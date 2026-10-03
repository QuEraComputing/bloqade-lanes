"""Structural validation of qmove IR: state rules, call rules, frame rules.

Every check is a direct walk, not a ``Forward`` analysis (which only visits
reachable code), following ``FlatBlockValidation``. Checks that need qubit
identity (distinct qubits within a statement, the general case of F4, whether a
call's frame precondition holds) are left to later analyses.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, ClassVar

from bloqade.types import QubitType
from kirin import ir, types
from kirin.analysis import const
from kirin.dialects import func, ilist, scf
from kirin.validation import ValidationPass

from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.dialects import move, qmove
from bloqade.lanes.dialects.qmove import Frame, MachineFrame
from bloqade.lanes.types import StateType
from bloqade.lanes.validation.spectator import SpectatorPolicy, format_location

Path = list[tuple[ir.Statement, int]]


def _is_state(value: ir.SSAValue) -> bool:
    # Bottom is a subtype of everything; a value whose type inference failed is
    # not a machine state.
    return not value.type.is_subseteq(types.Bottom) and value.type.is_subseteq(
        StateType
    )


def is_qmove_method(mt: ir.Method) -> bool:
    """Whether ``mt`` has been lowered to qmove: it opens its chain with ``enter``."""
    return any(isinstance(s, qmove.Enter) for s in mt.callable_region.walk())


def method_frame(mt: ir.Method) -> Frame | MachineFrame | None:
    """The frame on ``mt``'s ``enter``; ``None`` for a hole or a non-qmove method."""
    for stmt in mt.callable_region.walk():
        if isinstance(stmt, qmove.Enter):
            return stmt.frame
    return None


def _state_values(mt: ir.Method) -> Iterator[ir.SSAValue]:
    for stmt in mt.callable_region.walk():
        yield from (r for r in stmt.results if _is_state(r))
        for region in stmt.regions:
            for block in region.blocks:
                yield from (a for a in block.args if _is_state(a))


def _defining_block(value: ir.SSAValue) -> ir.Block | None:
    if isinstance(value, ir.BlockArgument):
        return value.block
    assert isinstance(value, ir.ResultValue)
    return value.stmt.parent_block


def _path(stmt: ir.Statement, top: ir.Block | None) -> Path | None:
    """``(statement, region index)`` pairs enclosing ``stmt``, below ``top``."""
    path: Path = []
    block = stmt.parent_block
    while block is not top:
        if block is None:
            return None
        region = block.parent
        if region is None:
            return None
        owner = region.parent_node
        if not isinstance(owner, ir.Statement):
            return None
        path.append((owner, owner.regions.index(region)))
        block = owner.parent_block
    path.reverse()
    return path


def _mutually_exclusive(a: Path, b: Path) -> bool:
    """Whether two uses sit in different arms of a common ``scf.IfElse``."""
    for (stmt_a, region_a), (stmt_b, region_b) in zip(a, b):
        if stmt_a is not stmt_b:
            return False
        if region_a != region_b:
            return isinstance(stmt_a, scf.IfElse)
    return False  # one path is a prefix of the other: same execution path


def check_use_def(mt: ir.Method) -> list[ir.ValidationError]:
    """V1: every state is used, and no path consumes it twice (ignoring Store)."""
    errors = []
    for value in _state_values(mt):
        node = value.stmt if isinstance(value, ir.ResultValue) else mt.code
        if not value.uses:
            errors.append(
                ir.ValidationError(
                    node, "V1: state is never used; its update is dropped"
                )
            )
            continue
        top = _defining_block(value)
        consumers: list[tuple[ir.Statement, Path]] = []
        for use in value.uses:
            if isinstance(use.stmt, move.Store):
                continue
            path = _path(use.stmt, top)
            if path is None:
                continue
            if any(isinstance(s, (scf.For, func.Lambda)) for s, _ in path):
                errors.append(
                    ir.ValidationError(
                        use.stmt,
                        "V1: state consumed inside a loop body that does not define it",
                    )
                )
            consumers.append((use.stmt, path))
        for i, (first, first_path) in enumerate(consumers):
            for second, second_path in consumers[i + 1 :]:
                if not _mutually_exclusive(first_path, second_path):
                    errors.append(
                        ir.ValidationError(
                            second,
                            f"V1: state consumed twice on one path, by {first.name} "
                            f"and {second.name}",
                        )
                    )
    return errors


def check_chain_ends(mt: ir.Method) -> list[ir.ValidationError]:
    """V2: exactly one ``enter`` and one ``exit``, top level; no ``load``/``store``."""
    errors = []
    top = mt.callable_region.blocks[0]
    stmts = list(mt.callable_region.walk())
    for stmt in stmts:
        if isinstance(stmt, (move.Load, move.Store)):
            errors.append(
                ir.ValidationError(
                    stmt,
                    f"V2: {stmt.name} is not allowed in qmove IR; "
                    "a method's chain uses enter/exit",
                )
            )
    ends = [s for s in stmts if isinstance(s, (qmove.Enter, qmove.Exit))]
    for stmt in ends:
        if stmt.parent_block is not top:
            errors.append(
                ir.ValidationError(
                    stmt, f"V2: {stmt.name} must be in the method's top-level block"
                )
            )
    stateful = any(_is_state(v) for s in stmts for v in (*s.args, *s.results))
    if ends or stateful:
        for kind in (qmove.Enter, qmove.Exit):
            count = sum(isinstance(s, kind) for s in ends)
            if count != 1:
                errors.append(
                    ir.ValidationError(
                        mt.code,
                        f"V2: a qmove method needs exactly one {kind.name}, "
                        f"found {count}",
                    )
                )
    return errors


def _const(value: ir.SSAValue) -> Any:
    hint = value.hints.get("const")
    return hint.data if isinstance(hint, const.Value) else None


def _list_len(typ: types.TypeAttribute) -> int | None:
    if (
        isinstance(typ, types.Generic)
        and typ.is_subseteq(ilist.IListType)
        and isinstance(typ.vars[1], types.Literal)
        and isinstance(typ.vars[1].data, int)
    ):
        return typ.vars[1].data
    return None


def check_statements(mt: ir.Method) -> list[ir.ValidationError]:
    """V3: per-statement checks wherever the operands are constant."""
    errors = []
    for stmt in mt.callable_region.walk():
        if isinstance(stmt, qmove.CZ):
            controls, targets = _list_len(stmt.controls.type), _list_len(
                stmt.targets.type
            )
            if controls is not None and targets is not None and controls != targets:
                errors.append(
                    ir.ValidationError(
                        stmt, f"V3: cz has {controls} controls but {targets} targets"
                    )
                )
        elif isinstance(stmt, qmove.Permute):
            perm = _const(stmt.perm)
            if perm is not None and sorted(int(p) for p in perm) != list(
                range(len(perm))
            ):
                errors.append(
                    ir.ValidationError(
                        stmt, f"V3: perm {tuple(perm)} is not a permutation"
                    )
                )
        elif isinstance(stmt, qmove.MoveTo):
            locations, count = _const(stmt.locations), _list_len(stmt.qubits.type)
            if locations is not None and count is not None and len(locations) != count:
                errors.append(
                    ir.ValidationError(
                        stmt,
                        f"V3: move_to has {len(locations)} locations for {count} qubits",
                    )
                )
    return errors


def check_calls(mt: ir.Method) -> list[ir.ValidationError]:
    """Call rules, plus F5's whole-machine half (it holds whatever the caller's frame)."""
    errors = []
    caller_is_machine = isinstance(method_frame(mt), MachineFrame)
    for stmt in mt.callable_region.walk():
        if isinstance(stmt, (qmove.Invoke, qmove.Prepare)):
            if not is_qmove_method(stmt.callee):
                errors.append(
                    ir.ValidationError(
                        stmt,
                        f"{stmt.name} target {stmt.callee.sym_name} is not a qmove "
                        "method",
                    )
                )
            elif (
                isinstance(method_frame(stmt.callee), MachineFrame)
                and not caller_is_machine
            ):
                errors.append(
                    ir.ValidationError(
                        stmt,
                        f"F5: whole-machine method {stmt.callee.sym_name} can only "
                        "be called from a whole-machine method",
                    )
                )
        elif isinstance(stmt, func.Invoke) and is_qmove_method(stmt.callee):
            errors.append(
                ir.ValidationError(
                    stmt,
                    f"qmove method {stmt.callee.sym_name} must be called with "
                    "qmove.invoke",
                )
            )
    return errors


def expected_param_slots(
    mt: ir.Method,
) -> tuple[tuple[tuple[int, int], ...], list[str]]:
    slots: list[tuple[int, int]] = []
    problems: list[str] = []
    for index, typ in enumerate(mt.arg_types):
        if typ.is_subseteq(QubitType):
            slots.append((index, 1))
        elif typ.is_subseteq(ilist.IListType[QubitType, types.Any]):
            length = _list_len(typ)
            if length is None:
                problems.append(
                    f"parameter {index} is {typ}; a framed subroutine needs "
                    "IList[Qubit, Literal[N]]"
                )
            else:
                slots.append((index, length))
    return tuple(slots), problems


def check_frame(mt: ir.Method, arch: ArchSpec) -> list[ir.ValidationError]:
    """F1-F5, for a method under a partial ``Frame``.

    A ``MachineFrame`` is the whole machine and has nothing to check; a hole
    (``None``) is not known yet. F5's whole-machine half is in ``check_calls``.

    F4 (relabels compose to the identity) is judged only when every relabel
    ``Permute`` (``insert_moves=False``) in the method is a top-level, constant,
    well-formed permutation of a parameter. Without qubit identity, a relabel
    that is nested, non-constant, malformed or on a non-parameter operand cannot
    be composed with the others, so F4 abstains for the whole method rather than
    judge an inconsistent subset.
    """
    frame = method_frame(mt)
    if not isinstance(frame, Frame):
        return []
    errors: list[ir.ValidationError] = []

    def error(node: ir.Statement, message: str) -> None:
        errors.append(ir.ValidationError(node, message))

    if len(frame.binding) != frame.shape.total_slots:
        error(
            mt.code,
            f"F1: binding has {len(frame.binding)} locations for "
            f"{frame.shape.total_slots} slots",
        )
    if len(set(frame.binding)) != len(frame.binding):
        error(mt.code, "F1: binding locations are not distinct")
    for problem in arch.check_location_group(list(set(frame.binding))):
        error(mt.code, f"F1: {problem}")
    for zone in frame.effects.cz_zones | frame.effects.measure_zones:
        if not 0 <= zone.zone_id < len(arch.zones):
            error(mt.code, f"F1: zone {zone.zone_id} does not exist")

    slots, problems = expected_param_slots(mt)
    for problem in problems:
        error(mt.code, f"F2: {problem}")
    if not problems and tuple(sorted(frame.shape.param_slots)) != slots:
        error(
            mt.code,
            f"F2: frame slots {frame.shape.param_slots} do not match parameters {slots}",
        )

    footprint = frame.footprint
    effects = frame.effects
    relabels: dict[ir.SSAValue, list[tuple[int, ...]]] = {}
    f4_abstains = False
    top = mt.callable_region.blocks[0]
    for stmt in mt.callable_region.walk():
        if isinstance(stmt, qmove.MoveTo):
            for loc in _const(stmt.locations) or ():
                if loc not in footprint:
                    error(
                        stmt,
                        f"F3: move_to target {format_location(loc)} is outside the frame",
                    )
        elif isinstance(stmt, move.CZ) and stmt.zone_address not in effects.cz_zones:
            error(
                stmt,
                f"F3: cz in zone {stmt.zone_address.zone_id} is not in the frame's effects",
            )
        elif isinstance(stmt, (move.Measure, move.EndMeasure)):
            if not set(stmt.zone_addresses) <= effects.measure_zones:
                error(stmt, "F3: measurement zone is not in the frame's effects")
        elif (
            isinstance(stmt, (move.GlobalR, move.GlobalRz))
            and not effects.global_pulses
        ):
            error(stmt, "F3: global pulse is not in the frame's effects")
        elif isinstance(stmt, move.Move):
            for lane in stmt.lanes:
                src, dst = arch.get_endpoints(lane)
                if src not in footprint or dst not in footprint:
                    error(stmt, f"F3: lane {lane} leaves the frame")
        elif isinstance(stmt, qmove.Permute) and not stmt.insert_moves:
            perm = _const(stmt.perm)
            if (
                stmt.parent_block is top
                and perm is not None
                and isinstance(stmt.qubits, ir.BlockArgument)
                and stmt.qubits.block is top
                and sorted(int(p) for p in perm) == list(range(len(perm)))
            ):
                relabels.setdefault(stmt.qubits, []).append(tuple(int(p) for p in perm))
            else:
                f4_abstains = True
        elif isinstance(stmt, qmove.Invoke):
            inner = method_frame(stmt.callee)
            if isinstance(inner, Frame) and (
                not inner.footprint <= footprint
                or not inner.effects.is_subset_of(effects)
            ):
                error(
                    stmt,
                    f"F5: frame of {stmt.callee.sym_name} is not inside this frame",
                )

    if not f4_abstains:
        for perms in relabels.values():
            # Validate all perms have the same length before composition.
            perm_len = len(perms[0])
            if any(len(p) != perm_len for p in perms):
                continue
            net = list(range(perm_len))
            for perm in perms:
                net = [net[i] for i in perm]
            if net != list(range(len(net))):
                error(
                    mt.code,
                    f"F4: relabels {perms} leave the binding permuted at exit",
                )
    return errors


def get_qmove_validation(
    arch: ArchSpec, policy: SpectatorPolicy
) -> type[ValidationPass]:
    """``ValidationSuite`` builds passes with no arguments, hence the factory."""

    @dataclass
    class QMoveValidation(ValidationPass):
        ARCH: ClassVar[ArchSpec] = arch
        POLICY: ClassVar[SpectatorPolicy] = policy

        def name(self) -> str:
            return "lanes.qmove.validation"

        def run(self, method: ir.Method) -> tuple[Any, list[ir.ValidationError]]:
            errors = (
                check_use_def(method)
                + check_chain_ends(method)
                + check_statements(method)
                + check_calls(method)
                + check_frame(method, self.ARCH)
            )
            if isinstance(frame := method_frame(method), Frame):
                errors += [
                    ir.ValidationError(method.code, problem)
                    for problem in self.POLICY.check_frame(frame, self.ARCH)
                ]
            return None, errors

    return QMoveValidation
