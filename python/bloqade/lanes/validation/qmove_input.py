"""Reject input the qmove lowering does not support, reporting every problem.

Runs on native IR (after ``lower_to_native``), once per method.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, ClassVar

from bloqade.native.dialects.gate import dialect as native_gate, stmts as gate
from bloqade.squin import gate as squin_gate, noise as squin_noise
from kirin import ir, types
from kirin.dialects import func, ilist, py, scf
from kirin.validation import ValidationPass

from bloqade import qubit
from bloqade.gemini.common.dialects import arrange, qubit as gemini_qubit
from bloqade.gemini.logical.dialects.extensions import dialect as logical_extensions
from bloqade.gemini.logical.dialects.operations import dialect as logical_operations

LOGICAL_DIALECTS = frozenset({logical_operations, logical_extensions})
QUANTUM_DIALECTS = (
    frozenset(
        {
            squin_gate.dialect,
            squin_noise.dialect,
            native_gate,
            qubit.dialect,
            gemini_qubit.dialect,
            arrange.dialect,
        }
    )
    | LOGICAL_DIALECTS
)
ALLOCATION = (qubit.stmts.New, gemini_qubit.stmts.NewAt)
SUPPORTED = ALLOCATION + (
    qubit.stmts.Measure,
    qubit.stmts.IsZero,
    qubit.stmts.IsOne,
    qubit.stmts.IsLost,
    gate.CZ,
    gate.R,
    gate.Rz,
    arrange.stmts.MoveTo,
    arrange.stmts.Permute,
)
HIGHER_ORDER = (ilist.Map, ilist.ForEach, ilist.Foldl, ilist.Foldr, ilist.Scan)
ALLOCATION_MESSAGE = (
    "qubits may only be allocated under a whole-machine frame (the entry kernel, "
    "or a subroutine pinned to MachineFrame())"
)


def _function_code(value: ir.SSAValue) -> ir.Statement | None:
    """The body behind a function value: a lambda, or a constant method."""
    if not isinstance(value, ir.ResultValue):
        return None
    owner = value.stmt
    if isinstance(owner, func.Lambda):
        return owner
    if (
        isinstance(owner, py.Constant)
        and isinstance(owner.value, ir.PyAttr)
        and isinstance(owner.value.data, ir.Method)
    ):
        return owner.value.data.code
    return None


def _reachable(code: ir.Statement) -> Iterator[ir.Statement]:
    """Statements in ``code`` and, transitively, in every method it calls."""
    seen: set[int] = set()
    stack = [code]
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        for stmt in current.walk():
            yield stmt
            if isinstance(stmt, func.Invoke):
                stack.append(stmt.callee.code)
            elif (
                isinstance(stmt, func.Call)
                and (callee := _function_code(stmt.callee)) is not None
            ):
                stack.append(callee)


def _applies_gates(code: ir.Statement) -> bool:
    return any(
        s.dialect in QUANTUM_DIALECTS and not isinstance(s, ALLOCATION)
        for s in _reachable(code)
    )


def _allocates(code: ir.Statement) -> bool:
    return any(isinstance(s, ALLOCATION) for s in _reachable(code))


def get_input_validation(
    may_allocate: bool, clones: frozenset[ir.Method] = frozenset()
) -> type[ValidationPass]:
    """``ValidationSuite`` builds passes with no arguments, hence the factory.

    ``may_allocate`` is true for a method under a ``MachineFrame`` (the entry
    kernel, or a whole-machine subroutine). ``clones`` are the lowered subroutine
    methods; a ``func.Invoke`` of any other method is a call that survived
    inlining.
    """

    @dataclass
    class QMoveInputValidation(ValidationPass):
        MAY_ALLOCATE: ClassVar[bool] = may_allocate
        CLONES: ClassVar[frozenset[ir.Method]] = clones

        def name(self) -> str:
            return "lanes.qmove.input"

        def run(self, method: ir.Method) -> tuple[Any, list[ir.ValidationError]]:
            errors: list[ir.ValidationError] = []

            def error(node: ir.Statement, message: str) -> None:
                errors.append(ir.ValidationError(node, message))

            regions = [method.callable_region]
            for stmt in method.callable_region.walk():
                if not isinstance(stmt, func.Lambda):
                    regions.extend(stmt.regions)

                if isinstance(stmt, func.Return) and isinstance(
                    stmt.parent_stmt, (scf.IfElse, scf.For)
                ):
                    error(stmt, "early return inside scf is not supported")
                if isinstance(stmt, qubit.stmts.Reset):
                    error(stmt, "qubit.reset is not supported")
                elif stmt.dialect in LOGICAL_DIALECTS:
                    error(
                        stmt,
                        f"{stmt.name} is a logical-pipeline statement; "
                        "qmove targets the physical pipeline",
                    )
                elif stmt.dialect in QUANTUM_DIALECTS and not isinstance(
                    stmt, SUPPORTED
                ):
                    error(stmt, f"{stmt.name} is not supported by the qmove lowering")

                if (
                    stmt.dialect in QUANTUM_DIALECTS or isinstance(stmt, func.Invoke)
                ) and any(r.type.is_subseteq(types.Bottom) for r in stmt.results):
                    # Type inference found no valid type, e.g. a wrongly typed
                    # argument. Reporting it here keeps the later state checks
                    # from blaming stdlib code for the user's type error.
                    error(
                        stmt,
                        f"{stmt.name} result has no valid type (Bottom); "
                        "check its argument types",
                    )
                if isinstance(stmt, func.Call):
                    error(stmt, "calls through a function value are not supported")
                if isinstance(stmt, func.Invoke) and stmt.callee not in self.CLONES:
                    # An un-inlined call stays off the state chain, silently
                    # dropping the effects of the gates behind it.
                    error(
                        stmt,
                        f"call of {stmt.callee.sym_name} survived inlining and is "
                        "not a listed subroutine",
                    )
                if not self.MAY_ALLOCATE and isinstance(stmt, ALLOCATION):
                    error(stmt, ALLOCATION_MESSAGE)
                if (
                    isinstance(stmt, HIGHER_ORDER)
                    and (code := _function_code(stmt.fn)) is not None
                ):
                    if _applies_gates(code):
                        error(
                            stmt,
                            f"{stmt.name} applies gates through a function value; "
                            "use a for loop",
                        )
                    elif not self.MAY_ALLOCATE and _allocates(code):
                        error(stmt, ALLOCATION_MESSAGE)

            for region in regions:
                if len(region.blocks) > 1:
                    error(
                        region.parent_node or method.code,
                        f"region has {len(region.blocks)} blocks; only structured "
                        "(scf) control flow is supported",
                    )
            return None, errors

    return QMoveInputValidation
