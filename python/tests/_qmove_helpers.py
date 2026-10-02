"""Test helpers for the qmove lowering: a deep IR comparator and an eraser.

kirin's ``is_structurally_equal`` cannot compare lowered IR. ``Region``'s version
records every block pair in its context before comparing, so ``Block``'s
version returns early and nested region contents are never compared, and result
types are never compared. ``blocks_equal`` recurses into every nested region
itself, block by block.
"""

from __future__ import annotations

from typing import TypeVar

from bloqade.native.dialects.gate import stmts as gate
from kirin import ir, types
from kirin.dialects import func, scf

from bloqade import qubit
from bloqade.gemini.common.dialects.arrange import stmts as arrange
from bloqade.lanes.dialects import move, qmove
from bloqade.lanes.types import StateType

Context = dict[ir.SSAValue, ir.SSAValue]
T = TypeVar("T", bound=ir.Statement)


def statements_of(node: ir.Method | ir.Statement, kind: type[T]) -> list[T]:
    """Every ``kind`` statement inside ``node``, nested regions included."""
    root = node.code if isinstance(node, ir.Method) else node
    return [s for s in root.walk() if isinstance(s, kind)]


def first_of(node: ir.Method | ir.Statement, kind: type[T]) -> T:
    first, *_ = statements_of(node, kind)
    return first


def top_level(mt: ir.Method) -> list[ir.Statement]:
    return list(mt.callable_region.blocks[0].stmts)


def _attr_equal(a: ir.Attribute, b: ir.Attribute) -> bool:
    if isinstance(a, ir.PyAttr) and isinstance(b, ir.PyAttr):
        if isinstance(a.data, ir.Method) and isinstance(b.data, ir.Method):
            return a.data.sym_name == b.data.sym_name
        return a.type == b.type and a.data == b.data
    return a == b


def _stmts_equal(a: ir.Statement, b: ir.Statement, ctx: Context) -> str | None:
    if type(a) is not type(b):
        return f"{type(a).__name__} != {type(b).__name__}"
    if (len(a.args), len(a.results), len(a.regions)) != (
        len(b.args),
        len(b.results),
        len(b.regions),
    ):
        return f"{a.name}: arity differs"
    if a.attributes.keys() != b.attributes.keys():
        return f"{a.name}: attribute names differ"
    for key in a.attributes:
        if not _attr_equal(a.attributes[key], b.attributes[key]):
            return f"{a.name}: attribute {key!r} differs"
    for x, y in zip(a.args, b.args):
        if ctx.get(x, x) is not y:
            return f"{a.name}: operand differs"
    for ra, rb in zip(a.regions, b.regions):
        if len(ra.blocks) != len(rb.blocks):
            return f"{a.name}: region has {len(ra.blocks)} != {len(rb.blocks)} blocks"
        for ba, bb in zip(ra.blocks, rb.blocks):
            if (err := blocks_equal(ba, bb, ctx)) is not None:
                return f"in {a.name}: {err}"
    for x, y in zip(a.results, b.results):
        if x.type != y.type:
            return f"{a.name}: result type {x.type} != {y.type}"
        ctx[x] = y
    return None


def blocks_equal(a: ir.Block, b: ir.Block, ctx: Context | None = None) -> str | None:
    """``None`` if the blocks match, recursively; otherwise the first difference."""
    ctx = {} if ctx is None else ctx
    if len(a.args) != len(b.args):
        return f"{len(a.args)} != {len(b.args)} block arguments"
    for x, y in zip(a.args, b.args):
        if x.type != y.type:
            return f"block argument type {x.type} != {y.type}"
        ctx[x] = y
    left, right = list(a.stmts), list(b.stmts)
    if len(left) != len(right):
        return f"{len(left)} != {len(right)} statements"
    for x, y in zip(left, right):
        if (err := _stmts_equal(x, y, ctx)) is not None:
            return err
    return None


def assert_methods_match(got: ir.Method, expected: ir.Method) -> None:
    err = blocks_equal(
        got.callable_region.blocks[0], expected.callable_region.blocks[0]
    )
    if err is not None:
        raise AssertionError(
            f"{err}\n--- got ---\n{got.print_str()}\n--- expected ---\n"
            f"{expected.print_str()}"
        )


def _is_state(value: ir.SSAValue) -> bool:
    return not value.type.is_subseteq(types.Bottom) and value.type.is_subseteq(
        StateType
    )


def _native(stmt: ir.Statement) -> ir.Statement | None:
    if isinstance(stmt, qmove.CZ):
        return gate.CZ(stmt.controls, stmt.targets)
    if isinstance(stmt, qmove.R):
        return gate.R(stmt.axis_angle, stmt.rotation_angle, stmt.qubits)
    if isinstance(stmt, qmove.Rz):
        return gate.Rz(stmt.rotation_angle, stmt.qubits)
    if isinstance(stmt, qmove.MoveTo):
        return arrange.MoveTo(
            stmt.qubits, stmt.locations, multi_move_warning=stmt.multi_move_warning
        )
    if isinstance(stmt, qmove.Permute):
        return arrange.Permute(stmt.qubits, stmt.perm, insert_moves=stmt.insert_moves)
    if isinstance(stmt, qmove.Measure):
        return qubit.stmts.Measure(stmt.qubits)
    if isinstance(stmt, qmove.Invoke):
        return func.Invoke(tuple(stmt.inputs), callee=stmt.callee)
    return None


def erase_qmove(block: ir.Block) -> None:
    """Undo the qmove lowering in place: drop all state plumbing."""
    for stmt in list(block.stmts):
        if (
            isinstance(stmt, (scf.IfElse, scf.For))
            and stmt.results
            and _is_state(stmt.results[0])
        ):
            _erase_scf(stmt)
            continue
        native = _native(stmt)
        if native is None:
            continue
        native.insert_before(stmt)
        for old, new in zip(stmt.results[1:], native.results):
            new.name = old.name
            new.type = old.type
            old.replace_by(new)
        stmt.results[0].replace_by(stmt.args[0])
        stmt.delete()
    for stmt in list(block.stmts):
        if isinstance(stmt, (move.Store, qmove.Exit)):
            stmt.delete()
    for stmt in list(block.stmts):
        if isinstance(stmt, (move.Load, qmove.Enter)):
            stmt.delete()


def _erase_scf(stmt: scf.IfElse | scf.For) -> None:
    yielded: list[ir.SSAValue] = []
    for region in stmt.regions:
        body = region.blocks[0]
        erase_qmove(body)
        old_yield = body.last_stmt
        assert isinstance(old_yield, scf.Yield)
        # With the arm erased, this is the state the arm started from.
        yielded.append(old_yield.values[0])
        old_yield.replace_by(scf.Yield(*old_yield.values[1:]))
    if isinstance(stmt, scf.For):
        body = stmt.body.blocks[0]
        body.args.delete(body.args[1])
        state_in = stmt.initializers[0]
        region = stmt.body
        region.detach()
        new: ir.Statement = scf.For(stmt.iterable, region, *stmt.initializers[1:])
    else:
        state_in = yielded[0]  # the state both arms captured
        then_body, else_body = stmt.then_body, stmt.else_body
        then_body.detach()
        else_body.detach()
        new = scf.IfElse(stmt.cond, then_body, else_body)
    new.insert_before(stmt)
    stmt.results[0].replace_by(state_in)
    for old, replacement in zip(stmt.results[1:], new.results, strict=True):
        replacement.name = old.name
        replacement.type = old.type
        old.replace_by(replacement)
    stmt.delete()
