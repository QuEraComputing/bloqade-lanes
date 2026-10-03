"""Thread the machine state through a method's ``load; qmove.X; store`` chains.

After ``RewriteNativeToQMove`` every stateful statement sits in its own
``load``/``store`` pair. ``thread_method`` joins them into one chain per method,
opened by ``qmove.enter(frame)`` and closed by ``qmove.exit``, and threads the
state explicitly through ``scf.IfElse`` (both arms capture it and yield it back)
and ``scf.For`` (a loop-carried value, ahead of any existing ``iter_args``). No
``load``/``store`` is left afterwards.

This is a direct recursive traversal, not ``kirin.rewrite.Walk``: ``Walk`` visits
a region's blocks in reverse and a statement's regions before the statement
(``python/tests/rewrite/test_walk_order.py``), and threading needs execution
order. ``rewrite.state.RewriteLoadStore`` cannot be reused either: it finds
stateful statements by their ``ConsumesState``/``EmitsState`` traits, and a
threaded ``scf.IfElse`` has a ``State`` result but no trait (kirin owns ``scf``),
so it would skip the branch and drop its effect.
"""

from __future__ import annotations

from kirin import ir, types
from kirin.dialects import scf

from bloqade.lanes.dialects import move, qmove
from bloqade.lanes.dialects.qmove import Frame, MachineFrame
from bloqade.lanes.types import StateType


def _is_state(value: ir.SSAValue) -> bool:
    # Bottom is a subtype of everything; a value whose type inference failed is
    # not a machine state.
    return not value.type.is_subseteq(types.Bottom) and value.type.is_subseteq(
        StateType
    )


def touches_state(stmt: ir.Statement) -> bool:
    """Whether ``stmt``'s regions contain anything that reads or writes the state."""
    for region in stmt.regions:
        for inner in region.walk():
            if isinstance(inner, (move.Load, move.Store)):
                return True
            if any(map(_is_state, inner.args)) or any(map(_is_state, inner.results)):
                return True
    return False


def _advances_state(stmt: ir.Statement) -> bool:
    return bool(
        stmt.args
        and stmt.results
        and _is_state(stmt.args[0])
        and _is_state(stmt.results[0])
    )


def thread_block(
    block: ir.Block, state: ir.SSAValue, *, skip: ir.Statement | None = None
) -> ir.SSAValue:
    """Thread ``state`` through ``block``; return the state at its end."""
    for stmt in list(block.stmts):
        if stmt is skip:
            continue
        if isinstance(stmt, move.Load):
            stmt.result.replace_by(state)
            stmt.delete()
        elif isinstance(stmt, move.Store):
            stmt.delete()
        elif isinstance(stmt, scf.IfElse) and touches_state(stmt):
            state = _thread_if_else(stmt, state)
        elif isinstance(stmt, scf.For) and touches_state(stmt):
            state = _thread_for(stmt, state)
        elif _advances_state(stmt):
            state = stmt.results[0]
    return state


def _prepend_to_yield(block: ir.Block, value: ir.SSAValue) -> None:
    old = block.last_stmt
    assert isinstance(old, scf.Yield), f"expected scf.yield, got {old}"
    old.replace_by(scf.Yield(value, *old.values))


def _replace_scf(old: ir.Statement, new: ir.Statement) -> ir.SSAValue:
    new.insert_before(old)
    for old_result, new_result in zip(old.results, new.results[1:], strict=True):
        new_result.name = old_result.name
        new_result.type = old_result.type
        old_result.replace_by(new_result)
    old.delete()
    return new.results[0]


def _thread_if_else(stmt: scf.IfElse, state: ir.SSAValue) -> ir.SSAValue:
    if not stmt.else_body.blocks:
        # Python lowering always emits an else block; hand-built IR may not.
        block = ir.Block()
        block.args.append_from(stmt.cond.type)
        block.stmts.append(scf.Yield())
        stmt.else_body.blocks.append(block)
    for region in (stmt.then_body, stmt.else_body):
        body = region.blocks[0]
        _prepend_to_yield(body, thread_block(body, state))
    # kirin fixes a statement's result count when it is built, so rebuild it.
    # Reusing the regions requires detaching them first.
    then_body, else_body = stmt.then_body, stmt.else_body
    then_body.detach()
    else_body.detach()
    return _replace_scf(stmt, scf.IfElse(stmt.cond, then_body, else_body))


def _thread_for(stmt: scf.For, state: ir.SSAValue) -> ir.SSAValue:
    body = stmt.body.blocks[0]
    carried = body.args.insert_from(1, StateType, "state")
    _prepend_to_yield(body, thread_block(body, carried))
    region = stmt.body
    region.detach()
    return _replace_scf(stmt, scf.For(stmt.iterable, region, state, *stmt.initializers))


def thread_method(mt: ir.Method, *, frame: Frame | MachineFrame | None) -> None:
    """Open the chain with ``qmove.enter(frame)``, thread it, close it with ``exit``.

    Every method gets the pair, even one with no quantum statements, so "is a
    qmove method" is the same as "has an ``enter``" and a frame is never dropped.
    """
    block = mt.callable_region.blocks[0]
    opener = qmove.Enter(frame=frame)
    first = block.first_stmt
    assert first is not None
    opener.insert_before(first)
    final = thread_block(block, opener.result, skip=opener)
    terminator = block.last_stmt
    assert terminator is not None
    qmove.Exit(final).insert_before(terminator)
