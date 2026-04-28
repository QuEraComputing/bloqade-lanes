"""Tests for FuseAdjacentGates rewrite rule.

Verifies that adjacent same-opcode same-parameter quantum statements with
disjoint qubit sets inside a StaticPlacement body get fused into a single
statement covering the union of qubits.

All test IR is hand-built using kirin.ir primitives; no upstream lowering
is involved.
"""

from kirin import ir, rewrite, types as kirin_types
from kirin.rewrite.abc import RewriteResult

from bloqade import types as bloqade_types
from bloqade.lanes import types as lanes_types
from bloqade.lanes.dialects import place
from bloqade.lanes.rewrite.fuse_gates import FuseAdjacentGates

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wrap_in_static_placement(
    body_block: ir.Block,
    num_qubits: int = 4,
) -> tuple[place.StaticPlacement, ir.Block]:
    """Wrap a populated body block in a StaticPlacement and an outer block.

    Caller is responsible for appending statements to ``body_block`` and for
    setting up its entry-state block argument (see ``_new_body_block``).
    Returns the StaticPlacement and the outer block used as the rewrite target.
    """
    sp_qubits = tuple(
        ir.TestValue(type=bloqade_types.QubitType) for _ in range(num_qubits)
    )
    sp = place.StaticPlacement(qubits=sp_qubits, body=ir.Region(body_block))
    outer = ir.Block([sp])
    return sp, outer


def _new_body_block() -> tuple[ir.Block, ir.SSAValue]:
    """Return an empty body block + its entry-state SSA value."""
    body_block = ir.Block()
    entry_state = body_block.args.append_from(lanes_types.StateType, name="entry_state")
    return body_block, entry_state


def _run(outer_block: ir.Block) -> RewriteResult:
    """Apply Fixpoint(Walk(FuseAdjacentGates())) and return the result."""
    return rewrite.Fixpoint(rewrite.Walk(FuseAdjacentGates())).rewrite(outer_block)


# ---------------------------------------------------------------------------
# Skeleton: applying the rule on a single-statement body is a no-op.
# ---------------------------------------------------------------------------


def test_single_statement_body_is_unchanged():
    """A body with one R statement is left untouched by the fusion rule."""
    body_block, entry_state = _new_body_block()
    axis = ir.TestValue(type=kirin_types.Float)
    angle = ir.TestValue(type=kirin_types.Float)

    r = place.R(entry_state, axis_angle=axis, rotation_angle=angle, qubits=(0,))
    body_block.stmts.append(r)

    sp, outer = _wrap_in_static_placement(body_block)

    result = _run(outer)

    assert not result.has_done_something
    body_stmts = list(sp.body.blocks[0].stmts)
    assert len(body_stmts) == 1
    assert body_stmts[0] is r
