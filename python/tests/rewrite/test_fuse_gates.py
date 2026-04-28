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


# ---------------------------------------------------------------------------
# Two-way R fusion (happy path)
# ---------------------------------------------------------------------------


def test_two_adjacent_r_fuses():
    """Two R(state, axis=%a, angle=%φ, qubits=...) with disjoint qubits fuse.

    The merged R has state_before from the first, qubits = concat in order,
    same axis/angle SSA values; the second R is gone.
    """
    body_block, entry_state = _new_body_block()
    axis = ir.TestValue(type=kirin_types.Float)
    angle = ir.TestValue(type=kirin_types.Float)

    r1 = place.R(entry_state, axis_angle=axis, rotation_angle=angle, qubits=(0,))
    body_block.stmts.append(r1)
    r2 = place.R(r1.state_after, axis_angle=axis, rotation_angle=angle, qubits=(1, 2))
    body_block.stmts.append(r2)

    sp, outer = _wrap_in_static_placement(body_block)

    result = _run(outer)

    assert result.has_done_something
    body_stmts = list(sp.body.blocks[0].stmts)
    assert len(body_stmts) == 1
    merged = body_stmts[0]
    assert isinstance(merged, place.R)
    assert merged.qubits == (0, 1, 2)
    assert merged.axis_angle is axis
    assert merged.rotation_angle is angle
    assert merged.state_before is entry_state


# ---------------------------------------------------------------------------
# R-fusion: predicate negative cases
# ---------------------------------------------------------------------------


def test_overlapping_qubits_does_not_fuse():
    """R(qubits=[0,1]) + R(qubits=[1,2]) overlap on qubit 1 → no fusion."""
    body_block, entry_state = _new_body_block()
    axis = ir.TestValue(type=kirin_types.Float)
    angle = ir.TestValue(type=kirin_types.Float)

    r1 = place.R(entry_state, axis_angle=axis, rotation_angle=angle, qubits=(0, 1))
    body_block.stmts.append(r1)
    r2 = place.R(r1.state_after, axis_angle=axis, rotation_angle=angle, qubits=(1, 2))
    body_block.stmts.append(r2)

    sp, outer = _wrap_in_static_placement(body_block)

    result = _run(outer)

    assert not result.has_done_something
    body_stmts = list(sp.body.blocks[0].stmts)
    assert body_stmts == [r1, r2]


def test_different_axis_ssa_does_not_fuse():
    """Two R with different axis_angle SSA values → no fusion (SSA-identity)."""
    body_block, entry_state = _new_body_block()
    axis_a = ir.TestValue(type=kirin_types.Float)
    axis_b = ir.TestValue(type=kirin_types.Float)
    angle = ir.TestValue(type=kirin_types.Float)

    r1 = place.R(entry_state, axis_angle=axis_a, rotation_angle=angle, qubits=(0,))
    body_block.stmts.append(r1)
    r2 = place.R(r1.state_after, axis_angle=axis_b, rotation_angle=angle, qubits=(1,))
    body_block.stmts.append(r2)

    sp, outer = _wrap_in_static_placement(body_block)

    result = _run(outer)

    assert not result.has_done_something
    body_stmts = list(sp.body.blocks[0].stmts)
    assert body_stmts == [r1, r2]


def test_different_rotation_angle_ssa_does_not_fuse():
    """Two R with different rotation_angle SSA values → no fusion."""
    body_block, entry_state = _new_body_block()
    axis = ir.TestValue(type=kirin_types.Float)
    angle_a = ir.TestValue(type=kirin_types.Float)
    angle_b = ir.TestValue(type=kirin_types.Float)

    r1 = place.R(entry_state, axis_angle=axis, rotation_angle=angle_a, qubits=(0,))
    body_block.stmts.append(r1)
    r2 = place.R(r1.state_after, axis_angle=axis, rotation_angle=angle_b, qubits=(1,))
    body_block.stmts.append(r2)

    sp, outer = _wrap_in_static_placement(body_block)

    result = _run(outer)

    assert not result.has_done_something
    body_stmts = list(sp.body.blocks[0].stmts)
    assert body_stmts == [r1, r2]


def test_different_opcode_between_blocks_fusion():
    """R; Rz; R does NOT fuse the two Rs even though their qubits are disjoint.

    Strict adjacency: the Rz between them flushes the group.
    """
    body_block, entry_state = _new_body_block()
    axis = ir.TestValue(type=kirin_types.Float)
    angle_r = ir.TestValue(type=kirin_types.Float)
    angle_rz = ir.TestValue(type=kirin_types.Float)

    r1 = place.R(entry_state, axis_angle=axis, rotation_angle=angle_r, qubits=(0,))
    body_block.stmts.append(r1)
    rz = place.Rz(r1.state_after, rotation_angle=angle_rz, qubits=(1,))
    body_block.stmts.append(rz)
    r2 = place.R(rz.state_after, axis_angle=axis, rotation_angle=angle_r, qubits=(2,))
    body_block.stmts.append(r2)

    sp, outer = _wrap_in_static_placement(body_block)

    result = _run(outer)

    assert not result.has_done_something
    body_stmts = list(sp.body.blocks[0].stmts)
    assert body_stmts == [r1, rz, r2]


# ---------------------------------------------------------------------------
# Rz fusion
# ---------------------------------------------------------------------------


def test_two_adjacent_rz_fuses():
    """Two Rz(state, angle=%θ, qubits=...) with disjoint qubits fuse."""
    body_block, entry_state = _new_body_block()
    angle = ir.TestValue(type=kirin_types.Float)

    rz1 = place.Rz(entry_state, rotation_angle=angle, qubits=(0,))
    body_block.stmts.append(rz1)
    rz2 = place.Rz(rz1.state_after, rotation_angle=angle, qubits=(1, 2))
    body_block.stmts.append(rz2)

    sp, outer = _wrap_in_static_placement(body_block)

    result = _run(outer)

    assert result.has_done_something
    body_stmts = list(sp.body.blocks[0].stmts)
    assert len(body_stmts) == 1
    merged = body_stmts[0]
    assert isinstance(merged, place.Rz)
    assert merged.qubits == (0, 1, 2)
    assert merged.rotation_angle is angle
    assert merged.state_before is entry_state


def test_rz_with_different_angle_does_not_fuse():
    """Two Rz with different rotation_angle SSA values → no fusion."""
    body_block, entry_state = _new_body_block()
    angle_a = ir.TestValue(type=kirin_types.Float)
    angle_b = ir.TestValue(type=kirin_types.Float)

    rz1 = place.Rz(entry_state, rotation_angle=angle_a, qubits=(0,))
    body_block.stmts.append(rz1)
    rz2 = place.Rz(rz1.state_after, rotation_angle=angle_b, qubits=(1,))
    body_block.stmts.append(rz2)

    sp, outer = _wrap_in_static_placement(body_block)

    result = _run(outer)

    assert not result.has_done_something
    body_stmts = list(sp.body.blocks[0].stmts)
    assert body_stmts == [rz1, rz2]
