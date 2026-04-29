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
    Appends a ``place.Yield`` terminator threading the final state so the
    body satisfies ``StaticPlacement.check()``.
    Returns the StaticPlacement and the outer block used as the rewrite target.
    """
    last = body_block.last_stmt
    if isinstance(last, place.QuantumStmt):
        final_state = last.state_after
    else:
        final_state = body_block.args[0]
    body_block.stmts.append(place.Yield(final_state))

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


def _body_stmts(sp: place.StaticPlacement) -> list[ir.Statement]:
    """Return body statements excluding the trailing ``place.Yield`` terminator."""
    stmts = list(sp.body.blocks[0].stmts)
    assert stmts and isinstance(stmts[-1], place.Yield)
    return stmts[:-1]


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
    body_stmts = _body_stmts(sp)
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
    body_stmts = _body_stmts(sp)
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
    body_stmts = _body_stmts(sp)
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
    body_stmts = _body_stmts(sp)
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
    body_stmts = _body_stmts(sp)
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
    body_stmts = _body_stmts(sp)
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
    body_stmts = _body_stmts(sp)
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
    body_stmts = _body_stmts(sp)
    assert body_stmts == [rz1, rz2]


# ---------------------------------------------------------------------------
# CZ fusion with controls-then-targets re-interleaving
# ---------------------------------------------------------------------------


def test_two_adjacent_cz_fuses_with_reinterleaved_qubits():
    """Two CZ with disjoint qubits fuse; merged.qubits = controls0+controls1+targets0+targets1.

    Verifies the controls-then-targets convention enforced by place.CZ.controls
    and place.CZ.targets (which split qubits in half) is preserved.
    """
    body_block, entry_state = _new_body_block()

    # CZ#0: controls=(0,), targets=(2,)  → qubits=(0, 2)
    cz1 = place.CZ(entry_state, qubits=(0, 2))
    body_block.stmts.append(cz1)
    # CZ#1: controls=(1,), targets=(3,)  → qubits=(1, 3)
    cz2 = place.CZ(cz1.state_after, qubits=(1, 3))
    body_block.stmts.append(cz2)

    sp, outer = _wrap_in_static_placement(body_block)

    result = _run(outer)

    assert result.has_done_something
    body_stmts = _body_stmts(sp)
    assert len(body_stmts) == 1
    merged = body_stmts[0]
    assert isinstance(merged, place.CZ)
    # merged.controls = (0, 1), merged.targets = (2, 3) → qubits = (0, 1, 2, 3)
    assert merged.qubits == (0, 1, 2, 3)
    assert merged.controls == (0, 1)
    assert merged.targets == (2, 3)
    assert merged.state_before is entry_state


def test_cz_overlapping_controls_does_not_fuse():
    """Two CZ sharing a control qubit do not fuse."""
    body_block, entry_state = _new_body_block()

    cz1 = place.CZ(entry_state, qubits=(0, 2))  # control=0, target=2
    body_block.stmts.append(cz1)
    cz2 = place.CZ(cz1.state_after, qubits=(0, 3))  # control=0, target=3 (overlaps)
    body_block.stmts.append(cz2)

    sp, outer = _wrap_in_static_placement(body_block)

    result = _run(outer)

    assert not result.has_done_something
    body_stmts = _body_stmts(sp)
    assert body_stmts == [cz1, cz2]


def test_cz_three_way_fusion_preserves_control_target_order():
    """Three CZ statements collapse with all controls first, then all targets."""
    body_block, entry_state = _new_body_block()

    cz1 = place.CZ(entry_state, qubits=(0, 4))  # c=0, t=4
    body_block.stmts.append(cz1)
    cz2 = place.CZ(cz1.state_after, qubits=(1, 5))  # c=1, t=5
    body_block.stmts.append(cz2)
    cz3 = place.CZ(cz2.state_after, qubits=(2, 6))  # c=2, t=6
    body_block.stmts.append(cz3)

    sp, outer = _wrap_in_static_placement(body_block, num_qubits=7)

    result = _run(outer)

    assert result.has_done_something
    body_stmts = _body_stmts(sp)
    assert len(body_stmts) == 1
    merged = body_stmts[0]
    assert isinstance(merged, place.CZ)
    assert merged.qubits == (0, 1, 2, 4, 5, 6)
    assert merged.controls == (0, 1, 2)
    assert merged.targets == (4, 5, 6)


# ---------------------------------------------------------------------------
# N-way fusion in a single pass
# ---------------------------------------------------------------------------


def test_four_adjacent_r_collapse_in_one_pass():
    """Four adjacent fusable R statements collapse to one in a single rewrite invocation."""
    body_block, entry_state = _new_body_block()
    axis = ir.TestValue(type=kirin_types.Float)
    angle = ir.TestValue(type=kirin_types.Float)

    r1 = place.R(entry_state, axis_angle=axis, rotation_angle=angle, qubits=(0,))
    body_block.stmts.append(r1)
    r2 = place.R(r1.state_after, axis_angle=axis, rotation_angle=angle, qubits=(1,))
    body_block.stmts.append(r2)
    r3 = place.R(r2.state_after, axis_angle=axis, rotation_angle=angle, qubits=(2, 3))
    body_block.stmts.append(r3)
    r4 = place.R(r3.state_after, axis_angle=axis, rotation_angle=angle, qubits=(4,))
    body_block.stmts.append(r4)

    sp, outer = _wrap_in_static_placement(body_block, num_qubits=5)

    result = _run(outer)

    assert result.has_done_something
    body_stmts = _body_stmts(sp)
    assert len(body_stmts) == 1
    merged = body_stmts[0]
    assert isinstance(merged, place.R)
    assert merged.qubits == (0, 1, 2, 3, 4)
    assert merged.state_before is entry_state


# ---------------------------------------------------------------------------
# Boundary statements: Initialize and EndMeasure flush groups.
# ---------------------------------------------------------------------------


def test_initialize_flushes_group_does_not_start_new_one():
    """An Initialize between two R groups flushes the first and does not start a new one.

    Initialize takes its own state_before SSA value (the previous statement's
    state_after) and produces a state_after; subsequent fusable statements
    that thread through Initialize start a fresh group.
    """
    body_block, entry_state = _new_body_block()
    axis = ir.TestValue(type=kirin_types.Float)
    angle = ir.TestValue(type=kirin_types.Float)
    init_theta = ir.TestValue(type=kirin_types.Float)
    init_phi = ir.TestValue(type=kirin_types.Float)
    init_lam = ir.TestValue(type=kirin_types.Float)

    r1 = place.R(entry_state, axis_angle=axis, rotation_angle=angle, qubits=(0,))
    body_block.stmts.append(r1)
    r2 = place.R(r1.state_after, axis_angle=axis, rotation_angle=angle, qubits=(1,))
    body_block.stmts.append(r2)
    init = place.Initialize(
        r2.state_after,
        theta=init_theta,
        phi=init_phi,
        lam=init_lam,
        qubits=(2,),
    )
    body_block.stmts.append(init)
    r3 = place.R(init.state_after, axis_angle=axis, rotation_angle=angle, qubits=(3,))
    body_block.stmts.append(r3)
    r4 = place.R(r3.state_after, axis_angle=axis, rotation_angle=angle, qubits=(4,))
    body_block.stmts.append(r4)

    sp, outer = _wrap_in_static_placement(body_block, num_qubits=5)

    result = _run(outer)

    assert result.has_done_something
    body_stmts = _body_stmts(sp)
    # Expected: [merged(r1+r2), init, merged(r3+r4)]
    assert len(body_stmts) == 3
    merged_first, init_seen, merged_second = body_stmts
    assert isinstance(merged_first, place.R)
    assert merged_first.qubits == (0, 1)
    assert init_seen is init
    assert isinstance(merged_second, place.R)
    assert merged_second.qubits == (3, 4)


def test_endmeasure_flushes_preceding_group():
    """An EndMeasure flushes a preceding R run; the EndMeasure itself is untouched."""
    body_block, entry_state = _new_body_block()
    axis = ir.TestValue(type=kirin_types.Float)
    angle = ir.TestValue(type=kirin_types.Float)

    r1 = place.R(entry_state, axis_angle=axis, rotation_angle=angle, qubits=(0,))
    body_block.stmts.append(r1)
    r2 = place.R(r1.state_after, axis_angle=axis, rotation_angle=angle, qubits=(1,))
    body_block.stmts.append(r2)
    em = place.EndMeasure(r2.state_after, qubits=(0, 1))
    body_block.stmts.append(em)

    sp, outer = _wrap_in_static_placement(body_block, num_qubits=2)

    result = _run(outer)

    assert result.has_done_something
    body_stmts = _body_stmts(sp)
    assert len(body_stmts) == 2
    merged, em_seen = body_stmts
    assert isinstance(merged, place.R)
    assert merged.qubits == (0, 1)
    assert em_seen is em


# ---------------------------------------------------------------------------
# Idempotence and degenerate bodies.
# ---------------------------------------------------------------------------


def test_idempotence_second_application_is_noop():
    """Running the rule a second time on already-fused IR returns has_done_something=False."""
    body_block, entry_state = _new_body_block()
    axis = ir.TestValue(type=kirin_types.Float)
    angle = ir.TestValue(type=kirin_types.Float)

    r1 = place.R(entry_state, axis_angle=axis, rotation_angle=angle, qubits=(0,))
    body_block.stmts.append(r1)
    r2 = place.R(r1.state_after, axis_angle=axis, rotation_angle=angle, qubits=(1,))
    body_block.stmts.append(r2)

    _, outer = _wrap_in_static_placement(body_block)

    result_first = _run(outer)
    assert result_first.has_done_something

    # Second invocation: nothing more to do.
    result_second = _run(outer)
    assert not result_second.has_done_something


def test_empty_body_is_unchanged():
    """An empty body is a no-op."""
    body_block, entry_state = _new_body_block()

    sp, outer = _wrap_in_static_placement(body_block)

    result = _run(outer)

    assert not result.has_done_something
    assert _body_stmts(sp) == []


def test_no_fusable_groups_is_unchanged():
    """A body of only non-fusable statements (Initialize + EndMeasure) is a no-op."""
    body_block, entry_state = _new_body_block()
    init_theta = ir.TestValue(type=kirin_types.Float)
    init_phi = ir.TestValue(type=kirin_types.Float)
    init_lam = ir.TestValue(type=kirin_types.Float)

    init = place.Initialize(
        entry_state,
        theta=init_theta,
        phi=init_phi,
        lam=init_lam,
        qubits=(0,),
    )
    body_block.stmts.append(init)
    em = place.EndMeasure(init.state_after, qubits=(0,))
    body_block.stmts.append(em)

    sp, outer = _wrap_in_static_placement(body_block, num_qubits=1)

    result = _run(outer)

    assert not result.has_done_something
    body_stmts = _body_stmts(sp)
    assert body_stmts == [init, em]
