"""Tests for ReorderStaticPlacement rewrite rule and asap_reorder_policy."""

from kirin import ir, rewrite, types as kirin_types

from bloqade import types as bloqade_types
from bloqade.lanes import types as lanes_types
from bloqade.lanes.dialects import place
from bloqade.lanes.rewrite.reorder_static_placement import (
    ReorderStaticPlacement,
    asap_reorder_policy,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _new_body_block() -> tuple[ir.Block, ir.SSAValue]:
    block = ir.Block()
    entry_state = block.args.append_from(lanes_types.StateType, name="entry_state")
    return block, entry_state


def _wrap_in_static_placement(
    body_block: ir.Block, num_qubits: int = 4
) -> tuple[place.StaticPlacement, ir.Block]:
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


def _run(outer: ir.Block) -> bool:
    result = rewrite.Walk(ReorderStaticPlacement(asap_reorder_policy)).rewrite(outer)
    return result.has_done_something


def _get_sp(outer: ir.Block) -> place.StaticPlacement:
    for stmt in outer.stmts:
        if isinstance(stmt, place.StaticPlacement):
            return stmt
    raise AssertionError("No StaticPlacement in outer block")


def _body_stmts(sp: place.StaticPlacement) -> list[ir.Statement]:
    stmts = list(sp.body.blocks[0].stmts)
    assert stmts and isinstance(stmts[-1], place.Yield)
    return stmts[:-1]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_single_stmt_unchanged():
    """A body with one statement is never reordered."""
    body_block, entry_state = _new_body_block()
    axis = ir.TestValue(type=kirin_types.Float)
    angle = ir.TestValue(type=kirin_types.Float)
    r = place.R(entry_state, axis_angle=axis, rotation_angle=angle, qubits=(0,))
    body_block.stmts.append(r)

    _, outer = _wrap_in_static_placement(body_block, num_qubits=1)
    assert not _run(outer)


def test_two_independent_gates_already_optimal():
    """R(q0), R(q1) are both layer 0 — already optimal, no reorder."""
    body_block, entry_state = _new_body_block()
    axis = ir.TestValue(type=kirin_types.Float)
    angle = ir.TestValue(type=kirin_types.Float)
    r0 = place.R(entry_state, axis_angle=axis, rotation_angle=angle, qubits=(0,))
    body_block.stmts.append(r0)
    r1 = place.R(r0.state_after, axis_angle=axis, rotation_angle=angle, qubits=(1,))
    body_block.stmts.append(r1)

    _, outer = _wrap_in_static_placement(body_block, num_qubits=2)
    assert not _run(outer)


def test_dependent_gate_not_moved_before_predecessor():
    """R(q0), CZ(q0,q1), R(q1): CZ depends on R(q0); R(q1) depends on CZ. Order unchanged."""
    body_block, entry_state = _new_body_block()
    axis = ir.TestValue(type=kirin_types.Float)
    angle = ir.TestValue(type=kirin_types.Float)
    r0 = place.R(entry_state, axis_angle=axis, rotation_angle=angle, qubits=(0,))
    body_block.stmts.append(r0)
    cz = place.CZ(r0.state_after, qubits=(0, 1))  # control=0, target=1
    body_block.stmts.append(cz)
    r1 = place.R(cz.state_after, axis_angle=axis, rotation_angle=angle, qubits=(1,))
    body_block.stmts.append(r1)

    _, outer = _wrap_in_static_placement(body_block, num_qubits=2)
    assert not _run(outer)


def test_independent_gate_moves_earlier():
    """R(q0), R(q1), CZ(q0,q1), R(q2): R(q2) is layer 0, moves before CZ which is layer 1."""
    body_block, entry_state = _new_body_block()
    axis = ir.TestValue(type=kirin_types.Float)
    angle = ir.TestValue(type=kirin_types.Float)
    r0 = place.R(entry_state, axis_angle=axis, rotation_angle=angle, qubits=(0,))
    body_block.stmts.append(r0)
    r1 = place.R(r0.state_after, axis_angle=axis, rotation_angle=angle, qubits=(1,))
    body_block.stmts.append(r1)
    cz = place.CZ(r1.state_after, qubits=(0, 1))  # control=0, target=1
    body_block.stmts.append(cz)
    r2 = place.R(cz.state_after, axis_angle=axis, rotation_angle=angle, qubits=(2,))
    body_block.stmts.append(r2)

    _, outer = _wrap_in_static_placement(body_block, num_qubits=3)
    assert _run(outer)  # r2 moved before CZ

    sp = _get_sp(outer)
    stmts = _body_stmts(sp)
    assert len(stmts) == 4
    # Three Rs come before CZ
    types_in_order = [type(s) for s in stmts]
    assert types_in_order == [place.R, place.R, place.R, place.CZ]
    # R(q2) is among the first three; CZ is last
    r_qubits = {s.qubits for s in stmts[:3]}  # type: ignore[union-attr]
    assert (2,) in r_qubits
    assert stmts[3].qubits == (0, 1)  # type: ignore[union-attr]


def test_barrier_prevents_reorder_across():
    """R(q0), Initialize(q1), R(q0): Initialize is a barrier; second R stays after it."""
    body_block, entry_state = _new_body_block()
    axis = ir.TestValue(type=kirin_types.Float)
    angle = ir.TestValue(type=kirin_types.Float)
    theta = ir.TestValue(type=kirin_types.Float)
    phi = ir.TestValue(type=kirin_types.Float)
    lam = ir.TestValue(type=kirin_types.Float)

    r0 = place.R(entry_state, axis_angle=axis, rotation_angle=angle, qubits=(0,))
    body_block.stmts.append(r0)
    init = place.Initialize(r0.state_after, theta=theta, phi=phi, lam=lam, qubits=(1,))
    body_block.stmts.append(init)
    r0b = place.R(init.state_after, axis_angle=axis, rotation_angle=angle, qubits=(0,))
    body_block.stmts.append(r0b)

    _, outer = _wrap_in_static_placement(body_block, num_qubits=2)
    # Both segments are length 1 → no reorder possible
    assert not _run(outer)


def test_multiple_layers_correct_ordering():
    """R(q0), CZ(q0,q1), R(q2): R(q2) is layer 0 and moves before CZ at layer 1."""
    body_block, entry_state = _new_body_block()
    axis = ir.TestValue(type=kirin_types.Float)
    angle = ir.TestValue(type=kirin_types.Float)

    r0 = place.R(entry_state, axis_angle=axis, rotation_angle=angle, qubits=(0,))
    body_block.stmts.append(r0)
    cz = place.CZ(r0.state_after, qubits=(0, 1))  # control=0, target=1
    body_block.stmts.append(cz)
    r2 = place.R(cz.state_after, axis_angle=axis, rotation_angle=angle, qubits=(2,))
    body_block.stmts.append(r2)

    _, outer = _wrap_in_static_placement(body_block, num_qubits=3)
    assert _run(outer)  # r2 moves before cz

    sp = _get_sp(outer)
    stmts = _body_stmts(sp)
    assert len(stmts) == 3
    assert isinstance(stmts[0], place.R) and stmts[0].qubits == (0,)  # type: ignore[union-attr]
    assert isinstance(stmts[1], place.R) and stmts[1].qubits == (2,)  # type: ignore[union-attr]
    assert isinstance(stmts[2], place.CZ)


def test_idempotence():
    """Second application of the reorder pass is a no-op."""
    body_block, entry_state = _new_body_block()
    axis = ir.TestValue(type=kirin_types.Float)
    angle = ir.TestValue(type=kirin_types.Float)

    r0 = place.R(entry_state, axis_angle=axis, rotation_angle=angle, qubits=(0,))
    body_block.stmts.append(r0)
    cz = place.CZ(r0.state_after, qubits=(0, 1))
    body_block.stmts.append(cz)
    r2 = place.R(cz.state_after, axis_angle=axis, rotation_angle=angle, qubits=(2,))
    body_block.stmts.append(r2)

    _, outer = _wrap_in_static_placement(body_block, num_qubits=3)

    assert _run(outer)  # first pass reorders
    assert not _run(outer)  # second pass is no-op


def test_same_type_gates_cluster_within_layer():
    """R(q0,a,b), Rz(q1,c), R(q2,a,b) all at layer 0: the two R gates cluster together.

    Both R gates share the same axis_angle and rotation_angle SSA values.
    _group_within_layer opens the (R, id(a), id(b)) group on the first R,
    then Rz opens a new group, then the second R lands in the first group.
    Result: R(q0), R(q2), Rz(q1).
    """
    body_block, entry_state = _new_body_block()
    axis = ir.TestValue(type=kirin_types.Float)
    angle_r = ir.TestValue(type=kirin_types.Float)
    angle_rz = ir.TestValue(type=kirin_types.Float)

    r0 = place.R(entry_state, axis_angle=axis, rotation_angle=angle_r, qubits=(0,))
    body_block.stmts.append(r0)
    rz1 = place.Rz(r0.state_after, rotation_angle=angle_rz, qubits=(1,))
    body_block.stmts.append(rz1)
    r2 = place.R(rz1.state_after, axis_angle=axis, rotation_angle=angle_r, qubits=(2,))
    body_block.stmts.append(r2)

    _, outer = _wrap_in_static_placement(body_block, num_qubits=3)
    assert _run(outer)  # order changes: R(q2) moves before Rz(q1)

    sp = _get_sp(outer)
    stmts = _body_stmts(sp)
    assert len(stmts) == 3
    assert isinstance(stmts[0], place.R) and stmts[0].qubits == (0,)  # type: ignore[union-attr]
    assert isinstance(stmts[1], place.R) and stmts[1].qubits == (2,)  # type: ignore[union-attr]
    assert isinstance(stmts[2], place.Rz) and stmts[2].qubits == (1,)  # type: ignore[union-attr]
