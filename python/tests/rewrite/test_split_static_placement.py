"""Tests for SplitStaticPlacement rewrite rule and cz_layer_split_policy."""

from kirin import ir, rewrite, types as kirin_types

from bloqade import types as bloqade_types
from bloqade.lanes import types as lanes_types
from bloqade.lanes.dialects import place
from bloqade.lanes.rewrite.split_static_placement import (
    SplitStaticPlacement,
    cz_layer_split_policy,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _new_body_block() -> tuple[ir.Block, ir.SSAValue]:
    block = ir.Block()
    entry_state = block.args.append_from(lanes_types.StateType, name="entry_state")
    return block, entry_state


def _make_sp(
    num_qubits: int = 4,
) -> tuple[ir.Block, ir.SSAValue, tuple[ir.SSAValue, ...]]:
    """Return (body_block, entry_state, sp_qubits). Caller appends statements
    then calls _finalize_sp to create the StaticPlacement and outer block."""
    body_block, entry_state = _new_body_block()
    sp_qubits: tuple[ir.SSAValue, ...] = tuple(
        ir.TestValue(type=bloqade_types.QubitType) for _ in range(num_qubits)
    )
    return body_block, entry_state, sp_qubits


def _finalize_sp(
    body_block: ir.Block,
    sp_qubits: tuple[ir.SSAValue, ...],
    final_state: ir.SSAValue,
) -> tuple[place.StaticPlacement, ir.Block]:
    body_block.stmts.append(place.Yield(final_state))
    sp = place.StaticPlacement(qubits=sp_qubits, body=ir.Region(body_block))
    outer = ir.Block([sp])
    return sp, outer


def _run(outer: ir.Block) -> bool:
    result = rewrite.Walk(SplitStaticPlacement(cz_layer_split_policy)).rewrite(outer)
    return result.has_done_something


def _get_all_sps(outer: ir.Block) -> list[place.StaticPlacement]:
    return [s for s in outer.stmts if isinstance(s, place.StaticPlacement)]


def _body_stmts(sp: place.StaticPlacement) -> list[ir.Statement]:
    stmts = list(sp.body.blocks[0].stmts)
    assert stmts and isinstance(stmts[-1], place.Yield)
    return stmts[:-1]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_cz_no_split():
    """A body with only SQ gates and no CZ is not split."""
    body_block, entry_state, sp_qubits = _make_sp(num_qubits=2)
    axis = ir.TestValue(type=kirin_types.Float)
    angle = ir.TestValue(type=kirin_types.Float)

    r0 = place.R(entry_state, axis_angle=axis, rotation_angle=angle, qubits=(0,))
    body_block.stmts.append(r0)
    r1 = place.R(r0.state_after, axis_angle=axis, rotation_angle=angle, qubits=(1,))
    body_block.stmts.append(r1)

    _, outer = _finalize_sp(body_block, sp_qubits, r1.state_after)
    assert not _run(outer)
    assert len(_get_all_sps(outer)) == 1


def test_single_cz_groups_preceding_sq():
    """R(q0), R(q1), CZ(q0,q1) all go into one group — no split."""
    body_block, entry_state, sp_qubits = _make_sp(num_qubits=2)
    axis = ir.TestValue(type=kirin_types.Float)
    angle = ir.TestValue(type=kirin_types.Float)

    r0 = place.R(entry_state, axis_angle=axis, rotation_angle=angle, qubits=(0,))
    body_block.stmts.append(r0)
    r1 = place.R(r0.state_after, axis_angle=axis, rotation_angle=angle, qubits=(1,))
    body_block.stmts.append(r1)
    cz = place.CZ(r1.state_after, qubits=(0, 1))
    body_block.stmts.append(cz)

    _, outer = _finalize_sp(body_block, sp_qubits, cz.state_after)
    assert not _run(outer)
    assert len(_get_all_sps(outer)) == 1


def test_two_cz_layers_produce_two_groups():
    """R(q0), CZ(q0,q1), R(q2), CZ(q2,q3) → two StaticPlacements."""
    body_block, entry_state, sp_qubits = _make_sp(num_qubits=4)
    axis = ir.TestValue(type=kirin_types.Float)
    angle = ir.TestValue(type=kirin_types.Float)

    r0 = place.R(entry_state, axis_angle=axis, rotation_angle=angle, qubits=(0,))
    body_block.stmts.append(r0)
    cz01 = place.CZ(r0.state_after, qubits=(0, 1))
    body_block.stmts.append(cz01)
    r2 = place.R(cz01.state_after, axis_angle=axis, rotation_angle=angle, qubits=(2,))
    body_block.stmts.append(r2)
    cz23 = place.CZ(r2.state_after, qubits=(2, 3))
    body_block.stmts.append(cz23)

    _, outer = _finalize_sp(body_block, sp_qubits, cz23.state_after)
    assert _run(outer)

    sps = _get_all_sps(outer)
    assert len(sps) == 2

    # SP1: R(q0), CZ(q0,q1)
    s1 = _body_stmts(sps[0])
    assert len(s1) == 2
    assert isinstance(s1[0], place.R) and s1[0].qubits == (0,)  # type: ignore[union-attr]
    assert isinstance(s1[1], place.CZ) and s1[1].qubits == (0, 1)  # type: ignore[union-attr]

    # SP2: R(q2), CZ(q2,q3)
    s2 = _body_stmts(sps[1])
    assert len(s2) == 2
    assert isinstance(s2[0], place.R) and s2[0].qubits == (2,)  # type: ignore[union-attr]
    assert isinstance(s2[1], place.CZ) and s2[1].qubits == (2, 3)  # type: ignore[union-attr]


def test_trailing_sq_after_last_cz_forms_own_group():
    """R(q0), CZ(q0,q1), R(q2) → SP1=[R(q0),CZ], SP2=[R(q2)]."""
    body_block, entry_state, sp_qubits = _make_sp(num_qubits=3)
    axis = ir.TestValue(type=kirin_types.Float)
    angle = ir.TestValue(type=kirin_types.Float)

    r0 = place.R(entry_state, axis_angle=axis, rotation_angle=angle, qubits=(0,))
    body_block.stmts.append(r0)
    cz = place.CZ(r0.state_after, qubits=(0, 1))
    body_block.stmts.append(cz)
    r2 = place.R(cz.state_after, axis_angle=axis, rotation_angle=angle, qubits=(2,))
    body_block.stmts.append(r2)

    _, outer = _finalize_sp(body_block, sp_qubits, r2.state_after)
    assert _run(outer)

    sps = _get_all_sps(outer)
    assert len(sps) == 2

    s1 = _body_stmts(sps[0])
    assert len(s1) == 2
    assert isinstance(s1[0], place.R) and s1[0].qubits == (0,)  # type: ignore[union-attr]
    assert isinstance(s1[1], place.CZ)

    s2 = _body_stmts(sps[1])
    assert len(s2) == 1
    assert isinstance(s2[0], place.R) and s2[0].qubits == (2,)  # type: ignore[union-attr]


def test_policy_a_full_example():
    """The 8-statement example from the design doc splits into 3 StaticPlacements.

    Input body (ASAP order):
        R(q2), R(q0), R(q1), CZ(q0,q1), R(q0), R(q3), CZ(q2,q3), R(q1)

    Expected groups:
        SP1: R(q2), R(q0), R(q1), CZ(q0,q1)
        SP2: R(q0), R(q3), CZ(q2,q3)
        SP3: R(q1)
    """
    body_block, entry_state, sp_qubits = _make_sp(num_qubits=4)
    axis = ir.TestValue(type=kirin_types.Float)
    angle = ir.TestValue(type=kirin_types.Float)

    s = entry_state
    r_q2 = place.R(s, axis_angle=axis, rotation_angle=angle, qubits=(2,))
    body_block.stmts.append(r_q2)
    s = r_q2.state_after

    r_q0a = place.R(s, axis_angle=axis, rotation_angle=angle, qubits=(0,))
    body_block.stmts.append(r_q0a)
    s = r_q0a.state_after

    r_q1a = place.R(s, axis_angle=axis, rotation_angle=angle, qubits=(1,))
    body_block.stmts.append(r_q1a)
    s = r_q1a.state_after

    cz01 = place.CZ(s, qubits=(0, 1))
    body_block.stmts.append(cz01)
    s = cz01.state_after

    r_q0b = place.R(s, axis_angle=axis, rotation_angle=angle, qubits=(0,))
    body_block.stmts.append(r_q0b)
    s = r_q0b.state_after

    r_q3 = place.R(s, axis_angle=axis, rotation_angle=angle, qubits=(3,))
    body_block.stmts.append(r_q3)
    s = r_q3.state_after

    cz23 = place.CZ(s, qubits=(2, 3))
    body_block.stmts.append(cz23)
    s = cz23.state_after

    r_q1b = place.R(s, axis_angle=axis, rotation_angle=angle, qubits=(1,))
    body_block.stmts.append(r_q1b)
    s = r_q1b.state_after

    _, outer = _finalize_sp(body_block, sp_qubits, s)
    assert _run(outer)

    sps = _get_all_sps(outer)
    assert len(sps) == 3

    # SP1: R(2), R(0), R(1), CZ(0,1)
    s1 = _body_stmts(sps[0])
    assert len(s1) == 4
    assert isinstance(s1[0], place.R) and s1[0].qubits == (2,)  # type: ignore[union-attr]
    assert isinstance(s1[1], place.R) and s1[1].qubits == (0,)  # type: ignore[union-attr]
    assert isinstance(s1[2], place.R) and s1[2].qubits == (1,)  # type: ignore[union-attr]
    assert isinstance(s1[3], place.CZ) and s1[3].qubits == (0, 1)  # type: ignore[union-attr]

    # SP2: R(0), R(3), CZ(2,3)
    s2 = _body_stmts(sps[1])
    assert len(s2) == 3
    assert isinstance(s2[0], place.R) and s2[0].qubits == (0,)  # type: ignore[union-attr]
    assert isinstance(s2[1], place.R) and s2[1].qubits == (3,)  # type: ignore[union-attr]
    assert isinstance(s2[2], place.CZ) and s2[2].qubits == (2, 3)  # type: ignore[union-attr]

    # SP3: R(1)
    s3 = _body_stmts(sps[2])
    assert len(s3) == 1
    assert isinstance(s3[0], place.R) and s3[0].qubits == (1,)  # type: ignore[union-attr]
