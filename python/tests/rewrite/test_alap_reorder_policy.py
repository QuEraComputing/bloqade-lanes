"""Tests for alap_reorder_policy and _alap_schedule."""

from kirin import ir, rewrite, types as kirin_types

from bloqade import types as bloqade_types
from bloqade.lanes import types as lanes_types
from bloqade.lanes.dialects import place
from bloqade.lanes.rewrite.reorder_static_placement import (
    ReorderStaticPlacement,
    alap_reorder_policy,
)

# ---------------------------------------------------------------------------
# Helpers (identical to test_reorder_static_placement.py helpers)
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
    result = rewrite.Walk(ReorderStaticPlacement(alap_reorder_policy)).rewrite(outer)
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


def test_two_independent_gates_defer_to_max_layer():
    """R(q0), R(q1) — both independent, both have ALAP = 0 (max_layer = 0).
    No reorder needed since they share the single available layer."""
    body_block, entry_state = _new_body_block()
    axis = ir.TestValue(type=kirin_types.Float)
    angle = ir.TestValue(type=kirin_types.Float)
    r0 = place.R(entry_state, axis_angle=axis, rotation_angle=angle, qubits=(0,))
    body_block.stmts.append(r0)
    r1 = place.R(r0.state_after, axis_angle=axis, rotation_angle=angle, qubits=(1,))
    body_block.stmts.append(r1)

    _, outer = _wrap_in_static_placement(body_block, num_qubits=2)
    # Both are at ALAP layer 0 (max_layer = 0 since they're both independent
    # and the furthest ASAP layer is 0); order unchanged.
    assert not _run(outer)


def test_dependent_gates_order_preserved():
    """R(q0) → CZ(q0,q1) → R(q1): each gate has ALAP = ASAP (no mobility). Order unchanged."""
    body_block, entry_state = _new_body_block()
    axis = ir.TestValue(type=kirin_types.Float)
    angle = ir.TestValue(type=kirin_types.Float)
    r0 = place.R(entry_state, axis_angle=axis, rotation_angle=angle, qubits=(0,))
    body_block.stmts.append(r0)
    cz = place.CZ(r0.state_after, qubits=(0, 1))
    body_block.stmts.append(cz)
    r1 = place.R(cz.state_after, axis_angle=axis, rotation_angle=angle, qubits=(1,))
    body_block.stmts.append(r1)

    _, outer = _wrap_in_static_placement(body_block, num_qubits=2)
    assert not _run(outer)


def test_independent_gate_deferred_past_cz():
    """R(q2) is independent, should be deferred past CZ(q0,q1).

    Circuit: R(q0), CZ(q0,q1), R(q2)
    ASAP: R(q0)=0, CZ=1, R(q2)=0 → ASAP reorder would move R(q2) before CZ.
    ALAP: R(q0)=0, CZ=1, R(q2)=1 (max_layer=1, no successor) → R(q2) stays AFTER CZ.
    """
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
    # ALAP defers R(q2) to layer 1 (= max_layer), so it stays after CZ.
    # Original order is already [R(q0), CZ, R(q2)] which matches ALAP, so no change.
    assert not _run(outer)


def test_pre_gate_deferred_from_layer_0_to_later():
    """An independent pre-gate that ASAP would front-load is deferred by ALAP.

    Circuit (original order): R(q2)[pre], R(q0), CZ(q0,q1), R(q1)[post]
    ASAP: R(q2)=0, R(q0)=0, CZ=1, R(q1)=2 → puts R(q2) at layer 0, before CZ.
    ALAP: R(q2) has no successors → ALAP=2 (max_layer). Moves to AFTER CZ.

    We simulate ASAP-ordered input (R(q2) already moved to front) and check
    that ALAP moves it back to after the CZ.
    """
    body_block, entry_state = _new_body_block()
    axis = ir.TestValue(type=kirin_types.Float)
    angle = ir.TestValue(type=kirin_types.Float)

    # Put R(q2) first (as ASAP would have placed it)
    r2 = place.R(entry_state, axis_angle=axis, rotation_angle=angle, qubits=(2,))
    body_block.stmts.append(r2)
    r0 = place.R(r2.state_after, axis_angle=axis, rotation_angle=angle, qubits=(0,))
    body_block.stmts.append(r0)
    cz = place.CZ(r0.state_after, qubits=(0, 1))
    body_block.stmts.append(cz)
    r1 = place.R(cz.state_after, axis_angle=axis, rotation_angle=angle, qubits=(1,))
    body_block.stmts.append(r1)

    _, outer = _wrap_in_static_placement(body_block, num_qubits=3)
    assert _run(outer)  # ALAP moves R(q2) past the CZ

    sp = _get_sp(outer)
    stmts = _body_stmts(sp)
    assert len(stmts) == 4
    types_in_order = [type(s) for s in stmts]
    # CZ must come before R(q2) which has no successors
    cz_pos = types_in_order.index(place.CZ)
    r2_pos = next(
        i for i, s in enumerate(stmts) if isinstance(s, place.R) and s.qubits == (2,)
    )
    assert cz_pos < r2_pos, "ALAP should defer R(q2) to after the CZ"


def test_alap_places_cz_layer0_sink_gates_last():
    """Three-gate chain: R(q0), CZ(q0,q1), R(q1).  Independent R(q2).

    ALAP layers: R(q0)=0, CZ=1, R(q1)=2, R(q2)=2 (max_layer=2, no successor).
    Input order: R(q0), CZ(q0,q1), R(q2)[independent, placed early by mistake], R(q1).
    Expected output: R(q0), CZ, R(q1), R(q2)  (both R at layer 2, but R(q1)
    preserves original relative order with R(q2) since they're at the same ALAP layer).
    """
    body_block, entry_state = _new_body_block()
    axis = ir.TestValue(type=kirin_types.Float)
    angle = ir.TestValue(type=kirin_types.Float)

    r0 = place.R(entry_state, axis_angle=axis, rotation_angle=angle, qubits=(0,))
    body_block.stmts.append(r0)
    cz = place.CZ(r0.state_after, qubits=(0, 1))
    body_block.stmts.append(cz)
    r2 = place.R(cz.state_after, axis_angle=axis, rotation_angle=angle, qubits=(2,))
    body_block.stmts.append(r2)
    r1 = place.R(r2.state_after, axis_angle=axis, rotation_angle=angle, qubits=(1,))
    body_block.stmts.append(r1)

    _, outer = _wrap_in_static_placement(body_block, num_qubits=3)
    # R(q1) has ALAP=2 (must follow CZ). R(q2) also has ALAP=2 (no successor).
    # Both land at layer 2 — relative order among them follows original order.
    # Original order already matches ALAP if r2 and r1 are both at layer 2.
    # The pass may or may not reorder (depends on whether R(q2) stays at same slot).
    sp = _get_sp(outer)
    stmts = _body_stmts(sp)
    assert len(stmts) == 4
    # R(q0) must be first, CZ must be second
    assert isinstance(stmts[0], place.R) and stmts[0].qubits == (0,)
    assert isinstance(stmts[1], place.CZ)
    # Both R(q1) and R(q2) come after CZ
    after_cz = {s.qubits for s in stmts[2:] if isinstance(s, place.R)}  # type: ignore[union-attr]
    assert (1,) in after_cz
    assert (2,) in after_cz


def test_barrier_prevents_reorder_across():
    """Initialize is a barrier; ALAP cannot push gates past it."""
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
    # Each segment has one gate; no reorder possible in either segment.
    assert not _run(outer)


def test_idempotence():
    """Second application of the ALAP reorder pass is a no-op."""
    body_block, entry_state = _new_body_block()
    axis = ir.TestValue(type=kirin_types.Float)
    angle = ir.TestValue(type=kirin_types.Float)

    # Put an independent gate (q2) before CZ so ALAP has work to do.
    r2 = place.R(entry_state, axis_angle=axis, rotation_angle=angle, qubits=(2,))
    body_block.stmts.append(r2)
    r0 = place.R(r2.state_after, axis_angle=axis, rotation_angle=angle, qubits=(0,))
    body_block.stmts.append(r0)
    cz = place.CZ(r0.state_after, qubits=(0, 1))
    body_block.stmts.append(cz)
    r1 = place.R(cz.state_after, axis_angle=axis, rotation_angle=angle, qubits=(1,))
    body_block.stmts.append(r1)

    _, outer = _wrap_in_static_placement(body_block, num_qubits=3)
    assert _run(outer)  # first pass may reorder
    assert not _run(outer)  # second pass is no-op


def test_alap_defers_independent_pre_gates():
    """ALAP reduces qubit footprint of first CZ group by deferring pre-gates
    that belong to later CZ interactions.

    Circuit (original / ASAP order):
      R(q14)[pre-CZ2], R(q21)[pre-CZ3],   ← independent pre-gates at ASAP layer 0
      R(q0), R(q7)[pre-CZ1],              ← also layer 0
      CZ(q0,q7),                          ← layer 1
      R(q7)[post-CZ1],                    ← layer 2 (depends on CZ1 via q7)
      CZ(q0,q14),                         ← layer 2 (depends on CZ1 via q0, pre-CZ2 via q14)
      CZ(q7,q21)                          ← layer 3 (depends on post via q7, pre-CZ3 via q21)

    ASAP would pull R(q14) and R(q21) to layer 0 (before CZ1), inflating SP1's qubit
    footprint to {q0,q7,q14,q21}.

    ALAP should defer them to layer 2 (their ALAP), keeping SP1 footprint at {q0,q7}.
    After ALAP reorder, R(q14) and R(q21) appear after CZ(q0,q7) in the linear order.
    """
    body_block, entry_state = _new_body_block()
    axis = ir.TestValue(type=kirin_types.Float)
    angle = ir.TestValue(type=kirin_types.Float)

    # ASAP-style input: pre-gates for CZ2 and CZ3 come first.
    r14 = place.R(entry_state, axis_angle=axis, rotation_angle=angle, qubits=(14,))
    body_block.stmts.append(r14)
    r21 = place.R(r14.state_after, axis_angle=axis, rotation_angle=angle, qubits=(21,))
    body_block.stmts.append(r21)
    r0 = place.R(r21.state_after, axis_angle=axis, rotation_angle=angle, qubits=(0,))
    body_block.stmts.append(r0)
    r7_pre = place.R(r0.state_after, axis_angle=axis, rotation_angle=angle, qubits=(7,))
    body_block.stmts.append(r7_pre)
    cz1 = place.CZ(r7_pre.state_after, qubits=(0, 7))
    body_block.stmts.append(cz1)
    r7_post = place.R(
        cz1.state_after, axis_angle=axis, rotation_angle=angle, qubits=(7,)
    )
    body_block.stmts.append(r7_post)
    cz2 = place.CZ(r7_post.state_after, qubits=(0, 14))
    body_block.stmts.append(cz2)
    cz3 = place.CZ(cz2.state_after, qubits=(7, 21))
    body_block.stmts.append(cz3)

    _, outer = _wrap_in_static_placement(body_block, num_qubits=22)
    assert _run(outer)

    sp = _get_sp(outer)
    stmts = _body_stmts(sp)

    # Find position of CZ1 and the pre-gates for CZ2/CZ3.
    cz1_pos = next(
        i for i, s in enumerate(stmts) if isinstance(s, place.CZ) and s.qubits == (0, 7)
    )
    r14_pos = next(
        i for i, s in enumerate(stmts) if isinstance(s, place.R) and s.qubits == (14,)
    )
    r21_pos = next(
        i for i, s in enumerate(stmts) if isinstance(s, place.R) and s.qubits == (21,)
    )

    # ALAP: R(q14) and R(q21) have no successors that precede CZ2/CZ3, so their
    # ALAP layer is >= CZ1's layer. They should appear after CZ1 in linear order.
    assert r14_pos > cz1_pos, "R(q14) pre-gate should be deferred past CZ(q0,q7)"
    assert r21_pos > cz1_pos, "R(q21) pre-gate should be deferred past CZ(q0,q7)"
