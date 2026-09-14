"""Tests for the EliminateRz rewrite rule, on hand-built native-dialect IR."""

from typing import cast

import pytest
from bloqade.native.dialects import gate as native_gate
from kirin import ir, rewrite, types as kirin_types
from kirin.dialects import ilist, py

from bloqade import qubit as squin_qubit, types as bloqade_types
from bloqade.gemini.logical.dialects.operations import stmts as operations
from bloqade.lanes.rewrite.eliminate_rz import EliminateRz, EliminateRzError


def _qubits(block: ir.Block, count: int) -> list[ir.SSAValue]:
    """Append ``count`` qubit allocations to ``block`` and return their values."""
    values = []
    for _ in range(count):
        new = squin_qubit.stmts.New()
        block.stmts.append(new)
        values.append(new.result)
    return values


def _register(block: ir.Block, values) -> ir.SSAValue:
    reg = ilist.New(values=tuple(values), elem_type=bloqade_types.QubitType)
    block.stmts.append(reg)
    return reg.result


def _const(block: ir.Block, value: float) -> ir.SSAValue:
    const = py.Constant(value)
    block.stmts.append(const)
    return const.result


def _of_type(block: ir.Block, kind) -> list[ir.Statement]:
    return [stmt for stmt in block.stmts if isinstance(stmt, kind)]


def _axis(stmt) -> float:
    return stmt.axis_angle.owner.value.unwrap()


def _axis_value(stmt) -> ir.SSAValue:
    return stmt.axis_angle


def _run(block: ir.Block):
    """Drive the rule the way the pipeline does -- a forward Walk.

    Returns the rule (so tests can read the residual frame) and the result.
    """
    rule = EliminateRz()
    return rule, rewrite.Walk(rule).rewrite(block)


def test_rz_is_deleted():
    block = ir.Block()
    (q0,) = _qubits(block, 1)
    reg = _register(block, [q0])
    angle = _const(block, 0.25)
    block.stmts.append(native_gate.stmts.Rz(rotation_angle=angle, qubits=reg))

    _, result = _run(block)

    assert result.has_done_something
    assert _of_type(block, native_gate.stmts.Rz) == []


def test_r_axis_is_shifted_by_the_pending_frame():
    block = ir.Block()
    (q0,) = _qubits(block, 1)
    reg = _register(block, [q0])
    quarter = _const(block, 0.25)
    zero = _const(block, 0.0)
    block.stmts.append(native_gate.stmts.Rz(rotation_angle=quarter, qubits=reg))
    block.stmts.append(
        native_gate.stmts.R(axis_angle=zero, rotation_angle=quarter, qubits=reg)
    )

    _run(block)

    rs = _of_type(block, native_gate.stmts.R)
    assert len(rs) == 1
    # (0.0 - 0.25) mod 1 == 0.75
    assert _axis(rs[0]) == 0.75


def test_frames_accumulate_across_multiple_rz():
    block = ir.Block()
    (q0,) = _qubits(block, 1)
    reg = _register(block, [q0])
    quarter = _const(block, 0.25)
    half = _const(block, 0.5)
    zero = _const(block, 0.0)
    block.stmts.append(native_gate.stmts.Rz(rotation_angle=quarter, qubits=reg))
    block.stmts.append(native_gate.stmts.Rz(rotation_angle=half, qubits=reg))
    block.stmts.append(
        native_gate.stmts.R(axis_angle=zero, rotation_angle=quarter, qubits=reg)
    )

    _run(block)

    (r,) = _of_type(block, native_gate.stmts.R)
    # (0.0 - 0.75) mod 1 == 0.25
    assert _axis(r) == 0.25


def test_r_on_an_untouched_qubit_is_left_alone():
    block = ir.Block()
    q0, q1 = _qubits(block, 2)
    reg0, reg1 = _register(block, [q0]), _register(block, [q1])
    quarter = _const(block, 0.25)
    zero = _const(block, 0.0)
    block.stmts.append(native_gate.stmts.Rz(rotation_angle=quarter, qubits=reg0))
    original = native_gate.stmts.R(axis_angle=zero, rotation_angle=quarter, qubits=reg1)
    block.stmts.append(original)

    _run(block)

    assert _of_type(block, native_gate.stmts.R) == [
        original
    ], "an untouched qubit's R must be left in place, not rebuilt"


def test_cz_passes_through_and_the_frame_survives_it():
    block = ir.Block()
    q0, q1 = _qubits(block, 2)
    controls, targets = _register(block, [q0]), _register(block, [q1])
    quarter = _const(block, 0.25)
    block.stmts.append(native_gate.stmts.Rz(rotation_angle=quarter, qubits=controls))
    cz = native_gate.stmts.CZ(controls=controls, targets=targets)
    block.stmts.append(cz)

    rule, _ = _run(block)

    assert _of_type(block, native_gate.stmts.CZ) == [cz]
    # Diagonal, so it commutes with the frame exactly -- nothing changes.
    assert rule._frame[q0] == 0.25


def test_rule_is_idempotent():
    block = ir.Block()
    (q0,) = _qubits(block, 1)
    reg = _register(block, [q0])
    quarter = _const(block, 0.25)
    zero = _const(block, 0.0)
    block.stmts.append(native_gate.stmts.Rz(rotation_angle=quarter, qubits=reg))
    block.stmts.append(
        native_gate.stmts.R(axis_angle=zero, rotation_angle=quarter, qubits=reg)
    )

    _run(block)
    _, second = _run(block)

    assert not second.has_done_something


def test_non_angle_constant_is_not_reused_as_axis():
    block = ir.Block()
    (q0,) = _qubits(block, 1)
    reg = _register(block, [q0])
    # An unrelated constant (e.g. a loop bound or count) that happens to
    # normalize to the same key as the shifted axis below (7.0 % 1.0 == 0.0).
    seven = _const(block, 7.0)
    quarter = _const(block, 0.25)
    block.stmts.append(native_gate.stmts.Rz(rotation_angle=quarter, qubits=reg))
    block.stmts.append(
        native_gate.stmts.R(axis_angle=quarter, rotation_angle=quarter, qubits=reg)
    )

    _run(block)

    (r,) = _of_type(block, native_gate.stmts.R)
    # (0.25 - 0.25) mod 1 == 0.0, same cache key as 7.0's normalized value --
    # but 7.0 itself is not a normalized angle, so it must not be reused.
    assert _axis_value(r) is not seven
    assert _axis(r) == 0.0


def test_r_splits_when_its_qubits_carry_different_frames():
    """One pulse cannot carry two axis angles, so the statement must split."""
    block = ir.Block()
    q0, q1 = _qubits(block, 2)
    only_q0 = _register(block, [q0])
    both = _register(block, [q0, q1])
    quarter = _const(block, 0.25)
    zero = _const(block, 0.0)
    block.stmts.append(native_gate.stmts.Rz(rotation_angle=quarter, qubits=only_q0))
    block.stmts.append(
        native_gate.stmts.R(axis_angle=zero, rotation_angle=quarter, qubits=both)
    )

    _run(block)

    rs = _of_type(block, native_gate.stmts.R)
    assert len(rs) == 2
    rs = cast(list[native_gate.stmts.R], rs)
    by_axis = {_axis(r): tuple(r.qubits.owner.values) for r in rs}  # type: ignore[attr-defined]
    assert by_axis == {0.75: (q0,), 0.0: (q1,)}


def test_split_covers_every_original_qubit_exactly_once():
    block = ir.Block()
    q0, q1, q2 = _qubits(block, 3)
    only_q0 = _register(block, [q0])
    only_q2 = _register(block, [q2])
    all_three = _register(block, [q0, q1, q2])
    quarter = _const(block, 0.25)
    half = _const(block, 0.5)
    zero = _const(block, 0.0)
    block.stmts.append(native_gate.stmts.Rz(rotation_angle=quarter, qubits=only_q0))
    block.stmts.append(native_gate.stmts.Rz(rotation_angle=half, qubits=only_q2))
    block.stmts.append(
        native_gate.stmts.R(axis_angle=zero, rotation_angle=quarter, qubits=all_three)
    )

    _run(block)

    rs = cast(list[native_gate.stmts.R], _of_type(block, native_gate.stmts.R))
    covered = [value for r in rs for value in r.qubits.owner.values]  # type: ignore[attr-defined]
    assert sorted(map(id, covered)) == sorted(map(id, [q0, q1, q2]))


def test_qubits_with_equal_frames_stay_in_one_statement():
    block = ir.Block()
    q0, q1 = _qubits(block, 2)
    both = _register(block, [q0, q1])
    quarter = _const(block, 0.25)
    zero = _const(block, 0.0)
    block.stmts.append(native_gate.stmts.Rz(rotation_angle=quarter, qubits=both))
    block.stmts.append(
        native_gate.stmts.R(axis_angle=zero, rotation_angle=quarter, qubits=both)
    )

    _run(block)

    rs = cast(list[native_gate.stmts.R], _of_type(block, native_gate.stmts.R))
    assert len(rs) == 1
    assert tuple(rs[0].qubits.owner.values) == (q0, q1)  # type: ignore[attr-defined]


def test_equal_angles_share_one_constant_ssa_value():
    """FuseAdjacentGates matches on SSA identity, so equal angles must share."""
    block = ir.Block()
    q0, q1 = _qubits(block, 2)
    reg0, reg1 = _register(block, [q0]), _register(block, [q1])
    both = _register(block, [q0, q1])
    quarter = _const(block, 0.25)
    zero = _const(block, 0.0)
    block.stmts.append(native_gate.stmts.Rz(rotation_angle=quarter, qubits=both))
    block.stmts.append(
        native_gate.stmts.R(axis_angle=zero, rotation_angle=quarter, qubits=reg0)
    )
    block.stmts.append(
        native_gate.stmts.R(axis_angle=zero, rotation_angle=quarter, qubits=reg1)
    )

    _run(block)

    rs = cast(list[native_gate.stmts.R], _of_type(block, native_gate.stmts.R))
    assert len(rs) == 2
    assert rs[0].axis_angle is rs[1].axis_angle


def test_an_existing_constant_is_reused_rather_than_duplicated():
    """The shifted angle already exists in the block, so no new constant."""
    block = ir.Block()
    (q0,) = _qubits(block, 1)
    reg = _register(block, [q0])
    quarter = _const(block, 0.25)
    three_quarter = _const(block, 0.75)
    zero = _const(block, 0.0)
    block.stmts.append(native_gate.stmts.Rz(rotation_angle=quarter, qubits=reg))
    block.stmts.append(
        native_gate.stmts.R(axis_angle=zero, rotation_angle=quarter, qubits=reg)
    )

    _run(block)

    (r,) = cast(list[native_gate.stmts.R], _of_type(block, native_gate.stmts.R))
    assert r.axis_angle is three_quarter


def test_statement_carrying_a_region_raises():
    """Un-unrolled control flow must not be silently scanned past.

    This rule runs before scf2cf, so a surviving scf.For is a statement with a
    region inside one block -- not a second block. Its body closes over qubit
    values instead of taking them as arguments, so the qubit-reachability guard
    would miss it and the gates inside would vanish from the scan.
    """
    from kirin.dialects import scf

    block = ir.Block()
    _qubits(block, 1)
    block.stmts.append(
        scf.For(ir.TestValue(type=kirin_types.Any), ir.Region(ir.Block()))
    )

    with pytest.raises(EliminateRzError, match="region"):
        _run(block)


def test_multi_block_region_raises():
    """A phase frame cannot cross a block boundary.

    It has no IR representation to travel in -- unlike the state that
    stack_move2move and state.py thread through block arguments -- so a second
    block would start from zero and silently lose the first block's phases.
    """
    region = ir.Region(ir.Block())
    region.blocks.append(ir.Block())

    with pytest.raises(EliminateRzError, match="single-block"):
        EliminateRz().rewrite_Region(region)


def test_unknown_statement_touching_a_qubit_raises():
    """The scan cannot know whether an unrecognised gate is diagonal."""
    block = ir.Block()
    (q0,) = _qubits(block, 1)
    reg = _register(block, [q0])
    block.stmts.append(squin_qubit.stmts.Measure(qubits=reg))

    with pytest.raises(EliminateRzError, match="phase-neutral"):
        _run(block)


def test_non_ilist_register_raises():
    block = ir.Block()
    angle = _const(block, 0.25)
    opaque = ir.TestValue(
        type=ilist.IListType[bloqade_types.QubitType, kirin_types.Any]
    )
    block.stmts.append(native_gate.stmts.Rz(rotation_angle=angle, qubits=opaque))

    with pytest.raises(EliminateRzError, match="ilist.New"):
        _run(block)


def test_duplicate_qubit_in_one_register_raises():
    block = ir.Block()
    (q0,) = _qubits(block, 1)
    doubled = _register(block, [q0, q0])
    angle = _const(block, 0.25)
    block.stmts.append(native_gate.stmts.Rz(rotation_angle=angle, qubits=doubled))

    with pytest.raises(EliminateRzError, match="more than once"):
        _run(block)


def test_non_constant_angle_raises():
    block = ir.Block()
    (q0,) = _qubits(block, 1)
    reg = _register(block, [q0])
    angle = ir.TestValue(type=kirin_types.Float)
    block.stmts.append(native_gate.stmts.Rz(rotation_angle=angle, qubits=reg))

    with pytest.raises(EliminateRzError, match="constant"):
        _run(block)


def test_initialize_with_a_pending_frame_raises():
    """Initialize sits at the head of a wire; a mid-wire one would drop a phase."""
    block = ir.Block()
    (q0,) = _qubits(block, 1)
    reg = _register(block, [q0])
    quarter = _const(block, 0.25)
    zero = _const(block, 0.0)
    block.stmts.append(native_gate.stmts.Rz(rotation_angle=quarter, qubits=reg))
    block.stmts.append(
        operations.Initialize(theta=zero, phi=zero, lam=zero, qubits=reg)
    )

    with pytest.raises(EliminateRzError, match="Initialize"):
        _run(block)


def test_initialize_at_the_head_of_a_wire_is_fine():
    block = ir.Block()
    (q0,) = _qubits(block, 1)
    reg = _register(block, [q0])
    zero = _const(block, 0.0)
    init = operations.Initialize(theta=zero, phi=zero, lam=zero, qubits=reg)
    block.stmts.append(init)

    _run(block)

    assert _of_type(block, operations.Initialize) == [init]


def test_star_rz_is_untouched_and_the_frame_commutes_past_it():
    block = ir.Block()
    (q0,) = _qubits(block, 1)
    reg = _register(block, [q0])
    quarter = _const(block, 0.25)
    star_angle = _const(block, 0.03)
    block.stmts.append(native_gate.stmts.Rz(rotation_angle=quarter, qubits=reg))
    star = operations.StarRz(rotation_angle=star_angle, qubits=reg)
    block.stmts.append(star)

    rule, _ = _run(block)

    assert _of_type(block, operations.StarRz) == [star]
    assert rule._frame[q0] == 0.25


def test_terminal_measurement_discards_the_frame():
    block = ir.Block()
    (q0,) = _qubits(block, 1)
    reg = _register(block, [q0])
    quarter = _const(block, 0.25)
    block.stmts.append(native_gate.stmts.Rz(rotation_angle=quarter, qubits=reg))
    block.stmts.append(operations.TerminalLogicalMeasurement(qubits=reg))

    rule, _ = _run(block)

    assert rule._frame == {}


def test_frame_left_at_end_of_block_is_simply_discarded():
    """The shape left by RemovePostProcessing: no measurement statement at all."""
    block = ir.Block()
    (q0,) = _qubits(block, 1)
    reg = _register(block, [q0])
    quarter = _const(block, 0.25)
    block.stmts.append(native_gate.stmts.Rz(rotation_angle=quarter, qubits=reg))

    rule, result = _run(block)

    assert result.has_done_something
    assert _of_type(block, native_gate.stmts.Rz) == []
    assert rule._frame == {q0: 0.25}  # residual, discarded by the caller
