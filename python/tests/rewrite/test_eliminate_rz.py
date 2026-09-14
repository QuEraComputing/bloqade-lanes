"""Tests for the EliminateRz rewrite rule, on hand-built native-dialect IR."""

from bloqade.native.dialects import gate as native_gate
from kirin import ir, rewrite
from kirin.dialects import ilist, py

from bloqade import qubit as squin_qubit, types as bloqade_types
from bloqade.lanes.rewrite.eliminate_rz import EliminateRz


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
