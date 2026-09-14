"""The rule must preserve the unitary up to the residual frame it leaves behind.

    U_before == Rz(residual) . U_after      (up to global phase)

A sign error in the commutation identity passes every structural test, so this
builds both circuits as matrices from the IR itself and compares them.
"""

from typing import cast

import numpy as np
import pytest
from bloqade.native.dialects import gate as native_gate
from kirin import ir, rewrite
from kirin.dialects import ilist, py

from bloqade import qubit as squin_qubit, types as bloqade_types
from bloqade.lanes.rewrite.eliminate_rz import EliminateRz

_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]])


def _r(axis: float, rotation: float) -> np.ndarray:
    """R(axis, rotation), both angles in turns."""
    phi, theta = 2 * np.pi * axis, 2 * np.pi * rotation
    return np.cos(theta / 2) * _I - 1j * np.sin(theta / 2) * (
        np.cos(phi) * _X + np.sin(phi) * _Y
    )


def _rz(angle: float) -> np.ndarray:
    return np.diag([1.0, np.exp(1j * 2 * np.pi * angle)]).astype(complex)


def _embed(op: np.ndarray, target: int, num_qubits: int) -> np.ndarray:
    out = np.array([[1.0]], dtype=complex)
    for index in range(num_qubits):
        out = np.kron(out, op if index == target else _I)
    return out


def _cz(num_qubits: int, a: int, b: int) -> np.ndarray:
    dim = 2**num_qubits
    diagonal = np.ones(dim, dtype=complex)
    for state in range(dim):
        if (state >> (num_qubits - 1 - a)) & 1 and (state >> (num_qubits - 1 - b)) & 1:
            diagonal[state] = -1.0
    return np.diag(diagonal)


def _float_operand(value: ir.SSAValue) -> float:
    """Read the float literal out of a ``py.Constant``-owned SSA value."""
    owner = cast(py.Constant, value.owner)
    return float(owner.value.unwrap())


def _qubit_operands(register: ir.SSAValue) -> tuple[ir.SSAValue, ...]:
    """Read the qubit values out of an ``ilist.New``-owned SSA value."""
    owner = cast(ilist.New, register.owner)
    return tuple(owner.values)


def _unitary(block: ir.Block, order: list[ir.SSAValue]) -> np.ndarray:
    """Build the block's unitary. ``order`` fixes the qubit tensor ordering."""
    index = {value: position for position, value in enumerate(order)}
    num_qubits = len(order)
    total = np.eye(2**num_qubits, dtype=complex)
    for stmt in block.stmts:
        if isinstance(stmt, native_gate.stmts.Rz):
            angle = _float_operand(stmt.rotation_angle)
            for value in _qubit_operands(stmt.qubits):
                total = _embed(_rz(angle), index[value], num_qubits) @ total
        elif isinstance(stmt, native_gate.stmts.R):
            axis = _float_operand(stmt.axis_angle)
            rotation = _float_operand(stmt.rotation_angle)
            for value in _qubit_operands(stmt.qubits):
                total = _embed(_r(axis, rotation), index[value], num_qubits) @ total
        elif isinstance(stmt, native_gate.stmts.CZ):
            (control,) = _qubit_operands(stmt.controls)
            (target,) = _qubit_operands(stmt.targets)
            total = _cz(num_qubits, index[control], index[target]) @ total
    return total


def _equal_up_to_phase(lhs: np.ndarray, rhs: np.ndarray) -> bool:
    position = np.unravel_index(np.argmax(np.abs(lhs)), lhs.shape)
    return np.allclose(lhs / lhs[position], rhs / rhs[position], atol=1e-9)


def _qubits(block: ir.Block, count: int) -> list[ir.SSAValue]:
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


def _run(block: ir.Block):
    """Drive the rule the way the pipeline does -- a forward Walk.

    Returns the rule (so tests can read the residual frame) and the result.
    """
    rule = EliminateRz()
    return rule, rewrite.Walk(rule).rewrite(block)


def _teleportation_block():
    """The h/s/cx kernel's native gate sequence, acting on logical qubit 1."""
    block = ir.Block()
    q0, q1 = _qubits(block, 2)
    r0, r1 = _register(block, [q0]), _register(block, [q1])
    minus_quarter = _const(block, -0.25)
    zero = _const(block, 0.0)
    quarter = _const(block, 0.25)
    add = block.stmts.append
    add(native_gate.stmts.Rz(rotation_angle=minus_quarter, qubits=r1))
    add(native_gate.stmts.R(axis_angle=zero, rotation_angle=minus_quarter, qubits=r1))
    add(native_gate.stmts.Rz(rotation_angle=minus_quarter, qubits=r1))
    add(native_gate.stmts.Rz(rotation_angle=minus_quarter, qubits=r1))
    add(
        native_gate.stmts.R(axis_angle=quarter, rotation_angle=minus_quarter, qubits=r1)
    )
    add(native_gate.stmts.CZ(controls=r0, targets=r1))
    add(native_gate.stmts.R(axis_angle=quarter, rotation_angle=quarter, qubits=r1))
    return block, [q0, q1]


def _split_forcing_block():
    """Frames diverge across a broadcast R, forcing a split."""
    block = ir.Block()
    q0, q1 = _qubits(block, 2)
    only_q0 = _register(block, [q0])
    both = _register(block, [q0, q1])
    r0, r1 = _register(block, [q0]), _register(block, [q1])
    quarter = _const(block, 0.25)
    zero = _const(block, 0.0)
    add = block.stmts.append
    add(native_gate.stmts.Rz(rotation_angle=quarter, qubits=only_q0))
    add(native_gate.stmts.R(axis_angle=zero, rotation_angle=quarter, qubits=both))
    add(native_gate.stmts.CZ(controls=r0, targets=r1))
    return block, [q0, q1]


@pytest.mark.parametrize(
    "build", [_teleportation_block, _split_forcing_block], ids=["teleport", "split"]
)
def test_rule_preserves_the_unitary_up_to_the_residual(build):
    block, order = build()
    before = _unitary(block, order)

    rule, _ = _run(block)

    after = _unitary(block, order)
    residual = np.eye(2 ** len(order), dtype=complex)
    for value, angle in rule._frame.items():
        residual = _embed(_rz(angle), order.index(value), len(order)) @ residual

    assert _equal_up_to_phase(before, residual @ after)


def test_a_flipped_sign_would_be_caught():
    """Guard the guard: shifting the axis the wrong way must break the check."""
    block, order = _teleportation_block()
    before = _unitary(block, order)

    rule, _ = _run(block)
    after = _unitary(block, order)

    wrong = np.eye(4, dtype=complex)
    for value, angle in rule._frame.items():
        wrong = _embed(_rz(-angle), order.index(value), 2) @ wrong

    assert not _equal_up_to_phase(before, wrong @ after)
