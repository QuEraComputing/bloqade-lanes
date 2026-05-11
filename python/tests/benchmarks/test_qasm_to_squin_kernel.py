from __future__ import annotations

import cirq
import numpy as np
from benchmarks.utils.qasm_to_squin_kernel import (
    _load_qasm_circuit,
    _qubit_sort_key,
    _statevectors_close,
    _strip_qasm_barriers,
    circuit_to_squin_decorator_source,
)


def test_strip_qasm_barriers_handles_global_and_explicit_statements():
    qasm = "\n".join(
        [
            "OPENQASM 2.0;",
            'include "qelib1.inc";',
            "qreg q[2];",
            "barrier;",
            "barrier q[0], q[1];",
            "x q[0];",
        ]
    )

    stripped, dropped = _strip_qasm_barriers(qasm)

    assert dropped == 2
    assert "barrier;" not in stripped
    assert "barrier q[0], q[1];" not in stripped
    assert "x q[0];" in stripped


def test_load_qasm_circuit_accepts_barrier_forms():
    qasm = "\n".join(
        [
            "OPENQASM 2.0;",
            'include "qelib1.inc";',
            "qreg q[2];",
            "barrier;",
            "x q[0];",
            "barrier q[0], q[1];",
            "cz q[0], q[1];",
        ]
    )

    circuit, dropped = _load_qasm_circuit(qasm)

    assert dropped >= 0
    assert len(list(circuit.all_operations())) == 2


def test_qubit_sort_key_orders_line_qubits_numerically():
    qubits = [cirq.LineQubit(10), cirq.LineQubit(2)]
    ordered = sorted(qubits, key=_qubit_sort_key)
    assert ordered == [cirq.LineQubit(2), cirq.LineQubit(10)]


def test_circuit_to_squin_decorator_source_is_deterministic_within_moment():
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit1 = cirq.Circuit(cirq.Moment([cirq.Z(q2), cirq.X(q1), cirq.Y(q0)]))
    circuit2 = cirq.Circuit(cirq.Moment([cirq.Y(q0), cirq.Z(q2), cirq.X(q1)]))

    source1 = circuit_to_squin_decorator_source(circuit1, "kernel")
    source2 = circuit_to_squin_decorator_source(circuit2, "kernel")

    assert source1 == source2


def test_statevectors_close_skips_phase_alignment_for_tiny_overlap():
    lhs = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    orthogonal = np.array([1.0, -1.0], dtype=np.complex128) / np.sqrt(2.0)
    epsilon = 1e-13
    phase = np.exp(1j * 1.1)
    rhs = orthogonal + epsilon * phase * lhs
    rhs /= np.linalg.norm(rhs)

    ok, max_abs_diff = _statevectors_close(lhs, rhs, atol=1e-10, rtol=1e-10)
    expected_diff = float(np.max(np.abs(lhs - rhs)))

    assert not ok
    assert np.isclose(max_abs_diff, expected_diff)
