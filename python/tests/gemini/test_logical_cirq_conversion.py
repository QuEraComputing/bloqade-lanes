import cirq
import pytest
from bloqade.cirq_utils import emit_circuit, load_circuit
from kirin import interp, lowering
from kirin.dialects import ilist

from bloqade import qubit, squin
from bloqade.gemini import logical
from bloqade.gemini.common.dialects.qubit import new_at
from bloqade.gemini.common.dialects.qubit.stmts import NewAt
from bloqade.gemini.device import GeminiLogicalSimulator
from bloqade.gemini.logical.cirq_conversion import GeminiLogicalQubit
from bloqade.gemini.logical.dialects.operations.stmts import (
    TerminalLogicalMeasurement,
)


def test_load_plain_cirq_as_logical_kernel():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(q0), cirq.CX(q0, q1), cirq.measure(q0, q1))

    method = load_circuit(circuit, dialects=logical.kernel)

    assert method.dialects is logical.kernel
    assert (
        sum(
            isinstance(stmt, TerminalLogicalMeasurement)
            for stmt in method.callable_region.walk()
        )
        == 1
    )
    assert (
        sum(isinstance(stmt, qubit.stmts.New) for stmt in method.callable_region.walk())
        == 2
    )
    method.verify()
    method.verify_type()


def test_pinned_logical_qubit_round_trip():
    @logical.kernel
    def program():
        register = ilist.IList([new_at(0, 0, 0), qubit.new()])
        squin.h(register[0])
        squin.cx(register[0], register[1])
        logical.terminal_measure(register)

    circuit = emit_circuit(program)
    qids = circuit.all_qubits()
    assert GeminiLogicalQubit(0, (0, 0, 0)) in qids
    assert cirq.LineQubit(1) in qids

    reloaded = load_circuit(circuit, dialects=logical.kernel)
    pins = [stmt for stmt in reloaded.callable_region.walk() if isinstance(stmt, NewAt)]
    assert len(pins) == 1
    assert (
        sum(
            isinstance(stmt, TerminalLogicalMeasurement)
            for stmt in reloaded.callable_region.walk()
        )
        == 1
    )
    reloaded.verify()
    reloaded.verify_type()
    assert emit_circuit(reloaded) == circuit


def test_pinned_emission_rejects_mismatched_circuit_qubits():
    @logical.kernel
    def program():
        register = ilist.IList([new_at(0, 0, 0)])
        logical.terminal_measure(register)

    with pytest.raises(interp.exceptions.InterpreterError, match="matching"):
        emit_circuit(program, circuit_qubits=[cirq.LineQubit(0)])


def test_logical_loader_rejects_mid_circuit_measurement():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.measure(q0), cirq.H(q1), cirq.measure(q1))

    with pytest.raises(lowering.BuildError, match="one final measurement"):
        load_circuit(circuit, dialects=logical.kernel)


def test_logical_loader_rejects_inverted_measurement():
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.measure(q, invert_mask=(True,)))

    with pytest.raises(lowering.BuildError, match="invert masks"):
        load_circuit(circuit, dialects=logical.kernel)


def test_logical_loader_rejects_register_argument_mode():
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.measure(q))

    with pytest.raises(lowering.BuildError, match="register_as_argument"):
        load_circuit(circuit, dialects=logical.kernel, register_as_argument=True)


def test_emit_rejects_logical_measurement_postprocessing():
    @logical.kernel
    def program():
        register = qubit.qalloc(1)
        return logical.default_post_processing(register)

    with pytest.raises(
        interp.exceptions.InterpreterError, match="detector/observable post-processing"
    ):
        emit_circuit(program, ignore_returns=True)


def test_emit_bell_kernel_returning_terminal_measurement():
    @logical.kernel
    def logical_bell():
        register = squin.qalloc(2)
        squin.h(register[0])
        squin.cx(register[0], register[1])
        return logical.terminal_measure(register)

    circuit = emit_circuit(logical_bell, ignore_returns=True)
    q0, q1 = cirq.LineQubit.range(2)
    assert circuit == cirq.Circuit(cirq.H(q0), cirq.CX(q0, q1), cirq.measure(q0, q1))


def test_emit_compiled_bell_has_one_physical_measurement():
    @logical.kernel
    def logical_bell():
        register = squin.qalloc(2)
        squin.h(register[0])
        squin.cx(register[0], register[1])
        return logical.terminal_measure(register)

    physical = (
        GeminiLogicalSimulator().task(logical_bell).noiseless_physical_squin_kernel
    )
    physical_measurements = [
        stmt
        for stmt in physical.callable_region.walk()
        if isinstance(stmt, qubit.stmts.Measure)
    ]
    assert len(physical_measurements) == 1
    assert isinstance(physical_measurements[0].qubits.owner, ilist.New)
    assert len(physical_measurements[0].qubits.owner.values) == 14

    circuit = emit_circuit(physical, ignore_returns=True)
    cirq_measurements = [
        op
        for op in circuit.all_operations()
        if isinstance(op.gate, cirq.MeasurementGate)
    ]
    assert len(cirq_measurements) == 1
    assert len(cirq_measurements[0].qubits) == 14
