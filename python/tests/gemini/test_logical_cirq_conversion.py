import cirq
import pytest
from bloqade.cirq_utils import emit_circuit, load_circuit
from bloqade.cirq_utils.emit.base import EmitCirq, EmitCirqFrame
from kirin import interp, ir, lowering
from kirin.dialects import ilist
from kirin.ir.exception import ValidationErrorGroup

from bloqade import qubit, squin
from bloqade.gemini import logical
from bloqade.gemini.common.cirq_conversion import (
    GeminiCirqLowerer,
    GeminiQubit,
    _GeminiQubitCirqMethods,
)
from bloqade.gemini.common.dialects.qubit import new_at
from bloqade.gemini.common.dialects.qubit.stmts import NewAt
from bloqade.gemini.device import GeminiLogicalSimulator
from bloqade.gemini.logical.cirq_conversion import (
    GeminiLogicalQubit,
    _LogicalCirqMethods,
)
from bloqade.gemini.logical.dialects.operations import stmts as logical_ops
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


def test_logical_loader_accepts_unmeasured_subcircuit_with_final_measurement():
    q0, q1 = cirq.LineQubit.range(2)
    body = cirq.FrozenCircuit(cirq.H(q0), cirq.CX(q0, q1))
    circuit = cirq.Circuit(cirq.CircuitOperation(body), cirq.measure(q0, q1))

    method = load_circuit(circuit, dialects=logical.kernel)

    assert (
        sum(
            isinstance(stmt, TerminalLogicalMeasurement)
            for stmt in method.callable_region.walk()
        )
        == 1
    )


def test_logical_loader_preserves_subcircuit_qubit_mapping():
    q0, q1 = cirq.LineQubit.range(2)
    operation = cirq.CircuitOperation(cirq.FrozenCircuit(cirq.H(q0)))
    operation = operation.with_qubit_mapping({q0: q1})
    circuit = cirq.Circuit(cirq.X(q0), operation, cirq.measure(q0, q1))

    method = load_circuit(circuit, dialects=logical.kernel)
    emitted = emit_circuit(method)

    assert cirq.H(q1) in emitted.all_operations()
    assert cirq.H(q0) not in emitted.all_operations()


def test_logical_loader_preserves_terminal_measurement_order():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(q0), cirq.measure(q1, q0))

    method = load_circuit(circuit, dialects=logical.kernel)

    assert emit_circuit(method) == circuit


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

    with pytest.raises(ValidationErrorGroup, match="Multiple terminal measurements"):
        load_circuit(circuit, dialects=logical.kernel)


def test_logical_loader_rejects_partial_terminal_measurement():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(q0), cirq.H(q1), cirq.measure(q0))

    with pytest.raises(ValidationErrorGroup, match="only 1 were measured"):
        load_circuit(circuit, dialects=logical.kernel)


def test_logical_loader_rejects_missing_terminal_measurement():
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q0))

    with pytest.raises(ValidationErrorGroup, match="exactly one"):
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


def test_emit_rejects_logical_star_rz_extension():
    @logical.kernel
    def program():
        register = qubit.qalloc(1)
        logical.extensions.star_rz(0.1, register[0])
        logical.terminal_measure(register)

    with pytest.raises(interp.exceptions.InterpreterError, match="star_rz"):
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


def test_gemini_qid_rejects_negative_index_and_invalid_pin():
    with pytest.raises(ValueError, match="nonnegative"):
        GeminiQubit(-1)
    with pytest.raises(ValueError, match="three integer coordinates"):
        GeminiQubit(0, (0, 1))  # type: ignore[arg-type]


def test_gemini_qid_has_readable_pinned_and_unpinned_names():
    unpinned = GeminiQubit(0)
    pinned = GeminiQubit(1, (0, 2, 3))

    assert str(unpinned) == "Q0"
    assert str(pinned) == "Q1@(0,2,3)"
    assert repr(pinned) == "GeminiQubit(index=1, pin=(0, 2, 3))"
    assert GeminiLogicalQubit is GeminiQubit


def test_logical_loader_rejects_unsupported_qid_type():
    with pytest.raises(lowering.BuildError, match="Unsupported Gemini qubit"):
        GeminiCirqLowerer._qubit_index(cirq.GridQubit(0, 0))


def test_logical_loader_rejects_mixed_qid_types():
    pinned = GeminiLogicalQubit(0, (0, 0, 0))
    grid = cirq.GridQubit(0, 1)
    circuit = cirq.Circuit(cirq.measure(pinned, grid))

    with pytest.raises(lowering.BuildError, match="cannot be mixed"):
        load_circuit(circuit, dialects=logical.kernel)


def test_logical_loader_requires_contiguous_logical_indices():
    qid = GeminiLogicalQubit(1)
    circuit = cirq.Circuit(cirq.measure(qid))

    with pytest.raises(lowering.BuildError, match="contiguous from zero"):
        load_circuit(circuit, dialects=logical.kernel)


def test_pinned_emission_rejects_noninteger_address():
    zone, word, site = (ir.TestValue() for _ in range(3))
    stmt = NewAt(zone, word, site)
    frame = EmitCirqFrame(stmt, entries={zone: 0, word: 1.5, site: 0}, qubit_index=0)

    with pytest.raises(
        interp.exceptions.InterpreterError, match="compile-time integers"
    ):
        _GeminiQubitCirqMethods().new_at(EmitCirq(), frame, stmt)


def test_emit_rejects_logical_initialize_statement():
    stmt = logical_ops.Initialize(*(ir.TestValue() for _ in range(4)))

    with pytest.raises(interp.exceptions.InterpreterError, match="initialize"):
        _LogicalCirqMethods().unsupported(EmitCirq(), EmitCirqFrame(stmt), stmt)
