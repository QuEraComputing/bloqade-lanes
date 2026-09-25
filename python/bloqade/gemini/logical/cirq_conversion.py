"""Cirq conversion for the supported Gemini logical circuit subset.

Imported lazily by ``bloqade.cirq_utils`` so importing Gemini logical kernels
does not require the optional Cirq dependency.
"""

from dataclasses import dataclass
from typing import Any, cast

import cirq
from bloqade.cirq_utils.emit.base import EmitCirq, EmitCirqFrame
from bloqade.cirq_utils.lowering import Squin
from kirin import interp, ir, lowering
from kirin.dialects import func, ilist, py

from bloqade import qubit
from bloqade.gemini.common.dialects.qubit import stmts as gemini_qubit
from bloqade.gemini.common.dialects.qubit._dialect import (
    dialect as gemini_qubit_dialect,
)
from bloqade.gemini.logical.dialects.operations import stmts as logical_ops
from bloqade.gemini.logical.dialects.operations._dialect import (
    dialect as logical_dialect,
)


@dataclass(frozen=True, eq=False, repr=False)
class GeminiLogicalQubit(cirq.Qid):
    """A logical Cirq qubit with an optional pinned Gemini address."""

    index: int
    pin: tuple[int, int, int] | None = None

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("Logical qubit indices must be nonnegative")
        if self.pin is not None and (
            len(self.pin) != 3 or any(not isinstance(value, int) for value in self.pin)
        ):
            raise ValueError("A pinned address must be three integer coordinates")

    @property
    def dimension(self) -> int:
        return 2

    def _comparison_key(self) -> tuple[int, tuple[int, int, int]]:
        return (self.index, self.pin or (-1, -1, -1))

    def __str__(self) -> str:
        if self.pin is None:
            return f"L{self.index}"
        zone, word, site = self.pin
        return f"L{self.index}@({zone},{word},{site})"

    def __repr__(self) -> str:
        return f"GeminiLogicalQubit(index={self.index!r}, pin={self.pin!r})"


class GeminiLogicalCirqLowerer(Squin):
    """Import terminal-measurement Cirq circuits as Gemini logical IR.

    Cirq's final measurement is a structural marker here: its ideal bit values
    are not the seven physical readouts of logical.terminal_measure.
    """

    def run(
        self,
        stmt: cirq.Circuit,
        *,
        register_as_argument: bool = False,
        **kwargs: Any,
    ) -> ir.Region:
        if register_as_argument:
            raise lowering.BuildError(
                "Gemini logical Cirq loading does not support register_as_argument"
            )
        return super().run(stmt, register_as_argument=register_as_argument, **kwargs)

    @staticmethod
    def _logical_index(qid: cirq.Qid) -> int:
        if isinstance(qid, GeminiLogicalQubit):
            return qid.index
        if isinstance(qid, cirq.LineQubit):
            return qid.x
        raise lowering.BuildError(f"Unsupported logical qubit identifier: {qid!r}")

    def __post_init__(self) -> None:
        qids = self.circuit.all_qubits()
        annotated = any(isinstance(qid, GeminiLogicalQubit) for qid in qids)
        if annotated:
            if not all(
                isinstance(qid, (GeminiLogicalQubit, cirq.LineQubit)) for qid in qids
            ):
                raise lowering.BuildError(
                    "GeminiLogicalQubit cannot be mixed with other Cirq Qid types"
                )
            ordered = sorted(qids, key=self._logical_index)
            indices = [self._logical_index(qid) for qid in ordered]
            if indices != list(range(len(qids))):
                raise lowering.BuildError(
                    "Logical qubit indices must be unique and contiguous from zero"
                )
        else:
            ordered = sorted(qids)
        self.qreg_index = {qid: index for index, qid in enumerate(ordered)}

    def allocate_register(self, state: lowering.State[cirq.Circuit]) -> ir.SSAValue:
        values: list[ir.SSAValue] = []
        for qid in self.qreg_index:
            if isinstance(qid, GeminiLogicalQubit) and qid.pin is not None:
                coords = [
                    state.current_frame.push(py.Constant(value)).result
                    for value in qid.pin
                ]
                stmt = gemini_qubit.NewAt(*coords)
            else:
                stmt = qubit.stmts.New()
            values.append(state.current_frame.push(stmt).results[0])
        return state.current_frame.push(ilist.New(values=values)).result

    def visit_Circuit(
        self,
        state: lowering.State[cirq.Circuit],
        node: cirq.Circuit | cirq.FrozenCircuit,
    ) -> lowering.Result:
        operations = list(node.all_operations())
        measurements = [
            operation
            for operation in operations
            if isinstance(operation.gate, cirq.MeasurementGate)
        ]
        if self.qreg_index:
            ordered_qids = tuple(self.qreg_index)
            if (
                len(measurements) != 1
                or measurements[0] is not operations[-1]
                or measurements[0].qubits != ordered_qids
            ):
                raise lowering.BuildError(
                    "Gemini logical Cirq circuits require one final measurement "
                    "of every qubit in allocation order"
                )
            gate = cast(cirq.MeasurementGate, measurements[0].gate)
            if any(gate.invert_mask) or gate.confusion_map:
                raise lowering.BuildError(
                    "Gemini logical terminal measurement does not support "
                    "Cirq invert masks or confusion maps"
                )
        elif measurements:
            raise lowering.BuildError("Cannot measure an empty logical register")
        return super().visit_Circuit(state, node)

    def visit_MeasurementGate(
        self, state: lowering.State[cirq.Circuit], node: cirq.GateOperation
    ) -> ir.Statement:
        return state.current_frame.push(
            logical_ops.TerminalLogicalMeasurement(qubits=self.qreg)
        )


@gemini_qubit_dialect.register(key="emit.cirq")
class _GeminiQubitCirqMethods(interp.MethodTable):
    @interp.impl(gemini_qubit.NewAt)
    def new_at(
        self, emit: EmitCirq, frame: EmitCirqFrame, stmt: gemini_qubit.NewAt
    ) -> tuple[GeminiLogicalQubit]:
        pin = (
            frame.get(stmt.zone_id),
            frame.get(stmt.word_id),
            frame.get(stmt.site_id),
        )
        if not all(isinstance(coord, int) for coord in pin):
            raise interp.exceptions.InterpreterError(
                "Pinned Gemini addresses must be compile-time integers"
            )
        qid = GeminiLogicalQubit(frame.qubit_index, pin=pin)
        if frame.qubits is not None:
            supplied = frame.qubits[frame.qubit_index]
            if supplied != qid:
                raise interp.exceptions.InterpreterError(
                    "A circuit_qubits entry for NewAt must be the matching "
                    "GeminiLogicalQubit with the same index and pinned address"
                )
        frame.qubit_index += 1
        return (qid,)


@logical_dialect.register(key="emit.cirq")
class _LogicalCirqMethods(interp.MethodTable):
    @interp.impl(logical_ops.TerminalLogicalMeasurement)
    def terminal_measure(
        self,
        emit: EmitCirq,
        frame: EmitCirqFrame,
        stmt: logical_ops.TerminalLogicalMeasurement,
    ) -> tuple[None]:
        # A direct return is discarded by emit_circuit(ignore_returns=True).
        # Other consumers (e.g. detector/observable post-processing) cannot be
        # represented by an ordinary Cirq measurement.
        if any(not isinstance(use.stmt, func.Return) for use in stmt.result.uses):
            raise interp.exceptions.InterpreterError(
                "Cirq emission cannot preserve logical measurement results or "
                "their detector/observable post-processing"
            )
        qids = frame.get(stmt.qubits)
        emit.circuit.append(cirq.measure(*qids), strategy=cirq.InsertStrategy.NEW)
        return (None,)

    @interp.impl(logical_ops.Initialize)
    @interp.impl(logical_ops.StarRz)
    def unsupported(
        self, emit: EmitCirq, frame: EmitCirqFrame, stmt: ir.Statement
    ) -> tuple[Any, ...]:
        raise interp.exceptions.InterpreterError(
            f"Cirq emission does not yet support {stmt.name}"
        )
