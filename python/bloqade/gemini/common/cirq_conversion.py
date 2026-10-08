"""Shared Cirq support for Gemini qubits with pinned architecture addresses.

Import this module lazily so Cirq remains an optional dependency of Gemini.
"""

from dataclasses import dataclass
from typing import Any

import cirq
from bloqade.cirq_utils.emit.base import EmitCirq, EmitCirqFrame
from bloqade.cirq_utils.lowering import Squin
from kirin import interp, ir, lowering
from kirin.dialects import ilist, py

from bloqade import qubit
from bloqade.gemini.common.dialects.qubit import stmts as gemini_qubit
from bloqade.gemini.common.dialects.qubit._dialect import (
    dialect as gemini_qubit_dialect,
)


@dataclass(frozen=True, eq=False, repr=False)
class GeminiQubit(cirq.Qid):
    """A Cirq qubit with an optional pinned Gemini architecture address."""

    index: int
    pin: tuple[int, int, int] | None = None

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("Gemini qubit indices must be nonnegative")
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
            return f"Q{self.index}"
        zone, word, site = self.pin
        return f"Q{self.index}@({zone},{word},{site})"

    def __repr__(self) -> str:
        return f"GeminiQubit(index={self.index!r}, pin={self.pin!r})"


class GeminiCirqLowerer(Squin):
    """Load pinned Qids as ``new_at`` and other Qids as ordinary qubits."""

    def visit_CircuitOperation(
        self, state: lowering.State[cirq.Circuit], node: cirq.CircuitOperation
    ) -> lowering.Result:
        if node.qubit_map:
            if node.repetitions != 1:
                raise lowering.BuildError(
                    "Mapped CircuitOperations with repetitions are not supported"
                )
            return self.visit(state, node.mapped_circuit())
        return super().visit_CircuitOperation(state, node)

    def run(
        self,
        stmt: cirq.Circuit,
        *,
        register_as_argument: bool = False,
        **kwargs: Any,
    ) -> ir.Region:
        if register_as_argument and any(
            isinstance(qid, GeminiQubit) and qid.pin is not None
            for qid in self.circuit.all_qubits()
        ):
            raise lowering.BuildError(
                "A pinned Gemini qubit cannot be loaded as a register argument"
            )
        return super().run(stmt, register_as_argument=register_as_argument, **kwargs)

    @staticmethod
    def _qubit_index(qid: cirq.Qid) -> int:
        if isinstance(qid, GeminiQubit):
            return qid.index
        if isinstance(qid, cirq.LineQubit):
            return qid.x
        raise lowering.BuildError(f"Unsupported Gemini qubit identifier: {qid!r}")

    def __post_init__(self) -> None:
        qids = self.circuit.all_qubits()
        annotated = any(isinstance(qid, GeminiQubit) for qid in qids)
        if annotated:
            if not all(isinstance(qid, (GeminiQubit, cirq.LineQubit)) for qid in qids):
                raise lowering.BuildError(
                    "GeminiQubit cannot be mixed with other Cirq Qid types"
                )
            ordered = sorted(qids, key=self._qubit_index)
            indices = [self._qubit_index(qid) for qid in ordered]
            if indices != list(range(len(qids))):
                raise lowering.BuildError(
                    "Gemini qubit indices must be unique and contiguous from zero"
                )
        else:
            ordered = sorted(qids)
        self.qreg_index = {qid: index for index, qid in enumerate(ordered)}

    def allocate_register(self, state: lowering.State[cirq.Circuit]) -> ir.SSAValue:
        values: list[ir.SSAValue] = []
        for qid in self.qreg_index:
            if isinstance(qid, GeminiQubit) and qid.pin is not None:
                coords = [
                    state.current_frame.push(py.Constant(value)).result
                    for value in qid.pin
                ]
                stmt = gemini_qubit.NewAt(*coords)
            else:
                stmt = qubit.stmts.New()
            values.append(state.current_frame.push(stmt).results[0])
        return state.current_frame.push(ilist.New(values=values)).result


@gemini_qubit_dialect.register(key="emit.cirq")
class _GeminiQubitCirqMethods(interp.MethodTable):
    @interp.impl(gemini_qubit.NewAt)
    def new_at(
        self, emit: EmitCirq, frame: EmitCirqFrame, stmt: gemini_qubit.NewAt
    ) -> tuple[GeminiQubit]:
        pin = (
            frame.get(stmt.zone_id),
            frame.get(stmt.word_id),
            frame.get(stmt.site_id),
        )
        if not all(isinstance(coord, int) for coord in pin):
            raise interp.exceptions.InterpreterError(
                "Pinned Gemini addresses must be compile-time integers"
            )
        qid = GeminiQubit(frame.qubit_index, pin=pin)
        if frame.qubits is not None:
            supplied = frame.qubits[frame.qubit_index]
            if supplied != qid:
                raise interp.exceptions.InterpreterError(
                    "A circuit_qubits entry for NewAt must be the matching "
                    "GeminiQubit with the same index and pinned address"
                )
        frame.qubit_index += 1
        return (qid,)
