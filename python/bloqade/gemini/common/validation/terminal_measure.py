"""Terminal-measurement validation for physical Gemini programs (post-unroll).

``PhysicalTerminalMeasurementValidation`` checks that the unrolled squin kernel
has exactly one ``qubit.stmts.Measure`` statement and that it consumes every
qubit allocated in the circuit.

Run this after ``SquinToNative`` + ``AggressiveUnroll`` so that
``qubit.stmts.New`` and ``qubit.stmts.Measure`` are present as direct IR
statements rather than Invokes.

The ``_PhysicalTerminalMeasurementAnalysis`` Forward pass accumulates a
``measure_count`` on the interpreter; the per-statement impl is registered in
``bloqade.gemini.common.impl.terminal_measure`` under the key
``"gemini.validate.physical.terminal_measure"``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bloqade.analysis import address
from bloqade.analysis.address.lattice import AddressQubit, AddressReg, PartialIList
from kirin import ir
from kirin.analysis import Forward, ForwardFrame
from kirin.lattice import EmptyLattice
from kirin.validation import ValidationPass


def _collect_qubit_ids(addr: address.Address) -> set[int]:
    """Recursively collect all qubit IDs from an Address lattice value."""
    if isinstance(addr, AddressQubit):
        return {addr.data}
    if isinstance(addr, AddressReg):
        return set(addr.data)
    if isinstance(addr, PartialIList):
        result: set[int] = set()
        for elem in addr.data:
            result |= _collect_qubit_ids(elem)
        return result
    raise ValueError(f"Unhandled Address type: {type(addr)}")


@dataclass
class _PhysicalTerminalMeasurementAnalysis(Forward[EmptyLattice]):
    """Forward dataflow pass that validates the physical terminal measurement.

    ``measure_count`` accumulates across the walk via the impl registered in
    ``bloqade.gemini.common.impl.terminal_measure``.
    """

    keys = ("gemini.validate.physical.terminal_measure",)
    lattice = EmptyLattice

    address_frame: ForwardFrame
    total_qubits: int
    measure_count: int = field(init=False, default=0)

    def eval_fallback(self, frame: ForwardFrame, node: ir.Statement):
        return tuple(self.lattice.bottom() for _ in node.results)

    def method_self(self, method: ir.Method) -> EmptyLattice:
        return self.lattice.bottom()


@dataclass
class PhysicalTerminalMeasurementValidation(ValidationPass):
    """Validate that a physical (post-unroll) circuit contains exactly one
    ``qubit.stmts.Measure`` consuming all allocated qubits.
    """

    def name(self) -> str:
        return "Physical Terminal Measurement Validation"

    def run(self, method: ir.Method) -> tuple[Any, list[ir.ValidationError]]:
        addr_analysis = address.AddressAnalysis(dialects=method.dialects)
        address_frame, _ = addr_analysis.run(method)
        total_qubits = addr_analysis.qubit_count

        analysis = _PhysicalTerminalMeasurementAnalysis(
            method.dialects,
            address_frame=address_frame,
            total_qubits=total_qubits,
        )
        frame, _ = analysis.run(method)

        errors = list(analysis.get_validation_errors())

        if analysis.measure_count == 0:
            return_stmt = method.callable_region.blocks[0].last_stmt
            if return_stmt is None:
                raise RuntimeError(
                    "PhysicalTerminalMeasurementValidation: method has no statements; "
                    "cannot attach missing-measure error."
                )
            errors.append(
                ir.ValidationError(
                    return_stmt,
                    "Physical circuit must end with exactly one terminal measure "
                    "consuming all allocated qubits; none found.",
                )
            )

        return frame, errors
