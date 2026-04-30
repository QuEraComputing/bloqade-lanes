"""Pre-compilation validation for physical circuits (post-unroll).

Checks that the unrolled squin kernel has exactly one qubit.stmts.Measure statement
and that it consumes every qubit allocated in the circuit.

Run this after SquinToNative + AggressiveUnroll so that qubit.stmts.New and
qubit.stmts.Measure are present as direct IR statements rather than Invokes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import bloqade.qubit as qubit_dialect
from bloqade.analysis import address
from bloqade.analysis.address.lattice import AddressQubit, AddressReg, PartialIList
from kirin import ir
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
    return set()


@dataclass
class PhysicalTerminalMeasurementValidation(ValidationPass):
    """Validate that a physical circuit has exactly one terminal measure
    covering all allocated qubits.

    Pre-condition: IR has been through SquinToNative + AggressiveUnroll so that
    qubit.stmts.New and qubit.stmts.Measure are direct IR statements.

    Checks:
    1. Exactly one qubit.stmts.Measure statement exists.
    2. Its qubits argument covers all qubits allocated in the circuit.
    """

    analysis_cache: dict = field(default_factory=dict)

    def name(self) -> str:
        return "Physical Terminal Measurement Validation"

    def get_required_analyses(self) -> list[type]:
        return []

    def set_analysis_cache(self, cache: dict[type, Any]) -> None:
        self.analysis_cache.update(cache)

    def run(self, method: ir.Method) -> tuple[Any, list[ir.ValidationError]]:
        measure_stmts = [
            s
            for s in method.callable_region.walk()
            if isinstance(s, qubit_dialect.stmts.Measure)
        ]

        addr_analysis = address.AddressAnalysis(dialects=method.dialects)
        frame, _ = addr_analysis.run(method)
        total_qubits = addr_analysis.qubit_count

        errors: list[ir.ValidationError] = []

        # Anchor method-level errors to the return statement.
        return_stmt = method.callable_region.blocks[0].last_stmt
        assert return_stmt is not None

        if len(measure_stmts) == 0:
            errors.append(
                ir.ValidationError(
                    return_stmt,
                    "Physical circuit must end with exactly one terminal measure "
                    "consuming all allocated qubits; none found.",
                )
            )
            return None, errors

        if len(measure_stmts) > 1:
            for extra in measure_stmts[1:]:
                errors.append(
                    ir.ValidationError(
                        extra,
                        "Multiple qubit.stmts.Measure statements found; only one "
                        "terminal measure is allowed in a physical circuit.",
                    )
                )

        measure = measure_stmts[0]
        measured_ids = _collect_qubit_ids(frame.get(measure.qubits))
        expected_ids = set(range(total_qubits))
        if measured_ids != expected_ids:
            errors.append(
                ir.ValidationError(
                    measure,
                    f"Terminal measure covers {len(measured_ids)} qubit(s) but "
                    f"{total_qubits} were allocated; all qubits must be measured.",
                )
            )

        return None, errors
