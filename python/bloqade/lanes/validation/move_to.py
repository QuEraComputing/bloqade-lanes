"""Eager per-statement validation for ``arrange.MoveTo`` (failures 1-5 from spec §4).

Registered against the lanes validation interpreter key (``move.address.validation``).
The impl checks:

1. ``len(qubits) == len(locations)``
2. ``locations`` must be compile-time constants (const-foldable)
3. Every ``LocationAddress`` is in-range for the arch spec
4. No duplicate destination ``LocationAddress`` values within one call
5. No duplicate Qubit SSA values within one call
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kirin import ir
from kirin.validation import ValidationPass

from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.dialects import arch, arrange
from bloqade.lanes.validation._method_tables import require_method_tables
from bloqade.lanes.validation.address import _ValidationAnalysis


@dataclass
class MoveToValidation(ValidationPass):
    """Eager per-statement validation for ``arrange.MoveTo`` (failures 1-5)."""

    arch_spec: ArchSpec = field(kw_only=True)

    def name(self) -> str:
        return "lanes.move_to.validation"

    def run(self, method: ir.Method) -> tuple[Any, list[ir.ValidationError]]:
        require_method_tables(
            method,
            [
                (arch.dialect, "move.address.validation"),
                (arrange.dialect, "move.address.validation"),
            ],
        )

        analysis = _ValidationAnalysis(
            method.dialects,
            arch_spec=self.arch_spec,
        )
        frame, _ = analysis.run(method)
        return frame, analysis.get_validation_errors()
