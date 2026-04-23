"""measure_lower — validate + rewrite move.Measure to move.EndMeasure."""

from __future__ import annotations

from dataclasses import dataclass

from kirin import ir
from kirin.rewrite.abc import RewriteResult, RewriteRule

from bloqade.lanes.dialects import move


class MeasureLowerError(RuntimeError):
    """Raised when the measure_lower invariants are violated."""


@dataclass
class MeasureLower(RewriteRule):
    """Lower move.Measure stmts to move.EndMeasure.

    Requires the program-wide count of final measurements (from
    AtomAnalysis). Enforces:

    1. Each move.Measure covers exactly one zone (read directly from
       the ``zone_addresses`` attribute on the node).
    2. The program contains exactly one final measurement.
    """

    final_measurement_count: int

    @classmethod
    def from_method(cls, method: ir.Method, arch_spec) -> "MeasureLower":
        """Build a MeasureLower by running AtomAnalysis on the given method.

        Populates ``final_measurement_count`` from
        ``AtomInterpreter.final_measurement_count``.
        """
        from bloqade.lanes.analysis.atom import AtomInterpreter

        interp = AtomInterpreter(method.dialects, arch_spec=arch_spec)
        interp.run(method)
        return cls(final_measurement_count=interp.final_measurement_count)

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        if not isinstance(node, move.Measure):
            return RewriteResult()
        if self.final_measurement_count != 1:
            raise MeasureLowerError(
                f"expected exactly one final measurement, "
                f"found {self.final_measurement_count}"
            )
        if len(node.zone_addresses) != 1:
            raise MeasureLowerError(
                f"move.Measure spans {len(node.zone_addresses)} zones; "
                f"expected exactly 1"
            )
        (zone_addr,) = node.zone_addresses
        replacement = move.EndMeasure(
            current_state=node.current_state,
            zone_addresses=(zone_addr,),
        )
        node.replace_by(replacement)
        return RewriteResult(has_done_something=True)
