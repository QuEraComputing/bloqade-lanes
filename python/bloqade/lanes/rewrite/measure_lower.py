"""measure_lower — validate + rewrite move.Measure to move.EndMeasure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from kirin import ir
from kirin.rewrite.abc import RewriteResult, RewriteRule

from bloqade.lanes.dialects import move
from bloqade.lanes.layout.encoding import ZoneAddress


class MeasureLowerError(RuntimeError):
    """Raised when the measure_lower invariants are violated."""


@dataclass
class MeasureLower(RewriteRule):
    """Lower move.Measure stmts to move.EndMeasure.

    Requires the zone set per Measure site (from AtomAnalysis) and the
    program-wide count of final measurements. Enforces:

    1. Each move.Measure covers exactly one zone.
    2. The program contains exactly one final measurement.
    """

    zone_sets: Mapping[move.Measure, frozenset[int]]
    final_measurement_count: int

    @classmethod
    def from_method(cls, method: ir.Method, arch_spec) -> "MeasureLower":
        """Build a MeasureLower by running AtomAnalysis on the given method.

        Populates ``zone_sets`` and ``final_measurement_count`` from
        ``AtomInterpreter.measure_sites`` / ``.final_measurement_count``.
        """
        from bloqade.lanes.analysis.atom import AtomInterpreter

        interp = AtomInterpreter(method.dialects, arch_spec=arch_spec)
        interp.run(method)
        # pyright flags move.Measure as unhashable when used as a dict-literal
        # key (ir.Statement.__hash__ = id(self) at runtime, so this is a false
        # positive). Build via assignment to sidestep the analyzer.
        zone_sets: dict[move.Measure, frozenset[int]] = {}
        for site in interp.measure_sites:
            stmt = site["stmt"]
            zone_ids = frozenset(z.zone_id for z in site["zones"])
            zone_sets[stmt] = zone_ids
        return cls(
            zone_sets=zone_sets,
            final_measurement_count=interp.final_measurement_count,
        )

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        if not isinstance(node, move.Measure):
            return RewriteResult()
        if self.final_measurement_count != 1:
            raise MeasureLowerError(
                f"expected exactly one final measurement, "
                f"found {self.final_measurement_count}"
            )
        zones = self.zone_sets.get(node)
        if zones is None:
            raise MeasureLowerError(f"no analysis result for {node}")
        if len(zones) != 1:
            raise MeasureLowerError(
                f"move.Measure spans {len(zones)} zones; expected exactly 1"
            )
        (zone_id,) = zones
        replacement = move.EndMeasure(
            current_state=node.current_state,
            zone_addresses=(ZoneAddress(zone_id),),
        )
        node.replace_by(replacement)
        return RewriteResult(has_done_something=True)
