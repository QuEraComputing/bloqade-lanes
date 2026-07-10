from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar

from kirin import ir, passes, rewrite
from kirin.rewrite.abc import RewriteResult

from bloqade.lanes.rewrite import circuit2place
from bloqade.lanes.rewrite.circuit2place import (
    HoistNewQubitsUp,
    MergeStaticPlacement,
    always_merge,
)
from bloqade.lanes.rewrite.fuse_gates import FuseAdjacentGates
from bloqade.lanes.rewrite.remove_debug import RemoveDebugStatements
from bloqade.lanes.rewrite.reorder_static_placement import (
    ReorderStaticPlacement,
    alap_reorder_policy,
    asap_reorder_policy,
)
from bloqade.lanes.rewrite.transversal import (
    RewriteGetItem,
    RewriteLocations,
    RewriteLogicalInitialize,
    RewriteLogicalToPhysicalConversion,
    RewriteMoves,
    RewriteStarRz,
)


@dataclass
class SequentialPlacePass(passes.Pass):
    """Merge all adjacent StaticPlacement blocks, preserving original gate order.

    Merging CZ placements into a single region (rather than leaving each CZ
    isolated) is required for correctness: the placement analysis threads the
    atom layout through the merged region, so each CZ layer is placed from the
    *current* (possibly displaced) layout. Isolated CZ blocks were each placed
    from the initial home layout, which is only valid when return moves restore
    home between layers. This is the default place-dialect optimization pass.
    """

    name: ClassVar[str] = "sequential_place"

    def unsafe_run(self, mt: ir.Method) -> RewriteResult:
        result = RewriteResult()
        result = result.join(
            rewrite.Walk(circuit2place.HoistConstants()).rewrite(mt.code)
        )
        result = result.join(
            rewrite.Fixpoint(rewrite.Walk(MergeStaticPlacement(always_merge))).rewrite(
                mt.code
            )
        )
        result = result.join(rewrite.Walk(HoistNewQubitsUp()).rewrite(mt.code))
        result = result.join(
            rewrite.Fixpoint(rewrite.Walk(MergeStaticPlacement(always_merge))).rewrite(
                mt.code
            )
        )
        return result


@dataclass
class ASAPPlacePass(passes.Pass):
    """ASAP scheduling optimization for the place dialect.

    Merges all pure-gate placements (R, Rz, CZ), reorders gates by ASAP
    dependency scheduling, then fuses adjacent compatible gates.

    ``debug.Info`` statements are stripped at the start of this pass.
    ASAP scheduling cannot reorder across opaque debug nodes, so they must
    be removed before the merge/reorder/fuse sequence.  Use
    ``SequentialPlacePass`` if debug statements must be preserved.
    """

    name: ClassVar[str] = "asap_place"

    def unsafe_run(self, mt: ir.Method) -> RewriteResult:
        result = RewriteResult()
        result = result.join(rewrite.Walk(RemoveDebugStatements()).rewrite(mt.code))
        result = result.join(
            rewrite.Walk(circuit2place.HoistConstants()).rewrite(mt.code)
        )
        result = result.join(
            rewrite.Fixpoint(rewrite.Walk(MergeStaticPlacement(always_merge))).rewrite(
                mt.code
            )
        )
        result = result.join(rewrite.Walk(HoistNewQubitsUp()).rewrite(mt.code))
        result = result.join(
            rewrite.Fixpoint(rewrite.Walk(MergeStaticPlacement(always_merge))).rewrite(
                mt.code
            )
        )
        result = result.join(
            rewrite.Walk(ReorderStaticPlacement(asap_reorder_policy)).rewrite(mt.code)
        )
        result = result.join(
            rewrite.Fixpoint(rewrite.Walk(FuseAdjacentGates())).rewrite(mt.code)
        )
        return result


@dataclass
class ALAPPlacePass(passes.Pass):
    """ALAP scheduling optimization for the place dialect.

    Same pipeline as ``ASAPPlacePass`` but uses ALAP (As-Late-As-Possible)
    dependency scheduling.  Deferring single-qubit gates to their latest
    valid layer reduces the qubit footprint of early CZ-anchored
    StaticPlacement regions, lowering atom-move overhead compared to ASAP.

    ``debug.Info`` statements are stripped at the start of this pass (same
    reason as ``ASAPPlacePass``).
    """

    name: ClassVar[str] = "alap_place"

    def unsafe_run(self, mt: ir.Method) -> RewriteResult:
        result = RewriteResult()
        result = result.join(rewrite.Walk(RemoveDebugStatements()).rewrite(mt.code))
        result = result.join(
            rewrite.Walk(circuit2place.HoistConstants()).rewrite(mt.code)
        )
        result = result.join(
            rewrite.Fixpoint(rewrite.Walk(MergeStaticPlacement(always_merge))).rewrite(
                mt.code
            )
        )
        result = result.join(rewrite.Walk(HoistNewQubitsUp()).rewrite(mt.code))
        result = result.join(
            rewrite.Fixpoint(rewrite.Walk(MergeStaticPlacement(always_merge))).rewrite(
                mt.code
            )
        )
        result = result.join(
            rewrite.Walk(ReorderStaticPlacement(alap_reorder_policy)).rewrite(mt.code)
        )
        result = result.join(
            rewrite.Fixpoint(rewrite.Walk(FuseAdjacentGates())).rewrite(mt.code)
        )
        return result


@dataclass
class TransversalRewritePass(passes.Pass):
    """Rewrite from logical to physical addresses."""

    name: ClassVar[str] = "transversal"
    transversal_location_map: Callable[..., Any] = field(kw_only=True)
    rewrite_logical_initialize: bool = field(kw_only=True, default=True)

    def unsafe_run(self, mt: ir.Method) -> RewriteResult:
        rules = []

        if self.rewrite_logical_initialize:
            rules.append(RewriteLogicalInitialize(self.transversal_location_map))

        # handles rewriting logical location addresses to groups of physical addresses
        rules += [
            RewriteLocations(self.transversal_location_map),
            RewriteMoves(self.transversal_location_map),
            RewriteStarRz(self.transversal_location_map),
            # handles the rewrite of physical to logical measurement results
            RewriteGetItem(self.transversal_location_map),
            RewriteLogicalToPhysicalConversion(),
        ]

        return rewrite.Walk(rewrite.Chain(*rules)).rewrite(mt.code)
