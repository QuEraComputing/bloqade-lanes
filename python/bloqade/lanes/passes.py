from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from kirin import ir, passes, rewrite
from kirin.rewrite.abc import RewriteResult

from bloqade.lanes.rewrite import circuit2place
from bloqade.lanes.rewrite.circuit2place import (
    HoistNewQubitsUp,
    MergeStaticPlacement,
    gate_only_merge,
    sq_only_merge,
)
from bloqade.lanes.rewrite.fuse_gates import FuseAdjacentGates
from bloqade.lanes.rewrite.remove_debug import RemoveDebugStatements
from bloqade.lanes.rewrite.reorder_static_placement import (
    ReorderStaticPlacement,
    alap_reorder_policy,
    asap_reorder_policy,
)
from bloqade.lanes.rewrite.split_static_placement import (
    SplitStaticPlacement,
    cz_layer_split_policy,
)


@dataclass
class SequentialPlacePass(passes.Pass):
    """Preserve gate order; merge only adjacent single-qubit-gate placements (R, Rz).

    CZ placements remain isolated, preserving the original program order.
    This is the default behavior.
    """

    name: ClassVar[str] = "sequential_place"

    def unsafe_run(self, mt: ir.Method) -> RewriteResult:
        result = RewriteResult()
        result = result.join(
            rewrite.Walk(circuit2place.HoistConstants()).rewrite(mt.code)
        )
        result = result.join(
            rewrite.Fixpoint(rewrite.Walk(MergeStaticPlacement(sq_only_merge))).rewrite(
                mt.code
            )
        )
        result = result.join(rewrite.Walk(HoistNewQubitsUp()).rewrite(mt.code))
        result = result.join(
            rewrite.Fixpoint(rewrite.Walk(MergeStaticPlacement(sq_only_merge))).rewrite(
                mt.code
            )
        )
        return result


@dataclass
class ASAPPlacePass(passes.Pass):
    """ASAP scheduling optimization for the place dialect.

    Merges all pure-gate placements (R, Rz, CZ), reorders gates by ASAP
    dependency scheduling, fuses adjacent compatible gates, then re-splits
    on CZ-anchored boundaries.

    ``debug.Info`` statements are stripped at the start of this pass.
    ASAP scheduling cannot reorder across opaque debug nodes, so they must
    be removed before the merge/reorder/fuse/split sequence.  Use
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
            rewrite.Fixpoint(
                rewrite.Walk(MergeStaticPlacement(gate_only_merge))
            ).rewrite(mt.code)
        )
        result = result.join(rewrite.Walk(HoistNewQubitsUp()).rewrite(mt.code))
        result = result.join(
            rewrite.Fixpoint(
                rewrite.Walk(MergeStaticPlacement(gate_only_merge))
            ).rewrite(mt.code)
        )
        result = result.join(
            rewrite.Walk(ReorderStaticPlacement(asap_reorder_policy)).rewrite(mt.code)
        )
        result = result.join(
            rewrite.Fixpoint(rewrite.Walk(FuseAdjacentGates())).rewrite(mt.code)
        )
        result = result.join(
            rewrite.Walk(SplitStaticPlacement(cz_layer_split_policy)).rewrite(mt.code)
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
            rewrite.Fixpoint(
                rewrite.Walk(MergeStaticPlacement(gate_only_merge))
            ).rewrite(mt.code)
        )
        result = result.join(rewrite.Walk(HoistNewQubitsUp()).rewrite(mt.code))
        result = result.join(
            rewrite.Fixpoint(
                rewrite.Walk(MergeStaticPlacement(gate_only_merge))
            ).rewrite(mt.code)
        )
        result = result.join(
            rewrite.Walk(ReorderStaticPlacement(alap_reorder_policy)).rewrite(mt.code)
        )
        result = result.join(
            rewrite.Fixpoint(rewrite.Walk(FuseAdjacentGates())).rewrite(mt.code)
        )
        result = result.join(
            rewrite.Walk(SplitStaticPlacement(cz_layer_split_policy)).rewrite(mt.code)
        )
        return result
