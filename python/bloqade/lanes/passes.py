from abc import ABC, abstractmethod
from dataclasses import dataclass

from kirin import rewrite
from kirin.ir.method import Method

from bloqade.lanes.rewrite.circuit2place import (
    HoistNewQubitsUp,
    MergeStaticPlacement,
    gate_only_merge,
    sq_only_merge,
)
from bloqade.lanes.rewrite.fuse_gates import FuseAdjacentGates
from bloqade.lanes.rewrite.reorder_static_placement import (
    ReorderStaticPlacement,
    asap_reorder_policy,
)
from bloqade.lanes.rewrite.split_static_placement import (
    SplitStaticPlacement,
    cz_layer_split_policy,
)


@dataclass
class PlaceOptimizationPass(ABC):
    """Base class for place-dialect optimization passes.

    Called once from the compilation pipeline without a fixpoint wrapper.
    Any internal fixpoint logic must be self-contained in the implementation.
    """

    @abstractmethod
    def __call__(self, mt: Method) -> None: ...


@dataclass
class SequentialPlacePass(PlaceOptimizationPass):
    """Preserve gate order; merge only adjacent single-qubit-gate placements (R, Rz).

    CZ placements remain isolated, preserving the original program order.
    This is the default behavior.
    """

    def __call__(self, mt: Method) -> None:
        rewrite.Fixpoint(rewrite.Walk(MergeStaticPlacement(sq_only_merge))).rewrite(
            mt.code
        )
        rewrite.Walk(HoistNewQubitsUp()).rewrite(mt.code)
        rewrite.Fixpoint(rewrite.Walk(MergeStaticPlacement(sq_only_merge))).rewrite(
            mt.code
        )


@dataclass
class ASAPPlacePass(PlaceOptimizationPass):
    """ASAP scheduling optimization for the place dialect.

    Merges all pure-gate placements (R, Rz, CZ), reorders gates by ASAP
    dependency scheduling, fuses adjacent compatible gates, then re-splits
    on CZ-anchored boundaries.
    """

    def __call__(self, mt: Method) -> None:
        rewrite.Fixpoint(rewrite.Walk(MergeStaticPlacement(gate_only_merge))).rewrite(
            mt.code
        )
        rewrite.Walk(HoistNewQubitsUp()).rewrite(mt.code)
        rewrite.Fixpoint(rewrite.Walk(MergeStaticPlacement(gate_only_merge))).rewrite(
            mt.code
        )
        rewrite.Walk(ReorderStaticPlacement(asap_reorder_policy)).rewrite(mt.code)
        rewrite.Fixpoint(rewrite.Walk(FuseAdjacentGates())).rewrite(mt.code)
        rewrite.Walk(SplitStaticPlacement(cz_layer_split_policy)).rewrite(mt.code)
