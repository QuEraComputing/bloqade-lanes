from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from kirin import ir, rewrite
from kirin.ir.exception import ValidationErrorGroup
from kirin.ir.method import Method

from bloqade.gemini.common.validation.terminal_measure import (
    PhysicalTerminalMeasurementValidation,
)
from bloqade.lanes.analysis import layout, placement
from bloqade.lanes.arch.gemini.physical import get_arch_spec as get_physical_arch_spec
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.heuristics.physical.layout import (
    PhysicalLayoutHeuristicGraphPartitionCenterOut,
)
from bloqade.lanes.heuristics.physical.placement import PhysicalPlacementStrategy
from bloqade.lanes.rewrite import circuit2place

from .base import _default_merge_heuristic, _NativeToPlaceBase, _PlaceToMove


@dataclass
class _PhysicalNativeToPlace(_NativeToPlaceBase):
    def _post_unroll_validation(self, out: Method) -> None:
        _, errors = PhysicalTerminalMeasurementValidation().run(out)
        if errors:
            raise ValidationErrorGroup(
                f"Physical circuit validation failed with {len(errors)} error(s)",
                errors=errors,
            )

    def _lower_qubits(self, out: Method) -> None:
        rewrite.Walk(circuit2place.RewriteQubitsToPinnedQubits()).rewrite(out.code)
        rewrite.Walk(circuit2place.RewritePhysicalMeasure()).rewrite(out.code)


@dataclass
class PhysicalPipeline:
    """Compile a physical squin kernel to the move dialect.

    Qubits are lowered to place.NewPinnedQubit; no logical initialization
    sequence is inserted.  Validates that the kernel has exactly one terminal
    measure covering all allocated qubits (post-unroll).
    """

    arch_spec: ArchSpec = field(default_factory=get_physical_arch_spec)
    layout_heuristic: layout.LayoutHeuristicABC | None = None
    placement_strategy: placement.PlacementStrategyABC | None = None
    insert_return_moves: bool = True
    merge_heuristic: Callable[[ir.Region, ir.Region], bool] = field(
        default=_default_merge_heuristic
    )

    def emit(self, mt: Method, no_raise: bool = True) -> Method:
        heuristic = (
            self.layout_heuristic
            or PhysicalLayoutHeuristicGraphPartitionCenterOut(arch_spec=self.arch_spec)
        )
        strategy = self.placement_strategy or PhysicalPlacementStrategy(
            arch_spec=self.arch_spec
        )

        out = _PhysicalNativeToPlace(
            merge_heuristic=self.merge_heuristic,
            arch_spec=self.arch_spec,
        ).emit(mt, no_raise=no_raise)

        out = _PlaceToMove(
            layout_heuristic=heuristic,
            placement_strategy=strategy,
            insert_initialize=False,
            insert_return_moves=self.insert_return_moves,
        ).emit(out, no_raise=no_raise)

        return out
