from __future__ import annotations

from dataclasses import dataclass, field

from bloqade.analysis.validation.simple_nocloning import FlatKernelNoCloningValidation
from bloqade.rewrite.passes.callgraph import CallGraphPass
from bloqade.squin.rewrite.non_clifford_to_U3 import RewriteNonCliffordToU3
from kirin import passes, rewrite
from kirin.ir.method import Method
from kirin.validation import ValidationSuite

from bloqade.gemini.logical.rewrite.initialize import _RewriteU3ToInitialize
from bloqade.gemini.logical.validation.clifford.analysis import GeminiLogicalValidation
from bloqade.gemini.logical.validation.measurement.analysis import (
    GeminiTerminalMeasurementValidation,
)
from bloqade.lanes.analysis import layout, placement
from bloqade.lanes.arch.gemini.logical import get_arch_spec as get_logical_arch_spec
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.heuristics.logical.layout import LogicalLayoutHeuristic
from bloqade.lanes.heuristics.logical.placement import LogicalPlacementStrategyNoHome
from bloqade.lanes.passes import SequentialPlacePass
from bloqade.lanes.rewrite import circuit2place

from .base import _NativeToPlaceBase, _PlaceToMove


@dataclass
class _LogicalNativeToPlace(_NativeToPlaceBase):
    def _pre_native_rewrites(self, mt: Method, out: Method, no_raise: bool) -> Method:
        validator = ValidationSuite(
            [
                GeminiLogicalValidation,
                GeminiTerminalMeasurementValidation,
                FlatKernelNoCloningValidation,
            ]
        )
        result = validator.validate(mt)
        if not result.is_valid and not no_raise:
            result.raise_if_invalid()

        rule = rewrite.Chain(
            rewrite.Walk(RewriteNonCliffordToU3()),
            rewrite.Walk(_RewriteU3ToInitialize()),
        )
        CallGraphPass(mt.dialects, rule)(out)
        return out

    def _lower_qubits(self, out: Method) -> None:
        rewrite.Walk(circuit2place.RewriteInitializeToLogicalInitialize()).rewrite(
            out.code
        )
        rewrite.Walk(circuit2place.RewriteLogicalInitializeToNewLogical()).rewrite(
            out.code
        )
        rewrite.Walk(circuit2place.CleanUpLogicalInitialize()).rewrite(out.code)
        rewrite.Walk(circuit2place.InitializeNewQubits()).rewrite(out.code)


@dataclass
class LogicalPipeline:
    """Compile a logical squin kernel to the move dialect.

    Includes Clifford/measurement/no-cloning validation and the full logical
    initialization sequence (InsertInitialize for LogicalQubits).
    """

    layout_heuristic: layout.LayoutHeuristicABC = field(
        default_factory=LogicalLayoutHeuristic
    )
    placement_strategy: placement.PlacementStrategyABC = field(
        default_factory=LogicalPlacementStrategyNoHome
    )
    insert_return_moves: bool = True
    arch_spec: ArchSpec = field(default_factory=get_logical_arch_spec)
    place_opt_type: type[passes.Pass] = field(default=SequentialPlacePass)

    def emit(self, mt: Method, no_raise: bool = True) -> Method:
        out = _LogicalNativeToPlace(
            arch_spec=self.arch_spec,
            place_opt_type=self.place_opt_type,
        ).emit(mt, no_raise=no_raise)

        out = _PlaceToMove(
            layout_heuristic=self.layout_heuristic,
            placement_strategy=self.placement_strategy,
            insert_initialize=True,
            insert_return_moves=self.insert_return_moves,
        ).emit(out, no_raise=no_raise)

        return out
