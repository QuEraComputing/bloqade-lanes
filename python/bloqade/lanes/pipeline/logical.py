from __future__ import annotations

from dataclasses import dataclass, field

from bloqade.analysis.validation.simple_nocloning import FlatKernelNoCloningValidation
from bloqade.rewrite.passes.callgraph import CallGraphPass
from bloqade.squin.rewrite.non_clifford_to_U3 import RewriteNonCliffordToU3
from kirin import passes, rewrite
from kirin.ir.method import Method
from kirin.rewrite.abc import RewriteRule
from kirin.validation import ValidationSuite

from bloqade.gemini.logical.rewrite.initialize import _RewriteU3ToInitialize
from bloqade.gemini.logical.rewrite.steane_transversal import (
    RewriteSteaneTransversalCliffordAdjoints,
)
from bloqade.gemini.logical.validation.clifford.analysis import GeminiLogicalValidation
from bloqade.gemini.logical.validation.measurement.analysis import (
    GeminiTerminalMeasurementValidation,
)
from bloqade.lanes.analysis import layout, placement
from bloqade.lanes.analysis.placement import PalindromePlacementStrategy
from bloqade.lanes.arch.gemini.logical import get_arch_spec as get_logical_arch_spec
from bloqade.lanes.arch.gemini.logical.upstream import steane7_transversal_map
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.heuristics.logical.layout import LogicalLayoutHeuristic
from bloqade.lanes.heuristics.logical.placement import LogicalPlacementStrategyNoHome
from bloqade.lanes.passes import SequentialPlacePass, TransversalRewritePass
from bloqade.lanes.rewrite import circuit2place

from .base import _NativeToPlaceBase, _PlaceToMove


def transversal_rewrites(mt: Method) -> Method:
    """Apply transversal rewrite rules to a squin method.

    Expands logical operations into their transversal (physical qubit) equivalents
    using the Steane [[7,1,3]] transversal map. The method is rewritten in place.

    Args:
        mt (Method): The squin method to rewrite.

    Returns:
        Method: The rewritten method (same object, mutated in place).

    """

    TransversalRewritePass(
        mt.dialects, transversal_location_map=steane7_transversal_map
    )(mt)

    return mt


@dataclass
class _LogicalNativeToPlace(_NativeToPlaceBase):
    transversal_rewrite: bool = False

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

        rules: list[RewriteRule] = []
        if self.transversal_rewrite:
            # For [[7,1,3]] Steane code, logical sqrt-X and sqrt-Z are implemented
            # as transversal sqrt-X-adj and sqrt-Z-adj, respectively.
            rules.append(rewrite.Walk(RewriteSteaneTransversalCliffordAdjoints()))
        rules += [
            rewrite.Walk(RewriteNonCliffordToU3()),
            rewrite.Walk(_RewriteU3ToInitialize()),
        ]
        CallGraphPass(mt.dialects, rewrite.Chain(*rules))(out)
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

    ``arch_spec`` is the single source of truth: when ``layout_heuristic`` or
    ``placement_strategy`` are ``None`` (the default), they are constructed in
    ``emit`` using ``self.arch_spec``, guaranteeing consistency.  Pass explicit
    instances only when you need a fully custom heuristic or strategy; in that
    case the caller is responsible for arch-spec consistency.
    """

    arch_spec: ArchSpec = field(default_factory=get_logical_arch_spec)
    layout_heuristic: layout.LayoutHeuristicABC | None = None
    placement_strategy: placement.PlacementStrategyABC | None = None
    place_opt_type: type[passes.Pass] = field(default=SequentialPlacePass)
    transversal_rewrite: bool = False

    def emit(self, mt: Method, no_raise: bool = True) -> Method:
        heuristic = (
            LogicalLayoutHeuristic(arch_spec=self.arch_spec)
            if self.layout_heuristic is None
            else self.layout_heuristic
        )
        strategy = (
            PalindromePlacementStrategy(
                inner=LogicalPlacementStrategyNoHome(arch_spec=self.arch_spec)
            )
            if self.placement_strategy is None
            else self.placement_strategy
        )

        out = _LogicalNativeToPlace(
            arch_spec=self.arch_spec, transversal_rewrite=self.transversal_rewrite
        ).emit(mt, no_raise=no_raise)
        self.place_opt_type(out.dialects, no_raise=no_raise)(out)

        out = _PlaceToMove(
            layout_heuristic=heuristic,
            placement_strategy=strategy,
            insert_initialize=True,
        ).emit(out, no_raise=no_raise)

        if self.transversal_rewrite:
            out = transversal_rewrites(out)

        return out
