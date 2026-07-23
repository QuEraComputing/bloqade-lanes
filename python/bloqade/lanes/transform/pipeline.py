from __future__ import annotations

import warnings
from dataclasses import dataclass, field

from kirin import passes
from kirin.ir.method import Method

from bloqade.lanes.analysis import layout, placement
from bloqade.lanes.analysis.placement import PalindromePlacementStrategy
from bloqade.lanes.arch.gemini.logical import get_arch_spec as get_logical_arch_spec
from bloqade.lanes.arch.gemini.logical.upstream import steane7_transversal_map
from bloqade.lanes.arch.gemini.physical import get_arch_spec as get_physical_arch_spec
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.heuristics.logical.layout import LogicalLayoutHeuristic
from bloqade.lanes.heuristics.logical.placement import LogicalPlacementStrategyNoHome
from bloqade.lanes.heuristics.physical.layout import (
    PhysicalLayoutHeuristicGraphPartitionCenterOut,
)
from bloqade.lanes.heuristics.physical.movement import make_physical_placement_strategy
from bloqade.lanes.passes import SequentialPlacePass, TransversalRewritePass
from bloqade.lanes.transform.native_to_place import (
    LogicalNativeToPlace,
    PhysicalNativeToPlace,
)
from bloqade.lanes.transform.place_to_move import PlaceToMove


def transversal_rewrites(mt: Method, rewrite_logical_initialize: bool = True) -> Method:
    """Apply transversal rewrite rules to a squin method.

    Expands logical operations into their transversal (physical qubit) equivalents
    using the Steane [[7,1,3]] transversal map. The method is rewritten in place.

    Args:
        mt (Method): The squin method to rewrite.
        rewrite_logical_initialize (bool): Whether to rewrite the logical initialize statements.

    Returns:
        Method: The rewritten method (same object, mutated in place).

    """
    TransversalRewritePass(
        mt.dialects,
        transversal_location_map=steane7_transversal_map,
        rewrite_logical_initialize=rewrite_logical_initialize,
    )(mt)

    return mt


@dataclass
class PhysicalPipeline:
    """Compile a physical squin kernel to the move dialect.

    Qubits are lowered to place.NewPinnedQubit; no logical initialization
    sequence is inserted.  Validates that the kernel has exactly one terminal
    measure covering all allocated qubits (post-unroll).

    ``arch_spec`` is the single source of truth: when ``layout_heuristic`` or
    ``placement_strategy`` are ``None`` (the default), they are constructed in
    ``emit`` using ``self.arch_spec``, guaranteeing consistency.  Pass explicit
    instances only when you need a fully custom heuristic or strategy; in that
    case the caller is responsible for arch-spec consistency.
    """

    arch_spec: ArchSpec = field(default_factory=get_physical_arch_spec)
    layout_heuristic: layout.LayoutHeuristicABC | None = None
    placement_strategy: placement.PlacementStrategyABC | None = None
    place_opt_type: type[passes.Pass] = field(default=SequentialPlacePass)

    @property
    def resolved_layout_heuristic(self) -> layout.LayoutHeuristicABC:
        """Return the active layout heuristic, constructing it from ``arch_spec`` if unset.

        Emits a warning if an explicit heuristic is set whose ``arch_spec``
        differs from the pipeline's ``arch_spec``.
        """
        if self.layout_heuristic is None:
            return PhysicalLayoutHeuristicGraphPartitionCenterOut(
                arch_spec=self.arch_spec
            )
        if self.layout_heuristic.arch_spec != self.arch_spec:
            warnings.warn(
                "PhysicalPipeline.layout_heuristic was constructed with a different "
                "arch_spec than the pipeline. Initial qubit layout may not match the "
                "pipeline architecture. Leave layout_heuristic=None to have it built "
                "automatically from arch_spec.",
                stacklevel=2,
            )
        return self.layout_heuristic

    @property
    def resolved_placement_strategy(self) -> placement.PlacementStrategyABC:
        """Return the active placement strategy, constructing it from ``arch_spec`` if unset.

        Emits a warning if an explicit strategy is set whose ``arch_spec``
        differs from the pipeline's ``arch_spec``.
        """
        if self.placement_strategy is None:
            return make_physical_placement_strategy(arch_spec=self.arch_spec)
        if self.placement_strategy.arch_spec != self.arch_spec:
            warnings.warn(
                "PhysicalPipeline.placement_strategy was constructed with a different "
                "arch_spec than the pipeline. Compiled moves may not match the pipeline "
                "architecture. Leave placement_strategy=None to have it built automatically "
                "from arch_spec, or pass arch_spec= explicitly to "
                "make_physical_placement_strategy().",
                stacklevel=2,
            )
        return self.placement_strategy

    def emit(self, mt: Method, no_raise: bool = True) -> Method:
        out = PhysicalNativeToPlace(arch_spec=self.arch_spec).emit(
            mt, no_raise=no_raise
        )
        self.place_opt_type(out.dialects, no_raise=no_raise)(out)

        out = PlaceToMove(
            layout_heuristic=self.resolved_layout_heuristic,
            placement_strategy=self.resolved_placement_strategy,
            insert_initialize=False,
        ).emit(out, no_raise=no_raise)

        return out


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
    simulation: bool = True

    @property
    def resolved_layout_heuristic(self) -> layout.LayoutHeuristicABC:
        """Return the active layout heuristic, constructing it from ``arch_spec`` if unset.

        Emits a warning if an explicit heuristic is set whose ``arch_spec``
        differs from the pipeline's ``arch_spec``.
        """
        if self.layout_heuristic is None:
            return LogicalLayoutHeuristic(arch_spec=self.arch_spec)
        if self.layout_heuristic.arch_spec != self.arch_spec:
            warnings.warn(
                "LogicalPipeline.layout_heuristic was constructed with a different "
                "arch_spec than the pipeline. Initial qubit layout may not match the "
                "pipeline architecture. Leave layout_heuristic=None to have it built "
                "automatically from arch_spec.",
                stacklevel=2,
            )
        return self.layout_heuristic

    @property
    def resolved_placement_strategy(self) -> placement.PlacementStrategyABC:
        """Return the active placement strategy, constructing it from ``arch_spec`` if unset.

        Emits a warning if an explicit strategy is set whose ``arch_spec``
        differs from the pipeline's ``arch_spec``.
        """
        if self.placement_strategy is None:
            return PalindromePlacementStrategy(
                inner=LogicalPlacementStrategyNoHome(arch_spec=self.arch_spec)
            )
        if self.placement_strategy.arch_spec != self.arch_spec:
            warnings.warn(
                "LogicalPipeline.placement_strategy was constructed with a different "
                "arch_spec than the pipeline. Compiled moves may not match the pipeline "
                "architecture. Leave placement_strategy=None to have it built automatically "
                "from arch_spec, or construct the strategy with the same arch_spec instance.",
                stacklevel=2,
            )
        return self.placement_strategy

    def emit(self, mt: Method, no_raise: bool = True) -> Method:
        out = LogicalNativeToPlace(
            arch_spec=self.arch_spec, transversal_rewrite=self.transversal_rewrite
        ).emit(mt, no_raise=no_raise)
        self.place_opt_type(out.dialects, no_raise=no_raise)(out)

        out = PlaceToMove(
            layout_heuristic=self.resolved_layout_heuristic,
            placement_strategy=self.resolved_placement_strategy,
            insert_initialize=True,
        ).emit(out, no_raise=no_raise)

        if self.transversal_rewrite:
            # If running this compilation for simulation purposes we
            # need to rewrite the logical initialize statement
            transversal_rewrites(out, rewrite_logical_initialize=self.simulation)

        return out
