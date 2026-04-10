from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from bloqade.lanes import layout
from bloqade.lanes.analysis.placement import (
    AtomState,
    ConcreteState,
    ExecuteCZ,
    ExecuteMeasure,
    PlacementStrategyABC,
)
from bloqade.lanes.arch.gemini.physical import get_arch_spec as get_physical_arch_spec
from bloqade.lanes.search import (
    BFSTraversal,
    CandidateScorer,
    ConfigurationTree,
    EntropyGuidedTraversal,
    ExhaustiveMoveGenerator,
    GreedyBestFirstTraversal,
    HeuristicMoveGenerator,
    SearchParams,
    placement_goal,
)

if TYPE_CHECKING:
    from bloqade.lanes.search.configuration import ConfigurationNode
    from bloqade.lanes.search.traversal.goal import SearchResult
    from bloqade.lanes.search.traversal.step_info import StepInfo


OnSearchStep = Callable[[str, "ConfigurationNode", "StepInfo"], None]


class PlacementTraversalABC(abc.ABC):
    """Placement-facing traversal API for target-configuration search."""

    @abc.abstractmethod
    def path_to_target_config(
        self,
        *,
        tree: ConfigurationTree,
        target: dict[int, layout.LocationAddress],
    ) -> SearchResult:
        """Run search and return one or more goal nodes."""
        ...


def _mismatch_heuristic(
    target: dict[int, layout.LocationAddress],
) -> Callable[[ConfigurationNode], float]:
    def h(node: ConfigurationNode) -> float:
        return float(
            sum(
                1
                for qid, desired_loc in target.items()
                if node.configuration.get(qid) != desired_loc
            )
        )

    return h


@dataclass(frozen=True)
class EntropyPlacementTraversal(PlacementTraversalABC):
    """Placement traversal adapter backed by entropy-guided search."""

    search_params: SearchParams = field(default_factory=SearchParams)
    max_depth: int | None = None
    max_expansions: int | None = 2
    on_search_step: OnSearchStep | None = None

    def path_to_target_config(
        self,
        *,
        tree: ConfigurationTree,
        target: dict[int, layout.LocationAddress],
    ) -> SearchResult:
        params = self.search_params
        scorer = CandidateScorer(params=params, target=target)
        generator = HeuristicMoveGenerator(
            scorer=scorer,
            params=params,
            search_nodes={},
        )
        traversal = EntropyGuidedTraversal(
            target=target,
            params=params,
            on_step=self.on_search_step,
        )
        return traversal.search(
            tree=tree,
            generator=generator,
            goal=placement_goal(target),
            max_depth=self.max_depth,
            max_expansions=self.max_expansions,
        )


@dataclass(frozen=True)
class GreedyPlacementTraversal(PlacementTraversalABC):
    """Placement traversal adapter backed by greedy best-first search."""

    max_expansions: int | None = 300

    def path_to_target_config(
        self,
        *,
        tree: ConfigurationTree,
        target: dict[int, layout.LocationAddress],
    ) -> SearchResult:
        return GreedyBestFirstTraversal(heuristic=_mismatch_heuristic(target)).search(
            tree=tree,
            generator=ExhaustiveMoveGenerator(),
            goal=placement_goal(target),
            max_expansions=self.max_expansions,
        )


@dataclass(frozen=True)
class BFSPlacementTraversal(PlacementTraversalABC):
    """Placement traversal adapter backed by breadth-first search."""

    max_depth: int | None = None
    max_expansions: int | None = 300

    def path_to_target_config(
        self,
        *,
        tree: ConfigurationTree,
        target: dict[int, layout.LocationAddress],
    ) -> SearchResult:
        return BFSTraversal().search(
            tree=tree,
            generator=ExhaustiveMoveGenerator(),
            goal=placement_goal(target),
            max_depth=self.max_depth,
            max_expansions=self.max_expansions,
        )


@dataclass
class PhysicalPlacementStrategy(PlacementStrategyABC):
    """Physical placement strategy backed by configuration-tree search."""

    arch_spec: layout.ArchSpec = field(default_factory=get_physical_arch_spec)
    traversal: PlacementTraversalABC = field(default_factory=EntropyPlacementTraversal)

    _cz_counter: int = field(default=0, init=False, repr=False)
    _trace_cz_index: int | None = field(default=None, init=False, repr=False)
    _traced_tree: ConfigurationTree | None = field(default=None, init=False, repr=False)
    _traced_target: dict[int, layout.LocationAddress] = field(
        default_factory=dict, init=False, repr=False
    )

    @property
    def traced_tree(self) -> ConfigurationTree | None:
        return self._traced_tree

    @property
    def traced_target(self) -> dict[int, layout.LocationAddress]:
        return dict(self._traced_target)

    @property
    def trace_cz_index(self) -> int | None:
        return self._trace_cz_index

    @trace_cz_index.setter
    def trace_cz_index(self, value: int | None) -> None:
        self._trace_cz_index = value

    def __post_init__(self) -> None:
        if not isinstance(self.traversal, PlacementTraversalABC):
            raise TypeError(
                "traversal must implement PlacementTraversalABC "
                "(e.g., EntropyPlacementTraversal())"
            )

    def validate_initial_layout(
        self,
        initial_layout: tuple[layout.LocationAddress, ...],
    ) -> None:
        _ = initial_layout

    def _target_from_stage_controls_only(
        self,
        placement: dict[int, layout.LocationAddress],
        controls: tuple[int, ...],
        targets: tuple[int, ...],
    ) -> dict[int, layout.LocationAddress]:
        if len(placement) == 0:
            return {}
        target = dict(placement)
        for control_qid, target_qid in zip(controls, targets):
            target_loc = placement[target_qid]
            blockade_partner = self.arch_spec.get_blockaded_location(target_loc)
            assert blockade_partner is not None, f"No blockade partner for {target_loc}"
            target[control_qid] = blockade_partner
        return target

    def _run_search(
        self,
        tree: ConfigurationTree,
        target: dict[int, layout.LocationAddress],
        traversal: PlacementTraversalABC | None = None,
    ) -> SearchResult:
        active_traversal = self.traversal if traversal is None else traversal
        return active_traversal.path_to_target_config(
            tree=tree,
            target=target,
        )

    def cz_placements(
        self,
        state: AtomState,
        controls: tuple[int, ...],
        targets: tuple[int, ...],
        lookahead_cz_layers: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = (),
    ) -> AtomState:
        _ = lookahead_cz_layers
        if len(controls) != len(targets) or state == AtomState.bottom():
            return AtomState.bottom()
        if not isinstance(state, ConcreteState):
            return AtomState.top()

        placement = {qid: loc for qid, loc in enumerate(state.layout)}
        target = self._target_from_stage_controls_only(placement, controls, targets)
        tree = ConfigurationTree.from_initial_placement(
            self.arch_spec,
            placement,
            blocked_locations=state.occupied,
        )

        should_trace = (
            self._trace_cz_index is None or self._cz_counter == self._trace_cz_index
        )
        traversal = self.traversal
        if isinstance(traversal, EntropyPlacementTraversal) and not should_trace:
            traversal = replace(traversal, on_search_step=None)
        if should_trace:
            self._traced_tree = tree
            self._traced_target = dict(target)

        result = self._run_search(tree, target, traversal)
        self._cz_counter += 1

        if not result.goal_nodes:
            return AtomState.bottom()

        best_goal = result.goal_nodes[0]
        move_program = best_goal.to_move_program()
        goal_layout_map = best_goal.configuration
        goal_layout = tuple(goal_layout_map[qid] for qid in range(len(state.layout)))
        move_count = tuple(
            mc + int(src != dst)
            for mc, src, dst in zip(state.move_count, state.layout, goal_layout)
        )
        return ExecuteCZ(
            occupied=state.occupied,
            layout=goal_layout,
            move_count=move_count,
            active_cz_zones=frozenset([layout.ZoneAddress(0)]),
            move_layers=move_program,
        )

    def sq_placements(self, state: AtomState, qubits: tuple[int, ...]) -> AtomState:
        _ = qubits
        if isinstance(state, ConcreteState):
            return ConcreteState(
                occupied=state.occupied,
                layout=state.layout,
                move_count=state.move_count,
            )
        return state

    def measure_placements(
        self,
        state: AtomState,
        qubits: tuple[int, ...],
    ) -> AtomState:
        if not isinstance(state, ConcreteState):
            return state
        if len(qubits) != len(state.layout):
            return AtomState.bottom()
        return ExecuteMeasure(
            occupied=state.occupied,
            layout=state.layout,
            move_count=state.move_count,
            zone_maps=tuple(layout.ZoneAddress(0) for _ in qubits),
        )
