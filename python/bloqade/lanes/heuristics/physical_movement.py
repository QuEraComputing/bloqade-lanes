from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

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
    ConfigurationTree,
    ExhaustiveMoveGenerator,
    SearchParams,
    bfs,
    entropy_guided_search,
    greedy_best_first,
    placement_goal,
)

if TYPE_CHECKING:
    from bloqade.lanes.search.configuration import ConfigurationNode
    from bloqade.lanes.search.traversal.goal import SearchResult
    from bloqade.lanes.search.traversal.step_info import StepInfo


TraversalName = Literal["entropy", "greedy", "bfs"]
OnSearchStep = Callable[[str, "ConfigurationNode", "StepInfo"], None]


@dataclass
class PhysicalPlacementStrategy(PlacementStrategyABC):
    """Physical placement strategy backed by configuration-tree search."""

    arch_spec: layout.ArchSpec = field(default_factory=get_physical_arch_spec)
    traversal: TraversalName = "entropy"
    search_params: SearchParams = field(default_factory=SearchParams)
    max_depth: int | None = None
    max_expansions: int | None = None
    on_search_step: OnSearchStep | None = None
    trace_cz_index: int | None = None

    _cz_counter: int = field(default=0, init=False, repr=False)
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

    def validate_initial_layout(
        self,
        initial_layout: tuple[layout.LocationAddress, ...],
    ) -> None:
        _ = initial_layout

    @staticmethod
    def _paired_site(site_id: int, half: int) -> int:
        if site_id < half:
            return site_id + half
        return site_id - half

    def _target_from_stage_controls_only(
        self,
        placement: dict[int, layout.LocationAddress],
        controls: tuple[int, ...],
        targets: tuple[int, ...],
    ) -> dict[int, layout.LocationAddress]:
        if len(placement) == 0:
            return {}
        n_sites = len(self.arch_spec.words[0].site_indices)
        half = n_sites // 2
        target = dict(placement)
        for control_qid, target_qid in zip(controls, targets):
            target_loc = placement[target_qid]
            dst_site = self._paired_site(target_loc.site_id, half)
            target[control_qid] = target_loc.replace(site_id=dst_site)
        return target

    @staticmethod
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

    def _run_search(
        self,
        tree: ConfigurationTree,
        target: dict[int, layout.LocationAddress],
        callback: OnSearchStep | None,
    ) -> SearchResult:
        goal = placement_goal(target)
        if self.traversal == "entropy":
            return entropy_guided_search(
                tree=tree,
                target=target,
                goal=goal,
                params=self.search_params,
                max_depth=self.max_depth,
                max_expansions=self.max_expansions,
                on_step=callback,
            )

        generator = ExhaustiveMoveGenerator()
        if self.traversal == "greedy":
            return greedy_best_first(
                tree=tree,
                generator=generator,
                goal=goal,
                heuristic=self._mismatch_heuristic(target),
                max_expansions=self.max_expansions,
            )
        if self.traversal == "bfs":
            return bfs(
                tree=tree,
                generator=generator,
                goal=goal,
                max_depth=self.max_depth,
                max_expansions=self.max_expansions,
            )
        raise ValueError(f"Unknown traversal strategy: {self.traversal}")

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
        tree = ConfigurationTree.from_initial_placement(self.arch_spec, placement)

        should_trace = (
            self.trace_cz_index is None or self._cz_counter == self.trace_cz_index
        )
        callback = (
            self.on_search_step
            if should_trace and self.traversal == "entropy"
            else None
        )
        if should_trace:
            self._traced_tree = tree
            self._traced_target = dict(target)

        result = self._run_search(tree, target, callback)
        self._cz_counter += 1

        if result.goal_node is None:
            return AtomState.bottom()

        move_program = result.goal_node.to_move_program()
        goal_layout_map = result.goal_node.configuration
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
