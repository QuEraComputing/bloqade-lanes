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
from bloqade.lanes.bytecode._native import MoveSolver
from bloqade.lanes.layout import (
    Direction,
    LaneAddress,
    LocationAddress,
    MoveType,
    ZoneAddress,
)
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


TraversalName = Literal["entropy", "greedy", "bfs", "rust"]
OnSearchStep = Callable[[str, "ConfigurationNode", "StepInfo"], None]


@dataclass
class PhysicalPlacementStrategy(PlacementStrategyABC):
    """Physical placement strategy backed by configuration-tree search."""

    arch_spec: layout.ArchSpec = field(default_factory=get_physical_arch_spec)
    traversal: TraversalName = "entropy"
    search_params: SearchParams = field(default_factory=SearchParams)
    max_depth: int | None = None
    max_expansions: int | None = 300
    on_search_step: OnSearchStep | None = None
    trace_cz_index: int | None = None

    _cz_counter: int = field(default=0, init=False, repr=False)
    _traced_tree: ConfigurationTree | None = field(default=None, init=False, repr=False)
    _traced_target: dict[int, layout.LocationAddress] = field(
        default_factory=dict, init=False, repr=False
    )
    _rust_solver: MoveSolver | None = field(default=None, init=False, repr=False)

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

        if self.traversal == "rust":
            return self._cz_placements_rust(state, controls, targets)

        placement = {qid: loc for qid, loc in enumerate(state.layout)}
        target = self._target_from_stage_controls_only(placement, controls, targets)
        tree = ConfigurationTree.from_initial_placement(
            self.arch_spec,
            placement,
            blocked_locations=state.occupied,
        )

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

    def _get_rust_solver(self) -> MoveSolver:
        if self._rust_solver is None:
            self._rust_solver = MoveSolver.from_arch_spec(self.arch_spec._inner)
        return self._rust_solver

    _DIR_MAP = {0: Direction.FORWARD, 1: Direction.BACKWARD}
    _MT_MAP = {0: MoveType.SITE, 1: MoveType.WORD}

    def _cz_placements_rust(
        self,
        state: ConcreteState,
        controls: tuple[int, ...],
        targets: tuple[int, ...],
    ) -> AtomState:
        placement = {qid: loc for qid, loc in enumerate(state.layout)}
        target = self._target_from_stage_controls_only(placement, controls, targets)

        initial = [(qid, loc.word_id, loc.site_id) for qid, loc in placement.items()]
        target_tuples = [(qid, loc.word_id, loc.site_id) for qid, loc in target.items()]
        blocked = [(loc.word_id, loc.site_id) for loc in state.occupied]

        solver = self._get_rust_solver()
        result = solver.solve(initial, target_tuples, blocked, self.max_expansions)

        if result is None:
            return AtomState.bottom()

        move_layers = tuple(
            tuple(
                LaneAddress(self._MT_MAP[mt], word, site, bus, self._DIR_MAP[d])
                for d, mt, word, site, bus in step
            )
            for step in result.move_layers
        )

        goal_map = {qid: LocationAddress(w, s) for qid, w, s in result.goal_config}
        goal_layout = tuple(goal_map[qid] for qid in range(len(state.layout)))

        move_count = tuple(
            mc + int(src != dst)
            for mc, src, dst in zip(state.move_count, state.layout, goal_layout)
        )

        return ExecuteCZ(
            occupied=state.occupied,
            layout=goal_layout,
            move_count=move_count,
            active_cz_zones=frozenset([ZoneAddress(0)]),
            move_layers=move_layers,
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
