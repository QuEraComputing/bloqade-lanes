from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, cast

from bloqade.lanes import layout
from bloqade.lanes.analysis.placement import (
    AtomState,
    ConcreteState,
    ExecuteCZ,
    ExecuteMeasure,
    PlacementStrategyABC,
)
from bloqade.lanes.analysis.placement.strategy import assert_single_cz_zone
from bloqade.lanes.arch.gemini.physical import get_arch_spec as get_physical_arch_spec
from bloqade.lanes.bytecode import _native
from bloqade.lanes.bytecode._native import EntropyTrace, MoveSolver
from bloqade.lanes.heuristics.physical.target_generator import (
    DefaultTargetGenerator,
    TargetContext,
    TargetGeneratorABC,
    TargetGeneratorCallable,
    _coerce_target_generator,
    _validate_candidate,
)
from bloqade.lanes.layout import (
    Direction,
    LaneAddress,
    LocationAddress,
    MoveType,
)
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

_STRATEGY_MAP: dict[str, _native.SearchStrategy] = {
    "astar": _native.SearchStrategy.ASTAR,
    "dfs": _native.SearchStrategy.DFS,
    "bfs": _native.SearchStrategy.BFS,
    "greedy": _native.SearchStrategy.GREEDY,
    "ids": _native.SearchStrategy.IDS,
    "cascade": _native.SearchStrategy.CASCADE_IDS,
    "cascade-ids": _native.SearchStrategy.CASCADE_IDS,
    "cascade-dfs": _native.SearchStrategy.CASCADE_DFS,
    "cascade-entropy": _native.SearchStrategy.CASCADE_ENTROPY,
    "entropy": _native.SearchStrategy.ENTROPY,
}


class PlacementTraversalABC(abc.ABC):
    """Placement-facing traversal API for target-configuration search."""

    max_expansions: int | None

    @abc.abstractmethod
    def path_to_target_config(
        self,
        *,
        tree: ConfigurationTree,
        target: dict[int, layout.LocationAddress],
    ) -> SearchResult:
        """Run search and return one or more goal nodes."""
        ...

    def with_max_expansions(
        self, max_expansions: int | None
    ) -> "PlacementTraversalABC":
        """Return a copy of this traversal with an overridden budget.

        The default implementation uses :func:`dataclasses.replace`, which
        requires the subclass to be a dataclass (all built-in subclasses are).
        Override this method if your subclass is not a dataclass.
        """
        return replace(cast(Any, self), max_expansions=max_expansions)


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
    max_expansions: int | None = 300
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

    max_expansions: int | None = 1000

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


@dataclass(frozen=True)
class RustPlacementTraversal:
    """Config for the Rust MoveSolver — bypasses ConfigurationTree entirely.

    Note: The Rust ``MoveSolver.solve()`` accepts additional tuning parameters
    (weight, restarts, lookahead, deadlock_policy, w_t) that are not yet
    exposed here; Rust defaults are used. These will be threaded through
    once the parameters are validated via Rust-only benchmarking.
    """

    strategy: Literal[
        "astar",
        "dfs",
        "bfs",
        "greedy",
        "ids",
        "cascade",
        "cascade-ids",
        "cascade-dfs",
        "cascade-entropy",
        "entropy",
    ] = "astar"
    max_movesets_per_group: int = 3
    max_goal_candidates: int = 3
    max_expansions: int | None = 300
    restarts: int = 1
    lookahead: bool = False
    collect_entropy_trace: bool = False


@dataclass
class PhysicalPlacementStrategy(PlacementStrategyABC):
    """Physical placement strategy backed by configuration-tree search."""

    arch_spec: layout.ArchSpec = field(default_factory=get_physical_arch_spec)
    traversal: PlacementTraversalABC | RustPlacementTraversal = field(
        default_factory=EntropyPlacementTraversal
    )
    target_generator: TargetGeneratorABC | TargetGeneratorCallable | None = None

    _cz_counter: int = field(default=0, init=False, repr=False)
    _trace_cz_index: int | None = field(default=None, init=False, repr=False)
    _traced_tree: ConfigurationTree | None = field(default=None, init=False, repr=False)
    _traced_target: dict[int, layout.LocationAddress] = field(
        default_factory=dict, init=False, repr=False
    )
    _rust_solver: MoveSolver | None = field(default=None, init=False, repr=False)
    _rust_nodes_expanded_total: int = field(default=0, init=False, repr=False)
    _traced_rust_entropy_trace: EntropyTrace | None = field(
        default=None, init=False, repr=False
    )
    _resolved_target_generator: TargetGeneratorABC | None = field(
        default=None, init=False, repr=False
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
        assert_single_cz_zone(self.arch_spec, type(self).__name__)
        if not isinstance(
            self.traversal, (PlacementTraversalABC, RustPlacementTraversal)
        ):
            raise TypeError(
                "traversal must implement PlacementTraversalABC or be a "
                "RustPlacementTraversal instance"
            )
        if self.target_generator is not None and not (
            isinstance(self.target_generator, TargetGeneratorABC)
            or callable(self.target_generator)
        ):
            raise TypeError(
                "target_generator must be a TargetGeneratorABC, a callable, " "or None"
            )
        self._resolved_target_generator = _coerce_target_generator(
            self.target_generator
        )

    def validate_initial_layout(
        self,
        initial_layout: tuple[layout.LocationAddress, ...],
    ) -> None:
        _ = initial_layout

    def _build_candidates(
        self,
        ctx: TargetContext,
    ) -> list[dict[int, LocationAddress]]:
        """Build the ordered candidate list: plugin output + default-as-fallback.

        Dedups plugin candidates by dict equality (preserving order) and
        appends the default candidate only if it is not already present.
        Validates every candidate against ``_validate_candidate`` before
        returning; a malformed candidate raises ``ValueError``.
        """
        plugin = self._resolved_target_generator
        plugin_candidates: list[dict[int, LocationAddress]] = (
            [] if plugin is None else list(plugin.generate(ctx))
        )

        deduped: list[dict[int, LocationAddress]] = []
        for candidate in plugin_candidates:
            _validate_candidate(ctx, candidate)
            if candidate not in deduped:
                deduped.append(candidate)

        default = DefaultTargetGenerator().generate(ctx)[0]
        if default not in deduped:
            deduped.append(default)
        return deduped

    def _run_search(
        self,
        tree: ConfigurationTree,
        target: dict[int, layout.LocationAddress],
        traversal: PlacementTraversalABC | None = None,
    ) -> SearchResult:
        active_traversal = self.traversal if traversal is None else traversal
        assert isinstance(active_traversal, PlacementTraversalABC)
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
        if len(controls) != len(targets) or state == AtomState.bottom():
            return AtomState.bottom()
        if not isinstance(state, ConcreteState):
            return AtomState.top()

        if isinstance(self.traversal, RustPlacementTraversal):
            return self._cz_placements_rust(
                state, controls, targets, lookahead_cz_layers
            )

        ctx = TargetContext(
            arch_spec=self.arch_spec,
            state=state,
            controls=controls,
            targets=targets,
            lookahead_cz_layers=lookahead_cz_layers,
            cz_stage_index=self._cz_counter,
        )
        candidates = self._build_candidates(ctx)

        tree = ConfigurationTree.from_initial_placement(
            self.arch_spec,
            ctx.placement,
            blocked_locations=state.occupied,
        )

        should_trace = (
            self._trace_cz_index is None or self._cz_counter == self._trace_cz_index
        )
        base_traversal = self.traversal
        assert isinstance(base_traversal, PlacementTraversalABC)
        if isinstance(base_traversal, EntropyPlacementTraversal) and not should_trace:
            base_traversal = replace(base_traversal, on_search_step=None)
        if should_trace:
            self._traced_tree = tree
            self._traced_target = dict(candidates[0])

        remaining = base_traversal.max_expansions
        best_goal = None
        for candidate in candidates:
            if remaining is not None and remaining <= 0:
                break
            per_call_traversal = (
                base_traversal
                if remaining == base_traversal.max_expansions
                else base_traversal.with_max_expansions(remaining)
            )
            result = self._run_search(tree, candidate, per_call_traversal)
            if remaining is not None:
                remaining -= int(result.nodes_expanded)
            if result.goal_nodes:
                best_goal = result.goal_nodes[0]
                break

        self._cz_counter += 1

        if best_goal is None:
            return AtomState.bottom()

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
            active_cz_zones=self.arch_spec.cz_zone_addresses,
            move_layers=move_program,
        )

    def _get_rust_solver(self) -> MoveSolver:
        if self._rust_solver is None:
            self._rust_solver = MoveSolver.from_arch_spec(self.arch_spec._inner)
        return self._rust_solver

    @property
    def rust_nodes_expanded_total(self) -> int:
        """Total Rust solver node expansions for this strategy instance."""
        return self._rust_nodes_expanded_total

    @property
    def traced_rust_entropy_trace(self) -> EntropyTrace | None:
        return self._traced_rust_entropy_trace

    def _cz_placements_rust(
        self,
        state: ConcreteState,
        controls: tuple[int, ...],
        targets: tuple[int, ...],
        lookahead_cz_layers: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = (),
    ) -> AtomState:
        assert isinstance(self.traversal, RustPlacementTraversal)
        ctx = TargetContext(
            arch_spec=self.arch_spec,
            state=state,
            controls=controls,
            targets=targets,
            lookahead_cz_layers=lookahead_cz_layers,
            cz_stage_index=self._cz_counter,
        )
        candidates = self._build_candidates(ctx)
        tree = ConfigurationTree.from_initial_placement(
            self.arch_spec,
            ctx.placement,
            blocked_locations=state.occupied,
        )
        should_trace = (
            self._trace_cz_index is None or self._cz_counter == self._trace_cz_index
        )
        if should_trace:
            self._traced_tree = tree
            self._traced_target = dict(candidates[0])
            self._traced_rust_entropy_trace = None

        solver = self._get_rust_solver()
        initial_native = {qid: loc._inner for qid, loc in ctx.placement.items()}
        blocked_native = [loc._inner for loc in state.occupied]
        opts = _native.SolveOptions(
            strategy=_STRATEGY_MAP[self.traversal.strategy],
            max_movesets_per_group=self.traversal.max_movesets_per_group,
            max_goal_candidates=self.traversal.max_goal_candidates,
            restarts=self.traversal.restarts,
            lookahead=self.traversal.lookahead,
            collect_entropy_trace=(
                should_trace and self.traversal.collect_entropy_trace
            ),
        )

        remaining = self.traversal.max_expansions
        winning_result = None
        for candidate in candidates:
            if remaining is not None and remaining <= 0:
                break
            target_native = {qid: loc._inner for qid, loc in candidate.items()}
            result = solver.solve(
                initial_native,
                target_native,
                blocked_native,
                max_expansions=remaining,
                options=opts,
            )
            self._rust_nodes_expanded_total += int(result.nodes_expanded)
            if remaining is not None:
                remaining -= int(result.nodes_expanded)
            if result.status == "solved":
                winning_result = result
                if should_trace and self.traversal.collect_entropy_trace:
                    self._traced_rust_entropy_trace = result.entropy_trace
                break

        self._cz_counter += 1

        if winning_result is None:
            return AtomState.bottom()

        move_layers = tuple(
            tuple(LaneAddress.from_inner(lane) for lane in step)
            for step in winning_result.move_layers
        )

        goal_map = {
            qid: LocationAddress(loc.word_id, loc.site_id, loc.zone_id)
            for qid, loc in winning_result.goal_config.items()
        }
        goal_layout = tuple(goal_map[qid] for qid in range(len(state.layout)))
        move_count = tuple(
            mc + int(src != dst)
            for mc, src, dst in zip(state.move_count, state.layout, goal_layout)
        )
        return ExecuteCZ(
            occupied=state.occupied,
            layout=goal_layout,
            move_count=move_count,
            active_cz_zones=self.arch_spec.cz_zone_addresses,
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
            zone_maps=tuple(layout.ZoneAddress(loc.zone_id) for loc in state.layout),
        )
