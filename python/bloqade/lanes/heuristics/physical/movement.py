from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

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
from bloqade.lanes.bytecode._native import MoveSolver
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


_DIR_MAP = {0: Direction.FORWARD, 1: Direction.BACKWARD}
_MT_MAP = {0: MoveType.SITE, 1: MoveType.WORD, 2: MoveType.ZONE}


def _convert_move_layers(
    raw_layers: list[list[tuple[int, int, int, int, int, int]]],
) -> tuple[tuple[LaneAddress, ...], ...]:
    """Convert Rust solver move_layers to Python LaneAddress tuples."""
    return tuple(
        tuple(
            LaneAddress(
                _MT_MAP[mt],
                word,
                site,
                bus,
                _DIR_MAP[d],
                zone,
            )
            for d, mt, zone, word, site, bus in step
        )
        for step in raw_layers
    )


@dataclass(frozen=True)
class RustPlacementTraversal:
    """Config for the Rust MoveSolver.

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
    collect_entropy_trace: bool = False


@dataclass
class PhysicalPlacementStrategy(PlacementStrategyABC):
    """Physical placement strategy backed by the Rust MoveSolver."""

    arch_spec: layout.ArchSpec = field(default_factory=get_physical_arch_spec)
    traversal: RustPlacementTraversal = field(
        default_factory=lambda: RustPlacementTraversal(strategy="entropy")
    )
    target_generator: TargetGeneratorABC | TargetGeneratorCallable | None = None

    _cz_counter: int = field(default=0, init=False, repr=False)
    _trace_cz_index: int | None = field(default=None, init=False, repr=False)
    _rust_solver: MoveSolver | None = field(default=None, init=False, repr=False)
    _rust_nodes_expanded_total: int = field(default=0, init=False, repr=False)
    _traced_rust_trace_json: str | None = field(default=None, init=False, repr=False)
    _resolved_target_generator: TargetGeneratorABC | None = field(
        default=None, init=False, repr=False
    )

    @property
    def trace_cz_index(self) -> int | None:
        return self._trace_cz_index

    @trace_cz_index.setter
    def trace_cz_index(self, value: int | None) -> None:
        self._trace_cz_index = value

    def __post_init__(self) -> None:
        assert_single_cz_zone(self.arch_spec, type(self).__name__)
        if not isinstance(self.traversal, RustPlacementTraversal):
            raise TypeError("traversal must be a RustPlacementTraversal instance")
        if self.target_generator is not None and not (
            isinstance(self.target_generator, TargetGeneratorABC)
            or callable(self.target_generator)
        ):
            raise TypeError(
                "target_generator must be a TargetGeneratorABC, a callable, or None"
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
        return self._cz_placements_rust(state, controls, targets, lookahead_cz_layers)

    def _get_rust_solver(self) -> MoveSolver:
        if self._rust_solver is None:
            self._rust_solver = MoveSolver.from_arch_spec(self.arch_spec._inner)
        return self._rust_solver

    @property
    def rust_nodes_expanded_total(self) -> int:
        """Total Rust solver node expansions for this strategy instance."""
        return self._rust_nodes_expanded_total

    @property
    def traced_rust_trace_json(self) -> str | None:
        return self._traced_rust_trace_json

    def _cz_placements_rust(
        self,
        state: ConcreteState,
        controls: tuple[int, ...],
        targets: tuple[int, ...],
        lookahead_cz_layers: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = (),
    ) -> AtomState:
        ctx = TargetContext(
            arch_spec=self.arch_spec,
            state=state,
            controls=controls,
            targets=targets,
            lookahead_cz_layers=lookahead_cz_layers,
            cz_stage_index=self._cz_counter,
        )
        candidates = self._build_candidates(ctx)
        should_trace = (
            self._trace_cz_index is None or self._cz_counter == self._trace_cz_index
        )
        if should_trace:
            self._traced_rust_trace_json = None

        solver = self._get_rust_solver()
        initial_native = {qid: loc._inner for qid, loc in ctx.placement.items()}
        blocked_native = [loc._inner for loc in state.occupied]
        opts = _native.SolveOptions(
            strategy=_STRATEGY_MAP[self.traversal.strategy],
            max_movesets_per_group=self.traversal.max_movesets_per_group,
            max_goal_candidates=self.traversal.max_goal_candidates,
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
                    self._traced_rust_trace_json = result.entropy_trace_json
                break

        self._cz_counter += 1

        if winning_result is None:
            return AtomState.bottom()

        move_layers = _convert_move_layers(winning_result.move_layers)

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
