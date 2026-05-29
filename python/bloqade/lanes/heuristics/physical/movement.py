from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from bloqade.lanes.analysis.placement import (
    AtomState,
    ConcreteState,
    ExecuteCZ,
    ExecuteMeasure,
    PalindromePlacementStrategy,
    PlacementStrategyABC,
)
from bloqade.lanes.analysis.placement.lattice import UserMoved
from bloqade.lanes.analysis.placement.strategy import assert_single_cz_zone
from bloqade.lanes.arch.gemini.physical import get_arch_spec as get_physical_arch_spec
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode import _native
from bloqade.lanes.bytecode._native import EntropyTrace, SearchEngine
from bloqade.lanes.bytecode.encoding import (
    LaneAddress,
    LocationAddress,
    ZoneAddress,
)
from bloqade.lanes.heuristics.physical._solver_dispatch import _STRATEGY_MAP
from bloqade.lanes.heuristics.physical.target_generator import (
    DefaultTargetGenerator,
    TargetContext,
    TargetGeneratorABC,
    TargetGeneratorCallable,
    _coerce_target_generator,
    _validate_candidate,
)

SearchStrategyName = Literal[
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
]


def convert_move_layers(
    raw_layers,
) -> tuple[tuple[LaneAddress, ...], ...]:
    """Wrap Rust solver move_layers into Python ``LaneAddress`` tuples."""
    return tuple(
        tuple(LaneAddress.from_inner(lane) for lane in step) for step in raw_layers
    )


@dataclass(frozen=True)
class RustPlacementTraversal:
    """Config for the Rust search engine (``TargetSolver``).

    ``restarts`` and ``lookahead`` are now exposed and threaded into
    ``SolveOptions``; per-strategy entropy knobs (``max_movesets_per_group``,
    ``max_goal_candidates``, ``collect_entropy_trace``) feed ``EntropyOptions``
    via :func:`_move_search_from_traversal`.

    Not yet exposed (Rust defaults used): ``weight``, ``deadlock_policy``,
    ``w_t``. These will be threaded through once validated via Rust-only
    benchmarking.
    """

    strategy: SearchStrategyName = "entropy"
    max_movesets_per_group: int = 3
    max_goal_candidates: int = 3
    max_expansions: int | None = 300
    restarts: int = 1
    lookahead: bool = False
    collect_entropy_trace: bool = False


def _move_search_from_traversal(
    traversal: RustPlacementTraversal,
    *,
    collect_entropy_trace: bool = False,
) -> _native.MoveSearch:
    """Build a ``MoveSearch`` bundle from a ``RustPlacementTraversal``.

    Strategy, restarts, lookahead, and entropy knobs are baked into a single
    immutable value; ``collect_entropy_trace`` overrides the traversal flag.
    """
    solve_opts = _native.SolveOptions(
        strategy=_STRATEGY_MAP[traversal.strategy],
        restarts=traversal.restarts,
        lookahead=traversal.lookahead,
    )
    entropy_opts = _native.EntropyOptions(
        max_movesets_per_group=traversal.max_movesets_per_group,
        max_goal_candidates=traversal.max_goal_candidates,
        collect_entropy_trace=collect_entropy_trace,
    )
    return (
        _native.MoveSearch.entropy()
        .with_options(solve_opts)
        .with_entropy_options(entropy_opts)
    )


@dataclass
class PhysicalPlacementStrategy(PlacementStrategyABC):
    """Physical placement strategy backed by the Rust ``TargetSolver``."""

    arch_spec: ArchSpec = field(default_factory=get_physical_arch_spec)
    traversal: RustPlacementTraversal = field(
        default_factory=lambda: RustPlacementTraversal(strategy="entropy")
    )
    target_generator: TargetGeneratorABC | TargetGeneratorCallable | None = None

    _cz_counter: int = field(default=0, init=False, repr=False)
    _trace_cz_index: int | None = field(default=None, init=False, repr=False)
    _engine: SearchEngine | None = field(default=None, init=False, repr=False)
    _rust_nodes_expanded_total: int = field(default=0, init=False, repr=False)
    _rust_entropy_fallback_count: int = field(default=0, init=False, repr=False)
    _traced_rust_entropy_trace: EntropyTrace | None = field(
        default=None, init=False, repr=False
    )
    _traced_target: dict[int, LocationAddress] = field(
        default_factory=dict, init=False, repr=False
    )
    _traced_blocked_locations: tuple[LocationAddress, ...] = field(
        default=(), init=False, repr=False
    )
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
        initial_layout: tuple[LocationAddress, ...],
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

    def _get_engine(self) -> SearchEngine:
        if self._engine is None:
            self._engine = SearchEngine.from_arch_spec(self.arch_spec._inner)
        return self._engine

    def _make_target_solver(
        self, move_search: _native.MoveSearch
    ) -> _native.TargetSolver:
        return _native.TargetSolver(self._get_engine(), move_search)

    @property
    def rust_nodes_expanded_total(self) -> int:
        """Total Rust solver node expansions for this strategy instance."""
        return self._rust_nodes_expanded_total

    @property
    def rust_entropy_fallback_count(self) -> int:
        """Number of solved Rust entropy stages that used sequential fallback."""
        return self._rust_entropy_fallback_count

    @property
    def traced_rust_entropy_trace(self) -> EntropyTrace | None:
        return self._traced_rust_entropy_trace

    @property
    def traced_target(self) -> dict[int, LocationAddress]:
        """First candidate target for the traced CZ layer (used by visualizers)."""
        return dict(self._traced_target)

    @property
    def traced_blocked_locations(self) -> tuple[LocationAddress, ...]:
        """Spectator atom positions for the traced CZ layer (atoms not in the active placement)."""
        return self._traced_blocked_locations

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
            self._traced_rust_entropy_trace = None
            self._traced_target = dict(candidates[0])
            active_locations = set(ctx.placement.values())
            self._traced_blocked_locations = tuple(
                loc for loc in state.occupied if loc not in active_locations
            )

        initial_native = {qid: loc._inner for qid, loc in ctx.placement.items()}
        blocked_native = [loc._inner for loc in state.occupied]
        move_search = _move_search_from_traversal(
            self.traversal,
            collect_entropy_trace=(
                should_trace and self.traversal.collect_entropy_trace
            ),
        )
        solver = self._make_target_solver(move_search)

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
                remaining,
            )
            self._rust_nodes_expanded_total += int(result.nodes_expanded)
            if remaining is not None:
                # Invariant: the Rust solver expands ≥ 1 node per call (even
                # when unsolvable), so the shared budget makes forward progress
                # across candidates and this loop terminates.
                remaining -= int(result.nodes_expanded)
            if result.status == "solved":
                winning_result = result
                if should_trace and self.traversal.collect_entropy_trace:
                    trace = result.entropy_trace
                    self._traced_rust_entropy_trace = trace
                    if trace is not None and any(
                        step.event == "fallback_start" for step in trace.steps
                    ):
                        self._rust_entropy_fallback_count += 1
                break

        self._cz_counter += 1

        if winning_result is None:
            return AtomState.bottom()

        move_layers = convert_move_layers(winning_result.move_layers)

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
        if isinstance(state, UserMoved):
            return AtomState.bottom()  # move_to before SQ gate is invalid
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
        if isinstance(state, UserMoved):
            return AtomState.bottom()  # move_to before measurement is invalid
        if not isinstance(state, ConcreteState):
            return state
        if len(qubits) != len(state.layout):
            return AtomState.bottom()
        return ExecuteMeasure(
            occupied=state.occupied,
            layout=state.layout,
            move_count=state.move_count,
            zone_maps=tuple(ZoneAddress(loc.zone_id) for loc in state.layout),
        )


def make_physical_placement_strategy(
    *,
    move_solutions_per_layer: int = 3,
    search_budget: int | None = 300,
    strategy: SearchStrategyName = "entropy",
    arch_spec: ArchSpec | None = None,
    return_moves: bool = True,
) -> PlacementStrategyABC:
    """Build a physical placement strategy from user-facing search knobs.

    Uses :class:`~bloqade.lanes.heuristics.physical.nohome.NoHomePlacementStrategy`
    backed by the Rust ``NoHomeCzPlacement`` solver.

    ``move_solutions_per_layer`` maps to ``k_candidates`` (candidate home
    sites per qubit in the Hungarian assignment).  ``search_budget`` maps to
    ``max_expansions``.  ``lambda_lookahead`` is fixed at ``0`` because
    palindrome return always moves atoms back to their original home position,
    so future-layer proximity penalties carry no signal.
    """
    from bloqade.lanes.heuristics.physical.nohome import NoHomePlacementStrategy

    if move_solutions_per_layer < 1:
        raise ValueError("move_solutions_per_layer must be >= 1")
    if search_budget is not None and search_budget < 1:
        raise ValueError("search_budget must be None or >= 1")

    inner = NoHomePlacementStrategy(
        arch_spec=get_physical_arch_spec() if arch_spec is None else arch_spec,
        strategy=_STRATEGY_MAP[strategy],
        max_expansions=search_budget,
        k_candidates=move_solutions_per_layer,
        lambda_lookahead=0.0,
    )

    return PalindromePlacementStrategy(inner=inner) if return_moves else inner
