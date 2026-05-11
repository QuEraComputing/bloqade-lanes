"""Receding-horizon (MPC-style) loose-goal placement strategy.

At each stage, generates K diverse Hungarian candidate assignments, runs
short forward IDS rollouts of each, commits the best branch's path, and
re-plans. Targeted at high-occupancy regimes where the baseline
``LooseGoalPlacementStrategy`` under-uses parallelism — in low-density
regimes, the K-branch rollout structure is overhead and the baseline is
preferred.

See ``plans/2026-05-11-receding-horizon-loose-goal-design.md`` for the
full design rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from bloqade.lanes.analysis.placement import (
    AtomState,
    ConcreteState,
    ExecuteCZ,
    ExecuteMeasure,
    PlacementStrategyABC,
)
from bloqade.lanes.analysis.placement.strategy import assert_single_cz_zone
from bloqade.lanes.bytecode import _native
from bloqade.lanes.bytecode._native import MoveSolver
from bloqade.lanes.bytecode.encoding import (
    LaneAddress,
    LocationAddress,
    ZoneAddress,
)


@dataclass
class RecedingHorizonLooseGoalPlacementStrategy(PlacementStrategyABC):
    """Loose-goal placement using a receding-horizon orchestration.

    Differs from :class:`LooseGoalPlacementStrategy` in that it does **not**
    commit to one Hungarian assignment up front. Instead, at each stage:

    1. Generates ``k_candidates`` diverse Hungarian candidates from the
       current state (using the configured ``weight_grid``).
    2. Runs each candidate forward for ``rollout_horizon`` move layers via
       the existing IDS infrastructure.
    3. Picks the best branch by stratified score (tier-0: goal reached
       mid-rollout; tier-1: completed full horizon; tier-2: dropped).
    4. Commits the full path of a tier-0 winner, or the first
       ``commit_depth`` layers of a tier-1 winner, then re-plans.

    Positioning
    -----------
    Use this strategy **when atom density is high** and the baseline
    loose-goal under-uses parallelism (the "1.23 vs 1.49 lanes/layer" gap
    flagged in the PR description). In low-density regimes, most rollouts
    will hit the constraint goal before depth ``x``, collapsing the
    algorithm to "K parallel searches, pick the shortest" — which the
    baseline restart mechanism already does more cheaply.

    Parameters
    ----------
    arch_spec
        Architecture specification.
    strategy
        Inner search strategy used for rollouts. Currently only ``"ids"``
        is exercised; passing other values still works but the receding-
        horizon orchestration is tuned for IDS.
    max_expansions
        Optional cap on **total** node expansions across all stages of one
        restart's trajectory.
    restarts
        Number of parallel restart trajectories. Each restart runs its own
        independent receding-horizon solve with a distinct seed; the
        lowest-cost trajectory wins. For evaluation, use ``restarts=1`` to
        isolate the algorithm from the orthogonal restart-parallelism
        axis; the design's intended high-density-regime gains should be
        visible per-restart.
    congestion_weight, occupancy_penalty, hungarian_horizon
        Underlying Hungarian cost-matrix knobs passed to every candidate
        generation call (same semantics as
        :class:`LooseGoalPlacementStrategy`). The weight_grid varies
        ``congestion_weight`` and ``occupancy_penalty`` *per candidate*;
        these field values are used by the noise top-up fallback when the
        grid produces fewer than K unique assignments.
    top_c
        Per-qubit move-candidate pruning cap inside ``HeuristicGenerator``
        (same as base strategy).
    deadlock_policy
        ``"skip"`` | ``"move_blockers"`` (default) | ``"all_moves"``.
    k_candidates
        ``K``: number of Hungarian candidates tried per stage. Default 10.
    rollout_horizon
        ``x``: maximum number of move layers each branch's inner search
        explores. Default 5.
    commit_depth
        ``m``: number of layers from the winning *tier-1* branch's path
        that get committed before re-planning. Must satisfy
        ``1 <= commit_depth <= rollout_horizon``. Tier-0 winners (rollouts
        that reached the goal) always commit their full path regardless of
        this value. Default 1.
    tier0_next_h_weight
        α: weight on next-layer Hungarian cost when ranking tier-0
        branches. ``0.0`` ignores next-layer setup; higher values trade
        current-layer commitment depth for better next-layer staging.
        Default 0.5.
    weight_grid
        Sequence of ``(congestion_weight, occupancy_penalty)`` tuples used
        to generate K candidates per stage. ``None`` uses a default 10-
        entry grid spanning ``congestion ∈ {0, 0.5, 1, 2, 5}`` ×
        ``occupancy ∈ {0.5, 2.0}``.
    fallback_x_decrement
        When all K branches drop (tier-2) at the current horizon, retry the
        stage with ``x ← x − decrement``. Default 1.
    branch_parallel
        Run the K rollouts in parallel via rayon within a stage. Default
        ``True``. When ``restarts > 1``, set ``False`` to reserve cores for
        restart parallelism.
    max_expansions_per_rollout
        Per-rollout expansion budget. Caps any single rollout to bound
        runaway. Default 1000.
    """

    strategy: str = "ids"
    max_expansions: int | None = 100
    restarts: int = 1
    congestion_weight: float = 0.0
    occupancy_penalty: float = 1.0
    hungarian_horizon: int | None = 4
    top_c: int | None = 3
    deadlock_policy: str = "move_blockers"

    # Receding-horizon-specific orchestration knobs.
    # Defaults calibrated from the 40q × 20-pair × depth-3 sweep
    # (scripts/eval_sweep_m.py): K=5 / m=5 is the cost/quality knee — ~24%
    # move-layer reduction vs LooseGoal(cw=1.0) at ~8× baseline wall-clock.
    # For best quality at higher cost, use k_candidates=10 (~3% better, 2×
    # slower). For most re-planning (commit-depth=1), expect ~3% better
    # quality at ~3.5× slower.
    k_candidates: int = 5
    rollout_horizon: int = 5
    commit_depth: int = 5
    tier0_next_h_weight: float = 0.5
    weight_grid: tuple[tuple[float, float], ...] | None = None
    fallback_x_decrement: int = 1
    branch_parallel: bool = True
    max_expansions_per_rollout: int = 1000

    _solver: MoveSolver | None = field(default=None, init=False, repr=False)

    _STRATEGY_MAP: dict[str, _native.SearchStrategy] = field(
        default_factory=lambda: {
            "ids": _native.SearchStrategy.IDS,
            "cascade": _native.SearchStrategy.CASCADE_IDS,
            "cascade-ids": _native.SearchStrategy.CASCADE_IDS,
            "entropy": _native.SearchStrategy.ENTROPY,
            "astar": _native.SearchStrategy.ASTAR,
        },
        init=False,
        repr=False,
    )

    _DEADLOCK_MAP: dict[str, _native.DeadlockPolicy] = field(
        default_factory=lambda: {
            "skip": _native.DeadlockPolicy.SKIP,
            "move_blockers": _native.DeadlockPolicy.MOVE_BLOCKERS,
            "all_moves": _native.DeadlockPolicy.ALL_MOVES,
        },
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        assert_single_cz_zone(self.arch_spec, type(self).__name__)
        if self.commit_depth < 1 or self.commit_depth > self.rollout_horizon:
            raise ValueError(
                "commit_depth must satisfy 1 <= commit_depth <= rollout_horizon "
                f"(got commit_depth={self.commit_depth}, "
                f"rollout_horizon={self.rollout_horizon})"
            )

    def _get_solver(self) -> MoveSolver:
        if self._solver is None:
            self._solver = MoveSolver.from_arch_spec(self.arch_spec._inner)
        return self._solver

    def _make_options(
        self,
    ) -> tuple[
        _native.SolveOptions, _native.EntanglingOptions, _native.RecedingHorizonOptions
    ]:
        opts = _native.SolveOptions(
            strategy=self._STRATEGY_MAP[self.strategy],
            restarts=self.restarts,
            lookahead=True,
            deadlock_policy=self._DEADLOCK_MAP[self.deadlock_policy],
            top_c=self.top_c,
        )
        ent_opts = _native.EntanglingOptions(
            congestion_weight=self.congestion_weight,
            occupancy_penalty=self.occupancy_penalty,
            hungarian_horizon=self.hungarian_horizon,
        )
        grid: list[tuple[float, float]] | None = (
            [(float(cw), float(op)) for cw, op in self.weight_grid]
            if self.weight_grid is not None
            else None
        )
        rh_opts = _native.RecedingHorizonOptions(
            k_candidates=self.k_candidates,
            rollout_horizon=self.rollout_horizon,
            commit_depth=self.commit_depth,
            tier0_next_h_weight=self.tier0_next_h_weight,
            weight_grid=grid,
            fallback_x_decrement=self.fallback_x_decrement,
            branch_parallel=self.branch_parallel,
            max_expansions_per_rollout=self.max_expansions_per_rollout,
        )
        return opts, ent_opts, rh_opts

    def validate_initial_layout(
        self, initial_layout: tuple[LocationAddress, ...]
    ) -> None:
        pass

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

        solver = self._get_solver()
        options, entangling_options, rh_options = self._make_options()

        initial = {qid: state.layout[qid]._inner for qid in range(len(state.layout))}
        cz_pairs = list(zip(controls, targets))
        blocked = [loc._inner for loc in state.occupied]

        future = None
        if len(lookahead_cz_layers) > 1:
            future = [list(zip(ctrls, tgts)) for ctrls, tgts in lookahead_cz_layers[1:]]

        result = solver.solve_entangling_rh(
            initial,
            cz_pairs,
            blocked,
            max_expansions=self.max_expansions,
            options=options,
            entangling_options=entangling_options,
            rh_options=rh_options,
            future_cz_layers=future,
        )

        if result.status != "solved":
            return AtomState.bottom()

        move_layers = tuple(
            tuple(LaneAddress.from_inner(lane) for lane in step)
            for step in result.move_layers
        )

        goal_map = {
            qid: LocationAddress(loc.word_id, loc.site_id, loc.zone_id)
            for qid, loc in result.goal_config.items()
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
        if isinstance(state, ConcreteState):
            return ConcreteState(
                occupied=state.occupied,
                layout=state.layout,
                move_count=state.move_count,
            )
        return state

    def measure_placements(
        self, state: AtomState, qubits: tuple[int, ...]
    ) -> AtomState:
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
