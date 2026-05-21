"""Receding-horizon (MPC-style) no-return placement strategy.

At each stage, generates K diverse Hungarian candidate assignments, runs
short forward IDS rollouts of each, commits the best branch's path, and
re-plans. Targeted at high-occupancy regimes where the baseline
:class:`NoReturnPlacementStrategy` under-uses parallelism — in low-density
regimes, the K-branch rollout structure is overhead and the baseline is
preferred.

See ``docs/superpowers/plans/2026-05-11-receding-horizon-loose-goal-design.md`` for the
full design rationale.
"""

from __future__ import annotations

from dataclasses import dataclass

from bloqade.lanes.bytecode import _native
from bloqade.lanes.bytecode._native import MoveSolver, SolveResult
from bloqade.lanes.heuristics.physical._no_return_base import NoReturnStrategyBase


@dataclass
class RecedingHorizonNoReturnPlacementStrategy(NoReturnStrategyBase):
    """No-return placement using a receding-horizon orchestration.

    Differs from :class:`NoReturnPlacementStrategy` in that it does **not**
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
    no-return loose-goal solver under-uses parallelism (the "1.23 vs 1.49
    lanes/layer" gap flagged in the PR description). In low-density
    regimes, most rollouts will hit the constraint goal before depth
    ``x``, collapsing the algorithm to "K parallel searches, pick the
    shortest" — which the baseline restart mechanism already does more
    cheaply.

    Parameters
    ----------
    arch_spec
        Architecture specification.
    strategy
        Inner search strategy used for rollouts as a :class:`SearchStrategy`
        enum. Default :py:attr:`SearchStrategy.IDS` (the orchestration is
        tuned for IDS; other values still work but aren't routinely
        exercised).
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
        generation call (same semantics as :class:`NoReturnPlacementStrategy`).
        The ``weight_grid`` varies ``congestion_weight`` and
        ``occupancy_penalty`` *per candidate*; these field values are used
        by the noise top-up fallback when the grid produces fewer than
        ``k_candidates`` unique assignments.
    top_c
        Per-qubit move-candidate pruning cap inside ``HeuristicGenerator``
        (same as the base no-return strategy).
    deadlock_policy
        :class:`DeadlockPolicy` enum value (default
        :py:attr:`DeadlockPolicy.MOVE_BLOCKERS`).
    k_candidates
        ``K``: number of Hungarian candidates tried per stage. Default 5.
    rollout_horizon
        ``x``: maximum number of move layers each branch's inner search
        explores. Default 5.
    commit_depth
        ``m``: number of layers from the winning *tier-1* branch's path
        that get committed before re-planning. Must satisfy
        ``1 <= commit_depth <= rollout_horizon``. Tier-0 winners (rollouts
        that reached the goal) always commit their full path regardless
        of this value. Default 3.
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
        When all K branches drop (tier-2) at the current horizon, retry
        the stage with ``x ← x − decrement``. Default 1.
    branch_parallel
        Run the K rollouts in parallel via rayon within a stage. Default
        ``True``. When ``restarts > 1``, set ``False`` to reserve cores
        for restart parallelism.
    max_expansions_per_rollout
        Per-rollout expansion budget. Caps any single rollout to bound
        runaway. Default 300.
    greedy_first, inner_beam_width
        Inner-rollout cheap-beam-then-IDS knobs. See the Rust struct
        ``RecedingHorizonOptions`` doc for details.
    """

    top_c: int | None = 3
    congestion_weight: float = 0.0
    occupancy_penalty: float = 1.0
    hungarian_horizon: int | None = 4

    # Receding-horizon-specific orchestration knobs.
    # Defaults calibrated from the 80q × 30-pair × depth-3 sweep
    # (scripts/eval_sweep_m_80q.py): K=5 / m=3 is the cost/quality knee.
    # See receding_horizon.rs for the detailed rationale.
    k_candidates: int = 5
    rollout_horizon: int = 5
    commit_depth: int = 3
    tier0_next_h_weight: float = 0.5
    weight_grid: tuple[tuple[float, float], ...] | None = None
    fallback_x_decrement: int = 1
    branch_parallel: bool = True
    max_expansions_per_rollout: int = 300
    greedy_first: bool = True
    inner_beam_width: int = 2

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.commit_depth < 1 or self.commit_depth > self.rollout_horizon:
            raise ValueError(
                "commit_depth must satisfy 1 <= commit_depth <= rollout_horizon "
                f"(got commit_depth={self.commit_depth}, "
                f"rollout_horizon={self.rollout_horizon})"
            )

    def _build_entangling_options(self) -> _native.EntanglingOptions:
        return _native.EntanglingOptions(
            congestion_weight=self.congestion_weight,
            occupancy_penalty=self.occupancy_penalty,
            hungarian_horizon=self.hungarian_horizon,
        )

    def _build_rh_options(self) -> _native.RecedingHorizonOptions:
        grid: list[tuple[float, float]] | None = (
            [(float(cw), float(op)) for cw, op in self.weight_grid]
            if self.weight_grid is not None
            else None
        )
        return _native.RecedingHorizonOptions(
            k_candidates=self.k_candidates,
            rollout_horizon=self.rollout_horizon,
            commit_depth=self.commit_depth,
            tier0_next_h_weight=self.tier0_next_h_weight,
            weight_grid=grid,
            fallback_x_decrement=self.fallback_x_decrement,
            branch_parallel=self.branch_parallel,
            max_expansions_per_rollout=self.max_expansions_per_rollout,
            greedy_first=self.greedy_first,
            inner_beam_width=self.inner_beam_width,
        )

    def _invoke_solver(
        self,
        solver: MoveSolver,
        initial: dict[int, "_native.LocationAddress"],
        cz_pairs: list[tuple[int, int]],
        blocked: list["_native.LocationAddress"],
        future_cz_layers: list[list[tuple[int, int]]] | None,
    ) -> SolveResult:
        return solver.solve_entangling_rh(
            initial,
            cz_pairs,
            blocked,
            max_expansions=self.max_expansions,
            options=self._build_solve_options(),
            entangling_options=self._build_entangling_options(),
            rh_options=self._build_rh_options(),
            future_cz_layers=future_cz_layers,
        )
