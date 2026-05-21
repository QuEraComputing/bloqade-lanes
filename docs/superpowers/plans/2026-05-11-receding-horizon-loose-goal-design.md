# Receding-Horizon Loose-Goal Placement Search

**Date:** 2026-05-11
**Builds on:** `LooseGoalPlacementStrategy` (PR adding `solve_entangling`), the loose-goal infrastructure in `crates/bloqade-lanes-search/`

## Problem

The current `LooseGoalPlacementStrategy` collapses the feasible set of entangling configurations to a single point: it computes one Hungarian assignment of CZ pairs → entangling slots, writes that to `ctx.targets`, and the move generator's compass rigidly aims at that point until the constraint goal is reached.

The Hungarian assignment uses **sum of per-atom distances** as its cost. This cost is a poor proxy for the true cost (**number of move layers**) because:

- **Parallelism (Hungarian over-counts).** Many atom-hops can collapse into one bus layer when atoms share a bus, move in the same direction, and form a valid AOD rectangle (`aod_grid.rs:91`). Hungarian sees N independent distances; the hardware executes them in O(N / parallelism) layers.
- **Blockers (Hungarian under-counts).** An assignment that looks short by distance may require routing around occupied positions, adding move layers the cost matrix never accounted for.

So Hungarian's *best* assignment by distance is often not the best by move layers, particularly in the high-occupancy regime — exactly where the current strategy underperforms (PR_DESCRIPTION reports loose-goal at 1.23 lanes/layer vs baseline 1.49, listed as an open follow-up).

Restarts are not the answer: each restart runs an independent IDS from `s_0` with a perturbed Hungarian, picks the best end-to-end. There is no opportunity to *re-evaluate* mid-trajectory — once a restart commits to its assignment, it can only follow that compass to completion. The actual progress measured in move layers along the way never feeds back into target selection.

## Idea

Treat the Hungarian cost as a **suggestion, not a commitment**. At each stage of the compilation:

1. Generate **K diverse candidate assignments** (Hungarian under varying cost weights).
2. Run a short forward rollout for each candidate, using the candidate as the compass and the existing IDS search infrastructure, terminating at **rollout depth `x`**.
3. Evaluate each rollout's leaf state by re-running Hungarian — the leaf-state Hungarian cost is an estimate of remaining work.
4. **Commit only the first `m` move layers** (`m << x`) of the winning rollout's path.
5. Re-plan from the new state: generate K fresh candidates and repeat.

This is receding-horizon (MPC-style) control with the existing IDS search acting as the rollout simulator and Hungarian as the leaf evaluator. The horizon `x` controls how far ahead each candidate is tested; the commit depth `m` controls how aggressively we lock in moves before re-evaluation.

Three properties this gives us that pure restarts cannot:

- **Re-evaluation along the trajectory.** If a candidate's rollout reveals that its compass led to a state from which the remaining problem is easy (low leaf Hungarian), that candidate wins — even if its initial Hungarian cost was worse than others.
- **Parallelism-aware selection.** Branches that produce highly parallelizable move sequences naturally win because their actual `d_i` (rollout depth) is the *real* committed move layers, not an atom-hop sum.
- **Robustness to wrong initial assignment.** A bad first stage costs at most `m` layers before re-planning.

## Scope

In scope:

- New Rust orchestrator that drives the receding-horizon loop, calling existing `entangling.rs` Hungarian variants and existing `IdsFrontier` search.
- PyO3 bindings for the new options struct.
- New Python `RecedingHorizonLooseGoalPlacementStrategy` that composes the existing loose-goal options with horizon/commit/K parameters.

Out of scope:

- Adaptive K, x, m (fixed per-solve in v1; tunable via options).
- Murty's k-best Hungarian (weight-perturbation diversity should be enough for v1; revisit if benchmarks show structural-collapse).
- Online calibration of the tier-0 scoring weight α.
- Auto-escalation from `LooseGoalPlacementStrategy` to receding-horizon based on search difficulty (positioned as a separate strategy the user picks for high-density regimes).
- Per-branch parallelism inside one stage (the K rollouts within a stage run sequentially in v1; restart-level parallelism across cores is unchanged).

## Positioning

This strategy is **not a drop-in replacement** for `LooseGoalPlacementStrategy`. It is targeted at the high-occupancy regime where the current strategy's "1.23 lanes/layer vs baseline 1.49" gap is largest. In low-density regimes, the K-branch rollout structure is overhead — most rollouts will hit the constraint goal before depth `x`, collapsing the algorithm to "K parallel searches, pick the shortest," which restarts already do better and cheaper.

Documentation and the strategy's Python docstring should be explicit: **use this when atom density is high and the baseline loose-goal underperforms on move-layer count**.

## Algorithm

### Top level (one solve)

```
state ← initial_config
path ← []

while not state.satisfies(EntanglingConstraintGoal):
    candidates ← generate_K_assignments(state, K, weight_grid)
    branches ← []
    for A_i in candidates:
        rollout ← inner_search(state, compass=A_i, max_depth=x)
        if rollout.reached_goal:
            next_h ← hungarian_cost(rollout.leaf_state, next_cz_layer)
            branches.append(Tier0(d=rollout.depth, next_h=next_h, rollout=rollout))
        elif rollout.depth == x:
            c_prime ← hungarian_cost(rollout.leaf_state, current_cz_pairs)
            branches.append(Tier1(c_prime=c_prime, rollout=rollout))
        # else: drop (Tier 2, rollout failed to reach x)

    if branches is empty:
        if x > 1:
            retry stage with x ← x − 1
        else:
            fall back to LooseGoalPlacementStrategy from current state
            return

    best ← argmin over branches of score(branch)
    committed ← first m layers of best.rollout.path
    state ← apply(state, committed)
    path.extend(committed)

return path
```

### Scoring

Stratified — never add `d_i` (move layers) and `c_prime` (Hungarian cost) directly, except in the explicit α-weighted tier-0 case.

| Tier | Condition | Score |
|------|-----------|-------|
| 0    | Rollout reached `EntanglingConstraintGoal` at depth `d_i ≤ x` | `d_i + α · next_h` |
| 1    | Rollout completed exactly `x` layers without reaching goal | `c_prime` |
| 2    | Rollout reached depth `d_i < x` and could not extend | — (dropped) |

**Cross-tier ordering: Tier 0 always beats Tier 1.** A branch that reached the constraint goal is strictly preferable to one still searching.

**Tier-0 scoring rationale.** Among tier-0 branches, we want to pick the one with both (i) low committed cost (`d_i`) and (ii) a leaf state that is good for the *next* CZ layer (low `next_h`). These quantities have different units (layers vs. atom-hops), but for tier-0 there is no way to avoid combining them — they reflect different concerns. The α knob trades the two off:

- α = 0 → ignore next-layer setup, pure greedy on current-layer commitment.
- α large → next-layer setup dominates.

Default α = 0.5 (to be calibrated against benchmark sweeps). Exposed as a hyperparameter.

When the current CZ layer is the last in the circuit (no next layer), `next_h = 0` and tier-0 reduces to ranking by `d_i` alone.

**Tier-1 unit-mismatch sidestep.** All tier-1 branches have `d_i = x` by definition, so adding `d_i` to `c_prime` would not change the ranking. We drop the `d_i` term entirely for tier-1 to keep the comparison in a single unit.

**Tier-2 drop rationale.** A branch that could not advance `x` layers either deadlocked or hit an expansion budget early. Comparing such a branch numerically against tier-1 branches via `d_i + c_prime` would require a unit conversion. Discarding sidesteps the problem; the all-empty case is handled by the explicit fallback.

### Inner rollout

```
inner_search(state, compass, max_depth=x):
    targets ← compass (the Hungarian-assigned per-qubit targets)
    generator ← LooseTargetGenerator with cached_targets = targets
    frontier ← IdsFrontier with max_depth = x
    goal ← EntanglingConstraintGoal (same as outer; not strict target-match)
    return run_search(state, goal, generator, frontier, max_depth=x)
```

The inner search reuses the existing infrastructure with one tweak: `LooseTargetGenerator` must accept pre-computed targets rather than computing them from a seed (currently it only supports the latter; see `loose_target.rs:147–182`). This is a small refactor — add a constructor that takes pre-computed `Vec<(qid, target_loc)>` and bypasses the cached-on-first-call path.

The goal predicate is `EntanglingConstraintGoal` — same as the outer loop. A branch that lands on **any** valid entangling configuration during its rollout is accepted as tier-0. Forcing strict target-match would punish branches for finding a different (but equally valid) entangling configuration, which contradicts the loose-goal premise.

### Generating K candidate assignments

`generate_K_assignments(state, K, weight_grid)`:

1. For each `(congestion_weight, occupancy_penalty)` in `weight_grid`, call `assign_pairs_with_blockers` (the existing entry point in `entangling.rs:494+`) with those weights and any lookahead settings inherited from the outer options.
2. Hash-dedup each result: signature = `frozenset((qid, target_loc) for (qid, target_loc) in assignment)`. Drop duplicates.
3. If `len(unique) < K`, top up with additive cost-matrix noise using the existing seed-perturbation path (`entangling.rs:397-408`) until `len(unique) ≥ K` or a retry budget is exhausted.

Default `weight_grid`: 2D over `congestion_weight ∈ {0.0, 0.5, 1.0, 2.0, 5.0}` × `occupancy_penalty ∈ {0.5, 2.0}` = 10 combinations.

When the outer options enable lookahead Hungarian (`hungarian_horizon > 0`), each candidate uses `lookahead_assign_pairs` instead of `assign_pairs_with_blockers`, biasing candidates toward configurations that are good for the next CZ layer. This is orthogonal to (and complements) the tier-0 `next_h` term.

## Hyperparameters

| Name | Symbol | Default | Notes |
|------|--------|---------|-------|
| Candidates per stage | `K` | 10 | Number of distinct Hungarian assignments to try per stage. |
| Rollout horizon | `x` | 5 | How many move layers each branch's inner search runs. |
| Commit depth | `m` | 1 | How many of the winning branch's layers are committed before re-planning. Must satisfy `1 ≤ m ≤ x`. |
| Tier-0 next-layer weight | `α` | 0.5 | Trades current-layer commitment against next-layer setup quality. To be calibrated. |
| Weight grid | — | 2D `congestion × occupancy` (10 combos) | List of `(congestion_weight, occupancy_penalty)` tuples. |
| Fallback x decrement | — | 1 | When all branches drop (tier 2), retry with `x ← x − decrement`. Continues until `x = 1`, then falls back to standard loose-goal. |

## API

### Rust

```rust
// crates/bloqade-lanes-search/src/solve.rs (or new receding_horizon.rs)

pub struct RecedingHorizonOptions {
    pub k_candidates: usize,            // K, default 10
    pub rollout_horizon: u32,           // x, default 5
    pub commit_depth: u32,              // m, default 1
    pub tier0_next_h_weight: f64,       // α, default 0.5
    pub weight_grid: Vec<(f64, f64)>,   // (congestion, occupancy) pairs
    pub fallback_x_decrement: u32,      // default 1
}

impl MoveSolver {
    pub fn solve_entangling_rh(
        &self,
        initial: Config,
        cz_pairs: &[(u32, u32)],
        blocked: &HashSet<u64>,
        options: SolveOptions,
        entropy_options: EntropyOptions,
        entangling_options: EntanglingOptions,
        rh_options: RecedingHorizonOptions,
        future_layers: &[Vec<(u32, u32)>],  // for next_h and lookahead
        max_expansions: Option<u32>,
    ) -> SolveResult;
}
```

The new entry point reuses `solve_entangling`'s setup phase (`SearchContext`, `DistanceTable`, `WordPairDistances`, `EntanglingConstraintGoal`) — only the search loop differs.

### Python

```python
# python/bloqade/lanes/heuristics/physical/receding_horizon.py

class RecedingHorizonLooseGoalPlacementStrategy(LooseGoalPlacementStrategy):
    def __init__(
        self,
        *,
        k_candidates: int = 10,
        rollout_horizon: int = 5,
        commit_depth: int = 1,
        tier0_next_h_weight: float = 0.5,
        weight_grid: list[tuple[float, float]] | None = None,
        fallback_x_decrement: int = 1,
        **loose_goal_kwargs,
    ):
        super().__init__(**loose_goal_kwargs)
        # store RH-specific knobs
        ...

    def cz_placements(self, ...):
        # dispatch to solver.solve_entangling_rh instead of solve_entangling
        ...
```

Existing `LooseGoalPlacementStrategy` is unchanged. Users opt in by selecting the new class.

## Implementation Plan (high level)

1. **Refactor `LooseTargetGenerator` to accept pre-computed targets.** Currently it computes them lazily from a seed (`loose_target.rs:147–182`); add a constructor `from_targets(targets: Vec<(u32, u64)>)` that skips that path. (~30 LOC.)
2. **Add `RecedingHorizonOptions` struct** in `solve.rs` (~30 LOC).
3. **Implement `solve_entangling_rh`** — the outer loop, branch evaluation, tiering, commitment, fallback. Either in `solve.rs` or a new `receding_horizon.rs` module if `solve.rs` exceeds 2k LOC (currently 1876). (~300–400 LOC.)
4. **PyO3 binding `PyRecedingHorizonOptions`** in `crates/bloqade-lanes-bytecode-python/src/search_python.rs` (~80 LOC).
5. **Python strategy `RecedingHorizonLooseGoalPlacementStrategy`** in `python/bloqade/lanes/heuristics/physical/receding_horizon.py` (~120 LOC).
6. **Tests:**
   - Rust: tier-0/1/2 classification, commitment-step accounting, all-drop fallback, weight-grid dedup.
   - Python: 2–3 smoke tests mirroring `test_loose_goal_placement.py`.
7. **Benchmarks:** add to `scripts/bench_sweep.py` (untracked) — sweep over `K ∈ {5, 10, 20}`, `x ∈ {3, 5, 8}`, `m ∈ {1, 2, x}`, on Gemini physical 80q depths {3, 5, 8, 12}. Goal: confirm ≥5% move-layer reduction vs `LooseGoalPlacementStrategy(congestion_weight=1.0)` at depth ≥ 8 (high-density regime).

## Risks & Known Limitations

- **Greedy commitment uses leaf score only.** Two branches with the same `c_prime` could have very different path quality. Mitigation: A_i drives the path strongly enough that good-leaf ≈ good-path on average. Re-evaluate if benchmarks show pathological cases.
- **Diversity may collapse to one Hungarian basin.** Different weights can converge to the same assignment when the problem is "easy". Hash-dedup catches identical assignments; structurally-near-identical ones are not handled in v1. If benchmarks reveal frequent collapse, add forced word-pair-distribution diversity in v2.
- **Per-stage cost ~K× single search.** Sequential K rollouts × ~`circuit_layers / m` stages. Embarrassingly parallelizable inside a stage; v1 keeps it sequential to preserve restart-level parallelism. If cost is prohibitive, parallelize K branches inside a stage in v2.
- **Blocker resolution silently consumes `x`.** Under `MoveBlockers` deadlock policy, blocker resolution adds real move layers. A rollout of nominal depth 5 may be "3 productive + 2 blocker-resolution". Not a bug, but a calibration consideration when choosing `x`.
- **α (tier-0 next-h weight) has unit-mismatch sensitivity.** The user-facing default 0.5 is a guess; calibration against the benchmark sweep is required. If the optimal α varies widely across regimes, document that and consider exposing as a per-call argument.
- **Easy-regime overhead.** In low-occupancy layouts, most rollouts hit the constraint goal before depth `x` and the receding-horizon structure does not pay for itself. Documentation positions the strategy as a high-density tool to avoid confusion.

## Open questions (for v2)

- Adaptive `K`, `x`, `m` based on observed search difficulty.
- Online α calibration: observe the empirical layers-per-Hungarian-cost ratio along the trajectory and adjust α dynamically.
- Auto-escalation: `LooseGoalPlacementStrategy` detects "I'm struggling" (high expansion count, low h-score progress) and silently switches to receding-horizon for the remainder of the solve.
- Murty's k-best Hungarian as a third diversity layer below weight-perturbation and noise.
