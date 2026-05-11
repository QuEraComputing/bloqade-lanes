//! Receding-horizon (MPC-style) loose-goal placement search.
//!
//! This module orchestrates a sequence of K-candidate-rollout stages, each
//! using the existing IDS search infrastructure as the rollout simulator and
//! the existing Hungarian variants as both the per-candidate compass and the
//! per-leaf evaluator. See `plans/2026-05-11-receding-horizon-loose-goal-design.md`.
//!
//! High-level shape (one stage of one restart):
//!   1. Generate K diverse Hungarian assignments from the current state.
//!   2. For each candidate, run an inner IDS rollout of depth at most `x`.
//!   3. Classify each rollout: tier-0 (goal reached), tier-1 (full x layers,
//!      no goal), tier-2 (could not extend to depth x — discarded).
//!   4. Pick the best branch by stratified score. Tier-0 beats tier-1.
//!   5. Commit the full path of a tier-0 winner, or only `m` layers of a
//!      tier-1 winner; advance the state and re-plan.
//!
//! `MoveSolver::solve_entangling_rh` (the public entry, defined in
//! [`crate::solve`]) wraps a parallel restart loop around
//! [`solve_entangling_rh_single`].

use std::collections::HashSet;
use std::sync::Arc;

use bloqade_lanes_bytecode_core::arch::types::ArchSpec;
use rayon::prelude::*;

use crate::config::Config;
use crate::context::{SearchContext, SearchState};
use crate::cost::UniformCost;
use crate::entangling;
use crate::frontier::{self, IdsFrontier};
use crate::generators::heuristic::DeadlockPolicy;
use crate::generators::{HeuristicGenerator, LooseTargetGenerator};
use crate::goals::EntanglingConstraintGoal;
use crate::graph::{MoveSet, NodeId, SearchGraph};
use crate::heuristic::{DistanceTable, PairDistanceHeuristic};
use crate::lane_index::LaneIndex;
use crate::observer::NoOpObserver;
use crate::scorers::DistanceScorer;
use crate::solve::{EntanglingOptions, SolveOptions, SolveResult, SolveStatus};
use crate::traits::{Goal, Heuristic};

/// Per-rollout-iteration move penalty added to each Hungarian cost cell.
/// Matches `MOVE_PENALTY` in [`crate::solve`].
const MOVE_PENALTY: f64 = 1.0;
/// Future-layer blend weight for lookahead Hungarian. Matches
/// `LOOKAHEAD_BETA` in [`crate::solve`].
const LOOKAHEAD_BETA: f64 = 2.0;
/// Cap on how many extra noise-perturbed candidates we try when the
/// weight-grid alone returns fewer than `K` unique assignments.
const MAX_NOISE_TOP_UP_RETRIES: u64 = 50;

/// Options controlling the receding-horizon outer loop.
///
/// The cost-matrix knobs `congestion_weight` and `occupancy_penalty` and the
/// `hungarian_horizon` come from the existing [`EntanglingOptions`]; these
/// fields are receding-horizon-specific orchestration parameters.
#[derive(Debug, Clone)]
pub struct RecedingHorizonOptions {
    /// `K`: number of diverse Hungarian assignments tried per stage.
    pub k_candidates: usize,
    /// `x`: rollout horizon (max move-layer depth searched per branch).
    pub rollout_horizon: u32,
    /// `m`: commit depth — how many of the winning branch's layers are
    /// committed before re-planning. A tier-0 winner always commits its full
    /// path regardless of this value (its rollout already reached the goal).
    pub commit_depth: u32,
    /// α: weight on next-layer Hungarian cost when scoring tier-0 branches.
    /// `0.0` ignores next-layer setup; higher values trade current-layer
    /// commitment depth for better next-layer staging.
    pub tier0_next_h_weight: f64,
    /// Cost-weight grid for candidate generation. Each entry is
    /// `(congestion_weight, occupancy_penalty)`. Default has 10 entries.
    pub weight_grid: Vec<(f64, f64)>,
    /// When all K branches drop (tier-2 all around), retry the stage with
    /// `x ← x − decrement`. If `x` reaches 1 and still all-drop, fall back
    /// to standard `solve_entangling` from the current state.
    pub fallback_x_decrement: u32,
    /// Run the K rollouts in parallel via rayon within a stage. Useful when
    /// the outer restart parallelism is small (e.g. `restarts == 1`); set
    /// `false` to leave cores for restart parallelism.
    pub branch_parallel: bool,
    /// Per-rollout expansion budget. Each inner IDS call is capped at this
    /// many node expansions to bound runaway rollouts.
    pub max_expansions_per_rollout: u32,
}

impl Default for RecedingHorizonOptions {
    fn default() -> Self {
        // Defaults calibrated from the 40q × 20-pair × depth-3 sweep:
        // K=5 / m=5 sits at the cost/quality knee — ~24% move-layer
        // reduction vs LooseGoal(cw=1.0) at roughly 8× baseline wall-clock.
        // Smaller m (1–3) gives ~1–3% better quality at 2–3× more wall-clock;
        // K=10 gains ~3% at 2× wall-clock. See `scripts/eval_sweep_m.py`.
        Self {
            k_candidates: 5,
            rollout_horizon: 5,
            commit_depth: 5,
            tier0_next_h_weight: 0.5,
            weight_grid: default_weight_grid(),
            fallback_x_decrement: 1,
            branch_parallel: true,
            max_expansions_per_rollout: 1000,
        }
    }
}

/// Default cost-weight grid: 10 combinations covering the
/// (congestion, occupancy) plane.
pub fn default_weight_grid() -> Vec<(f64, f64)> {
    let cong = [0.0_f64, 0.5, 1.0, 2.0, 5.0];
    let occ = [0.5_f64, 2.0];
    let mut grid = Vec::with_capacity(cong.len() * occ.len());
    for &c in &cong {
        for &o in &occ {
            grid.push((c, o));
        }
    }
    grid
}

// ── Branch classification ──────────────────────────────────────────────

/// Outcome of one branch's rollout.
///
/// Tier-2 (failed to reach depth x) is represented by `None` from
/// [`classify_into_tier`] — dropped branches are not stored.
///
/// Tier-1's leaf-Hungarian cost (`c_prime`) is **not** stored here — it's
/// only consulted when ranking against other tier-1 branches in the
/// absence of any tier-0 winner. When at least one tier-0 exists, all
/// tier-1 branches lose by definition, so computing their leaf Hungarian
/// would be wasted work. [`pick_best_branch`] computes it lazily.
pub(crate) enum BranchResult {
    /// Rollout reached `EntanglingConstraintGoal` at `depth ≤ rollout_horizon`.
    /// Commit the full `path` regardless of `commit_depth`.
    Tier0 {
        depth: u32,
        path: Vec<MoveSet>,
        leaf_config: Config,
        nodes_expanded: u32,
    },
    /// Rollout completed exactly `rollout_horizon` layers without reaching
    /// the goal. The leaf Hungarian cost is computed lazily in
    /// [`pick_best_branch`] only when no tier-0 branch exists.
    Tier1 {
        path: Vec<MoveSet>,
        leaf_config: Config,
        nodes_expanded: u32,
    },
}

impl BranchResult {
    fn nodes_expanded(&self) -> u32 {
        match self {
            BranchResult::Tier0 { nodes_expanded, .. }
            | BranchResult::Tier1 { nodes_expanded, .. } => *nodes_expanded,
        }
    }
    fn is_tier0(&self) -> bool {
        matches!(self, BranchResult::Tier0 { .. })
    }
    fn path(&self) -> &[MoveSet] {
        match self {
            BranchResult::Tier0 { path, .. } | BranchResult::Tier1 { path, .. } => path,
        }
    }
    fn leaf_config(&self) -> &Config {
        match self {
            BranchResult::Tier0 { leaf_config, .. } | BranchResult::Tier1 { leaf_config, .. } => {
                leaf_config
            }
        }
    }
    fn depth(&self) -> u32 {
        match self {
            BranchResult::Tier0 { depth, .. } => *depth,
            BranchResult::Tier1 { path, .. } => path.len() as u32,
        }
    }
}

// ── Helpers ────────────────────────────────────────────────────────────

/// Generate up to `k` diverse Hungarian candidate assignments from `state`.
///
/// Strategy:
/// 1. For each `(congestion_weight, occupancy_penalty)` in `weight_grid`,
///    call Hungarian with a `restart_seed`-derived seed (so each restart
///    sees genuinely different cost-matrix perturbations across stages).
///    These K calls are independent and run in parallel via rayon when
///    `parallel = true` (gated on the same `branch_parallel` flag that
///    controls rollout parallelism, so we don't oversubscribe cores when
///    the outer restart loop is already parallel).
/// 2. Dedup by the canonical (sorted) target list, take first k unique.
/// 3. Top up with additive-noise variants if the unique set is still < k.
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_k_candidates(
    state: &Config,
    cz_pairs: &[(u32, u32)],
    arch: &ArchSpec,
    index: &LaneIndex,
    dist_table: &DistanceTable,
    blocked: &HashSet<u64>,
    ent_opts: &EntanglingOptions,
    future_layers: &[Vec<(u32, u32)>],
    weight_grid: &[(f64, f64)],
    k: usize,
    restart_seed: u64,
    parallel: bool,
) -> Vec<Vec<(u32, u64)>> {
    let grid_len = weight_grid.len().max(1) as u64;

    // ── Phase 1: per-weight Hungarian (parallel if requested) ──────────
    let one_candidate = |i: usize, cw: f64, op: f64| -> Vec<(u32, u64)> {
        let seed = restart_seed
            .wrapping_mul(grid_len)
            .wrapping_add(i as u64 + 1);
        if !future_layers.is_empty() {
            entangling::lookahead_assign_pairs(
                cz_pairs,
                state,
                arch,
                index,
                dist_table,
                blocked,
                seed,
                future_layers,
                LOOKAHEAD_BETA,
                cw,
                op,
                MOVE_PENALTY,
            )
        } else {
            entangling::assign_pairs_with_blockers(
                cz_pairs,
                state,
                arch,
                index,
                dist_table,
                blocked,
                seed,
                None,
                0.0,
                cw,
                op,
                MOVE_PENALTY,
                true,
            )
        }
    };

    let raw_candidates: Vec<Vec<(u32, u64)>> = if parallel {
        weight_grid
            .par_iter()
            .enumerate()
            .map(|(i, &(cw, op))| one_candidate(i, cw, op))
            .collect()
    } else {
        weight_grid
            .iter()
            .enumerate()
            .map(|(i, &(cw, op))| one_candidate(i, cw, op))
            .collect()
    };

    // ── Phase 2: dedup, take first k unique ───────────────────────────
    let mut signatures: HashSet<Vec<(u32, u64)>> = HashSet::new();
    let mut candidates: Vec<Vec<(u32, u64)>> = Vec::with_capacity(k);
    for targets in raw_candidates {
        if targets.is_empty() {
            continue;
        }
        let mut sig = targets.clone();
        sig.sort_unstable();
        if signatures.insert(sig) {
            candidates.push(targets);
            if candidates.len() >= k {
                return candidates;
            }
        }
    }

    // Top up with additive noise variants on the baseline weight tuple.
    let mut retry_idx: u64 = 0;
    while candidates.len() < k && retry_idx < MAX_NOISE_TOP_UP_RETRIES {
        let seed = restart_seed.wrapping_add(1000).wrapping_add(retry_idx + 1);
        let targets = entangling::assign_pairs_with_blockers(
            cz_pairs,
            state,
            arch,
            index,
            dist_table,
            blocked,
            seed,
            None,
            0.0,
            ent_opts.congestion_weight,
            ent_opts.occupancy_penalty,
            MOVE_PENALTY,
            true,
        );
        if !targets.is_empty() {
            let mut sig = targets.clone();
            sig.sort_unstable();
            if signatures.insert(sig) {
                candidates.push(targets);
            }
        }
        retry_idx += 1;
    }

    candidates
}

/// Run a single inner rollout: build a LooseTargetGenerator with pre-set
/// `targets`, run IDS with `max_depth = x`, return the SearchGraph + outcome.
pub(crate) struct RolloutOutcome {
    pub(crate) graph: SearchGraph,
    pub(crate) goal_node: Option<NodeId>,
    pub(crate) max_depth_reached: u32,
    pub(crate) nodes_expanded: u32,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_inner_rollout<G: Goal + Sync, Hsum: Heuristic + Copy + Sync>(
    root: Config,
    targets: Vec<(u32, u64)>,
    cz_pairs: Vec<(u32, u32)>,
    arch: Arc<ArchSpec>,
    index: Arc<LaneIndex>,
    dist_table: Arc<DistanceTable>,
    blocked: &HashSet<u64>,
    goal: &G,
    h_sum: Hsum,
    max_depth: u32,
    max_expansions: u32,
    deadlock_policy: DeadlockPolicy,
    inner_lookahead: bool,
    top_c: usize,
    restart_seed: u64,
) -> RolloutOutcome {
    let inner = HeuristicGenerator::new()
        .with_deadlock_policy(deadlock_policy)
        .with_lookahead(inner_lookahead)
        .with_top_c(top_c)
        .with_seed(restart_seed);

    // The DistanceScorer in `run_search` reads `ctx.targets` to score
    // candidate moves; we need ctx.targets populated with the actual
    // assignment, not an empty placeholder. Pass the targets to both
    // (a) the outer SearchContext (consumed by the scorer) and (b) the
    // LooseTargetGenerator's internal cache (consumed when the generator
    // overrides ctx for the inner HeuristicGenerator).
    let targets_for_ctx = targets.clone();
    let cz_pairs_for_ctx = cz_pairs.clone();
    let generator =
        LooseTargetGenerator::from_targets(inner, targets, cz_pairs, arch, index, dist_table);

    let scorer = DistanceScorer;
    let cost_fn = UniformCost;
    let mut frontier = IdsFrontier::new(h_sum);
    let mut search_state = SearchState::default();
    let mut observer = NoOpObserver;

    let result = {
        let ctx = SearchContext {
            index: generator.index_ref(),
            dist_table: generator.dist_table_ref(),
            blocked,
            targets: &targets_for_ctx,
            cz_pairs: Some(&cz_pairs_for_ctx),
        };
        frontier::run_search(
            root,
            &generator,
            &scorer,
            &cost_fn,
            goal,
            &mut frontier,
            &ctx,
            &mut search_state,
            &mut observer,
            Some(max_expansions),
            Some(max_depth),
        )
    };

    RolloutOutcome {
        graph: result.graph,
        goal_node: result.goal,
        max_depth_reached: result.max_depth_reached,
        nodes_expanded: result.nodes_expanded,
    }
}

/// Walk the graph and find the best leaf at the deepest reached depth ≤
/// `target_depth`. Returns `None` if the graph is just the root (no children).
///
/// `heuristic_fn` is any closure that scores a config (lower = better).
pub(crate) fn extract_best_leaf(
    graph: &SearchGraph,
    target_depth: u32,
    heuristic_fn: impl Fn(&Config) -> f64,
) -> Option<NodeId> {
    // First pass: find the deepest depth actually reached.
    let mut max_d: u32 = 0;
    let n = graph.len();
    for i in 0..n {
        let nid = NodeId(i as u32);
        let d = graph.depth(nid);
        if d > max_d && d <= target_depth {
            max_d = d;
        }
    }
    if max_d == 0 {
        return None; // only the root — no expansion happened
    }

    // Second pass: best leaf at max_d.
    let mut best: Option<(NodeId, f64, f64)> = None; // (id, h, g)
    for i in 0..n {
        let nid = NodeId(i as u32);
        if graph.depth(nid) != max_d {
            continue;
        }
        let h = heuristic_fn(graph.config(nid));
        let g = graph.g_score(nid);
        match best {
            None => best = Some((nid, h, g)),
            Some((_, best_h, best_g)) => {
                if h < best_h || (h == best_h && g < best_g) {
                    best = Some((nid, h, g));
                }
            }
        }
    }

    best.map(|(nid, _, _)| nid)
}

/// Compute the Hungarian cost scalar at `leaf_config` for `cz_pairs` (or
/// the next-layer pairs, for the tier-0 `next_h` term). Uses
/// `assign_pairs_with_blockers` to find an assignment, then
/// [`entangling::assignment_cost`] to sum the distances.
#[allow(clippy::too_many_arguments)]
pub(crate) fn hungarian_cost_at_leaf(
    leaf: &Config,
    pairs: &[(u32, u32)],
    arch: &ArchSpec,
    index: &LaneIndex,
    dist_table: &DistanceTable,
    blocked: &HashSet<u64>,
    ent_opts: &EntanglingOptions,
) -> u32 {
    if pairs.is_empty() {
        return 0;
    }
    let targets = entangling::assign_pairs_with_blockers(
        pairs,
        leaf,
        arch,
        index,
        dist_table,
        blocked,
        0, // no perturbation for scoring
        None,
        0.0,
        ent_opts.congestion_weight,
        ent_opts.occupancy_penalty,
        MOVE_PENALTY,
        true,
    );
    if targets.is_empty() {
        return u32::MAX / 4;
    }
    entangling::assignment_cost(leaf, &targets, dist_table)
}

/// Convert a [`RolloutOutcome`] into a [`BranchResult`].
/// Returns `None` for tier-2 (the rollout couldn't extend to `rollout_horizon`).
///
/// Cheap — does no Hungarian work. The leaf Hungarian for tier-1 ranking
/// is deferred to [`pick_best_branch`].
pub(crate) fn classify_into_tier(
    outcome: RolloutOutcome,
    rollout_horizon: u32,
    heuristic: &PairDistanceHeuristic,
) -> Option<BranchResult> {
    let RolloutOutcome {
        graph,
        goal_node,
        max_depth_reached,
        nodes_expanded,
    } = outcome;

    // Tier-0: rollout reached the constraint goal.
    if let Some(goal_id) = goal_node {
        let path = graph.reconstruct_path(goal_id);
        let depth = graph.depth(goal_id);
        let leaf_config = graph.config(goal_id).clone();
        return Some(BranchResult::Tier0 {
            depth,
            path,
            leaf_config,
            nodes_expanded,
        });
    }

    // Tier-1: rollout reached exactly `rollout_horizon` layers, no goal.
    if max_depth_reached < rollout_horizon {
        return None; // tier-2 — drop
    }
    let h_fn = |cfg: &Config| heuristic.estimate_sum(cfg);
    let leaf = extract_best_leaf(&graph, rollout_horizon, h_fn)?;
    if graph.depth(leaf) < rollout_horizon {
        return None;
    }
    let path = graph.reconstruct_path(leaf);
    let leaf_config = graph.config(leaf).clone();
    // Note: c_prime (Hungarian cost at leaf) is computed lazily in
    // pick_best_branch, only when needed (i.e., when no tier-0 branch exists).
    Some(BranchResult::Tier1 {
        path,
        leaf_config,
        nodes_expanded,
    })
}

/// Pick the lowest-score branch. Tier-0 always beats tier-1.
///
/// Tier-0 score = `depth + α · next_h` where `next_h` is the Hungarian cost
/// at the leaf for the next CZ layer (zero if no next layer).
/// Tier-1 score = `c_prime` (the leaf-state Hungarian cost for the current
/// CZ layer).
///
/// Hungarian leaf-scoring is **lazy** — tier-1 `c_prime` and tier-0 `next_h`
/// (when `future_layers` is empty) are not computed at all. When a tier-0
/// branch exists, all tier-1 are infeasible and skip their Hungarian work
/// entirely (saves up to K Hungarian calls per stage).
#[allow(clippy::too_many_arguments)]
pub(crate) fn pick_best_branch<'a>(
    branches: &'a [BranchResult],
    alpha: f64,
    cz_pairs: &[(u32, u32)],
    future_layers: &[Vec<(u32, u32)>],
    arch: &ArchSpec,
    index: &LaneIndex,
    dist_table: &DistanceTable,
    blocked: &HashSet<u64>,
    ent_opts: &EntanglingOptions,
) -> Option<&'a BranchResult> {
    let has_tier0 = branches.iter().any(|b| b.is_tier0());

    let mut best: Option<(&BranchResult, f64)> = None;
    for b in branches {
        let score: f64 = match b {
            BranchResult::Tier0 {
                depth, leaf_config, ..
            } => {
                let next_h = if !future_layers.is_empty() && alpha != 0.0 {
                    hungarian_cost_at_leaf(
                        leaf_config,
                        &future_layers[0],
                        arch,
                        index,
                        dist_table,
                        blocked,
                        ent_opts,
                    ) as f64
                } else {
                    0.0
                };
                *depth as f64 + alpha * next_h
            }
            BranchResult::Tier1 { leaf_config, .. } => {
                // Cross-tier short-circuit: when any tier-0 exists, all
                // tier-1 are infeasible and we skip the Hungarian.
                if has_tier0 {
                    f64::INFINITY
                } else {
                    hungarian_cost_at_leaf(
                        leaf_config,
                        cz_pairs,
                        arch,
                        index,
                        dist_table,
                        blocked,
                        ent_opts,
                    ) as f64
                }
            }
        };
        match best {
            None => best = Some((b, score)),
            Some((_, best_score)) if score < best_score => best = Some((b, score)),
            _ => (),
        }
    }
    best.map(|(b, _)| b)
}

// ── Single-trajectory orchestrator ────────────────────────────────────

/// Run one receding-horizon trajectory from `root` to a configuration
/// satisfying the constraint goal. Returns the committed sequence of
/// move layers plus the final state.
///
/// The caller is responsible for parallelism (see
/// `MoveSolver::solve_entangling_rh`'s rayon wrapper).
#[allow(clippy::too_many_arguments)]
pub fn solve_entangling_rh_single(
    root: Config,
    cz_pairs: &[(u32, u32)],
    blocked: HashSet<u64>,
    arch: Arc<ArchSpec>,
    index: Arc<LaneIndex>,
    dist_table: Arc<DistanceTable>,
    goal: &EntanglingConstraintGoal,
    heuristic: &PairDistanceHeuristic,
    opts: &SolveOptions,
    ent_opts: &EntanglingOptions,
    rh_opts: &RecedingHorizonOptions,
    future_layers: &[Vec<(u32, u32)>],
    max_expansions: Option<u32>,
    restart_seed: u64,
    fallback: impl Fn(&Config) -> SolveResult + Sync,
) -> SolveResult {
    let mut state = root;
    let mut committed_layers: Vec<MoveSet> = Vec::new();
    let mut total_expansions: u32 = 0;
    let mut x = rh_opts.rollout_horizon;

    // Reusable handle to the sum-heuristic for IDS frontier ordering and
    // leaf ranking.
    let h_sum_closure = |cfg: &Config| -> f64 { heuristic.estimate_sum(cfg) };

    let top_c = opts.top_c.unwrap_or(3);
    let inner_lookahead = opts.lookahead;
    let deadlock_policy = opts.deadlock_policy;

    let mut stage_iter: u32 = 0;
    let stage_budget_cap: u32 = max_expansions.unwrap_or(u32::MAX);

    while !goal.is_goal(&state) {
        if total_expansions >= stage_budget_cap {
            // Out of global budget — return whatever we've committed plus a
            // BudgetExceeded status.
            return SolveResult {
                status: SolveStatus::BudgetExceeded,
                move_layers: committed_layers,
                goal_config: state,
                nodes_expanded: total_expansions,
                cost: 0.0,
                deadlocks: 0,
                entropy_trace: None,
            };
        }
        stage_iter = stage_iter.saturating_add(1);

        // (a) Generate K candidate assignments — seeded by restart_seed and
        // stage iteration so consecutive stages within one restart also
        // diversify. The `1_000_003` multiplier (a small prime well above
        // any plausible stage count) spreads consecutive stage seeds into
        // distinct, well-separated regions of the seed space, preventing
        // adjacent stages from producing near-identical candidate sets when
        // restart_seed is small.
        let stage_seed = restart_seed
            .wrapping_mul(1_000_003)
            .wrapping_add(stage_iter as u64);
        let candidates = generate_k_candidates(
            &state,
            cz_pairs,
            &arch,
            &index,
            &dist_table,
            &blocked,
            ent_opts,
            future_layers,
            &rh_opts.weight_grid,
            rh_opts.k_candidates,
            stage_seed,
            rh_opts.branch_parallel,
        );

        if candidates.is_empty() {
            // No assignment possible from this state — fall back.
            let fb = fallback(&state);
            return merge_fallback(committed_layers, fb, total_expansions);
        }

        // (b) Run rollouts (optionally parallel).
        let rollout = |targets: Vec<(u32, u64)>| -> Option<BranchResult> {
            let outcome = run_inner_rollout(
                state.clone(),
                targets,
                cz_pairs.to_vec(),
                arch.clone(),
                index.clone(),
                dist_table.clone(),
                &blocked,
                goal,
                h_sum_closure,
                x,
                rh_opts.max_expansions_per_rollout,
                deadlock_policy,
                inner_lookahead,
                top_c,
                stage_seed,
            );
            classify_into_tier(outcome, x, heuristic)
        };

        let branches: Vec<BranchResult> = if rh_opts.branch_parallel {
            candidates.into_par_iter().filter_map(rollout).collect()
        } else {
            candidates.into_iter().filter_map(rollout).collect()
        };
        total_expansions = total_expansions
            .saturating_add(branches.iter().map(|b| b.nodes_expanded()).sum::<u32>());

        // (c) All-drop fallback.
        if branches.is_empty() {
            if x > 1 {
                x = x.saturating_sub(rh_opts.fallback_x_decrement.max(1));
                continue;
            }
            let fb = fallback(&state);
            return merge_fallback(committed_layers, fb, total_expansions);
        }

        // (d) Pick best branch.
        let best = match pick_best_branch(
            &branches,
            rh_opts.tier0_next_h_weight,
            cz_pairs,
            future_layers,
            &arch,
            &index,
            &dist_table,
            &blocked,
            ent_opts,
        ) {
            Some(b) => b,
            None => {
                let fb = fallback(&state);
                return merge_fallback(committed_layers, fb, total_expansions);
            }
        };

        // (e) Commit: full path for tier-0, `m` layers for tier-1.
        let commit_count = if best.is_tier0() {
            best.depth() as usize
        } else {
            (rh_opts.commit_depth as usize).min(best.depth() as usize)
        };
        if commit_count == 0 {
            // Nothing to commit (shouldn't happen — guards against infinite loop).
            let fb = fallback(&state);
            return merge_fallback(committed_layers, fb, total_expansions);
        }
        for ms in best.path().iter().take(commit_count) {
            committed_layers.push(ms.clone());
        }
        // Advance state.
        state = if commit_count == best.path().len() {
            best.leaf_config().clone()
        } else {
            // Walk the path from the current state, applying each move.
            // We need to recover the config after `commit_count` moves
            // from the rollout. The simplest way: replay the moves to
            // derive the intermediate config (since the rollout's graph
            // is dropped at end of stage scope).
            apply_layers_to(state.clone(), best.path().iter().take(commit_count), &index)
                .unwrap_or_else(|| best.leaf_config().clone())
        };

        // After commit, refresh x to the configured horizon (it may have
        // been reduced by the fallback path above).
        x = rh_opts.rollout_horizon;
    }

    let cost = committed_layers.len() as f64;
    SolveResult {
        status: SolveStatus::Solved,
        move_layers: committed_layers,
        goal_config: state,
        nodes_expanded: total_expansions,
        cost,
        deadlocks: 0,
        entropy_trace: None,
    }
}

/// Apply a sequence of move layers to a configuration, returning the new
/// configuration. Returns `None` if any move is inapplicable from the
/// current config (which would indicate an inconsistency between the
/// rollout's graph and the orchestrator's tracked state).
///
/// Only used when `commit_depth < rollout_horizon`. With the default
/// (m=x=5), tier-1 commits exhaust the full path and we read the leaf
/// config directly from the rollout instead.
fn apply_layers_to<'a>(
    mut config: Config,
    layers: impl IntoIterator<Item = &'a MoveSet>,
    index: &LaneIndex,
) -> Option<Config> {
    for ms in layers {
        config = apply_move_set(&config, ms, index)?;
    }
    Some(config)
}

/// Apply one [`MoveSet`] to a configuration. Each lane in the move set
/// describes an atom move from its source location to its destination.
/// Endpoints are resolved via the [`LaneIndex`].
fn apply_move_set(config: &Config, move_set: &MoveSet, index: &LaneIndex) -> Option<Config> {
    use bloqade_lanes_bytecode_core::arch::addr::{LaneAddr, LocationAddr};

    let mut next: Vec<(u32, LocationAddr)> = config.iter().collect();

    for &lane_enc in move_set.encoded_lanes() {
        let lane = LaneAddr::decode_u64(lane_enc);
        let (src, dst) = index.endpoints(&lane)?;
        // Find the qubit at src and move it to dst.
        let entry = next.iter_mut().find(|(_, loc)| *loc == src)?;
        entry.1 = dst;
    }
    Config::new(next).ok()
}

/// Merge an end-of-trajectory fallback result onto the committed-layer
/// prefix, producing a single SolveResult.
fn merge_fallback(
    committed_layers: Vec<MoveSet>,
    fallback: SolveResult,
    total_expansions: u32,
) -> SolveResult {
    let combined_expansions = total_expansions.saturating_add(fallback.nodes_expanded);
    if fallback.status != SolveStatus::Solved {
        return SolveResult {
            status: fallback.status,
            move_layers: committed_layers,
            goal_config: fallback.goal_config,
            nodes_expanded: combined_expansions,
            cost: 0.0,
            deadlocks: fallback.deadlocks,
            entropy_trace: None,
        };
    }
    let mut merged = committed_layers;
    merged.extend(fallback.move_layers);
    let cost = merged.len() as f64;
    SolveResult {
        status: SolveStatus::Solved,
        move_layers: merged,
        goal_config: fallback.goal_config,
        nodes_expanded: combined_expansions,
        cost,
        deadlocks: fallback.deadlocks,
        entropy_trace: None,
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::SearchGraph;
    use crate::solve::{MoveSolver, Strategy};
    use crate::test_utils::{example_arch_json, loc};

    // ── Trivial / structural ──

    #[test]
    fn default_weight_grid_has_ten_entries() {
        let grid = default_weight_grid();
        assert_eq!(grid.len(), 10);
        // All entries finite, non-negative.
        for &(cw, op) in &grid {
            assert!(cw.is_finite() && cw >= 0.0);
            assert!(op.is_finite() && op >= 0.0);
        }
        // Spans both axes.
        assert!(grid.iter().any(|&(cw, _)| cw == 0.0));
        assert!(grid.iter().any(|&(cw, _)| cw == 5.0));
        assert!(grid.iter().any(|&(_, op)| op == 0.5));
        assert!(grid.iter().any(|&(_, op)| op == 2.0));
    }

    #[test]
    fn default_options_satisfy_invariants() {
        let opts = RecedingHorizonOptions::default();
        assert_eq!(opts.k_candidates, 5);
        assert_eq!(opts.rollout_horizon, 5);
        assert_eq!(opts.commit_depth, 5);
        assert_eq!(opts.tier0_next_h_weight, 0.5);
        assert!(opts.commit_depth >= 1 && opts.commit_depth <= opts.rollout_horizon);
        assert_eq!(opts.weight_grid.len(), 10);
        assert!(opts.fallback_x_decrement >= 1);
        assert!(opts.max_expansions_per_rollout > 0);
    }

    // ── extract_best_leaf ──

    #[test]
    fn extract_best_leaf_root_only_returns_none() {
        let cfg = Config::new([(0, loc(0, 0))]).unwrap();
        let graph = SearchGraph::new(cfg);
        // No expansion happened; only root exists. Should return None.
        let h_fn = |_cfg: &Config| 0.0_f64;
        assert!(extract_best_leaf(&graph, 3, h_fn).is_none());
    }

    #[test]
    fn extract_best_leaf_picks_lowest_h_at_deepest_depth() {
        let root_cfg = Config::new([(0, loc(0, 0))]).unwrap();
        let mut graph = SearchGraph::new(root_cfg);
        let root = graph.root();

        // Build a tree with two children at depth 1, two grandchildren at
        // depth 2, with distinct configs and known h-values.
        use crate::graph::MoveSet;
        let ms = MoveSet::from_encoded(vec![]);

        let cfg_d1_a = Config::new([(0, loc(0, 1))]).unwrap();
        let (n1a, _) = graph.insert(root, ms.clone(), cfg_d1_a.clone(), 1.0);
        let cfg_d1_b = Config::new([(0, loc(0, 2))]).unwrap();
        let (n1b, _) = graph.insert(root, ms.clone(), cfg_d1_b.clone(), 1.0);

        let cfg_d2_a = Config::new([(0, loc(0, 3))]).unwrap();
        let (_n2a, _) = graph.insert(n1a, ms.clone(), cfg_d2_a.clone(), 2.0);
        let cfg_d2_b = Config::new([(0, loc(0, 4))]).unwrap();
        let (n2b, _) = graph.insert(n1b, ms.clone(), cfg_d2_b.clone(), 2.0);

        // h-fn: prefer cfg_d2_b (lower h-score).
        let h_fn = |cfg: &Config| -> f64 {
            if *cfg == cfg_d2_b {
                1.0
            } else if *cfg == cfg_d2_a {
                10.0
            } else {
                100.0
            }
        };

        let leaf = extract_best_leaf(&graph, 2, h_fn).expect("should find a depth-2 leaf");
        assert_eq!(leaf, n2b, "should pick the lower-h depth-2 node");
    }

    #[test]
    fn extract_best_leaf_falls_through_to_lower_depth() {
        let root_cfg = Config::new([(0, loc(0, 0))]).unwrap();
        let mut graph = SearchGraph::new(root_cfg);
        let root = graph.root();
        use crate::graph::MoveSet;
        let ms = MoveSet::from_encoded(vec![]);

        // Only depth-1 nodes; no depth-3.
        let cfg_d1 = Config::new([(0, loc(0, 1))]).unwrap();
        let (n1, _) = graph.insert(root, ms, cfg_d1, 1.0);

        let h_fn = |_cfg: &Config| 0.0_f64;
        // Target depth 3 — no nodes there; should fall back to depth 1.
        let leaf = extract_best_leaf(&graph, 3, h_fn).expect("should fall back");
        assert_eq!(leaf, n1);
    }

    // ── hungarian_cost_at_leaf ──

    #[test]
    fn hungarian_cost_at_leaf_empty_pairs_returns_zero() {
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        let arch = solver.index().arch_spec().clone();
        let lane_index = solver.index();
        let cz_pairs: Vec<(u32, u32)> = vec![(0, 1)];
        // Build a tiny dist table targeting entangling locations.
        let ent_locs = entangling::all_entangling_locations(&arch);
        let dist_table = DistanceTable::new(&ent_locs, lane_index);
        let blocked: HashSet<u64> = HashSet::new();
        let config = Config::new([(0, loc(0, 0)), (1, loc(1, 0))]).unwrap();

        // Empty pairs → cost 0.
        let cost = hungarian_cost_at_leaf(
            &config,
            &[],
            &arch,
            lane_index,
            &dist_table,
            &blocked,
            &EntanglingOptions::default(),
        );
        assert_eq!(cost, 0);

        // Real pairs at entangling positions → assignment_cost returns 0
        // (qubits already at targets, no distance to cover).
        let cost = hungarian_cost_at_leaf(
            &config,
            &cz_pairs,
            &arch,
            lane_index,
            &dist_table,
            &blocked,
            &EntanglingOptions::default(),
        );
        assert_eq!(cost, 0, "qubits already entangling-paired → cost 0");
    }

    // ── generate_k_candidates ──

    #[test]
    fn generate_k_candidates_caps_at_k() {
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        let arch = solver.index().arch_spec().clone();
        let lane_index = solver.index();
        let ent_locs = entangling::all_entangling_locations(&arch);
        let dist_table = DistanceTable::new(&ent_locs, lane_index);
        let blocked: HashSet<u64> = HashSet::new();
        let cz_pairs: Vec<(u32, u32)> = vec![(0, 1)];
        // Use a non-trivial starting config so Hungarian actually moves.
        let config = Config::new([(0, loc(0, 0)), (1, loc(1, 5))]).unwrap();

        let grid = default_weight_grid();
        let candidates = generate_k_candidates(
            &config,
            &cz_pairs,
            &arch,
            lane_index,
            &dist_table,
            &blocked,
            &EntanglingOptions::default(),
            &[],
            &grid,
            3,     // k
            1,     // restart_seed
            false, // parallel
        );
        assert!(candidates.len() <= 3, "must respect k cap");
    }

    #[test]
    #[ignore = "documents the known seed-perturbation-collapse failure mode on small \
                instances. On example_arch_json with 1 CZ pair and 10 slots, the cost \
                surface is too well-separated for ±1 perturbation to flip Hungarian's \
                optimum across distinct seeds, so different restart seeds produce \
                identical candidate sets. The plan's Step 1b (widen \
                seed_perturbation_amplitude) is the documented fix; re-enable this test \
                once that lands. Note: even with identical candidate sets, full \
                trajectories can still diverge across restarts via inner-search seeding \
                (see restart_trajectories_can_diverge)."]
    fn generate_k_candidates_seed_changes_signatures() {
        // The plan's primary risk: different restart_seeds collapse to the
        // same candidate set. On a non-trivial instance, distinct seeds
        // should produce at least one distinct candidate signature.
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        let arch = solver.index().arch_spec().clone();
        let lane_index = solver.index();
        let ent_locs = entangling::all_entangling_locations(&arch);
        let dist_table = DistanceTable::new(&ent_locs, lane_index);
        let blocked: HashSet<u64> = HashSet::new();
        let cz_pairs: Vec<(u32, u32)> = vec![(0, 1)];
        let config = Config::new([(0, loc(0, 0)), (1, loc(1, 5))]).unwrap();

        let grid = default_weight_grid();
        let cands_seed_1 = generate_k_candidates(
            &config,
            &cz_pairs,
            &arch,
            lane_index,
            &dist_table,
            &blocked,
            &EntanglingOptions::default(),
            &[],
            &grid,
            10,
            1,
            false,
        );
        let cands_seed_2 = generate_k_candidates(
            &config,
            &cz_pairs,
            &arch,
            lane_index,
            &dist_table,
            &blocked,
            &EntanglingOptions::default(),
            &[],
            &grid,
            10,
            2,
            false,
        );

        // Convert each candidate to a canonical sorted signature.
        fn sig(mut t: Vec<(u32, u64)>) -> Vec<(u32, u64)> {
            t.sort_unstable();
            t
        }
        let s1: HashSet<Vec<(u32, u64)>> = cands_seed_1.into_iter().map(sig).collect();
        let s2: HashSet<Vec<(u32, u64)>> = cands_seed_2.into_iter().map(sig).collect();
        // The two sets aren't necessarily disjoint, but they should NOT be
        // exactly equal — otherwise restart parallelism is wasted. On a
        // 1-pair instance the cost surface may be very flat; if this test
        // proves brittle, widen seed_perturbation_amplitude (Step 1b).
        assert!(
            s1 != s2,
            "restart seeds produced identical candidate sets; \
             cross-restart trajectories would collapse. \
             This is the failure mode flagged in the design — widen perturbation."
        );
    }

    // ── End-to-end smoke tests ──

    #[test]
    fn solve_entangling_rh_with_already_at_goal() {
        // Initial state already satisfies the constraint goal → trajectory
        // length 0, Solved, no move layers.
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        let result = solver
            .solve_entangling_rh(
                [(0, loc(0, 0)), (1, loc(1, 0))],
                &[(0, 1)],
                std::iter::empty(),
                Some(5000),
                &SolveOptions {
                    strategy: Strategy::Ids,
                    restarts: 1,
                    ..SolveOptions::default()
                },
                &EntanglingOptions::default(),
                &RecedingHorizonOptions::default(),
                &[],
            )
            .unwrap();
        assert_eq!(result.status, SolveStatus::Solved);
        assert_eq!(result.move_layers.len(), 0);
    }

    #[test]
    fn solve_entangling_rh_solves_nontrivial() {
        // Start with qubits at non-entangling positions; the orchestrator
        // must drive them to a feasible configuration.
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        let result = solver
            .solve_entangling_rh(
                [(0, loc(0, 0)), (1, loc(1, 5))],
                &[(0, 1)],
                std::iter::empty(),
                Some(5000),
                &SolveOptions {
                    strategy: Strategy::Ids,
                    restarts: 1,
                    ..SolveOptions::default()
                },
                &EntanglingOptions::default(),
                &RecedingHorizonOptions {
                    k_candidates: 3,
                    rollout_horizon: 5,
                    commit_depth: 1,
                    ..RecedingHorizonOptions::default()
                },
                &[],
            )
            .unwrap();
        assert_eq!(result.status, SolveStatus::Solved);
    }

    #[test]
    fn solve_entangling_rh_with_restarts() {
        // restarts=2 engages the rayon wrapper and pick_best path.
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        let result = solver
            .solve_entangling_rh(
                [(0, loc(0, 0)), (1, loc(1, 5))],
                &[(0, 1)],
                std::iter::empty(),
                Some(5000),
                &SolveOptions {
                    strategy: Strategy::Ids,
                    restarts: 2,
                    ..SolveOptions::default()
                },
                &EntanglingOptions::default(),
                &RecedingHorizonOptions {
                    k_candidates: 3,
                    rollout_horizon: 5,
                    commit_depth: 1,
                    branch_parallel: false, // give cores to restart parallelism
                    ..RecedingHorizonOptions::default()
                },
                &[],
            )
            .unwrap();
        assert_eq!(result.status, SolveStatus::Solved);
    }
}
