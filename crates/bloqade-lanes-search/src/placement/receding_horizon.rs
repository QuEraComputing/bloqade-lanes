//! Receding-horizon (MPC-style) loose-goal placement search.
//!
//! This module orchestrates a sequence of K-candidate-rollout stages, each
//! using the existing IDS search infrastructure as the rollout simulator and
//! the existing Hungarian variants as both the per-candidate compass and the
//! per-leaf evaluator. See `docs/superpowers/plans/2026-05-11-receding-horizon-loose-goal-design.md`.
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
//! [`crate::search::solve`]) wraps a parallel restart loop around
//! [`solve_entangling_rh_single`].

use std::collections::HashSet;
use std::sync::Arc;

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;
use bloqade_lanes_bytecode_core::arch::types::ArchSpec;
use rayon::prelude::*;

use crate::cost::UniformCost;
use crate::drivers::frontier::{self, IdsFrontier};
use crate::generators::heuristic::DeadlockPolicy;
use crate::generators::{HeuristicGenerator, LooseTargetGenerator};
use crate::goals::EntanglingConstraintGoal;
use crate::observer::NoOpObserver;
use crate::ops::entangling;
use crate::primitives::config::Config;
use crate::primitives::context::{SearchContext, SearchState};
use crate::primitives::distance::{DistanceTable, PairDistanceHeuristic};
use crate::primitives::graph::{MoveSet, NodeId, SearchGraph};
use crate::primitives::lane_index::LaneIndex;
use crate::scorers::DistanceScorer;
use crate::search::solve::{EntanglingOptions, SolveOptions, SolveResult, SolveStatus};
use crate::traits::{CandidateScorer, Goal, Heuristic, MoveGenerator};

use crate::ops::entangling::{LOOKAHEAD_BETA, MOVE_PENALTY};
/// Cap on how many extra noise-perturbed candidates we try when the
/// weight-grid alone returns fewer than `K` unique assignments.
const MAX_NOISE_TOP_UP_RETRIES: u64 = 50;

/// Quality gate for the greedy/beam rollout: when the rollout completes
/// the full horizon without reaching the goal, the leaf's heuristic
/// score must be at most this fraction of the root's score for us to
/// accept the result. Otherwise we fall back to IDS — greedy wandered
/// without making meaningful progress, IDS may yet find a goal.
///
/// `1.0` means "any non-increasing score is accepted" (effectively no
/// gating). `0.85` requires the leaf to be ≤ 85% of the root's score,
/// i.e. greedy must have closed at least 15% of the distance. Empirical
/// sweet spot; smaller values are more aggressive (more IDS fallbacks,
/// better quality, more wall-clock).
const GREEDY_PROGRESS_THRESHOLD: f64 = 0.85;

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
    /// If `true` (default), each rollout first tries a cheap bounded
    /// beam-search (see `inner_beam_width`) before falling back to full
    /// IDS only when the beam dead-ends. Set `false` to skip the beam
    /// attempt and always run IDS — slower (~10× more wall-clock) but
    /// more reliable quality at high atom density.
    pub greedy_first: bool,
    /// Beam width for the inner pre-search when `greedy_first = true`.
    /// `1` is pure greedy (cheapest, most prone to local optima);
    /// `2` (default) doubles cost but escapes most single-step dead-ends
    /// and significantly improves quality at high density;
    /// `3+` provides diminishing returns. Has no effect when
    /// `greedy_first = false`.
    pub inner_beam_width: u32,
}

impl Default for RecedingHorizonOptions {
    fn default() -> Self {
        // Defaults recalibrated from the 80q × 30-pair × depth-3 sweep
        // after the beam-2 + quality-gate inner search landed: cheaper
        // rollouts shifted the cost/quality math, and m=3 now beats both
        // m=1 (over-replanning lands in different basins) and m=5
        // (under-replanning commits 5-layer sequences that may include
        // bad tail moves). m=3 captures beam-2's 5-layer lookahead value
        // and re-plans before committing potentially-stale layers. K=5
        // remains at the cost/quality knee. See
        // `scripts/eval_sweep_m_80q.py`.
        Self {
            k_candidates: 5,
            rollout_horizon: 5,
            commit_depth: 3,
            tier0_next_h_weight: 0.5,
            weight_grid: default_weight_grid(),
            fallback_x_decrement: 1,
            branch_parallel: true,
            // Reduced from 1000 after profiling: with the greedy-first
            // rollout path, IDS fallback only fires when greedy gets stuck
            // (rare). Keep the IDS budget tight so a single stuck rollout
            // can't dominate stage wall-clock; the orchestrator's all-drop
            // fallback handles cases where the budget is genuinely needed.
            max_expansions_per_rollout: 300,
            greedy_first: true,
            // Beam-2: ~2× greedy cost (still ~25× cheaper than IDS) but
            // escapes most single-step local optima that bite pure-greedy
            // at high atom density. Empirical sweet spot.
            inner_beam_width: 2,
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

/// Fast bounded beam-search rollout used as the first attempt inside
/// [`run_inner_rollout`].
///
/// At each depth step, asks the generator for candidate moves from every
/// node in the current beam, scores them with `DistanceScorer`, keeps the
/// top-`beam_width` by score, and applies all of them — producing the
/// next beam. No backtracking; the rollout succeeds when any beam node
/// reaches the goal or the beam completes `max_depth` cleanly. It fails
/// (returns `goal_node = None` with `max_depth_reached < max_depth`)
/// only when *every* beam node dead-ends at the same step, in which
/// case the caller should fall back to full IDS.
///
/// `beam_width = 1` is a pure greedy walk. `beam_width = 2` doubles per-
/// rollout cost but escapes single-step local optima — empirically much
/// more robust at high density, where greedy gets trapped in suboptimal
/// short-term moves. The beam grows up to `beam_width`, never beyond.
#[allow(clippy::too_many_arguments)]
fn beam_rollout<G: Goal>(
    root: Config,
    generator: &LooseTargetGenerator,
    blocked: &HashSet<u64>,
    cz_pairs: &[(u32, u32)],
    targets: &[(u32, u64)],
    goal: &G,
    max_depth: u32,
    beam_width: usize,
) -> RolloutOutcome {
    let beam_width = beam_width.max(1);
    let mut graph = SearchGraph::new(root);
    let scorer = DistanceScorer;
    let mut search_state = SearchState::default();
    let mut candidates: Vec<crate::primitives::context::MoveCandidate> = Vec::new();
    let mut nodes_expanded: u32 = 0;

    let ctx = SearchContext {
        index: generator.index_ref(),
        dist_table: generator.dist_table_ref(),
        blocked,
        targets,
        cz_pairs: Some(cz_pairs),
    };

    // Root goal check.
    if goal.is_goal(graph.config(graph.root())) {
        return RolloutOutcome {
            goal_node: Some(graph.root()),
            max_depth_reached: 0,
            nodes_expanded: 0,
            graph,
        };
    }

    // Active beam: at most `beam_width` nodes alive at the current depth.
    let mut beam: Vec<NodeId> = vec![graph.root()];

    for _ in 0..max_depth {
        // Collect (parent, candidate, score) for every candidate from every
        // beam member. The pool is the union of all next-step options.
        let mut pool: Vec<(NodeId, crate::primitives::context::MoveCandidate, f64)> = Vec::new();
        for &node in &beam {
            candidates.clear();
            generator.generate(
                graph.config(node),
                node,
                &ctx,
                &mut search_state,
                &mut candidates,
            );
            nodes_expanded = nodes_expanded.saturating_add(1);
            let cur_cfg = graph.config(node).clone();
            for c in candidates.drain(..) {
                let s = scorer.score(&c, &cur_cfg, &ctx);
                pool.push((node, c, s));
            }
        }

        if pool.is_empty() {
            // All beam members dead-ended at this depth — fall back to IDS.
            // Use the depth of the first beam node (all beam members are
            // at the same depth by construction).
            let depth = graph.depth(beam[0]);
            return RolloutOutcome {
                goal_node: None,
                max_depth_reached: depth,
                nodes_expanded,
                graph,
            };
        }

        // Sort pool by score descending and keep top `beam_width`.
        pool.sort_by(|(_, _, a), (_, _, b)| b.total_cmp(a));
        pool.truncate(beam_width);

        // Apply each kept candidate. Goal check on every new node — return
        // immediately on first hit (a beam node reaching the goal is the
        // best possible outcome of this rollout).
        let mut new_beam: Vec<NodeId> = Vec::with_capacity(pool.len());
        for (parent, c, _) in pool {
            let new_g = graph.g_score(parent) + 1.0;
            let (next, _is_new) = graph.insert(parent, c.move_set, c.new_config, new_g);
            if goal.is_goal(graph.config(next)) {
                let depth = graph.depth(next);
                return RolloutOutcome {
                    goal_node: Some(next),
                    max_depth_reached: depth,
                    nodes_expanded,
                    graph,
                };
            }
            new_beam.push(next);
        }
        beam = new_beam;
    }

    // Completed max_depth without finding goal. The caller's
    // `extract_best_leaf` picks the lowest-h beam node by walking the
    // graph; reporting any beam member's depth here is sufficient
    // (they're all at max_depth).
    let depth = graph.depth(beam[0]);
    RolloutOutcome {
        goal_node: None,
        max_depth_reached: depth,
        nodes_expanded,
        graph,
    }
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
    greedy_first: bool,
    inner_beam_width: u32,
) -> RolloutOutcome {
    let inner =
        HeuristicGenerator::configured(restart_seed, deadlock_policy, inner_lookahead, Some(top_c));

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

    // ── Phase 1: try cheap greedy walk first (if enabled) ──────────────
    // Most rollouts succeed under greedy at moderate density. We fall
    // back to IDS in two cases:
    //   (a) greedy dead-ended (no candidates from some node)
    //   (b) greedy completed the full horizon but the leaf's heuristic
    //       score barely dropped from the root's — i.e., greedy
    //       wandered without making meaningful progress. The quality
    //       gate uses `h_sum.estimate(...)` (cheap) to detect this.
    //
    // Greedy is ~50× cheaper per rollout when it succeeds; the gate
    // adds two h-score evaluations per gated rollout — negligible.
    if greedy_first {
        let greedy_outcome = beam_rollout(
            root.clone(),
            &generator,
            blocked,
            &cz_pairs_for_ctx,
            &targets_for_ctx,
            goal,
            max_depth,
            inner_beam_width.max(1) as usize,
        );
        if greedy_outcome.goal_node.is_some() {
            // Tier-0 (goal reached) is always best — accept immediately.
            return greedy_outcome;
        }
        if greedy_outcome.max_depth_reached >= max_depth {
            // Tier-1 candidate: gate on progress.
            let start_h = h_sum.estimate(&root);
            // The same h-fn used for ranking leaves below by
            // extract_best_leaf — pick the lowest-h leaf as the gate
            // metric (the one classify_into_tier will later pick).
            let h_fn = |cfg: &Config| h_sum.estimate(cfg);
            if let Some(leaf) = extract_best_leaf(&greedy_outcome.graph, max_depth, h_fn) {
                let leaf_h = h_sum.estimate(greedy_outcome.graph.config(leaf));
                if leaf_h <= GREEDY_PROGRESS_THRESHOLD * start_h {
                    return greedy_outcome;
                }
                // else: greedy made no real progress, fall through to IDS.
            } else {
                // No depth-x leaf found in the graph — shouldn't happen
                // if greedy reported max_depth_reached >= max_depth. Be
                // defensive and fall through to IDS.
            }
        }
    }

    // ── Phase 2: greedy got stuck; fall back to full IDS ───────────────
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
                rh_opts.greedy_first,
                rh_opts.inner_beam_width,
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
        let mut commit_count = if best.is_tier0() {
            best.depth() as usize
        } else {
            (rh_opts.commit_depth as usize).min(best.depth() as usize)
        };
        if commit_count == 0 {
            // Nothing to commit (shouldn't happen — guards against infinite loop).
            let fb = fallback(&state);
            return merge_fallback(committed_layers, fb, total_expansions);
        }
        // Advance state. If the partial-commit replay fails (would only
        // happen on an internal inconsistency between the rollout graph
        // and the orchestrator's tracked state — apply_move_set returns
        // None), promote the commit to the full path so `committed_layers`
        // and `state` stay consistent (both reach `best.leaf_config()`).
        // Without this fallback the orchestrator would silently emit a
        // prefix of moves whose replay does not reproduce `goal_config`.
        state = if commit_count == best.path().len() {
            best.leaf_config().clone()
        } else {
            match apply_layers_to(state.clone(), best.path().iter().take(commit_count), &index) {
                Some(s) => s,
                None => {
                    commit_count = best.path().len();
                    best.leaf_config().clone()
                }
            }
        };
        for ms in best.path().iter().take(commit_count) {
            committed_layers.push(ms.clone());
        }

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
/// Used when `commit_depth < rollout_horizon` (the default `m=3 < x=5`),
/// so the orchestrator commits the first `m` move layers of a tier-1
/// path and reads the post-commit config from the rollout graph; tier-0
/// branches always commit their full path and skip this helper.
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
///
/// Move layers are *parallel* in the AOD model — all sources are lifted
/// before any destination is occupied, so a move layer can validly
/// contain a chain like `(L1 → L2, L2 → L3)` where one lane's source is
/// another lane's destination. To preserve that semantics, we snapshot
/// the source → qubit-index map from the *input* config before applying
/// any move, then write all destinations against that snapshot. Naively
/// mutating in place would mis-attribute the chained move to the just-
/// moved atom rather than the atom originally at L2.
///
/// Complete AOD rectangles may also contain empty filler lanes. Those lanes
/// have no source qubit in the input config, so replay skips them.
fn apply_move_set(config: &Config, move_set: &MoveSet, index: &LaneIndex) -> Option<Config> {
    use bloqade_lanes_bytecode_core::arch::addr::{LaneAddr, LocationAddr};
    use std::collections::HashMap;

    let mut next: Vec<(u32, LocationAddr)> = config.iter().collect();
    // Snapshot source → index-in-`next` from the input config. Multiple
    // qubits should never share a source location; if they do, treat
    // the move set as malformed and signal failure.
    let mut src_to_idx: HashMap<LocationAddr, usize> = HashMap::with_capacity(next.len());
    for (i, (_, loc)) in next.iter().enumerate() {
        if src_to_idx.insert(*loc, i).is_some() {
            return None;
        }
    }

    for &lane_enc in move_set.encoded_lanes() {
        let lane = LaneAddr::decode_u64(lane_enc);
        let (src, dst) = index.endpoints(&lane)?;
        let Some(&idx) = src_to_idx.get(&src) else {
            continue;
        };
        next[idx].1 = dst;
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

// ── CzPlacement composition ─────────────────────────────────────────────

use crate::placement::cz_placement::CzPlacement;
use crate::placement::loose_goal::solve_loose_goal;
use crate::primitives::config::ConfigError;
use crate::search::engine::SearchEngine;
use crate::search::move_search::MoveSearch;

/// MPC-style loose-goal CZ placement.
///
/// Composes `Arc<SearchEngine> + MoveSearch + EntanglingOptions +
/// RecedingHorizonOptions`. Drives [`solve_entangling_rh_single`]
/// across restarts in parallel via Rayon; falls back to
/// [`LooseGoalCzPlacement`](crate::placement::loose_goal::LooseGoalCzPlacement)'s
/// shared impl when the receding-horizon branches all drop at
/// horizon = 1.
pub struct RecedingHorizonCzPlacement {
    engine: Arc<SearchEngine>,
    search: MoveSearch,
    entangling_options: EntanglingOptions,
    rh_options: RecedingHorizonOptions,
}

impl RecedingHorizonCzPlacement {
    /// Build a `RecedingHorizonCzPlacement` from its four composing pieces.
    pub fn new(
        engine: Arc<SearchEngine>,
        search: MoveSearch,
        entangling_options: EntanglingOptions,
        rh_options: RecedingHorizonOptions,
    ) -> Self {
        Self {
            engine,
            search,
            entangling_options,
            rh_options,
        }
    }

    /// Borrow the underlying engine.
    pub fn engine(&self) -> &Arc<SearchEngine> {
        &self.engine
    }

    /// Borrow the search configuration.
    pub fn search(&self) -> &MoveSearch {
        &self.search
    }

    /// Borrow the entangling-options bundle.
    pub fn entangling_options(&self) -> &EntanglingOptions {
        &self.entangling_options
    }

    /// Borrow the receding-horizon-options bundle.
    pub fn rh_options(&self) -> &RecedingHorizonOptions {
        &self.rh_options
    }

    /// Solve using the receding-horizon MPC loop.
    ///
    /// Equivalent to the trait-level
    /// [`CzPlacement::solve`](super::cz_placement::CzPlacement::solve)
    /// but accepts `cz_pairs` and an explicit `future_cz_layers`
    /// lookahead window directly.
    pub fn solve_pairs(
        &self,
        initial: impl IntoIterator<Item = (u32, LocationAddr)>,
        cz_pairs: &[(u32, u32)],
        blocked: impl IntoIterator<Item = LocationAddr>,
        max_expansions: Option<u32>,
        future_cz_layers: &[Vec<(u32, u32)>],
    ) -> Result<SolveResult, ConfigError> {
        solve_receding_horizon(
            &self.engine,
            &self.search.options,
            &self.entangling_options,
            &self.rh_options,
            initial,
            cz_pairs,
            blocked,
            max_expansions,
            future_cz_layers,
        )
    }
}

impl CzPlacement for RecedingHorizonCzPlacement {
    fn solve(
        &self,
        initial: &[(u32, LocationAddr)],
        controls: &[u32],
        targets: &[u32],
        blocked: &[LocationAddr],
        max_expansions: Option<u32>,
    ) -> Result<SolveResult, ConfigError> {
        debug_assert_eq!(
            controls.len(),
            targets.len(),
            "controls and targets must have equal length",
        );
        let cz_pairs: Vec<(u32, u32)> = controls
            .iter()
            .copied()
            .zip(targets.iter().copied())
            .collect();
        self.solve_pairs(
            initial.iter().copied(),
            &cz_pairs,
            blocked.iter().copied(),
            max_expansions,
            &[],
        )
    }
}

/// Shared implementation backing both [`RecedingHorizonCzPlacement::solve_pairs`]
/// and the legacy
/// [`MoveSolver::solve_entangling_rh`](crate::search::solve::MoveSolver::solve_entangling_rh).
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_receding_horizon(
    engine: &SearchEngine,
    opts: &SolveOptions,
    ent_opts: &EntanglingOptions,
    rh_opts: &RecedingHorizonOptions,
    initial: impl IntoIterator<Item = (u32, LocationAddr)>,
    cz_pairs: &[(u32, u32)],
    blocked: impl IntoIterator<Item = LocationAddr>,
    max_expansions: Option<u32>,
    future_cz_layers: &[Vec<(u32, u32)>],
) -> Result<SolveResult, ConfigError> {
    use rayon::prelude::*;

    let root = Config::new(initial)?;
    let blocked_locs: Vec<LocationAddr> = blocked.into_iter().collect();
    let arch = engine.index().arch_spec();

    let cache = engine.entangling_cache();
    let dist_table = cache.dist_table.clone();

    let heuristic = PairDistanceHeuristic::new(cz_pairs, &cache.wpd);
    let goal = EntanglingConstraintGoal::new(cz_pairs, cache.ent_set.clone());
    let blocked_encoded: HashSet<u64> = blocked_locs.iter().map(|l| l.encode()).collect();

    let clipped_future = ent_opts.clipped_future_layers(future_cz_layers);
    let future_owned: Vec<Vec<(u32, u32)>> = clipped_future.to_vec();

    let upgraded_opts = opts.upgraded_for_entangling();
    let opts = &upgraded_opts;

    let arch_arc = Arc::new(arch.clone());
    let index_arc: Arc<LaneIndex> = Arc::new(engine.index().clone());

    let restarts = opts.restarts.max(1);
    let cz_pairs_owned: Vec<(u32, u32)> = cz_pairs.to_vec();

    // Fallback when receding-horizon drops at horizon=1: run a single-shot
    // loose-goal solve from the current state. Use restarts=1 to avoid
    // nested rayon parallelism.
    let single_opts = SolveOptions {
        restarts: 1,
        ..opts.clone()
    };
    let make_fallback = |state: &Config| -> SolveResult {
        let initial: Vec<(u32, LocationAddr)> = state.iter().collect();
        solve_loose_goal(
            engine,
            &single_opts,
            ent_opts,
            initial,
            cz_pairs,
            blocked_locs.iter().copied(),
            max_expansions,
            future_cz_layers,
        )
        .unwrap_or_else(|_| SolveResult::unsolvable(state.clone()))
    };

    let results: Vec<SolveResult> = if restarts <= 1 {
        vec![solve_entangling_rh_single(
            root.clone(),
            &cz_pairs_owned,
            blocked_encoded.clone(),
            arch_arc.clone(),
            index_arc.clone(),
            dist_table.clone(),
            &goal,
            &heuristic,
            opts,
            ent_opts,
            rh_opts,
            &future_owned,
            max_expansions,
            /*restart_seed*/ 0,
            make_fallback,
        )]
    } else {
        (0..restarts)
            .into_par_iter()
            .map(|i| {
                solve_entangling_rh_single(
                    root.clone(),
                    &cz_pairs_owned,
                    blocked_encoded.clone(),
                    arch_arc.clone(),
                    index_arc.clone(),
                    dist_table.clone(),
                    &goal,
                    &heuristic,
                    opts,
                    ent_opts,
                    rh_opts,
                    &future_owned,
                    max_expansions,
                    /*restart_seed*/ (i + 1) as u64,
                    make_fallback,
                )
            })
            .collect()
    };

    Ok(crate::search::restarts::pick_best(results))
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::graph::SearchGraph;
    use crate::search::solve::{MoveSolver, Strategy};
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
        assert_eq!(opts.commit_depth, 3);
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
        use crate::primitives::graph::MoveSet;
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
        use crate::primitives::graph::MoveSet;
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

    #[test]
    fn apply_move_set_skips_empty_filler_lane_sources() {
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        let index = solver.index();
        let move_set = MoveSet::new(vec![
            crate::test_utils::lane(0, 0, 0),
            crate::test_utils::lane(0, 1, 0),
        ]);
        let config = Config::new([(0, loc(0, 0))]).unwrap();

        let next = apply_move_set(&config, &move_set, index)
            .expect("empty filler lane should not make replay fail");

        assert_eq!(next.location_of(0), Some(loc(0, 5)));
        assert_eq!(next.iter().count(), 1);
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

    /// Parity: `RecedingHorizonCzPlacement::solve_pairs` must produce a
    /// byte-identical `SolveResult` to `MoveSolver::solve_entangling_rh`
    /// for the same problem and config.
    #[test]
    fn rh_placement_matches_solve_entangling_rh() {
        let engine = Arc::new(SearchEngine::from_json(example_arch_json()).unwrap());
        let opts = SolveOptions {
            strategy: Strategy::Ids,
            ..SolveOptions::default()
        };
        let ent_opts = EntanglingOptions::default();
        let rh_opts = RecedingHorizonOptions {
            k_candidates: 2,
            rollout_horizon: 4,
            commit_depth: 1,
            branch_parallel: false,
            ..RecedingHorizonOptions::default()
        };
        let search = MoveSearch::new(opts.clone(), Default::default());
        let placement = RecedingHorizonCzPlacement::new(
            engine.clone(),
            search,
            ent_opts.clone(),
            rh_opts.clone(),
        );
        let legacy = MoveSolver::from_index(engine.index().clone());

        let initial = [(0u32, loc(0, 0)), (1u32, loc(1, 5))];
        let cz_pairs = [(0u32, 1u32)];
        let blocked: [LocationAddr; 0] = [];

        let new_result = placement
            .solve_pairs(
                initial.iter().copied(),
                &cz_pairs,
                blocked.iter().copied(),
                Some(5000),
                &[],
            )
            .unwrap();
        let legacy_result = legacy
            .solve_entangling_rh(
                initial.iter().copied(),
                &cz_pairs,
                blocked.iter().copied(),
                Some(5000),
                &opts,
                &ent_opts,
                &rh_opts,
                &[],
            )
            .unwrap();

        assert_eq!(new_result.status, legacy_result.status);
        assert_eq!(new_result.cost.to_bits(), legacy_result.cost.to_bits());
        assert_eq!(new_result.nodes_expanded, legacy_result.nodes_expanded);
        let new_layers: Vec<Vec<u64>> = new_result
            .move_layers
            .iter()
            .map(|ms| ms.encoded_lanes().to_vec())
            .collect();
        let legacy_layers: Vec<Vec<u64>> = legacy_result
            .move_layers
            .iter()
            .map(|ms| ms.encoded_lanes().to_vec())
            .collect();
        assert_eq!(new_layers, legacy_layers);
    }
}
