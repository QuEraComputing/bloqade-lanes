//! Search-tuning option bundles + the strategy enum.
//!
//! - [`Strategy`] / [`InnerStrategy`] — algorithm selectors.
//! - [`SolveOptions`] — core knobs every solver entry point takes.
//! - [`EntropyOptions`] — entropy-strategy-specific knobs.
//! - [`EntanglingOptions`] — loose-goal Hungarian-assignment knobs.

use crate::generators::heuristic::DeadlockPolicy;
use crate::ops::entangling::OCCUPANCY_PENALTY_DEFAULT;

/// Inner strategy for the cascade's Phase 1 (fast feasibility search).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InnerStrategy {
    /// Iterative Diving Search.
    Ids,
    /// Heuristic depth-first search.
    Dfs,
    /// Entropy-guided search.
    Entropy,
}

/// Search strategy for the solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    /// A* / Weighted A* search: `f = g + weight * h`, goal on pop.
    /// weight=1.0 is standard A* (optimal); weight>1.0 is bounded suboptimal.
    AStar,
    /// Heuristic depth-first search: fast, bounded memory, not optimal.
    HeuristicDfs,
    /// Breadth-first search: finds shallowest solution, no heuristic.
    Bfs,
    /// Greedy best-first: fast, uses heuristic only (no path cost).
    GreedyBestFirst,
    /// Iterative Diving Search: depth-first with heuristic jump-back.
    Ids,
    /// Cascade: fast inner strategy first, then weighted A* bounded by inner cost.
    /// Restarts apply to the inner phase only; A* runs once with the tightest bound.
    Cascade { inner: InnerStrategy },
    /// Entropy-guided search: single-path DFS with entropy-based backtracking.
    Entropy,
}

/// Core search-tuning parameters shared by every solver entry point.
///
/// Strategy-specific knobs live in [`EntropyOptions`] (entropy-search
/// parameters) and [`EntanglingOptions`] (loose-goal Hungarian parameters).
/// Problem-specific data (`initial`, `target`, `blocked`, `max_expansions`)
/// remain as direct arguments.
#[derive(Debug, Clone)]
pub struct SolveOptions {
    /// Search strategy to use.
    pub strategy: Strategy,
    /// Heuristic weight for A* (1.0 = standard, >1.0 = bounded suboptimal).
    pub weight: f64,
    /// Number of parallel restarts with perturbed scoring (1 = no restarts).
    pub restarts: u32,
    /// How to handle deadlocks (no improving moves).
    pub deadlock_policy: DeadlockPolicy,
    /// Enable 2-step lookahead scoring inside
    /// [`HeuristicGenerator`](crate::generators::HeuristicGenerator).
    /// Affects every strategy that goes through that generator.
    pub lookahead: bool,
    /// Per-qubit move-candidate pruning passed to
    /// [`HeuristicGenerator`](crate::generators::HeuristicGenerator).
    /// `None` keeps all scored triples (default for the basic
    /// [`MoveSolver::solve`](crate::search::solve::MoveSolver::solve)
    /// path); `Some(n)` keeps the top `n` bus options per qubit by score.
    /// [`MoveSolver::solve_entangling`](crate::search::solve::MoveSolver::solve_entangling)
    /// defaults this to `Some(3)` when not set.
    pub top_c: Option<usize>,
}

impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            strategy: Strategy::AStar,
            weight: 1.0,
            restarts: 1,
            deadlock_policy: DeadlockPolicy::Skip,
            lookahead: false,
            top_c: None,
        }
    }
}

impl SolveOptions {
    /// Upgrade [`DeadlockPolicy::Skip`] to [`DeadlockPolicy::MoveBlockers`]
    /// for entangling-style routing — the loose-goal solver needs at
    /// least `MoveBlockers` to handle qubits competing for entangling
    /// positions. `MoveBlockers` and `AllMoves` pass through unchanged.
    pub fn upgraded_for_entangling(&self) -> SolveOptions {
        if matches!(self.deadlock_policy, DeadlockPolicy::Skip) {
            SolveOptions {
                deadlock_policy: DeadlockPolicy::MoveBlockers,
                ..self.clone()
            }
        } else {
            self.clone()
        }
    }
}

/// Entropy-strategy-specific parameters.
///
/// Only consumed when [`SolveOptions::strategy`] is [`Strategy::Entropy`]
/// (or a [`Strategy::Cascade`] variant whose inner is entropy). Pass via
/// [`MoveSolver::solve`](crate::search::solve::MoveSolver::solve)'s
/// optional `entropy_opts` argument; otherwise defaults are used.
#[derive(Debug, Clone)]
pub struct EntropyOptions {
    /// Max movesets generated per bus group.
    pub max_movesets_per_group: usize,
    /// Number of goal candidates to collect before stopping entropy search.
    pub max_goal_candidates: usize,
    /// Time-distance blend weight (0.0 = hop-count only, 1.0 = time only).
    pub w_t: f64,
    /// Collect entropy-step trace payload for visualization/debugging.
    pub collect_entropy_trace: bool,
    /// Base RNG seed for score perturbations.
    ///
    /// With a single restart (`SolveOptions::restarts == 1`), `0` (default)
    /// disables perturbations entirely for fully deterministic results.
    /// With multiple restarts, a zero base seed still assigns non-zero seeds
    /// per restart (1, 2, …) to preserve pre-existing restart diversity.
    /// A non-zero base seed starts the per-restart sequence at that value
    /// so every run is reproducible.
    pub seed: u64,
}

impl Default for EntropyOptions {
    fn default() -> Self {
        Self {
            max_movesets_per_group: 3,
            max_goal_candidates: 3,
            w_t: 0.05,
            collect_entropy_trace: false,
            seed: 0,
        }
    }
}

/// Loose-goal entangling-search parameters consumed by
/// [`MoveSolver::solve_entangling`](crate::search::solve::MoveSolver::solve_entangling).
///
/// Ignored by [`MoveSolver::solve`](crate::search::solve::MoveSolver::solve)
/// and [`MoveSolver::solve_nohome`](crate::search::solve::MoveSolver::solve_nohome).
#[derive(Debug, Clone)]
pub struct EntanglingOptions {
    /// Congestion penalty weight for the entangling Hungarian assignment.
    ///
    /// `0.0` (default): standard min-sum-distance assignment.
    /// `> 0.0`: iteratively re-runs Hungarian, adding a penalty
    /// `congestion_weight × (load - ideal_load)` to slots on overloaded
    /// entangling word pairs. Spreads CZ pairs across word pairs to reduce
    /// routing serialization at high occupancy.
    pub congestion_weight: f64,
    /// Spectator-occupancy penalty (in lane-hop units) added to each
    /// entangling Hungarian cost cell *per spectator-occupied slot half*.
    /// A spectator is an atom that is *not* in any CZ pair of the current
    /// layer.
    ///
    /// `0.0`: occupancy-blind (legacy behaviour). `> 0.0`: bias the
    /// assignment away from slots that would force the search to evict a
    /// non-participating atom. Default `1.0` was selected by sweep on
    /// the 80q / depth 3 / max_pairs 10 regime; deeper sparse-pair
    /// circuits prefer larger values (~2–3), full-layer circuits are
    /// unaffected since they have no spectators. Atoms that are
    /// themselves part of another CZ pair this layer are *not* penalised
    /// — they will be reassigned by the Hungarian and move out of the
    /// way naturally.
    ///
    /// Internally the per-cell contribution is
    /// `(spectator_half_count * occupancy_penalty).round() as u32`, so
    /// fractional values (e.g. `0.5`, `1.5`) provide meaningful sub-hop
    /// granularity for sweeps. Must be finite and non-negative.
    pub occupancy_penalty: f64,
    /// Cap on the number of future CZ layers fed to the Hungarian
    /// forward/backward sweep. `None` is unbounded; `Some(0)` disables
    /// lookahead entirely (single-layer Hungarian); `Some(n)` for n > 0
    /// keeps the first `n` future layers. Default `Some(4)` keeps solve
    /// time bounded regardless of circuit depth.
    pub hungarian_horizon: Option<usize>,
}

impl Default for EntanglingOptions {
    fn default() -> Self {
        Self {
            congestion_weight: 0.0,
            occupancy_penalty: OCCUPANCY_PENALTY_DEFAULT,
            hungarian_horizon: Some(4),
        }
    }
}

impl EntanglingOptions {
    /// Apply this options bundle's Hungarian future-layer horizon to a
    /// list of upcoming CZ layers.
    ///
    /// `hungarian_horizon == None` is unbounded; `Some(0)` disables
    /// multi-layer lookahead entirely; `Some(n > 0)` keeps the first
    /// `n` layers.
    pub fn clipped_future_layers<'a>(
        &self,
        future_cz_layers: &'a [Vec<(u32, u32)>],
    ) -> &'a [Vec<(u32, u32)>] {
        match self.hungarian_horizon {
            Some(0) => &[],
            Some(n) => &future_cz_layers[..future_cz_layers.len().min(n)],
            None => future_cz_layers,
        }
    }
}
