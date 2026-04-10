//! High-level solver that ties together the lane index, expander, heuristic,
//! and A* search into a single reusable object.
//!
//! [`MoveSolver`] is constructed once per architecture (parsing JSON and
//! building indexes) and can then solve multiple placement problems.

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;
use rayon::prelude::*;

use crate::astar::SearchResult;
use crate::config::{Config, ConfigError};
use crate::frontier::{self, BfsFrontier, DfsFrontier, IdsFrontier, PriorityFrontier};
use crate::graph::MoveSet;
use crate::heuristic::{DistanceTable, HopDistanceHeuristic};
use crate::heuristic_expander::{DeadlockPolicy, FreeRiderPolicy, HeuristicExpander};
use crate::lane_index::LaneIndex;

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

/// Outcome status of a solve attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveStatus {
    /// A solution was found.
    Solved,
    /// The search space was fully explored — no solution exists.
    Unsolvable,
    /// The expansion budget was exhausted before finding a solution or
    /// proving unsolvability.
    BudgetExceeded,
}

/// Result of a solve attempt.
///
/// Always returned (never `None`). Check [`status`](SolveResult::status) to
/// determine whether a solution was found.
#[derive(Debug)]
pub struct SolveResult {
    /// Whether the solve succeeded, was unsolvable, or ran out of budget.
    pub status: SolveStatus,
    /// Sequence of parallel move steps from initial to goal configuration.
    /// Empty when `status` is not `Solved`.
    pub move_layers: Vec<MoveSet>,
    /// Final qubit positions (the goal configuration).
    /// Equals the initial configuration when `status` is not `Solved`.
    pub goal_config: Config,
    /// Number of nodes expanded during search.
    pub nodes_expanded: u32,
    /// Total path cost. 0.0 when `status` is not `Solved`.
    pub cost: f64,
    /// Number of deadlocks encountered during search.
    pub deadlocks: u32,
}

/// Reusable move synthesis solver.
///
/// Constructed once per architecture — parses the arch spec JSON and builds
/// the lane index. Then [`solve`](MoveSolver::solve) can be called multiple
/// times with different initial/target placements.
///
/// Works for both physical and logical architectures (same interface,
/// different arch spec JSON).
#[derive(Debug)]
pub struct MoveSolver {
    index: LaneIndex,
}

impl MoveSolver {
    /// Construct from an [`ArchSpec`] JSON string.
    ///
    /// Parses the JSON, builds the lane index (precomputes all lane lookups,
    /// endpoints, and positions).
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        let arch_spec = serde_json::from_str(json)?;
        Ok(Self {
            index: LaneIndex::new(arch_spec),
        })
    }

    /// Construct from an existing [`LaneIndex`].
    pub fn from_index(index: LaneIndex) -> Self {
        Self { index }
    }

    /// Access the underlying lane index.
    pub fn index(&self) -> &LaneIndex {
        &self.index
    }

    /// Solve a move synthesis problem.
    ///
    /// Finds the minimum-cost sequence of parallel move steps to move
    /// qubits from `initial` placement to `target` placement, avoiding
    /// `blocked` locations.
    ///
    /// # Arguments
    ///
    /// * `initial` — Starting qubit positions: `(qubit_id, location)` pairs.
    /// * `target` — Desired qubit positions: `(qubit_id, location)` pairs.
    /// * `blocked` — Locations occupied by external atoms (immovable obstacles).
    /// * `max_expansions` — Optional limit on node expansions.
    /// * `strategy` — Search strategy to use.
    /// * `top_c` — Top bus options per qubit in the heuristic expander.
    /// * `max_movesets_per_group` — Max movesets generated per bus group.
    /// * `weight` — Heuristic weight for A* (1.0 = standard, >1.0 = bounded suboptimal).
    /// * `mobility_weight` — Weight for mobility bonus in expander scoring (0.0 = disabled).
    /// * `restarts` — Number of parallel restarts with perturbed scoring (1 = no restarts).
    ///
    /// # Returns
    ///
    /// A [`SolveResult`] whose [`status`](SolveResult::status) indicates
    /// whether a solution was found, the problem is unsolvable, or the
    /// expansion budget was exceeded.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] if `initial` contains duplicate qubit IDs.
    #[allow(clippy::too_many_arguments)]
    pub fn solve(
        &self,
        initial: impl IntoIterator<Item = (u32, LocationAddr)>,
        target: impl IntoIterator<Item = (u32, LocationAddr)>,
        blocked: impl IntoIterator<Item = LocationAddr>,
        max_expansions: Option<u32>,
        strategy: Strategy,
        top_c: usize,
        max_movesets_per_group: usize,
        weight: f64,
        mobility_weight: f64,
        restarts: u32,
        free_rider_policy: FreeRiderPolicy,
        lookahead: bool,
    ) -> Result<SolveResult, ConfigError> {
        let root = Config::new(initial)?;
        let target_pairs: Vec<(u32, LocationAddr)> = target.into_iter().collect();
        let blocked_locs: Vec<LocationAddr> = blocked.into_iter().collect();

        // Build goal predicate.
        let target_encoded: Vec<(u32, u32)> =
            target_pairs.iter().map(|&(q, l)| (q, l.encode())).collect();
        let goal = |config: &Config| -> bool {
            target_encoded.iter().all(|&(qid, target_enc)| {
                config
                    .location_of(qid)
                    .is_some_and(|l| l.encode() == target_enc)
            })
        };

        // Build distance table and heuristic (shared across restarts).
        let target_locs: Vec<u32> = target_encoded.iter().map(|&(_, l)| l).collect();
        let dist_table = DistanceTable::new(&target_locs, &self.index);
        let heuristic = HopDistanceHeuristic::new(target_pairs.iter().copied(), &dist_table);
        // Max heuristic (admissible) for A*/Greedy, sum heuristic for IDS/DFS ordering.
        let h_max = |config: &Config| -> f64 { heuristic.estimate_max(config) };
        let h_sum = |config: &Config| -> f64 { heuristic.estimate_sum(config) };

        // Build an expander with the given seed and deadlock policy.
        let make_expander = |seed: u64, policy: DeadlockPolicy| {
            HeuristicExpander::new(
                &self.index,
                blocked_locs.iter().copied(),
                target_pairs.iter().copied(),
                &dist_table,
                top_c,
                max_movesets_per_group,
                mobility_weight,
                seed,
                policy,
                free_rider_policy,
                lookahead,
            )
        };

        // Extract a SolveResult from a SearchResult.
        let extract = |result: SearchResult, deadlocks: u32, max_exp: Option<u32>| -> SolveResult {
            match result.goal {
                Some(goal_id) => {
                    let move_layers = result.solution_path().unwrap_or_default();
                    let goal_config = result.graph.config(goal_id).clone();
                    let cost = result.graph.g_score(goal_id);
                    SolveResult {
                        status: SolveStatus::Solved,
                        move_layers,
                        goal_config,
                        nodes_expanded: result.nodes_expanded,
                        cost,
                        deadlocks,
                    }
                }
                None => {
                    let root_config = result.graph.config(result.graph.root()).clone();
                    let status = if max_exp.is_some_and(|max| result.nodes_expanded >= max) {
                        SolveStatus::BudgetExceeded
                    } else {
                        SolveStatus::Unsolvable
                    };
                    SolveResult {
                        status,
                        move_layers: Vec::new(),
                        goal_config: root_config,
                        nodes_expanded: result.nodes_expanded,
                        cost: 0.0,
                        deadlocks,
                    }
                }
            }
        };

        // Pre-compute blocked set for entropy (avoids per-restart allocation).
        let blocked_encoded: std::collections::HashSet<u32> =
            blocked_locs.iter().map(|l| l.encode()).collect();

        // Helper: pick best result (prefer solved, then lowest cost).
        let pick_best = |results: Vec<SolveResult>| -> SolveResult {
            results
                .into_iter()
                .min_by(|a, b| {
                    let a_solved = a.status == SolveStatus::Solved;
                    let b_solved = b.status == SolveStatus::Solved;
                    b_solved.cmp(&a_solved).then(a.cost.total_cmp(&b.cost))
                })
                .expect("non-empty results")
        };

        // Helper: run a single inner strategy with the given seed.
        let run_inner = |inner: InnerStrategy, seed: u64| -> SolveResult {
            match inner {
                InnerStrategy::Ids => {
                    let expander = make_expander(seed, DeadlockPolicy::Skip);
                    let mut f = IdsFrontier::new(h_sum);
                    let result = frontier::run_search(
                        root.clone(),
                        goal,
                        &expander,
                        &mut f,
                        max_expansions,
                        None,
                    );
                    extract(result, expander.deadlock_count(), max_expansions)
                }
                InnerStrategy::Dfs => {
                    let expander = make_expander(seed, DeadlockPolicy::Skip);
                    let mut f = DfsFrontier::new(h_sum);
                    let result = frontier::run_search(
                        root.clone(),
                        goal,
                        &expander,
                        &mut f,
                        max_expansions,
                        None,
                    );
                    extract(result, expander.deadlock_count(), max_expansions)
                }
                InnerStrategy::Entropy => {
                    let entropy_params = crate::entropy::EntropyParams {
                        top_c,
                        max_movesets_per_group,
                        lookahead,
                        ..crate::entropy::EntropyParams::default()
                    };
                    let result = crate::entropy::entropy_search(
                        root.clone(),
                        goal,
                        &entropy_params,
                        &self.index,
                        &dist_table,
                        &target_encoded,
                        &blocked_encoded,
                        max_expansions,
                        None,
                        seed,
                    );
                    extract(result, 0, max_expansions)
                }
            }
        };

        // Helper: run inner strategy with parallel restarts, return best.
        let run_inner_with_restarts = |inner: InnerStrategy| -> SolveResult {
            if restarts <= 1 {
                run_inner(inner, 0)
            } else {
                let results: Vec<SolveResult> = (0..restarts)
                    .into_par_iter()
                    .map(|i| run_inner(inner, i as u64 + 1))
                    .collect();
                pick_best(results)
            }
        };

        // ── Cascade: inner restarts + single A* refinement ─────────
        if let Strategy::Cascade { inner } = strategy {
            // Phase 1: run inner strategy with restarts.
            let inner_result = run_inner_with_restarts(inner);

            if inner_result.status != SolveStatus::Solved {
                return Ok(inner_result);
            }

            // Phase 2: single weighted A* bounded by inner cost.
            let max_depth = Some((inner_result.cost as u32).saturating_sub(1));
            let astar_expander = make_expander(0, DeadlockPolicy::MoveBlockers);
            let mut astar_f = PriorityFrontier::astar(h_max, weight);
            let astar_result = frontier::run_search(
                root.clone(),
                goal,
                &astar_expander,
                &mut astar_f,
                max_expansions,
                max_depth,
            );
            let astar_solve = extract(
                astar_result,
                astar_expander.deadlock_count(),
                max_expansions,
            );

            if astar_solve.status == SolveStatus::Solved {
                return Ok(pick_best(vec![inner_result, astar_solve]));
            }
            return Ok(inner_result);
        }

        // ── Non-cascade strategies ─────────────────────────────────

        // Run a single search with the given seed.
        let run_once = |seed: u64| -> SolveResult {
            match strategy {
                Strategy::Entropy => run_inner(InnerStrategy::Entropy, seed),
                Strategy::Ids => run_inner(InnerStrategy::Ids, seed),
                Strategy::HeuristicDfs => run_inner(InnerStrategy::Dfs, seed),
                _ => {
                    let expander = make_expander(seed, DeadlockPolicy::MoveBlockers);
                    let result = Self::run_strategy(
                        strategy,
                        root.clone(),
                        goal,
                        h_max,
                        &expander,
                        max_expansions,
                        weight,
                    );
                    extract(result, expander.deadlock_count(), max_expansions)
                }
            }
        };

        if restarts <= 1 {
            Ok(run_once(0))
        } else {
            let results: Vec<SolveResult> = (0..restarts)
                .into_par_iter()
                .map(|i| run_once(i as u64 + 1))
                .collect();
            Ok(pick_best(results))
        }
    }

    /// Dispatch to the appropriate search strategy.
    fn run_strategy(
        strategy: Strategy,
        root: Config,
        goal: impl Fn(&Config) -> bool + Copy,
        heuristic_fn: impl Fn(&Config) -> f64 + Copy,
        expander: &HeuristicExpander<'_>,
        max_expansions: Option<u32>,
        weight: f64,
    ) -> SearchResult {
        match strategy {
            Strategy::AStar => {
                let mut f = PriorityFrontier::astar(heuristic_fn, weight);
                frontier::run_search(root, goal, expander, &mut f, max_expansions, None)
            }
            Strategy::Bfs => {
                let mut f = BfsFrontier::new();
                frontier::run_search(root, goal, expander, &mut f, max_expansions, None)
            }
            Strategy::GreedyBestFirst => {
                let mut f = PriorityFrontier::greedy(heuristic_fn);
                frontier::run_search(root, goal, expander, &mut f, max_expansions, None)
            }
            _ => {
                unreachable!("IDS/DFS/Cascade/Entropy handled before run_strategy")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{example_arch_json, loc};

    #[test]
    fn solve_simple_one_step() {
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        let result = solver
            .solve(
                [(0, loc(0, 0))],
                [(0, loc(0, 5))],
                std::iter::empty(),
                Some(100),
                Strategy::AStar,
                3,
                3,
                1.0,
                0.0,
                1,
                FreeRiderPolicy::Off,
                false,
            )
            .unwrap();

        assert_eq!(result.status, SolveStatus::Solved);
        assert_eq!(result.cost, 1.0);
        assert_eq!(result.move_layers.len(), 1);
        assert_eq!(result.goal_config.location_of(0), Some(loc(0, 5)));
    }

    #[test]
    fn solve_already_at_target() {
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        let result = solver
            .solve(
                [(0, loc(0, 5))],
                [(0, loc(0, 5))],
                std::iter::empty(),
                Some(100),
                Strategy::AStar,
                3,
                3,
                1.0,
                0.0,
                1,
                FreeRiderPolicy::Off,
                false,
            )
            .unwrap();

        assert_eq!(result.status, SolveStatus::Solved);
        assert_eq!(result.cost, 0.0);
        assert!(result.move_layers.is_empty());
        assert_eq!(result.nodes_expanded, 0);
    }

    #[test]
    fn solve_cross_word() {
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        // Move qubit from word 0 site 5 to word 1 site 5 (one word bus hop).
        let result = solver
            .solve(
                [(0, loc(0, 5))],
                [(0, loc(1, 5))],
                std::iter::empty(),
                Some(100),
                Strategy::AStar,
                3,
                3,
                1.0,
                0.0,
                1,
                FreeRiderPolicy::Off,
                false,
            )
            .unwrap();

        assert_eq!(result.status, SolveStatus::Solved);
        assert_eq!(result.cost, 1.0);
        assert_eq!(result.move_layers.len(), 1);
        assert_eq!(result.goal_config.location_of(0), Some(loc(1, 5)));
    }

    #[test]
    fn solve_multi_step() {
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        // Word 0 site 0 → word 1 site 5: needs site bus + word bus = 2 steps.
        let result = solver
            .solve(
                [(0, loc(0, 0))],
                [(0, loc(1, 5))],
                std::iter::empty(),
                Some(1000),
                Strategy::AStar,
                3,
                3,
                1.0,
                0.0,
                1,
                FreeRiderPolicy::Off,
                false,
            )
            .unwrap();

        assert_eq!(result.status, SolveStatus::Solved);
        assert_eq!(result.cost, 2.0);
        assert_eq!(result.move_layers.len(), 2);
    }

    #[test]
    fn solve_no_solution() {
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        // Target a nonexistent location.
        let result = solver
            .solve(
                [(0, loc(0, 0))],
                [(0, loc(99, 99))],
                std::iter::empty(),
                Some(100),
                Strategy::AStar,
                3,
                3,
                1.0,
                0.0,
                1,
                FreeRiderPolicy::Off,
                false,
            )
            .unwrap();

        assert_ne!(result.status, SolveStatus::Solved);
        assert!(result.move_layers.is_empty());
    }

    #[test]
    fn solver_reusable() {
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();

        let r1 = solver
            .solve(
                [(0, loc(0, 0))],
                [(0, loc(0, 5))],
                std::iter::empty(),
                Some(100),
                Strategy::AStar,
                3,
                3,
                1.0,
                0.0,
                1,
                FreeRiderPolicy::Off,
                false,
            )
            .unwrap();

        let r2 = solver
            .solve(
                [(0, loc(0, 5))],
                [(0, loc(0, 0))],
                std::iter::empty(),
                Some(100),
                Strategy::AStar,
                3,
                3,
                1.0,
                0.0,
                1,
                FreeRiderPolicy::Off,
                false,
            )
            .unwrap();

        assert_eq!(r1.cost, 1.0);
        assert_eq!(r2.cost, 1.0);
    }

    #[test]
    fn solve_with_blocked() {
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        // Qubit at site 0, target site 5, but site 5 is blocked.
        // Should find no solution (or a longer path if one exists).
        let result = solver
            .solve(
                [(0, loc(0, 0))],
                [(0, loc(0, 5))],
                [loc(0, 5)],
                Some(100),
                Strategy::AStar,
                3,
                3,
                1.0,
                0.0,
                1,
                FreeRiderPolicy::Off,
                false,
            )
            .unwrap();

        // Can't reach blocked destination.
        assert_ne!(result.status, SolveStatus::Solved);
        assert!(result.move_layers.is_empty());
    }

    #[test]
    fn solve_multiple_qubits() {
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        // Move two qubits from sites 0,1 to sites 5,6 (parallel site bus move).
        let result = solver
            .solve(
                [(0, loc(0, 0)), (1, loc(0, 1))],
                [(0, loc(0, 5)), (1, loc(0, 6))],
                std::iter::empty(),
                Some(1000),
                Strategy::AStar,
                3,
                3,
                1.0,
                0.0,
                1,
                FreeRiderPolicy::Off,
                false,
            )
            .unwrap();

        // Should find the parallel move in 1 step.
        assert_eq!(result.status, SolveStatus::Solved);
        assert_eq!(result.cost, 1.0);
        assert_eq!(result.goal_config.location_of(0), Some(loc(0, 5)));
        assert_eq!(result.goal_config.location_of(1), Some(loc(0, 6)));
    }

    #[test]
    fn cascade_finds_equal_or_better_than_ids() {
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        // Multi-step problem: word 0 site 0 → word 1 site 5.
        let ids_result = solver
            .solve(
                [(0, loc(0, 0))],
                [(0, loc(1, 5))],
                std::iter::empty(),
                Some(1000),
                Strategy::Ids,
                3,
                3,
                1.0,
                0.0,
                1,
                FreeRiderPolicy::Off,
                false,
            )
            .unwrap();

        let cascade_result = solver
            .solve(
                [(0, loc(0, 0))],
                [(0, loc(1, 5))],
                std::iter::empty(),
                Some(1000),
                Strategy::Cascade {
                    inner: InnerStrategy::Ids,
                },
                3,
                3,
                1.0,
                0.0,
                1,
                FreeRiderPolicy::Off,
                false,
            )
            .unwrap();

        assert!(cascade_result.cost <= ids_result.cost);
    }
}
