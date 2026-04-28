//! High-level solver that ties together the lane index, expander, heuristic,
//! and A* search into a single reusable object.
//!
//! [`MoveSolver`] is constructed once per architecture (parsing JSON and
//! building indexes) and can then solve multiple placement problems.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;
use rayon::prelude::*;

use crate::astar::SearchResult;
use crate::config::{Config, ConfigError};
use crate::context::{SearchContext, SearchState};
use crate::cost::UniformCost;
use crate::entangling::{self, WordPairDistances};
use crate::entropy::EntropyTrace;
use crate::frontier::{self, BfsFrontier, DfsFrontier, IdsFrontier, PriorityFrontier};
use crate::generators::HeuristicGenerator;
use crate::generators::heuristic::DeadlockPolicy;
use crate::goals::AllAtTarget;
use crate::graph::MoveSet;
use crate::heuristic::{DistanceTable, HopDistanceHeuristic};
use crate::lane_index::LaneIndex;
use crate::observer::NoOpObserver;
use crate::scorers::DistanceScorer;
use crate::target_generator::{TargetContext, TargetGenerator, validate_candidate};
use crate::traits::{Goal, Heuristic, MoveGenerator};

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
    /// Optional entropy-search trace payload for visualization/debugging.
    pub entropy_trace: Option<EntropyTrace>,
}

/// Grouping of search-tuning parameters for [`MoveSolver::solve`].
///
/// Keeps the `solve()` signature compact. Problem-specific data
/// (`initial`, `target`, `blocked`, `max_expansions`) remain as direct
/// arguments.
#[derive(Debug, Clone)]
pub struct SolveOptions {
    /// Search strategy to use.
    pub strategy: Strategy,
    /// Max movesets generated per bus group.
    pub max_movesets_per_group: usize,
    /// Number of goal candidates to collect before stopping entropy search.
    pub max_goal_candidates: usize,
    /// Heuristic weight for A* (1.0 = standard, >1.0 = bounded suboptimal).
    pub weight: f64,
    /// Number of parallel restarts with perturbed scoring (1 = no restarts).
    pub restarts: u32,
    /// Enable 2-step lookahead scoring.
    pub lookahead: bool,
    /// How to handle deadlocks (no improving moves).
    pub deadlock_policy: DeadlockPolicy,
    /// Time-distance blend weight (0.0 = hop-count only, 1.0 = time only).
    pub w_t: f64,
    /// Collect entropy-step trace payload when using entropy strategy.
    pub collect_entropy_trace: bool,
    /// Whether to recompute per-qubit targets dynamically on every node
    /// expansion in [`MoveSolver::solve_entangling`].
    ///
    /// `false` (default): use a static greedy pre-assignment computed once.
    /// `true`: recompute targets per expansion, avoiding premature qubit
    /// "sleep" at the cost of O(pairs × word_pairs) extra per node.
    ///
    /// Ignored by [`MoveSolver::solve`] (always uses fixed targets).
    pub dynamic_targets: bool,
    /// How often to recompute targets when `dynamic_targets` is true.
    ///
    /// `1` = every expansion (most fresh, most expensive).
    /// `10-50` = periodic recomputation (good balance).
    /// Only used by [`MoveSolver::solve_entangling`]; ignored by [`MoveSolver::solve`].
    pub recompute_interval: u32,
}

impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            strategy: Strategy::AStar,
            max_movesets_per_group: 3,
            max_goal_candidates: 3,
            weight: 1.0,
            restarts: 1,
            lookahead: false,
            deadlock_policy: DeadlockPolicy::Skip,
            w_t: 0.05,
            collect_entropy_trace: false,
            dynamic_targets: false,
            recompute_interval: 1,
        }
    }
}

// ── Shared helpers ────────────────────────────────────────────────

/// Extract a [`SolveResult`] from a [`SearchResult`].
fn extract(result: SearchResult, deadlocks: u32, max_exp: Option<u32>) -> SolveResult {
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
                entropy_trace: None,
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
                entropy_trace: None,
            }
        }
    }
}

/// Pick the best result from multiple restarts (prefer solved, then lowest cost).
fn pick_best(results: Vec<SolveResult>) -> SolveResult {
    results
        .into_iter()
        .min_by(|a, b| {
            let a_solved = a.status == SolveStatus::Solved;
            let b_solved = b.status == SolveStatus::Solved;
            b_solved.cmp(&a_solved).then(a.cost.total_cmp(&b.cost))
        })
        .expect("non-empty results")
}

/// Shared strategy dispatch + restart logic.
///
/// Both [`MoveSolver::solve`] and [`MoveSolver::solve_entangling`] delegate
/// here after constructing their specific goal, heuristic, and generator.
#[allow(clippy::too_many_arguments)]
fn run_with_components<Go, Gen, Hmax, Hsum, MkGen>(
    root: Config,
    goal: &Go,
    make_generator: MkGen,
    h_max: Hmax,
    h_sum: Hsum,
    ctx: &SearchContext,
    max_expansions: Option<u32>,
    opts: &SolveOptions,
) -> SolveResult
where
    Go: Goal + Sync,
    Gen: MoveGenerator,
    Hmax: Heuristic + Copy + Sync,
    Hsum: Heuristic + Copy + Sync,
    MkGen: Fn(u64, DeadlockPolicy) -> Gen + Sync,
{
    let strategy = opts.strategy;
    let weight = opts.weight;
    let restarts = opts.restarts;
    let deadlock_policy = opts.deadlock_policy;
    let max_movesets_per_group = opts.max_movesets_per_group;
    let max_goal_candidates = opts.max_goal_candidates;
    let collect_entropy_trace = opts.collect_entropy_trace;

    let scorer = DistanceScorer;
    let cost_fn = UniformCost;

    // Helper: run a single inner strategy with the given seed and budget.
    let run_inner = |inner: InnerStrategy, seed: u64, budget: Option<u32>| -> SolveResult {
        match inner {
            InnerStrategy::Ids => {
                let move_gen = make_generator(seed, deadlock_policy);
                let mut f = IdsFrontier::new(h_sum);
                let result = frontier::run_search(
                    root.clone(),
                    &move_gen,
                    &scorer,
                    &cost_fn,
                    goal,
                    &mut f,
                    ctx,
                    &mut SearchState::default(),
                    &mut NoOpObserver,
                    budget,
                    None,
                );
                extract(result, move_gen.deadlock_count(), budget)
            }
            InnerStrategy::Dfs => {
                let move_gen = make_generator(seed, deadlock_policy);
                let mut f = DfsFrontier::new(h_sum);
                let result = frontier::run_search(
                    root.clone(),
                    &move_gen,
                    &scorer,
                    &cost_fn,
                    goal,
                    &mut f,
                    ctx,
                    &mut SearchState::default(),
                    &mut NoOpObserver,
                    budget,
                    None,
                );
                extract(result, move_gen.deadlock_count(), budget)
            }
            InnerStrategy::Entropy => {
                let entropy_params = crate::entropy::EntropyParams {
                    max_movesets_per_group,
                    max_goal_candidates,
                    lookahead: opts.lookahead,
                    w_t: opts.w_t,
                    ..crate::entropy::EntropyParams::default()
                };
                let mut entropy_trace = if collect_entropy_trace {
                    Some(EntropyTrace::default())
                } else {
                    None
                };
                let result = crate::entropy::entropy_search(
                    root.clone(),
                    goal,
                    &entropy_params,
                    ctx,
                    budget,
                    None,
                    seed,
                    entropy_trace.as_mut(),
                );
                let mut solve = extract(result, 0, budget);
                solve.entropy_trace = entropy_trace;
                solve
            }
        }
    };

    // Helper: run inner strategy with parallel restarts, return best.
    let run_inner_with_restarts = |inner: InnerStrategy| -> SolveResult {
        if restarts <= 1 {
            run_inner(inner, 0, max_expansions)
        } else {
            let results: Vec<SolveResult> = (0..restarts)
                .into_par_iter()
                .map(|i| run_inner(inner, i as u64 + 1, max_expansions))
                .collect();
            pick_best(results)
        }
    };

    // ── Cascade: inner restarts + single A* refinement ─────────
    if let Strategy::Cascade { inner } = strategy {
        let inner_result = run_inner_with_restarts(inner);

        if inner_result.status != SolveStatus::Solved {
            return inner_result;
        }

        let max_depth = Some(inner_result.cost.ceil() as u32);
        let astar_move_gen = make_generator(0, DeadlockPolicy::MoveBlockers);
        let mut astar_f = PriorityFrontier::astar(h_max, weight);
        let astar_result = frontier::run_search(
            root.clone(),
            &astar_move_gen,
            &scorer,
            &cost_fn,
            goal,
            &mut astar_f,
            ctx,
            &mut SearchState::default(),
            &mut NoOpObserver,
            max_expansions,
            max_depth,
        );
        let astar_solve = extract(
            astar_result,
            astar_move_gen.deadlock_count(),
            max_expansions,
        );

        if astar_solve.status == SolveStatus::Solved {
            return pick_best(vec![inner_result, astar_solve]);
        }
        return inner_result;
    }

    // ── Non-cascade strategies ─────────────────────────────────

    let run_once = |seed: u64, budget: Option<u32>| -> SolveResult {
        match strategy {
            Strategy::Entropy => run_inner(InnerStrategy::Entropy, seed, budget),
            Strategy::Ids => run_inner(InnerStrategy::Ids, seed, budget),
            Strategy::HeuristicDfs => run_inner(InnerStrategy::Dfs, seed, budget),
            _ => {
                let move_gen = make_generator(seed, DeadlockPolicy::MoveBlockers);
                let result = run_strategy_v2(
                    strategy,
                    root.clone(),
                    &move_gen,
                    &scorer,
                    &cost_fn,
                    goal,
                    ctx,
                    h_max,
                    budget,
                    weight,
                );
                extract(result, move_gen.deadlock_count(), budget)
            }
        }
    };

    if restarts <= 1 {
        run_once(0, max_expansions)
    } else {
        let results: Vec<SolveResult> = (0..restarts)
            .into_par_iter()
            .map(|i| run_once(i as u64 + 1, max_expansions))
            .collect();
        pick_best(results)
    }
}

/// Dispatch to the appropriate frontier-based search strategy.
#[allow(clippy::too_many_arguments)]
fn run_strategy_v2<Go, Gen, Hmax>(
    strategy: Strategy,
    root: Config,
    generator: &Gen,
    scorer: &DistanceScorer,
    cost_fn: &UniformCost,
    goal: &Go,
    ctx: &SearchContext<'_>,
    heuristic_fn: Hmax,
    max_expansions: Option<u32>,
    weight: f64,
) -> SearchResult
where
    Go: Goal,
    Gen: MoveGenerator,
    Hmax: Heuristic + Copy,
{
    match strategy {
        Strategy::AStar => {
            let mut f = PriorityFrontier::astar(heuristic_fn, weight);
            frontier::run_search(
                root,
                generator,
                scorer,
                cost_fn,
                goal,
                &mut f,
                ctx,
                &mut SearchState::default(),
                &mut NoOpObserver,
                max_expansions,
                None,
            )
        }
        Strategy::Bfs => {
            let mut f = BfsFrontier::new();
            frontier::run_search(
                root,
                generator,
                scorer,
                cost_fn,
                goal,
                &mut f,
                ctx,
                &mut SearchState::default(),
                &mut NoOpObserver,
                max_expansions,
                None,
            )
        }
        Strategy::GreedyBestFirst => {
            let mut f = PriorityFrontier::greedy(heuristic_fn);
            frontier::run_search(
                root,
                generator,
                scorer,
                cost_fn,
                goal,
                &mut f,
                ctx,
                &mut SearchState::default(),
                &mut NoOpObserver,
                max_expansions,
                None,
            )
        }
        _ => {
            unreachable!("IDS/DFS/Cascade/Entropy handled before run_strategy_v2")
        }
    }
}

/// Cached architecture-dependent data for [`MoveSolver::solve_entangling`].
///
/// All fields depend only on the architecture (lane index), not on per-call
/// data (initial positions, CZ pairs). Built once on first
/// `solve_entangling` call and reused for all subsequent calls.
pub(crate) struct EntanglingCache {
    pub ent_set: HashSet<(u64, u64)>,
    pub partner_map: HashMap<u64, u64>,
    pub dist_table: Arc<DistanceTable>,
    pub wpd: WordPairDistances,
}

/// Reusable move synthesis solver.
///
/// Constructed once per architecture — parses the arch spec JSON and builds
/// the lane index. Then [`solve`](MoveSolver::solve) can be called multiple
/// times with different initial/target placements.
///
/// Works for both physical and logical architectures (same interface,
/// different arch spec JSON).
pub struct MoveSolver {
    index: LaneIndex,
    entangling_cache: OnceLock<EntanglingCache>,
}

impl std::fmt::Debug for MoveSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MoveSolver")
            .field("index", &self.index)
            .finish_non_exhaustive()
    }
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
            entangling_cache: OnceLock::new(),
        })
    }

    /// Construct from an existing [`LaneIndex`].
    pub fn from_index(index: LaneIndex) -> Self {
        Self {
            index,
            entangling_cache: OnceLock::new(),
        }
    }

    /// Access the underlying lane index.
    pub fn index(&self) -> &LaneIndex {
        &self.index
    }

    /// Get or build the cached entangling precomputation.
    fn entangling_cache(&self) -> &EntanglingCache {
        self.entangling_cache.get_or_init(|| {
            let arch = self.index.arch_spec();
            let word_pairs = entangling::enumerate_word_pairs(arch);
            let ent_locs = entangling::all_entangling_locations(arch);
            let ent_set = entangling::build_entangling_set(arch);
            let partner_map = entangling::build_partner_map(&ent_set);
            // Always include time distances — callers with w_t=0.0 just
            // ignore them (hop-count fields are separate).
            let dist_table = Arc::new(
                DistanceTable::new(&ent_locs, &self.index).with_time_distances(&self.index),
            );
            let wpd =
                entangling::WordPairDistances::from_dist_table(&word_pairs, arch, &dist_table);
            EntanglingCache {
                ent_set,
                partner_map,
                dist_table,
                wpd,
            }
        })
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
    /// * `opts` — Search-tuning parameters (strategy, weight, restarts, etc.).
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
    pub fn solve(
        &self,
        initial: impl IntoIterator<Item = (u32, LocationAddr)>,
        target: impl IntoIterator<Item = (u32, LocationAddr)>,
        blocked: impl IntoIterator<Item = LocationAddr>,
        max_expansions: Option<u32>,
        opts: &SolveOptions,
    ) -> Result<SolveResult, ConfigError> {
        let root = Config::new(initial)?;
        let target_pairs: Vec<(u32, LocationAddr)> = target.into_iter().collect();
        let blocked_locs: Vec<LocationAddr> = blocked.into_iter().collect();

        // Build goal predicate.
        let target_encoded: Vec<(u32, u64)> =
            target_pairs.iter().map(|&(q, l)| (q, l.encode())).collect();

        // Build distance table and heuristic (shared across restarts).
        let target_locs: Vec<u64> = target_encoded.iter().map(|&(_, l)| l).collect();
        let dist_table = if opts.w_t > 0.0 {
            DistanceTable::new(&target_locs, &self.index).with_time_distances(&self.index)
        } else {
            DistanceTable::new(&target_locs, &self.index)
        };
        let heuristic = HopDistanceHeuristic::new(target_pairs.iter().copied(), &dist_table);
        let h_max = |config: &Config| -> f64 { heuristic.estimate_max(config) };
        let h_sum = |config: &Config| -> f64 { heuristic.estimate_sum(config) };

        let goal_obj = AllAtTarget::new(&target_encoded);
        let blocked_encoded: std::collections::HashSet<u64> =
            blocked_locs.iter().map(|l| l.encode()).collect();
        let ctx = SearchContext {
            index: &self.index,
            dist_table: &dist_table,
            blocked: &blocked_encoded,
            targets: &target_encoded,
            cz_pairs: None,
        };

        let lookahead = opts.lookahead;
        let make_generator = |seed: u64, policy: DeadlockPolicy| {
            HeuristicGenerator::new()
                .with_deadlock_policy(policy)
                .with_lookahead(lookahead)
                .with_seed(seed)
        };

        Ok(run_with_components(
            root,
            &goal_obj,
            make_generator,
            h_max,
            h_sum,
            &ctx,
            max_expansions,
            opts,
        ))
    }

    /// Solve a loose-goal entangling placement + routing problem.
    ///
    /// Instead of fixed target locations, the solver receives CZ pair
    /// constraints and simultaneously discovers both the entangling
    /// placement and the routing. The goal is satisfied when every
    /// CZ pair occupies a valid entangling position (same zone,
    /// entangling word pair, same site).
    ///
    /// # Arguments
    ///
    /// * `initial` — Starting qubit positions: `(qubit_id, location)` pairs.
    /// * `cz_pairs` — Required CZ pairs: `(qubit_a, qubit_b)` that must
    ///   end up at entangling positions.
    /// * `blocked` — Locations occupied by external atoms (immovable obstacles).
    /// * `max_expansions` — Optional limit on node expansions.
    /// * `opts` — Search-tuning parameters (strategy, weight, restarts, etc.).
    ///
    /// # Returns
    ///
    /// A [`SolveResult`] whose [`goal_config`](SolveResult::goal_config)
    /// contains the discovered entangling placement.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] if `initial` contains duplicate qubit IDs.
    pub fn solve_entangling(
        &self,
        initial: impl IntoIterator<Item = (u32, LocationAddr)>,
        cz_pairs: &[(u32, u32)],
        blocked: impl IntoIterator<Item = LocationAddr>,
        max_expansions: Option<u32>,
        opts: &SolveOptions,
    ) -> Result<SolveResult, ConfigError> {
        use crate::goals::EntanglingConstraintGoal;
        use crate::heuristic::PairDistanceHeuristic;

        let root = Config::new(initial)?;
        let blocked_locs: Vec<LocationAddr> = blocked.into_iter().collect();
        let arch = self.index.arch_spec();

        // Reuse cached architecture-dependent data (built on first call).
        let cache = self.entangling_cache();
        let dist_table = cache.dist_table.clone(); // Arc clone (cheap)

        // Per-call: heuristic, goal, greedy assignment.
        let heuristic = PairDistanceHeuristic::new(cz_pairs, &cache.wpd);
        let h_max = |config: &Config| -> f64 { heuristic.estimate_max(config) };
        let h_sum = |config: &Config| -> f64 { heuristic.estimate_sum(config) };

        let goal = EntanglingConstraintGoal::new(cz_pairs, cache.ent_set.clone());

        let greedy_targets = entangling::greedy_assign_pairs(cz_pairs, &root, arch, &dist_table, 0);

        let blocked_encoded: HashSet<u64> = blocked_locs.iter().map(|l| l.encode()).collect();
        let ctx = SearchContext {
            index: &self.index,
            dist_table: &dist_table,
            blocked: &blocked_encoded,
            targets: &greedy_targets,
            cz_pairs: Some(cz_pairs),
        };

        let lookahead = opts.lookahead;
        // Loose-goal search needs at least MoveBlockers to handle blocking
        // between qubits competing for entangling positions. Upgrade Skip
        // to MoveBlockers; preserve AllMoves if explicitly requested.
        let deadlock_policy = match opts.deadlock_policy {
            DeadlockPolicy::Skip => DeadlockPolicy::MoveBlockers,
            other => other,
        };
        let opts = &SolveOptions {
            deadlock_policy,
            ..opts.clone()
        };

        let mut result = if opts.dynamic_targets {
            use crate::generators::LooseTargetGenerator;

            let arch_arc = Arc::new(arch.clone());
            let dt_arc = dist_table.clone(); // Arc clone (cheap)
            let recompute_interval = opts.recompute_interval;

            let cz_pairs_owned: Vec<(u32, u32)> = cz_pairs.to_vec();
            let make_generator = move |seed: u64, policy: DeadlockPolicy| {
                let inner = HeuristicGenerator::new()
                    .with_deadlock_policy(policy)
                    .with_lookahead(lookahead)
                    .with_top_c(3)
                    .with_seed(seed);
                LooseTargetGenerator::new(
                    inner,
                    cz_pairs_owned.clone(),
                    arch_arc.clone(),
                    dt_arc.clone(),
                    seed,
                    recompute_interval,
                )
            };

            run_with_components(
                root,
                &goal,
                make_generator,
                h_max,
                h_sum,
                &ctx,
                max_expansions,
                opts,
            )
        } else {
            let make_generator = |seed: u64, policy: DeadlockPolicy| {
                HeuristicGenerator::new()
                    .with_deadlock_policy(policy)
                    .with_lookahead(lookahead)
                    .with_top_c(3)
                    .with_seed(seed)
            };

            run_with_components(
                root,
                &goal,
                make_generator,
                h_max,
                h_sum,
                &ctx,
                max_expansions,
                opts,
            )
        };

        // Post-solve cleanup: move spectator qubits out of accidental CZ positions.
        if result.status == SolveStatus::Solved {
            let cz_qubit_set: HashSet<u32> = cz_pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
            let accidental = entangling::find_accidental_cz(
                &result.goal_config,
                &cz_qubit_set,
                &cache.partner_map,
            );

            if !accidental.is_empty() {
                let mut cleanup_targets: Vec<(u32, LocationAddr)> =
                    result.goal_config.iter().collect();

                for &(qid, move_loc) in &accidental {
                    for &lane in self.index.outgoing_lanes(move_loc) {
                        if let Some((_, dst)) = self.index.endpoints(&lane) {
                            if result.goal_config.is_occupied(dst) {
                                continue;
                            }
                            let safe = arch.get_cz_partner(&dst).is_none_or(|p| {
                                !result.goal_config.is_occupied(p)
                                    || cz_qubit_set.contains(
                                        &result.goal_config.qubit_at(p).unwrap_or(u32::MAX),
                                    )
                            });
                            if safe {
                                if let Some(entry) =
                                    cleanup_targets.iter_mut().find(|(q, _)| *q == qid)
                                {
                                    entry.1 = dst;
                                }
                                break;
                            }
                        }
                    }
                }

                let cleanup_result = self.solve(
                    result.goal_config.iter(),
                    cleanup_targets,
                    blocked_locs.iter().copied(),
                    max_expansions,
                    opts,
                );

                if let Ok(cleanup) = cleanup_result
                    && cleanup.status == SolveStatus::Solved
                {
                    result.move_layers.extend(cleanup.move_layers);
                    result.goal_config = cleanup.goal_config;
                    result.cost += cleanup.cost;
                    result.nodes_expanded += cleanup.nodes_expanded;
                }
            }
        }

        Ok(result)
    }
}

// ── Multi-candidate solve ──

/// Per-candidate debug info recorded during [`MoveSolver::solve_with_generator`].
#[derive(Debug, Clone)]
pub struct CandidateAttempt {
    /// Index of this candidate in the generator's output.
    pub candidate_index: usize,
    /// Outcome status of the solve attempt for this candidate.
    pub status: SolveStatus,
    /// Number of nodes expanded for this candidate.
    pub nodes_expanded: u32,
}

/// Result of a multi-candidate solve attempt via [`MoveSolver::solve_with_generator`].
#[derive(Debug)]
pub struct MultiSolveResult {
    /// The solve result from the winning candidate (or the last attempted).
    pub result: SolveResult,
    /// Index of the candidate that succeeded (`None` if all failed).
    pub candidate_index: Option<usize>,
    /// Total nodes expanded across all candidates.
    pub total_expansions: u32,
    /// Number of candidates actually attempted (excludes validation failures).
    pub candidates_tried: usize,
    /// Per-candidate attempt details for debugging.
    pub attempts: Vec<CandidateAttempt>,
}

impl MoveSolver {
    /// Solve using a target generator: generates candidates, validates each,
    /// and tries them in order with a shared expansion budget.
    ///
    /// Returns on the first successful solve, or the result of the last
    /// candidate if all fail or the budget runs out.
    #[allow(clippy::too_many_arguments)]
    pub fn solve_with_generator(
        &self,
        initial: impl IntoIterator<Item = (u32, LocationAddr)>,
        blocked: impl IntoIterator<Item = LocationAddr>,
        controls: &[u32],
        targets: &[u32],
        generator: &dyn TargetGenerator,
        max_expansions: Option<u32>,
        opts: &SolveOptions,
    ) -> Result<MultiSolveResult, ConfigError> {
        let initial_pairs: Vec<(u32, LocationAddr)> = initial.into_iter().collect();
        let blocked_locs: Vec<LocationAddr> = blocked.into_iter().collect();

        let ctx = TargetContext {
            placement: &initial_pairs,
            controls,
            targets,
            index: &self.index,
        };

        let candidates = generator.generate(&ctx);

        if candidates.is_empty() {
            let root = Config::new(initial_pairs.iter().copied())?;
            return Ok(MultiSolveResult {
                result: SolveResult {
                    status: SolveStatus::Unsolvable,
                    move_layers: Vec::new(),
                    goal_config: root,
                    nodes_expanded: 0,
                    cost: 0.0,
                    deadlocks: 0,
                    entropy_trace: None,
                },
                candidate_index: None,
                total_expansions: 0,
                candidates_tried: 0,
                attempts: Vec::new(),
            });
        }

        let mut total_expansions: u32 = 0;
        let mut remaining_budget = max_expansions;
        let mut last_result = None;
        let mut attempts = Vec::new();

        for (i, candidate) in candidates.iter().enumerate() {
            if validate_candidate(candidate, controls, targets, &self.index).is_err() {
                continue;
            }

            let result = self.solve(
                initial_pairs.iter().copied(),
                candidate.iter().copied(),
                blocked_locs.iter().copied(),
                remaining_budget,
                opts,
            )?;

            total_expansions += result.nodes_expanded;
            attempts.push(CandidateAttempt {
                candidate_index: i,
                status: result.status,
                nodes_expanded: result.nodes_expanded,
            });

            if result.status == SolveStatus::Solved {
                return Ok(MultiSolveResult {
                    result,
                    candidate_index: Some(i),
                    total_expansions,
                    candidates_tried: attempts.len(),
                    attempts,
                });
            }

            if let Some(budget) = remaining_budget.as_mut() {
                *budget = budget.saturating_sub(result.nodes_expanded);
                if *budget == 0 {
                    return Ok(MultiSolveResult {
                        result,
                        candidate_index: None,
                        total_expansions,
                        candidates_tried: attempts.len(),
                        attempts,
                    });
                }
            }

            last_result = Some(result);
        }

        let result = last_result.unwrap_or_else(|| {
            let root =
                Config::new(initial_pairs.iter().copied()).expect("initial was valid on entry");
            SolveResult {
                status: SolveStatus::Unsolvable,
                move_layers: Vec::new(),
                goal_config: root,
                nodes_expanded: 0,
                cost: 0.0,
                deadlocks: 0,
                entropy_trace: None,
            }
        });

        Ok(MultiSolveResult {
            result,
            candidate_index: None,
            total_expansions,
            candidates_tried: attempts.len(),
            attempts,
        })
    }

    /// Generate and validate candidate target configurations without solving.
    ///
    /// Useful for inspecting what a generator would produce.
    /// Returns only candidates that pass validation.
    pub fn generate_candidates(
        &self,
        initial: &[(u32, LocationAddr)],
        controls: &[u32],
        targets: &[u32],
        generator: &dyn TargetGenerator,
    ) -> Vec<Vec<(u32, LocationAddr)>> {
        let ctx = TargetContext {
            placement: initial,
            controls,
            targets,
            index: &self.index,
        };

        generator
            .generate(&ctx)
            .into_iter()
            .filter(|c| validate_candidate(c, controls, targets, &self.index).is_ok())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{example_arch_json, loc};

    /// Default test options: A*, w_t=0.0.
    fn default_opts() -> SolveOptions {
        SolveOptions {
            w_t: 0.0,
            ..SolveOptions::default()
        }
    }

    #[test]
    fn solve_simple_one_step() {
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        let result = solver
            .solve(
                [(0, loc(0, 0))],
                [(0, loc(0, 5))],
                std::iter::empty(),
                Some(100),
                &default_opts(),
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
                &default_opts(),
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
                &default_opts(),
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
                &default_opts(),
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
                &default_opts(),
            )
            .unwrap();

        assert_ne!(result.status, SolveStatus::Solved);
        assert!(result.move_layers.is_empty());
    }

    #[test]
    fn solver_reusable() {
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        let opts = default_opts();

        let r1 = solver
            .solve(
                [(0, loc(0, 0))],
                [(0, loc(0, 5))],
                std::iter::empty(),
                Some(100),
                &opts,
            )
            .unwrap();

        let r2 = solver
            .solve(
                [(0, loc(0, 5))],
                [(0, loc(0, 0))],
                std::iter::empty(),
                Some(100),
                &opts,
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
                &default_opts(),
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
                &default_opts(),
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
                &SolveOptions {
                    strategy: Strategy::Ids,
                    w_t: 0.0,
                    ..SolveOptions::default()
                },
            )
            .unwrap();

        let cascade_result = solver
            .solve(
                [(0, loc(0, 0))],
                [(0, loc(1, 5))],
                std::iter::empty(),
                Some(1000),
                &SolveOptions {
                    strategy: Strategy::Cascade {
                        inner: InnerStrategy::Ids,
                    },
                    w_t: 0.0,
                    ..SolveOptions::default()
                },
            )
            .unwrap();

        assert_eq!(ids_result.status, SolveStatus::Solved);
        assert_eq!(cascade_result.status, SolveStatus::Solved);
        assert!(cascade_result.cost <= ids_result.cost);
    }

    // ── solve_with_generator tests ──

    #[test]
    fn solve_with_generator_default_solves_cz() {
        use crate::target_generator::DefaultTargetGenerator;

        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        // Qubit 0 at word 0 site 0, qubit 1 at word 1 site 0.
        // CZ pair: word 0 ↔ word 1. DefaultTargetGenerator should produce
        // a candidate where qubit 0 stays at word 0 (CZ partner of word 1).
        let result = solver
            .solve_with_generator(
                [(0, loc(0, 0)), (1, loc(1, 0))],
                std::iter::empty(),
                &[0],
                &[1],
                &DefaultTargetGenerator,
                Some(1000),
                &default_opts(),
            )
            .unwrap();

        assert_eq!(result.result.status, SolveStatus::Solved);
        assert_eq!(result.candidate_index, Some(0));
        assert_eq!(result.candidates_tried, 1);
        assert_eq!(result.attempts.len(), 1);
    }

    #[test]
    fn solve_with_generator_empty_candidates() {
        use crate::target_generator::DefaultTargetGenerator;

        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        // Qubit 1 missing from placement — DefaultTargetGenerator returns empty.
        let result = solver
            .solve_with_generator(
                [(0, loc(0, 0))],
                std::iter::empty(),
                &[0],
                &[1],
                &DefaultTargetGenerator,
                Some(1000),
                &default_opts(),
            )
            .unwrap();

        assert_eq!(result.result.status, SolveStatus::Unsolvable);
        assert_eq!(result.candidate_index, None);
        assert_eq!(result.candidates_tried, 0);
        assert!(result.attempts.is_empty());
    }

    #[test]
    fn generate_candidates_returns_valid_only() {
        use crate::target_generator::DefaultTargetGenerator;

        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        let initial = vec![(0, loc(0, 0)), (1, loc(1, 0))];
        let candidates = solver.generate_candidates(&initial, &[0], &[1], &DefaultTargetGenerator);
        assert_eq!(candidates.len(), 1);
    }

    #[test]
    fn entropy_strategy_can_collect_trace() {
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        let result = solver
            .solve(
                [(0, loc(0, 0))],
                [(0, loc(0, 5))],
                std::iter::empty(),
                Some(100),
                &SolveOptions {
                    strategy: Strategy::Entropy,
                    w_t: 0.0,
                    collect_entropy_trace: true,
                    ..SolveOptions::default()
                },
            )
            .unwrap();

        assert_eq!(result.status, SolveStatus::Solved);
        let trace = result
            .entropy_trace
            .as_ref()
            .expect("entropy trace should be populated");
        assert_eq!(trace.root_node_id, 0);
        assert!(!trace.steps.is_empty(), "trace should include step events");
    }

    // ── solve_entangling tests ──

    #[test]
    fn solve_entangling_finds_solution() {
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        let result = solver
            .solve_entangling(
                [(0, loc(0, 0)), (1, loc(1, 0))],
                &[(0, 1)],
                std::iter::empty(),
                Some(5000),
                &default_opts(),
            )
            .unwrap();

        assert_eq!(result.status, SolveStatus::Solved);
        // Verify goal config satisfies the entangling constraint.
        let arch: bloqade_lanes_bytecode_core::arch::types::ArchSpec =
            serde_json::from_str(example_arch_json()).unwrap();
        let eset = crate::entangling::build_entangling_set(&arch);
        let loc_a = result.goal_config.location_of(0).unwrap().encode();
        let loc_b = result.goal_config.location_of(1).unwrap().encode();
        assert!(
            eset.contains(&(loc_a, loc_b)),
            "goal config should satisfy entangling constraint"
        );
    }

    #[test]
    fn solve_entangling_already_at_goal() {
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        // Qubits already at entangling positions.
        let result = solver
            .solve_entangling(
                [(0, loc(0, 5)), (1, loc(1, 5))],
                &[(0, 1)],
                std::iter::empty(),
                Some(100),
                &default_opts(),
            )
            .unwrap();

        assert_eq!(result.status, SolveStatus::Solved);
        assert_eq!(result.cost, 0.0);
        assert!(result.move_layers.is_empty());
    }

    #[test]
    fn solve_entangling_multiple_pairs() {
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        let result = solver
            .solve_entangling(
                [
                    (0, loc(0, 0)),
                    (1, loc(1, 0)),
                    (2, loc(0, 1)),
                    (3, loc(1, 1)),
                ],
                &[(0, 1), (2, 3)],
                std::iter::empty(),
                Some(10000),
                &default_opts(),
            )
            .unwrap();

        assert_eq!(result.status, SolveStatus::Solved);
        // Verify both pairs satisfy the constraint.
        let arch: bloqade_lanes_bytecode_core::arch::types::ArchSpec =
            serde_json::from_str(example_arch_json()).unwrap();
        let eset = crate::entangling::build_entangling_set(&arch);
        for &(qa, qb) in &[(0u32, 1u32), (2, 3)] {
            let la = result.goal_config.location_of(qa).unwrap().encode();
            let lb = result.goal_config.location_of(qb).unwrap().encode();
            assert!(
                eset.contains(&(la, lb)),
                "pair ({qa}, {qb}) should be at entangling positions"
            );
        }
    }

    #[test]
    fn solve_entangling_spectator_qubits() {
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        // q0/q1 are a CZ pair, q2 is a spectator (not in any pair).
        let result = solver
            .solve_entangling(
                [(0, loc(0, 0)), (1, loc(1, 0)), (2, loc(0, 3))],
                &[(0, 1)],
                std::iter::empty(),
                Some(5000),
                &default_opts(),
            )
            .unwrap();

        assert_eq!(result.status, SolveStatus::Solved);
        // Spectator q2 should remain at its initial position.
        assert_eq!(result.goal_config.location_of(2), Some(loc(0, 3)));
    }

    #[test]
    fn solve_entangling_with_ids() {
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        let result = solver
            .solve_entangling(
                [(0, loc(0, 0)), (1, loc(1, 0))],
                &[(0, 1)],
                std::iter::empty(),
                Some(5000),
                &SolveOptions {
                    strategy: Strategy::Ids,
                    w_t: 0.0,
                    ..SolveOptions::default()
                },
            )
            .unwrap();

        assert_eq!(result.status, SolveStatus::Solved);
    }

    #[test]
    fn solve_entangling_dynamic_targets() {
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        let result = solver
            .solve_entangling(
                [(0, loc(0, 0)), (1, loc(1, 0))],
                &[(0, 1)],
                std::iter::empty(),
                Some(5000),
                &SolveOptions {
                    dynamic_targets: true,
                    w_t: 0.0,
                    ..SolveOptions::default()
                },
            )
            .unwrap();

        assert_eq!(result.status, SolveStatus::Solved);
    }

    #[test]
    fn solve_entangling_with_cascade() {
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        let result = solver
            .solve_entangling(
                [(0, loc(0, 0)), (1, loc(1, 0))],
                &[(0, 1)],
                std::iter::empty(),
                Some(5000),
                &SolveOptions {
                    strategy: Strategy::Cascade {
                        inner: InnerStrategy::Ids,
                    },
                    w_t: 0.0,
                    ..SolveOptions::default()
                },
            )
            .unwrap();

        assert_eq!(result.status, SolveStatus::Solved);
    }
}
