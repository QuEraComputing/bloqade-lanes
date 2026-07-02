//! Strategy dispatch, restart orchestration, and `SearchResult →
//! SolveResult` extraction.
//!
//! Every solver entry point (`TargetSolver::solve` and the
//! `placement::*` drivers) delegates to
//! [`run_with_components`] after building its goal, heuristic, and
//! generator factory. The free helpers [`extract`] and [`pick_best`]
//! are the only places that translate raw frontier output into a
//! [`SolveResult`] — keeping them in one file ensures the
//! "Empty/Unsolvable SolveResult literal" pattern doesn't re-creep
//! back into the orchestration code.

use rayon::prelude::*;

use crate::cost::UniformCost;
use crate::drivers::entropy::EntropyTrace;
use crate::drivers::frontier::{BfsFrontier, DfsFrontier, Frontier, IdsFrontier, PriorityFrontier};
use crate::drivers::result::SearchResult;
use crate::generators::heuristic::DeadlockPolicy;
use crate::observer::NoOpObserver;
use crate::primitives::config::Config;
use crate::primitives::context::{SearchContext, SearchState};
use crate::scorers::DistanceScorer;
use crate::search::options::{EntropyOptions, InnerStrategy, SolveOptions, Strategy};
use crate::search::result::{SolveResult, SolveStatus};
use crate::traits::{Goal, Heuristic, MoveGenerator};

/// Extract a [`SolveResult`] from a [`SearchResult`].
pub(crate) fn extract(result: SearchResult, deadlocks: u32, max_exp: Option<u32>) -> SolveResult {
    match result.goal {
        Some(goal_id) => {
            let move_layers = result.solution_path().unwrap_or_default();
            let goal_config = result.graph.config(goal_id).clone();
            let cost = result.graph.g_score(goal_id);
            SolveResult::solved(
                goal_config,
                move_layers,
                cost,
                result.nodes_expanded,
                deadlocks,
            )
        }
        None => {
            let root_config = result.graph.config(result.graph.root()).clone();
            let status = if max_exp.is_some_and(|max| result.nodes_expanded >= max) {
                SolveStatus::BudgetExceeded
            } else {
                SolveStatus::Unsolvable
            };
            SolveResult::unsolved(status, root_config, result.nodes_expanded, deadlocks)
        }
    }
}

/// Pick the best result from multiple restarts (prefer solved, then lowest
/// cost). Returns `None` only when `results` is empty.
pub(crate) fn pick_best(results: Vec<SolveResult>) -> Option<SolveResult> {
    results.into_iter().min_by(|a, b| {
        let a_solved = a.status == SolveStatus::Solved;
        let b_solved = b.status == SolveStatus::Solved;
        b_solved.cmp(&a_solved).then(a.cost.total_cmp(&b.cost))
    })
}

/// Run the trait-based frontier search with the scorer, cost, state, and
/// observer fixed to the values every call site in this module uses
/// identically. Removes those four boilerplate arguments from
/// `frontier::run_search`.
fn run_frontier<Gen, Go, F>(
    root: &Config,
    generator: &Gen,
    goal: &Go,
    ctx: &SearchContext,
    frontier: &mut F,
    max_expansions: Option<u32>,
    max_depth: Option<u32>,
) -> SearchResult
where
    Gen: MoveGenerator,
    Go: Goal,
    F: Frontier,
{
    crate::drivers::frontier::run_search(
        root.clone(),
        generator,
        &DistanceScorer,
        &UniformCost,
        goal,
        frontier,
        ctx,
        &mut SearchState::default(),
        &mut NoOpObserver,
        max_expansions,
        max_depth,
    )
}

/// Shared strategy dispatch + restart logic.
///
/// Both [`solve_with_engine`](crate::search::target_solver::solve_with_engine)
/// and [`solve_loose_goal`](crate::placement::loose_goal::solve_loose_goal)
/// delegate here after constructing their specific goal, heuristic, and
/// generator.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_with_components<Go, Gen, Hmax, Hsum, MkGen>(
    root: Config,
    goal: &Go,
    make_generator: MkGen,
    h_max: Hmax,
    h_sum: Hsum,
    ctx: &SearchContext,
    max_expansions: Option<u32>,
    opts: &SolveOptions,
    entropy_opts: Option<&EntropyOptions>,
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
    let entropy_defaults = EntropyOptions::default();
    let entropy = entropy_opts.unwrap_or(&entropy_defaults);
    let max_movesets_per_group = entropy.max_movesets_per_group;
    let max_goal_candidates = entropy.max_goal_candidates;
    let collect_entropy_trace = entropy.collect_entropy_trace;
    let w_t = entropy.w_t;
    let base_seed = entropy.seed;
    let aod_grid_strategy = entropy.aod_grid_strategy;

    // Helper: run a single inner strategy with the given seed and budget.
    let run_inner = |inner: InnerStrategy, seed: u64, budget: Option<u32>| -> SolveResult {
        match inner {
            InnerStrategy::Ids => {
                let move_gen = make_generator(seed, deadlock_policy);
                let mut f = IdsFrontier::new(h_sum);
                let result = run_frontier(&root, &move_gen, goal, ctx, &mut f, budget, None);
                extract(result, move_gen.deadlock_count(), budget)
            }
            InnerStrategy::Dfs => {
                let move_gen = make_generator(seed, deadlock_policy);
                let mut f = DfsFrontier::new(h_sum);
                let result = run_frontier(&root, &move_gen, goal, ctx, &mut f, budget, None);
                extract(result, move_gen.deadlock_count(), budget)
            }
            InnerStrategy::Entropy => {
                let entropy_params = crate::drivers::entropy::EntropyParams {
                    max_movesets_per_group,
                    max_goal_candidates,
                    lookahead: opts.lookahead,
                    w_t,
                    aod_grid_strategy,
                    ..crate::drivers::entropy::EntropyParams::default()
                };
                let mut entropy_trace = if collect_entropy_trace {
                    Some(EntropyTrace::for_params(&entropy_params))
                } else {
                    None
                };
                let result = {
                    let mut noop = crate::observer::NoOpObserver;
                    let observer: &mut dyn crate::observer::SearchObserver =
                        match entropy_trace.as_mut() {
                            Some(trace) => trace,
                            None => &mut noop,
                        };
                    crate::drivers::entropy::entropy_search(
                        root.clone(),
                        goal,
                        &entropy_params,
                        ctx,
                        budget,
                        None,
                        seed,
                        observer,
                    )
                };
                let mut solve = extract(result, 0, budget);
                solve.entropy_trace = entropy_trace;
                solve
            }
        }
    };

    // Helper: run inner strategy with parallel restarts, return best.
    // Seed semantics:
    //   base_seed == 0, restarts == 1 → seed 0 (no perturbation, deterministic)
    //   base_seed == 0, restarts > 1  → seeds 1, 2, … (preserves pre-existing diversity)
    //   base_seed != 0, restarts == 1 → seed base_seed
    //   base_seed != 0, restarts > 1  → seeds base_seed, base_seed+1, …
    // base_seed.max(1) unifies the two multi-restart cases without an explicit branch.
    let run_inner_with_restarts = |inner: InnerStrategy| -> SolveResult {
        if restarts <= 1 {
            run_inner(inner, base_seed, max_expansions)
        } else {
            let start = base_seed.max(1);
            let results: Vec<SolveResult> = (0..restarts)
                .into_par_iter()
                .map(|i| run_inner(inner, start.saturating_add(i as u64), max_expansions))
                .collect();
            pick_best(results).expect("restarts > 1 yields a non-empty result set")
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
        let astar_result = run_frontier(
            &root,
            &astar_move_gen,
            goal,
            ctx,
            &mut astar_f,
            max_expansions,
            max_depth,
        );
        let astar_solve = extract(
            astar_result,
            astar_move_gen.deadlock_count(),
            max_expansions,
        );

        if astar_solve.status == SolveStatus::Solved {
            return pick_best(vec![inner_result, astar_solve])
                .expect("two-element vec is non-empty");
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
        run_once(base_seed, max_expansions)
    } else {
        let start = base_seed.max(1);
        let results: Vec<SolveResult> = (0..restarts)
            .into_par_iter()
            .map(|i| run_once(start.saturating_add(i as u64), max_expansions))
            .collect();
        pick_best(results).expect("restarts > 1 yields a non-empty result set")
    }
}

/// Dispatch to the appropriate frontier-based search strategy.
#[allow(clippy::too_many_arguments)]
fn run_strategy_v2<Go, Gen, Hmax>(
    strategy: Strategy,
    root: Config,
    generator: &Gen,
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
            run_frontier(&root, generator, goal, ctx, &mut f, max_expansions, None)
        }
        Strategy::Bfs => {
            let mut f = BfsFrontier::new();
            run_frontier(&root, generator, goal, ctx, &mut f, max_expansions, None)
        }
        Strategy::GreedyBestFirst => {
            let mut f = PriorityFrontier::greedy(heuristic_fn);
            run_frontier(&root, generator, goal, ctx, &mut f, max_expansions, None)
        }
        _ => {
            unreachable!("IDS/DFS/Cascade/Entropy handled before run_strategy_v2")
        }
    }
}
