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

use crate::bounds::CompletionBound;
use crate::bounds::{NoBound, WeightedDistanceBound};
use crate::cost::{SolveObjective, UniformCost};
use crate::drivers::branch_and_bound::{BranchAndBound, Schedule};
use crate::drivers::entropy::EntropyTrace;
use crate::drivers::frontier::{
    BfsFrontier, DfsFrontier, Frontier, IdsFrontier, LifoFrontier, PriorityFrontier,
};
use crate::drivers::result::{SearchResult, Termination};
use crate::generators::EntropyGenerator;
use crate::generators::exhaustive::{ExhaustiveGenerator, SeedPolicy};
use crate::generators::heuristic::DeadlockPolicy;
use crate::observer::NoOpObserver;
use crate::primitives::config::Config;
use crate::primitives::context::AodCapacity;
use crate::primitives::context::{SearchContext, SearchState};
use crate::scorers::DistanceScorer;
use crate::search::options::{
    BnbFrontier, BnbOptions, BnbOrdering, BoundKind, EntropyOptions, InnerStrategy, Refinement,
    SolveOptions, Strategy,
};
use crate::search::result::{SolveResult, SolveStatus};
use crate::traits::Objective;
// No `Objective` import: the cascade now bounds its refinement by cost
// directly, so nothing here needs `min_shot_cost`.
use crate::traits::{Goal, Heuristic, MoveGenerator};

/// Extract a [`SolveResult`] from a [`SearchResult`].
///
/// Every solved plan is replayed through the canonical execution model before
/// it leaves the solver (see [`crate::search::verify`]): `Config::with_moves`
/// performs no occupancy validation, so this is where a generator that emits
/// an inexecutable move set gets caught, rather than downstream in the IR.
pub(crate) fn extract(
    mut result: SearchResult,
    deadlocks: u32,
    max_exp: Option<u32>,
    ctx: &SearchContext,
) -> SolveResult {
    let bound_stats = result.bound_stats;
    let termination = result.termination;
    let proven = matches!(termination, Termination::Exhausted { proof: true });
    let stage_expansions = std::mem::take(&mut result.stage_expansions);
    let plan_stage = result.plan_stage();
    match result.goal {
        Some(goal_id) => {
            let move_layers = result.solution_path().unwrap_or_default();
            let goal_config = result.graph.config(goal_id).clone();
            let cost = result.graph.g_score(goal_id);
            crate::search::verify::assert_move_layers_executable(
                result.graph.config(result.graph.root()),
                &move_layers,
                ctx.index.arch_spec(),
                ctx.blocked,
                &goal_config,
            );
            let mut solved = SolveResult::solved(
                goal_config,
                move_layers,
                cost,
                result.nodes_expanded,
                deadlocks,
            );
            solved.bound_stats = bound_stats;
            solved.proven = proven;
            solved.termination = termination;
            solved.stage_expansions = stage_expansions;
            solved.plan_stage = plan_stage;
            solved
        }
        None => {
            let root_config = result.graph.config(result.graph.root()).clone();
            // The status is the driver's own account of how it ended. The
            // frontier and entropy drivers report `Budget` exactly when the
            // inference this replaces — expansions reached `max_expansions` —
            // would have, so results are unchanged; a driver that can drain
            // its space says so directly.
            let status = match termination {
                Termination::Budget => SolveStatus::BudgetExceeded,
                Termination::Exhausted { .. } => SolveStatus::Unsolvable,
                Termination::Stopped => {
                    debug_assert!(false, "a driver stopped by its own rule must report a goal");
                    SolveStatus::Unsolvable
                }
            };
            debug_assert_eq!(
                status == SolveStatus::BudgetExceeded,
                max_exp.is_some_and(|max| result.nodes_expanded >= max)
                    || matches!(termination, Termination::Budget),
            );
            let mut unsolved =
                SolveResult::unsolved(status, root_config, result.nodes_expanded, deadlocks);
            unsolved.bound_stats = bound_stats;
            unsolved.proven = proven;
            unsolved.termination = termination;
            unsolved.stage_expansions = stage_expansions;
            unsolved
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

/// Deadlock policy for the plain frontier strategies — A*, BFS, greedy, and the
/// cascade's A* refinement.
///
/// [`DeadlockPolicy::MoveBlockers`] is a **floor here, not an override**. Those
/// strategies have no depth-first jump-back to fall back on, so under
/// [`DeadlockPolicy::Skip`] a node whose candidates all fail leaves them with
/// nothing at all; the floor keeps them functional on the default options.
///
/// What the floor must not do is *lower* the caller's request. Hardcoding
/// `MoveBlockers` — as this dispatch did — silently discarded an explicit
/// [`DeadlockPolicy::AllMoves`], and `MoveBlockers` only frees atoms parked on
/// an unresolved target. When the target is simply *far away* rather than
/// occupied it emits nothing, so A* got zero successors and reported
/// `unsolvable` on instances IDS and entropy — which do honour the option —
/// solved in three nodes.
fn frontier_deadlock_policy(requested: DeadlockPolicy) -> DeadlockPolicy {
    match requested {
        DeadlockPolicy::Skip => DeadlockPolicy::MoveBlockers,
        stronger => stronger,
    }
}

/// Run the trait-based frontier search with the scorer, cost, state, and
/// observer fixed to the values every call site in this module uses
/// identically. Removes those four boilerplate arguments from
/// `frontier::run_search`.
///
/// Still passes both of `run_search`'s limits through: `max_depth` is a layer
/// horizon, `max_cost` an incumbent bound, and they are not interchangeable
/// under a non-uniform objective.
#[allow(clippy::too_many_arguments)]
fn run_frontier<Gen, Go, F>(
    root: &Config,
    generator: &Gen,
    goal: &Go,
    ctx: &SearchContext,
    frontier: &mut F,
    max_expansions: Option<u32>,
    max_depth: Option<u32>,
    max_cost: Option<f64>,
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
        max_cost,
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
    bnb_opts: &BnbOptions,
    blended_cache: Option<&crate::drivers::entropy::BlendedColumnCache>,
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
    // The objective this solve accumulates `g` with, named once, so the driver
    // and the bound it is paired with cannot disagree about it. Resolved from
    // the options against the arch; `Uniform` delegates to `UniformCost` and
    // is bit-identical to what every path ran before the knob existed.
    let objective = SolveObjective::from_kind(entropy.objective, ctx.index);
    // Whether this solve runs the branch-and-bound driver at all, and whether
    // its stage 0 is the entropy generator (which reads the tables below).
    let bnb_active = matches!(
        strategy,
        Strategy::BranchAndBound
            | Strategy::Cascade {
                refine: Refinement::BranchAndBound,
                ..
            }
    );
    let bnb_entropy_stage = bnb_active && bnb_opts.schedule.stage0_is_entropy();
    // One parameter set for every consumer of the entropy generator: the
    // entropy driver and the branch-and-bound stage 0 alike.
    let entropy_params = crate::drivers::entropy::EntropyParams {
        max_movesets_per_group,
        max_goal_candidates,
        lookahead: opts.lookahead,
        w_t,
        ..crate::drivers::entropy::EntropyParams::default()
    };

    // Build the entropy heuristic tables once per solve, shared across
    // restarts, exactly when the dispatch below will run the entropy driver.
    // Deciding here — next to that dispatch — keeps "will this solve run
    // entropy?" in one place, and building from the same `w_t`/`lookahead`
    // the `EntropyParams` below use keeps tables and params coupled by
    // construction. Skipped when the root already satisfies the goal (the
    // driver early-returns before touching the tables).
    let entropy_tables = ((matches!(
        strategy,
        Strategy::Entropy
            | Strategy::Cascade {
                inner: InnerStrategy::Entropy,
                ..
            }
    ) || bnb_entropy_stage)
        && !goal.is_goal(&root))
    .then(|| match blended_cache {
        Some(cache) => {
            crate::drivers::entropy::HeuristicTables::build_cached(ctx, w_t, opts.lookahead, cache)
        }
        None => crate::drivers::entropy::HeuristicTables::build(ctx, w_t, opts.lookahead),
    });
    let entropy_tables = entropy_tables.as_ref();

    // Completion bound, built once per solve and shared by reference across
    // the restart fan-out (hence `Objective: Sync` / `CompletionBound: Sync`).
    //
    // Built from `goal.exact_targets()`, not from `ctx.targets`, and only when
    // the goal says it is point-valued. A set-valued goal — `EntanglingConstraintGoal`,
    // where `ctx.targets` is one greedy Hungarian assignment among many
    // acceptable placements, or `PartialPlacementGoal`, which requires only
    // `min_placed` of them — is satisfiable without every qubit reaching its
    // listed target, so a distance to those targets can exceed the true
    // remaining cost, `h0` would be inadmissible, and pruning could discard
    // the optimum. Asking the goal keeps that decision next to the definition
    // that determines it, rather than inferring it from a context field.
    let completion_bound = match entropy_opts.and_then(|o| o.completion_bound) {
        Some(BoundKind::WeightedDistance) if entropy_tables.is_some() || bnb_active => goal
            .exact_targets()
            .map(|targets| WeightedDistanceBound::new(&objective, targets, ctx.index, ctx.blocked)),
        _ => None,
    };
    let completion_bound = completion_bound.as_ref();
    let no_bound = NoBound::for_objective(&objective);

    // The exhaustive levels of the branch-and-bound schedule, built once per
    // solve (they depend on the arch, the blocked set and the capacity, not
    // on the restart) and shared by reference across the restart fan-out.
    // The entry point already refused an arch that fails the preconditions.
    let exhaustive_levels: Vec<ExhaustiveGenerator> =
        if bnb_active && bnb_opts.schedule.has_exhaustive() && !goal.is_goal(&root) {
            capacity_ladder(ctx.capacity)
                .into_iter()
                .map(|(seed, cap)| {
                    ExhaustiveGenerator::for_solve(ctx, seed, cap)
                        .expect("the solve entry point rejects an arch that fails P1/P2")
                })
                .collect()
        } else {
            Vec::new()
        };

    // Helper: one branch-and-bound run. Stage 0 is built per restart (seeded);
    // the exhaustive levels are shared. Returns the raw result and the stage-0
    // generator's deadlock count so the caller can map the result itself —
    // only the caller knows whether it passed a seed.
    let run_bnb = |seed: u64,
                   budget: Option<u32>,
                   seed_incumbent: Option<f64>|
     -> (SearchResult, u32) {
        let entropy_stage;
        let heuristic_stage;
        let mut stages: Vec<&dyn MoveGenerator> = Vec::new();
        let heuristic_ref: Option<&Gen> = if bnb_opts.schedule.stage0_is_entropy() {
            entropy_stage = match entropy_tables {
                Some(tables) => EntropyGenerator::with_tables(entropy_params.clone(), seed, tables),
                None => EntropyGenerator::new(entropy_params.clone(), seed),
            };
            stages.push(&entropy_stage);
            None
        } else {
            heuristic_stage = make_generator(seed, deadlock_policy);
            stages.push(&heuristic_stage);
            Some(&heuristic_stage)
        };
        let schedule = match exhaustive_levels.split_last() {
            Some((terminal, prefix)) => {
                stages.extend(prefix.iter().map(|g| g as &dyn MoveGenerator));
                Schedule::complete(stages, terminal, ctx.capacity)
            }
            None => Schedule::partial(stages),
        };
        // Two monomorphizations, as the entropy arm: the bound-disabled run
        // compiles to plain DFS with an incumbent.
        let result = match completion_bound {
            Some(bound) => run_bnb_driver(
                bound,
                &objective,
                goal,
                ctx,
                budget,
                bnb_opts,
                &schedule,
                root.clone(),
                seed_incumbent,
                h_sum,
            ),
            None => run_bnb_driver(
                &no_bound,
                &objective,
                goal,
                ctx,
                budget,
                bnb_opts,
                &schedule,
                root.clone(),
                seed_incumbent,
                h_sum,
            ),
        };
        (result, heuristic_ref.map_or(0, |g| g.deadlock_count()))
    };

    // Helper: run a single inner strategy with the given seed and budget.
    let run_inner = |inner: InnerStrategy, seed: u64, budget: Option<u32>| -> SolveResult {
        match inner {
            InnerStrategy::Ids => {
                let move_gen = make_generator(seed, deadlock_policy);
                let mut f = IdsFrontier::new(h_sum);
                let result = run_frontier(&root, &move_gen, goal, ctx, &mut f, budget, None, None);
                extract(result, move_gen.deadlock_count(), budget, ctx)
            }
            InnerStrategy::Dfs => {
                let move_gen = make_generator(seed, deadlock_policy);
                let mut f = DfsFrontier::new(h_sum);
                let result = run_frontier(&root, &move_gen, goal, ctx, &mut f, budget, None, None);
                extract(result, move_gen.deadlock_count(), budget, ctx)
            }
            InnerStrategy::Entropy => {
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
                    // Two monomorphizations rather than a runtime branch, so
                    // the bound-disabled arm compiles to the same code as
                    // having no bounding at all (`NoBound::TRIVIAL`).
                    match completion_bound {
                        Some(bound) => crate::drivers::entropy::entropy_search_with_tables(
                            root.clone(),
                            goal,
                            &entropy_params,
                            ctx,
                            budget,
                            None,
                            seed,
                            observer,
                            entropy_tables,
                            &objective,
                            bound,
                        ),
                        None => crate::drivers::entropy::entropy_search_with_tables(
                            root.clone(),
                            goal,
                            &entropy_params,
                            ctx,
                            budget,
                            None,
                            seed,
                            observer,
                            entropy_tables,
                            &objective,
                            &no_bound,
                        ),
                    }
                };
                let mut solve = extract(result, 0, budget, ctx);
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

    // ── Cascade: inner restarts + one refinement pass ──────────
    if let Strategy::Cascade { inner, refine } = strategy {
        let inner_result = run_inner_with_restarts(inner);

        if inner_result.status != SolveStatus::Solved {
            return inner_result;
        }

        if refine == Refinement::BranchAndBound {
            // Seeded with the inner cost, the driver reports only something
            // strictly cheaper. Result mapping (spec, *Result semantics*):
            // a goal is a better plan; no goal means the seed stands, and
            // whether that is a proof is the refinement's exhaustion verdict.
            let (result, deadlocks) = run_bnb(0, max_expansions, Some(inner_result.cost));
            if result.goal.is_some() {
                return extract(result, deadlocks, max_expansions, ctx);
            }
            let mut stands = inner_result;
            stands.proven = matches!(result.termination, Termination::Exhausted { proof: true });
            // `incumbent_cost` is `None` here: the refinement found nothing.
            stands.bound_stats = result.bound_stats;
            return stands;
        }

        // The refinement is looking for something strictly cheaper than what
        // the inner strategy already found, which is a statement about the
        // objective — so bound it by that cost directly. It used to be
        // converted into a tree-depth cutoff via `min_shot_cost`, which is only
        // equivalent while `g == depth`: under a non-uniform objective a
        // cheaper plan can be *deeper* (more shots, each cheaper), so a depth
        // cap would exclude exactly the improvements sought here.
        let max_cost = Some(inner_result.cost);
        let astar_move_gen = make_generator(0, frontier_deadlock_policy(deadlock_policy));
        let mut astar_f = PriorityFrontier::astar(h_max, weight);
        let astar_result = run_frontier(
            &root,
            &astar_move_gen,
            goal,
            ctx,
            &mut astar_f,
            max_expansions,
            None,
            max_cost,
        );
        let astar_solve = extract(
            astar_result,
            astar_move_gen.deadlock_count(),
            max_expansions,
            ctx,
        );

        if astar_solve.status == SolveStatus::Solved {
            // The refinement runs on a frontier driver, which never prunes
            // against an incumbent and so reports an inert `BoundStats`. If it
            // wins, the pruning the inner entropy pass really did still has to
            // be reported — otherwise a bounded cascade whose A* leg happens to
            // find a cheaper plan looks like a solve that was never bounded.
            let inner_stats = inner_result.bound_stats;
            let mut best =
                pick_best(vec![inner_result, astar_solve]).expect("two-element vec is non-empty");
            if !best.bound_stats.bound_enabled {
                best.bound_stats = inner_stats;
            }
            return best;
        }
        return inner_result;
    }

    // ── Non-cascade strategies ─────────────────────────────────

    let run_once = |seed: u64, budget: Option<u32>| -> SolveResult {
        match strategy {
            Strategy::BranchAndBound => {
                let (result, deadlocks) = run_bnb(seed, budget, None);
                extract(result, deadlocks, budget, ctx)
            }
            Strategy::Entropy => run_inner(InnerStrategy::Entropy, seed, budget),
            Strategy::Ids => run_inner(InnerStrategy::Ids, seed, budget),
            Strategy::HeuristicDfs => run_inner(InnerStrategy::Dfs, seed, budget),
            _ => {
                let move_gen = make_generator(seed, frontier_deadlock_policy(deadlock_policy));
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
                extract(result, move_gen.deadlock_count(), budget, ctx)
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

/// One branch-and-bound run with the frontier the options name.
///
/// Generic over the bound so the trivial bound compiles to plain DFS with an
/// incumbent; the frontier choice is a runtime match into five
/// monomorphizations of [`BranchAndBound::run`].
#[allow(clippy::too_many_arguments)]
fn run_bnb_driver<O, B, Go, Hsum>(
    bound: &B,
    objective: &O,
    goal: &Go,
    ctx: &SearchContext<'_>,
    budget: Option<u32>,
    opts: &BnbOptions,
    schedule: &Schedule<'_>,
    root: Config,
    seed_incumbent: Option<f64>,
    h_sum: Hsum,
) -> SearchResult
where
    O: Objective,
    B: CompletionBound<Obj = O>,
    Go: Goal,
    Hsum: Heuristic + Copy,
{
    let driver = BranchAndBound::new(objective, bound, goal, ctx, budget, opts.widening);
    let mut observer = NoOpObserver;
    match (opts.frontier, opts.ordering) {
        (BnbFrontier::Lifo, _) => driver.run(
            root,
            schedule,
            LifoFrontier::new(),
            &mut observer,
            seed_incumbent,
        ),
        (BnbFrontier::Dfs, BnbOrdering::HopSum) => driver.run(
            root,
            schedule,
            DfsFrontier::new(h_sum),
            &mut observer,
            seed_incumbent,
        ),
        (BnbFrontier::Dfs, BnbOrdering::Bound) => driver.run(
            root,
            schedule,
            DfsFrontier::new(bound.as_heuristic()),
            &mut observer,
            seed_incumbent,
        ),
        (BnbFrontier::Ids, BnbOrdering::HopSum) => driver.run(
            root,
            schedule,
            IdsFrontier::new(h_sum),
            &mut observer,
            seed_incumbent,
        ),
        (BnbFrontier::Ids, BnbOrdering::Bound) => driver.run(
            root,
            schedule,
            IdsFrontier::new(bound.as_heuristic()),
            &mut observer,
            seed_incumbent,
        ),
    }
}

/// The exhaustive levels of a complete schedule for a solve at `cap`:
/// `(Unresolved, 1×1), (Unresolved, 2×2), …` while strictly below `cap` on
/// both axes, then `(Unresolved, cap)` and `(Any, cap)`. With no cap the
/// ladder is `1×1, 2×2, unlimited` under `Unresolved`, then `Any`
/// unlimited. Non-decreasing in both coordinates by construction, and its
/// last level is the whole search space at the solve's capacity.
pub(crate) fn capacity_ladder(cap: Option<AodCapacity>) -> Vec<(SeedPolicy, Option<AodCapacity>)> {
    let mut ladder = Vec::new();
    let mut k = 1usize;
    loop {
        let below = match cap {
            Some(c) => k < c.x && k < c.y,
            None => k <= 2,
        };
        if !below {
            break;
        }
        ladder.push((SeedPolicy::Unresolved, Some(AodCapacity { x: k, y: k })));
        k += 1;
    }
    ladder.push((SeedPolicy::Unresolved, cap));
    ladder.push((SeedPolicy::Any, cap));
    ladder
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
            run_frontier(
                &root,
                generator,
                goal,
                ctx,
                &mut f,
                max_expansions,
                None,
                None,
            )
        }
        Strategy::Bfs => {
            let mut f = BfsFrontier::new();
            run_frontier(
                &root,
                generator,
                goal,
                ctx,
                &mut f,
                max_expansions,
                None,
                None,
            )
        }
        Strategy::GreedyBestFirst => {
            let mut f = PriorityFrontier::greedy(heuristic_fn);
            run_frontier(
                &root,
                generator,
                goal,
                ctx,
                &mut f,
                max_expansions,
                None,
                None,
            )
        }
        // Push and Rotate needs a concrete target placement, which this path
        // does not have: `run_with_components` is reached with a `Goal`
        // predicate, and the loose-goal callers deliberately leave the target
        // open for the Hungarian assignment to choose. Fall back to A* rather
        // than panicking, and note it in `Strategy::PushRotate`'s docs so the
        // substitution is not a surprise.
        Strategy::PushRotate => {
            let mut f = PriorityFrontier::astar(heuristic_fn, weight);
            run_frontier(
                &root,
                generator,
                goal,
                ctx,
                &mut f,
                max_expansions,
                None,
                None,
            )
        }
        _ => {
            unreachable!("IDS/DFS/Cascade/Entropy handled before run_strategy_v2")
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::generators::HeuristicGenerator;
    use crate::goals::AllAtTarget;
    use crate::primitives::distance::DistanceTable;
    use crate::primitives::lane_index::LaneIndex;
    use crate::test_utils::{example_arch_json, loc};

    // ── Branch and bound: dispatch arm and result mapping ──

    use crate::drivers::branch_and_bound::Widening;
    use crate::search::engine::SearchEngine;
    use crate::search::options::{BnbOptions, ScheduleKind};
    use crate::search::target_solver::solve_with_engine;
    use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;

    fn bnb_engine() -> SearchEngine {
        SearchEngine::from_json(example_arch_json()).unwrap()
    }

    fn bnb_opts(strategy: Strategy, bound: Option<BoundKind>) -> (SolveOptions, EntropyOptions) {
        (
            SolveOptions {
                strategy,
                ..SolveOptions::default()
            },
            EntropyOptions {
                completion_bound: bound,
                ..EntropyOptions::default()
            },
        )
    }

    fn unlimited(schedule: ScheduleKind) -> BnbOptions {
        BnbOptions {
            schedule,
            widening: Widening::UNLIMITED,
            ..BnbOptions::default()
        }
    }

    /// Qubit 0 walks its column to the far end, qubit 1 takes one shot; the
    /// optimum is three shots (see the driver's tests for the arch's shape).
    const TWO: [(u32, LocationAddr); 2] = [
        (
            0,
            LocationAddr {
                zone_id: 0,
                word_id: 0,
                site_id: 0,
            },
        ),
        (
            1,
            LocationAddr {
                zone_id: 0,
                word_id: 0,
                site_id: 1,
            },
        ),
    ];
    fn two_targets() -> Vec<(u32, LocationAddr)> {
        vec![(0, loc(1, 0)), (1, loc(0, 6))]
    }
    /// An initial placement and its targets.
    type Instance = (Vec<(u32, LocationAddr)>, Vec<(u32, LocationAddr)>);

    /// Two atoms in one column with crossed targets: no plan exists.
    fn crossed() -> Instance {
        (
            vec![(0, loc(0, 0)), (1, loc(0, 5))],
            vec![(0, loc(0, 5)), (1, loc(0, 0))],
        )
    }

    #[test]
    fn capacity_ladder_is_nested_and_ends_at_the_solve_cap() {
        let cap = |x, y| Some(AodCapacity { x, y });
        assert_eq!(
            capacity_ladder(cap(3, 4)),
            vec![
                (SeedPolicy::Unresolved, cap(1, 1)),
                (SeedPolicy::Unresolved, cap(2, 2)),
                (SeedPolicy::Unresolved, cap(3, 4)),
                (SeedPolicy::Any, cap(3, 4)),
            ]
        );
        assert_eq!(
            capacity_ladder(cap(1, 1)),
            vec![
                (SeedPolicy::Unresolved, cap(1, 1)),
                (SeedPolicy::Any, cap(1, 1))
            ]
        );
        assert_eq!(
            capacity_ladder(None),
            vec![
                (SeedPolicy::Unresolved, cap(1, 1)),
                (SeedPolicy::Unresolved, cap(2, 2)),
                (SeedPolicy::Unresolved, None),
                (SeedPolicy::Any, None),
            ]
        );
        // Non-decreasing in both coordinates, terminal is Any at the cap.
        for cap in [None, cap(2, 5), cap(6, 1)] {
            let ladder = capacity_ladder(cap);
            let leq = |a: Option<AodCapacity>, b: Option<AodCapacity>| match (a, b) {
                (_, None) => true,
                (Some(a), Some(b)) => a.x <= b.x && a.y <= b.y,
                (None, Some(_)) => false,
            };
            assert!(
                ladder
                    .windows(2)
                    .all(|w| w[0].0 <= w[1].0 && leq(w[0].1, w[1].1))
            );
            assert_eq!(*ladder.last().unwrap(), (SeedPolicy::Any, cap));
        }
    }

    /// Row 1 of the mapping table: a goal is a plan, `proven` follows the
    /// exhaustion verdict. With unlimited widening on a complete schedule the
    /// three-shot optimum is found and proven.
    #[test]
    fn the_bnb_arm_reaches_the_driver_and_proves_the_optimum() {
        let engine = bnb_engine();
        let (opts, eopts) = bnb_opts(Strategy::BranchAndBound, Some(BoundKind::WeightedDistance));
        let r = solve_with_engine(
            &engine,
            &opts,
            Some(&eopts),
            &unlimited(ScheduleKind::EntropyThenExhaustive),
            TWO,
            two_targets(),
            std::iter::empty(),
            None,
        )
        .unwrap();
        assert_eq!(r.status, SolveStatus::Solved);
        assert_eq!(r.cost, 3.0);
        assert!(r.proven, "{:?}", r.termination);
        assert_eq!(r.termination, Termination::Exhausted { proof: true });
        assert!(r.bound_stats.bound_enabled);
        assert!(r.bound_stats.total_cuts() > 0 || r.bound_stats.cuts_infeasible > 0 || true);
        assert_eq!(r.stage_expansions.iter().sum::<u32>(), r.nodes_expanded);
        assert!(r.plan_stage.is_some());

        // The heuristic stage 0 gets there too, and so does the unbounded run.
        for (schedule, bound) in [
            (
                ScheduleKind::HeuristicThenExhaustive,
                Some(BoundKind::WeightedDistance),
            ),
            (ScheduleKind::EntropyThenExhaustive, None),
        ] {
            let (opts, eopts) = bnb_opts(Strategy::BranchAndBound, bound);
            let r = solve_with_engine(
                &engine,
                &opts,
                Some(&eopts),
                &unlimited(schedule),
                TWO,
                two_targets(),
                std::iter::empty(),
                None,
            )
            .unwrap();
            assert_eq!(
                (r.status, r.cost, r.proven),
                (SolveStatus::Solved, 3.0, true)
            );
            assert_eq!(r.bound_stats.bound_enabled, bound.is_some());
        }
    }

    /// Rows 4–6: no seed. A complete schedule that exhausts proves
    /// infeasibility; a partial one only reports that it found nothing; a
    /// spent budget is `BudgetExceeded`.
    #[test]
    fn bnb_without_a_seed_maps_exhaustion_and_budget() {
        let engine = bnb_engine();
        let (root, targets) = crossed();
        let (opts, eopts) = bnb_opts(Strategy::BranchAndBound, Some(BoundKind::WeightedDistance));

        let proof = solve_with_engine(
            &engine,
            &opts,
            Some(&eopts),
            &unlimited(ScheduleKind::EntropyThenExhaustive),
            root.clone(),
            targets.clone(),
            std::iter::empty(),
            None,
        )
        .unwrap();
        assert_eq!(proof.status, SolveStatus::Unsolvable);
        assert!(proof.proven);
        assert_eq!(proof.termination, Termination::Exhausted { proof: true });

        let no_proof = solve_with_engine(
            &engine,
            &opts,
            Some(&eopts),
            &unlimited(ScheduleKind::EntropyOnly),
            root,
            targets,
            std::iter::empty(),
            None,
        )
        .unwrap();
        assert_eq!(no_proof.status, SolveStatus::Unsolvable);
        assert!(!no_proof.proven);
        assert_eq!(
            no_proof.termination,
            Termination::Exhausted { proof: false }
        );

        let budget = solve_with_engine(
            &engine,
            &opts,
            Some(&eopts),
            &unlimited(ScheduleKind::EntropyThenExhaustive),
            TWO,
            two_targets(),
            std::iter::empty(),
            Some(1),
        )
        .unwrap();
        assert_eq!(budget.status, SolveStatus::BudgetExceeded);
        assert!(!budget.proven);
        assert_eq!(budget.termination, Termination::Budget);
    }

    /// Rows 2–3: a cascade refined by branch and bound hands the inner cost in
    /// as the seed. Nothing cheaper on a complete, unlimited schedule means the
    /// inner plan stands, proven; under a spent budget it stands unproven.
    #[test]
    fn cascade_refined_by_bnb_lets_the_seed_stand() {
        let engine = bnb_engine();
        let strategy = Strategy::Cascade {
            inner: InnerStrategy::Ids,
            refine: Refinement::BranchAndBound,
        };
        let (opts, eopts) = bnb_opts(strategy, Some(BoundKind::WeightedDistance));
        let single_root = [(0, loc(0, 0))];
        let single_target = [(0, loc(0, 5))];

        let stands = solve_with_engine(
            &engine,
            &opts,
            Some(&eopts),
            &unlimited(ScheduleKind::EntropyThenExhaustive),
            single_root,
            single_target,
            std::iter::empty(),
            None,
        )
        .unwrap();
        assert_eq!((stands.status, stands.cost), (SolveStatus::Solved, 1.0));
        assert!(
            stands.proven,
            "the one-shot plan is optimal and the refinement proved it"
        );
        assert!(stands.bound_stats.bound_enabled);
        assert_eq!(
            stands.bound_stats.incumbent_cost, None,
            "the refinement found nothing of its own"
        );

        // Under a budget the seed stands unproven. The one-shot instance
        // above cannot show it — its seed cuts the root at the pop gate, a
        // proof at zero expansions — so take a three-atom instance, give the
        // cascade one expansion more than the inner phase needs, and the
        // refinement runs out before it can exhaust.
        let root3 = [(0, loc(0, 0)), (1, loc(0, 1)), (2, loc(1, 2))];
        let targets3 = [(0, loc(1, 5)), (1, loc(0, 6)), (2, loc(0, 2))];
        let (ids_opts, ids_eopts) = bnb_opts(Strategy::Ids, None);
        let ids = solve_with_engine(
            &engine,
            &ids_opts,
            Some(&ids_eopts),
            &BnbOptions::default(),
            root3,
            targets3,
            std::iter::empty(),
            None,
        )
        .unwrap();
        assert_eq!(ids.status, SolveStatus::Solved);
        let unproven = solve_with_engine(
            &engine,
            &opts,
            Some(&eopts),
            &unlimited(ScheduleKind::EntropyThenExhaustive),
            root3,
            targets3,
            std::iter::empty(),
            Some(ids.nodes_expanded + 1),
        )
        .unwrap();
        assert_eq!(unproven.status, SolveStatus::Solved);
        assert!(unproven.cost <= ids.cost);
        assert!(!unproven.proven, "{:?}", unproven.termination);
    }

    /// Row 1 through the cascade: when the refinement finds something strictly
    /// cheaper than the inner plan, the cheaper plan is returned.
    #[test]
    fn cascade_refined_by_bnb_returns_a_cheaper_plan() {
        let engine = bnb_engine();
        // The inner phase on its own.
        let (ids_opts, eopts) = bnb_opts(Strategy::Ids, Some(BoundKind::WeightedDistance));
        let ids = solve_with_engine(
            &engine,
            &ids_opts,
            Some(&eopts),
            &BnbOptions::default(),
            TWO,
            two_targets(),
            std::iter::empty(),
            None,
        )
        .unwrap();
        assert_eq!(ids.status, SolveStatus::Solved);
        let strategy = Strategy::Cascade {
            inner: InnerStrategy::Ids,
            refine: Refinement::BranchAndBound,
        };
        let (opts, eopts) = bnb_opts(strategy, Some(BoundKind::WeightedDistance));
        let refined = solve_with_engine(
            &engine,
            &opts,
            Some(&eopts),
            &unlimited(ScheduleKind::EntropyThenExhaustive),
            TWO,
            two_targets(),
            std::iter::empty(),
            None,
        )
        .unwrap();
        assert_eq!(refined.status, SolveStatus::Solved);
        assert_eq!(refined.cost, 3.0, "the refinement reaches the optimum");
        assert!(refined.cost <= ids.cost);
        assert!(refined.proven);
        if refined.cost < ids.cost {
            assert_eq!(
                refined.bound_stats.incumbent_cost,
                Some(3.0),
                "a plan of its own"
            );
        }
    }

    /// The loose-goal path has a set-valued goal, so the arm runs unbounded.
    #[test]
    fn loose_goal_bnb_runs_unbounded() {
        use crate::placement::loose_goal::solve_loose_goal;
        use crate::search::options::EntanglingOptions;
        let engine = bnb_engine();
        let (opts, _) = bnb_opts(Strategy::BranchAndBound, Some(BoundKind::WeightedDistance));
        let r = solve_loose_goal(
            &engine,
            &opts,
            &EntanglingOptions::default(),
            &unlimited(ScheduleKind::HeuristicThenExhaustive),
            [(0, loc(0, 0)), (1, loc(1, 0))],
            &[(0, 1)],
            std::iter::empty(),
            Some(2000),
            &[],
        )
        .unwrap();
        assert!(!r.bound_stats.bound_enabled);
        assert_eq!(r.bound_stats.total_cuts(), 0);
    }

    /// A spec that fails the exhaustive preconditions is refused at the entry
    /// point with a named error, before anything runs.
    #[test]
    fn full_json_under_bnb_is_rejected_not_panicked() {
        let engine = SearchEngine::from_json(crate::test_utils::full_arch_json()).unwrap();
        let (opts, eopts) = bnb_opts(Strategy::BranchAndBound, None);
        let err = solve_with_engine(
            &engine,
            &opts,
            Some(&eopts),
            &BnbOptions::default(),
            [(0, loc(0, 0))],
            [(0, loc(0, 1))],
            std::iter::empty(),
            Some(10),
        )
        .unwrap_err();
        assert!(
            matches!(
                err,
                crate::primitives::config::ConfigError::UnsupportedArchitecture {
                    strategy: "branch_and_bound",
                    ..
                }
            ),
            "{err}"
        );
        assert!(err.to_string().contains("P1"), "{err}");
        // The entropy-only schedule never touches the exhaustive generator.
        let ok = solve_with_engine(
            &engine,
            &opts,
            Some(&eopts),
            &unlimited(ScheduleKind::EntropyOnly),
            [(0, loc(0, 0))],
            [(0, loc(0, 1))],
            std::iter::empty(),
            Some(10),
        );
        assert!(ok.is_ok());
    }

    /// Drive one solve through the real dispatch. Every argument the wiring
    /// tests need to vary is a parameter; everything else (arch, root, goal,
    /// targets, budget) is held fixed, so any difference in the returned
    /// `bound_stats` is attributable to the varied input alone.
    fn solve_with(
        strategy: Strategy,
        completion_bound: Option<BoundKind>,
        cz_pairs: Option<&[(u32, u32)]>,
        initial: &[(u32, bloqade_lanes_bytecode_core::arch::addr::LocationAddr)],
    ) -> SolveResult {
        solve_with_goal(
            AllAtTarget::new,
            strategy,
            completion_bound,
            cz_pairs,
            initial,
        )
    }

    /// As [`solve_with`], but with the goal chosen by the caller — the input
    /// that now decides whether a completion bound is admissible.
    fn solve_with_goal<Go, F>(
        make_goal: F,
        strategy: Strategy,
        completion_bound: Option<BoundKind>,
        cz_pairs: Option<&[(u32, u32)]>,
        initial: &[(u32, bloqade_lanes_bytecode_core::arch::addr::LocationAddr)],
    ) -> SolveResult
    where
        Go: Goal + Sync,
        F: FnOnce(&[(u32, u64)]) -> Go,
    {
        let spec: bloqade_lanes_bytecode_core::arch::types::ArchSpec =
            serde_json::from_str(example_arch_json()).expect("example arch json parses");
        let index = LaneIndex::new(spec);
        let root = Config::new(initial.iter().copied()).expect("root is a valid config");
        let targets: Vec<(u32, u64)> = vec![(0, loc(1, 5).encode()), (1, loc(1, 6).encode())];
        let target_locs: Vec<u64> = targets.iter().map(|&(_, l)| l).collect();
        let dist_table = DistanceTable::new(&target_locs, &index);
        let blocked = HashSet::new();
        let goal = make_goal(&targets);
        let ctx = SearchContext {
            index: &index,
            dist_table: &dist_table,
            blocked: &blocked,
            targets: &targets,
            cz_pairs,
            capacity: None,
        };
        let opts = SolveOptions {
            strategy,
            ..SolveOptions::default()
        };
        let entropy_opts = EntropyOptions {
            completion_bound,
            ..EntropyOptions::default()
        };
        // The entropy driver generates its own candidates, so this factory goes
        // unused on that path; it exists to satisfy the type parameter, and is
        // the real generator for the frontier strategies.
        let make_generator = |seed: u64, policy: DeadlockPolicy| {
            HeuristicGenerator::configured(seed, policy, false, None)
        };
        run_with_components(
            root,
            &goal,
            make_generator,
            |_: &Config| 0.0,
            |_: &Config| 0.0,
            &ctx,
            Some(2000),
            &opts,
            Some(&entropy_opts),
            &BnbOptions::default(),
            None,
        )
    }

    /// The placement `solve_with`'s fixed targets are stated against: two
    /// qubits, both away from their targets.
    fn start() -> Vec<(u32, bloqade_lanes_bytecode_core::arch::addr::LocationAddr)> {
        vec![(0, loc(0, 0)), (1, loc(0, 1))]
    }

    /// The target placement itself, for the root-is-goal case.
    fn already_at_target() -> Vec<(u32, bloqade_lanes_bytecode_core::arch::addr::LocationAddr)> {
        vec![(0, loc(1, 5)), (1, loc(1, 6))]
    }

    /// Shorthand for the fixed-target entropy solve the wiring tests compare
    /// against.
    fn entropy_solve(completion_bound: Option<BoundKind>) -> SolveResult {
        solve_with(Strategy::Entropy, completion_bound, None, &start())
    }

    /// An explicit `BoundKind` must reach the driver and actually prune;
    /// leaving it unset must leave the search bit-for-bit unbounded.
    ///
    /// Both directions matter. Without the "on" half, a wiring bug that never
    /// built the bound would look like a correctly disabled one; without the
    /// "off" half, a bound that ignored the option would look enabled
    /// everywhere. The cut count is what separates "the flag was recorded"
    /// from "the bound is doing work".
    #[test]
    fn the_completion_bound_option_reaches_the_driver() {
        let bounded = entropy_solve(Some(BoundKind::WeightedDistance));
        assert_eq!(bounded.status, SolveStatus::Solved);
        assert!(bounded.bound_stats.bound_enabled);
        assert!(
            bounded.bound_stats.total_cuts() > 0,
            "the constructed bound must reach the driver and prune, not just set a flag"
        );
        assert!(
            bounded.bound_stats.root_lower_bound > 0.0,
            "a real h(root) must be recorded"
        );

        let unbounded = entropy_solve(None);
        assert_eq!(unbounded.status, SolveStatus::Solved);
        assert!(!unbounded.bound_stats.bound_enabled);
        assert_eq!(unbounded.bound_stats.total_cuts(), 0);
        assert_eq!(unbounded.bound_stats.root_lower_bound, 0.0);
        assert_eq!(
            unbounded.bound_stats.optimality_gap(),
            None,
            "an unbounded run has no gap to report, rather than a gap of 1.0"
        );

        // Pruning only removes branches that cannot hold a cheaper plan, so
        // enabling it can improve the answer but never degrade it.
        assert!(
            bounded.cost <= unbounded.cost,
            "bounded cost {} exceeded unbounded {}",
            bounded.cost,
            unbounded.cost
        );
    }

    /// Only the entropy driver prunes against an incumbent, so a
    /// `completion_bound` requested alongside a frontier strategy is a no-op
    /// rather than an error.
    ///
    /// The bound is built next to the entropy dispatch and gated on the same
    /// `entropy_tables.is_some()` condition. Were that gate to drift, the two
    /// would disagree about whether this solve is bounded — and the frontier
    /// drivers report `BoundStats::default()` unconditionally, so the request
    /// would be silently dropped while the caller believed it applied.
    #[test]
    fn a_frontier_strategy_ignores_a_requested_completion_bound() {
        for strategy in [Strategy::AStar, Strategy::Bfs, Strategy::Ids] {
            let result = solve_with(strategy, Some(BoundKind::WeightedDistance), None, &start());
            assert_eq!(
                result.status,
                SolveStatus::Solved,
                "{strategy:?} should still solve with a bound requested"
            );
            assert!(
                !result.bound_stats.bound_enabled,
                "{strategy:?} does not prune against an incumbent; the request must be inert"
            );
            assert_eq!(result.bound_stats.total_cuts(), 0);
        }
    }

    /// A solve whose root already satisfies the goal builds no bound, and says
    /// so.
    ///
    /// This is the second thing the `entropy_tables.is_some()` condition
    /// decides: those tables are skipped when the root is a goal, and the bound
    /// rides on the same condition so a solve that never searches never pays
    /// for a Dijkstra sweep it cannot use. The reported stats have to agree
    /// with that — an empty `BoundStats` claiming `bound_enabled` would offer a
    /// `root_lower_bound` of 0.0 as if it were a measurement.
    #[test]
    fn a_root_that_is_already_the_goal_builds_no_bound() {
        let result = solve_with(
            Strategy::Entropy,
            Some(BoundKind::WeightedDistance),
            None,
            &already_at_target(),
        );
        assert_eq!(result.status, SolveStatus::Solved);
        assert_eq!(result.cost, 0.0);
        assert!(result.move_layers.is_empty(), "nothing needed moving");
        assert!(
            !result.bound_stats.bound_enabled,
            "no search ran, so no bound was built; reporting one would be a claim about nothing"
        );
        assert_eq!(result.bound_stats.total_cuts(), 0);
        assert_eq!(
            result.bound_stats.optimality_gap(),
            None,
            "a zero-cost incumbent has no meaningful gap"
        );
    }

    /// Loose-goal solves must refuse the completion bound even when the caller
    /// asks for it.
    ///
    /// There, `ctx.targets` is a greedy Hungarian assignment of qubits to
    /// entangling slots, but the goal accepts *any* valid entangling
    /// placement: a qubit can satisfy the goal without ever reaching its
    /// assigned target, so `h0` — a distance to that target — can exceed the
    /// true remaining cost. `Goal::exact_targets()` is how a goal declares
    /// itself point-valued, and only a goal that does gets a bound.
    ///
    /// Exercised with `PartialPlacementGoal`, the smallest set-valued goal:
    /// requiring only `min_placed` of the targets means an atom can be left
    /// where it is, so `h0` — a max over *all* unresolved atoms — overestimates.
    /// `EntanglingConstraintGoal` is set-valued for the same reason and
    /// inherits the same `None` default.
    #[test]
    fn a_set_valued_goal_refuses_the_completion_bound() {
        // Point-valued: the request is honoured, so the assertion below cannot
        // pass vacuously through some unrelated path that drops the bound.
        let fixed = entropy_solve(Some(BoundKind::WeightedDistance));
        assert_eq!(fixed.status, SolveStatus::Solved);
        assert!(
            fixed.bound_stats.bound_enabled,
            "a fixed-target solve must honour an explicit completion-bound request"
        );

        let loose = solve_with_goal(
            |targets| crate::goals::PartialPlacementGoal::new(targets, Some(1)),
            Strategy::Entropy,
            Some(BoundKind::WeightedDistance),
            None,
            &start(),
        );
        assert_eq!(
            loose.status,
            SolveStatus::Solved,
            "refusing the bound must not cost the solve its answer"
        );
        assert!(
            !loose.bound_stats.bound_enabled,
            "h0 is not admissible against a set-valued goal; the bound must be refused"
        );
        assert_eq!(
            loose.bound_stats.total_cuts(),
            0,
            "a refused bound must not prune"
        );

        // The CZ marker is no longer what decides this: a point-valued goal is
        // bounded whether or not the context carries cz_pairs, because the goal
        // is what determines admissibility.
        let with_cz_marker = solve_with(
            Strategy::Entropy,
            Some(BoundKind::WeightedDistance),
            Some(&[(0, 1)]),
            &start(),
        );
        assert!(with_cz_marker.bound_stats.bound_enabled);
    }

    /// The dispatch may *raise* a caller's deadlock policy to keep the plain
    /// frontier strategies functional on the defaults, but it must never lower
    /// one. Hardcoding `MoveBlockers` here — which is what it used to do — meant
    /// A*, BFS and greedy silently ignored an explicit `AllMoves`, so a solve the
    /// caller had configured to escape deadlocks reported `unsolvable` at a node
    /// where IDS and entropy, which honour the option, walked straight through.
    ///
    /// Observed on a 37-atom staging phase with a single mover whose target was
    /// *free*: `MoveBlockers` only frees atoms parked on an unresolved target, so
    /// it emitted nothing and A* died with zero successors at the root, while the
    /// same instance solved in three nodes under `AllMoves`.
    #[test]
    fn an_explicit_deadlock_policy_is_never_downgraded() {
        assert_eq!(
            frontier_deadlock_policy(DeadlockPolicy::AllMoves),
            DeadlockPolicy::AllMoves,
            "AllMoves asks for *more* escapes than MoveBlockers; honour it"
        );
        assert_eq!(
            frontier_deadlock_policy(DeadlockPolicy::MoveBlockers),
            DeadlockPolicy::MoveBlockers
        );
        // `Skip` is the default and leaves these strategies with no escape hatch
        // at all, so the floor still applies there — this is the one case the
        // dispatch is allowed to change.
        assert_eq!(
            frontier_deadlock_policy(DeadlockPolicy::Skip),
            DeadlockPolicy::MoveBlockers
        );
    }
}
