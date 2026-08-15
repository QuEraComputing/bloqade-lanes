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

use crate::bounds::{NoBound, WeightedDistanceBound};
use crate::cost::UniformCost;
use crate::drivers::entropy::EntropyTrace;
use crate::drivers::frontier::{BfsFrontier, DfsFrontier, Frontier, IdsFrontier, PriorityFrontier};
use crate::drivers::result::SearchResult;
use crate::generators::heuristic::DeadlockPolicy;
use crate::observer::NoOpObserver;
use crate::primitives::config::Config;
use crate::primitives::context::{SearchContext, SearchState};
use crate::scorers::DistanceScorer;
use crate::search::options::{BoundKind, EntropyOptions, InnerStrategy, SolveOptions, Strategy};
use crate::search::result::{SolveResult, SolveStatus};
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
    result: SearchResult,
    deadlocks: u32,
    max_exp: Option<u32>,
    ctx: &SearchContext,
) -> SolveResult {
    let bound_stats = result.bound_stats;
    match result.goal {
        Some(goal_id) => {
            let move_layers = result.solution_path().unwrap_or_default();
            let goal_config = result.graph.config(goal_id).clone();
            let cost = result.graph.g_score(goal_id);
            crate::search::verify::assert_move_layers_executable(
                result.graph.config(result.graph.root()),
                &move_layers,
                ctx.index.arch_spec(),
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
            solved
        }
        None => {
            let root_config = result.graph.config(result.graph.root()).clone();
            let status = if max_exp.is_some_and(|max| result.nodes_expanded >= max) {
                SolveStatus::BudgetExceeded
            } else {
                SolveStatus::Unsolvable
            };
            let mut unsolved =
                SolveResult::unsolved(status, root_config, result.nodes_expanded, deadlocks);
            unsolved.bound_stats = bound_stats;
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
    // and the bound it is paired with cannot disagree about it.
    let objective = UniformCost;

    // Build the entropy heuristic tables once per solve, shared across
    // restarts, exactly when the dispatch below will run the entropy driver.
    // Deciding here — next to that dispatch — keeps "will this solve run
    // entropy?" in one place, and building from the same `w_t`/`lookahead`
    // the `EntropyParams` below use keeps tables and params coupled by
    // construction. Skipped when the root already satisfies the goal (the
    // driver early-returns before touching the tables).
    let entropy_tables = (matches!(
        strategy,
        Strategy::Entropy
            | Strategy::Cascade {
                inner: InnerStrategy::Entropy
            }
    ) && !goal.is_goal(&root))
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
    // Deliberately **not** built for loose-goal solves. There, `ctx.targets`
    // is a greedy assignment of qubits to entangling slots, but the goal
    // (`EntanglingConstraintGoal`) accepts *any* valid entangling placement:
    // a qubit can satisfy the goal without ever reaching its assigned target,
    // so the distance to that target can exceed the true remaining cost.
    // `h0` would not be admissible and pruning could discard the optimum.
    // `cz_pairs.is_some()` is exactly the loose-goal marker.
    let completion_bound = match entropy_opts.and_then(|o| o.completion_bound) {
        Some(BoundKind::WeightedDistance) if entropy_tables.is_some() && ctx.cz_pairs.is_none() => {
            Some(WeightedDistanceBound::new(
                &objective,
                ctx.targets,
                ctx.index,
                ctx.blocked,
            ))
        }
        _ => None,
    };
    let completion_bound = completion_bound.as_ref();
    let no_bound = NoBound::for_objective(&objective);

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
                let entropy_params = crate::drivers::entropy::EntropyParams {
                    max_movesets_per_group,
                    max_goal_candidates,
                    lookahead: opts.lookahead,
                    w_t,
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

    // ── Cascade: inner restarts + single A* refinement ─────────
    if let Strategy::Cascade { inner } = strategy {
        let inner_result = run_inner_with_restarts(inner);

        if inner_result.status != SolveStatus::Solved {
            return inner_result;
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
        let spec: bloqade_lanes_bytecode_core::arch::types::ArchSpec =
            serde_json::from_str(example_arch_json()).expect("example arch json parses");
        let index = LaneIndex::new(spec);
        let root = Config::new(initial.iter().copied()).expect("root is a valid config");
        let targets: Vec<(u32, u64)> = vec![(0, loc(1, 5).encode()), (1, loc(1, 6).encode())];
        let target_locs: Vec<u64> = targets.iter().map(|&(_, l)| l).collect();
        let dist_table = DistanceTable::new(&target_locs, &index);
        let blocked = HashSet::new();
        let goal = AllAtTarget::new(&targets);
        let ctx = SearchContext {
            index: &index,
            dist_table: &dist_table,
            blocked: &blocked,
            targets: &targets,
            cz_pairs,
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
    /// true remaining cost. Honouring the request would make pruning
    /// inadmissible and silently discard optimal plans, with nothing in the
    /// output to show for it. `cz_pairs.is_some()` is the marker for that
    /// shape, and this pins that flipping it is what disables the bound.
    #[test]
    fn a_loose_goal_solve_refuses_the_completion_bound() {
        // Fixed-target: the request is honoured, so the loose-goal assertion
        // below cannot pass vacuously through some unrelated path that drops
        // the bound anyway.
        let fixed = entropy_solve(Some(BoundKind::WeightedDistance));
        assert_eq!(fixed.status, SolveStatus::Solved);
        assert!(
            fixed.bound_stats.bound_enabled,
            "a fixed-target solve must honour an explicit completion-bound request"
        );

        let loose = solve_with(
            Strategy::Entropy,
            Some(BoundKind::WeightedDistance),
            Some(&[(0, 1)]),
            &start(),
        );
        assert_eq!(
            loose.status,
            SolveStatus::Solved,
            "refusing the bound must not cost the solve its answer"
        );
        assert!(
            !loose.bound_stats.bound_enabled,
            "h0 is not admissible against a loose entangling goal; the bound must be refused"
        );
        assert_eq!(
            loose.bound_stats.total_cuts(),
            0,
            "a refused bound must not prune"
        );
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
