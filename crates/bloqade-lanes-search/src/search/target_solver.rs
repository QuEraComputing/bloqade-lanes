//! Single fixed-target solve.
//!
//! [`TargetSolver`] is the composition: `Arc<SearchEngine>` (the
//! arch-bound state) + [`MoveSearch`] (the search configuration). Its
//! `solve(initial, target, blocked, max_expansions)` is the single
//! entry point callers should use.
//!
//! The implementation lives in [`solve_with_engine`] so future tuning and
//! observer wiring happens in exactly one place.

use std::collections::HashSet;
use std::sync::Arc;

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;

use crate::generators::HeuristicGenerator;
use crate::generators::heuristic::DeadlockPolicy;
use crate::goals::AllAtTarget;
use crate::primitives::config::{
    Config, ConfigError, validate_initial_placement, validate_target_assignment,
};
use crate::primitives::context::SearchContext;
use crate::primitives::distance::{DistanceTable, HopDistanceHeuristic};
use crate::push_rotate::{DEFAULT_MOVE_BUDGET, solve_push_rotate};
use crate::search::engine::SearchEngine;
use crate::search::move_search::MoveSearch;
use crate::search::options::{EntropyOptions, SolveOptions, Strategy};
use crate::search::restarts::run_with_components;
use crate::search::result::SolveResult;
use crate::search::result::SolveStatus;

/// Single-target move-synthesis solver.
///
/// Composes an [`Arc<SearchEngine>`] (arch-bound state) with a
/// [`MoveSearch`] (configuration). `solve` accepts an `(initial,
/// target, blocked)` triple and returns a [`SolveResult`].
///
/// Fixed-target routing is this type's job; the loose-goal entangling
/// variants live in the `placement::*CzPlacement` peers. All of them
/// share the same underlying search via [`solve_with_engine`].
pub struct TargetSolver {
    engine: Arc<SearchEngine>,
    search: MoveSearch,
}

impl TargetSolver {
    /// Build a `TargetSolver` from a shared engine and a search config.
    pub fn new(engine: Arc<SearchEngine>, search: MoveSearch) -> Self {
        Self { engine, search }
    }

    /// Borrow the underlying engine.
    pub fn engine(&self) -> &Arc<SearchEngine> {
        &self.engine
    }

    /// Borrow the search configuration.
    pub fn search(&self) -> &MoveSearch {
        &self.search
    }

    /// Solve a single-target move-synthesis problem.
    ///
    /// # Arguments
    ///
    /// * `initial` — Starting qubit positions: `(qubit_id, location)` pairs.
    /// * `target` — Desired qubit positions: `(qubit_id, location)` pairs.
    /// * `blocked` — Locations occupied by external atoms (immovable obstacles).
    /// * `max_expansions` — Optional limit on node expansions.
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
    ) -> Result<SolveResult, ConfigError> {
        solve_with_engine(
            &self.engine,
            &self.search.options,
            Some(&self.search.entropy_options),
            initial,
            target,
            blocked,
            max_expansions,
        )
    }
}

/// Whether swapping `initial` and `target` would change the instance's
/// meaning with respect to `blocked`.
///
/// A `blocked` location is an external atom, so no move may *land* on one —
/// but nothing forbids the root placement from sitting on one, and the
/// generators never check it. That makes an endpoint that overlaps `blocked`
/// asymmetric: as a target it is unreachable, as a root it is merely a
/// starting point the atoms move off. Mirroring across such an endpoint would
/// "solve" an instance that is genuinely unsolvable (target on a blocked
/// location) by producing a plan that parks an atom on top of an external one
/// — which the replay verifier cannot catch, since blocked atoms are not in
/// the configuration. Skip the mirror instead.
fn mirroring_breaks_blocked(
    blocked: &[LocationAddr],
    initial_pairs: &[(u32, LocationAddr)],
    target_pairs: &[(u32, LocationAddr)],
) -> bool {
    if blocked.is_empty() {
        return false;
    }
    let blocked_set: HashSet<u64> = blocked.iter().map(|l| l.encode()).collect();
    initial_pairs
        .iter()
        .chain(target_pairs)
        .any(|(_, loc)| blocked_set.contains(&loc.encode()))
}

/// Shared implementation backing [`TargetSolver::solve`].
///
/// Builds the distance table, heuristic, goal predicate, search
/// context, and generator factory from the supplied arch (`engine`)
/// and options, then dispatches to
/// [`run_with_components`](crate::search::restarts::run_with_components).
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_with_engine(
    engine: &SearchEngine,
    opts: &SolveOptions,
    entropy_opts: Option<&EntropyOptions>,
    initial: impl IntoIterator<Item = (u32, LocationAddr)>,
    target: impl IntoIterator<Item = (u32, LocationAddr)>,
    blocked: impl IntoIterator<Item = LocationAddr>,
    max_expansions: Option<u32>,
) -> Result<SolveResult, ConfigError> {
    let root = Config::new(initial)?;
    validate_initial_placement(&root)?;
    let target_pairs: Vec<(u32, LocationAddr)> = target.into_iter().collect();
    // A non-injective target assignment (two qubits on one location, or one
    // qubit given two locations) is a malformed request. Reject it here —
    // ahead of every strategy, the Push and Rotate branch, and any
    // feasibility pass — rather than letting it surface as a verdict.
    validate_target_assignment(&target_pairs)?;
    let blocked_locs: Vec<LocationAddr> = blocked.into_iter().collect();
    let initial_pairs: Vec<(u32, LocationAddr)> = root.iter().collect();

    // Mirroring: solve `target -> initial` and turn the plan around.
    //
    // Only well-defined when the target is a total assignment over the
    // initial placement's qubits: a mirrored instance needs a single
    // concrete configuration to start from. A partial target
    // has no configuration to start the mirrored solve from, so the option
    // no-ops there rather than mirroring something else.
    //
    // The test below is cardinality only, which is weaker than that: with
    // `validate_target_assignment` already past, an equal-length target can
    // still name a *different* qubit-id set than `root` (root `{0, 1}`,
    // target `{0, 2}`), and such a target is admitted here. Harmless rather
    // than wrong, because `AllAtTarget::is_goal` requires each target qubit
    // to be present in the config: the mirror's goal names a qubit its root
    // does not contain, so the mirror reports `Unsolvable` and that verdict
    // is returned — the same verdict the forward solve reaches, for the
    // mirror-image reason. So the loose check costs a futile search on
    // malformed input, never a wrong plan.
    if opts.backwards_search
        && target_pairs.len() == root.len()
        && !mirroring_breaks_blocked(&blocked_locs, &initial_pairs, &target_pairs)
    {
        // `backwards_search: false` is the recursion guard: the mirrored solve must
        // run forward or this recurses forever.
        let mirrored_opts = SolveOptions {
            backwards_search: false,
            ..opts.clone()
        };
        let mirrored = solve_with_engine(
            engine,
            &mirrored_opts,
            entropy_opts,
            target_pairs.iter().copied(),
            initial_pairs.iter().copied(),
            blocked_locs.iter().copied(),
            max_expansions,
        )?;
        if mirrored.status != SolveStatus::Solved {
            // An unsolved result reports the configuration the *caller's*
            // solve started from, not the mirror's.
            return Ok(SolveResult {
                goal_config: root,
                ..mirrored
            });
        }

        // The mirrored plan `[m1, …, mk]` runs `target -> initial`: applying
        // `m1` to the target configuration yields `c1`, and `mk` yields
        // `initial`. Undoing it from `initial` therefore applies `mk` first,
        // backwards — so the plan for `initial -> target` is
        // `[mk⁻¹, m(k-1)⁻¹, …, m1⁻¹]`: the list reversed *and* every element
        // inverted. Doing only one of the two yields a plan that is often
        // still executable but lands somewhere else entirely.
        let layers: Vec<_> = mirrored
            .move_layers
            .iter()
            .rev()
            .map(|layer| layer.inverse())
            .collect();
        // `Config::new` cannot fail: `validate_target_assignment` already
        // rejected duplicate qubit ids.
        let goal_config = Config::new(target_pairs.iter().copied())?;
        // For lane-endpoint and occupancy semantics this is the check that
        // distinguishes a correct transform from a plausible-looking wrong
        // one — it is what rejects reversing without inverting (or the
        // converse), which yields a plan that still executes but lands
        // elsewhere. Runs in release, like every other packaging-time replay
        // in this crate.
        //
        // It is *not* a check on AOD geometry under inversion, and is
        // structurally blind to it: `check_lane_group_geometry` builds its
        // grid from each lane's raw `(zone_id, word_id, site_id)` — the
        // forward source by convention — and never consults
        // `lane_endpoints`, so a lane and its inverse always get the same
        // verdict, and an inverted group is checked at its drop side rather
        // than its pickup side. What makes that sound is a property of the
        // arch, not of this transform: the src -> dst coordinate map of a bus
        // must be *separable* (dst x a function of src x alone, dst y of src
        // y alone), which is what lets a rectangle on one side certify a
        // rectangle on the other. Every bus of the bundled Gemini physical
        // spec is separable (3 site buses, 19 word buses, no zone buses), so
        // the one-sided check is equivalent to a two-sided one there. The
        // property is currently unstated and unenforced — arch build time
        // validates that each bus's *full* src and dst sets are rectangles,
        // which does not imply separability for lane *subsets*. Pre-existing,
        // and orthogonal to mirroring.
        //
        // Note this is the *second* replay a mirrored plan goes through: the
        // mirror's own plan was already verified inside `run_with_components`
        // (see `extract` in `search/restarts.rs`).
        crate::search::verify::assert_move_layers_executable(
            &root,
            &layers,
            engine.index().arch_spec(),
            &goal_config,
        );
        // `nodes_expanded`, `deadlocks` and `cost` describe the search that
        // actually ran; inversion preserves the layer count that `cost`
        // measures, so they carry over unchanged.
        return Ok(SolveResult {
            move_layers: layers,
            goal_config,
            ..mirrored
        });
    }

    // Push and Rotate is not a search, so it bypasses the whole
    // frontier/generator apparatus below rather than being a `Frontier`.
    //
    // `max_expansions` is deliberately NOT mapped onto the planner's move
    // budget: it caps search *node expansions*, a knob for frontier
    // strategies, while the planner is rule-based and needs no exploration
    // budget — its `DEFAULT_MOVE_BUDGET` is a runaway guard, not a search
    // budget. Mapping the (often small) expansion cap onto emitted moves
    // would strangle exactly the solves this strategy exists to finish.
    if opts.strategy == Strategy::PushRotate {
        return solve_push_rotate(
            engine.index(),
            &initial_pairs,
            &target_pairs,
            &blocked_locs,
            DEFAULT_MOVE_BUDGET,
        );
    }

    // Build goal predicate.
    let target_encoded: Vec<(u32, u64)> =
        target_pairs.iter().map(|&(q, l)| (q, l.encode())).collect();

    // Build distance table and heuristic (shared across restarts).
    let target_locs: Vec<u64> = target_encoded.iter().map(|&(_, l)| l).collect();
    let w_t = entropy_opts.map_or(EntropyOptions::default().w_t, |e| e.w_t);
    let dist_table = if w_t > 0.0 {
        DistanceTable::new(&target_locs, engine.index()).with_time_distances(engine.index())
    } else {
        DistanceTable::new(&target_locs, engine.index())
    };
    let heuristic = HopDistanceHeuristic::new(target_pairs.iter().copied(), &dist_table);
    let h_max = |config: &Config| -> f64 { heuristic.estimate_max(config) };
    let h_sum = |config: &Config| -> f64 { heuristic.estimate_sum(config) };

    let goal_obj = AllAtTarget::new(&target_encoded);
    let blocked_encoded: HashSet<u64> = blocked_locs.iter().map(|l| l.encode()).collect();
    let ctx = SearchContext {
        index: engine.index(),
        dist_table: &dist_table,
        blocked: &blocked_encoded,
        targets: &target_encoded,
        cz_pairs: None,
    };

    let lookahead = opts.lookahead;
    let top_c = opts.top_c;
    let make_generator = |seed: u64, policy: DeadlockPolicy| {
        HeuristicGenerator::configured(seed, policy, lookahead, top_c)
    };

    // The engine's cross-solve blended-column cache: entropy strategies
    // build their heuristic tables from it inside `run_with_components`, so
    // repeated solves against recurring target locations (one solve per
    // candidate layout per CZ layer) skip the distance-column fill entirely.
    let result = run_with_components(
        root.clone(),
        &goal_obj,
        make_generator,
        h_max,
        h_sum,
        &ctx,
        max_expansions,
        opts,
        entropy_opts,
        Some(engine.blended_cache()),
    );

    // Opt-in reliability net. Push and Rotate is complete, so this converts
    // "the search gave up" into either a schedule or a proof that none
    // exists. Only the *failure* path pays for it.
    if opts.fallback_push_rotate && result.status != SolveStatus::Solved {
        let fallback = solve_push_rotate(
            engine.index(),
            &initial_pairs,
            &target_pairs,
            &blocked_locs,
            DEFAULT_MOVE_BUDGET,
        )?;
        if fallback.status == SolveStatus::Solved {
            return Ok(fallback);
        }
        // Both failed. Prefer the planner's verdict when it is a *proof* of
        // unsolvability; the search's `Unsolvable` only means its frontier
        // drained, which says nothing.
        if fallback.status == SolveStatus::Unsolvable {
            return Ok(fallback);
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::move_search::MoveSearch;
    use crate::search::result::SolveStatus;
    use crate::test_utils::{chain_arch_json, example_arch_json, loc};
    use std::sync::Arc;

    fn make_engine() -> Arc<SearchEngine> {
        Arc::new(SearchEngine::from_json(example_arch_json()).unwrap())
    }

    /// The conveyor-chain fixture (`0→1→2→3→4` along one row). Its bus
    /// destinations overlap its sources, so plans there are genuinely
    /// direction-dependent — unlike the example arch, where each site column
    /// is an isolated 4-node path and every plan is forced.
    fn make_chain_engine() -> Arc<SearchEngine> {
        Arc::new(SearchEngine::from_json(&chain_arch_json()).unwrap())
    }

    #[test]
    fn target_solver_solves_simple_move() {
        let engine = make_engine();
        let search = MoveSearch::astar(1.0);
        let solver = TargetSolver::new(engine, search);

        let result = solver
            .solve(
                [(0, loc(0, 0))],
                [(0, loc(0, 5))],
                std::iter::empty(),
                Some(1000),
            )
            .unwrap();

        assert_eq!(result.status, SolveStatus::Solved);
        assert!(!result.move_layers.is_empty());
        assert_eq!(result.goal_config.location_of(0), Some(loc(0, 5)));
    }
    // ── `SolveOptions::backwards_search` ──
    //
    // The transform under test is "reverse the layer list AND invert each
    // layer". Getting exactly one of the two right still produces a plausible
    // plan, so every test here leans on the replay verifier
    // (`assert_move_layers_executable`, called inside `solve_with_engine` on
    // the transformed plan) to reject it — a wrong transform panics rather
    // than returning a quietly-wrong result.

    fn backwards_options(strategy: Strategy) -> SolveOptions {
        SolveOptions {
            strategy,
            backwards_search: true,
            ..SolveOptions::default()
        }
    }

    #[test]
    fn backwards_search_solves_and_lands_on_the_requested_target() {
        let engine = make_engine();
        let result = solve_with_engine(
            &engine,
            &backwards_options(Strategy::AStar),
            None,
            [(0, loc(0, 0)), (1, loc(0, 1))],
            [(0, loc(1, 5)), (1, loc(1, 6))],
            std::iter::empty(),
            Some(2000),
        )
        .unwrap();

        assert_eq!(result.status, SolveStatus::Solved);
        assert_eq!(result.goal_config.location_of(0), Some(loc(1, 5)));
        assert_eq!(result.goal_config.location_of(1), Some(loc(1, 6)));
        assert!(!result.move_layers.is_empty());
    }

    #[test]
    fn backwards_and_forward_agree_on_the_goal_configuration() {
        let engine = make_engine();
        let initial = [(0, loc(0, 0)), (1, loc(0, 1))];
        let target = [(0, loc(1, 5)), (1, loc(1, 6))];

        let forward = solve_with_engine(
            &engine,
            &SolveOptions {
                strategy: Strategy::Entropy,
                ..SolveOptions::default()
            },
            None,
            initial,
            target,
            std::iter::empty(),
            Some(2000),
        )
        .unwrap();
        let backwards = solve_with_engine(
            &engine,
            &backwards_options(Strategy::Entropy),
            None,
            initial,
            target,
            std::iter::empty(),
            Some(2000),
        )
        .unwrap();

        assert_eq!(forward.status, SolveStatus::Solved);
        assert_eq!(backwards.status, SolveStatus::Solved);
        for qubit in [0, 1] {
            assert_eq!(
                backwards.goal_config.location_of(qubit),
                forward.goal_config.location_of(qubit),
                "both directions must land qubit {qubit} on the same location"
            );
        }
    }

    #[test]
    fn backwards_search_returns_the_transformed_mirror_plan() {
        // The discriminating test: a backwards solve must return exactly the
        // mirrored solve's plan, reversed and inverted. A `backwards_search` flag that
        // was silently ignored would return the forward plan instead, which on
        // this conveyor-chain instance is a different plan.
        let engine = make_chain_engine();
        let initial = [(0, loc(0, 0)), (1, loc(0, 2))];
        let target = [(0, loc(0, 3)), (1, loc(0, 4))];

        let mirrored = solve_with_engine(
            &engine,
            &SolveOptions::default(),
            None,
            target,
            initial,
            std::iter::empty(),
            Some(2000),
        )
        .unwrap();
        let backwards = solve_with_engine(
            &engine,
            &backwards_options(Strategy::AStar),
            None,
            initial,
            target,
            std::iter::empty(),
            Some(2000),
        )
        .unwrap();

        assert_eq!(mirrored.status, SolveStatus::Solved);
        assert_eq!(backwards.status, SolveStatus::Solved);
        let expected: Vec<_> = mirrored
            .move_layers
            .iter()
            .rev()
            .map(|layer| layer.inverse())
            .collect();
        assert_eq!(backwards.move_layers, expected);
        assert_eq!(backwards.nodes_expanded, mirrored.nodes_expanded);
        assert_eq!(backwards.cost, mirrored.cost);
        assert_eq!(backwards.deadlocks, mirrored.deadlocks);

        // Guard the fixture itself: if the forward plan ever coincides with the
        // transformed mirror plan, the assertion above stops discriminating.
        let forward = solve_with_engine(
            &engine,
            &SolveOptions::default(),
            None,
            initial,
            target,
            std::iter::empty(),
            Some(2000),
        )
        .unwrap();
        assert_ne!(
            forward.move_layers, expected,
            "the fixture must be direction-sensitive for this test to mean anything"
        );
    }

    #[test]
    fn backwards_search_no_ops_on_a_partial_target() {
        // Two qubits in the initial placement, one target: the mirrored
        // instance is not well-defined (the "initial" of the mirror would not
        // cover every qubit), so the option must be ignored rather than
        // producing a bogus plan.
        let engine = make_engine();
        let initial = [(0, loc(0, 0)), (1, loc(1, 0))];
        let target = [(0, loc(0, 5))];
        let result = solve_with_engine(
            &engine,
            &backwards_options(Strategy::AStar),
            None,
            initial,
            target,
            std::iter::empty(),
            Some(2000),
        )
        .unwrap();
        let plain = solve_with_engine(
            &engine,
            &SolveOptions::default(),
            None,
            initial,
            target,
            std::iter::empty(),
            Some(2000),
        )
        .unwrap();

        assert_eq!(result.status, SolveStatus::Solved);
        assert_eq!(result.goal_config.location_of(0), Some(loc(0, 5)));
        assert_eq!(
            result.move_layers, plain.move_layers,
            "the option must be a no-op, not a different (mirrored) solve"
        );
    }

    #[test]
    fn backwards_search_reports_the_root_placement_when_the_mirror_fails() {
        // Site columns are disjoint on the example arch, so no plan connects
        // column 0 to column 1 in either direction. An unsolved result must
        // report the *original* initial configuration, not the mirror's.
        let engine = make_engine();
        let result = solve_with_engine(
            &engine,
            &backwards_options(Strategy::AStar),
            None,
            [(0, loc(0, 0))],
            [(0, loc(0, 6))],
            std::iter::empty(),
            Some(200),
        )
        .unwrap();

        assert_ne!(result.status, SolveStatus::Solved);
        assert!(result.move_layers.is_empty());
        assert_eq!(result.goal_config.location_of(0), Some(loc(0, 0)));
    }

    #[test]
    fn backwards_search_no_ops_when_an_endpoint_sits_on_a_blocked_location() {
        // The target location holds an external atom, so the instance is
        // unsolvable. The mirror would start *on* that location and happily
        // move away, "solving" it with a plan that parks qubit 0 on top of the
        // blocker — and the replay verifier cannot see blocked atoms. The
        // option must decline to mirror here.
        let engine = make_engine();
        let result = solve_with_engine(
            &engine,
            &backwards_options(Strategy::AStar),
            None,
            [(0, loc(0, 0))],
            [(0, loc(0, 5))],
            [loc(0, 5)],
            Some(200),
        )
        .unwrap();

        assert_ne!(
            result.status,
            SolveStatus::Solved,
            "a target on a blocked location has no valid plan"
        );
        assert!(result.move_layers.is_empty());
    }

    #[test]
    fn inverting_and_reversing_a_plan_twice_is_the_identity() {
        // The transform is its own inverse, which is exactly why solving the
        // mirror and applying it yields a plan for the original instance.
        let engine = make_engine();
        let forward = solve_with_engine(
            &engine,
            &SolveOptions::default(),
            None,
            [(0, loc(0, 0)), (1, loc(0, 1))],
            [(0, loc(1, 5)), (1, loc(1, 6))],
            std::iter::empty(),
            Some(2000),
        )
        .unwrap();
        assert_eq!(forward.status, SolveStatus::Solved);
        assert!(forward.move_layers.len() > 1, "need a non-trivial plan");

        let once: Vec<_> = forward
            .move_layers
            .iter()
            .rev()
            .map(|layer| layer.inverse())
            .collect();
        let twice: Vec<_> = once.iter().rev().map(|layer| layer.inverse()).collect();
        assert_eq!(twice, forward.move_layers);
        assert_ne!(
            once, forward.move_layers,
            "a single application must actually change the plan"
        );
    }
}
