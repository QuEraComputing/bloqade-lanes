//! Loose-goal CZ placement strategy.
//!
//! [`LooseGoalCzPlacement`] drives a
//! [`MoveSearch`](crate::search::move_search::MoveSearch) directly
//! against an `EntanglingConstraintGoal` (every CZ pair must occupy
//! *some* valid entangling site, not a pre-decided fixed target).
//! Internally uses [`LooseTargetGenerator`], which computes a Hungarian
//! target assignment once per restart (lazily, on its first `generate`
//! call) and steers every later expansion in that restart toward the
//! cached assignment. Diversity comes from the parallel restarts: each
//! restart's seed perturbs its assignment, and `pick_best` keeps the
//! best result. Because the goal accepts *any* valid entangling
//! placement, the final placement can differ from the cached
//! assignment.
//!
//! Unlike [`SingleHeuristicCzPlacement`](super::single_heuristic::SingleHeuristicCzPlacement),
//! there is *no* [`TargetSolver`](crate::search::target_solver::TargetSolver)
//! involvement — the search predicate is set-membership rather than
//! point-equality, so the per-call routing is fundamentally a
//! different problem shape. The two placement variants compose
//! differently but both satisfy the same
//! [`CzPlacement`](super::cz_placement::CzPlacement) trait.
//!
//! [`LooseGoalCzPlacement`]'s [`CzPlacement::place`] delegates to the free
//! [`solve_loose_goal`] function.

use std::collections::HashSet;
use std::sync::Arc;

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;

use crate::generators::heuristic::DeadlockPolicy;
use crate::generators::{HeuristicGenerator, LooseTargetGenerator};
use crate::goals::EntanglingConstraintGoal;
use crate::ops::entangling::{self, LOOKAHEAD_BETA, MOVE_PENALTY};
use crate::placement::cz_placement::{CzPlacement, CzStage, PlacementBudget, PlacementResult};
use crate::primitives::config::{Config, ConfigError};
use crate::primitives::context::SearchContext;
use crate::primitives::distance::PairDistanceHeuristic;
use crate::primitives::lane_index::LaneIndex;
use crate::search::engine::SearchEngine;
use crate::search::move_search::MoveSearch;
use crate::search::options::{EntanglingOptions, EntropyOptions, SolveOptions};
use crate::search::restarts::run_with_components;
use crate::search::result::SolveResult;

/// CZ placement that simultaneously discovers entangling positions and
/// the routing to reach them.
///
/// Composes:
///
/// - `engine` — the arch-bound state.
/// - `search` — the search algorithm + tuning knobs.
/// - `entangling_options` — Hungarian-assignment knobs
///   (`congestion_weight`, `occupancy_penalty`, `hungarian_horizon`).
pub struct LooseGoalCzPlacement {
    engine: Arc<SearchEngine>,
    search: MoveSearch,
    entangling_options: EntanglingOptions,
}

impl LooseGoalCzPlacement {
    /// Build a `LooseGoalCzPlacement` from its three composing pieces.
    pub fn new(
        engine: Arc<SearchEngine>,
        search: MoveSearch,
        entangling_options: EntanglingOptions,
    ) -> Self {
        Self {
            engine,
            search,
            entangling_options,
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
}

impl CzPlacement for LooseGoalCzPlacement {
    /// `budget.max_expansions` caps the loose-goal solve (each restart
    /// inside it gets the full cap).
    fn place(
        &self,
        stage: &CzStage<'_>,
        budget: &PlacementBudget,
    ) -> Result<PlacementResult, ConfigError> {
        solve_loose_goal(
            &self.engine,
            &self.search.options,
            Some(&self.search.entropy_options),
            &self.entangling_options,
            stage.initial.iter().copied(),
            stage.pairs,
            stage.blocked.iter().copied(),
            budget.max_expansions,
            stage.future_layers,
        )
        .map(PlacementResult::single)
    }
}

/// Shared implementation backing [`LooseGoalCzPlacement`]'s
/// [`CzPlacement::place`].
///
/// Phases:
///
/// 1. Pull the cached `EntanglingCache` (Hungarian word-pair distances
///    + entangling-pair set) from the engine.
/// 2. Run a Hungarian assignment (with optional multi-layer lookahead)
///    to produce the initial `targets` list the search will steer
///    toward.
/// 3. Drive the search via [`run_with_components`] with a
///    [`LooseTargetGenerator`] factory that re-runs Hungarian per
///    restart seed.
///
/// There is no post-solve spectator cleanup: [`EntanglingConstraintGoal`]
/// already rejects any configuration with two spectators on partner sites,
/// so a solved result never contains an accidental CZ.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_loose_goal(
    engine: &SearchEngine,
    opts: &SolveOptions,
    entropy_opts: Option<&EntropyOptions>,
    ent_opts: &EntanglingOptions,
    initial: impl IntoIterator<Item = (u32, LocationAddr)>,
    cz_pairs: &[(u32, u32)],
    blocked: impl IntoIterator<Item = LocationAddr>,
    max_expansions: Option<u32>,
    future_cz_layers: &[Vec<(u32, u32)>],
) -> Result<SolveResult, ConfigError> {
    let root = Config::new(initial)?;
    let blocked_locs: Vec<LocationAddr> = blocked.into_iter().collect();

    // Reuse cached architecture-dependent data (built on first call).
    let cache = engine.entangling_cache();
    let dist_table = cache.dist_table.clone(); // Arc clone (cheap)

    // Per-call: heuristic, goal, greedy assignment.
    let heuristic = PairDistanceHeuristic::new(cz_pairs, &cache.wpd);
    let h_max = |config: &Config| -> f64 { heuristic.estimate_max(config) };
    let h_sum = |config: &Config| -> f64 { heuristic.estimate_sum(config) };

    let goal = EntanglingConstraintGoal::new(cz_pairs, cache.ent_set.clone());

    let blocked_encoded: HashSet<u64> = blocked_locs.iter().map(|l| l.encode()).collect();

    let clipped_future = ent_opts.clipped_future_layers(future_cz_layers);

    // Use lookahead assignment if (clipped) future layers are available.
    let greedy_targets = if !clipped_future.is_empty() {
        entangling::lookahead_assign_pairs(
            cz_pairs,
            &root,
            engine.index(),
            &dist_table,
            &blocked_encoded,
            0,
            clipped_future,
            LOOKAHEAD_BETA,
            ent_opts.congestion_weight,
            ent_opts.occupancy_penalty,
            MOVE_PENALTY,
        )
    } else {
        entangling::assign_pairs_with_blockers(
            cz_pairs,
            &root,
            engine.index(),
            &dist_table,
            &blocked_encoded,
            0,
            None,
            0.0,
            ent_opts.congestion_weight,
            ent_opts.occupancy_penalty,
            MOVE_PENALTY,
            true,
        )
    };

    let ctx = SearchContext {
        index: engine.index(),
        dist_table: &dist_table,
        blocked: &blocked_encoded,
        targets: &greedy_targets,
        cz_pairs: Some(cz_pairs),
    };

    let lookahead = opts.lookahead;
    let top_c = opts.top_c.unwrap_or(3);
    let upgraded_opts = opts.upgraded_for_entangling();
    let opts = &upgraded_opts;

    let result = {
        let index_arc: Arc<LaneIndex> = Arc::new(engine.index().clone());
        let dt_arc = dist_table.clone();
        let congestion_weight = ent_opts.congestion_weight;
        let occupancy_penalty = ent_opts.occupancy_penalty;

        let cz_pairs_owned: Vec<(u32, u32)> = cz_pairs.to_vec();
        let future_layers_owned: Vec<Vec<(u32, u32)>> = clipped_future.to_vec();
        let make_generator = move |seed: u64, policy: DeadlockPolicy| {
            let inner = HeuristicGenerator::configured(seed, policy, lookahead, Some(top_c));
            let mut generator = LooseTargetGenerator::new(
                inner,
                cz_pairs_owned.clone(),
                index_arc.clone(),
                dt_arc.clone(),
                seed,
                congestion_weight,
                occupancy_penalty,
                MOVE_PENALTY,
            );
            if !future_layers_owned.is_empty() {
                generator = generator.with_lookahead(future_layers_owned.clone(), LOOKAHEAD_BETA);
            }
            generator
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
            entropy_opts,
            Some(engine.blended_cache()),
        )
    };

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::move_search::MoveSearch;
    use crate::test_utils::{example_arch_json, loc};

    /// A placement that routes once reports no candidates, and its total is
    /// its one solve's expansions.
    #[test]
    fn place_reports_a_single_solve() {
        let engine = Arc::new(SearchEngine::from_json(example_arch_json()).unwrap());
        let placement =
            LooseGoalCzPlacement::new(engine, MoveSearch::default(), EntanglingOptions::default());

        let initial = vec![(0u32, loc(0, 0)), (1u32, loc(0, 1))];
        let placed = (&placement as &dyn CzPlacement)
            .place(
                &CzStage::new(&initial, &[(0, 1)], &[]),
                &PlacementBudget::new(Some(2000)),
            )
            .unwrap();

        assert_eq!(placed.chosen, None);
        assert!(placed.attempts.is_empty());
        assert_eq!(placed.total_expansions, placed.result.nodes_expanded);
    }
}
