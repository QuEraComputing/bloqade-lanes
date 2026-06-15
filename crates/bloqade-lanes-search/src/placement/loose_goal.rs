//! Loose-goal CZ placement strategy.
//!
//! [`LooseGoalCzPlacement`] drives a
//! [`MoveSearch`](crate::search::move_search::MoveSearch) directly
//! against an `EntanglingConstraintGoal` (every CZ pair must occupy
//! *some* valid entangling site, not a pre-decided fixed target).
//! Internally uses [`LooseTargetGenerator`] which re-runs the
//! Hungarian assignment per search step so the "target" co-evolves
//! with the current placement.
//!
//! Unlike [`SingleHeuristicCzPlacement`](super::single_heuristic::SingleHeuristicCzPlacement),
//! there is *no* [`TargetSolver`](crate::search::target_solver::TargetSolver)
//! involvement — the search predicate is set-membership rather than
//! point-equality, so the per-call routing is fundamentally a
//! different problem shape. The two placement variants compose
//! differently but both satisfy the same
//! [`CzPlacement`](super::cz_placement::CzPlacement) trait.
//!
//! Both [`LooseGoalCzPlacement::solve_pairs`] and the free
//! [`solve_loose_goal`] function share the same implementation.

use std::collections::HashSet;
use std::sync::Arc;

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;

use crate::generators::heuristic::DeadlockPolicy;
use crate::generators::{HeuristicGenerator, LooseTargetGenerator};
use crate::goals::EntanglingConstraintGoal;
use crate::ops::entangling::{self, LOOKAHEAD_BETA, MOVE_PENALTY};
use crate::placement::cz_placement::CzPlacement;
use crate::primitives::config::{Config, ConfigError};
use crate::primitives::context::SearchContext;
use crate::primitives::distance::PairDistanceHeuristic;
use crate::primitives::lane_index::LaneIndex;
use crate::search::engine::SearchEngine;
use crate::search::move_search::MoveSearch;
use crate::search::options::{EntanglingOptions, SolveOptions};
use crate::search::restarts::run_with_components;
use crate::search::result::{SolveResult, SolveStatus};
use crate::search::target_solver::solve_with_engine;

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

    /// Solve a loose-goal entangling placement + routing problem.
    ///
    /// Equivalent to the trait-level
    /// [`CzPlacement::solve`](super::cz_placement::CzPlacement::solve)
    /// but accepts `cz_pairs` as a `&[(u32, u32)]` directly and an
    /// explicit `future_cz_layers` lookahead window (which the trait
    /// signature doesn't expose).
    pub fn solve_pairs(
        &self,
        initial: impl IntoIterator<Item = (u32, LocationAddr)>,
        cz_pairs: &[(u32, u32)],
        blocked: impl IntoIterator<Item = LocationAddr>,
        max_expansions: Option<u32>,
        future_cz_layers: &[Vec<(u32, u32)>],
    ) -> Result<SolveResult, ConfigError> {
        solve_loose_goal(
            &self.engine,
            &self.search.options,
            &self.entangling_options,
            initial,
            cz_pairs,
            blocked,
            max_expansions,
            future_cz_layers,
        )
    }
}

impl CzPlacement for LooseGoalCzPlacement {
    fn solve(
        &self,
        initial: &[(u32, LocationAddr)],
        controls: &[u32],
        targets: &[u32],
        blocked: &[LocationAddr],
        max_expansions: Option<u32>,
    ) -> Result<SolveResult, ConfigError> {
        assert_eq!(
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

/// Shared implementation backing [`LooseGoalCzPlacement::solve_pairs`].
///
/// Phases:
///
/// 1. Pull the cached `EntanglingCache` (Hungarian word-pair distances
///    + entangling-pair set + partner map) from the engine.
/// 2. Run a Hungarian assignment (with optional multi-layer lookahead)
///    to produce the initial `targets` list the search will steer
///    toward.
/// 3. Drive the search via [`run_with_components`] with a
///    [`LooseTargetGenerator`] factory that re-runs Hungarian per
///    restart seed.
/// 4. If the search solved, run an accidental-CZ cleanup pass:
///    spectator qubits that landed at an entangling-partner site are
///    nudged off via a follow-on [`solve_with_engine`] call.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_loose_goal(
    engine: &SearchEngine,
    opts: &SolveOptions,
    ent_opts: &EntanglingOptions,
    initial: impl IntoIterator<Item = (u32, LocationAddr)>,
    cz_pairs: &[(u32, u32)],
    blocked: impl IntoIterator<Item = LocationAddr>,
    max_expansions: Option<u32>,
    future_cz_layers: &[Vec<(u32, u32)>],
) -> Result<SolveResult, ConfigError> {
    let root = Config::new(initial)?;
    let blocked_locs: Vec<LocationAddr> = blocked.into_iter().collect();
    let arch = engine.index().arch_spec();

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
            arch,
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
            arch,
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

    let mut result = {
        let arch_arc = Arc::new(arch.clone());
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
                arch_arc.clone(),
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
            None,
        )
    };

    // Post-solve cleanup: move spectator qubits out of accidental CZ positions.
    if result.status == SolveStatus::Solved {
        let cz_qubit_set: HashSet<u32> = cz_pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
        let accidental =
            entangling::find_accidental_cz(&result.goal_config, &cz_qubit_set, &cache.partner_map);

        if !accidental.is_empty() {
            let mut cleanup_targets: Vec<(u32, LocationAddr)> = result.goal_config.iter().collect();

            for &(qid, move_loc) in &accidental {
                for &lane in engine.index().outgoing_lanes(move_loc) {
                    if let Some((_, dst)) = engine.index().endpoints(&lane) {
                        if result.goal_config.is_occupied(dst) {
                            continue;
                        }
                        let safe = arch.get_cz_partner(&dst).is_none_or(|p| {
                            !result.goal_config.is_occupied(p)
                                || cz_qubit_set
                                    .contains(&result.goal_config.qubit_at(p).unwrap_or(u32::MAX))
                        });
                        if safe {
                            if let Some(entry) = cleanup_targets.iter_mut().find(|(q, _)| *q == qid)
                            {
                                entry.1 = dst;
                            }
                            break;
                        }
                    }
                }
            }

            let cleanup_result = solve_with_engine(
                engine,
                opts,
                None,
                result.goal_config.iter(),
                cleanup_targets,
                blocked_locs.iter().copied(),
                max_expansions,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::move_search::MoveSearch;
    use crate::test_utils::{example_arch_json, loc};

    /// Trait-level CzPlacement::solve converts (controls, targets) into
    /// cz_pairs and produces the same result as solve_pairs.
    #[test]
    fn cz_placement_trait_matches_solve_pairs() {
        let engine = Arc::new(SearchEngine::from_json(example_arch_json()).unwrap());
        let placement =
            LooseGoalCzPlacement::new(engine, MoveSearch::default(), EntanglingOptions::default());

        let initial = vec![(0u32, loc(0, 0)), (1u32, loc(0, 1))];
        let blocked: Vec<LocationAddr> = Vec::new();

        let via_pairs = placement
            .solve_pairs(
                initial.iter().copied(),
                &[(0, 1)],
                blocked.iter().copied(),
                Some(2000),
                &[],
            )
            .unwrap();
        let via_trait = (&placement as &dyn CzPlacement)
            .solve(&initial, &[0], &[1], &blocked, Some(2000))
            .unwrap();

        assert_eq!(via_trait.status, via_pairs.status);
        assert_eq!(via_trait.cost.to_bits(), via_pairs.cost.to_bits());
    }
}
