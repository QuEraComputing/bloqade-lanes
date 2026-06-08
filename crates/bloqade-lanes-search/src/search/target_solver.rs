//! Single fixed-target solve.
//!
//! [`TargetSolver`] is the composition: `Arc<SearchEngine>` (the
//! arch-bound state) + [`MoveSearch`] (the search configuration). Its
//! `solve(initial, target, blocked, max_expansions)` is the single
//! entry point new callers should use.
//!
//! The same implementation backs the legacy
//! [`MoveSolver::solve`](super::solve::MoveSolver::solve) facade —
//! both paths funnel through [`solve_with_engine`] so future tuning
//! and observer wiring happens in exactly one place.

use std::collections::HashSet;
use std::sync::Arc;

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;

use crate::generators::HeuristicGenerator;
use crate::generators::heuristic::DeadlockPolicy;
use crate::goals::AllAtTarget;
use crate::primitives::config::{Config, ConfigError};
use crate::primitives::context::SearchContext;
use crate::primitives::distance::{DistanceTable, HopDistanceHeuristic};
use crate::search::engine::SearchEngine;
use crate::search::move_search::MoveSearch;
use crate::search::options::{EntropyOptions, SolveOptions};
use crate::search::restarts::run_with_components;
use crate::search::result::SolveResult;

/// Single-target move-synthesis solver.
///
/// Composes an [`Arc<SearchEngine>`] (arch-bound state) with a
/// [`MoveSearch`] (configuration). `solve` accepts an `(initial,
/// target, blocked)` triple and returns a [`SolveResult`].
///
/// The legacy [`MoveSolver`](super::solve::MoveSolver) facade
/// dissolves into this type plus the upcoming `CzPlacement` peers;
/// both `TargetSolver` and `MoveSolver` share the same underlying
/// implementation via [`solve_with_engine`].
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

/// Shared implementation backing both [`TargetSolver::solve`] and the
/// legacy [`MoveSolver::solve`](super::solve::MoveSolver::solve).
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
    let target_pairs: Vec<(u32, LocationAddr)> = target.into_iter().collect();
    let blocked_locs: Vec<LocationAddr> = blocked.into_iter().collect();

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

    Ok(run_with_components(
        root,
        &goal_obj,
        make_generator,
        h_max,
        h_sum,
        &ctx,
        max_expansions,
        opts,
        entropy_opts,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::move_search::MoveSearch;
    use crate::search::options::{EntropyOptions, SolveOptions, Strategy};
    use crate::search::result::SolveStatus;
    use crate::search::solve::MoveSolver;
    use crate::test_utils::{example_arch_json, loc};
    use std::sync::Arc;

    fn make_engine() -> Arc<SearchEngine> {
        Arc::new(SearchEngine::from_json(example_arch_json()).unwrap())
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

    /// Parity check: `TargetSolver::solve` and the legacy
    /// `MoveSolver::solve` must produce identical results when given
    /// the same engine, search config, and inputs — both now funnel
    /// through `solve_with_engine`.
    #[test]
    fn target_solver_matches_move_solver_solve() {
        let engine = make_engine();
        let opts = SolveOptions {
            strategy: Strategy::AStar,
            weight: 1.0,
            ..SolveOptions::default()
        };
        let entropy_opts = EntropyOptions::default();
        let search = MoveSearch::new(opts.clone(), entropy_opts.clone());
        let target_solver = TargetSolver::new(engine.clone(), search);

        // The legacy facade shares the engine (no rebuild cost).
        let move_solver = MoveSolver::from_index(engine.index().clone());

        let initial = [(0u32, loc(0, 0))];
        let target = [(0u32, loc(0, 5))];
        let blocked: [LocationAddr; 0] = [];

        let ts_result = target_solver
            .solve(
                initial.iter().copied(),
                target.iter().copied(),
                blocked.iter().copied(),
                Some(1000),
            )
            .unwrap();
        let ms_result = move_solver
            .solve(
                initial.iter().copied(),
                target.iter().copied(),
                blocked.iter().copied(),
                Some(1000),
                &opts,
                Some(&entropy_opts),
            )
            .unwrap();

        assert_eq!(ts_result.status, ms_result.status);
        assert_eq!(ts_result.cost.to_bits(), ms_result.cost.to_bits());
        assert_eq!(ts_result.nodes_expanded, ms_result.nodes_expanded);
        assert_eq!(ts_result.deadlocks, ms_result.deadlocks);
        let ts_layers: Vec<Vec<u64>> = ts_result
            .move_layers
            .iter()
            .map(|ms| ms.encoded_lanes().to_vec())
            .collect();
        let ms_layers: Vec<Vec<u64>> = ms_result
            .move_layers
            .iter()
            .map(|ms| ms.encoded_lanes().to_vec())
            .collect();
        assert_eq!(ts_layers, ms_layers);
    }
}
