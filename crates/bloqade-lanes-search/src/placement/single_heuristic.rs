//! Single-heuristic CZ placement strategy.
//!
//! [`SingleHeuristicCzPlacement`] composes a
//! [`TargetSolver`](crate::search::target_solver::TargetSolver) with a
//! [`TargetGenerator`](crate::placement::target_generator::TargetGenerator):
//! the generator proposes candidate target placements for the
//! `(controls, targets)` qubit IDs at this CZ layer, and the
//! `TargetSolver` routes from `initial` to each candidate in turn —
//! returning the first successful route, or the last failure if all
//! candidates fail.
//!
//! The legacy
//! [`MoveSolver::solve_with_generator`](crate::search::solve::MoveSolver::solve_with_generator)
//! delegates to the same shared implementation
//! ([`solve_single_heuristic`]) so both paths produce identical results.

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;

use crate::placement::cz_placement::CzPlacement;
use crate::placement::target_generator::{TargetContext, TargetGenerator, validate_candidate};
use crate::primitives::config::{Config, ConfigError};
use crate::search::engine::SearchEngine;
use crate::search::options::{EntropyOptions, SolveOptions};
use crate::search::result::{SolveResult, SolveStatus};
use crate::search::solve::{CandidateAttempt, MultiSolveResult};
use crate::search::target_solver::{TargetSolver, solve_with_engine};

/// CZ placement that uses a [`TargetGenerator`] to propose candidate
/// fixed-target placements, then routes via a [`TargetSolver`].
///
/// Composes:
///
/// - `target_solver` — the routing solver (carries the
///   [`Arc<SearchEngine>`](crate::search::engine::SearchEngine) and
///   [`MoveSearch`](crate::search::move_search::MoveSearch)).
/// - `target_generator` — the plug-in that emits candidate target
///   layouts for a CZ stage's `(controls, targets)`.
pub struct SingleHeuristicCzPlacement {
    target_solver: TargetSolver,
    target_generator: Box<dyn TargetGenerator>,
}

impl SingleHeuristicCzPlacement {
    /// Build a `SingleHeuristicCzPlacement` from its two composing
    /// pieces.
    pub fn new(target_solver: TargetSolver, target_generator: Box<dyn TargetGenerator>) -> Self {
        Self {
            target_solver,
            target_generator,
        }
    }

    /// Borrow the composed target solver.
    pub fn target_solver(&self) -> &TargetSolver {
        &self.target_solver
    }

    /// Borrow the composed target generator.
    pub fn target_generator(&self) -> &dyn TargetGenerator {
        self.target_generator.as_ref()
    }

    /// Solve, returning the full per-candidate attempt detail.
    ///
    /// The trait-level [`CzPlacement::solve`] returns just the winning
    /// (or last) [`SolveResult`]; this method exposes the per-candidate
    /// attempt list, candidate-index of the winner (if any), and
    /// total-expansion accounting.
    pub fn solve_with_attempts(
        &self,
        initial: impl IntoIterator<Item = (u32, LocationAddr)>,
        controls: &[u32],
        targets: &[u32],
        blocked: impl IntoIterator<Item = LocationAddr>,
        max_expansions: Option<u32>,
    ) -> Result<MultiSolveResult, ConfigError> {
        let search = self.target_solver.search();
        solve_single_heuristic(
            self.target_solver.engine(),
            &search.options,
            Some(&search.entropy_options),
            self.target_generator.as_ref(),
            initial,
            controls,
            targets,
            blocked,
            max_expansions,
        )
    }
}

impl CzPlacement for SingleHeuristicCzPlacement {
    fn solve(
        &self,
        initial: &[(u32, LocationAddr)],
        controls: &[u32],
        targets: &[u32],
        blocked: &[LocationAddr],
        max_expansions: Option<u32>,
    ) -> Result<SolveResult, ConfigError> {
        let multi = self.solve_with_attempts(
            initial.iter().copied(),
            controls,
            targets,
            blocked.iter().copied(),
            max_expansions,
        )?;
        Ok(multi.result)
    }
}

/// Shared implementation backing both
/// [`SingleHeuristicCzPlacement::solve_with_attempts`] and the legacy
/// [`MoveSolver::solve_with_generator`](crate::search::solve::MoveSolver::solve_with_generator).
///
/// Generates candidates via `target_generator`, validates each, and
/// runs them through [`solve_with_engine`] in order with a shared
/// expansion budget. Returns on the first successful solve, or the
/// result of the last candidate if all fail / budget runs out.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_single_heuristic(
    engine: &SearchEngine,
    opts: &SolveOptions,
    entropy_opts: Option<&EntropyOptions>,
    target_generator: &dyn TargetGenerator,
    initial: impl IntoIterator<Item = (u32, LocationAddr)>,
    controls: &[u32],
    targets: &[u32],
    blocked: impl IntoIterator<Item = LocationAddr>,
    max_expansions: Option<u32>,
) -> Result<MultiSolveResult, ConfigError> {
    let initial_pairs: Vec<(u32, LocationAddr)> = initial.into_iter().collect();
    let blocked_locs: Vec<LocationAddr> = blocked.into_iter().collect();

    let ctx = TargetContext {
        placement: &initial_pairs,
        controls,
        targets,
        index: engine.index(),
    };

    let candidates = target_generator.generate(&ctx);

    if candidates.is_empty() {
        let root = Config::new(initial_pairs.iter().copied())?;
        return Ok(MultiSolveResult {
            result: SolveResult::unsolvable(root),
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
        if validate_candidate(candidate, controls, targets, engine.index()).is_err() {
            continue;
        }

        let result = solve_with_engine(
            engine,
            opts,
            entropy_opts,
            initial_pairs.iter().copied(),
            candidate.iter().copied(),
            blocked_locs.iter().copied(),
            remaining_budget,
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
        let root = Config::new(initial_pairs.iter().copied()).expect("initial was valid on entry");
        SolveResult::unsolvable(root)
    });

    Ok(MultiSolveResult {
        result,
        candidate_index: None,
        total_expansions,
        candidates_tried: attempts.len(),
        attempts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placement::target_generator::DefaultTargetGenerator;
    use crate::search::engine::SearchEngine;
    use crate::search::move_search::MoveSearch;
    use crate::search::options::{EntropyOptions, SolveOptions, Strategy};
    use crate::search::solve::MoveSolver;
    use crate::search::target_solver::TargetSolver;
    use crate::test_utils::{example_arch_json, loc};
    use std::sync::Arc;

    /// Parity check: `SingleHeuristicCzPlacement::solve_with_attempts` and
    /// the legacy `MoveSolver::solve_with_generator` funnel through the
    /// same `solve_single_heuristic` impl and must produce identical
    /// `MultiSolveResult`s for the same inputs.
    #[test]
    fn single_heuristic_matches_solve_with_generator() {
        let engine = Arc::new(SearchEngine::from_json(example_arch_json()).unwrap());
        let opts = SolveOptions {
            strategy: Strategy::AStar,
            weight: 1.0,
            ..SolveOptions::default()
        };
        let entropy_opts = EntropyOptions::default();
        let search = MoveSearch::new(opts.clone(), entropy_opts.clone());

        let target_solver = TargetSolver::new(engine.clone(), search);
        let placement =
            SingleHeuristicCzPlacement::new(target_solver, Box::new(DefaultTargetGenerator));
        let legacy = MoveSolver::from_index(engine.index().clone());

        let initial = [(0u32, loc(0, 0)), (1u32, loc(0, 1))];
        let controls = [0u32];
        let targets = [1u32];
        let blocked: Vec<LocationAddr> = Vec::new();

        let new_result = placement
            .solve_with_attempts(
                initial.iter().copied(),
                &controls,
                &targets,
                blocked.iter().copied(),
                Some(2000),
            )
            .unwrap();
        let legacy_result = legacy
            .solve_with_generator(
                initial.iter().copied(),
                blocked.iter().copied(),
                &controls,
                &targets,
                &DefaultTargetGenerator,
                Some(2000),
                &opts,
                Some(&entropy_opts),
            )
            .unwrap();

        assert_eq!(new_result.result.status, legacy_result.result.status);
        assert_eq!(new_result.candidate_index, legacy_result.candidate_index);
        assert_eq!(new_result.candidates_tried, legacy_result.candidates_tried);
        assert_eq!(new_result.total_expansions, legacy_result.total_expansions);
        assert_eq!(new_result.attempts.len(), legacy_result.attempts.len());
        let new_layers: Vec<Vec<u64>> = new_result
            .result
            .move_layers
            .iter()
            .map(|ms| ms.encoded_lanes().to_vec())
            .collect();
        let legacy_layers: Vec<Vec<u64>> = legacy_result
            .result
            .move_layers
            .iter()
            .map(|ms| ms.encoded_lanes().to_vec())
            .collect();
        assert_eq!(new_layers, legacy_layers);
    }

    /// The trait-level `CzPlacement::solve` returns the same SolveResult
    /// as `solve_with_attempts(...).result`.
    #[test]
    fn cz_placement_trait_returns_inner_result() {
        let engine = Arc::new(SearchEngine::from_json(example_arch_json()).unwrap());
        let search = MoveSearch::astar(1.0);
        let target_solver = TargetSolver::new(engine, search);
        let placement =
            SingleHeuristicCzPlacement::new(target_solver, Box::new(DefaultTargetGenerator));

        let initial = vec![(0u32, loc(0, 0)), (1u32, loc(0, 1))];
        let blocked: Vec<LocationAddr> = Vec::new();

        let via_attempts = placement
            .solve_with_attempts(
                initial.iter().copied(),
                &[0],
                &[1],
                blocked.iter().copied(),
                Some(2000),
            )
            .unwrap();
        let via_trait = (&placement as &dyn CzPlacement)
            .solve(&initial, &[0], &[1], &blocked, Some(2000))
            .unwrap();

        assert_eq!(via_trait.status, via_attempts.result.status);
        assert_eq!(via_trait.cost.to_bits(), via_attempts.result.cost.to_bits());
    }
}
