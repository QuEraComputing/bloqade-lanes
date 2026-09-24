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
//! [`SingleHeuristicCzPlacement`]'s [`CzPlacement::place`] delegates to the
//! free [`solve_single_heuristic`] function.

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;

use crate::placement::cz_placement::{
    CandidateAttempt, CzPlacement, CzStage, PlacementBudget, PlacementResult,
};
use crate::placement::target_generator::{TargetContext, TargetGenerator, validate_candidate};
use crate::primitives::config::{Config, ConfigError};
use crate::search::engine::SearchEngine;
use crate::search::options::{EntropyOptions, SolveOptions};
use crate::search::result::{SolveResult, SolveStatus};
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
}

impl CzPlacement for SingleHeuristicCzPlacement {
    /// `budget.max_expansions` is shared across candidates: each candidate
    /// routes on what the earlier ones left (and, inside a solve, each
    /// restart gets that remainder).
    fn place(
        &self,
        stage: &CzStage<'_>,
        budget: &PlacementBudget,
    ) -> Result<PlacementResult, ConfigError> {
        let search = self.target_solver.search();
        solve_single_heuristic(
            self.target_solver.engine(),
            &search.options,
            Some(&search.entropy_options),
            self.target_generator.as_ref(),
            stage.initial.iter().copied(),
            stage.pairs,
            stage.blocked.iter().copied(),
            budget.max_expansions,
        )
    }
}

/// Shared implementation backing [`SingleHeuristicCzPlacement`]'s
/// [`CzPlacement::place`].
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
    pairs: &[(u32, u32)],
    blocked: impl IntoIterator<Item = LocationAddr>,
    max_expansions: Option<u32>,
) -> Result<PlacementResult, ConfigError> {
    let initial_pairs: Vec<(u32, LocationAddr)> = initial.into_iter().collect();
    let blocked_locs: Vec<LocationAddr> = blocked.into_iter().collect();
    // `TargetGenerator` still takes parallel slices (the plan's open
    // decision 8), so the pairs are unzipped for it.
    let (controls, targets): (Vec<u32>, Vec<u32>) = pairs.iter().copied().unzip();
    let (controls, targets) = (controls.as_slice(), targets.as_slice());

    let ctx = TargetContext {
        placement: &initial_pairs,
        controls,
        targets,
        index: engine.index(),
    };

    let candidates = target_generator.generate(&ctx);

    if candidates.is_empty() {
        let root = Config::new(initial_pairs.iter().copied())?;
        return Ok(PlacementResult {
            result: SolveResult::unsolvable(root),
            chosen: None,
            attempts: Vec::new(),
            total_expansions: 0,
        });
    }

    let mut total_expansions: u32 = 0;
    let mut remaining_budget = max_expansions;
    let mut last_result = None;
    let mut attempts = Vec::new();

    for (i, candidate) in candidates.iter().enumerate() {
        if validate_candidate(candidate, &initial_pairs, controls, targets, engine.index()).is_err()
        {
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
            score: None,
        });

        if result.status == SolveStatus::Solved {
            return Ok(PlacementResult {
                result,
                chosen: Some(i),
                attempts,
                total_expansions,
            });
        }

        if let Some(budget) = remaining_budget.as_mut() {
            *budget = budget.saturating_sub(result.nodes_expanded);
            if *budget == 0 {
                return Ok(PlacementResult {
                    result,
                    chosen: None,
                    attempts,
                    total_expansions,
                });
            }
        }

        last_result = Some(result);
    }

    let result = last_result.unwrap_or_else(|| {
        let root = Config::new(initial_pairs.iter().copied()).expect("initial was valid on entry");
        SolveResult::unsolvable(root)
    });

    Ok(PlacementResult {
        result,
        chosen: None,
        attempts,
        total_expansions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placement::target_generator::DefaultTargetGenerator;
    use crate::search::engine::SearchEngine;
    use crate::search::move_search::MoveSearch;
    use crate::search::target_solver::TargetSolver;
    use crate::test_utils::{example_arch_json, loc};
    use std::sync::Arc;

    /// `place` through `dyn CzPlacement` reports the winning candidate and
    /// an attempt log whose expansions sum to the total.
    #[test]
    fn place_reports_the_winner_and_the_attempt_log() {
        let engine = Arc::new(SearchEngine::from_json(example_arch_json()).unwrap());
        let search = MoveSearch::astar(1.0);
        let target_solver = TargetSolver::new(engine, search);
        let placement =
            SingleHeuristicCzPlacement::new(target_solver, Box::new(DefaultTargetGenerator));

        // Qubit 1 sits on qubit 0's CZ partner word, as in
        // `single_heuristic_default_solves_cz`.
        let initial = vec![(0u32, loc(0, 0)), (1u32, loc(1, 0))];
        let placed = (&placement as &dyn CzPlacement)
            .place(
                &CzStage::new(&initial, &[(0, 1)], &[]),
                &PlacementBudget::new(Some(2000)),
            )
            .unwrap();

        assert_eq!(placed.result.status, SolveStatus::Solved);
        assert_eq!(
            placed.chosen,
            Some(placed.attempts.last().unwrap().candidate_index)
        );
        assert_eq!(
            placed.total_expansions,
            placed
                .attempts
                .iter()
                .map(|a| a.nodes_expanded)
                .sum::<u32>()
        );
        assert!(placed.attempts.iter().all(|a| a.score.is_none()));
    }
}
