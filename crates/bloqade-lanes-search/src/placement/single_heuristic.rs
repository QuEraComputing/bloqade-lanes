//! Single-heuristic CZ placement strategy.
//!
//! [`SingleHeuristicCzPlacement`] composes a
//! [`TargetSolver`](crate::search::target_solver::TargetSolver) with a
//! [`TargetGenerator`](crate::placement::target_generator::TargetGenerator):
//! the generator proposes candidate target placements for the
//! `(controls, targets)` qubit IDs at this CZ layer, and the
//! `TargetSolver` routes from `initial` to each candidate in turn —
//! returning the first successful route, or a failure describing the stage
//! if all candidates fail.
//!
//! [`SingleHeuristicCzPlacement`]'s [`CzPlacement::place`] delegates to the
//! free [`solve_single_heuristic`] function.

use std::collections::HashSet;

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;

use crate::placement::cz_placement::{
    CandidateAttempt, CzPlacement, CzStage, PlacementBudget, PlacementResult, failed_stage_verdict,
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
/// Generates candidates via `target_generator`, validates each (skipping
/// any that place a qubit on a blocked site), and runs them through
/// [`solve_with_engine`] in order with a shared expansion budget. Returns on
/// the first successful solve. If all fail or the budget runs out, the
/// result is the last candidate's with the stage's verdict: `BudgetExceeded`
/// if any candidate ran out of budget, else `Unsolvable`, never a proof.
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
    let mut any_budget = false;
    let mut attempts = Vec::new();
    let blocked_set: HashSet<u64> = blocked_locs.iter().map(|l| l.encode()).collect();

    for (i, candidate) in candidates.iter().enumerate() {
        // A candidate that puts a qubit on a blocked site is unreachable by
        // construction; routing it would only spend the shared budget.
        if validate_candidate(candidate, &initial_pairs, controls, targets, engine.index()).is_err()
            || candidate
                .iter()
                .any(|(_, loc)| blocked_set.contains(&loc.encode()))
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

        any_budget |= result.status == SolveStatus::BudgetExceeded;
        let out_of_budget = remaining_budget.as_mut().is_some_and(|budget| {
            *budget = budget.saturating_sub(result.nodes_expanded);
            *budget == 0
        });
        last_result = Some(result);
        if out_of_budget {
            break;
        }
    }

    // Nothing routed. The verdict describes the stage, not the last
    // candidate's target; see `failed_stage_verdict`.
    let result = match last_result {
        Some(failed) => {
            let (status, termination) = failed_stage_verdict(any_budget);
            SolveResult {
                status,
                termination,
                ..failed
            }
        }
        None => SolveResult::unsolvable(Config::new(initial_pairs.iter().copied())?),
    };

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

    use crate::placement::target_generator::CandidateList;
    use crate::search::options::Strategy;

    /// A candidate that puts a qubit on a blocked site is skipped like one
    /// that fails validation, so it cannot spend the shared budget, and the
    /// next candidate routes.
    #[test]
    fn a_candidate_on_a_blocked_site_is_not_routed() {
        let engine = Arc::new(SearchEngine::from_json(example_arch_json()).unwrap());
        let initial = [(0u32, loc(0, 0)), (1u32, loc(1, 5))];
        let on_blocked = vec![(0u32, loc(0, 5)), (1u32, loc(1, 5))];
        let valid = vec![(0u32, loc(0, 0)), (1u32, loc(1, 0))];
        let placed = SingleHeuristicCzPlacement::new(
            TargetSolver::new(engine, MoveSearch::astar(1.0)),
            Box::new(CandidateList(vec![on_blocked, valid])),
        )
        .place(
            &CzStage::new(&initial, &[(0, 1)], &[loc(0, 5)]),
            &PlacementBudget::new(Some(1000)),
        )
        .unwrap();

        assert_eq!(placed.result.status, SolveStatus::Solved);
        assert_eq!(placed.chosen, Some(1));
        assert_eq!(placed.candidates_tried(), 1);
    }

    /// When every candidate fails, the verdict describes the stage rather
    /// than the last candidate's target: never a proof, and out of budget if
    /// any candidate was.
    #[test]
    fn a_failed_stage_reports_the_stage_verdict() {
        let engine = Arc::new(SearchEngine::from_json(example_arch_json()).unwrap());
        let push_rotate = MoveSearch::new(
            SolveOptions {
                strategy: Strategy::PushRotate,
                ..SolveOptions::default()
            },
            Default::default(),
        );
        let place = |initial: &[(u32, LocationAddr)], candidates| {
            SingleHeuristicCzPlacement::new(
                TargetSolver::new(engine.clone(), push_rotate.clone()),
                Box::new(CandidateList(candidates)),
            )
            .place(
                &CzStage::new(initial, &[(0, 1)], &[]),
                &PlacementBudget::new(Some(1000)),
            )
            .unwrap()
        };

        // Site columns are isolated on the example arch, so Push and Rotate
        // proves this candidate unroutable; the stage is still not proven.
        let initial = [(0u32, loc(0, 0)), (1u32, loc(1, 5))];
        let other_column = vec![(0u32, loc(0, 1)), (1u32, loc(1, 1))];
        let placed = place(&initial, vec![other_column]);
        assert_eq!(placed.attempts[0].status, SolveStatus::Unsolvable);
        assert_eq!(placed.result.status, SolveStatus::Unsolvable);
        assert!(!placed.result.proven(), "{:?}", placed.result.termination);

        // A first candidate Push and Rotate gives up on (one free site in the
        // column) makes the stage out of budget, whatever the last one did.
        let crowded = [(0u32, loc(0, 0)), (1u32, loc(1, 5)), (2u32, loc(0, 5))];
        let too_few_empty = vec![(0u32, loc(0, 0)), (1u32, loc(1, 0)), (2u32, loc(0, 5))];
        let unreachable = vec![(0u32, loc(0, 1)), (1u32, loc(1, 1)), (2u32, loc(0, 5))];
        let placed = place(&crowded, vec![too_few_empty, unreachable]);
        assert_eq!(placed.attempts[0].status, SolveStatus::BudgetExceeded);
        assert_eq!(placed.result.status, SolveStatus::BudgetExceeded);
        assert!(!placed.result.proven(), "{:?}", placed.result.termination);
    }
}
