//! Result types returned from every solver entry point.
//!
//! [`SolveStatus`] is the tristate outcome; [`SolveResult`] bundles the
//! status with the path, final placement, and search statistics.

use crate::drivers::entropy::EntropyTrace;
use crate::primitives::config::Config;
use crate::primitives::graph::MoveSet;

/// Outcome status of a solve attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveStatus {
    /// A solution was found.
    Solved,
    /// The search space was fully explored — no solution exists.
    Unsolvable,
    /// The expansion budget was exhausted before finding a solution or
    /// proving unsolvability.
    BudgetExceeded,
}

impl SolveStatus {
    /// Stable string label for status reporting (PyO3 wrappers, logs).
    pub fn as_label(&self) -> &'static str {
        match self {
            Self::Solved => "solved",
            Self::Unsolvable => "unsolvable",
            Self::BudgetExceeded => "budget_exceeded",
        }
    }
}

/// Result of a solve attempt.
///
/// Always returned (never `None`). Check [`status`](SolveResult::status) to
/// determine whether a solution was found.
#[derive(Debug)]
pub struct SolveResult {
    /// Whether the solve succeeded, was unsolvable, or ran out of budget.
    pub status: SolveStatus,
    /// Sequence of parallel move steps from initial to goal configuration.
    /// Empty when `status` is not `Solved`.
    pub move_layers: Vec<MoveSet>,
    /// Final qubit positions (the goal configuration).
    /// Equals the initial configuration when `status` is not `Solved`.
    pub goal_config: Config,
    /// Number of nodes expanded during search.
    pub nodes_expanded: u32,
    /// Total path cost. 0.0 when `status` is not `Solved`.
    pub cost: f64,
    /// Number of deadlocks encountered during search.
    pub deadlocks: u32,
    /// Optional entropy-search trace payload for visualization/debugging.
    pub entropy_trace: Option<EntropyTrace>,
}

impl SolveResult {
    /// Construct a [`SolveStatus::Solved`] result with the given path and counters.
    pub fn solved(
        goal_config: Config,
        move_layers: Vec<MoveSet>,
        cost: f64,
        nodes_expanded: u32,
        deadlocks: u32,
    ) -> Self {
        Self {
            status: SolveStatus::Solved,
            move_layers,
            goal_config,
            nodes_expanded,
            cost,
            deadlocks,
            entropy_trace: None,
        }
    }

    /// Construct a non-Solved result anchored at `root_config`.
    /// Use [`SolveStatus::Unsolvable`] or [`SolveStatus::BudgetExceeded`].
    /// `move_layers` is empty and `cost` is `0.0` by definition.
    pub fn unsolved(
        status: SolveStatus,
        root_config: Config,
        nodes_expanded: u32,
        deadlocks: u32,
    ) -> Self {
        debug_assert!(
            !matches!(status, SolveStatus::Solved),
            "SolveResult::unsolved cannot carry SolveStatus::Solved; use SolveResult::solved"
        );
        Self {
            status,
            move_layers: Vec::new(),
            goal_config: root_config,
            nodes_expanded,
            cost: 0.0,
            deadlocks,
            entropy_trace: None,
        }
    }

    /// Convenience for [`SolveStatus::Unsolvable`] with zero counters
    /// (no expansions happened, no deadlocks encountered).
    pub fn unsolvable(root_config: Config) -> Self {
        Self::unsolved(SolveStatus::Unsolvable, root_config, 0, 0)
    }
}

// ── Multi-candidate solve ──

/// Per-candidate debug info recorded during a multi-candidate solve
/// (see [`SingleHeuristicCzPlacement::solve_with_attempts`](crate::placement::single_heuristic::SingleHeuristicCzPlacement::solve_with_attempts)).
#[derive(Debug, Clone)]
pub struct CandidateAttempt {
    /// Index of this candidate in the generator's output.
    pub candidate_index: usize,
    /// Outcome status of the solve attempt for this candidate.
    pub status: SolveStatus,
    /// Number of nodes expanded for this candidate.
    pub nodes_expanded: u32,
}

/// Result of a multi-candidate solve attempt.
///
/// Surfaced through
/// [`SingleHeuristicCzPlacement::solve_with_attempts`](crate::placement::single_heuristic::SingleHeuristicCzPlacement::solve_with_attempts).
#[derive(Debug)]
pub struct MultiSolveResult {
    /// The solve result from the winning candidate (or the last attempted).
    pub result: SolveResult,
    /// Index of the candidate that succeeded (`None` if all failed).
    pub candidate_index: Option<usize>,
    /// Total nodes expanded across all candidates.
    pub total_expansions: u32,
    /// Number of candidates actually attempted (excludes validation failures).
    pub candidates_tried: usize,
    /// Per-candidate attempt details for debugging.
    pub attempts: Vec<CandidateAttempt>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placement::loose_goal::solve_loose_goal;
    use crate::placement::single_heuristic::solve_single_heuristic;
    use crate::placement::target_generator::DefaultTargetGenerator;
    use crate::search::engine::SearchEngine;
    use crate::search::options::{
        EntanglingOptions, EntropyOptions, InnerStrategy, SolveOptions, Strategy,
    };
    use crate::search::target_solver::solve_with_engine;
    use crate::test_utils::{example_arch_json, loc};

    /// Default test options: A*.
    fn default_opts() -> SolveOptions {
        SolveOptions::default()
    }

    // ── Fixed-target solve (routed through `solve_with_engine`, the shared
    //    implementation behind `TargetSolver::solve`) ──

    #[test]
    fn solve_simple_one_step() {
        let engine = SearchEngine::from_json(example_arch_json()).unwrap();
        let result = solve_with_engine(
            &engine,
            &default_opts(),
            None,
            [(0, loc(0, 0))],
            [(0, loc(0, 5))],
            std::iter::empty(),
            Some(100),
        )
        .unwrap();

        assert_eq!(result.status, SolveStatus::Solved);
        assert_eq!(result.cost, 1.0);
        assert_eq!(result.move_layers.len(), 1);
        assert_eq!(result.goal_config.location_of(0), Some(loc(0, 5)));
    }

    #[test]
    fn solve_already_at_target() {
        let engine = SearchEngine::from_json(example_arch_json()).unwrap();
        let result = solve_with_engine(
            &engine,
            &default_opts(),
            None,
            [(0, loc(0, 5))],
            [(0, loc(0, 5))],
            std::iter::empty(),
            Some(100),
        )
        .unwrap();

        assert_eq!(result.status, SolveStatus::Solved);
        assert_eq!(result.cost, 0.0);
        assert!(result.move_layers.is_empty());
        assert_eq!(result.nodes_expanded, 0);
    }

    #[test]
    fn solve_cross_word() {
        let engine = SearchEngine::from_json(example_arch_json()).unwrap();
        // Move qubit from word 0 site 5 to word 1 site 5 (one word bus hop).
        let result = solve_with_engine(
            &engine,
            &default_opts(),
            None,
            [(0, loc(0, 5))],
            [(0, loc(1, 5))],
            std::iter::empty(),
            Some(100),
        )
        .unwrap();

        assert_eq!(result.status, SolveStatus::Solved);
        assert_eq!(result.cost, 1.0);
        assert_eq!(result.move_layers.len(), 1);
        assert_eq!(result.goal_config.location_of(0), Some(loc(1, 5)));
    }

    #[test]
    fn solve_multi_step() {
        let engine = SearchEngine::from_json(example_arch_json()).unwrap();
        // Word 0 site 0 → word 1 site 5: needs site bus + word bus = 2 steps.
        let result = solve_with_engine(
            &engine,
            &default_opts(),
            None,
            [(0, loc(0, 0))],
            [(0, loc(1, 5))],
            std::iter::empty(),
            Some(1000),
        )
        .unwrap();

        assert_eq!(result.status, SolveStatus::Solved);
        assert_eq!(result.cost, 2.0);
        assert_eq!(result.move_layers.len(), 2);
    }

    #[test]
    fn solve_no_solution() {
        let engine = SearchEngine::from_json(example_arch_json()).unwrap();
        // Target a nonexistent location.
        let result = solve_with_engine(
            &engine,
            &default_opts(),
            None,
            [(0, loc(0, 0))],
            [(0, loc(99, 99))],
            std::iter::empty(),
            Some(100),
        )
        .unwrap();

        assert_ne!(result.status, SolveStatus::Solved);
        assert!(result.move_layers.is_empty());
    }

    #[test]
    fn engine_reusable() {
        let engine = SearchEngine::from_json(example_arch_json()).unwrap();
        let opts = default_opts();

        let r1 = solve_with_engine(
            &engine,
            &opts,
            None,
            [(0, loc(0, 0))],
            [(0, loc(0, 5))],
            std::iter::empty(),
            Some(100),
        )
        .unwrap();

        let r2 = solve_with_engine(
            &engine,
            &opts,
            None,
            [(0, loc(0, 5))],
            [(0, loc(0, 0))],
            std::iter::empty(),
            Some(100),
        )
        .unwrap();

        assert_eq!(r1.cost, 1.0);
        assert_eq!(r2.cost, 1.0);
    }

    #[test]
    fn solve_with_blocked() {
        let engine = SearchEngine::from_json(example_arch_json()).unwrap();
        // Qubit at site 0, target site 5, but site 5 is blocked.
        // Should find no solution (or a longer path if one exists).
        let result = solve_with_engine(
            &engine,
            &default_opts(),
            None,
            [(0, loc(0, 0))],
            [(0, loc(0, 5))],
            [loc(0, 5)],
            Some(100),
        )
        .unwrap();

        // Can't reach blocked destination.
        assert_ne!(result.status, SolveStatus::Solved);
        assert!(result.move_layers.is_empty());
    }

    #[test]
    fn solve_multiple_qubits() {
        let engine = SearchEngine::from_json(example_arch_json()).unwrap();
        // Move two qubits from sites 0,1 to sites 5,6 (parallel site bus move).
        let result = solve_with_engine(
            &engine,
            &default_opts(),
            None,
            [(0, loc(0, 0)), (1, loc(0, 1))],
            [(0, loc(0, 5)), (1, loc(0, 6))],
            std::iter::empty(),
            Some(1000),
        )
        .unwrap();

        // Should find the parallel move in 1 step.
        assert_eq!(result.status, SolveStatus::Solved);
        assert_eq!(result.cost, 1.0);
        assert_eq!(result.goal_config.location_of(0), Some(loc(0, 5)));
        assert_eq!(result.goal_config.location_of(1), Some(loc(0, 6)));
    }

    #[test]
    fn cascade_finds_equal_or_better_than_ids() {
        let engine = SearchEngine::from_json(example_arch_json()).unwrap();
        // Multi-step problem: word 0 site 0 → word 1 site 5.
        let ids_result = solve_with_engine(
            &engine,
            &SolveOptions {
                strategy: Strategy::Ids,
                ..SolveOptions::default()
            },
            None,
            [(0, loc(0, 0))],
            [(0, loc(1, 5))],
            std::iter::empty(),
            Some(1000),
        )
        .unwrap();

        let cascade_result = solve_with_engine(
            &engine,
            &SolveOptions {
                strategy: Strategy::Cascade {
                    inner: InnerStrategy::Ids,
                },
                ..SolveOptions::default()
            },
            None,
            [(0, loc(0, 0))],
            [(0, loc(1, 5))],
            std::iter::empty(),
            Some(1000),
        )
        .unwrap();

        assert_eq!(ids_result.status, SolveStatus::Solved);
        assert_eq!(cascade_result.status, SolveStatus::Solved);
        assert!(cascade_result.cost <= ids_result.cost);
    }

    #[test]
    fn entropy_strategy_can_collect_trace() {
        let engine = SearchEngine::from_json(example_arch_json()).unwrap();
        let entropy_opts = EntropyOptions {
            w_t: 0.0,
            collect_entropy_trace: true,
            ..EntropyOptions::default()
        };
        let result = solve_with_engine(
            &engine,
            &SolveOptions {
                strategy: Strategy::Entropy,
                ..SolveOptions::default()
            },
            Some(&entropy_opts),
            [(0, loc(0, 0))],
            [(0, loc(0, 5))],
            std::iter::empty(),
            Some(100),
        )
        .unwrap();

        assert_eq!(result.status, SolveStatus::Solved);
        let trace = result
            .entropy_trace
            .as_ref()
            .expect("entropy trace should be populated");
        assert_eq!(trace.root_node_id, 0);
        assert!(!trace.steps.is_empty(), "trace should include step events");
    }

    // ── Multi-candidate solve (routed through `solve_single_heuristic`) ──

    #[test]
    fn single_heuristic_default_solves_cz() {
        let engine = SearchEngine::from_json(example_arch_json()).unwrap();
        // Qubit 0 at word 0 site 0, qubit 1 at word 1 site 0.
        // CZ pair: word 0 ↔ word 1. DefaultTargetGenerator should produce
        // a candidate where qubit 0 stays at word 0 (CZ partner of word 1).
        let result = solve_single_heuristic(
            &engine,
            &default_opts(),
            None,
            &DefaultTargetGenerator,
            [(0, loc(0, 0)), (1, loc(1, 0))],
            &[0],
            &[1],
            std::iter::empty(),
            Some(1000),
        )
        .unwrap();

        assert_eq!(result.result.status, SolveStatus::Solved);
        assert_eq!(result.candidate_index, Some(0));
        assert_eq!(result.candidates_tried, 1);
        assert_eq!(result.attempts.len(), 1);
    }

    #[test]
    fn single_heuristic_empty_candidates() {
        let engine = SearchEngine::from_json(example_arch_json()).unwrap();
        // Qubit 1 missing from placement — DefaultTargetGenerator returns empty.
        let result = solve_single_heuristic(
            &engine,
            &default_opts(),
            None,
            &DefaultTargetGenerator,
            [(0, loc(0, 0))],
            &[0],
            &[1],
            std::iter::empty(),
            Some(1000),
        )
        .unwrap();

        assert_eq!(result.result.status, SolveStatus::Unsolvable);
        assert_eq!(result.candidate_index, None);
        assert_eq!(result.candidates_tried, 0);
        assert!(result.attempts.is_empty());
    }

    // ── Loose-goal entangling solve (routed through `solve_loose_goal`) ──

    #[test]
    fn solve_entangling_finds_solution() {
        let engine = SearchEngine::from_json(example_arch_json()).unwrap();
        let result = solve_loose_goal(
            &engine,
            &default_opts(),
            &EntanglingOptions::default(),
            [(0, loc(0, 0)), (1, loc(1, 0))],
            &[(0, 1)],
            std::iter::empty(),
            Some(5000),
            &[],
        )
        .unwrap();

        assert_eq!(result.status, SolveStatus::Solved);
        // Verify goal config satisfies the entangling constraint.
        let arch: bloqade_lanes_bytecode_core::arch::types::ArchSpec =
            serde_json::from_str(example_arch_json()).unwrap();
        let eset = crate::ops::entangling::build_entangling_set(&arch);
        let loc_a = result.goal_config.location_of(0).unwrap().encode();
        let loc_b = result.goal_config.location_of(1).unwrap().encode();
        assert!(
            eset.contains(&(loc_a, loc_b)),
            "goal config should satisfy entangling constraint"
        );
    }

    #[test]
    fn solve_entangling_already_at_goal() {
        let engine = SearchEngine::from_json(example_arch_json()).unwrap();
        // Qubits already at entangling positions.
        let result = solve_loose_goal(
            &engine,
            &default_opts(),
            &EntanglingOptions::default(),
            [(0, loc(0, 5)), (1, loc(1, 5))],
            &[(0, 1)],
            std::iter::empty(),
            Some(100),
            &[],
        )
        .unwrap();

        assert_eq!(result.status, SolveStatus::Solved);
        assert_eq!(result.cost, 0.0);
        assert!(result.move_layers.is_empty());
    }

    #[test]
    fn solve_entangling_multiple_pairs() {
        let engine = SearchEngine::from_json(example_arch_json()).unwrap();
        let result = solve_loose_goal(
            &engine,
            &default_opts(),
            &EntanglingOptions::default(),
            [
                (0, loc(0, 0)),
                (1, loc(1, 0)),
                (2, loc(0, 1)),
                (3, loc(1, 1)),
            ],
            &[(0, 1), (2, 3)],
            std::iter::empty(),
            Some(10000),
            &[],
        )
        .unwrap();

        assert_eq!(result.status, SolveStatus::Solved);
        // Verify both pairs satisfy the constraint.
        let arch: bloqade_lanes_bytecode_core::arch::types::ArchSpec =
            serde_json::from_str(example_arch_json()).unwrap();
        let eset = crate::ops::entangling::build_entangling_set(&arch);
        for &(qa, qb) in &[(0u32, 1u32), (2, 3)] {
            let la = result.goal_config.location_of(qa).unwrap().encode();
            let lb = result.goal_config.location_of(qb).unwrap().encode();
            assert!(
                eset.contains(&(la, lb)),
                "pair ({qa}, {qb}) should be at entangling positions"
            );
        }
    }

    #[test]
    fn solve_entangling_spectator_qubits() {
        let engine = SearchEngine::from_json(example_arch_json()).unwrap();
        // q0/q1 are a CZ pair, q2 is a spectator (not in any pair).
        let result = solve_loose_goal(
            &engine,
            &default_opts(),
            &EntanglingOptions::default(),
            [(0, loc(0, 0)), (1, loc(1, 0)), (2, loc(0, 3))],
            &[(0, 1)],
            std::iter::empty(),
            Some(5000),
            &[],
        )
        .unwrap();

        assert_eq!(result.status, SolveStatus::Solved);
        // Spectator q2 should remain at its initial position.
        assert_eq!(result.goal_config.location_of(2), Some(loc(0, 3)));
    }

    #[test]
    fn solve_entangling_with_ids() {
        let engine = SearchEngine::from_json(example_arch_json()).unwrap();
        let result = solve_loose_goal(
            &engine,
            &SolveOptions {
                strategy: Strategy::Ids,
                ..SolveOptions::default()
            },
            &EntanglingOptions::default(),
            [(0, loc(0, 0)), (1, loc(1, 0))],
            &[(0, 1)],
            std::iter::empty(),
            Some(5000),
            &[],
        )
        .unwrap();

        assert_eq!(result.status, SolveStatus::Solved);
    }

    #[test]
    fn solve_entangling_with_cascade() {
        let engine = SearchEngine::from_json(example_arch_json()).unwrap();
        let result = solve_loose_goal(
            &engine,
            &SolveOptions {
                strategy: Strategy::Cascade {
                    inner: InnerStrategy::Ids,
                },
                ..SolveOptions::default()
            },
            &EntanglingOptions::default(),
            [(0, loc(0, 0)), (1, loc(1, 0))],
            &[(0, 1)],
            std::iter::empty(),
            Some(5000),
            &[],
        )
        .unwrap();

        assert_eq!(result.status, SolveStatus::Solved);
    }
}
