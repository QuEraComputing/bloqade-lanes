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
