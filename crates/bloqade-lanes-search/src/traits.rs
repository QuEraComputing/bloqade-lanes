//! Core search traits defining the composable search API.

use crate::config::Config;
use crate::context::{MoveCandidate, SearchContext, SearchState};
use crate::graph::{MoveSet, NodeId};

/// Produces candidate move sets from a configuration.
pub trait MoveGenerator {
    fn generate(
        &self,
        config: &Config,
        node_id: NodeId,
        ctx: &SearchContext,
        state: &mut SearchState,
        out: &mut Vec<MoveCandidate>,
    );

    /// Number of deadlock occurrences tracked by this generator (default 0).
    fn deadlock_count(&self) -> u32 {
        0
    }
}

/// Ranks candidates produced by the generator.
/// Higher score = better candidate. Used to sort before graph insertion.
pub trait CandidateScorer {
    fn score(&self, candidate: &MoveCandidate, config: &Config, ctx: &SearchContext) -> f64;
}

/// Computes edge cost for g-score accumulation in the search graph.
/// Separate from candidate scoring -- this affects A* optimality guarantees.
pub trait CostFn {
    fn edge_cost(&self, move_set: &MoveSet, from: &Config, to: &Config) -> f64;
}

/// Decides when the search is complete.
pub trait Goal {
    fn is_goal(&self, config: &Config) -> bool;
}

/// Estimates cost-to-goal for A*/greedy search (h-function).
/// Must be admissible (never overestimates) for A* optimality.
pub trait Heuristic {
    fn estimate(&self, config: &Config) -> f64;
}

/// Blanket impl: any `Fn(&Config) -> f64` closure satisfies `Heuristic`.
impl<F: Fn(&Config) -> f64> Heuristic for F {
    fn estimate(&self, config: &Config) -> f64 {
        self(config)
    }
}
