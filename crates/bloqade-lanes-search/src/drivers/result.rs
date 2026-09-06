//! Shared search result type.
//!
//! [`SearchResult`] is produced by every search driver
//! ([`crate::drivers::frontier::run_search`] and
//! [`crate::drivers::entropy::entropy_search`]) and carries the goal node,
//! expansion statistics, and the [`SearchGraph`] for path reconstruction.

use crate::bounds::BoundStats;
use crate::primitives::graph::{MoveSet, NodeId, SearchGraph};

/// Why a search loop ended — a fact about the loop, recorded rather than
/// inferred from the counters afterwards.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Termination {
    /// The expansion budget ran out with entries still pending.
    Budget,
    /// The frontier (and, for a staged driver, its widening queue) drained.
    ///
    /// `proof` is `true` only when the driver can vouch that nothing was
    /// left unexplored: a complete schedule whose stages were never withheld
    /// from a surviving node. The single-stage drivers always report
    /// `false`: their generators offer less than the architecture allows, so
    /// draining says "nothing within this generator's vocabulary", not
    /// "nothing exists".
    Exhausted { proof: bool },
    /// The driver stopped by its own rule: first goal for the frontier
    /// drivers, goal quota for the entropy driver.
    Stopped,
}

/// Result of a search.
#[derive(Debug)]
pub struct SearchResult {
    /// The goal node, if found.
    pub goal: Option<NodeId>,
    /// Number of nodes expanded (popped from frontier and not in closed set).
    pub nodes_expanded: u32,
    /// Maximum depth reached during search.
    pub max_depth_reached: u32,
    /// The search graph, for path reconstruction and inspection.
    pub graph: SearchGraph,
    /// How the loop ended.
    pub termination: Termination,
    /// Expansions per stage of a staged driver; empty for single-stage
    /// drivers. Sums to `nodes_expanded` when non-empty.
    pub stage_expansions: Vec<u32>,
    /// Per node, the stage of the edge that reached it; empty for
    /// single-stage drivers. Indexed by `NodeId`.
    pub via_stage: Vec<u8>,
    /// Branch-and-bound pruning statistics.
    ///
    /// Every counter stays zero for drivers that do not bound and for runs with
    /// the bound disabled. Read [`BoundStats::bound_enabled`] to tell a measured
    /// run from an unmeasured one rather than testing the counters against zero;
    /// `incumbent_cost` is `Some` on any solve that found a goal, bounded or not.
    pub bound_stats: BoundStats,
}

impl SearchResult {
    /// Reconstruct the solution path (sequence of move sets from root to goal).
    ///
    /// Returns `None` if no goal was found.
    pub fn solution_path(&self) -> Option<Vec<MoveSet>> {
        self.goal.map(|id| self.graph.reconstruct_path(id))
    }

    /// The stage that produced the plan: the maximum `via_stage` along the
    /// goal's parent chain. `None` without a goal or for a single-stage
    /// driver (`via_stage` empty).
    pub fn plan_stage(&self) -> Option<u8> {
        let goal = self.goal?;
        if self.via_stage.is_empty() {
            return None;
        }
        let mut stage = 0u8;
        let mut current = goal;
        while let Some(parent) = self.graph.parent(current) {
            stage = stage.max(self.via_stage[current.0 as usize]);
            current = parent;
        }
        Some(stage)
    }

    /// The termination rule the single-stage drivers share at loop exit:
    /// `Budget` exactly when a budget was set and the expansions reached it,
    /// otherwise `Exhausted { proof: false }`.
    pub(crate) fn loop_exit_termination(
        nodes_expanded: u32,
        max_expansions: Option<u32>,
    ) -> Termination {
        if max_expansions.is_some_and(|m| nodes_expanded >= m) {
            Termination::Budget
        } else {
            Termination::Exhausted { proof: false }
        }
    }
}
