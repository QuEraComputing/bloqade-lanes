//! Shared search result type.
//!
//! [`SearchResult`] is produced by every search driver
//! ([`crate::drivers::frontier::run_search`] and
//! [`crate::drivers::entropy::entropy_search`]) and carries the goal node,
//! expansion statistics, and the [`SearchGraph`] for path reconstruction.

use crate::primitives::graph::{MoveSet, NodeId, SearchGraph};

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
}

impl SearchResult {
    /// Reconstruct the solution path (sequence of move sets from root to goal).
    ///
    /// Returns `None` if no goal was found.
    pub fn solution_path(&self) -> Option<Vec<MoveSet>> {
        self.goal.map(|id| self.graph.reconstruct_path(id))
    }
}
