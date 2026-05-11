//! Shared context and mutable state for search invocations.

use std::collections::HashMap;
use std::collections::HashSet;

use crate::config::Config;
use crate::graph::{MoveSet, NodeId};
use crate::heuristic::DistanceTable;
use crate::lane_index::LaneIndex;

/// A candidate move produced by a generator, before scoring or cost computation.
#[derive(Clone)]
pub struct MoveCandidate {
    pub move_set: MoveSet,
    pub new_config: Config,
}

/// Read-only context built once per solve() invocation.
pub struct SearchContext<'a> {
    pub index: &'a LaneIndex,
    pub dist_table: &'a DistanceTable,
    pub blocked: &'a HashSet<u64>,
    pub targets: &'a [(u32, u64)],
}

/// Per-node state for entropy-guided search.
#[derive(Debug, Clone)]
pub struct EntropyNodeState {
    pub entropy: u32,
    pub candidates_tried: u32,
}

/// Mutable state for a single search run.
#[derive(Default)]
pub struct SearchState {
    pub entropy_map: HashMap<NodeId, EntropyNodeState>,
}
