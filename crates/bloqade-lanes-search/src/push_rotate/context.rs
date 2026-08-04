//! Read-only context threaded through planning.
//!
//! Bundles everything a heuristic may consult so that adding a decision point
//! does not mean changing the signature of every operation in `ops`.
//!
//! ## Why geometry is here
//!
//! The planner proper works on a [`LaneGraph`], which knows adjacency and
//! nothing else. That is sufficient to *route* — Push and Rotate only ever
//! asks "is there an edge" — but it is not sufficient to route *for
//! parallelism*. An AOD operation batches lanes that share a bus group and
//! whose source positions form a complete X×Y rectangle, so any heuristic
//! trying to align moves needs the bus group and the (x, y) of each edge.
//!
//! [`EdgeInfo`] is precomputed for every edge once per solve rather than
//! resolved on demand: `LaneIndex::outgoing_lanes` is a linear scan, and an
//! alignment heuristic queries it in its inner loop.

use std::collections::HashMap;

use crate::feasibility::decomposition::Decomposition;
use crate::feasibility::graph::{LaneGraph, VertexId};
use crate::primitives::lane_index::LaneIndex;
use bloqade_lanes_bytecode_core::arch::addr::{LaneAddr, LocationAddr};

use crate::push_rotate::heuristics::PlanHeuristics;

/// Bus-group identity: `(move_type, bus_id, zone_id, direction)`.
///
/// Two lanes can share an AOD operation only if these match. Mirrors
/// `LaneIndex`'s own grouping.
pub type GroupKey = (u8, u32, u32, u8);

/// What a heuristic needs to know about a graph edge.
#[derive(Debug, Clone, Copy)]
pub struct EdgeInfo {
    /// The lane realising this edge.
    pub lane: LaneAddr,
    /// Which AOD bus group it belongs to.
    pub group: GroupKey,
    /// Source position, as raw f64 bits so it can be compared and hashed.
    /// Rectangle membership is an exact-coordinate question, so bit equality
    /// is the right comparison — these are grid coordinates read from the
    /// spec, not computed values that might drift by an ulp.
    pub src_pos: (u64, u64),
}

/// Everything planning reads but never writes.
pub struct PlanCtx<'a> {
    pub graph: &'a LaneGraph,
    pub index: &'a LaneIndex,
    pub decomp: &'a Decomposition,
    /// Goal vertex per agent, indexed by dense agent id.
    pub goal: &'a [VertexId],
    /// Qubit id per agent, indexed by dense agent id. Errors report qubit
    /// ids — the caller's vocabulary — never the internal dense index.
    pub qubits: &'a [u32],
    /// The strategy in force for this solve.
    pub heuristics: &'a dyn PlanHeuristics,
    edges: HashMap<(VertexId, VertexId), EdgeInfo>,
}

impl<'a> PlanCtx<'a> {
    pub fn new(
        graph: &'a LaneGraph,
        index: &'a LaneIndex,
        decomp: &'a Decomposition,
        goal: &'a [VertexId],
        qubits: &'a [u32],
        heuristics: &'a dyn PlanHeuristics,
    ) -> Self {
        let mut edges = HashMap::new();
        for from in graph.vertices() {
            let src = LocationAddr::decode(graph.location_of(from));
            let Some(src_pos) = index.position(src).map(|(x, y)| (x.to_bits(), y.to_bits())) else {
                continue;
            };
            for lane in index.outgoing_lanes(src) {
                let Some((_, dst)) = index.endpoints(lane) else {
                    continue;
                };
                let Some(to) = graph.vertex_of(dst.encode()) else {
                    continue;
                };
                edges.entry((from, to)).or_insert(EdgeInfo {
                    lane: *lane,
                    group: (
                        lane.move_type as u8,
                        lane.bus_id,
                        lane.zone_id,
                        lane.direction as u8,
                    ),
                    src_pos,
                });
            }
        }
        Self {
            graph,
            index,
            decomp,
            goal,
            qubits,
            heuristics,
            edges,
        }
    }

    /// Lane, bus group and source position for an edge, if one exists.
    pub fn edge(&self, from: VertexId, to: VertexId) -> Option<&EdgeInfo> {
        self.edges.get(&(from, to))
    }

    /// Whether two edges could ever share an AOD operation.
    ///
    /// Necessary but not sufficient: sharing a bus group is required, but the
    /// sources must also form a complete rectangle, which depends on the whole
    /// batch rather than a pair.
    pub fn same_group(&self, a: (VertexId, VertexId), b: (VertexId, VertexId)) -> bool {
        match (self.edge(a.0, a.1), self.edge(b.0, b.1)) {
            (Some(x), Some(y)) => x.group == y.group,
            _ => false,
        }
    }
}
