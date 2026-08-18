//! The lane graph, reversed and interned — shared substrate for every
//! distance-to-target table.
//!
//! Three tables in this crate answer "what does it cost to get from an
//! arbitrary location *to* a fixed target": [`DistanceTable`]'s hop counts,
//! its optional duration-weighted companion, and
//! [`WeightedDistanceTable`](super::weighted_distance::WeightedDistanceTable)'s
//! objective-weighted distances. All three need the same three things first,
//! and each used to build them itself:
//!
//! 1. one sweep over `bus_groups() × lanes_for() × endpoints()`,
//! 2. every surviving location interned to a dense index,
//! 3. predecessor lists keyed by *destination* — reversed, because one search
//!    from a target then fills that target's whole column.
//!
//! What legitimately differs between the three is only the **edge filter and
//! weight** (all lanes at unit cost; only lanes carrying a duration; every lane
//! at `Objective::lane_weight`) and whether `blocked` locations are carved out.
//! Both are parameters here.
//!
//! [`DistanceTable`]: super::distance::DistanceTable
//!
//! # Determinism
//!
//! `LaneIndex::bus_groups` iterates a `HashMap`, so the order edges are
//! discovered — and hence the order of each node's predecessor list — already
//! varies between processes. That is safe, and this module does not change it:
//! both traversals below compute order-independent minima, so the *values* are
//! reproducible even though the traversal is not. Anything added here must keep
//! that property.

use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

use bloqade_lanes_bytecode_core::arch::addr::LaneAddr;

use crate::primitives::lane_index::LaneIndex;

/// A location that survived the carve, plus its reversed in-edges.
pub(crate) struct ReverseLaneGraph {
    /// `encoded location → compact index`.
    loc_index: HashMap<u64, usize>,
    /// Inverse of `loc_index`.
    loc_by_index: Vec<u64>,
    /// `dst_idx → [(src_idx, weight)]`.
    preds: Vec<Vec<(usize, f64)>>,
}

impl ReverseLaneGraph {
    /// Sweep the lane graph and build the reversed, interned adjacency.
    ///
    /// `weight_of` returns the edge weight for a lane, or `None` to drop that
    /// lane entirely — which is how the duration-weighted table skips lanes
    /// with no transport-path data. A lane touching `blocked` at either end is
    /// dropped before `weight_of` is consulted: such a lane can never be taken,
    /// so it must not contribute a path.
    pub(crate) fn build(
        index: &LaneIndex,
        blocked: &HashSet<u64>,
        weight_of: impl Fn(LaneAddr) -> Option<f64>,
    ) -> Self {
        let mut graph = Self {
            loc_index: HashMap::new(),
            loc_by_index: Vec::new(),
            preds: Vec::new(),
        };

        for (mt, bus_id, zone_id, dir) in index.bus_groups() {
            for &lane in index.lanes_for(mt, bus_id, zone_id, dir) {
                let Some((src, dst)) = index.endpoints(&lane) else {
                    continue;
                };
                let (src_enc, dst_enc) = (src.encode(), dst.encode());
                if blocked.contains(&src_enc) || blocked.contains(&dst_enc) {
                    continue;
                }
                let Some(w) = weight_of(lane) else {
                    continue;
                };
                let src_idx = graph.intern(src_enc);
                let dst_idx = graph.intern(dst_enc);
                graph.preds[dst_idx].push((src_idx, w));
            }
        }
        graph
    }

    /// Intern `encoded`, returning its compact index.
    ///
    /// Public so callers can give an isolated target an index even when no
    /// surviving lane touches it — without which `distance(t, t) == 0` would
    /// report "unknown location" instead.
    pub(crate) fn intern(&mut self, encoded: u64) -> usize {
        if let Some(&idx) = self.loc_index.get(&encoded) {
            return idx;
        }
        let idx = self.loc_by_index.len();
        self.loc_index.insert(encoded, idx);
        self.loc_by_index.push(encoded);
        self.preds.push(Vec::new());
        idx
    }

    pub(crate) fn index_of(&self, encoded: u64) -> Option<usize> {
        self.loc_index.get(&encoded).copied()
    }

    pub(crate) fn encoded_at(&self, idx: usize) -> u64 {
        self.loc_by_index[idx]
    }

    pub(crate) fn len(&self) -> usize {
        self.loc_by_index.len()
    }

    /// Take ownership of the interning maps, for tables that keep them.
    pub(crate) fn into_index(self) -> (HashMap<u64, usize>, Vec<u64>) {
        (self.loc_index, self.loc_by_index)
    }

    /// Unweighted BFS from `source_idx` over the reversed edges.
    ///
    /// Returns hop counts by compact index, [`u32::MAX`] where unreachable.
    /// Kept separate from [`Self::dijkstra_from`] rather than folded into it at
    /// unit weight: BFS is O(V+E) against Dijkstra's O(E log V), and this runs
    /// once per target on every solve.
    pub(crate) fn bfs_hops_from(&self, source_idx: usize) -> Vec<u32> {
        let mut dist = vec![u32::MAX; self.len()];
        let mut queue: VecDeque<usize> = VecDeque::new();
        dist[source_idx] = 0;
        queue.push_back(source_idx);

        while let Some(current) = queue.pop_front() {
            let current_dist = dist[current];
            for &(pred, _) in &self.preds[current] {
                if dist[pred] == u32::MAX {
                    dist[pred] = current_dist + 1;
                    queue.push_back(pred);
                }
            }
        }
        dist
    }

    /// Weighted Dijkstra from `source_idx` over the reversed edges.
    ///
    /// Returns costs by compact index, [`f64::INFINITY`] where unreachable.
    pub(crate) fn dijkstra_from(&self, source_idx: usize) -> Vec<f64> {
        let mut dist = vec![f64::INFINITY; self.len()];
        dist[source_idx] = 0.0;
        let mut heap = BinaryHeap::new();
        heap.push(DijkstraEntry {
            cost: 0.0,
            node: source_idx,
        });

        while let Some(entry) = heap.pop() {
            // Stale heap entry — a shorter path to this node was settled.
            if entry.cost > dist[entry.node] {
                continue;
            }
            for &(pred, w) in &self.preds[entry.node] {
                let new_cost = entry.cost + w;
                if new_cost < dist[pred] {
                    dist[pred] = new_cost;
                    heap.push(DijkstraEntry {
                        cost: new_cost,
                        node: pred,
                    });
                }
            }
        }
        dist
    }
}

/// Min-heap entry for [`ReverseLaneGraph::dijkstra_from`].
///
/// `Ord` is **reversed** so `BinaryHeap` (a max-heap) pops the cheapest node.
struct DijkstraEntry {
    cost: f64,
    node: usize,
}

impl Eq for DijkstraEntry {}

impl PartialEq for DijkstraEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cost.total_cmp(&other.cost) == std::cmp::Ordering::Equal
    }
}

impl Ord for DijkstraEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.cost.total_cmp(&self.cost)
    }
}

impl PartialOrd for DijkstraEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::example_arch_json;
    use bloqade_lanes_bytecode_core::arch::types::ArchSpec;

    fn make_index() -> LaneIndex {
        let spec: ArchSpec = serde_json::from_str(example_arch_json()).unwrap();
        LaneIndex::new(spec)
    }

    fn unit(index: &LaneIndex) -> ReverseLaneGraph {
        ReverseLaneGraph::build(index, &HashSet::new(), |_| Some(1.0))
    }

    /// At unit weight the two traversals must agree exactly, for every node.
    /// This is what licenses keeping BFS for hop counts and Dijkstra for
    /// weights instead of folding them together.
    #[test]
    fn bfs_and_dijkstra_agree_at_unit_weight() {
        let index = make_index();
        let graph = unit(&index);
        let mut compared = 0_usize;
        for source in 0..graph.len() {
            let hops = graph.bfs_hops_from(source);
            let costs = graph.dijkstra_from(source);
            for idx in 0..graph.len() {
                match (hops[idx], costs[idx]) {
                    (u32::MAX, c) => assert!(c.is_infinite(), "hop-unreachable but cost {c}"),
                    (h, c) => assert_eq!(f64::from(h), c, "hop {h} vs cost {c}"),
                }
                compared += 1;
            }
        }
        assert!(compared > 0, "fixture should expose locations");
    }

    /// `weight_of` returning `None` drops the lane, so a graph that accepts
    /// nothing has no edges at all — the mechanism the duration-weighted table
    /// relies on to skip lanes without transport paths.
    #[test]
    fn a_rejecting_weight_drops_every_edge() {
        let index = make_index();
        let graph = ReverseLaneGraph::build(&index, &HashSet::new(), |_| None);
        assert_eq!(graph.len(), 0, "no lane accepted, so nothing interned");
    }

    /// Blocked locations are carved out before weighting, so they are absent
    /// from the graph entirely rather than present but unreachable.
    #[test]
    fn blocked_locations_are_not_interned() {
        let index = make_index();
        let open = unit(&index);
        let victim = open.encoded_at(0);
        let blocked: HashSet<u64> = [victim].into_iter().collect();
        let carved = ReverseLaneGraph::build(&index, &blocked, |_| Some(1.0));
        assert_eq!(carved.index_of(victim), None);
        assert!(carved.len() < open.len());
    }

    /// Interning is idempotent, so a target already reached by a lane keeps its
    /// index rather than getting a second one.
    #[test]
    fn interning_is_idempotent() {
        let index = make_index();
        let mut graph = unit(&index);
        let existing = graph.encoded_at(0);
        let before = graph.len();
        assert_eq!(graph.intern(existing), 0);
        assert_eq!(graph.len(), before);

        // A location no lane touches gets a fresh index and an empty pred list.
        let fresh = graph.intern(0xDEAD_BEEF);
        assert_eq!(fresh, before);
        assert_eq!(graph.len(), before + 1);
        assert!(graph.dijkstra_from(fresh)[fresh] == 0.0);
    }
}
