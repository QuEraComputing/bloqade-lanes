//! Undirected view of the lane graph, plus biconnected-component
//! decomposition.
//!
//! The feasibility analysis in [`super`] is a pebble-motion argument, and
//! pebble motion is defined over a *simple undirected* graph. The lane graph
//! is stored directionally ([`LaneAddr`] carries a
//! [`Direction`](bloqade_lanes_bytecode_core::arch::addr::Direction)), but the
//! direction field only selects which endpoint is reported as source vs
//! destination — every lane is traversable both ways. [`LaneGraph`] collapses
//! that into an undirected adjacency over compact vertex ids, dropping
//! self-loops and duplicate edges so the result is simple.
//!
//! Blocked locations are removed from the graph entirely rather than marked
//! occupied: an atom can never enter one, so for reachability purposes they
//! are not vertices at all. Note the consequence — a blocked location is *not*
//! an empty vertex, so it does not contribute to `m`.

use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;

use crate::primitives::lane_index::LaneIndex;

/// Compact vertex identifier — an index into [`LaneGraph::locations`].
pub type VertexId = usize;

/// Undirected, simple view of the lane graph with blocked locations removed.
#[derive(Debug)]
pub struct LaneGraph {
    /// Vertex id → encoded [`LocationAddr`](bloqade_lanes_bytecode_core::arch::addr::LocationAddr).
    locations: Vec<u64>,
    /// Encoded location → vertex id.
    by_location: HashMap<u64, VertexId>,
    /// Adjacency lists, sorted and deduplicated, without self-loops.
    adj: Vec<Vec<VertexId>>,
}

impl LaneGraph {
    /// Build the undirected lane graph, excluding every location in
    /// `blocked`.
    ///
    /// Vertices are the unblocked lane endpoints reachable through
    /// [`LaneIndex::bus_groups`] — the same location set that
    /// [`LaneIndex::num_locations`] counts, minus `blocked`. A location that
    /// is not an endpoint of any lane cannot participate in any move and is
    /// intentionally absent; a location whose lanes have all been cut by
    /// blocking is retained as an isolated vertex.
    pub fn build(index: &LaneIndex, blocked: &HashSet<u64>) -> Self {
        // Collect endpoints first, then assign vertex ids in sorted order.
        //
        // `LaneIndex::bus_groups` iterates a `HashMap`, so discovery order
        // varies between processes. Numbering vertices as they are discovered
        // would make the ids — and therefore every neighbour list, every
        // BFS tie-break, and every downstream traversal order — differ run to
        // run. The feasibility *verdict* is order-independent, but consumers
        // that pick among equally-good options (a planner choosing which lane
        // to take) would silently produce a different answer each run.
        let mut endpoint_pairs: Vec<(u64, u64)> = Vec::new();
        let mut seen: HashSet<u64> = HashSet::new();
        let mut locations: Vec<u64> = Vec::new();

        for (mt, bus_id, zone_id, dir) in index.bus_groups() {
            for lane in index.lanes_for(mt, bus_id, zone_id, dir) {
                let Some((src, dst)) = index.endpoints(lane) else {
                    continue;
                };
                let (src_enc, dst_enc) = (src.encode(), dst.encode());
                // Keep each unblocked endpoint even when the lane itself is
                // dropped. A location whose every lane leads to a blocked
                // site is still a location: an atom can sit on it, and it
                // counts toward `m`. Dropping it would both misreport such an
                // atom as "not on the graph" and understate the empty count.
                for enc in [src_enc, dst_enc] {
                    if !blocked.contains(&enc) && seen.insert(enc) {
                        locations.push(enc);
                    }
                }
                if !blocked.contains(&src_enc) && !blocked.contains(&dst_enc) {
                    endpoint_pairs.push((src_enc, dst_enc));
                }
            }
        }
        locations.sort_unstable();

        let by_location: HashMap<u64, VertexId> =
            locations.iter().enumerate().map(|(i, &l)| (l, i)).collect();

        let mut adj: Vec<Vec<VertexId>> = vec![Vec::new(); locations.len()];
        for (src_enc, dst_enc) in endpoint_pairs {
            let a = by_location[&src_enc];
            let b = by_location[&dst_enc];
            if a != b {
                adj[a].push(b);
                adj[b].push(a);
            }
        }
        for list in &mut adj {
            list.sort_unstable();
            list.dedup();
        }

        Self {
            locations,
            by_location,
            adj,
        }
    }

    /// Number of vertices.
    pub fn len(&self) -> usize {
        self.locations.len()
    }

    /// Whether the graph has no vertices.
    pub fn is_empty(&self) -> bool {
        self.locations.is_empty()
    }

    /// Vertex id for an encoded location, if present.
    pub fn vertex_of(&self, encoded: u64) -> Option<VertexId> {
        self.by_location.get(&encoded).copied()
    }

    /// Encoded location for a vertex id.
    pub fn location_of(&self, v: VertexId) -> u64 {
        self.locations[v]
    }

    /// Neighbours of `v`.
    pub fn neighbors(&self, v: VertexId) -> &[VertexId] {
        &self.adj[v]
    }

    /// Degree of `v`.
    pub fn degree(&self, v: VertexId) -> usize {
        self.adj[v].len()
    }

    /// Iterate every vertex id.
    pub fn vertices(&self) -> impl Iterator<Item = VertexId> + '_ {
        0..self.locations.len()
    }

    /// Connected-component id per vertex, and the component count.
    pub fn connected_components(&self) -> (Vec<usize>, usize) {
        let n = self.len();
        let mut comp = vec![usize::MAX; n];
        let mut count = 0;
        let mut queue: VecDeque<VertexId> = VecDeque::new();
        for start in 0..n {
            if comp[start] != usize::MAX {
                continue;
            }
            comp[start] = count;
            queue.push_back(start);
            while let Some(u) = queue.pop_front() {
                for &w in &self.adj[u] {
                    if comp[w] == usize::MAX {
                        comp[w] = count;
                        queue.push_back(w);
                    }
                }
            }
            count += 1;
        }
        (comp, count)
    }

    /// BFS hop distances from `from`. Unreachable vertices get [`u32::MAX`].
    ///
    /// `skip` is consulted per vertex; a vertex for which it returns `true`
    /// is treated as absent from the graph (never entered, never expanded).
    /// `from` itself is always entered regardless of `skip`.
    pub fn distances_from(&self, from: VertexId, skip: impl Fn(VertexId) -> bool) -> Vec<u32> {
        let mut dist = vec![u32::MAX; self.len()];
        dist[from] = 0;
        let mut queue = VecDeque::new();
        queue.push_back(from);
        while let Some(u) = queue.pop_front() {
            let d = dist[u];
            for &w in &self.adj[u] {
                if dist[w] == u32::MAX && !skip(w) {
                    dist[w] = d + 1;
                    queue.push_back(w);
                }
            }
        }
        dist
    }

    /// Biconnected-component decomposition (Hopcroft–Tarjan), `O(V + E)`.
    ///
    /// Implemented iteratively: the recursive formulation would recurse to
    /// the DFS depth, which is `O(V)` on a corridor-heavy architecture.
    pub fn biconnected(&self) -> Biconnected {
        let n = self.len();
        let mut disc = vec![0u32; n];
        let mut low = vec![0u32; n];
        let mut timer: u32 = 1;
        let mut edge_stack: Vec<(VertexId, VertexId)> = Vec::new();
        let mut components: Vec<Vec<VertexId>> = Vec::new();
        let mut edge_counts: Vec<usize> = Vec::new();
        let mut by_vertex: Vec<Vec<usize>> = vec![Vec::new(); n];

        // Drain `edge_stack` down to and including the edge `(p, u)`, and
        // record the popped edges as one biconnected component.
        let emit = |edge_stack: &mut Vec<(VertexId, VertexId)>,
                    p: VertexId,
                    u: VertexId,
                    components: &mut Vec<Vec<VertexId>>,
                    edge_counts: &mut Vec<usize>,
                    by_vertex: &mut Vec<Vec<usize>>| {
            let comp_id = components.len();
            let mut verts: Vec<VertexId> = Vec::new();
            let mut edges = 0usize;
            while let Some((a, b)) = edge_stack.pop() {
                verts.push(a);
                verts.push(b);
                edges += 1;
                if (a, b) == (p, u) {
                    break;
                }
            }
            verts.sort_unstable();
            verts.dedup();
            for &v in &verts {
                by_vertex[v].push(comp_id);
            }
            components.push(verts);
            edge_counts.push(edges);
        };

        for start in 0..n {
            if disc[start] != 0 {
                continue;
            }
            disc[start] = timer;
            low[start] = timer;
            timer += 1;
            // Frame: (vertex, parent, next index into adj[vertex]).
            let mut stack: Vec<(VertexId, VertexId, usize)> = vec![(start, usize::MAX, 0)];

            while let Some(&(u, parent, i)) = stack.last() {
                if i < self.adj[u].len() {
                    stack.last_mut().expect("stack is non-empty").2 += 1;
                    let v = self.adj[u][i];
                    if v == parent {
                        continue;
                    }
                    if disc[v] == 0 {
                        edge_stack.push((u, v));
                        disc[v] = timer;
                        low[v] = timer;
                        timer += 1;
                        stack.push((v, u, 0));
                    } else if disc[v] < disc[u] {
                        // Back edge to an ancestor.
                        edge_stack.push((u, v));
                        low[u] = low[u].min(disc[v]);
                    }
                } else {
                    stack.pop();
                    if let Some(&(p, _, _)) = stack.last() {
                        low[p] = low[p].min(low[u]);
                        if low[u] >= disc[p] {
                            emit(
                                &mut edge_stack,
                                p,
                                u,
                                &mut components,
                                &mut edge_counts,
                                &mut by_vertex,
                            );
                        }
                    }
                }
            }
        }

        Biconnected {
            components,
            edge_counts,
            by_vertex,
        }
    }
}

#[cfg(test)]
impl LaneGraph {
    /// Build a graph directly from an edge list, bypassing the architecture.
    ///
    /// Lets the graph and decomposition tests use the small hand-checkable
    /// examples the pebble-motion literature is stated over, and lets them
    /// cross-check verdicts against brute-force reachability.
    pub(crate) fn from_edges(n: usize, edges: &[(VertexId, VertexId)]) -> Self {
        let mut adj = vec![Vec::new(); n];
        for &(a, b) in edges {
            adj[a].push(b);
            adj[b].push(a);
        }
        for list in &mut adj {
            list.sort_unstable();
            list.dedup();
        }
        Self {
            locations: (0..n as u64).collect(),
            by_location: (0..n as u64).map(|i| (i, i as usize)).collect(),
            adj,
        }
    }
}

/// Biconnected components of a [`LaneGraph`].
#[derive(Debug)]
pub struct Biconnected {
    /// Component id → the vertices incident to that component's edges.
    pub components: Vec<Vec<VertexId>>,
    /// Component id → number of edges in the component.
    pub edge_counts: Vec<usize>,
    /// Vertex id → the component ids it belongs to.
    pub by_vertex: Vec<Vec<usize>>,
}

impl Biconnected {
    /// A component is *nontrivial* when it contains more than one edge —
    /// equivalently, when it contains a cycle. Definition 1 of de Wilde et
    /// al. (2014) calls single-edge components trivial; only nontrivial
    /// components let agents exchange position.
    pub fn is_nontrivial(&self, comp_id: usize) -> bool {
        self.edge_counts[comp_id] > 1
    }

    /// Join vertices (Definition 2): degree ≥ 3 and common to at least two
    /// biconnected components.
    pub fn join_vertices(&self, graph: &LaneGraph) -> Vec<VertexId> {
        graph
            .vertices()
            .filter(|&v| graph.degree(v) >= 3 && self.by_vertex[v].len() >= 2)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn from_edges(n: usize, edges: &[(VertexId, VertexId)]) -> LaneGraph {
        LaneGraph::from_edges(n, edges)
    }

    #[test]
    fn triangle_is_one_nontrivial_component() {
        let g = from_edges(3, &[(0, 1), (1, 2), (2, 0)]);
        let bcc = g.biconnected();
        assert_eq!(bcc.components.len(), 1);
        assert!(bcc.is_nontrivial(0));
        assert_eq!(bcc.edge_counts[0], 3);
    }

    #[test]
    fn path_is_all_trivial_components() {
        // 0 — 1 — 2 — 3: three bridges, each its own trivial component.
        let g = from_edges(4, &[(0, 1), (1, 2), (2, 3)]);
        let bcc = g.biconnected();
        assert_eq!(bcc.components.len(), 3);
        assert!(bcc.components.iter().enumerate().all(|(i, _)| {
            assert_eq!(bcc.edge_counts[i], 1);
            !bcc.is_nontrivial(i)
        }));
    }

    #[test]
    fn two_triangles_joined_by_a_bridge() {
        // Triangle {0,1,2} — bridge (2,3) — triangle {3,4,5}.
        let g = from_edges(6, &[(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3)]);
        let bcc = g.biconnected();
        assert_eq!(bcc.components.len(), 3, "two triangles plus one bridge");
        let nontrivial: Vec<usize> = (0..bcc.components.len())
            .filter(|&i| bcc.is_nontrivial(i))
            .collect();
        assert_eq!(nontrivial.len(), 2);
        // Vertices 2 and 3 are cut vertices, each in two components.
        assert_eq!(bcc.by_vertex[2].len(), 2);
        assert_eq!(bcc.by_vertex[3].len(), 2);
        assert_eq!(bcc.by_vertex[0].len(), 1);
    }

    #[test]
    fn join_vertices_need_degree_three_and_two_components() {
        // Triangle {0,1,2} with a pendant path 2 — 3 — 4.
        // Vertex 2 has degree 3 and sits in two components → join vertex.
        // Vertex 3 has degree 2 → not a join vertex despite two components.
        let g = from_edges(5, &[(0, 1), (1, 2), (2, 0), (2, 3), (3, 4)]);
        let bcc = g.biconnected();
        assert_eq!(bcc.join_vertices(&g), vec![2]);
    }

    #[test]
    fn disconnected_graph_decomposes_per_component() {
        // Triangle {0,1,2} and a separate edge (3,4).
        let g = from_edges(5, &[(0, 1), (1, 2), (2, 0), (3, 4)]);
        let bcc = g.biconnected();
        assert_eq!(bcc.components.len(), 2);
        let (comp, count) = g.connected_components();
        assert_eq!(count, 2);
        assert_eq!(comp[0], comp[1]);
        assert_ne!(comp[0], comp[3]);
    }

    #[test]
    fn deep_path_does_not_overflow_the_stack() {
        // The recursive Hopcroft–Tarjan formulation would recurse 200k
        // deep here; the iterative one must not.
        let n = 200_000;
        let edges: Vec<(VertexId, VertexId)> = (0..n - 1).map(|i| (i, i + 1)).collect();
        let g = from_edges(n, &edges);
        let bcc = g.biconnected();
        assert_eq!(bcc.components.len(), n - 1);
    }

    #[test]
    fn distances_from_respects_skip() {
        // 0 — 1 — 2 — 3, with vertex 2 skipped: 3 becomes unreachable.
        let g = from_edges(4, &[(0, 1), (1, 2), (2, 3)]);
        let d = g.distances_from(0, |v| v == 2);
        assert_eq!(d[1], 1);
        assert_eq!(d[2], u32::MAX);
        assert_eq!(d[3], u32::MAX);
    }
}
