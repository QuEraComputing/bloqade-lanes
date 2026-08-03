//! Kornhauser subgraph decomposition, as reconstructed by de Wilde, ter Mors
//! & Witteveen (JAIR 51, 2014), §3.1.
//!
//! Three stages, matching Algorithms 1–3 of that paper:
//!
//! 1. [`find_subgraphs`] (Algorithm 1) — partition the graph into subgraphs
//!    within which agents can reach any position.
//! 2. [`assign_agents`] (Algorithm 2) — decide which agents are *confined* to
//!    each subgraph and its planks.
//! 3. [`subgraph_priorities`] (Algorithm 3) — the partial order over
//!    subgraphs; a cycle in it proves the instance unsolvable (Proposition 2).
//!
//! Throughout, `m` is the number of empty vertices, matching the paper's
//! notation. Everything here assumes the ≥ 2 empty-vertex regime that
//! Push and Rotate is complete for.
//!
//! ## Soundness bias
//!
//! This module backs a *one-sided* infeasibility detector: it must never
//! report a solvable instance as infeasible, but it is allowed to miss
//! obstructions. Where the paper's pseudocode is ambiguous, we therefore take
//! the reading that assigns *fewer* agents to subgraphs. Under-assignment is
//! safe in both directions it matters: an unassigned agent is simply not
//! goal-containment checked, and it contributes no precedence edges that
//! could create a spurious cycle. Over-assignment would be unsound, since
//! goal containment is only justified for agents Proposition 1 proves are
//! confined.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::feasibility::graph::{Biconnected, LaneGraph, VertexId};

/// Sentinel for "this vertex belongs to no subgraph".
pub const NO_SUBGRAPH: usize = usize::MAX;

/// A plank of a subgraph (Definition 5): a unique, maximal path leaving the
/// subgraph at a join vertex, of length at most `m - 1`.
#[derive(Debug, Clone)]
pub struct Plank {
    /// The join vertex where the plank attaches. Belongs to the subgraph.
    pub start: VertexId,
    /// Plank vertices ordered outward from `start`, excluding `start`.
    pub vertices: Vec<VertexId>,
}

/// The full decomposition of an instance.
#[derive(Debug)]
pub struct Decomposition {
    /// Subgraph id → its vertex set, sorted.
    pub subgraphs: Vec<Vec<VertexId>>,
    /// Vertex id → owning subgraph, or [`NO_SUBGRAPH`].
    pub subgraph_of_vertex: Vec<usize>,
    /// Subgraph id → its planks.
    pub planks: Vec<Vec<Plank>>,
    /// Qubit id → the subgraph it is confined to. Absent means the agent is
    /// confined to an isthmus and belongs to no subgraph.
    pub assignment: HashMap<u32, usize>,
    /// Number of empty vertices, `m`.
    pub empty_count: usize,
}

impl Decomposition {
    /// Build the full decomposition for an occupancy pattern.
    ///
    /// `occupant[v]` is the qubit at vertex `v`, if any, and `empty_count` is
    /// the number of empty vertices `m`. Assumes the `m ≥ 2` regime — callers
    /// gate on [`crate::feasibility::MIN_EMPTY_VERTICES`].
    pub fn build(graph: &LaneGraph, occupant: &[Option<u32>], empty_count: usize) -> Self {
        let bcc = graph.biconnected();
        let (subgraphs, subgraph_of_vertex) = find_subgraphs(graph, &bcc, empty_count);
        let planks = find_planks(graph, &subgraphs, &subgraph_of_vertex, empty_count);
        let assignment = assign_agents(
            graph,
            &subgraphs,
            &subgraph_of_vertex,
            occupant,
            empty_count,
        );
        Self {
            subgraphs,
            subgraph_of_vertex,
            planks,
            assignment,
            empty_count,
        }
    }

    /// Whether `v` lies in subgraph `sub` or on one of its planks — the
    /// region Proposition 1 confines an assigned agent to.
    pub fn contains_in_subgraph_or_planks(&self, sub: usize, v: VertexId) -> bool {
        if self.subgraph_of_vertex[v] == sub {
            return true;
        }
        self.planks[sub].iter().any(|p| p.vertices.contains(&v))
    }
}

/// Algorithm 1: `find_subgraphs(G, m)`.
///
/// Starts from the nontrivial biconnected components plus the isolated
/// degree-≥3 vertices, then repeatedly merges any two classes whose vertex
/// sets come within `m - 2` hops, absorbing the connecting shortest path
/// (Definition 3 condition 3, and Definition 4 condition 3). Distance 0 —
/// two components sharing a cut vertex — is within `m - 2` hops for every
/// `m ≥ 2`, so vertex-sharing classes always merge in the regime this
/// decomposition runs in.
///
/// The merge loop is `O(rounds × |S| × (V + E))`. The result depends only on
/// the architecture, the blocked set, and `m`, so callers that solve many
/// layers against one architecture should cache it rather than recomputing
/// per solve.
pub fn find_subgraphs(
    graph: &LaneGraph,
    bcc: &Biconnected,
    empty_count: usize,
) -> (Vec<Vec<VertexId>>, Vec<usize>) {
    let n = graph.len();
    let mut owner = vec![NO_SUBGRAPH; n];
    let mut sets: Vec<Option<Vec<VertexId>>> = Vec::new();

    // Line 1: every nontrivial biconnected component becomes a class.
    for comp_id in 0..bcc.components.len() {
        if !bcc.is_nontrivial(comp_id) {
            continue;
        }
        let verts = bcc.components[comp_id].clone();
        let id = sets.len();
        for &v in &verts {
            owner[v] = id;
        }
        sets.push(Some(verts));
    }

    // Line 2: degree-≥3 vertices not already inside a nontrivial component.
    for v in graph.vertices() {
        if graph.degree(v) >= 3 && owner[v] == NO_SUBGRAPH {
            let id = sets.len();
            owner[v] = id;
            sets.push(Some(vec![v]));
        }
    }

    // Lines 3–5: merge classes within `m - 2` hops of each other, pulling in
    // the vertices of the connecting shortest path. `m < 2` makes the
    // threshold vacuous, so the loop is skipped entirely.
    if empty_count >= 2 {
        let radius = empty_count - 2;
        let mut changed = true;
        while changed {
            changed = false;
            for i in 0..sets.len() {
                if sets[i].is_none() {
                    continue;
                }
                // Multi-source BFS out of class `i`, bounded by `radius`,
                // recording parents so the connecting path can be absorbed.
                let sources = sets[i].as_ref().expect("checked above").clone();
                let mut dist = vec![u32::MAX; n];
                let mut parent = vec![usize::MAX; n];
                let mut queue: VecDeque<VertexId> = VecDeque::new();
                let mut hits: Vec<VertexId> = Vec::new();
                for &s in &sources {
                    dist[s] = 0;
                    // Distance 0: a vertex this class shares with another
                    // class (a cut vertex common to two nontrivial biconnected
                    // components). The BFS below can only discover vertices at
                    // distance ≥ 1, so shared vertices must be caught here —
                    // missing them left vertex-sharing components unmerged and
                    // produced provably wrong confinement claims at `m = 2`.
                    if owner[s] != i && owner[s] != NO_SUBGRAPH {
                        hits.push(s);
                    }
                    queue.push_back(s);
                }
                while let Some(u) = queue.pop_front() {
                    if dist[u] as usize >= radius {
                        continue;
                    }
                    for &w in graph.neighbors(u) {
                        if dist[w] != u32::MAX {
                            continue;
                        }
                        dist[w] = dist[u] + 1;
                        parent[w] = u;
                        if owner[w] != NO_SUBGRAPH && owner[w] != i {
                            hits.push(w);
                        }
                        queue.push_back(w);
                    }
                }

                for hit in hits {
                    let j = owner[hit];
                    if j == NO_SUBGRAPH || j == i || sets[j].is_none() {
                        continue;
                    }
                    // Absorb class `j` and the path connecting it to `i`.
                    let other = sets[j].take().expect("checked above");
                    let mut merged = sets[i].take().expect("class i is live");
                    merged.extend(other);
                    let mut step = hit;
                    while step != usize::MAX && dist[step] > 0 {
                        merged.push(step);
                        step = parent[step];
                    }
                    merged.sort_unstable();
                    merged.dedup();
                    for &v in &merged {
                        owner[v] = i;
                    }
                    sets[i] = Some(merged);
                    changed = true;
                }
            }
        }
    }

    // Compact: drop the merged-away holes and renumber.
    let subgraphs: Vec<Vec<VertexId>> = sets.into_iter().flatten().collect();
    let mut subgraph_of_vertex = vec![NO_SUBGRAPH; n];
    for (new_id, verts) in subgraphs.iter().enumerate() {
        for &v in verts {
            subgraph_of_vertex[v] = new_id;
        }
    }

    (subgraphs, subgraph_of_vertex)
}

/// Walk the unique path leaving `start` through its neighbour `first`,
/// collecting at most `max_len` vertices (`start` itself excluded).
///
/// A plank (Definition 5) is a *unique* path, so the walk stops after a
/// branch vertex (degree ≠ 2 has no forced continuation) or a vertex that
/// belongs to a subgraph; that terminal vertex is included in the result.
/// This is the single definition of "the plank leaving `start` via `first`" —
/// both [`find_planks`] and [`assign_agents`] traverse it, so the region an
/// agent is confined to and the set of agents claimed by a subgraph cannot
/// drift apart.
fn walk_plank(
    graph: &LaneGraph,
    subgraph_of_vertex: &[usize],
    start: VertexId,
    first: VertexId,
    max_len: usize,
) -> Vec<VertexId> {
    let mut path = Vec::new();
    if max_len == 0 {
        return path;
    }
    let mut prev = start;
    let mut cur = first;
    loop {
        path.push(cur);
        if path.len() >= max_len {
            break;
        }
        // A plank is a *unique* path: stop at any branch, and at any vertex
        // that belongs to a subgraph.
        if graph.degree(cur) != 2 || subgraph_of_vertex[cur] != NO_SUBGRAPH {
            break;
        }
        let Some(&next) = graph.neighbors(cur).iter().find(|&&w| w != prev) else {
            break;
        };
        prev = cur;
        cur = next;
    }
    path
}

/// Compute the planks of every subgraph (Definition 5).
///
/// From each subgraph vertex with a neighbour outside the subgraph, walk
/// outward along the forced continuation ([`walk_plank`]). Length is capped
/// at `m - 1` edges.
pub fn find_planks(
    graph: &LaneGraph,
    subgraphs: &[Vec<VertexId>],
    subgraph_of_vertex: &[usize],
    empty_count: usize,
) -> Vec<Vec<Plank>> {
    let max_len = empty_count.saturating_sub(1);
    let mut all = Vec::with_capacity(subgraphs.len());

    for (sub_id, verts) in subgraphs.iter().enumerate() {
        let mut planks = Vec::new();
        if max_len == 0 {
            all.push(planks);
            continue;
        }
        for &s in verts {
            for &u in graph.neighbors(s) {
                if subgraph_of_vertex[u] == sub_id {
                    continue;
                }
                planks.push(Plank {
                    start: s,
                    vertices: walk_plank(graph, subgraph_of_vertex, s, u, max_len),
                });
            }
        }
        all.push(planks);
    }
    all
}

/// Count unoccupied vertices reachable from `sources`, optionally with one
/// vertex and one undirected edge deleted from the graph.
fn count_unoccupied_reachable(
    graph: &LaneGraph,
    sources: &[VertexId],
    occupied: &[bool],
    skip_vertex: Option<VertexId>,
    skip_edge: Option<(VertexId, VertexId)>,
) -> usize {
    let mut seen = vec![false; graph.len()];
    let mut queue: VecDeque<VertexId> = VecDeque::new();
    for &s in sources {
        if Some(s) == skip_vertex || seen[s] {
            continue;
        }
        seen[s] = true;
        queue.push_back(s);
    }
    let mut count = 0usize;
    while let Some(u) = queue.pop_front() {
        if !occupied[u] {
            count += 1;
        }
        for &w in graph.neighbors(u) {
            if seen[w] || Some(w) == skip_vertex {
                continue;
            }
            if let Some((a, b)) = skip_edge
                && ((u == a && w == b) || (u == b && w == a))
            {
                continue;
            }
            seen[w] = true;
            queue.push_back(w);
        }
    }
    count
}

/// Algorithm 2: `assign_agents_to_subgraphs(G, A, A, S)`.
///
/// Returns the map from qubit id to the subgraph that agent is confined to.
/// Agents confined to an isthmus are absent from the map.
///
/// `occupant[v]` is the qubit at vertex `v`, if any.
pub fn assign_agents(
    graph: &LaneGraph,
    subgraphs: &[Vec<VertexId>],
    subgraph_of_vertex: &[usize],
    occupant: &[Option<u32>],
    empty_count: usize,
) -> HashMap<u32, usize> {
    let occupied: Vec<bool> = occupant.iter().map(|o| o.is_some()).collect();
    let mut assignment: HashMap<u32, usize> = HashMap::new();

    for (sub_id, verts) in subgraphs.iter().enumerate() {
        for &v in verts {
            let outside: Vec<VertexId> = graph
                .neighbors(v)
                .iter()
                .copied()
                .filter(|&u| subgraph_of_vertex[u] != sub_id)
                .collect();

            // Line 11–12: an inner vertex (no neighbour outside the
            // subgraph). Its occupant is confined outright.
            if outside.is_empty() {
                if let Some(a) = occupant[v] {
                    assignment.insert(a, sub_id);
                }
                continue;
            }

            // Line 5: empties reachable from the subgraph without using `v`.
            let m_dprime = count_unoccupied_reachable(graph, verts, &occupied, Some(v), None);

            for &u in &outside {
                // Line 7: empties reachable from `v` with edge (u, v) cut.
                let m_prime =
                    count_unoccupied_reachable(graph, &[v], &occupied, None, Some((u, v)));

                // Line 8. Note the `A^-1(v) != ⊥` conjunct: the paper gates
                // the whole body — including the plank walk on line 10 — on
                // the join vertex being occupied. We implement that
                // literally; see the module-level soundness note.
                let gate = (m_prime >= 1 && m_prime < empty_count) || m_dprime >= 1;
                let Some(join_agent) = occupant[v] else {
                    continue;
                };
                if !gate {
                    continue;
                }
                assignment.insert(join_agent, sub_id);

                // Line 10: follow the plank outward from `u`, assigning the
                // first `m' - 1` agents found along it. Only vertices that
                // belong to no subgraph count as plank vertices: an occupant
                // of a neighbouring subgraph's vertex is that subgraph's to
                // assign, and claiming it here would be over-assignment —
                // the unsound direction (see the module-level note).
                let mut budget = m_prime.saturating_sub(1);
                if budget == 0 {
                    continue;
                }
                let max_len = empty_count.saturating_sub(1);
                for &w in &walk_plank(graph, subgraph_of_vertex, v, u, max_len) {
                    if subgraph_of_vertex[w] != NO_SUBGRAPH {
                        break;
                    }
                    if let Some(a) = occupant[w] {
                        assignment.insert(a, sub_id);
                        budget -= 1;
                        if budget == 0 {
                            break;
                        }
                    }
                }
            }
        }
    }

    assignment
}

/// A precedence edge `before ≺ after` between two subgraphs.
pub type Precedence = (usize, usize);

/// Algorithm 3: the partial order over subgraphs.
///
/// For an agent `r` assigned to `S_j` with goal `g`, add `S_i ≺ S_j` when
/// either (1) `g` is the start vertex of a plank of `S_i`, or (2) `g` lies on
/// a plank of `S_i` and every plank vertex between `g` and the plank start is
/// the goal of an agent assigned to no subgraph.
pub fn subgraph_priorities(
    decomp: &Decomposition,
    goals: &HashMap<u32, VertexId>,
    unassigned_goal_vertices: &HashSet<VertexId>,
) -> Vec<Precedence> {
    let mut edges: HashSet<Precedence> = HashSet::new();

    for (&agent, &sub_j) in &decomp.assignment {
        let Some(&goal) = goals.get(&agent) else {
            continue;
        };
        for (sub_i, planks) in decomp.planks.iter().enumerate() {
            if sub_i == sub_j {
                continue;
            }
            for plank in planks {
                // Case 1: the goal is the plank's start vertex.
                if plank.start == goal {
                    edges.insert((sub_i, sub_j));
                    continue;
                }
                // Case 2: the goal sits on the plank, and everything between
                // it and the start is claimed by unassigned agents.
                let Some(pos) = plank.vertices.iter().position(|&w| w == goal) else {
                    continue;
                };
                let between_all_unassigned = plank.vertices[..pos]
                    .iter()
                    .all(|w| unassigned_goal_vertices.contains(w));
                if between_all_unassigned {
                    edges.insert((sub_i, sub_j));
                }
            }
        }
    }

    let mut out: Vec<Precedence> = edges.into_iter().collect();
    out.sort_unstable();
    out
}

/// Find a cycle in the subgraph precedence relation, if one exists.
///
/// Proposition 2: a cyclic precedence relation proves the instance
/// unsolvable. Returns the subgraph ids on the cycle, in order.
pub fn find_precedence_cycle(n_subgraphs: usize, edges: &[Precedence]) -> Option<Vec<usize>> {
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n_subgraphs];
    for &(a, b) in edges {
        adj[a].push(b);
    }

    // Iterative DFS with an explicit colour marking: 0 = unvisited,
    // 1 = on the current path, 2 = fully explored.
    let mut colour = vec![0u8; n_subgraphs];
    let mut path: Vec<usize> = Vec::new();

    for start in 0..n_subgraphs {
        if colour[start] != 0 {
            continue;
        }
        let mut stack: Vec<(usize, usize)> = vec![(start, 0)];
        colour[start] = 1;
        path.push(start);
        while let Some(&(u, i)) = stack.last() {
            if i < adj[u].len() {
                stack.last_mut().expect("stack is non-empty").1 += 1;
                let w = adj[u][i];
                match colour[w] {
                    0 => {
                        colour[w] = 1;
                        path.push(w);
                        stack.push((w, 0));
                    }
                    1 => {
                        // Back edge — the cycle is the path suffix from `w`.
                        let at = path
                            .iter()
                            .position(|&x| x == w)
                            .expect("a grey vertex is on the current path");
                        return Some(path[at..].to_vec());
                    }
                    _ => {}
                }
            } else {
                colour[u] = 2;
                path.pop();
                stack.pop();
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two triangles joined by a corridor:
    /// `{0,1,2}` — 2 – 3 – 4 – 5 – 6 — `{6,7,8}`.
    ///
    /// Atoms cannot pass each other in the corridor, so whether an atom can
    /// cross between the triangles depends entirely on how many empty
    /// vertices there are — exactly the regime the plank machinery encodes.
    fn dumbbell() -> LaneGraph {
        LaneGraph::from_edges(
            9,
            &[
                (0, 1),
                (1, 2),
                (2, 0),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 6),
            ],
        )
    }

    fn decompose(graph: &LaneGraph, occupant: &[Option<u32>]) -> Decomposition {
        let empty_count = occupant.iter().filter(|o| o.is_none()).count();
        Decomposition::build(graph, occupant, empty_count)
    }

    /// Exhaustive BFS over the configuration space: for every agent, the set
    /// of vertices it can ever occupy from `occupant`, moving one atom at a
    /// time into an empty neighbour. Ground truth for the confinement claims.
    fn reachable_vertices(
        graph: &LaneGraph,
        occupant: &[Option<u32>],
    ) -> HashMap<u32, HashSet<VertexId>> {
        let mut reach: HashMap<u32, HashSet<VertexId>> = HashMap::new();
        let record = |reach: &mut HashMap<u32, HashSet<VertexId>>, state: &[Option<u32>]| {
            for (v, o) in state.iter().enumerate() {
                if let Some(q) = o {
                    reach.entry(*q).or_default().insert(v);
                }
            }
        };
        let start: Vec<Option<u32>> = occupant.to_vec();
        let mut seen: HashSet<Vec<Option<u32>>> = HashSet::new();
        let mut queue: VecDeque<Vec<Option<u32>>> = VecDeque::new();
        record(&mut reach, &start);
        seen.insert(start.clone());
        queue.push_back(start);
        while let Some(state) = queue.pop_front() {
            for v in graph.vertices() {
                let Some(q) = state[v] else { continue };
                for &w in graph.neighbors(v) {
                    if state[w].is_some() {
                        continue;
                    }
                    let mut next = state.clone();
                    next[v] = None;
                    next[w] = Some(q);
                    if seen.insert(next.clone()) {
                        record(&mut reach, &next);
                        queue.push_back(next);
                    }
                }
            }
        }
        reach
    }

    /// Exhaustive BFS over the configuration space: can `agent` ever reach
    /// `goal` from `occupant`, moving one atom at a time into an empty
    /// neighbour? Ground truth for the confinement claims.
    fn agent_can_reach(
        graph: &LaneGraph,
        occupant: &[Option<u32>],
        agent: u32,
        goal: VertexId,
    ) -> bool {
        let start: Vec<Option<u32>> = occupant.to_vec();
        let mut seen: HashSet<Vec<Option<u32>>> = HashSet::new();
        let mut queue: VecDeque<Vec<Option<u32>>> = VecDeque::new();
        seen.insert(start.clone());
        queue.push_back(start);
        while let Some(state) = queue.pop_front() {
            if state[goal] == Some(agent) {
                return true;
            }
            for v in graph.vertices() {
                let Some(q) = state[v] else { continue };
                for &w in graph.neighbors(v) {
                    if state[w].is_some() {
                        continue;
                    }
                    let mut next = state.clone();
                    next[v] = None;
                    next[w] = Some(q);
                    if seen.insert(next.clone()) {
                        queue.push_back(next);
                    }
                }
            }
        }
        false
    }

    /// With two empty vertices the corridor is impassable, so an atom in the
    /// left triangle is confined there — and the decomposition must say so.
    /// Cross-checked against exhaustive search.
    #[test]
    fn confinement_matches_brute_force_when_corridor_is_blocked() {
        let graph = dumbbell();
        // 7 atoms, 2 empties (vertices 1 and 7 left free).
        let occupant: Vec<Option<u32>> = vec![
            Some(0), // v0, left triangle
            None,    // v1
            Some(1), // v2, join vertex
            Some(2), // v3, corridor
            Some(3), // v4, corridor
            Some(4), // v5, corridor
            Some(5), // v6, join vertex
            None,    // v7
            Some(6), // v8, right triangle
        ];
        assert_eq!(occupant.iter().filter(|o| o.is_none()).count(), 2);

        let decomp = decompose(&graph, &occupant);
        let sub_of_agent0 = decomp
            .assignment
            .get(&0)
            .copied()
            .expect("an atom on an inner vertex is always assigned");

        // Ground truth: agent 0 genuinely cannot cross to the right triangle.
        assert!(
            !agent_can_reach(&graph, &occupant, 0, 8),
            "brute force says v8 is reachable — the fixture is wrong"
        );
        // The decomposition must agree: v8 is outside agent 0's region.
        assert!(
            !decomp.contains_in_subgraph_or_planks(sub_of_agent0, 8),
            "decomposition failed to detect confinement"
        );

        // And it must not over-claim: vertices it says are in range really
        // are reachable.
        for v in graph.vertices() {
            if decomp.contains_in_subgraph_or_planks(sub_of_agent0, v) {
                assert!(
                    agent_can_reach(&graph, &occupant, 0, v),
                    "claimed v{v} reachable for agent 0, brute force disagrees"
                );
            }
        }
    }

    /// With enough empty vertices the corridor becomes passable, the two
    /// triangles merge into one subgraph, and nothing is confined.
    #[test]
    fn corridor_merges_subgraphs_when_empties_allow() {
        let graph = dumbbell();
        // 2 atoms, 7 empties: m - 2 = 5 ≥ the 4-hop gap between triangles.
        let mut occupant: Vec<Option<u32>> = vec![None; 9];
        occupant[0] = Some(0);
        occupant[8] = Some(1);

        let decomp = decompose(&graph, &occupant);
        assert_eq!(
            decomp.subgraphs.len(),
            1,
            "triangles within m-2 hops must merge into one subgraph"
        );
        let sub = decomp
            .assignment
            .get(&0)
            .copied()
            .expect("agent 0 assigned");
        assert!(decomp.contains_in_subgraph_or_planks(sub, 8));
        assert!(agent_can_reach(&graph, &occupant, 0, 8));
    }

    /// Definition 3 condition 3 is a threshold on `m`: the same graph must
    /// split or merge purely as a function of the empty count.
    #[test]
    fn subgraph_merging_is_a_threshold_in_empty_count() {
        let graph = dumbbell();
        let bcc = graph.biconnected();
        // Gap between the two triangles' join vertices (2 and 6) is 4 hops,
        // so they merge exactly when m - 2 >= 4, i.e. m >= 6.
        for m in 2..=5 {
            let (subs, _) = find_subgraphs(&graph, &bcc, m);
            assert_eq!(subs.len(), 2, "m={m} should keep the triangles separate");
        }
        for m in 6..=9 {
            let (subs, _) = find_subgraphs(&graph, &bcc, m);
            assert_eq!(subs.len(), 1, "m={m} should merge the triangles");
        }
    }

    #[test]
    fn precedence_cycle_detected() {
        let cycle = find_precedence_cycle(3, &[(0, 1), (1, 2), (2, 0)]);
        let cycle = cycle.expect("0 → 1 → 2 → 0 is a cycle");
        assert_eq!(cycle.len(), 3);
    }

    #[test]
    fn precedence_dag_has_no_cycle() {
        assert!(find_precedence_cycle(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]).is_none());
    }

    #[test]
    fn precedence_self_loop_is_a_cycle() {
        assert_eq!(find_precedence_cycle(2, &[(1, 1)]), Some(vec![1]));
    }

    /// Soundness invariants every decomposition must satisfy, checked against
    /// exhaustive configuration-space search:
    ///
    /// 1. The subgraphs partition their vertices — no vertex in two subgraphs.
    /// 2. Every confinement claim over-approximates reachability: an assigned
    ///    agent must never be able to reach a vertex outside its claimed
    ///    region, or goal containment would report a solvable instance as
    ///    infeasible.
    fn assert_confinement_sound(graph: &LaneGraph, occupant: &[Option<u32>], context: &str) {
        let decomp = decompose(graph, occupant);

        let mut owner_count = vec![0usize; graph.len()];
        for verts in &decomp.subgraphs {
            for &v in verts {
                owner_count[v] += 1;
            }
        }
        for (v, &count) in owner_count.iter().enumerate() {
            assert!(
                count <= 1,
                "{context}: vertex {v} appears in {count} subgraphs — not a partition"
            );
        }

        let reach = reachable_vertices(graph, occupant);
        for (&agent, &sub) in &decomp.assignment {
            for &v in &reach[&agent] {
                assert!(
                    decomp.contains_in_subgraph_or_planks(sub, v),
                    "{context}: agent {agent} (assigned subgraph {sub}) can reach \
                     v{v} by brute force, but the claimed region excludes it"
                );
            }
        }
    }

    /// Regression test: two 4-cycles sharing a cut vertex, `m = 2`.
    ///
    /// Vertex-sharing nontrivial biconnected components are at distance 0 —
    /// within `m - 2` hops for every `m ≥ 2` — so they must merge into one
    /// subgraph. Before that merge existed, the decomposition kept the cycles
    /// separate and confined the agent at v4 to the right cycle, while brute
    /// force shows it can cross through the shared vertex — i.e. `check`
    /// returned a false `Infeasible(GoalOutsideSubgraph)` for a solvable
    /// instance.
    #[test]
    fn shared_cut_vertex_merges_and_confinement_is_sound() {
        // Cycle A: 0-1-2-3-0. Cycle B: 0-4-5-6-0. Shared cut vertex 0.
        let graph = LaneGraph::from_edges(
            7,
            &[
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
                (0, 4),
                (4, 5),
                (5, 6),
                (6, 0),
            ],
        );
        // Atoms at 0, 1, 4, 5, 6; empties at 2, 3.
        let occupant: Vec<Option<u32>> =
            vec![Some(0), Some(1), None, None, Some(2), Some(3), Some(4)];
        let decomp = decompose(&graph, &occupant);
        assert_eq!(
            decomp.subgraphs.len(),
            1,
            "cycles sharing a cut vertex must merge at m = 2"
        );
        assert_confinement_sound(&graph, &occupant, "shared cut vertex");
    }

    /// Three cycles sharing one cut vertex must merge transitively.
    #[test]
    fn three_cycles_sharing_a_vertex_merge_transitively() {
        let graph = LaneGraph::from_edges(
            7,
            &[
                (0, 1),
                (1, 2),
                (2, 0),
                (0, 3),
                (3, 4),
                (4, 0),
                (0, 5),
                (5, 6),
                (6, 0),
            ],
        );
        for m in 2..=6 {
            let bcc = graph.biconnected();
            let (subs, sub_of_vertex) = find_subgraphs(&graph, &bcc, m);
            assert_eq!(subs.len(), 1, "m={m}: all three cycles must merge");
            assert!(
                graph.vertices().all(|v| sub_of_vertex[v] == 0),
                "m={m}: every vertex belongs to the merged subgraph"
            );
        }
    }

    /// Property test: on random small graphs, the decomposition's confinement
    /// claims must over-approximate brute-force reachability. This is the
    /// generalization of the hand-built fixtures above — it is what caught
    /// the vertex-sharing merge bug.
    #[test]
    fn confinement_never_underestimates_reachability_on_random_graphs() {
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};

        for seed in 0..120u64 {
            let mut rng = SmallRng::seed_from_u64(seed);
            let n = rng.random_range(4..=8);
            let mut edges: Vec<(VertexId, VertexId)> = Vec::new();
            for a in 0..n {
                for b in (a + 1)..n {
                    if rng.random_bool(0.4) {
                        edges.push((a, b));
                    }
                }
            }
            let graph = LaneGraph::from_edges(n, &edges);

            // Scatter atoms over distinct vertices, leaving at least 2 empty.
            let mut verts: Vec<VertexId> = (0..n).collect();
            for i in (1..n).rev() {
                let j = rng.random_range(0..=i);
                verts.swap(i, j);
            }
            let n_atoms = rng.random_range(1..=(n - 2));
            let mut occupant: Vec<Option<u32>> = vec![None; n];
            for (q, &v) in verts.iter().take(n_atoms).enumerate() {
                occupant[v] = Some(q as u32);
            }

            assert_confinement_sound(&graph, &occupant, &format!("seed {seed}"));
        }
    }
}
