//! Mutable planning state: where every agent is, and the moves made so far.
//!
//! The paper's algorithms pass `Π` (the move sequence) and `A` (the agent
//! placement) by reference and mutate both, with several operations
//! speculatively trying something and rolling back. [`PlanState`] is that
//! pair, with explicit checkpoint/rollback rather than the paper's
//! `A' ← A ; Π' ← [ ]` copies.
//!
//! Moves record `from` as well as `to`. The paper writes `move(Π, A, agent r
//! to vertex x)`, but [`reverse`](PlanState::reverse_with_roles) needs to know
//! where each agent came from, and recovering that by replaying the prefix
//! would be quadratic.

use crate::feasibility::graph::{LaneGraph, VertexId};

/// One agent stepping between adjacent vertices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Move {
    pub agent: u32,
    pub from: VertexId,
    pub to: VertexId,
}

/// A point in the move log to roll back to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Checkpoint(usize);

/// Agent placement plus the move log that produced it.
pub struct PlanState<'g> {
    graph: &'g LaneGraph,
    /// Vertex → the agent on it.
    occupant: Vec<Option<u32>>,
    /// Agent → the vertex it is on.
    position: Vec<VertexId>,
    moves: Vec<Move>,
}

impl<'g> PlanState<'g> {
    /// Build from an initial placement. `initial[i]` is agent `i`'s vertex.
    pub fn new(graph: &'g LaneGraph, initial: &[VertexId]) -> Self {
        let mut occupant = vec![None; graph.len()];
        for (agent, &v) in initial.iter().enumerate() {
            occupant[v] = Some(agent as u32);
        }
        Self {
            graph,
            occupant,
            position: initial.to_vec(),
            moves: Vec::new(),
        }
    }

    pub fn graph(&self) -> &'g LaneGraph {
        self.graph
    }

    pub fn agent_count(&self) -> usize {
        self.position.len()
    }

    /// The agent on `v`, if any.
    pub fn agent_at(&self, v: VertexId) -> Option<u32> {
        self.occupant[v]
    }

    pub fn position_of(&self, agent: u32) -> VertexId {
        self.position[agent as usize]
    }

    pub fn is_occupied(&self, v: VertexId) -> bool {
        self.occupant[v].is_some()
    }

    /// Move `agent` onto the adjacent, unoccupied vertex `to`.
    ///
    /// # Panics
    ///
    /// Debug-asserts adjacency and vacancy. Every caller in this crate is
    /// responsible for clearing `to` first — a violation is a planner bug, not
    /// an input error, so it should surface loudly in tests rather than
    /// silently corrupt the placement.
    pub fn step(&mut self, agent: u32, to: VertexId) {
        let from = self.position[agent as usize];
        debug_assert!(
            self.graph.neighbors(from).contains(&to),
            "step {agent}: {from} -> {to} is not an edge"
        );
        debug_assert!(
            self.occupant[to].is_none(),
            "step {agent}: {to} is occupied by {:?}",
            self.occupant[to]
        );
        self.occupant[from] = None;
        self.occupant[to] = Some(agent);
        self.position[agent as usize] = to;
        self.moves.push(Move { agent, from, to });
    }

    /// Move `agent` two steps, `via` then `to`. The paper's "move agent r
    /// through vertex v to vertex x".
    pub fn step_through(&mut self, agent: u32, via: VertexId, to: VertexId) {
        self.step(agent, via);
        self.step(agent, to);
    }

    /// Current position in the move log.
    pub fn checkpoint(&self) -> Checkpoint {
        Checkpoint(self.moves.len())
    }

    /// Moves recorded since `at`.
    pub fn moves_since(&self, at: Checkpoint) -> &[Move] {
        &self.moves[at.0..]
    }

    /// Undo every move since `at`, restoring the placement exactly.
    pub fn rollback(&mut self, at: Checkpoint) {
        while self.moves.len() > at.0 {
            let m = self.moves.pop().expect("length checked");
            self.occupant[m.to] = None;
            self.occupant[m.from] = Some(m.agent);
            self.position[m.agent as usize] = m.from;
        }
    }

    /// Replay the moves in `[from, to)` backwards, with `r` and `s` swapped,
    /// and append the result to the log.
    ///
    /// This is the paper's `reverse(Π, A, Π'_{r/s})` (Algorithm 5 line 10).
    /// It is *not* a rollback: undoing the moves with the two roles exchanged
    /// returns every uninvolved agent to where it started while leaving `r`
    /// and `s` in each other's original positions — which, combined with the
    /// `exchange` at the swap vertex, is what makes `swap` a swap.
    ///
    /// The range is half-open and explicit because `swap` must reverse *only*
    /// `Π′` — the multipush and clear phases — while leaving the `exchange`
    /// that follows them intact. Reversing to the end of the log would undo
    /// the exchange too, and the agents would arrive back at non-adjacent
    /// vertices.
    pub fn reverse_range_with_roles(&mut self, from: Checkpoint, to: Checkpoint, r: u32, s: u32) {
        let to_undo: Vec<Move> = self.moves[from.0..to.0].to_vec();
        for m in to_undo.into_iter().rev() {
            let agent = if m.agent == r {
                s
            } else if m.agent == s {
                r
            } else {
                m.agent
            };
            self.step(agent, m.from);
        }
    }

    /// The full move log.
    pub fn moves(&self) -> &[Move] {
        &self.moves
    }

    /// Every unoccupied vertex.
    pub fn empty_vertices(&self) -> impl Iterator<Item = VertexId> + '_ {
        self.graph.vertices().filter(|&v| !self.is_occupied(v))
    }

    /// Number of unoccupied vertices — the paper's `m`.
    pub fn empty_count(&self) -> usize {
        self.empty_vertices().count()
    }

    /// Shortest path from `from` to `to` as a vertex list *excluding* `from`,
    /// treating any vertex in `blocked` as absent. Returns `None` if
    /// unreachable; an empty vec if `from == to`.
    pub fn shortest_path(
        &self,
        from: VertexId,
        to: VertexId,
        blocked: &[bool],
    ) -> Option<Vec<VertexId>> {
        self.shortest_path_scored(from, to, blocked, |_, _| 0.0)
    }

    /// Shortest path, with `score` breaking ties between equally short routes.
    ///
    /// `score(from, to)` rates a candidate step; **higher wins**. It is only
    /// consulted between predecessors that reach a vertex at the *same* BFS
    /// depth, so it can choose among equally short paths but can never make a
    /// path longer. That is what keeps a bad heuristic from being a
    /// correctness problem.
    ///
    /// With a constant score this is byte-for-byte the plain BFS it replaced:
    /// the first predecessor to claim a vertex keeps it, since a tie does not
    /// satisfy the strict `>`.
    pub fn shortest_path_scored(
        &self,
        from: VertexId,
        to: VertexId,
        blocked: &[bool],
        score: impl Fn(VertexId, VertexId) -> f64,
    ) -> Option<Vec<VertexId>> {
        if from == to {
            return Some(Vec::new());
        }
        let n = self.graph.len();
        let mut prev = vec![usize::MAX; n];
        let mut depth = vec![u32::MAX; n];
        let mut prev_score = vec![f64::NEG_INFINITY; n];
        let mut queue = std::collections::VecDeque::new();
        depth[from] = 0;
        queue.push_back(from);

        while let Some(u) = queue.pop_front() {
            for &w in self.graph.neighbors(u) {
                if blocked.get(w).copied().unwrap_or(false) {
                    continue;
                }
                let s = score(u, w);
                if depth[w] == u32::MAX {
                    depth[w] = depth[u] + 1;
                    prev[w] = u;
                    prev_score[w] = s;
                    queue.push_back(w);
                } else if depth[w] == depth[u] + 1 && s > prev_score[w] {
                    // Same distance, better step: re-parent. Strictly better
                    // only, so equal scores leave the first claimer in place.
                    prev[w] = u;
                    prev_score[w] = s;
                }
            }
        }

        if depth[to] == u32::MAX {
            return None;
        }
        let mut path = vec![to];
        let mut cur = to;
        while prev[cur] != from && prev[cur] != usize::MAX {
            cur = prev[cur];
            path.push(cur);
        }
        path.reverse();
        Some(path)
    }
}
