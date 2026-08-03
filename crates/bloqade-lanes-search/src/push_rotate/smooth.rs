//! Algorithm 9: `smooth` — remove redundant moves from a finished plan.
//!
//! From §5.1 of de Wilde, ter Mors & Witteveen (JAIR 51, 2014):
//!
//! > A sequence of agent moves is redundant if the agent visits a vertex for a
//! > second time, and no agents have visited the vertex in the meantime.
//!
//! Push and Rotate generates these constantly: `clear` shoves an agent aside,
//! the swap reversal shoves it back, and the agent later walks the same way
//! again. Every move in such a round trip can be deleted — the agent ends where
//! it started and, by the definition above, nothing else depended on it being
//! away.
//!
//! ## Why the linked lists
//!
//! Deleting a move can make *other* moves redundant, so the naive approach
//! rescans the whole plan until it reaches a fixed point. Algorithm 9 instead
//! threads four doubly-linked chains through the move list — previous/next
//! move **by agent**, previous/next move **to a vertex** — so that after a
//! deletion the newly-adjacent pair at that vertex can be checked in O(1).
//!
//! ## Deviation from the printed pseudocode
//!
//! Line 8 reads `while π′ ≠ NV(π)`, which stops *before* deleting the agent's
//! return to the vertex. That leaves the agent moving to a vertex it never
//! left. The prose one paragraph earlier is unambiguous — "from the sequence
//! [π, NA(π), NA(NA(π)), ⋯ , NV(π)] all moves except for the first can be
//! removed" — and the worked example in Figure 16 says the *final two* moves
//! go, including the return. So the deletion here is inclusive of `NV(π)`.
//!
//! ## Extension: virtual start arrivals
//!
//! Algorithm 9 detects a round trip by finding two *moves* arriving at the
//! same vertex. An agent's initial position is not a move, so a round trip
//! that leaves the starting vertex and comes back is invisible to the printed
//! algorithm — and Push and Rotate produces exactly those, since `clear` and
//! the swap reversal routinely shove a not-yet-planned agent aside and back.
//!
//! We therefore seed each agent's start vertex as a virtual arrival, ordered
//! before every real move. It is never itself removed. The soundness argument
//! is unchanged: the agent ends where it began and, by the same
//! no-one-else-visited condition, nothing depended on it being away.
//!
//! ## Why this is safe
//!
//! Removing a round trip can only make later moves *more* legal, never less:
//! a move's precondition is that its destination is empty, and deleting an
//! agent's excursion only frees vertices earlier. The vertex the agent
//! returned to is untouched by anyone else, which is exactly what the
//! redundancy condition guarantees. Plans are replayed against the graph in
//! the tests regardless.

use std::collections::VecDeque;

use crate::feasibility::graph::VertexId;

use crate::push_rotate::state::Move;

const NONE: usize = usize::MAX;

/// Remove redundant moves, returning the shortened plan.
pub fn smooth(moves: &[Move]) -> Vec<Move> {
    let n = moves.len();
    if n < 2 {
        return moves.to_vec();
    }

    // Nodes 0..n are real moves; n.. are the virtual start arrivals, one per
    // agent that moves at all. `agent_of` / `vertex_of` resolve either kind.
    let max_agent = moves.iter().map(|m| m.agent).max().unwrap_or(0) as usize;
    let mut start_vertex: Vec<Option<VertexId>> = vec![None; max_agent + 1];
    for m in moves {
        let a = m.agent as usize;
        if start_vertex[a].is_none() {
            start_vertex[a] = Some(m.from);
        }
    }
    let virtuals: Vec<(u32, VertexId)> = start_vertex
        .iter()
        .enumerate()
        .filter_map(|(a, v)| v.map(|v| (a as u32, v)))
        .collect();
    let total = n + virtuals.len();
    let agent_of = |i: usize| -> u32 {
        if i < n {
            moves[i].agent
        } else {
            virtuals[i - n].0
        }
    };
    let mut prev_agent = vec![NONE; total];
    let mut next_agent = vec![NONE; total];
    let mut prev_vertex = vec![NONE; total];
    let mut next_vertex = vec![NONE; total];
    let mut removed = vec![false; total];

    let mut last_by_agent = vec![NONE; max_agent + 1];
    let max_vertex = moves.iter().map(|m| m.to.max(m.from)).max().unwrap_or(0);
    let mut last_by_vertex = vec![NONE; max_vertex + 1];

    // Virtual arrivals first, so they head their vertex chains.
    for (k, &(a, v)) in virtuals.iter().enumerate() {
        let i = n + k;
        last_by_agent[a as usize] = i;
        last_by_vertex[v] = i;
    }

    for (i, m) in moves.iter().enumerate() {
        let a = m.agent as usize;
        if last_by_agent[a] != NONE {
            prev_agent[i] = last_by_agent[a];
            next_agent[last_by_agent[a]] = i;
        }
        last_by_agent[a] = i;

        let v: VertexId = m.to;
        if last_by_vertex[v] != NONE {
            prev_vertex[i] = last_by_vertex[v];
            next_vertex[last_by_vertex[v]] = i;
        }
        last_by_vertex[v] = i;
    }

    // Lines 2-4: seed with every arrival whose *previous* arrival at the same
    // vertex was by the same agent — the start of a round trip.
    let mut queue: VecDeque<usize> = VecDeque::new();
    let mut queued = vec![false; total];
    for (i, &p) in prev_vertex.iter().enumerate().take(n) {
        if p != NONE && agent_of(p) == agent_of(i) && !queued[p] {
            queued[p] = true;
            queue.push_back(p);
        }
    }

    #[allow(clippy::too_many_arguments)]
    let unlink = |i: usize,
                  removed: &mut Vec<bool>,
                  prev_agent: &mut Vec<usize>,
                  next_agent: &mut Vec<usize>,
                  prev_vertex: &mut Vec<usize>,
                  next_vertex: &mut Vec<usize>,
                  queue: &mut VecDeque<usize>,
                  queued: &mut Vec<bool>| {
        removed[i] = true;
        let (pa, na) = (prev_agent[i], next_agent[i]);
        if pa != NONE {
            next_agent[pa] = na;
        }
        if na != NONE {
            prev_agent[na] = pa;
        }
        let (pv, nv) = (prev_vertex[i], next_vertex[i]);
        if pv != NONE {
            next_vertex[pv] = nv;
        }
        if nv != NONE {
            prev_vertex[nv] = pv;
        }
        // Line 12: the pair now adjacent at this vertex may itself be a round
        // trip.
        if pv != NONE && nv != NONE && agent_of(pv) == agent_of(nv) && !queued[pv] {
            queued[pv] = true;
            queue.push_back(pv);
        }
    };

    // Lines 5-14.
    while let Some(start) = queue.pop_front() {
        queued[start] = false;
        if removed[start] {
            continue;
        }
        // Capture the return arrival before any unlinking rewrites pointers.
        let target = next_vertex[start];
        if target == NONE || removed[target] || agent_of(target) != agent_of(start) {
            continue;
        }

        let mut cur = next_agent[start];
        while cur != NONE {
            let next = next_agent[cur];
            unlink(
                cur,
                &mut removed,
                &mut prev_agent,
                &mut next_agent,
                &mut prev_vertex,
                &mut next_vertex,
                &mut queue,
                &mut queued,
            );
            if cur == target {
                break;
            }
            cur = next;
        }
    }

    moves
        .iter()
        .enumerate()
        .filter(|(i, _)| !removed[*i])
        .map(|(_, m)| *m)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn mv(agent: u32, from: VertexId, to: VertexId) -> Move {
        Move { agent, from, to }
    }

    /// Each agent's initial vertex, taken from its first move in the original
    /// plan. Seeding both replays with this is what makes an agent that ends
    /// where it began comparable to one that never moved.
    fn starts(plan: &[Move]) -> BTreeMap<u32, VertexId> {
        let mut out: BTreeMap<u32, VertexId> = BTreeMap::new();
        for m in plan {
            out.entry(m.agent).or_insert(m.from);
        }
        out
    }

    /// Replay a move list from `starts`, asserting each agent departs from
    /// where it actually is, and return the final placement.
    fn final_positions(
        starts: &BTreeMap<u32, VertexId>,
        moves: &[Move],
    ) -> BTreeMap<u32, VertexId> {
        let mut pos = starts.clone();
        for m in moves {
            let at = pos
                .get_mut(&m.agent)
                .expect("agent seen in the original plan");
            assert_eq!(
                *at, m.from,
                "agent {} departs from the wrong vertex",
                m.agent
            );
            *at = m.to;
        }
        pos
    }

    /// Smoothing must not change where anyone ends up, and must not grow the
    /// plan. Returns the smoothed plan.
    fn check(plan: &[Move]) -> Vec<Move> {
        let s = starts(plan);
        let before = final_positions(&s, plan);
        let out = smooth(plan);
        let after = final_positions(&s, &out);
        assert_eq!(before, after, "smoothing changed the final placement");
        assert!(out.len() <= plan.len(), "smoothing grew the plan");
        out
    }

    /// The paper's Figure 16: `a3` is pushed into its goal by `clear`, put back
    /// by the swap reversal, then walks there again when it is planned.
    #[test]
    fn collapses_the_figure_16_round_trip() {
        let plan = vec![
            mv(3, 0, 1), // a3 makes room, arriving at its goal
            mv(1, 5, 6), // unrelated traffic
            mv(3, 1, 0), // a3 put back
            mv(3, 0, 1), // a3 walks to its goal again
        ];
        let out = check(&plan);
        assert_eq!(
            out.len(),
            2,
            "a3 should move once, not three times: {out:?}"
        );
    }

    #[test]
    fn keeps_a_round_trip_another_agent_interrupted() {
        // a1 visits vertex 1, a2 arrives at vertex 1 in between and moves on
        // elsewhere. a1's return to 1 is therefore not redundant.
        let plan = vec![
            mv(1, 0, 1),
            mv(1, 1, 2),
            mv(2, 9, 1),
            mv(2, 1, 8),
            mv(1, 2, 1),
        ];
        let out = check(&plan);
        assert_eq!(out, plan, "nothing here is redundant");
    }

    /// The virtual-start extension: a round trip back to an agent's *initial*
    /// vertex is invisible to the printed algorithm, which only sees moves.
    #[test]
    fn collapses_a_round_trip_to_the_starting_vertex() {
        let plan = vec![mv(1, 0, 1), mv(1, 1, 2), mv(1, 2, 1), mv(1, 1, 0)];
        assert!(
            check(&plan).is_empty(),
            "a closed walk from the start with no other traffic should vanish"
        );
    }

    #[test]
    fn cascades_to_newly_redundant_moves() {
        // Removing a1's inner round trip at vertex 2 exposes the outer one at
        // vertex 1, which exposes the return to the start.
        let plan = vec![
            mv(1, 0, 1),
            mv(1, 1, 2),
            mv(1, 2, 3),
            mv(1, 3, 2),
            mv(1, 2, 1),
            mv(1, 1, 0),
        ];
        assert!(check(&plan).is_empty());
    }

    #[test]
    fn leaves_a_straight_path_alone() {
        let plan = vec![mv(1, 0, 1), mv(1, 1, 2), mv(1, 2, 3)];
        assert_eq!(check(&plan), plan);
    }

    #[test]
    fn handles_trivial_inputs() {
        assert!(smooth(&[]).is_empty());
        let one = vec![mv(0, 0, 1)];
        assert_eq!(smooth(&one), one);
    }
}
