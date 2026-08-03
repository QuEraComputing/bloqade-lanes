//! The `push`, `swap`, and `rotate` operations, and their auxiliaries.
//!
//! Direct implementations of Algorithms 4, 5, 6 and 10–13 of de Wilde, ter
//! Mors & Witteveen (JAIR 51, 2014). Algorithm numbers are cited on each
//! function; where this deviates from the paper it says so and why.
//!
//! Blocked-vertex sets (`U` in the paper) are passed as `&[bool]` indexed by
//! vertex rather than as a set: they are consulted inside BFS inner loops, and
//! every call site already has a dense vertex space.

use crate::feasibility::graph::VertexId;

use crate::push_rotate::context::PlanCtx;
use crate::push_rotate::state::PlanState;

/// A blocked-vertex mask.
pub type Blocked = Vec<bool>;

/// An all-false mask sized for the graph.
pub fn no_blocked(n: usize) -> Blocked {
    vec![false; n]
}

/// Algorithm 10: `clear_vertex(Π, G, A, v, U)`.
///
/// Find a shortest path from some unoccupied vertex `u` to `v` avoiding `U`,
/// then shift every agent on that path one step towards `u`, vacating `v`.
///
/// Which empty vertex to drain into is §5 decision 1, delegated to
/// [`PlanHeuristics::rank_clear_target`](crate::push_rotate::heuristics::PlanHeuristics::rank_clear_target).
/// The default ranks by distance, which is what the paper's Appendix A notes
/// affects both runtime and solution length.
pub fn clear_vertex(ctx: &PlanCtx, state: &mut PlanState, v: VertexId, blocked: &Blocked) -> bool {
    if !state.is_occupied(v) {
        return true;
    }
    if blocked.get(v).copied().unwrap_or(false) {
        return false;
    }

    // BFS outward from `v`, one pass rather than one per empty vertex.
    //
    // The scan stops once it is past the depth of the first empty vertex
    // found, so the heuristic ranks only the *equally nearest* candidates.
    // That is deliberate: letting it reach past them would let a heuristic
    // lengthen the plan, and the whole point of ranking rather than choosing
    // is that a heuristic can reorder equally-good options but not make a
    // worse one. It also preserves the early exit — without it every call
    // would sweep the whole component.
    let n = state.graph().len();
    let mut prev = vec![usize::MAX; n];
    let mut seen = vec![false; n];
    let mut depth = vec![0u32; n];
    let mut queue = std::collections::VecDeque::new();
    seen[v] = true;
    queue.push_back(v);

    let mut best: Option<(i64, VertexId)> = None;
    let mut best_depth: Option<u32> = None;
    while let Some(u) = queue.pop_front() {
        if best_depth.is_some_and(|d| depth[u] > d) {
            break;
        }
        if !state.is_occupied(u) {
            let key = ctx.heuristics.rank_clear_target(ctx, state, v, u, depth[u]);
            if best.is_none_or(|(bk, _)| key < bk) {
                best = Some((key, u));
            }
            best_depth.get_or_insert(depth[u]);
        }
        for &w in state.graph().neighbors(u) {
            if seen[w] || blocked.get(w).copied().unwrap_or(false) {
                continue;
            }
            seen[w] = true;
            prev[w] = u;
            depth[w] = depth[u] + 1;
            queue.push_back(w);
        }
    }

    let Some((_, empty)) = best else {
        return false;
    };

    // Walk back v <- ... <- empty, shifting each agent one step outward.
    let mut chain = vec![empty];
    let mut cur = empty;
    while cur != v {
        cur = prev[cur];
        debug_assert_ne!(cur, usize::MAX, "clear_vertex parent chain broke");
        chain.push(cur);
    }
    // chain is [empty, ..., v]; move the agent adjacent to `empty` into it,
    // and repeat towards v.
    for pair in chain.windows(2) {
        let (dst, src) = (pair[0], pair[1]);
        let agent = state.agent_at(src).expect("chain interior is occupied");
        state.step(agent, dst);
    }
    debug_assert!(!state.is_occupied(v));
    true
}

/// Algorithm 4: `push(Π, G, A, r, v, U)`.
///
/// Move agent `r` onto adjacent vertex `v`, clearing `v` first if needed.
/// `r`'s own vertex is added to the blocked set while clearing, so the clear
/// cannot route through the agent that is about to move.
pub fn push(ctx: &PlanCtx, state: &mut PlanState, r: u32, v: VertexId, blocked: &Blocked) -> bool {
    if state.is_occupied(v) {
        let mut u = blocked.clone();
        u[state.position_of(r)] = true;
        if !clear_vertex(ctx, state, v, &u) {
            return false;
        }
    }
    state.step(r, v);
    true
}

/// Algorithm 11: `multipush(Π, G, A, r', s', v)`.
///
/// Bring the adjacent pair `r'`/`s'` to `v`, leading with whichever is closer,
/// the other following into the vacated vertex.
fn multipush(
    ctx: &PlanCtx,
    state: &mut PlanState,
    r_prime: u32,
    s_prime: u32,
    v: VertexId,
) -> Option<(u32, u32)> {
    let none = no_blocked(state.graph().len());
    let dist = state.graph().distances_from(v, |_| false);
    let (r, s) = if dist[state.position_of(r_prime)] <= dist[state.position_of(s_prime)] {
        (r_prime, s_prime)
    } else {
        (s_prime, r_prime)
    };

    let path = state.shortest_path(state.position_of(r), v, &none)?;
    for x in path {
        let vr = state.position_of(r);
        let vs = state.position_of(s);
        if state.is_occupied(x) {
            let mut u = no_blocked(state.graph().len());
            u[vr] = true;
            u[vs] = true;
            if !clear_vertex(ctx, state, x, &u) {
                return None;
            }
        }
        state.step(r, x);
        state.step(s, vr);
    }
    Some((r, s))
}

/// Algorithm 13: `exchange(Π, G, A, r', s', v)`.
///
/// With `r` on `v`, `s` adjacent, and two empty neighbours of `v`, swap the
/// two agents in place.
fn exchange(state: &mut PlanState, r: u32, s: u32, v: VertexId) -> bool {
    let vs = state.position_of(s);
    let empties: Vec<VertexId> = state
        .graph()
        .neighbors(v)
        .iter()
        .copied()
        .filter(|&n| !state.is_occupied(n))
        .take(2)
        .collect();
    if empties.len() < 2 {
        return false;
    }
    let (v1, v2) = (empties[0], empties[1]);
    state.step(r, v1);
    state.step_through(s, v, v2);
    state.step_through(r, v, vs);
    state.step(s, v);
    true
}

/// Algorithm 12: `clear(Π, G, A, r', s', v)`.
///
/// Try to give `v` two empty neighbours, in the paper's four escalating
/// stages. `r` is on `v`, `s` on the adjacent `v'`.
fn clear(ctx: &PlanCtx, state: &mut PlanState, r: u32, s: u32, v: VertexId) -> bool {
    let n_v = state.graph().len();
    let v_prime = state.position_of(s);

    let empties = |st: &PlanState| -> Vec<VertexId> {
        st.graph()
            .neighbors(v)
            .iter()
            .copied()
            .filter(|&x| !st.is_occupied(x))
            .collect()
    };

    // Stage 1: push neighbours of `v` away from `v`.
    if empties(state).len() >= 2 {
        return true;
    }
    let neighbours: Vec<VertexId> = state.graph().neighbors(v).to_vec();
    for &n in &neighbours {
        if !state.is_occupied(n) || n == v_prime {
            continue;
        }
        let mut u = no_blocked(n_v);
        u[v] = true;
        u[v_prime] = true;
        for e in empties(state) {
            u[e] = true;
        }
        if clear_vertex(ctx, state, n, &u) && empties(state).len() >= 2 {
            return true;
        }
    }

    let current = empties(state);
    if current.is_empty() {
        return false;
    }
    let eps = current[0];

    // Stage 2: vacate a neighbour, then re-clear the empty one behind it.
    for &n in &neighbours {
        if n == v_prime || n == eps {
            continue;
        }
        let cp = state.checkpoint();
        let mut u = no_blocked(n_v);
        u[v] = true;
        u[v_prime] = true;
        if clear_vertex(ctx, state, n, &u) {
            let mut u2 = no_blocked(n_v);
            u2[v] = true;
            u2[v_prime] = true;
            u2[n] = true;
            if clear_vertex(ctx, state, eps, &u2) {
                return true;
            }
        }
        state.rollback(cp);
        break;
    }

    // Stage 3: move r and s back a step so a neighbour can vacate through v'.
    for &n in &neighbours {
        if n == v_prime || n == eps {
            continue;
        }
        let cp = state.checkpoint();
        state.step(r, eps);
        state.step(s, v);
        let mut u = no_blocked(n_v);
        u[v] = true;
        u[eps] = true;
        if clear_vertex(ctx, state, n, &u) {
            let mut u2 = no_blocked(n_v);
            u2[v] = true;
            u2[eps] = true;
            u2[n] = true;
            if clear_vertex(ctx, state, v_prime, &u2) {
                return true;
            }
        }
        state.rollback(cp);
        break;
    }

    // Stage 4: make room behind eps by routing a neighbour through v.
    let cp = state.checkpoint();
    let mut u = no_blocked(n_v);
    u[v] = true;
    if !clear_vertex(ctx, state, v_prime, &u) {
        state.rollback(cp);
        return false;
    }
    state.step(r, v_prime);
    let mut u2 = no_blocked(n_v);
    u2[v] = true;
    u2[v_prime] = true;
    u2[state.position_of(s)] = true;
    if !clear_vertex(ctx, state, eps, &u2) {
        state.rollback(cp);
        return false;
    }
    let Some(&n) = neighbours.iter().find(|&&x| x != v_prime && x != eps) else {
        state.rollback(cp);
        return false;
    };
    let Some(t) = state.agent_at(n) else {
        state.rollback(cp);
        return false;
    };
    if state.is_occupied(v) || state.is_occupied(eps) {
        state.rollback(cp);
        return false;
    }
    state.step_through(t, v, eps);
    state.step(r, v);
    state.step(s, v_prime);
    let mut u3 = no_blocked(n_v);
    u3[v] = true;
    u3[v_prime] = true;
    u3[n] = true;
    if clear_vertex(ctx, state, eps, &u3) {
        true
    } else {
        state.rollback(cp);
        false
    }
}

/// Algorithm 5: `swap(Π, G, A, r, s)`.
///
/// Exchange two adjacent agents by walking both to a degree-≥3 vertex in their
/// shared subgraph, exchanging there, and reversing the approach with the two
/// roles switched.
///
/// Proposition 3: this succeeds iff `r` and `s` are assigned to the same
/// subgraph.
pub fn swap(ctx: &PlanCtx, state: &mut PlanState, r: u32, s: u32) -> bool {
    let Some(&sub) = ctx.decomp.assignment.get(&r) else {
        return false;
    };
    if ctx.decomp.assignment.get(&s) != Some(&sub) {
        return false;
    }

    // Candidate swap vertices: degree ≥ 3 within r's subgraph, nearest first
    // (the paper: "we evaluate the vertices closest to r and s first").
    let from = state.position_of(r);
    let dist = state.graph().distances_from(from, |_| false);
    // Which swap vertex to use is §5 decision 2.
    let mut candidates: Vec<VertexId> = ctx.decomp.subgraphs[sub]
        .iter()
        .copied()
        .filter(|&x| state.graph().degree(x) >= 3 && dist[x] != u32::MAX)
        .collect();
    candidates.sort_by_key(|&x| {
        (
            ctx.heuristics
                .rank_swap_vertex(ctx, state, r, s, x, dist[x]),
            x,
        )
    });

    for v in candidates {
        let cp = state.checkpoint();
        let Some((lead, follow)) = multipush(ctx, state, r, s, v) else {
            state.rollback(cp);
            continue;
        };
        if !clear(ctx, state, lead, follow, v) {
            state.rollback(cp);
            continue;
        }
        // End of Π′ (multipush + clear). The exchange that follows is
        // appended to Π, not Π′, and must survive the reversal.
        let after_approach = state.checkpoint();
        if !exchange(state, lead, follow, v) {
            state.rollback(cp);
            continue;
        }
        state.reverse_range_with_roles(cp, after_approach, lead, follow);
        return true;
    }
    false
}

/// Algorithm 6: `rotate(Π, G, A, c)`.
///
/// Advance every agent on cycle `c` one step. `c` is in path order, and the
/// agent on `c[i]` moves to `c[i+1]` (wrapping).
pub fn rotate(ctx: &PlanCtx, state: &mut PlanState, cycle: &[VertexId]) -> bool {
    if cycle.len() < 2 {
        return false;
    }
    // `q` in Algorithm 8 accumulates vertices across successive agents, so a
    // repeated vertex does not on its own guarantee the suffix is a closed
    // walk. Verify before rotating: a non-cycle would produce non-adjacent
    // steps and corrupt the placement.
    for i in 0..cycle.len() {
        let a = cycle[i];
        let b = cycle[(i + 1) % cycle.len()];
        if !state.graph().neighbors(a).contains(&b) {
            return false;
        }
    }

    // Trivial case: some vertex on the cycle is already empty, so shift into
    // it and let each agent behind follow.
    if let Some(hole) = cycle.iter().position(|&v| !state.is_occupied(v)) {
        for k in 1..cycle.len() {
            let dst = cycle[(hole + cycle.len() - k + 1) % cycle.len()];
            let src = cycle[(hole + cycle.len() - k) % cycle.len()];
            if let Some(agent) = state.agent_at(src) {
                state.step(agent, dst);
            }
        }
        return true;
    }

    // Fully occupied: evict one agent, swap it into place, rotate, restore.
    let n_v = state.graph().len();
    for (i, &v) in cycle.iter().enumerate() {
        let Some(r) = state.agent_at(v) else { continue };
        let cp = state.checkpoint();
        let mut u = no_blocked(n_v);
        for &c in cycle {
            if c != v {
                u[c] = true;
            }
        }
        if !clear_vertex(ctx, state, v, &u) {
            state.rollback(cp);
            continue;
        }
        // Algorithm 6 line 9: Π′ is the clear_vertex phase alone. Everything
        // after — the swap and the rotation itself — stays.
        let after_clear = state.checkpoint();
        let v_prev = cycle[(i + cycle.len() - 1) % cycle.len()];
        let Some(r_prev) = state.agent_at(v_prev) else {
            state.rollback(cp);
            continue;
        };
        state.step(r_prev, v);
        if !swap(ctx, state, r, r_prev) {
            state.rollback(cp);
            continue;
        }
        // Algorithm 6 line 14: advance *every* cycle agent one step, starting
        // with the one moving into the hole at `v_prev` and walking backwards
        // around the cycle. The last step moves `r` out of `v`, which is what
        // leaves `v` free for the reversal below to put `r_prev` back into —
        // stopping an iteration early strands `r` on `v` and the reversal
        // then tries to move onto an occupied vertex.
        for k in 1..cycle.len() {
            let dst = cycle[(i + cycle.len() - k) % cycle.len()];
            let src = cycle[(i + cycle.len() - k - 1) % cycle.len()];
            if let Some(agent) = state.agent_at(src) {
                state.step(agent, dst);
            }
        }
        // Algorithm 6 line 15: `reverse(Pi, A, Pi'_{r/r'})`. Pi' is the
        // clear_vertex phase — the range [cp, after_clear) — so the agents
        // shoved aside to vacate `v` are put back. Reversing the *later*
        // range would undo the swap and the rotation instead, which is the
        // whole point of the operation.
        state.reverse_range_with_roles(cp, after_clear, r, r_prev);
        return true;
    }
    false
}
