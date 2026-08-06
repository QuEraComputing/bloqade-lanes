//! Push and Rotate: a complete multi-agent pathfinding planner.
//!
//! de Wilde, ter Mors & Witteveen, *Push and Rotate: a Complete Multi-agent
//! Pathfinding Algorithm*, JAIR 51 (2014) 443–492. Theorem 1: complete for
//! instances with at least two empty vertices per connected component where
//! an atom moves.
//!
//! This implementation meets the *solving* half of that claim (validated by
//! the brute-force property tests) and a **sound but partial** proving half:
//! [`PlanError::Unsolvable`] and [`PlanError::UnknownQubit`] are genuine
//! proofs, but some unsolvable in-regime instances come back as
//! [`PlanError::Stuck`] instead of a proof. The gap is deliberate — the
//! paper's `f = f'` test is only sound with its full-strength agent
//! assignment, and the shared decomposition in `crate::feasibility`
//! intentionally under-assigns (see the containment-form check in
//! [`plan_with`]). Closing it would need a planner-specific,
//! paper-faithful `assign_agents`.
//!
//! The Kornhauser decomposition it runs on (Algorithms 1–3) lives in the open
//! `crate::feasibility` module — it is the paper's own content
//! and is shared with the open feasibility checker. This module implements the
//! planner proper: Algorithms 4–8 and 10–13.
//!
//! ## Output shape
//!
//! The planner emits **single-atom moves**, one per step. That is not a usable
//! AOD schedule on its own — turning it into one is the condenser's job
//! (phase 2b), which merges consecutive moves into complete X×Y rectangle
//! batches. Sound by construction: a legal AOD move is exactly a set of
//! vertex-disjoint single moves into empty destinations.

pub mod context;
pub mod heuristics;
pub mod instances;
pub mod ops;
pub mod schedule;
pub mod smooth;
pub mod solver;
pub mod state;

pub use solver::{DEFAULT_MOVE_BUDGET, solve_push_rotate, solve_push_rotate_with};

use std::collections::HashMap;

use crate::feasibility::decomposition::{Decomposition, assign_agents};
use crate::feasibility::graph::{LaneGraph, VertexId};
use crate::primitives::lane_index::LaneIndex;

use crate::push_rotate::context::PlanCtx;
use crate::push_rotate::heuristics::{DefaultHeuristics, PlanHeuristics};
use crate::push_rotate::ops::{no_blocked, push, rotate, swap};
use crate::push_rotate::state::{Move, PlanState};

/// Why planning stopped without a solution.
///
/// The variants split into **proofs** and **non-proofs**, and the
/// distinction is load-bearing: the fallback path in the target solver
/// promotes the planner's verdict over the search's, so only
/// [`PlanError::Unsolvable`] and [`PlanError::UnknownQubit`] — the two
/// variants that genuinely prove no solution exists — may be surfaced as
/// [`SolveStatus::Unsolvable`](crate::search::result::SolveStatus::Unsolvable).
/// Everything else means "the planner did not find a plan", which proves
/// nothing.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PlanError {
    /// **Proof.** No solution exists: either the agent→subgraph assignment
    /// differs between the initial and goal placements (Algorithm 7 line 5 —
    /// the agent would have to leave the region Proposition 1 confines it
    /// to), or no path to the target exists at all.
    #[error("instance is unsolvable: {reason} (qubit {qubit})")]
    Unsolvable { qubit: u32, reason: &'static str },
    /// **Proof.** A target was given for a qubit that is not in the initial
    /// placement. No sequence of moves creates an atom that does not exist,
    /// so a goal that requires placing one is unsatisfiable. (This is a
    /// caller bug — but a loud verdict beats fabricating a placement.)
    #[error("target given for qubit {qubit}, which is not in the initial placement")]
    UnknownQubit { qubit: u32 },
    /// **Not a proof.** Fewer than two empty vertices in a connected
    /// component where an atom must move. Push and Rotate's completeness
    /// result does not cover this regime — the instance may still be
    /// solvable (single-hole 15-puzzle instances are).
    #[error(
        "push and rotate requires at least 2 empty vertices in every component \
         where an atom moves, found {found}"
    )]
    TooFewEmpty { found: usize },
    /// **Not a proof.** An operation the completeness argument says must
    /// succeed in-regime (rotate, swap, or the final placement check)
    /// failed. Either the instance slipped outside the regime in a way the
    /// gates did not catch, or this is a planner bug; in neither case is it
    /// evidence that no solution exists.
    #[error("planner stuck: {reason} (qubit {qubit}) — not a proof of unsolvability")]
    Stuck { qubit: u32, reason: &'static str },
    /// **Not a proof.** The planner exceeded its step budget.
    #[error("exceeded step budget of {budget} moves")]
    BudgetExceeded { budget: usize },
}

/// A completed plan: single-atom moves in execution order.
#[derive(Debug, Clone)]
pub struct Plan {
    /// The moves, after [`smooth`](smooth::smooth).
    pub moves: Vec<Move>,
    /// Move count before smoothing. Reported so the post-processing's
    /// contribution is visible rather than inferred — on Gemini instances it
    /// turns out to be near zero, which is worth being able to see.
    pub raw_move_count: usize,
}

/// Plan a rearrangement with Push and Rotate.
///
/// `initial` and `target` are `(qubit, vertex)` placements over `graph`.
pub fn plan(
    index: &LaneIndex,
    graph: &LaneGraph,
    initial: &[(u32, VertexId)],
    target: &[(u32, VertexId)],
    budget: usize,
) -> Result<Plan, PlanError> {
    plan_with(index, graph, initial, target, budget, &DefaultHeuristics)
}

/// Plan with an explicit heuristic strategy.
///
/// [`plan`] is this with [`DefaultHeuristics`], which is the unoptimised
/// baseline all current benchmark numbers were measured against.
pub fn plan_with(
    index: &LaneIndex,
    graph: &LaneGraph,
    initial: &[(u32, VertexId)],
    target: &[(u32, VertexId)],
    budget: usize,
    heuristics: &dyn PlanHeuristics,
) -> Result<Plan, PlanError> {
    let n_agents = initial.len();

    // Dense agent indexing: the planner works with 0..n_agents internally.
    let mut agent_of_qubit: HashMap<u32, u32> = HashMap::new();
    let mut start = vec![0usize; n_agents];
    let mut qubit_of_agent = vec![0u32; n_agents];
    for (i, &(qubit, v)) in initial.iter().enumerate() {
        agent_of_qubit.insert(qubit, i as u32);
        qubit_of_agent[i] = qubit;
        start[i] = v;
    }
    let mut goal = start.clone();
    for &(qubit, v) in target {
        let Some(&a) = agent_of_qubit.get(&qubit) else {
            // A target for an atom that does not exist can never be
            // satisfied; silently dropping it would let the solver report
            // `Solved` for a goal it did not (and cannot) meet.
            return Err(PlanError::UnknownQubit { qubit });
        };
        goal[a as usize] = v;
    }

    // ── Algorithm 7 ────────────────────────────────────────────────
    let occupancy = |placement: &[VertexId]| -> Vec<Option<u32>> {
        let mut occ = vec![None; graph.len()];
        for (a, &v) in placement.iter().enumerate() {
            occ[v] = Some(a as u32);
        }
        occ
    };

    // The decomposition (subgraphs, planks, per-component empty counts, and
    // the start-side assignment `f`) comes from the shared feasibility
    // builder, so the planner and the infeasibility oracle can never
    // disagree about the structure of an instance.
    let decomp = Decomposition::build(graph, &occupancy(&start));

    // Regime gate, per connected component: Theorem 1 covers each
    // pebble-motion instance separately, and empties in another component
    // cannot help maneuvering. A component where nothing needs to move is
    // fine with any empty count.
    for a in 0..n_agents {
        if start[a] != goal[a] {
            let found = decomp.empties_in_component[decomp.component_of_vertex[start[a]]];
            if found < 2 {
                return Err(PlanError::TooFewEmpty { found });
            }
        }
    }

    // Goal-side assignment `f'` reuses the same subgraphs and per-component
    // `m` — those depend on the graph and the empty counts, not on where
    // the atoms sit, and any solvable instance keeps each agent (and thus
    // each component's empty count) in its own component.
    let m_of_vertex: Vec<usize> = graph
        .vertices()
        .map(|v| decomp.empties_in_component[decomp.component_of_vertex[v]])
        .collect();
    let f_goal = assign_agents(
        graph,
        &decomp.subgraphs,
        &decomp.subgraph_of_vertex,
        &occupancy(&goal),
        &m_of_vertex,
    );

    // Line 5, in region-containment form rather than the paper's `f = f'`
    // equality. Our `assign_agents` deliberately under-assigns relative to
    // the paper (see the soundness note in the decomposition module), which
    // breaks the symmetry the equality test relies on: the two placements
    // can gate differently for an agent that is confined the same way in
    // both, turning `Some(S) != None` into a false proof. What Proposition 1
    // actually licenses — and what the feasibility module's brute-force
    // tests validate — is the region claim, applied in both time directions
    // since pebble moves are reversible:
    //
    // * an agent confined at the start must have its goal inside that
    //   region, and
    // * an agent confined at the goal must have its start inside that
    //   region (running the plan backwards).
    //
    // Subgraphs and planks depend only on the graph and the per-component
    // empty counts, so the one decomposition serves both directions.
    for (&a, &sub) in &decomp.assignment {
        if !decomp.contains_in_subgraph_or_planks(sub, goal[a as usize]) {
            return Err(PlanError::Unsolvable {
                qubit: qubit_of_agent[a as usize],
                reason: "the atom is confined to a region that excludes its goal",
            });
        }
    }
    for (&a, &sub) in &f_goal {
        if !decomp.contains_in_subgraph_or_planks(sub, start[a as usize]) {
            return Err(PlanError::Unsolvable {
                qubit: qubit_of_agent[a as usize],
                reason: "the atom's goal is confined to a region that excludes its start",
            });
        }
    }

    let ctx = PlanCtx::new(graph, index, &decomp, &goal, &qubit_of_agent, heuristics);
    solve(&ctx, &start, budget)
}

/// Algorithm 8: `solve`.
///
/// Move agents to their destinations one at a time along a shortest path,
/// pushing where possible and swapping where not. `q` accumulates the path
/// walked so far; a step back onto `q` means a cycle of displaced agents,
/// which `rotate` resolves. The second loop returns agents that a swap knocked
/// off their goal.
fn solve(ctx: &PlanCtx, start: &[VertexId], budget: usize) -> Result<Plan, PlanError> {
    let graph = ctx.graph;
    let goal = ctx.goal;
    let n_agents = start.len();
    let mut state = PlanState::new(graph, start);
    let mut finished = vec![false; n_agents];
    let mut q: Vec<VertexId> = Vec::new();
    let mut current: Option<u32> = None;

    // Line 5: on a polygon every vertex has degree 2, so no swap vertex
    // exists and paths must avoid finished agents entirely. Decided per
    // connected component — a graph mixing a polygon component with a
    // branching one must treat each by its own rule.
    let n_components = ctx.decomp.empties_in_component.len();
    let mut polygon_component = vec![true; n_components];
    for v in graph.vertices() {
        if graph.degree(v) != 2 {
            polygon_component[ctx.decomp.component_of_vertex[v]] = false;
        }
    }

    // Agents are planned in subgraph-precedence order; within that, agents
    // assigned to no subgraph go last (§3.1.3).
    let order = ctx.heuristics.agent_order(ctx, &state);
    debug_assert_eq!(
        order.len(),
        n_agents,
        "agent_order must return every agent exactly once"
    );
    let mut next_idx = 0usize;

    while finished.iter().any(|&f| !f) {
        if state.moves().len() > budget {
            return Err(PlanError::BudgetExceeded { budget });
        }

        let r = match current.take() {
            Some(r) => r,
            None => {
                // Line 8: next unfinished agent by priority.
                while next_idx < order.len() && finished[order[next_idx] as usize] {
                    next_idx += 1;
                }
                if next_idx >= order.len() {
                    break;
                }
                order[next_idx]
            }
        };

        let mut blocked_finished = no_blocked(graph.len());
        for a in 0..n_agents {
            if finished[a] {
                blocked_finished[state.position_of(a as u32)] = true;
            }
        }

        // Lines 9-12: choose the path.
        let is_polygon = polygon_component[ctx.decomp.component_of_vertex[state.position_of(r)]];
        let path_block = if is_polygon {
            blocked_finished.clone()
        } else {
            no_blocked(graph.len())
        };
        let goal_v = goal[r as usize];
        let Some(mut path) =
            state.shortest_path_scored(state.position_of(r), goal_v, &path_block, |from, to| {
                ctx.heuristics.score_step(ctx, &state, r, from, to)
            })
        else {
            // With no blocking (non-polygon) this is graph disconnection —
            // a genuine proof. On a polygon, settled agents can never be
            // displaced, so a path avoiding them failing to exist is the
            // paper's line 9-12 unsolvability criterion.
            return Err(PlanError::Unsolvable {
                qubit: ctx.qubits[r as usize],
                reason: "no path to the target exists",
            });
        };
        q.push(state.position_of(r));

        // Line 14: advance r one vertex at a time.
        let mut step_i = 0usize;
        while state.position_of(r) != goal_v {
            if state.moves().len() > budget {
                return Err(PlanError::BudgetExceeded { budget });
            }
            if step_i >= path.len() {
                // The placement shifted under us (a swap moved r); re-plan.
                let Some(p) = state.shortest_path_scored(
                    state.position_of(r),
                    goal_v,
                    &path_block,
                    |from, to| ctx.heuristics.score_step(ctx, &state, r, from, to),
                ) else {
                    return Err(PlanError::Unsolvable {
                        qubit: ctx.qubits[r as usize],
                        reason: "no path to the target exists",
                    });
                };
                path = p;
                step_i = 0;
                if path.is_empty() {
                    break;
                }
            }
            let v = path[step_i];
            step_i += 1;

            if let Some(pos) = q.iter().position(|&x| x == v) {
                // Lines 16-19: a cycle of displaced agents.
                let cycle: Vec<VertexId> = q[pos..].to_vec();
                q.truncate(pos);
                if rotate(ctx, &mut state, &cycle) {
                    // The rotate moved r; recompute the path.
                    step_i = usize::MAX;
                    continue;
                }
                // Not a rotatable cycle. `q` accumulates vertices across
                // successive agents (the return loop below can break early
                // on purpose), so a repeated vertex can be a stale artifact
                // of an earlier agent's path rather than a closed walk of
                // displaced atoms. Fall through to the ordinary push/swap
                // handling of `v` instead of aborting the whole plan.
            }

            let mut blocked_now = no_blocked(graph.len());
            for a in 0..n_agents {
                if finished[a] {
                    blocked_now[state.position_of(a as u32)] = true;
                }
            }
            if !push(ctx, &mut state, r, v, &blocked_now) {
                // Line 22: push failed, so (Lemma 1) the blocker shares r's
                // subgraph and swap must succeed. If either expectation
                // fails, that is the planner losing its footing — not
                // evidence about the instance.
                let Some(s) = state.agent_at(v) else {
                    return Err(PlanError::Stuck {
                        qubit: ctx.qubits[r as usize],
                        reason: "push failed with no blocking atom to swap with",
                    });
                };
                if !swap(ctx, &mut state, r, s) {
                    return Err(PlanError::Stuck {
                        qubit: ctx.qubits[r as usize],
                        reason: "swap failed where Lemma 1 requires it to succeed",
                    });
                }
                step_i = usize::MAX;
            }
            q.push(v);
        }

        finished[r as usize] = true;
        current = None;

        // Lines 26-35: return agents a swap knocked off their goal.
        while let Some(&v) = q.last() {
            if state.moves().len() > budget {
                return Err(PlanError::BudgetExceeded { budget });
            }
            let Some(s) = state.agent_at(v) else {
                q.pop();
                continue;
            };
            if finished[s as usize] && v != goal[s as usize] {
                let target_v = goal[s as usize];
                match state.agent_at(target_v) {
                    None => {
                        if graph.neighbors(v).contains(&target_v) {
                            state.step(s, target_v);
                        } else {
                            break;
                        }
                    }
                    Some(blocker) => {
                        // Line 34: restart the outer loop with the blocker.
                        finished[blocker as usize] = false;
                        current = Some(blocker);
                        break;
                    }
                }
            }
            q.pop();
        }
    }

    // Only a genuine success if every agent actually reached its goal. On a
    // correct implementation this is unreachable for in-regime instances —
    // if it fires, it is a planner-bug sentinel, never a proof.
    for (a, &want) in goal.iter().enumerate() {
        if state.position_of(a as u32) != want {
            return Err(PlanError::Stuck {
                qubit: ctx.qubits[a],
                reason: "final placement check failed after planning completed",
            });
        }
    }

    // Algorithm 9. Push and Rotate generates round trips constantly — `clear`
    // shoves an agent aside and the swap reversal shoves it back — and while
    // pnr emits one atom per operation, every move removed is an operation
    // removed.
    let raw_move_count = state.moves().len();
    Ok(Plan {
        moves: smooth::smooth(state.moves()),
        raw_move_count,
    })
}
