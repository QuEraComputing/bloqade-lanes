//! Push and Rotate: a complete multi-agent pathfinding planner.
//!
//! de Wilde, ter Mors & Witteveen, *Push and Rotate: a Complete Multi-agent
//! Pathfinding Algorithm*, JAIR 51 (2014) 443–492. Complete for instances with
//! at least two empty vertices (Theorem 1): it finds a solution whenever one
//! exists, and returns [`PlanError::Unsolvable`] otherwise.
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

use crate::feasibility::decomposition::{
    Decomposition, assign_agents, find_planks, find_subgraphs,
};
use crate::feasibility::graph::{LaneGraph, VertexId};
use crate::primitives::lane_index::LaneIndex;

use crate::push_rotate::context::PlanCtx;
use crate::push_rotate::heuristics::{DefaultHeuristics, PlanHeuristics};
use crate::push_rotate::ops::{no_blocked, push, rotate, swap};
use crate::push_rotate::state::{Move, PlanState};

/// Why planning stopped without a solution.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PlanError {
    /// Proven unsolvable. Algorithm 7 line 5: the agent→subgraph assignment
    /// computed from the initial placement differs from the one computed from
    /// the goal, so some agent would have to leave the region Proposition 1
    /// confines it to.
    #[error(
        "instance is unsolvable: agent {agent} is confined to a different subgraph in the goal"
    )]
    Unsolvable { agent: u32 },
    /// Fewer than two empty vertices. Push and Rotate's completeness result
    /// does not cover this regime.
    #[error("push and rotate requires at least 2 empty vertices, found {found}")]
    TooFewEmpty { found: usize },
    /// An atom or target is not a vertex of the lane graph.
    #[error("location {location:#x} for qubit {agent} is blocked or not on any lane")]
    OffGraph { agent: u32, location: u64 },
    /// The planner exceeded its step budget. Distinct from `Unsolvable`: this
    /// says nothing about whether a solution exists.
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
    let empty_count = graph.len().saturating_sub(n_agents);
    if empty_count < 2 {
        return Err(PlanError::TooFewEmpty { found: empty_count });
    }

    // Dense agent indexing: the planner works with 0..n_agents internally.
    let mut agent_of_qubit: HashMap<u32, u32> = HashMap::new();
    let mut start = vec![0usize; n_agents];
    for (i, &(qubit, v)) in initial.iter().enumerate() {
        agent_of_qubit.insert(qubit, i as u32);
        start[i] = v;
    }
    let mut goal = start.clone();
    for &(qubit, v) in target {
        if let Some(&a) = agent_of_qubit.get(&qubit) {
            goal[a as usize] = v;
        }
    }

    // ── Algorithm 7 ────────────────────────────────────────────────
    let bcc = graph.biconnected();
    let (subgraphs, subgraph_of_vertex) = find_subgraphs(graph, &bcc, empty_count);
    let planks = find_planks(graph, &subgraphs, &subgraph_of_vertex, empty_count);

    let occupancy = |placement: &[VertexId]| -> Vec<Option<u32>> {
        let mut occ = vec![None; graph.len()];
        for (a, &v) in placement.iter().enumerate() {
            occ[v] = Some(a as u32);
        }
        occ
    };

    let f_initial = assign_agents(
        graph,
        &subgraphs,
        &subgraph_of_vertex,
        &occupancy(&start),
        empty_count,
    );
    let f_goal = assign_agents(
        graph,
        &subgraphs,
        &subgraph_of_vertex,
        &occupancy(&goal),
        empty_count,
    );

    // Line 5: `if f = f'`. A mismatch means some agent is confined to one
    // subgraph at the start and a different one at the goal, which
    // Proposition 1 forbids — so the instance is unsolvable.
    for a in 0..n_agents as u32 {
        if f_initial.get(&a) != f_goal.get(&a) {
            let qubit = initial[a as usize].0;
            return Err(PlanError::Unsolvable { agent: qubit });
        }
    }

    let decomp = Decomposition {
        subgraphs,
        subgraph_of_vertex,
        planks,
        assignment: f_initial,
        empty_count,
    };

    let ctx = PlanCtx::new(graph, index, &decomp, &goal, heuristics);
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
    // exists and paths must avoid finished agents entirely.
    let is_polygon = graph.vertices().all(|v| graph.degree(v) == 2);

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
            return Err(PlanError::Unsolvable { agent: r });
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
                    return Err(PlanError::Unsolvable { agent: r });
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
                if !rotate(ctx, &mut state, &cycle) {
                    return Err(PlanError::Unsolvable { agent: r });
                }
                // The rotate moved r; recompute the path.
                step_i = usize::MAX;
                continue;
            }

            let mut blocked_now = no_blocked(graph.len());
            for a in 0..n_agents {
                if finished[a] {
                    blocked_now[state.position_of(a as u32)] = true;
                }
            }
            if !push(ctx, &mut state, r, v, &blocked_now) {
                // Line 22: push failed, so (Lemma 1) the blocker shares r's
                // subgraph and swap must succeed.
                let Some(s) = state.agent_at(v) else {
                    return Err(PlanError::Unsolvable { agent: r });
                };
                if !swap(ctx, &mut state, r, s) {
                    return Err(PlanError::Unsolvable { agent: r });
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

    // Only a genuine success if every agent actually reached its goal.
    for (a, &want) in goal.iter().enumerate() {
        if state.position_of(a as u32) != want {
            return Err(PlanError::Unsolvable { agent: a as u32 });
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
