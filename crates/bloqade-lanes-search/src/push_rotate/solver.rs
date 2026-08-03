//! [`SolveResult`]-producing entry point, so Push and Rotate composes with the
//! rest of the solver surface.
//!
//! Adapts the planner's `(LaneGraph, VertexId)` world to the
//! `(ArchSpec, LocationAddr)` one every other router speaks, and packages the
//! scheduled AOD operations as [`MoveSet`] layers.
//!
//! ## Why this exists as a peer rather than a `Strategy` inside the frontier
//!
//! The frontier drivers all share `pop, expand, push` and differ only in node
//! ordering. Push and Rotate is not a search at all — it has no frontier, no
//! node expansion and no heuristic to guide it, so `nodes_expanded` is
//! reported as `0`. What it shares with the others is the *contract*: take a
//! fixed target, return a `SolveResult`.

use std::collections::HashSet;

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;

use crate::feasibility::graph::{LaneGraph, VertexId};
use crate::primitives::config::{Config, ConfigError};
use crate::primitives::graph::MoveSet;
use crate::primitives::lane_index::LaneIndex;
use crate::push_rotate::heuristics::{DefaultHeuristics, PlanHeuristics};
use crate::push_rotate::{PlanError, plan_with, schedule::schedule};
use crate::search::result::{SolveResult, SolveStatus};

/// Default cap on emitted single-atom moves.
///
/// Push and Rotate is rule-based, so this is a runaway guard rather than a
/// search budget — a solvable instance is not expected to approach it.
pub const DEFAULT_MOVE_BUDGET: usize = 500_000;

/// Route `initial` to `target` with Push and Rotate.
///
/// Complete for instances with at least two empty locations: a
/// [`SolveStatus::Unsolvable`] result is a *proof* that no solution exists,
/// unlike the search drivers where it means the frontier drained.
///
/// # Errors
///
/// Returns [`ConfigError`] if `initial` contains duplicate qubit IDs.
pub fn solve_push_rotate(
    index: &LaneIndex,
    initial: &[(u32, LocationAddr)],
    target: &[(u32, LocationAddr)],
    blocked: &[LocationAddr],
    budget: usize,
) -> Result<SolveResult, ConfigError> {
    solve_push_rotate_with(index, initial, target, blocked, budget, &DefaultHeuristics)
}

/// As [`solve_push_rotate`], with an explicit heuristic strategy.
pub fn solve_push_rotate_with(
    index: &LaneIndex,
    initial: &[(u32, LocationAddr)],
    target: &[(u32, LocationAddr)],
    blocked: &[LocationAddr],
    budget: usize,
    heuristics: &dyn PlanHeuristics,
) -> Result<SolveResult, ConfigError> {
    let root = Config::new(initial.iter().copied())?;

    let blocked_set: HashSet<u64> = blocked.iter().map(|l| l.encode()).collect();
    let graph = LaneGraph::build(index, &blocked_set);

    // A location off the graph is not a planner failure — it is an instance
    // that cannot be expressed, so report it as unsolvable rather than
    // erroring.
    let Some(initial_v) = to_vertices(&graph, initial) else {
        return Ok(SolveResult::unsolvable(root));
    };
    let Some(target_v) = to_vertices(&graph, target) else {
        return Ok(SolveResult::unsolvable(root));
    };

    let plan = match plan_with(index, &graph, &initial_v, &target_v, budget, heuristics) {
        Ok(p) => p,
        Err(e) => {
            let status = match e {
                // Proven: the agent-to-subgraph assignment differs between
                // the initial and goal placements, so some atom would have to
                // leave the region it is confined to.
                PlanError::Unsolvable { .. } => SolveStatus::Unsolvable,
                // Not proven — say so rather than claiming impossibility.
                PlanError::BudgetExceeded { .. } => SolveStatus::BudgetExceeded,
                // Outside the completeness regime, or a malformed instance.
                PlanError::TooFewEmpty { .. } | PlanError::OffGraph { .. } => {
                    SolveStatus::Unsolvable
                }
            };
            return Ok(SolveResult::unsolved(status, root, 0, 0));
        }
    };

    let Some(batches) = schedule(index, &graph, &plan.moves) else {
        return Ok(SolveResult::unsolvable(root));
    };

    let move_layers: Vec<MoveSet> = batches
        .iter()
        .map(|b| MoveSet::new(b.lanes.to_vec()))
        .collect();

    let goal_config = Config::new(
        target_v
            .iter()
            .map(|&(q, v)| (q, LocationAddr::decode(graph.location_of(v)))),
    )?;

    // `nodes_expanded` is 0 by construction: this is not a search. Cost is the
    // operation count, matching `UniformCost` over the emitted layers so the
    // value is comparable with the frontier drivers'.
    let cost = move_layers.len() as f64;
    Ok(SolveResult::solved(goal_config, move_layers, cost, 0, 0))
}

fn to_vertices(graph: &LaneGraph, pairs: &[(u32, LocationAddr)]) -> Option<Vec<(u32, VertexId)>> {
    pairs
        .iter()
        .map(|&(q, loc)| graph.vertex_of(loc.encode()).map(|v| (q, v)))
        .collect()
}
