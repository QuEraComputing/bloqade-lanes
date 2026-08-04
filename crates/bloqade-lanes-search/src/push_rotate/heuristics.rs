//! Swappable decision points in the planner.
//!
//! Push and Rotate has no objective function — it is a completeness algorithm.
//! §5 of de Wilde, ter Mors & Witteveen identifies the four places where a
//! choice is free and affects solution quality, and this trait is exactly
//! those four:
//!
//! | § | Decision | Method |
//! |---|---|---|
//! | 1 | which empty vertex `clear_vertex` drains into | [`PlanHeuristics::rank_clear_target`] |
//! | 2 | which degree-≥3 vertex `swap` uses | [`PlanHeuristics::rank_swap_vertex`] |
//! | 3 | the order agents are planned in | [`PlanHeuristics::agent_order`] |
//! | 4 | which shortest path an agent takes | [`PlanHeuristics::score_step`] |
//!
//! The paper's authors explored only 3 and 4, and noted that agent ordering
//! "can be very important to solution quality". [`DefaultHeuristics`] does
//! neither — it is the unoptimised baseline the current benchmark numbers come
//! from.
//!
//! ## Why scores rather than choices
//!
//! Every method returns a *ranking key* instead of making the choice, and the
//! caller applies it as a tie-break on top of the existing rule. That keeps a
//! heuristic from breaking correctness: it can reorder equally-good options
//! but cannot select an illegal one or lengthen a path. A heuristic that
//! returns a constant is exactly the default.
//!
//! ## The one that matters next
//!
//! [`PlanHeuristics::score_step`] is where AOD alignment goes. `headroom`
//! shows the dependency DAG admits ~5x more parallelism than the scheduler
//! extracts, and the gap is entirely geometric: moves that are ready together
//! do not share a bus group or do not form a rectangle. Preferring steps that
//! align with other pending moves is what closes it, and [`PlanCtx::edge`]
//! exposes the bus group and source position needed to judge that.

use std::cell::RefCell;
use std::collections::HashMap;

use crate::feasibility::graph::VertexId;

use crate::push_rotate::context::{GroupKey, PlanCtx};
use crate::push_rotate::state::PlanState;

/// Strategy for the planner's free choices.
///
/// All methods have defaults reproducing the current behaviour, so an
/// implementation need only override what it cares about.
pub trait PlanHeuristics {
    /// Order in which agents are planned.
    ///
    /// Must contain every agent exactly once. Subgraph precedence still
    /// applies on top: agents assigned to no subgraph are planned last
    /// regardless (§3.1.3).
    fn agent_order(&self, ctx: &PlanCtx, state: &PlanState) -> Vec<u32> {
        let _ = state;
        let n = ctx.goal.len();
        let mut assigned: Vec<u32> = Vec::new();
        let mut unassigned: Vec<u32> = Vec::new();
        for a in 0..n as u32 {
            match ctx.decomp.assignment.get(&a) {
                Some(_) => assigned.push(a),
                None => unassigned.push(a),
            }
        }
        assigned.sort_by_key(|a| {
            (
                ctx.decomp.assignment.get(a).copied().unwrap_or(usize::MAX),
                *a,
            )
        });
        assigned.extend(unassigned);
        assigned
    }

    /// Preference for `agent` stepping `from -> to`, used to break ties
    /// between equally short paths. **Higher is better.**
    ///
    /// Only consulted among steps that are already on a shortest path, so it
    /// cannot lengthen a route.
    fn score_step(
        &self,
        ctx: &PlanCtx,
        state: &PlanState,
        agent: u32,
        from: VertexId,
        to: VertexId,
    ) -> f64 {
        let _ = (ctx, state, agent, from, to);
        0.0
    }

    /// Ranking key for an empty vertex that `clear_vertex` could drain into.
    /// **Lower is better.** `hops` is its distance from the vertex being
    /// cleared.
    fn rank_clear_target(
        &self,
        ctx: &PlanCtx,
        state: &PlanState,
        clearing: VertexId,
        candidate: VertexId,
        hops: u32,
    ) -> i64 {
        let _ = (ctx, state, clearing, candidate);
        hops as i64
    }

    /// Ranking key for a candidate swap vertex. **Lower is better.**
    /// `hops` is its distance from the agent initiating the swap.
    fn rank_swap_vertex(
        &self,
        ctx: &PlanCtx,
        state: &PlanState,
        r: u32,
        s: u32,
        candidate: VertexId,
        hops: u32,
    ) -> i64 {
        let _ = (ctx, state, r, s, candidate);
        hops as i64
    }
}

/// The unoptimised baseline: nearest-first everywhere, arbitrary agent order.
///
/// Every method is the trait default, so this is the behaviour all current
/// benchmark numbers were measured with. Keep it as the control when
/// evaluating anything else.
#[derive(Debug, Default, Clone, Copy)]
pub struct DefaultHeuristics;

impl PlanHeuristics for DefaultHeuristics {}

/// Alignment-aware strategy: steer concurrent moves onto a *shared bus*.
///
/// ## What the measurement said
///
/// The first version of this biased atoms towards shared rows, on the theory
/// that same-y atoms form a 1xN rectangle. It bought almost nothing, and
/// instrumenting the scheduler showed why. At k=16, with ~6.8 moves ready
/// simultaneously, the largest subset sharing a bus group averaged **1.57**,
/// and the largest same-row subset within a group averaged 1.18. The
/// histogram of "largest same-group subset in the ready set" was
/// `1 -> 73 steps, 2 -> 72, 3 -> 7`.
///
/// So rectangle *shape* was never the binding constraint. Concurrent moves
/// were simply on different buses, and two moves on different buses can never
/// share an operation whatever their geometry.
///
/// ## What this does instead
///
/// A bus moves a whole family of word pairs at once — Gemini's word bus 0
/// carries 0->1, 4->5, 8->9 and so on — so atoms making the *same relative
/// move* can batch even when far apart. This scores a step by how many **other
/// unfinished atoms could also make progress on that same bus right now**, and
/// so pulls concurrent moves onto shared buses.
///
/// ## Cost
///
/// Distance-to-goal per agent is a graph property and does not change as atoms
/// move, so it is computed once per solve. The per-bus support counts do
/// change, and are rebuilt only when the placement advances —
/// `PlanState::moves().len()` is a monotonic clock for that. A rebuild is
/// `O(agents x degree)`, a few hundred operations, against the thousands of
/// `score_step` calls a single BFS makes.
pub struct AlignmentHeuristics {
    /// Weight on routing onto a bus other atoms can also use. This is the
    /// whole heuristic: a separate flat preference for word buses was tried
    /// and ablation showed it contributed nothing on top of the support
    /// term (physical/logical k16 operation counts identical with and
    /// without it), so it was removed.
    pub bus_weight: f64,
    cache: RefCell<Cache>,
}

#[derive(Default)]
struct Cache {
    /// `state.moves().len()` when `support` was built.
    clock: usize,
    /// Hop distance to its goal for every agent, indexed by agent then vertex.
    /// Occupancy-independent, so computed once.
    goal_dist: Vec<Vec<u32>>,
    /// Bus group → how many unfinished atoms could make progress on it now.
    support: HashMap<GroupKey, u32>,
}

impl Default for AlignmentHeuristics {
    fn default() -> Self {
        Self {
            bus_weight: 1.0,
            cache: RefCell::new(Cache {
                clock: usize::MAX,
                ..Default::default()
            }),
        }
    }
}

impl AlignmentHeuristics {
    pub fn new(bus_weight: f64) -> Self {
        Self {
            bus_weight,
            ..Default::default()
        }
    }

    fn refresh(&self, ctx: &PlanCtx, state: &PlanState) {
        let clock = state.moves().len();
        let mut cache = self.cache.borrow_mut();
        if cache.clock == clock && !cache.goal_dist.is_empty() {
            return;
        }
        if cache.goal_dist.is_empty() {
            cache.goal_dist = ctx
                .goal
                .iter()
                .map(|&g| ctx.graph.distances_from(g, |_| false))
                .collect();
        }
        cache.support.clear();
        for a in 0..state.agent_count() {
            let at = state.position_of(a as u32);
            let Some(dist) = cache.goal_dist.get(a) else {
                continue;
            };
            let here = dist[at];
            if here == 0 {
                continue; // already home; it will not move again
            }
            // Every bus offering this atom a step closer to its goal counts as
            // support for that bus.
            let mut seen: Vec<GroupKey> = Vec::new();
            for &w in ctx.graph.neighbors(at) {
                if dist[w] >= here {
                    continue;
                }
                if let Some(info) = ctx.edge(at, w)
                    && !seen.contains(&info.group)
                {
                    seen.push(info.group);
                }
            }
            for g in seen {
                *cache.support.entry(g).or_insert(0) += 1;
            }
        }
        cache.clock = clock;
    }
}

impl PlanHeuristics for AlignmentHeuristics {
    // `_agent` is part of the trait signature (a heuristic may score by
    // agent identity); this one scores purely by bus-group company.
    fn score_step(
        &self,
        ctx: &PlanCtx,
        state: &PlanState,
        _agent: u32,
        from: VertexId,
        to: VertexId,
    ) -> f64 {
        let Some(info) = ctx.edge(from, to) else {
            return 0.0;
        };
        self.refresh(ctx, state);
        let cache = self.cache.borrow();

        // Discount this agent's own contribution, so the score reflects how
        // much *company* the step would have.
        let peers = cache
            .support
            .get(&info.group)
            .copied()
            .unwrap_or(0)
            .saturating_sub(1);
        self.bus_weight * peers as f64
    }
}
