//! Core search traits defining the composable search API.

use bloqade_lanes_bytecode_core::arch::addr::LaneAddr;

use crate::primitives::config::Config;
use crate::primitives::context::{MoveCandidate, SearchContext, SearchState};
use crate::primitives::graph::{MoveSet, NodeId};

/// Produces candidate move sets from a configuration.
pub trait MoveGenerator {
    fn generate(
        &self,
        config: &Config,
        node_id: NodeId,
        ctx: &SearchContext,
        state: &mut SearchState,
        out: &mut Vec<MoveCandidate>,
    );

    /// Number of deadlock occurrences tracked by this generator (default 0).
    fn deadlock_count(&self) -> u32 {
        0
    }
}

/// Ranks candidates produced by the generator.
/// Higher score = better candidate. Used to sort before graph insertion.
pub trait CandidateScorer {
    fn score(&self, candidate: &MoveCandidate, config: &Config, ctx: &SearchContext) -> f64;
}

/// Computes edge cost for g-score accumulation in the search graph.
/// Separate from candidate scoring -- this affects A* optimality guarantees.
pub trait CostFn {
    fn edge_cost(&self, move_set: &MoveSet, from: &Config, to: &Config) -> f64;
}

/// Identity of an [`Objective`] *instance*, its parameters included.
///
/// Two objectives with equal ids must agree on `edge_cost` and `lane_weight`
/// for every input. This is what lets a completion bound assert, at
/// construction, that it was built against the same objective instance the
/// driver accumulates `g` with — a bound paired to a different instance would
/// prune unsoundly, silently discarding better solutions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObjectiveId {
    /// Objective family, e.g. `"uniform"`.
    pub kind: &'static str,
    /// Bit-level digest of the instance's parameters; `0` when it has none.
    pub params: u64,
}

/// The quantity a search minimizes: the single source of truth for `g`.
///
/// [`CostFn::edge_cost`] is the per-shot increment `g` accumulates. This trait
/// extends it with the static per-lane data a completion bound needs, so that
/// an objective and the bounds admissible for it cannot drift apart.
///
/// # Stated constraints of the bound framework
///
/// These are requirements on implementors, not incidental properties of
/// today's cost model. Together they are exactly what makes a
/// weighted-distance completion bound admissible.
///
/// - **C1 — per-shot additive.** The cost of a plan is
///   `Σ_{shot ∈ plan} edge_cost(shot)`. A multiplicative objective (fidelity,
///   say) must be pre-transformed (`−log`) by the implementor; the framework
///   does not do it, and a non-additive objective is outside the framework.
/// - **C2 — non-negative.** `edge_cost(..) >= 0`.
/// - **C3 — lane floor.** For every shot `s` and every lane `l ∈ s`,
///   `edge_cost(s, ..) >= lane_weight(l)`.
/// - **C4 — shot floor.** `min_shot_cost() > 0` and is finite, and
///   `edge_cost(s, ..) >= min_shot_cost()` for *every* shot `s` — including an
///   empty one, which C3 says nothing about because it quantifies over the
///   lanes a shot contains. An implementor whose cost is
///   `base + f(lanes)` therefore has to seed `f`'s fold at its own minimum
///   rather than at zero.
///
/// `bounds::assert_objective_contract` — available under `cfg(test)` or the
/// `test-util` feature — checks C2, C3 and C4 mechanically over a lane sweep;
/// C1 is structural and cannot be tested from outside.
///
/// `Sync` because one objective is shared by reference across parallel
/// restarts.
pub trait Objective: CostFn + Sync {
    /// A floor on the cost of *any* shot containing `lane` (C3).
    ///
    /// Returning `0.0` is the explicit "no floor can be certified" opt-out:
    /// bounds derived from it are trivial (`h ≡ 0`), never unsound.
    fn lane_weight(&self, lane: LaneAddr) -> f64;

    /// A positive floor on any single shot's cost (C4).
    ///
    /// Converts a cost budget into a depth budget: no plan of cost `≤ c` is
    /// deeper than `floor(c / min_shot_cost())`. The pruning-depth
    /// instrumentation uses that to state how much *earlier* the bound cut than
    /// `g` alone could have. The cascade used to convert its incumbent this way
    /// too and no longer does — it bounds the refinement by cost directly,
    /// since the conversion only preserves the intended meaning while
    /// `g == depth`.
    fn min_shot_cost(&self) -> f64;

    /// Identity of this instance, parameters included.
    fn id(&self) -> ObjectiveId;
}

/// Decides when the search is complete.
pub trait Goal {
    fn is_goal(&self, config: &Config) -> bool;
}

/// Estimates cost-to-goal for A*/greedy search (h-function).
/// Must be admissible (never overestimates) for A* optimality.
pub trait Heuristic {
    fn estimate(&self, config: &Config) -> f64;
}

/// Blanket impl: any `Fn(&Config) -> f64` closure satisfies `Heuristic`.
impl<F: Fn(&Config) -> f64> Heuristic for F {
    fn estimate(&self, config: &Config) -> f64 {
        self(config)
    }
}
