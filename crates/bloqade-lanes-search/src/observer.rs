//! Search observability via event callbacks.
//!
//! The [`SearchObserver`] trait is the single hook for monitoring search
//! progress. Each driver — the frontier-based loop
//! [`run_search`](crate::drivers::frontier::run_search) and the entropy-guided
//! [`entropy_search`](crate::drivers::entropy::entropy_search) — fires
//! [`SearchEvent`]s at the points where a trace consumer needs a
//! snapshot. Use [`NoOpObserver`] when no observability is needed:
//! `on_event` is `#[inline(always)]` over an empty body so the dispatch
//! disappears under release-mode inlining.
//!
//! Implementations in this crate: [`NoOpObserver`] discards everything;
//! [`EntropyTrace`](crate::drivers::entropy::EntropyTrace) collects events into
//! a `Vec<EntropyTraceStep>` consumed by the Python visualization layer
//! (preserves the legacy step-record shape verbatim).

use crate::primitives::config::Config;
use crate::primitives::graph::{MoveSet, NodeId};

/// Event emitted during search.
///
/// The `'a` lifetime borrows from the driver's owned state (graph,
/// caches, move sets); observers that need to retain data past the
/// `on_event` call must copy out before returning.
#[derive(Debug, Clone)]
pub enum SearchEvent<'a> {
    // ── Frontier driver ────────────────────────────────────────────
    /// A goal node was discovered by the frontier loop.
    GoalFound {
        depth: u32,
        node_id: NodeId,
        config: &'a Config,
    },
    /// A node was popped from the frontier and expanded.
    NodeExpanded {
        depth: u32,
        num_candidates: usize,
        node_id: NodeId,
        config: &'a Config,
    },

    // ── Entropy driver ─────────────────────────────────────────────
    /// Entropy search descended from `parent_node_id` to a new
    /// `node_id` via the chosen `moveset` (selected by
    /// `candidate_index` from the parent's `candidate_movesets`).
    EntropyDescend {
        node_id: NodeId,
        parent_node_id: NodeId,
        depth: u32,
        entropy: u32,
        unresolved_count: u32,
        moveset: &'a MoveSet,
        candidate_movesets: &'a [MoveSet],
        candidate_index: u32,
        reason: Option<&'static str>,
        configuration: &'a Config,
        parent_configuration: &'a Config,
        moveset_score: f64,
        best_buffer_node_ids: &'a [u32],
    },
    /// Entropy search hit a goal node. Two sub-flavors:
    /// * fresh descent → `moveset == None`, `state_seen_node_id == None`,
    ///   `trigger_node_id == None`, `reason == None`.
    /// * transposition to an already-explored goal → `moveset` and
    ///   `candidate_index` describe the move that re-discovered it,
    ///   `state_seen_node_id` is the canonical node, `trigger_node_id`
    ///   is the node that proposed the move, `reason == Some("state-seen-goal")`.
    EntropyGoal {
        node_id: NodeId,
        parent_node_id: Option<NodeId>,
        depth: u32,
        entropy: u32,
        moveset: Option<&'a MoveSet>,
        candidate_movesets: &'a [MoveSet],
        candidate_index: Option<u32>,
        reason: Option<&'static str>,
        state_seen_node_id: Option<NodeId>,
        trigger_node_id: Option<NodeId>,
        configuration: &'a Config,
        parent_configuration: Option<&'a Config>,
        best_buffer_node_ids: &'a [u32],
    },
    /// Entropy search bumped a node's entropy counter. Two sub-flavors,
    /// distinguished by `reason`:
    /// * `"no-valid-moves"` — node has no improving candidates;
    ///   `moveset == None`, `no_valid_moves_qubit` names the first
    ///   unresolved qubit lacking a legal move.
    /// * `"state-seen"` — a tried candidate hit a transposition;
    ///   `moveset` and `candidate_index` describe it, and
    ///   `state_seen_node_id` is the previously-seen node.
    EntropyBump {
        node_id: NodeId,
        parent_node_id: Option<NodeId>,
        depth: u32,
        entropy: u32,
        unresolved_count: u32,
        moveset: Option<&'a MoveSet>,
        candidate_movesets: &'a [MoveSet],
        candidate_index: Option<u32>,
        reason: &'static str,
        state_seen_node_id: Option<NodeId>,
        no_valid_moves_qubit: Option<u32>,
        configuration: &'a Config,
        parent_configuration: Option<&'a Config>,
        best_buffer_node_ids: &'a [u32],
    },
    /// Entropy search reverted from `trigger_node_id` back to an
    /// ancestor (`node_id`) whose entropy is now `entropy`. The
    /// `trigger_entropy` value carries the entropy of the node that
    /// caused the revert.
    EntropyRevert {
        node_id: NodeId,
        parent_node_id: Option<NodeId>,
        depth: u32,
        entropy: u32,
        unresolved_count: u32,
        candidate_movesets: &'a [MoveSet],
        trigger_node_id: NodeId,
        trigger_entropy: u32,
        configuration: &'a Config,
        parent_configuration: Option<&'a Config>,
        best_buffer_node_ids: &'a [u32],
    },
    /// Entropy search exhausted its expansion budget without solving;
    /// sequential per-qubit fallback is about to run from the root.
    EntropyFallbackStart {
        node_id: NodeId,
        parent_node_id: Option<NodeId>,
        depth: u32,
        unresolved_count: u32,
        configuration: &'a Config,
        best_buffer_node_ids: &'a [u32],
    },
}

/// Callback trait for search observability.
///
/// Each search driver invokes `on_event` at the points where its trace
/// consumer needs a snapshot. The event payload borrows from the
/// driver's owned state — observers that need to retain data past
/// `on_event` must clone or transform before returning.
pub trait SearchObserver {
    fn on_event(&mut self, event: SearchEvent<'_>);
}

/// No-op observer that discards all events. Zero overhead.
pub struct NoOpObserver;

impl SearchObserver for NoOpObserver {
    #[inline(always)]
    fn on_event(&mut self, _event: SearchEvent<'_>) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::config::Config;
    use crate::test_utils::loc;

    /// Test-only observer that records the variant name of each event.
    struct LabelObserver {
        labels: Vec<&'static str>,
    }

    impl SearchObserver for LabelObserver {
        fn on_event(&mut self, event: SearchEvent<'_>) {
            self.labels.push(match event {
                SearchEvent::GoalFound { .. } => "GoalFound",
                SearchEvent::NodeExpanded { .. } => "NodeExpanded",
                SearchEvent::EntropyDescend { .. } => "EntropyDescend",
                SearchEvent::EntropyGoal { .. } => "EntropyGoal",
                SearchEvent::EntropyBump { .. } => "EntropyBump",
                SearchEvent::EntropyRevert { .. } => "EntropyRevert",
                SearchEvent::EntropyFallbackStart { .. } => "EntropyFallbackStart",
            });
        }
    }

    #[test]
    fn noop_observer_does_nothing() {
        let mut obs = NoOpObserver;
        let config = Config::new([(0, loc(0, 0))]).unwrap();
        obs.on_event(SearchEvent::GoalFound {
            depth: 0,
            node_id: NodeId(0),
            config: &config,
        });
    }

    #[test]
    fn observer_records_events_via_variant_name() {
        let mut obs = LabelObserver { labels: Vec::new() };
        let config = Config::new([(0, loc(0, 0))]).unwrap();
        obs.on_event(SearchEvent::GoalFound {
            depth: 5,
            node_id: NodeId(7),
            config: &config,
        });
        obs.on_event(SearchEvent::NodeExpanded {
            depth: 1,
            num_candidates: 3,
            node_id: NodeId(7),
            config: &config,
        });
        assert_eq!(obs.labels, vec!["GoalFound", "NodeExpanded"]);
    }
}
