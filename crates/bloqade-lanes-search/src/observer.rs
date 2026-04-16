//! Search observability via event callbacks.
//!
//! The [`SearchObserver`] trait provides a hook for monitoring search progress.
//! Implementations receive [`SearchEvent`]s at key points during search execution.
//! Use [`NoOpObserver`] when no observability is needed (zero overhead).

use crate::config::Config;

/// Event emitted during search for observability.
#[derive(Debug, Clone)]
pub enum SearchEvent {
    /// Search descended to a new child node.
    Descend { depth: u32, candidate_score: f64 },
    /// Goal configuration found.
    GoalFound { depth: u32 },
    /// Entropy bumped on a node (entropy search only).
    EntropyBump {
        node_depth: u32,
        new_entropy: u32,
        reason: &'static str,
    },
    /// Search reverted to ancestor (entropy search only).
    Revert {
        ancestor_depth: u32,
        reversion_steps: u32,
    },
    /// Sequential fallback started (entropy search only).
    FallbackStart { unresolved_count: usize },
    /// One step of sequential fallback.
    FallbackStep { qubit_id: u32 },
    /// Node expanded (popped from frontier and successors generated).
    NodeExpanded { depth: u32, num_candidates: usize },
}

/// Callback trait for search observability.
///
/// Implement this to receive events during search execution.
/// Each event includes the current configuration for context.
pub trait SearchObserver {
    fn on_event(&mut self, event: SearchEvent, config: &Config);
}

/// No-op observer that discards all events. Zero overhead.
pub struct NoOpObserver;

impl SearchObserver for NoOpObserver {
    #[inline(always)]
    fn on_event(&mut self, _event: SearchEvent, _config: &Config) {}
}

/// Observer that collects events into a Vec for later inspection.
#[cfg(test)]
pub(crate) struct CollectingObserver {
    pub events: Vec<SearchEvent>,
}

#[cfg(test)]
impl CollectingObserver {
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }
}

#[cfg(test)]
impl SearchObserver for CollectingObserver {
    fn on_event(&mut self, event: SearchEvent, _config: &Config) {
        self.events.push(event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::test_utils::loc;

    #[test]
    fn noop_observer_does_nothing() {
        let mut obs = NoOpObserver;
        let config = Config::new([(0, loc(0, 0))]).unwrap();
        obs.on_event(SearchEvent::GoalFound { depth: 0 }, &config);
        // No panic, no side effects.
    }

    #[test]
    fn collecting_observer_records_events() {
        let mut obs = CollectingObserver::new();
        let config = Config::new([(0, loc(0, 0))]).unwrap();
        obs.on_event(SearchEvent::GoalFound { depth: 5 }, &config);
        obs.on_event(
            SearchEvent::Descend {
                depth: 1,
                candidate_score: 3.0,
            },
            &config,
        );
        assert_eq!(obs.events.len(), 2);
        assert!(matches!(obs.events[0], SearchEvent::GoalFound { depth: 5 }));
    }
}
