//! Goal implementations.

use crate::config::Config;
use crate::traits::Goal;

/// Goal: all qubits are at their encoded target locations.
pub struct AllAtTarget {
    targets: Vec<(u32, u64)>,
}

impl AllAtTarget {
    /// Create a new goal from `(qubit_id, encoded_target_location)` pairs.
    pub fn new(targets: &[(u32, u64)]) -> Self {
        Self {
            targets: targets.to_vec(),
        }
    }
}

impl Goal for AllAtTarget {
    fn is_goal(&self, config: &Config) -> bool {
        self.targets.iter().all(|&(qid, target_enc)| {
            config
                .location_of(qid)
                .is_some_and(|l| l.encode() == target_enc)
        })
    }
}

/// Goal: at least `min_placed` qubits are at their target locations.
///
/// When `min_placed == targets.len()`, behaves identically to [`AllAtTarget`].
pub struct PartialPlacementGoal {
    targets: Vec<(u32, u64)>,
    min_placed: usize,
}

impl PartialPlacementGoal {
    /// Create a new partial placement goal.
    ///
    /// `min_placed` is the minimum number of qubits that must be at their target.
    /// If `None`, defaults to all qubits (same as `AllAtTarget`).
    pub fn new(targets: &[(u32, u64)], min_placed: Option<usize>) -> Self {
        Self {
            min_placed: min_placed.unwrap_or(targets.len()),
            targets: targets.to_vec(),
        }
    }
}

impl Goal for PartialPlacementGoal {
    fn is_goal(&self, config: &Config) -> bool {
        let placed = self
            .targets
            .iter()
            .filter(|&&(qid, target_enc)| {
                config
                    .location_of(qid)
                    .is_some_and(|l| l.encode() == target_enc)
            })
            .count();
        placed >= self.min_placed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::loc;

    #[test]
    fn all_at_target_when_matched() {
        let targets = vec![(0u32, loc(0, 5).encode())];
        let goal = AllAtTarget::new(&targets);
        let config = Config::new([(0, loc(0, 5))]).unwrap();
        assert!(goal.is_goal(&config));
    }

    #[test]
    fn all_at_target_when_not_matched() {
        let targets = vec![(0u32, loc(0, 5).encode())];
        let goal = AllAtTarget::new(&targets);
        let config = Config::new([(0, loc(0, 0))]).unwrap();
        assert!(!goal.is_goal(&config));
    }

    #[test]
    fn all_at_target_missing_qubit() {
        let targets = vec![(0u32, loc(0, 5).encode())];
        let goal = AllAtTarget::new(&targets);
        let config = Config::new([(1, loc(0, 5))]).unwrap();
        assert!(!goal.is_goal(&config));
    }

    // ── PartialPlacementGoal tests ──

    #[test]
    fn partial_all_placed() {
        let targets = vec![(0u32, loc(0, 5).encode()), (1, loc(0, 3).encode())];
        let goal = PartialPlacementGoal::new(&targets, Some(2));
        let config = Config::new([(0, loc(0, 5)), (1, loc(0, 3))]).unwrap();
        assert!(goal.is_goal(&config));
    }

    #[test]
    fn partial_one_of_two() {
        let targets = vec![(0u32, loc(0, 5).encode()), (1, loc(0, 3).encode())];
        let goal = PartialPlacementGoal::new(&targets, Some(1));
        let config = Config::new([(0, loc(0, 5)), (1, loc(0, 0))]).unwrap();
        assert!(goal.is_goal(&config)); // only q0 placed, but min_placed=1
    }

    #[test]
    fn partial_none_placed() {
        let targets = vec![(0u32, loc(0, 5).encode())];
        let goal = PartialPlacementGoal::new(&targets, Some(1));
        let config = Config::new([(0, loc(0, 0))]).unwrap();
        assert!(!goal.is_goal(&config));
    }

    #[test]
    fn partial_none_defaults_to_all() {
        let targets = vec![(0u32, loc(0, 5).encode()), (1, loc(0, 3).encode())];
        let goal = PartialPlacementGoal::new(&targets, None);
        // Only q0 placed — needs both
        let config = Config::new([(0, loc(0, 5)), (1, loc(0, 0))]).unwrap();
        assert!(!goal.is_goal(&config));
    }
}
