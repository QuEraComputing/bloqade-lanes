//! Goal implementations.

use std::collections::HashSet;

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

/// Goal: all qubits in the configuration are located within a specific zone.
///
/// A qubit is "in zone" if its location's `word_id` is in the zone's word set.
pub struct ZoneGoal {
    zone_words: HashSet<u32>,
}

impl ZoneGoal {
    /// Create a zone goal from the set of word IDs belonging to the target zone.
    pub fn new(zone_words: impl IntoIterator<Item = u32>) -> Self {
        Self {
            zone_words: zone_words.into_iter().collect(),
        }
    }
}

impl Goal for ZoneGoal {
    fn is_goal(&self, config: &Config) -> bool {
        config
            .iter()
            .all(|(_, loc)| self.zone_words.contains(&loc.word_id))
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

    // ── ZoneGoal tests ──

    #[test]
    fn zone_goal_all_in_zone() {
        // Zone with word_ids {0, 1}
        let goal = ZoneGoal::new([0, 1]);
        let config = Config::new([(0, loc(0, 0)), (1, loc(0, 1))]).unwrap();
        // loc(zone, site) creates LocationAddr { zone_id: zone, word_id: 0, site_id: site }
        // word_id=0 is in zone_words
        assert!(goal.is_goal(&config));
    }

    #[test]
    fn zone_goal_one_outside() {
        let goal = ZoneGoal::new([0]);
        // loc(0, 0) has word_id=0 (in zone), but we need to test with word_id not in set
        // loc helper sets word_id=0, so let's use a custom LocationAddr
        use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;
        let config = Config::new([
            (
                0,
                LocationAddr {
                    zone_id: 0,
                    word_id: 0,
                    site_id: 0,
                },
            ),
            (
                1,
                LocationAddr {
                    zone_id: 0,
                    word_id: 5,
                    site_id: 0,
                },
            ), // word_id=5 not in zone
        ])
        .unwrap();
        assert!(!goal.is_goal(&config));
    }
}
