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
}
