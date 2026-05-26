//! Goal implementations.

use std::collections::{HashMap, HashSet};

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;

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

/// Goal: all CZ pairs are at valid entangling positions AND no spectator
/// qubits are in accidental CZ positions.
///
/// A spectator qubit is one not listed in any CZ pair. An accidental CZ
/// occurs when two spectators occupy partner sites in the entangling set.
pub struct EntanglingConstraintGoal {
    /// Required CZ pairs: `(qubit_a, qubit_b)`.
    pairs: Vec<(u32, u32)>,
    /// Precomputed set of valid `(encoded_loc_a, encoded_loc_b)` pairs.
    /// Both orderings are stored.
    valid_placements: HashSet<(u64, u64)>,
    /// Qubits participating in CZ pairs (both sides of each pair).
    cz_qubits: HashSet<u32>,
    /// For each encoded entangling location, its CZ partner location.
    partner_map: HashMap<u64, u64>,
}

impl EntanglingConstraintGoal {
    /// Create from CZ pairs and a precomputed entangling set.
    ///
    /// Use [`crate::entangling::build_entangling_set`] to construct the set.
    pub fn new(pairs: &[(u32, u32)], valid_placements: HashSet<(u64, u64)>) -> Self {
        let cz_qubits: HashSet<u32> = pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
        let partner_map = crate::entangling::build_partner_map(&valid_placements);
        Self {
            pairs: pairs.to_vec(),
            valid_placements,
            cz_qubits,
            partner_map,
        }
    }
}

impl Goal for EntanglingConstraintGoal {
    fn is_goal(&self, config: &Config) -> bool {
        // Check all CZ pairs are at valid entangling positions.
        let pairs_ok = self.pairs.iter().all(|&(qa, qb)| {
            let loc_a = config.location_of(qa).map(|l| l.encode());
            let loc_b = config.location_of(qb).map(|l| l.encode());
            match (loc_a, loc_b) {
                (Some(a), Some(b)) => self.valid_placements.contains(&(a, b)),
                _ => false,
            }
        });
        if !pairs_ok {
            return false;
        }

        // Check no accidental CZ among spectators.
        for (qid, loc) in config.iter() {
            if self.cz_qubits.contains(&qid) {
                continue;
            }
            let loc_enc = loc.encode();
            if let Some(&partner_enc) = self.partner_map.get(&loc_enc)
                && let Some(other_qid) = config.qubit_at(LocationAddr::decode(partner_enc))
                && !self.cz_qubits.contains(&other_qid)
            {
                return false; // accidental CZ
            }
        }
        true
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

    // ── EntanglingConstraintGoal tests ──

    fn example_entangling_set() -> HashSet<(u64, u64)> {
        crate::entangling::build_entangling_set(
            &serde_json::from_str(crate::test_utils::example_arch_json()).unwrap(),
        )
    }

    #[test]
    fn entangling_goal_satisfied() {
        let eset = example_entangling_set();
        let goal = EntanglingConstraintGoal::new(&[(0, 1)], eset);
        // q0 on word 0 site 5, q1 on word 1 site 5 — valid entangling pair.
        let config = Config::new([(0, loc(0, 5)), (1, loc(1, 5))]).unwrap();
        assert!(goal.is_goal(&config));
    }

    #[test]
    fn entangling_goal_wrong_site() {
        let eset = example_entangling_set();
        let goal = EntanglingConstraintGoal::new(&[(0, 1)], eset);
        // Same words but different sites — not valid.
        let config = Config::new([(0, loc(0, 3)), (1, loc(1, 5))]).unwrap();
        assert!(!goal.is_goal(&config));
    }

    #[test]
    fn entangling_goal_same_word() {
        let eset = example_entangling_set();
        let goal = EntanglingConstraintGoal::new(&[(0, 1)], eset);
        // Both on same word — not an entangling pair.
        let config = Config::new([(0, loc(0, 5)), (1, loc(0, 6))]).unwrap();
        assert!(!goal.is_goal(&config));
    }

    #[test]
    fn entangling_goal_reversed_words() {
        let eset = example_entangling_set();
        let goal = EntanglingConstraintGoal::new(&[(0, 1)], eset);
        // q0 on word 1, q1 on word 0 — should still be valid (both orderings stored).
        let config = Config::new([(0, loc(1, 5)), (1, loc(0, 5))]).unwrap();
        assert!(goal.is_goal(&config));
    }

    #[test]
    fn entangling_goal_missing_qubit() {
        let eset = example_entangling_set();
        let goal = EntanglingConstraintGoal::new(&[(0, 1)], eset);
        // q1 not in config.
        let config = Config::new([(0, loc(0, 5))]).unwrap();
        assert!(!goal.is_goal(&config));
    }

    #[test]
    fn entangling_goal_multiple_pairs() {
        let eset = example_entangling_set();
        let goal = EntanglingConstraintGoal::new(&[(0, 1), (2, 3)], eset);
        // Both pairs satisfied.
        let config = Config::new([
            (0, loc(0, 5)),
            (1, loc(1, 5)),
            (2, loc(0, 6)),
            (3, loc(1, 6)),
        ])
        .unwrap();
        assert!(goal.is_goal(&config));
    }

    #[test]
    fn entangling_goal_one_pair_unsatisfied() {
        let eset = example_entangling_set();
        let goal = EntanglingConstraintGoal::new(&[(0, 1), (2, 3)], eset);
        // First pair ok, second pair on wrong sites.
        let config = Config::new([
            (0, loc(0, 5)),
            (1, loc(1, 5)),
            (2, loc(0, 6)),
            (3, loc(1, 7)),
        ])
        .unwrap();
        assert!(!goal.is_goal(&config));
    }

    #[test]
    fn entangling_goal_empty_pairs() {
        let eset = example_entangling_set();
        let goal = EntanglingConstraintGoal::new(&[], eset);
        // No pairs to satisfy — trivially true (no spectator conflict either).
        let config = Config::new([(0, loc(0, 0))]).unwrap();
        assert!(goal.is_goal(&config));
    }

    // ── Accidental CZ tests ──

    #[test]
    fn entangling_goal_rejects_accidental_cz() {
        let eset = example_entangling_set();
        // CZ pair (0, 1) — qubits 2 and 3 are spectators.
        let goal = EntanglingConstraintGoal::new(&[(0, 1)], eset);
        // q0/q1 at valid CZ positions. q2/q3 at partner sites = accidental CZ.
        let config = Config::new([
            (0, loc(0, 5)),
            (1, loc(1, 5)),
            (2, loc(0, 6)),
            (3, loc(1, 6)),
        ])
        .unwrap();
        assert!(!goal.is_goal(&config));
    }

    #[test]
    fn entangling_goal_accepts_spectator_with_empty_partner() {
        let eset = example_entangling_set();
        let goal = EntanglingConstraintGoal::new(&[(0, 1)], eset);
        // q0/q1 at valid CZ. q2 at entangling site but partner is empty.
        let config = Config::new([(0, loc(0, 5)), (1, loc(1, 5)), (2, loc(0, 6))]).unwrap();
        assert!(goal.is_goal(&config));
    }

    #[test]
    fn entangling_goal_accepts_spectator_paired_with_cz_participant() {
        let eset = example_entangling_set();
        let goal = EntanglingConstraintGoal::new(&[(0, 1)], eset);
        // q0 at (word 0, site 5), q1 at (word 1, site 5) — CZ pair.
        // q2 at (word 0, site 5) can't be — same location as q0.
        // Instead: q2 at (word 1, site 6), partner (word 0, site 6) is empty.
        let config = Config::new([(0, loc(0, 5)), (1, loc(1, 5)), (2, loc(1, 6))]).unwrap();
        assert!(goal.is_goal(&config));
    }
}
