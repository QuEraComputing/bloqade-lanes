//! Distance-based candidate scorer.
//!
//! Scores a [`MoveCandidate`] by how much closer it moves qubits to their targets,
//! measured in lane-hop distance.

use crate::config::Config;
use crate::context::{MoveCandidate, SearchContext};
use crate::traits::CandidateScorer;

/// Scores a candidate by the total reduction in hop distance across all target qubits.
///
/// For each target `(qid, target_enc)`, computes `d_before - d_after` where
/// `d_before` is the distance from the qubit's current location and `d_after`
/// is the distance from the qubit's location in the candidate configuration.
/// Returns the sum — positive means the candidate moves qubits closer overall.
pub struct DistanceScorer;

impl CandidateScorer for DistanceScorer {
    fn score(&self, candidate: &MoveCandidate, config: &Config, ctx: &SearchContext) -> f64 {
        let mut total: f64 = 0.0;
        for &(qid, target_enc) in ctx.targets {
            let before_enc = match config.location_of(qid) {
                Some(loc) => loc.encode(),
                None => continue,
            };
            let after_enc = match candidate.new_config.location_of(qid) {
                Some(loc) => loc.encode(),
                None => continue,
            };

            let d_before = ctx
                .dist_table
                .distance(before_enc, target_enc)
                .unwrap_or(u32::MAX) as f64;
            let d_after = ctx
                .dist_table
                .distance(after_enc, target_enc)
                .unwrap_or(u32::MAX) as f64;

            total += d_before - d_after;
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::context::{MoveCandidate, SearchContext};
    use crate::graph::MoveSet;
    use crate::heuristic::DistanceTable;
    use crate::lane_index::LaneIndex;
    use crate::test_utils::{example_arch_json, loc};
    use bloqade_lanes_bytecode_core::arch::types::ArchSpec;
    use std::collections::HashSet;

    fn make_index() -> LaneIndex {
        let spec: ArchSpec = serde_json::from_str(example_arch_json()).unwrap();
        LaneIndex::new(spec)
    }

    #[test]
    fn score_positive_when_closer() {
        let index = make_index();
        let target = loc(0, 5);
        let target_enc = target.encode();
        let targets_vec = vec![(0u32, target_enc)];
        let target_locs = vec![target_enc];
        let table = DistanceTable::new(&target_locs, &index);
        let blocked = HashSet::new();

        let ctx = SearchContext {
            index: &index,
            dist_table: &table,
            blocked: &blocked,
            targets: &targets_vec,
            cz_pairs: None,
        };

        // Qubit starts at site 0, candidate moves it to site 5 (the target).
        let config = Config::new([(0, loc(0, 0))]).unwrap();
        let new_config = Config::new([(0, loc(0, 5))]).unwrap();
        let candidate = MoveCandidate {
            move_set: MoveSet::new(std::iter::empty()),
            new_config,
        };

        let score = DistanceScorer.score(&candidate, &config, &ctx);
        assert!(
            score > 0.0,
            "score should be positive when moving closer, got {score}"
        );
    }

    #[test]
    fn score_zero_when_no_change() {
        let index = make_index();
        let target = loc(0, 5);
        let target_enc = target.encode();
        let targets_vec = vec![(0u32, target_enc)];
        let target_locs = vec![target_enc];
        let table = DistanceTable::new(&target_locs, &index);
        let blocked = HashSet::new();

        let ctx = SearchContext {
            index: &index,
            dist_table: &table,
            blocked: &blocked,
            targets: &targets_vec,
            cz_pairs: None,
        };

        let config = Config::new([(0, loc(0, 0))]).unwrap();
        let new_config = Config::new([(0, loc(0, 0))]).unwrap();
        let candidate = MoveCandidate {
            move_set: MoveSet::new(std::iter::empty()),
            new_config,
        };

        let score = DistanceScorer.score(&candidate, &config, &ctx);
        assert_eq!(score, 0.0, "score should be zero when position unchanged");
    }

    #[test]
    fn score_negative_when_farther() {
        let index = make_index();
        // Target is site 0, qubit starts at site 5 (1 hop), candidate moves to site in word 1 (farther).
        let target = loc(0, 0);
        let target_enc = target.encode();
        let targets_vec = vec![(0u32, target_enc)];
        let target_locs = vec![target_enc];
        let table = DistanceTable::new(&target_locs, &index);
        let blocked = HashSet::new();

        let ctx = SearchContext {
            index: &index,
            dist_table: &table,
            blocked: &blocked,
            targets: &targets_vec,
            cz_pairs: None,
        };

        // Start at site 5 (1 hop from target site 0 via site bus).
        // Move to word 1, site 5 (farther from target).
        let config = Config::new([(0, loc(0, 5))]).unwrap();
        let new_config = Config::new([(0, loc(1, 5))]).unwrap();
        let candidate = MoveCandidate {
            move_set: MoveSet::new(std::iter::empty()),
            new_config,
        };

        let score = DistanceScorer.score(&candidate, &config, &ctx);
        assert!(
            score < 0.0,
            "score should be negative when moving farther, got {score}"
        );
    }
}
