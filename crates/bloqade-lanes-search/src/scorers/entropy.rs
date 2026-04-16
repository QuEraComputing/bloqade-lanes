//! Entropy-weighted candidate scorer.
//!
//! Scores movesets using `alpha * distance_progress + beta * arrived + gamma * mobility_gain`,
//! matching the Python `CandidateScorer.score_moveset()` formula.

use std::collections::HashSet;

use crate::config::Config;
use crate::context::{MoveCandidate, SearchContext};
use crate::entropy::EntropyParams;
use crate::traits::CandidateScorer;

/// Scores candidates using the entropy moveset formula:
/// `alpha * distance_progress + beta * arrived + gamma * mobility_gain`.
///
/// This is the "Level 2" scoring from the Python `CandidateScorer`.
/// Level 1 (per-qubit-lane entropy-weighted filtering) happens inside
/// the `EntropyGenerator`.
pub struct EntropyScorer {
    params: EntropyParams,
}

impl EntropyScorer {
    pub fn new(params: EntropyParams) -> Self {
        Self { params }
    }
}

impl CandidateScorer for EntropyScorer {
    fn score(&self, candidate: &MoveCandidate, config: &Config, ctx: &SearchContext) -> f64 {
        // Build occupied set from current config + blocked.
        let mut occupied: HashSet<u64> = HashSet::with_capacity(ctx.blocked.len() + config.len());
        occupied.extend(ctx.blocked);
        for (_, loc) in config.iter() {
            occupied.insert(loc.encode());
        }

        crate::entropy::score_moveset(config, &candidate.new_config, &occupied, ctx, &self.params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heuristic::DistanceTable;
    use crate::lane_index::LaneIndex;
    use crate::test_utils::{example_arch_json, loc};
    use bloqade_lanes_bytecode_core::arch::types::ArchSpec;

    #[test]
    fn entropy_scorer_positive_for_improvement() {
        let spec: ArchSpec = serde_json::from_str(example_arch_json()).unwrap();
        let index = LaneIndex::new(spec);
        let targets = [(0u32, loc(0, 5))];
        let target_enc: Vec<(u32, u64)> = targets.iter().map(|&(q, l)| (q, l.encode())).collect();
        let locs: Vec<u64> = target_enc.iter().map(|&(_, l)| l).collect();
        let table = DistanceTable::new(&locs, &index);
        let blocked = HashSet::new();
        let ctx = SearchContext {
            index: &index,
            dist_table: &table,
            blocked: &blocked,
            targets: &target_enc,
        };

        let from = Config::new([(0, loc(0, 0))]).unwrap();
        let to = Config::new([(0, loc(0, 1))]).unwrap();
        let candidate = MoveCandidate {
            move_set: crate::graph::MoveSet::from_encoded(vec![]),
            new_config: to,
        };

        let scorer = EntropyScorer::new(EntropyParams::default());
        let score = scorer.score(&candidate, &from, &ctx);
        assert!(
            score > 0.0,
            "moving closer to target should have positive score, got {score}"
        );
    }

    #[test]
    fn entropy_scorer_zero_for_no_movement() {
        let spec: ArchSpec = serde_json::from_str(example_arch_json()).unwrap();
        let index = LaneIndex::new(spec);
        let targets = [(0u32, loc(0, 5))];
        let target_enc: Vec<(u32, u64)> = targets.iter().map(|&(q, l)| (q, l.encode())).collect();
        let locs: Vec<u64> = target_enc.iter().map(|&(_, l)| l).collect();
        let table = DistanceTable::new(&locs, &index);
        let blocked = HashSet::new();
        let ctx = SearchContext {
            index: &index,
            dist_table: &table,
            blocked: &blocked,
            targets: &target_enc,
        };

        let config = Config::new([(0, loc(0, 0))]).unwrap();
        let candidate = MoveCandidate {
            move_set: crate::graph::MoveSet::from_encoded(vec![]),
            new_config: config.clone(),
        };

        let scorer = EntropyScorer::new(EntropyParams::default());
        let score = scorer.score(&candidate, &config, &ctx);
        assert_eq!(score, 0.0, "no movement should score 0");
    }
}
