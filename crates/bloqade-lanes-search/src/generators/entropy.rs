//! Entropy-weighted move generator wrapping [`entropy::generate_candidates()`].

use crate::config::Config;
use crate::context::{MoveCandidate, SearchContext, SearchState};
use crate::entropy::EntropyParams;
use crate::graph::NodeId;
use crate::traits::MoveGenerator;

/// Entropy-weighted move generator.
///
/// Reads per-node entropy from [`SearchState::entropy_map`] and delegates
/// to [`entropy::generate_candidates()`](crate::entropy::generate_candidates)
/// for the actual scoring logic.
pub struct EntropyGenerator {
    params: EntropyParams,
    seed: u64,
}

impl EntropyGenerator {
    pub fn new(params: EntropyParams, seed: u64) -> Self {
        Self { params, seed }
    }
}

impl MoveGenerator for EntropyGenerator {
    fn generate(
        &self,
        config: &Config,
        node_id: NodeId,
        ctx: &SearchContext,
        state: &mut SearchState,
        out: &mut Vec<MoveCandidate>,
    ) {
        // Read entropy for this node (default 1 if not yet in map).
        let entropy = state.entropy_map.get(&node_id).map_or(1, |s| s.entropy);

        let raw =
            crate::entropy::generate_candidates(config, entropy, &self.params, ctx, self.seed);

        for (move_set, new_config, _cost) in raw {
            out.push(MoveCandidate {
                move_set,
                new_config,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::SearchState;
    use crate::heuristic::DistanceTable;
    use crate::lane_index::LaneIndex;
    use crate::test_utils::{example_arch_json, loc};
    use bloqade_lanes_bytecode_core::arch::types::ArchSpec;
    use std::collections::HashSet;

    #[test]
    fn entropy_generator_produces_candidates() {
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
        let mut state = SearchState::default();

        let generator = EntropyGenerator::new(EntropyParams::default(), 0);
        let config = crate::config::Config::new([(0, loc(0, 0))]).unwrap();
        let mut out = Vec::new();
        generator.generate(&config, NodeId(0), &ctx, &mut state, &mut out);
        assert!(!out.is_empty(), "should produce at least one candidate");
    }
}
