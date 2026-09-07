//! Entropy-weighted move generator wrapping [`entropy::generate_candidates()`].

use crate::drivers::entropy::{EntropyParams, HeuristicTables};
use crate::primitives::config::Config;
use crate::primitives::context::{MoveCandidate, SearchContext, SearchState};
use crate::primitives::graph::NodeId;
use crate::traits::MoveGenerator;

/// Entropy-weighted move generator.
///
/// Reads per-node entropy from [`SearchState::entropy_map`] and delegates
/// to [`entropy::generate_candidates()`](crate::drivers::entropy::generate_candidates)
/// for the actual scoring logic.
///
/// Optionally borrows the solve's [`HeuristicTables`], the per-solve memo of
/// the occupancy-independent heuristic terms the entropy driver builds once
/// and reads at every expansion. Without them every call recomputes the
/// blended distances — bit-identical results, just slower — which is a
/// handicap the entropy driver does not have and a comparison against it
/// should not carry.
pub struct EntropyGenerator<'t> {
    params: EntropyParams,
    seed: u64,
    tables: Option<&'t HeuristicTables>,
    pinned_entropy: Option<u32>,
}

impl<'t> EntropyGenerator<'t> {
    /// A generator that computes its heuristic terms directly.
    pub fn new(params: EntropyParams, seed: u64) -> Self {
        Self {
            params,
            seed,
            tables: None,
            pinned_entropy: None,
        }
    }

    /// A generator reading the solve's prebuilt tables. The tables must have
    /// been built with `params.w_t` (debug-asserted at the read sites).
    pub fn with_tables(params: EntropyParams, seed: u64, tables: &'t HeuristicTables) -> Self {
        Self {
            params,
            seed,
            tables: Some(tables),
            pinned_entropy: None,
        }
    }

    /// Score every node at a fixed entropy instead of reading the per-node
    /// map.
    ///
    /// The entropy sets the blend between distance-to-target and mobility, so
    /// a schedule of these at rising entropy is a *local* widening: revisit a
    /// node under a more mobility-weighted blend, which is the escalation the
    /// entropy driver performs on a dead end and the branch-and-bound
    /// schedule otherwise has no cheap counterpart for. `generate_candidates`
    /// clamps at `params.e_max`, so pinning above it changes nothing.
    pub fn with_entropy(mut self, entropy: u32) -> Self {
        self.pinned_entropy = Some(entropy);
        self
    }
}

impl MoveGenerator for EntropyGenerator<'_> {
    fn generate(
        &self,
        config: &Config,
        node_id: NodeId,
        ctx: &SearchContext,
        state: &mut SearchState,
        out: &mut Vec<MoveCandidate>,
    ) {
        // A pinned entropy makes this generator one rung of a widening
        // ladder; otherwise read the node's own (default 1 if unseen).
        let entropy = self
            .pinned_entropy
            .unwrap_or_else(|| state.entropy_map.get(&node_id).map_or(1, |s| s.entropy));

        let raw = crate::drivers::entropy::generate_candidates(
            config,
            entropy,
            &self.params,
            ctx,
            self.seed,
            self.tables,
        );

        for entry in raw {
            out.push(MoveCandidate {
                move_set: entry.move_set,
                new_config: entry.new_config,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::context::SearchState;
    use crate::primitives::distance::DistanceTable;
    use crate::primitives::lane_index::LaneIndex;
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
            cz_pairs: None,
            capacity: None,
        };
        let mut state = SearchState::default();

        let generator = EntropyGenerator::new(EntropyParams::default(), 0);
        let config = crate::primitives::config::Config::new([(0, loc(0, 0))]).unwrap();
        let mut out = Vec::new();
        generator.generate(&config, NodeId(0), &ctx, &mut state, &mut out);
        assert!(!out.is_empty(), "should produce at least one candidate");
    }

    /// Prebuilt tables change nothing but the work: the two constructors
    /// emit identical candidate sequences.
    #[test]
    fn with_tables_matches_direct_computation() {
        let spec: ArchSpec = serde_json::from_str(example_arch_json()).unwrap();
        let index = LaneIndex::new(spec);
        let targets = [(0u32, loc(1, 5)), (1, loc(1, 6)), (2, loc(0, 7))];
        let target_enc: Vec<(u32, u64)> = targets.iter().map(|&(q, l)| (q, l.encode())).collect();
        let locs: Vec<u64> = target_enc.iter().map(|&(_, l)| l).collect();
        let table = DistanceTable::new(&locs, &index).with_time_distances(&index);
        let blocked = HashSet::new();
        let ctx = SearchContext {
            index: &index,
            dist_table: &table,
            blocked: &blocked,
            targets: &target_enc,
            cz_pairs: None,
            capacity: None,
        };
        let params = EntropyParams::default();
        let tables = HeuristicTables::build(&ctx, params.w_t, params.lookahead);
        let config = crate::primitives::config::Config::new([
            (0, loc(0, 0)),
            (1, loc(0, 1)),
            (2, loc(1, 2)),
        ])
        .unwrap();

        let run = |g: &EntropyGenerator<'_>| {
            let mut out = Vec::new();
            g.generate(
                &config,
                NodeId(0),
                &ctx,
                &mut SearchState::default(),
                &mut out,
            );
            out.into_iter()
                .map(|c| {
                    (
                        c.move_set.encoded_lanes().to_vec(),
                        c.new_config.as_entries().to_vec(),
                    )
                })
                .collect::<Vec<_>>()
        };
        let direct = run(&EntropyGenerator::new(params.clone(), 3));
        let memoized = run(&EntropyGenerator::with_tables(params, 3, &tables));
        assert!(!direct.is_empty());
        assert_eq!(direct, memoized);
    }
}
