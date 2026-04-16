//! [`MoveGenerator`] wrapper around [`HeuristicExpander`].

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;

use crate::astar::Expander;
use crate::config::Config;
use crate::context::{MoveCandidate, SearchContext, SearchState};
use crate::graph::NodeId;
use crate::heuristic::DistanceTable;
use crate::heuristic_expander::{DeadlockPolicy, HeuristicExpander};
use crate::lane_index::LaneIndex;
use crate::traits::MoveGenerator;

/// Thin wrapper that adapts a [`HeuristicExpander`] to the [`MoveGenerator`]
/// trait, converting the legacy `(MoveSet, Config, f64)` output into
/// [`MoveCandidate`] values.
pub struct HeuristicGenerator<'a> {
    expander: HeuristicExpander<'a>,
}

impl<'a> HeuristicGenerator<'a> {
    /// Create a new generator with the given parameters.
    pub fn new(
        index: &'a LaneIndex,
        blocked: impl IntoIterator<Item = LocationAddr>,
        targets: impl IntoIterator<Item = (u32, LocationAddr)>,
        dist_table: &'a DistanceTable,
        top_c: usize,
        max_movesets_per_group: usize,
    ) -> Self {
        Self {
            expander: HeuristicExpander::new(
                index,
                blocked,
                targets,
                dist_table,
                top_c,
                max_movesets_per_group,
            ),
        }
    }

    /// Set the deadlock escape policy.
    pub fn with_deadlock_policy(mut self, policy: DeadlockPolicy) -> Self {
        self.expander = self.expander.with_deadlock_policy(policy);
        self
    }

    /// Enable or disable 2-step lookahead scoring.
    pub fn with_lookahead(mut self, enabled: bool) -> Self {
        self.expander = self.expander.with_lookahead(enabled);
        self
    }

    /// Set the seed for score perturbation.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.expander = self.expander.with_seed(seed);
        self
    }
}

impl MoveGenerator for HeuristicGenerator<'_> {
    fn generate(
        &self,
        config: &Config,
        _node_id: NodeId,
        _ctx: &SearchContext,
        _state: &mut SearchState,
        out: &mut Vec<MoveCandidate>,
    ) {
        let mut raw: Vec<(crate::graph::MoveSet, Config, f64)> = Vec::new();
        self.expander.expand(config, &mut raw);
        out.extend(
            raw.into_iter()
                .map(|(move_set, new_config, _)| MoveCandidate {
                    move_set,
                    new_config,
                }),
        );
    }

    fn deadlock_count(&self) -> u32 {
        self.expander.deadlock_count()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use bloqade_lanes_bytecode_core::arch::types::ArchSpec;

    use super::*;
    use crate::astar::Expander;
    use crate::heuristic::DistanceTable;
    use crate::heuristic_expander::HeuristicExpander;
    use crate::test_utils::{example_arch_json, loc};

    fn make_index() -> LaneIndex {
        let spec: ArchSpec = serde_json::from_str(example_arch_json()).unwrap();
        LaneIndex::new(spec)
    }

    #[test]
    fn generator_matches_expander() {
        let index = make_index();
        let targets_raw = [(0u32, loc(0, 5))];
        let target_locs: Vec<u64> = targets_raw.iter().map(|(_, l)| l.encode()).collect();
        let dist_table = DistanceTable::new(&target_locs, &index);
        let config = Config::new([(0, loc(0, 0))]).unwrap();

        // Old expander path.
        let expander = HeuristicExpander::new(
            &index,
            std::iter::empty(),
            targets_raw.iter().copied(),
            &dist_table,
            3,
            5,
        );
        let mut expander_out = Vec::new();
        expander.expand(&config, &mut expander_out);

        // New generator path.
        let generator = HeuristicGenerator::new(
            &index,
            std::iter::empty(),
            targets_raw.iter().copied(),
            &dist_table,
            3,
            5,
        );
        let ctx = SearchContext {
            index: &index,
            dist_table: &dist_table,
            blocked: &HashSet::new(),
            targets: &targets_raw
                .iter()
                .map(|(q, l)| (*q, l.encode()))
                .collect::<Vec<_>>(),
        };
        let mut state = SearchState::default();
        let mut gen_out = Vec::new();
        generator.generate(&config, NodeId(0), &ctx, &mut state, &mut gen_out);

        // Same number of candidates.
        assert_eq!(
            expander_out.len(),
            gen_out.len(),
            "candidate counts must match"
        );

        // Same move sets (order preserved).
        let expander_sets: Vec<_> = expander_out.iter().map(|(ms, _, _)| ms.clone()).collect();
        let gen_sets: Vec<_> = gen_out.iter().map(|c| c.move_set.clone()).collect();
        assert_eq!(expander_sets, gen_sets, "move sets must match");
    }
}
