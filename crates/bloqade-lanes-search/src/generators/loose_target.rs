//! Per-restart-seeded target generator for loose-goal entangling search.
//!
//! [`LooseTargetGenerator`] wraps a [`HeuristicGenerator`] with a target
//! assignment that's computed once per generator instance using its `seed`.
//!
//! When `solve_entangling` runs parallel restarts, each restart constructs
//! its own [`LooseTargetGenerator`] with a different seed. The seed feeds
//! [`greedy_assign_pairs`](entangling::greedy_assign_pairs)'s internal cost
//! perturbation, so each restart sees a slightly-different target
//! assignment. That diversity across restarts is what makes the parallel
//! `pick_best` strategy effective on hard problems.
//!
//! The targets are computed lazily on the first call to `generate` (because
//! the initial config is needed), then reused for every subsequent call
//! within the same restart. There is no dynamic recomputation — the
//! benefit comes from the per-restart seed, not from updating targets
//! during search.

use std::cell::{Cell, RefCell};
use std::sync::Arc;

use bloqade_lanes_bytecode_core::arch::types::ArchSpec;

use crate::Config;
use crate::context::{MoveCandidate, SearchContext, SearchState};
use crate::entangling;
use crate::generators::heuristic::HeuristicGenerator;
use crate::graph::NodeId;
use crate::heuristic::DistanceTable;
use crate::traits::MoveGenerator;

/// Move generator that wraps a [`HeuristicGenerator`] and overrides
/// `ctx.targets` with a per-instance seeded greedy assignment.
///
/// Targets are computed once on the first `generate` call using the
/// instance's `seed` and `congestion_weight`, then reused unchanged.
pub struct LooseTargetGenerator {
    inner: HeuristicGenerator,
    cz_pairs: Vec<(u32, u32)>,
    arch: Arc<ArchSpec>,
    dist_table: Arc<DistanceTable>,
    seed: u64,
    /// Forwarded to greedy_assign_pairs; 0.0 = standard min-sum assignment.
    congestion_weight: f64,
    cached_targets: RefCell<Vec<(u32, u64)>>,
    cache_initialized: Cell<bool>,
}

impl LooseTargetGenerator {
    /// Create a new seeded-target generator.
    ///
    /// # Arguments
    ///
    /// * `inner` — The underlying heuristic generator to delegate to.
    /// * `cz_pairs` — Required CZ pairs for target assignment.
    /// * `arch` — Architecture spec (shared, immutable).
    /// * `dist_table` — Distance table targeting entangling locations (shared).
    /// * `seed` — Seed for greedy-assignment perturbation. Different seeds
    ///   across parallel restarts give different target assignments.
    /// * `congestion_weight` — Forwarded to greedy assignment for spreading
    ///   targets across word pairs; 0.0 = standard min-sum.
    pub fn new(
        inner: HeuristicGenerator,
        cz_pairs: Vec<(u32, u32)>,
        arch: Arc<ArchSpec>,
        dist_table: Arc<DistanceTable>,
        seed: u64,
        congestion_weight: f64,
    ) -> Self {
        Self {
            inner,
            cz_pairs,
            arch,
            dist_table,
            seed,
            congestion_weight,
            cached_targets: RefCell::new(Vec::new()),
            cache_initialized: Cell::new(false),
        }
    }
}

impl MoveGenerator for LooseTargetGenerator {
    fn generate(
        &self,
        config: &Config,
        node_id: NodeId,
        ctx: &SearchContext,
        state: &mut SearchState,
        out: &mut Vec<MoveCandidate>,
    ) {
        // First call: compute initial targets using this instance's seed.
        // Subsequent calls reuse the same targets unchanged.
        if !self.cache_initialized.get() {
            let targets = entangling::greedy_assign_pairs(
                &self.cz_pairs,
                config,
                &self.arch,
                &self.dist_table,
                self.seed,
                None,
                0.0,
                self.congestion_weight,
            );
            *self.cached_targets.borrow_mut() = targets;
            self.cache_initialized.set(true);
        }

        // Run inner generator with the cached seeded targets.
        let targets = self.cached_targets.borrow();
        let loose_ctx = SearchContext {
            index: ctx.index,
            dist_table: ctx.dist_table,
            blocked: ctx.blocked,
            targets: &targets,
            cz_pairs: ctx.cz_pairs,
        };
        self.inner.generate(config, node_id, &loose_ctx, state, out);
    }

    fn deadlock_count(&self) -> u32 {
        self.inner.deadlock_count()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use bloqade_lanes_bytecode_core::arch::types::ArchSpec;

    use super::*;
    use crate::heuristic::DistanceTable;
    use crate::lane_index::LaneIndex;
    use crate::test_utils::{example_arch_json, loc};

    fn make_index() -> LaneIndex {
        let spec: ArchSpec = serde_json::from_str(example_arch_json()).unwrap();
        LaneIndex::new(spec)
    }

    fn make_arch() -> ArchSpec {
        serde_json::from_str(example_arch_json()).unwrap()
    }

    #[test]
    fn loose_target_produces_candidates() {
        let index = make_index();
        let arch = Arc::new(make_arch());
        let locs = entangling::all_entangling_locations(&arch);
        let dist_table = Arc::new(DistanceTable::new(&locs, &index));

        let cz_pairs = vec![(0u32, 1u32)];
        let inner = HeuristicGenerator::new();
        let generator = LooseTargetGenerator::new(
            inner,
            cz_pairs.clone(),
            arch.clone(),
            dist_table.clone(),
            0,
            0.0,
        );

        // q0 at word 0 site 0, q1 at word 0 site 5 — same column,
        // greedy assigns to entangling pair at site 5, q0 needs to move.
        let config = Config::new([(0, loc(0, 0)), (1, loc(0, 5))]).unwrap();

        let dummy_targets: Vec<(u32, u64)> = vec![];
        let blocked = HashSet::new();
        let ctx = SearchContext {
            index: &index,
            dist_table: &dist_table,
            blocked: &blocked,
            targets: &dummy_targets,
            cz_pairs: None,
        };

        let mut state = SearchState::default();
        let mut out = Vec::new();
        generator.generate(&config, NodeId(0), &ctx, &mut state, &mut out);

        assert!(!out.is_empty(), "should produce candidates");
    }

    #[test]
    fn loose_target_adapts_to_config() {
        let index = make_index();
        let arch = Arc::new(make_arch());
        let locs = entangling::all_entangling_locations(&arch);
        let dist_table = Arc::new(DistanceTable::new(&locs, &index));

        let cz_pairs = vec![(0u32, 1u32)];
        let inner1 = HeuristicGenerator::new();
        let generator1 = LooseTargetGenerator::new(
            inner1,
            cz_pairs.clone(),
            arch.clone(),
            dist_table.clone(),
            0,
            0.0,
        );
        let inner2 = HeuristicGenerator::new();
        let generator2 = LooseTargetGenerator::new(
            inner2,
            cz_pairs.clone(),
            arch.clone(),
            dist_table.clone(),
            0,
            0.0,
        );

        let dummy_targets: Vec<(u32, u64)> = vec![];
        let blocked = HashSet::new();
        let ctx = SearchContext {
            index: &index,
            dist_table: &dist_table,
            blocked: &blocked,
            targets: &dummy_targets,
            cz_pairs: None,
        };

        // Different configs in the same site column (0↔5), both needing moves.
        let config1 = Config::new([(0, loc(0, 0)), (1, loc(0, 5))]).unwrap();
        let config2 = Config::new([(0, loc(0, 5)), (1, loc(1, 0))]).unwrap();

        let mut out1 = Vec::new();
        let mut out2 = Vec::new();
        generator1.generate(
            &config1,
            NodeId(0),
            &ctx,
            &mut SearchState::default(),
            &mut out1,
        );
        generator2.generate(
            &config2,
            NodeId(0),
            &ctx,
            &mut SearchState::default(),
            &mut out2,
        );

        assert!(!out1.is_empty());
        assert!(!out2.is_empty());
    }
}
