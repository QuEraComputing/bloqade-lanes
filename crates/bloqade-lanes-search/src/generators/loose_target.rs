//! Dynamic-target move generator for loose-goal search.
//!
//! [`LooseTargetGenerator`] wraps a [`HeuristicGenerator`], recomputing
//! per-qubit targets on every expansion based on the current configuration.
//! This prevents the "premature sleep" problem where a qubit stops being
//! guided because it accidentally landed on its statically-assigned target
//! before its CZ partner arrived.

use std::cell::{Cell, RefCell};
use std::sync::Arc;

use bloqade_lanes_bytecode_core::arch::types::ArchSpec;

use crate::config::Config;
use crate::context::{MoveCandidate, SearchContext, SearchState};
use crate::entangling;
use crate::generators::heuristic::HeuristicGenerator;
use crate::graph::NodeId;
use crate::heuristic::DistanceTable;
use crate::traits::MoveGenerator;

/// Move generator that recomputes target assignments on deadlock and/or
/// on a periodic interval, applying new targets only if they improve cost.
///
/// Wraps a [`HeuristicGenerator`] and overrides `ctx.targets` with a greedy
/// assignment. Recomputation is triggered by:
/// - **Deadlock**: always checked — when the inner generator finds no
///   improving moves, targets are recomputed for the next expansion.
/// - **Interval** (optional): if `recompute_interval > 0`, also recompute
///   every N expansions.
///
/// New targets are only applied if their total assignment cost is strictly
/// lower than the cached targets. This prevents oscillation.
///
/// `recompute_interval = 0`: deadlock-triggered only (recommended).
/// `recompute_interval > 0`: deadlock + periodic.
pub struct LooseTargetGenerator {
    inner: HeuristicGenerator,
    cz_pairs: Vec<(u32, u32)>,
    arch: Arc<ArchSpec>,
    dist_table: Arc<DistanceTable>,
    seed: u64,
    /// 0 = deadlock-only, N > 0 = also recompute every N expansions.
    recompute_interval: u32,
    cached_targets: RefCell<Vec<(u32, u64)>>,
    cached_cost: Cell<u32>,
    cache_initialized: Cell<bool>,
    expansion_count: Cell<u32>,
}

impl LooseTargetGenerator {
    /// Create a new loose-target generator.
    ///
    /// # Arguments
    ///
    /// * `inner` — The underlying heuristic generator to delegate to.
    /// * `cz_pairs` — Required CZ pairs for target recomputation.
    /// * `arch` — Architecture spec (shared, immutable).
    /// * `dist_table` — Distance table targeting entangling locations (shared).
    /// * `seed` — Seed for greedy assignment perturbation (0 = no perturbation).
    /// * `recompute_interval` — 0 = deadlock-only, N > 0 = also every N expansions.
    pub fn new(
        inner: HeuristicGenerator,
        cz_pairs: Vec<(u32, u32)>,
        arch: Arc<ArchSpec>,
        dist_table: Arc<DistanceTable>,
        seed: u64,
        recompute_interval: u32,
    ) -> Self {
        Self {
            inner,
            cz_pairs,
            arch,
            dist_table,
            seed,
            recompute_interval,
            cached_targets: RefCell::new(Vec::new()),
            cached_cost: Cell::new(u32::MAX),
            cache_initialized: Cell::new(false),
            expansion_count: Cell::new(0),
        }
    }

    /// Recompute greedy assignment; apply only if strictly cheaper.
    fn maybe_recompute(&self, config: &Config) {
        let new_targets = entangling::greedy_assign_pairs(
            &self.cz_pairs,
            config,
            &self.arch,
            &self.dist_table,
            self.seed,
            None,
            0.0,
        );
        let new_cost = entangling::assignment_cost(config, &new_targets, &self.dist_table);
        if new_cost < self.cached_cost.get() {
            self.cached_cost.set(new_cost);
            *self.cached_targets.borrow_mut() = new_targets;
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
        // First call: always compute initial targets.
        if !self.cache_initialized.get() {
            let targets = entangling::greedy_assign_pairs(
                &self.cz_pairs,
                config,
                &self.arch,
                &self.dist_table,
                self.seed,
                None,
                0.0,
            );
            let cost = entangling::assignment_cost(config, &targets, &self.dist_table);
            self.cached_cost.set(cost);
            *self.cached_targets.borrow_mut() = targets;
            self.cache_initialized.set(true);
        }

        // Interval-based recomputation (if enabled).
        let count = self.expansion_count.get();
        if self.recompute_interval > 0 && count > 0 && count.is_multiple_of(self.recompute_interval)
        {
            self.maybe_recompute(config);
        }

        // Run inner generator with cached targets.
        let dl_before = self.inner.deadlock_count();
        {
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
        let dl_after = self.inner.deadlock_count();

        // Deadlock-triggered recomputation (for next expansion).
        if dl_after > dl_before {
            self.maybe_recompute(config);
        }

        self.expansion_count.set(count.wrapping_add(1));
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
            1,
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
            1,
        );
        let inner2 = HeuristicGenerator::new();
        let generator2 = LooseTargetGenerator::new(
            inner2,
            cz_pairs.clone(),
            arch.clone(),
            dist_table.clone(),
            0,
            1,
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
