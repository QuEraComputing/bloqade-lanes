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
use crate::generators::heuristic::HeuristicGenerator;
use crate::ops::entangling;
use crate::primitives::context::{MoveCandidate, SearchContext, SearchState};
use crate::primitives::distance::DistanceTable;
use crate::primitives::graph::NodeId;
use crate::primitives::lane_index::LaneIndex;
use crate::traits::MoveGenerator;

/// Move generator that wraps a [`HeuristicGenerator`] and overrides
/// `ctx.targets` with a per-instance seeded greedy assignment.
///
/// Targets are computed once on the first `generate` call using the
/// instance's `seed`, `congestion_weight`, and `occupancy_penalty`, then
/// reused unchanged. When `with_lookahead(...)` has supplied future layers
/// and a non-zero β, the per-restart computation uses
/// [`entangling::lookahead_assign_pairs`] instead of plain
/// [`entangling::greedy_assign_pairs`] so the search expansion sees the
/// same lookahead refinement that `solve_entangling` computes for
/// `ctx.targets`.
pub struct LooseTargetGenerator {
    inner: HeuristicGenerator,
    cz_pairs: Vec<(u32, u32)>,
    arch: Arc<ArchSpec>,
    index: Arc<LaneIndex>,
    dist_table: Arc<DistanceTable>,
    seed: u64,
    /// Forwarded to greedy_assign_pairs; 0.0 = standard min-sum assignment.
    congestion_weight: f64,
    /// Forwarded to greedy_assign_pairs; 0.0 = occupancy-blind assignment.
    occupancy_penalty: f64,
    /// Forwarded to assign_pairs_with_blockers; per-atom-moved cost added
    /// to each Hungarian cell to bias toward stay-in-place. 0.0 disables.
    move_penalty: f64,
    /// Future CZ layers for lookahead. Empty disables lookahead.
    future_layers: Vec<Vec<(u32, u32)>>,
    /// Lookahead blend weight (β). 0.0 disables lookahead.
    lookahead_beta: f64,
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
    /// * `index` — Lane index used by the iterative blocker detection
    ///   (Case-B egress check); shared, immutable.
    /// * `dist_table` — Distance table targeting entangling locations (shared).
    /// * `seed` — Seed for greedy-assignment perturbation. Different seeds
    ///   across parallel restarts give different target assignments.
    /// * `congestion_weight` — Forwarded to greedy assignment for spreading
    ///   targets across word pairs; 0.0 = standard min-sum.
    /// * `occupancy_penalty` — Forwarded to greedy assignment to bias
    ///   targets away from spectator-occupied entangling slot positions;
    ///   `0.0` = occupancy-blind. Must be finite and non-negative;
    ///   fractional values are supported.
    /// * `move_penalty` — Per-atom-moved cost added to each Hungarian
    ///   cell. Slightly biases the assignment toward stay-in-place
    ///   when the cost is otherwise close. `0.0` disables.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        inner: HeuristicGenerator,
        cz_pairs: Vec<(u32, u32)>,
        arch: Arc<ArchSpec>,
        index: Arc<LaneIndex>,
        dist_table: Arc<DistanceTable>,
        seed: u64,
        congestion_weight: f64,
        occupancy_penalty: f64,
        move_penalty: f64,
    ) -> Self {
        Self {
            inner,
            cz_pairs,
            arch,
            index,
            dist_table,
            seed,
            congestion_weight,
            occupancy_penalty,
            move_penalty,
            future_layers: Vec::new(),
            lookahead_beta: 0.0,
            cached_targets: RefCell::new(Vec::new()),
            cache_initialized: Cell::new(false),
        }
    }

    /// Enable lookahead-aware target assignment.
    ///
    /// When `future_layers` is non-empty and `beta > 0.0`, the per-restart
    /// target computation uses [`entangling::lookahead_assign_pairs`]
    /// instead of plain [`entangling::greedy_assign_pairs`]. The forward /
    /// backward sweep biases the current layer's assignment toward
    /// positions that are well-placed for the upcoming layers, weighted
    /// by `beta`.
    pub fn with_lookahead(mut self, future_layers: Vec<Vec<(u32, u32)>>, beta: f64) -> Self {
        self.future_layers = future_layers;
        self.lookahead_beta = beta;
        self
    }

    /// Construct with pre-computed targets, bypassing the lazy seed-based
    /// computation used by [`Self::new`]. Used by the receding-horizon
    /// orchestrator, which generates K candidate target assignments at each
    /// stage outside the generator and injects one per branch.
    ///
    /// `cz_pairs`, `arch`, `index`, and `dist_table` are still required for
    /// API parity with [`Self::new`] but are not consulted by `generate`
    /// when `cache_initialized = true` (the [`MoveGenerator::generate`]
    /// implementation uses `ctx.*` for those data, not the struct fields).
    pub fn from_targets(
        inner: HeuristicGenerator,
        targets: Vec<(u32, u64)>,
        cz_pairs: Vec<(u32, u32)>,
        arch: Arc<ArchSpec>,
        index: Arc<LaneIndex>,
        dist_table: Arc<DistanceTable>,
    ) -> Self {
        Self {
            inner,
            cz_pairs,
            arch,
            index,
            dist_table,
            seed: 0,
            congestion_weight: 0.0,
            occupancy_penalty: 0.0,
            move_penalty: 0.0,
            future_layers: Vec::new(),
            lookahead_beta: 0.0,
            cached_targets: RefCell::new(targets),
            cache_initialized: Cell::new(true),
        }
    }

    /// Internal accessor used by the receding-horizon orchestrator.
    pub(crate) fn index_ref(&self) -> &LaneIndex {
        &self.index
    }

    /// Internal accessor used by the receding-horizon orchestrator.
    pub(crate) fn dist_table_ref(&self) -> &DistanceTable {
        &self.dist_table
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
        // Subsequent calls reuse the same targets unchanged. Both branches
        // route through `assign_pairs_with_blockers` (directly or via
        // `lookahead_assign_pairs`), so the cached targets include any
        // spectator displacements needed to break Case-A or Case-B
        // deadlocks before the search starts.
        if !self.cache_initialized.get() {
            let targets = if self.future_layers.is_empty() || self.lookahead_beta == 0.0 {
                entangling::assign_pairs_with_blockers(
                    &self.cz_pairs,
                    config,
                    &self.arch,
                    &self.index,
                    &self.dist_table,
                    ctx.blocked,
                    self.seed,
                    None,
                    0.0,
                    self.congestion_weight,
                    self.occupancy_penalty,
                    self.move_penalty,
                    true,
                )
            } else {
                entangling::lookahead_assign_pairs(
                    &self.cz_pairs,
                    config,
                    &self.arch,
                    &self.index,
                    &self.dist_table,
                    ctx.blocked,
                    self.seed,
                    &self.future_layers,
                    self.lookahead_beta,
                    self.congestion_weight,
                    self.occupancy_penalty,
                    self.move_penalty,
                )
            };
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
    use crate::primitives::distance::DistanceTable;
    use crate::primitives::lane_index::LaneIndex;
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
        let index_arc = Arc::new(index.clone());
        let locs = entangling::all_entangling_locations(&arch);
        let dist_table = Arc::new(DistanceTable::new(&locs, &index));

        let cz_pairs = vec![(0u32, 1u32)];
        let inner = HeuristicGenerator::new();
        let generator = LooseTargetGenerator::new(
            inner,
            cz_pairs.clone(),
            arch.clone(),
            index_arc.clone(),
            dist_table.clone(),
            0,
            0.0,
            0.0,
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
        let index_arc = Arc::new(index.clone());
        let locs = entangling::all_entangling_locations(&arch);
        let dist_table = Arc::new(DistanceTable::new(&locs, &index));

        let cz_pairs = vec![(0u32, 1u32)];
        let inner1 = HeuristicGenerator::new();
        let generator1 = LooseTargetGenerator::new(
            inner1,
            cz_pairs.clone(),
            arch.clone(),
            index_arc.clone(),
            dist_table.clone(),
            0,
            0.0,
            0.0,
            0.0,
        );
        let inner2 = HeuristicGenerator::new();
        let generator2 = LooseTargetGenerator::new(
            inner2,
            cz_pairs.clone(),
            arch.clone(),
            index_arc.clone(),
            dist_table.clone(),
            0,
            0.0,
            0.0,
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
