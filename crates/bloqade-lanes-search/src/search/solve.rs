//! High-level solver that ties together the lane index, expander, heuristic,
//! and A* search into a single reusable object.
//!
//! [`MoveSolver`] is constructed once per architecture (parsing JSON and
//! building indexes) and can then solve multiple placement problems.
//!
//! Slice-1 of the §6 type split has factored the result types, options
//! bundles, and restart orchestration into [`super::result`],
//! [`super::options`], and [`super::restarts`]; the next slices will
//! extract `MoveSearch` / `TargetSolver` and the `CzPlacement` peers.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;
use bloqade_lanes_bytecode_core::arch::types::ArchSpec;
use rayon::prelude::*;

use crate::generators::HeuristicGenerator;
use crate::generators::heuristic::DeadlockPolicy;
use crate::goals::AllAtTarget;
use crate::ops::entangling::{self, WordPairDistances};
use crate::ops::entangling::{LOOKAHEAD_BETA, MOVE_PENALTY};
use crate::placement::nohome::{self, NoHomeOptions};
use crate::placement::target_generator::{TargetContext, TargetGenerator, validate_candidate};
use crate::primitives::config::{Config, ConfigError};
use crate::primitives::context::SearchContext;
use crate::primitives::distance::{DistanceTable, HopDistanceHeuristic};
use crate::primitives::lane_index::LaneIndex;
use crate::search::restarts::{pick_best, run_with_components};

// Re-export the moved types under the original `solve::*` path so existing
// `use crate::search::solve::SolveStatus` imports keep resolving until the
// consumer migration completes.
pub use crate::search::options::{
    EntanglingOptions, EntropyOptions, InnerStrategy, SolveOptions, Strategy,
};
pub use crate::search::result::{SolveResult, SolveStatus};

/// Cached architecture-dependent data for [`MoveSolver::solve_entangling`].
///
/// All fields depend only on the architecture (lane index), not on per-call
/// data (initial positions, CZ pairs). Built once on first
/// `solve_entangling` call and reused for all subsequent calls.
pub(crate) struct EntanglingCache {
    pub ent_set: HashSet<(u64, u64)>,
    pub partner_map: HashMap<u64, u64>,
    pub dist_table: Arc<DistanceTable>,
    pub wpd: WordPairDistances,
}

/// Cached architecture-dependent data for [`MoveSolver::solve_nohome`].
///
/// All fields depend only on the architecture (lane index). Built once on
/// first `solve_nohome` call and reused for all subsequent calls.
pub(crate) struct NoHomeCache {
    pub home_locs: Vec<u64>,
    pub home_set: HashSet<u64>,
    pub dist_table: Arc<DistanceTable>,
}

/// Reusable move synthesis solver.
///
/// Constructed once per architecture — parses the arch spec JSON and builds
/// the lane index. Then [`solve`](MoveSolver::solve) can be called multiple
/// times with different initial/target placements.
///
/// Works for both physical and logical architectures (same interface,
/// different arch spec JSON).
pub struct MoveSolver {
    index: LaneIndex,
    entangling_cache: OnceLock<EntanglingCache>,
    nohome_cache: OnceLock<NoHomeCache>,
}

impl std::fmt::Debug for MoveSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MoveSolver")
            .field("index", &self.index)
            .finish_non_exhaustive()
    }
}

impl MoveSolver {
    /// Construct from an [`ArchSpec`] JSON string.
    ///
    /// Parses the JSON, builds the lane index (precomputes all lane lookups,
    /// endpoints, and positions).
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        let arch_spec = serde_json::from_str(json)?;
        Ok(Self {
            index: LaneIndex::new(arch_spec),
            entangling_cache: OnceLock::new(),
            nohome_cache: OnceLock::new(),
        })
    }

    /// Construct from an existing [`LaneIndex`].
    pub fn from_index(index: LaneIndex) -> Self {
        Self {
            index,
            entangling_cache: OnceLock::new(),
            nohome_cache: OnceLock::new(),
        }
    }

    /// Construct from a borrowed [`ArchSpec`]. Avoids the JSON round-trip
    /// that callers holding a wrapper around an `ArchSpec` would otherwise
    /// pay to materialize an owned spec.
    pub fn from_arch_spec(arch_spec: &ArchSpec) -> Self {
        Self::from_index(LaneIndex::from_arch_spec(arch_spec))
    }

    /// Access the underlying lane index.
    pub fn index(&self) -> &LaneIndex {
        &self.index
    }

    /// Get or build the cached entangling precomputation.
    fn entangling_cache(&self) -> &EntanglingCache {
        self.entangling_cache.get_or_init(|| {
            let arch = self.index.arch_spec();
            let word_pairs = entangling::enumerate_word_pairs(arch);
            let ent_locs = entangling::all_entangling_locations(arch);
            let ent_set = entangling::build_entangling_set(arch);
            let partner_map = entangling::build_partner_map(&ent_set);
            // Always include time distances — callers with w_t=0.0 just
            // ignore them (hop-count fields are separate).
            let dist_table = Arc::new(
                DistanceTable::new(&ent_locs, &self.index).with_time_distances(&self.index),
            );
            let wpd =
                entangling::WordPairDistances::from_dist_table(&word_pairs, arch, &dist_table);
            EntanglingCache {
                ent_set,
                partner_map,
                dist_table,
                wpd,
            }
        })
    }

    /// Solve a move synthesis problem.
    ///
    /// Finds the minimum-cost sequence of parallel move steps to move
    /// qubits from `initial` placement to `target` placement, avoiding
    /// `blocked` locations.
    ///
    /// # Arguments
    ///
    /// * `initial` — Starting qubit positions: `(qubit_id, location)` pairs.
    /// * `target` — Desired qubit positions: `(qubit_id, location)` pairs.
    /// * `blocked` — Locations occupied by external atoms (immovable obstacles).
    /// * `max_expansions` — Optional limit on node expansions.
    /// * `opts` — Search-tuning parameters (strategy, weight, restarts, etc.).
    ///
    /// # Returns
    ///
    /// A [`SolveResult`] whose [`status`](SolveResult::status) indicates
    /// whether a solution was found, the problem is unsolvable, or the
    /// expansion budget was exceeded.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] if `initial` contains duplicate qubit IDs.
    pub fn solve(
        &self,
        initial: impl IntoIterator<Item = (u32, LocationAddr)>,
        target: impl IntoIterator<Item = (u32, LocationAddr)>,
        blocked: impl IntoIterator<Item = LocationAddr>,
        max_expansions: Option<u32>,
        opts: &SolveOptions,
        entropy_opts: Option<&EntropyOptions>,
    ) -> Result<SolveResult, ConfigError> {
        let root = Config::new(initial)?;
        let target_pairs: Vec<(u32, LocationAddr)> = target.into_iter().collect();
        let blocked_locs: Vec<LocationAddr> = blocked.into_iter().collect();

        // Build goal predicate.
        let target_encoded: Vec<(u32, u64)> =
            target_pairs.iter().map(|&(q, l)| (q, l.encode())).collect();

        // Build distance table and heuristic (shared across restarts).
        let target_locs: Vec<u64> = target_encoded.iter().map(|&(_, l)| l).collect();
        let w_t = entropy_opts.map_or(EntropyOptions::default().w_t, |e| e.w_t);
        let dist_table = if w_t > 0.0 {
            DistanceTable::new(&target_locs, &self.index).with_time_distances(&self.index)
        } else {
            DistanceTable::new(&target_locs, &self.index)
        };
        let heuristic = HopDistanceHeuristic::new(target_pairs.iter().copied(), &dist_table);
        let h_max = |config: &Config| -> f64 { heuristic.estimate_max(config) };
        let h_sum = |config: &Config| -> f64 { heuristic.estimate_sum(config) };

        let goal_obj = AllAtTarget::new(&target_encoded);
        let blocked_encoded: std::collections::HashSet<u64> =
            blocked_locs.iter().map(|l| l.encode()).collect();
        let ctx = SearchContext {
            index: &self.index,
            dist_table: &dist_table,
            blocked: &blocked_encoded,
            targets: &target_encoded,
            cz_pairs: None,
        };

        let lookahead = opts.lookahead;
        let top_c = opts.top_c;
        let make_generator = |seed: u64, policy: DeadlockPolicy| {
            HeuristicGenerator::configured(seed, policy, lookahead, top_c)
        };

        Ok(run_with_components(
            root,
            &goal_obj,
            make_generator,
            h_max,
            h_sum,
            &ctx,
            max_expansions,
            opts,
            entropy_opts,
        ))
    }

    /// Solve a loose-goal entangling placement + routing problem.
    ///
    /// Instead of fixed target locations, the solver receives CZ pair
    /// constraints and simultaneously discovers both the entangling
    /// placement and the routing. The goal is satisfied when every
    /// CZ pair occupies a valid entangling position (same zone,
    /// entangling word pair, same site).
    ///
    /// # Arguments
    ///
    /// * `initial` — Starting qubit positions: `(qubit_id, location)` pairs.
    /// * `cz_pairs` — Required CZ pairs: `(qubit_a, qubit_b)` that must
    ///   end up at entangling positions.
    /// * `blocked` — Locations occupied by external atoms (immovable obstacles).
    /// * `max_expansions` — Optional limit on node expansions.
    /// * `opts` — Search-tuning parameters (strategy, weight, restarts, etc.).
    ///
    /// # Returns
    ///
    /// A [`SolveResult`] whose [`goal_config`](SolveResult::goal_config)
    /// contains the discovered entangling placement.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] if `initial` contains duplicate qubit IDs.
    #[allow(clippy::too_many_arguments)]
    pub fn solve_entangling(
        &self,
        initial: impl IntoIterator<Item = (u32, LocationAddr)>,
        cz_pairs: &[(u32, u32)],
        blocked: impl IntoIterator<Item = LocationAddr>,
        max_expansions: Option<u32>,
        opts: &SolveOptions,
        ent_opts: &EntanglingOptions,
        future_cz_layers: &[Vec<(u32, u32)>],
    ) -> Result<SolveResult, ConfigError> {
        use crate::goals::EntanglingConstraintGoal;
        use crate::primitives::distance::PairDistanceHeuristic;

        let root = Config::new(initial)?;
        let blocked_locs: Vec<LocationAddr> = blocked.into_iter().collect();
        let arch = self.index.arch_spec();

        // Reuse cached architecture-dependent data (built on first call).
        let cache = self.entangling_cache();
        let dist_table = cache.dist_table.clone(); // Arc clone (cheap)

        // Per-call: heuristic, goal, greedy assignment.
        let heuristic = PairDistanceHeuristic::new(cz_pairs, &cache.wpd);
        let h_max = |config: &Config| -> f64 { heuristic.estimate_max(config) };
        let h_sum = |config: &Config| -> f64 { heuristic.estimate_sum(config) };

        let goal = EntanglingConstraintGoal::new(cz_pairs, cache.ent_set.clone());

        // Build the blocked set up-front so the Hungarian and the search
        // share the same view of immobile atoms.
        let blocked_encoded: HashSet<u64> = blocked_locs.iter().map(|l| l.encode()).collect();

        let clipped_future = ent_opts.clipped_future_layers(future_cz_layers);

        // Use lookahead assignment if (clipped) future layers are available.
        // Both branches go through `assign_pairs_with_blockers` (directly
        // or via `lookahead_assign_pairs`), so the produced target list
        // includes any spectator displacements needed to break Case-A,
        // Case-B, or accidental-CZ deadlocks.
        let greedy_targets = if !clipped_future.is_empty() {
            entangling::lookahead_assign_pairs(
                cz_pairs,
                &root,
                arch,
                &self.index,
                &dist_table,
                &blocked_encoded,
                0,
                clipped_future,
                LOOKAHEAD_BETA,
                ent_opts.congestion_weight,
                ent_opts.occupancy_penalty,
                MOVE_PENALTY,
            )
        } else {
            entangling::assign_pairs_with_blockers(
                cz_pairs,
                &root,
                arch,
                &self.index,
                &dist_table,
                &blocked_encoded,
                0,
                None,
                0.0,
                ent_opts.congestion_weight,
                ent_opts.occupancy_penalty,
                MOVE_PENALTY,
                true,
            )
        };

        let ctx = SearchContext {
            index: &self.index,
            dist_table: &dist_table,
            blocked: &blocked_encoded,
            targets: &greedy_targets,
            cz_pairs: Some(cz_pairs),
        };

        let lookahead = opts.lookahead;
        let top_c = opts.top_c.unwrap_or(3);
        let upgraded_opts = opts.upgraded_for_entangling();
        let opts = &upgraded_opts;

        let mut result = {
            use crate::generators::LooseTargetGenerator;

            let arch_arc = Arc::new(arch.clone());
            let index_arc: Arc<LaneIndex> = Arc::new(self.index.clone());
            let dt_arc = dist_table.clone(); // Arc clone (cheap)
            let congestion_weight = ent_opts.congestion_weight;
            let occupancy_penalty = ent_opts.occupancy_penalty;

            let cz_pairs_owned: Vec<(u32, u32)> = cz_pairs.to_vec();
            // Clone the (clipped) future layers once so the per-restart
            // generator closure can re-use them as its lookahead inputs.
            let future_layers_owned: Vec<Vec<(u32, u32)>> = clipped_future.to_vec();
            let make_generator = move |seed: u64, policy: DeadlockPolicy| {
                let inner = HeuristicGenerator::configured(seed, policy, lookahead, Some(top_c));
                let mut generator = LooseTargetGenerator::new(
                    inner,
                    cz_pairs_owned.clone(),
                    arch_arc.clone(),
                    index_arc.clone(),
                    dt_arc.clone(),
                    seed,
                    congestion_weight,
                    occupancy_penalty,
                    MOVE_PENALTY,
                );
                if !future_layers_owned.is_empty() {
                    generator =
                        generator.with_lookahead(future_layers_owned.clone(), LOOKAHEAD_BETA);
                }
                generator
            };

            run_with_components(
                root,
                &goal,
                make_generator,
                h_max,
                h_sum,
                &ctx,
                max_expansions,
                opts,
                None,
            )
        };

        // Post-solve cleanup: move spectator qubits out of accidental CZ positions.
        if result.status == SolveStatus::Solved {
            let cz_qubit_set: HashSet<u32> = cz_pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
            let accidental = entangling::find_accidental_cz(
                &result.goal_config,
                &cz_qubit_set,
                &cache.partner_map,
            );

            if !accidental.is_empty() {
                let mut cleanup_targets: Vec<(u32, LocationAddr)> =
                    result.goal_config.iter().collect();

                for &(qid, move_loc) in &accidental {
                    for &lane in self.index.outgoing_lanes(move_loc) {
                        if let Some((_, dst)) = self.index.endpoints(&lane) {
                            if result.goal_config.is_occupied(dst) {
                                continue;
                            }
                            let safe = arch.get_cz_partner(&dst).is_none_or(|p| {
                                !result.goal_config.is_occupied(p)
                                    || cz_qubit_set.contains(
                                        &result.goal_config.qubit_at(p).unwrap_or(u32::MAX),
                                    )
                            });
                            if safe {
                                if let Some(entry) =
                                    cleanup_targets.iter_mut().find(|(q, _)| *q == qid)
                                {
                                    entry.1 = dst;
                                }
                                break;
                            }
                        }
                    }
                }

                let cleanup_result = self.solve(
                    result.goal_config.iter(),
                    cleanup_targets,
                    blocked_locs.iter().copied(),
                    max_expansions,
                    opts,
                    None,
                );

                if let Ok(cleanup) = cleanup_result
                    && cleanup.status == SolveStatus::Solved
                {
                    result.move_layers.extend(cleanup.move_layers);
                    result.goal_config = cleanup.goal_config;
                    result.cost += cleanup.cost;
                    result.nodes_expanded += cleanup.nodes_expanded;
                }
            }
        }

        Ok(result)
    }

    /// Receding-horizon (MPC-style) loose-goal entangling solve.
    ///
    /// Instead of committing to one Hungarian assignment up-front like
    /// [`solve_entangling`](MoveSolver::solve_entangling), this entry runs a
    /// sequence of K-candidate-rollout stages. At each stage:
    ///   1. Generate K diverse Hungarian candidates from the current state.
    ///   2. Roll out each for `rollout_horizon` move layers via the existing
    ///      IDS infrastructure.
    ///   3. Pick the best branch by stratified score (tier-0: goal reached
    ///      mid-rollout; tier-1: completed full horizon; tier-2: dropped).
    ///   4. Commit the winning branch's path (full for tier-0, `commit_depth`
    ///      layers for tier-1) and re-plan from the new state.
    ///
    /// Restarts wrap the entire trajectory: `opts.restarts` parallel calls,
    /// each running its own independent receding-horizon trajectory with a
    /// distinct seed; `pick_best` returns the lowest-layer-count winner.
    ///
    /// Positioned as a tool for high-occupancy regimes where the baseline
    /// loose-goal under-uses parallelism. See
    /// `docs/superpowers/plans/2026-05-11-receding-horizon-loose-goal-design.md`.
    #[allow(clippy::too_many_arguments)]
    pub fn solve_entangling_rh(
        &self,
        initial: impl IntoIterator<Item = (u32, LocationAddr)>,
        cz_pairs: &[(u32, u32)],
        blocked: impl IntoIterator<Item = LocationAddr>,
        max_expansions: Option<u32>,
        opts: &SolveOptions,
        ent_opts: &EntanglingOptions,
        rh_opts: &crate::placement::receding_horizon::RecedingHorizonOptions,
        future_cz_layers: &[Vec<(u32, u32)>],
    ) -> Result<SolveResult, ConfigError> {
        use crate::goals::EntanglingConstraintGoal;
        use crate::primitives::distance::PairDistanceHeuristic;

        let root = Config::new(initial)?;
        let blocked_locs: Vec<LocationAddr> = blocked.into_iter().collect();
        let arch = self.index.arch_spec();

        let cache = self.entangling_cache();
        let dist_table = cache.dist_table.clone();

        let heuristic = PairDistanceHeuristic::new(cz_pairs, &cache.wpd);
        let goal = EntanglingConstraintGoal::new(cz_pairs, cache.ent_set.clone());
        let blocked_encoded: HashSet<u64> = blocked_locs.iter().map(|l| l.encode()).collect();

        let clipped_future = ent_opts.clipped_future_layers(future_cz_layers);
        let future_owned: Vec<Vec<(u32, u32)>> = clipped_future.to_vec();

        let upgraded_opts = opts.upgraded_for_entangling();
        let opts = &upgraded_opts;

        let arch_arc = Arc::new(arch.clone());
        let index_arc: Arc<LaneIndex> = Arc::new(self.index.clone());

        let restarts = opts.restarts.max(1);
        let cz_pairs_owned: Vec<(u32, u32)> = cz_pairs.to_vec();

        // Fallback closure: when receding-horizon gives up (all branches
        // drop at horizon=1, or candidate generation returns empty), fall
        // back to a standard `solve_entangling` from the current state. Use
        // `restarts = 1` for the inner call to avoid nested rayon
        // parallelism that could oversubscribe threads. Captures `&self` to
        // call back into the solver.
        let single_opts = SolveOptions {
            restarts: 1,
            ..opts.clone()
        };
        let make_fallback = |state: &Config| -> SolveResult {
            let initial: Vec<(u32, LocationAddr)> = state.iter().collect();
            self.solve_entangling(
                initial,
                cz_pairs,
                blocked_locs.iter().copied(),
                max_expansions,
                &single_opts,
                ent_opts,
                future_cz_layers,
            )
            .unwrap_or_else(|_| SolveResult::unsolvable(state.clone()))
        };

        // Run restarts in parallel.
        let results: Vec<SolveResult> = if restarts <= 1 {
            vec![
                crate::placement::receding_horizon::solve_entangling_rh_single(
                    root.clone(),
                    &cz_pairs_owned,
                    blocked_encoded.clone(),
                    arch_arc.clone(),
                    index_arc.clone(),
                    dist_table.clone(),
                    &goal,
                    &heuristic,
                    opts,
                    ent_opts,
                    rh_opts,
                    &future_owned,
                    max_expansions,
                    /*restart_seed*/ 0,
                    make_fallback,
                ),
            ]
        } else {
            (0..restarts)
                .into_par_iter()
                .map(|i| {
                    crate::placement::receding_horizon::solve_entangling_rh_single(
                        root.clone(),
                        &cz_pairs_owned,
                        blocked_encoded.clone(),
                        arch_arc.clone(),
                        index_arc.clone(),
                        dist_table.clone(),
                        &goal,
                        &heuristic,
                        opts,
                        ent_opts,
                        rh_opts,
                        &future_owned,
                        max_expansions,
                        /*restart_seed*/ (i + 1) as u64,
                        make_fallback,
                    )
                })
                .collect()
        };

        Ok(pick_best(results))
    }

    /// Get or build the cached no-home precomputation.
    fn nohome_cache(&self) -> &NoHomeCache {
        self.nohome_cache.get_or_init(|| {
            let arch = self.index.arch_spec();
            let home_locs = nohome::home_sites(arch);
            let home_set: HashSet<u64> = home_locs.iter().copied().collect();
            let dist_table = Arc::new(
                DistanceTable::new(&home_locs, &self.index).with_time_distances(&self.index),
            );
            NoHomeCache {
                home_locs,
                home_set,
                dist_table,
            }
        })
    }

    /// Two-phase no-home placement: return assignment + entangling routing.
    ///
    /// Phase 1 generates up to `1 + nohome_opts.top_bus_signatures` candidate
    /// home layouts (Hungarian with lane-signature reward variants), routes
    /// each via [`solve`](MoveSolver::solve), and keeps the candidate whose
    /// return-routing produces the fewest move layers (the hop-count
    /// substitute for routing time).
    ///
    /// Phase 2 picks one CZ-staging target per pair via a deterministic
    /// per-pair rule (cross-word pairs prefer moving the qubit at home;
    /// same-word pairs default to moving the control), then routes once.
    ///
    /// # Arguments
    ///
    /// * `initial` — Starting qubit positions (typically post-CZ).
    /// * `cz_pairs` — Required CZ pairs for the next layer.
    /// * `blocked` — Immovable obstacle locations.
    /// * `max_expansions` — Budget for node expansions (shared across phases).
    /// * `opts` — Search-tuning parameters for the routing phases.
    /// * `nohome_opts` — Tuning parameters for the return assignment.
    /// * `future_cz_layers` — Future CZ layers; gamma-decayed partner
    ///   distances bias the Phase-1 home assignment toward future-friendly
    ///   layouts via `nohome_opts.lambda_lookahead`.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] if `initial` contains duplicate qubit IDs.
    #[allow(clippy::too_many_arguments)]
    pub fn solve_nohome(
        &self,
        initial: impl IntoIterator<Item = (u32, LocationAddr)>,
        cz_pairs: &[(u32, u32)],
        blocked: impl IntoIterator<Item = LocationAddr>,
        max_expansions: Option<u32>,
        opts: &SolveOptions,
        nohome_opts: &NoHomeOptions,
        future_cz_layers: &[Vec<(u32, u32)>],
    ) -> Result<SolveResult, ConfigError> {
        // Both phases route under entangling-style contention.
        let upgraded_opts = opts.upgraded_for_entangling();
        let opts = &upgraded_opts;

        let root = Config::new(initial)?;
        let blocked_locs: Vec<LocationAddr> = blocked.into_iter().collect();
        let nh_cache = self.nohome_cache();
        let arch = self.index.arch_spec();

        let has_returners = root
            .iter()
            .any(|(_, loc)| !nh_cache.home_set.contains(&loc.encode()));

        let blocked_set: HashSet<u64> = blocked_locs.iter().map(|l| l.encode()).collect();

        // Helper: resolve fixed CZ-staging targets for Phase 2 from a config.
        // Mirrors Python LogicalPlacementMethods._pick_move (minus the
        // move_count comparison, which isn't tracked at the solver level):
        // for each CZ pair, pick which qubit moves to the CZ partner of the
        // *other* qubit's position. Cross-word pairs prefer moving the home
        // qubit; same-word pairs default to moving the control. (The
        // Python's `_pick_move_by_conflict` tiebreak collapses to a no-op
        // for same-word pairs because both candidate destinations land in
        // the same partner word — porting it would need real lane-conflict
        // analysis, which is out of scope here.)
        let resolve_cz_targets = |from: &Config| -> Vec<(u32, LocationAddr)> {
            let mut chosen: Vec<(u32, LocationAddr)> = Vec::with_capacity(cz_pairs.len());
            for &(c, t) in cz_pairs {
                let Some(c_addr) = from.location_of(c) else {
                    continue;
                };
                let Some(t_addr) = from.location_of(t) else {
                    continue;
                };
                let Some(c_dst) = arch.get_cz_partner(&t_addr) else {
                    continue;
                };
                let Some(t_dst) = arch.get_cz_partner(&c_addr) else {
                    continue;
                };

                let move_c = (c, c_dst);
                let move_t = (t, t_dst);

                let pick = if c_addr.word_id == t_addr.word_id {
                    // Same word — pick the control deterministically.
                    move_c
                } else if arch.is_home_position(&t_addr) {
                    // Cross-word — prefer moving the qubit currently at home.
                    move_t
                } else {
                    move_c
                };
                chosen.push(pick);
            }
            let chosen_map: HashMap<u32, LocationAddr> = chosen.iter().copied().collect();
            from.iter()
                .map(|(qid, loc)| (qid, chosen_map.get(&qid).copied().unwrap_or(loc)))
                .collect()
        };

        if !has_returners {
            // Skip the return phase — go directly to fixed-target entangling.
            let cz_targets = resolve_cz_targets(&root);
            return self.solve(
                root.iter(),
                cz_targets,
                blocked_locs.iter().copied(),
                max_expansions,
                opts,
                None,
            );
        }

        let occupied_set: HashSet<u64> = root.iter().map(|(_, loc)| loc.encode()).collect();
        let holes: Vec<u64> = nh_cache
            .home_locs
            .iter()
            .filter(|l| !occupied_set.contains(l) && !blocked_set.contains(l))
            .copied()
            .collect();

        let pw = nohome::partner_weights(future_cz_layers, nohome_opts.gamma);

        let candidates = nohome::candidate_return_layouts(
            &root,
            &nh_cache.home_set,
            &holes,
            &nh_cache.dist_table,
            &self.index,
            &pw,
            nohome_opts,
        );

        // Phase 1: route every candidate's return layout, pick the candidate
        // whose Phase-1 routing produces the fewest move layers (the
        // hop-count substitute for Python's _estimate_layers_time).
        let mut total_expanded: u32 = 0;
        let mut best_p1: Option<SolveResult> = None;
        let mut p1_saw_budget_exceeded = false;
        for candidate in &candidates {
            let return_target: Vec<(u32, LocationAddr)> = candidate.clone();
            let return_result = self.solve(
                root.iter(),
                return_target,
                blocked_locs.iter().copied(),
                max_expansions,
                opts,
                None,
            )?;

            total_expanded += return_result.nodes_expanded;

            match return_result.status {
                SolveStatus::Solved => {
                    if best_p1
                        .as_ref()
                        .is_none_or(|b| return_result.move_layers.len() < b.move_layers.len())
                    {
                        best_p1 = Some(return_result);
                    }
                }
                SolveStatus::BudgetExceeded => p1_saw_budget_exceeded = true,
                SolveStatus::Unsolvable => {}
            }
        }

        let Some(return_result) = best_p1 else {
            // No candidate routed successfully.
            let status = if p1_saw_budget_exceeded {
                SolveStatus::BudgetExceeded
            } else {
                SolveStatus::Unsolvable
            };
            return Ok(SolveResult::unsolved(status, root, total_expanded, 0));
        };

        // Phase 2: simple per-pair target picker, then route once.
        let cz_targets = resolve_cz_targets(&return_result.goal_config);
        let entangling_result = self.solve(
            return_result.goal_config.iter(),
            cz_targets,
            blocked_locs.iter().copied(),
            max_expansions,
            opts,
            None,
        )?;

        total_expanded += entangling_result.nodes_expanded;

        if entangling_result.status == SolveStatus::Solved {
            let total_cost = return_result.cost + entangling_result.cost;
            let mut combined_layers = return_result.move_layers;
            combined_layers.extend(entangling_result.move_layers);
            return Ok(SolveResult::solved(
                entangling_result.goal_config,
                combined_layers,
                total_cost,
                total_expanded,
                return_result.deadlocks + entangling_result.deadlocks,
            ));
        }

        Ok(SolveResult::unsolved(
            entangling_result.status,
            root,
            total_expanded,
            return_result.deadlocks + entangling_result.deadlocks,
        ))
    }
}

// ── Multi-candidate solve ──

/// Per-candidate debug info recorded during [`MoveSolver::solve_with_generator`].
#[derive(Debug, Clone)]
pub struct CandidateAttempt {
    /// Index of this candidate in the generator's output.
    pub candidate_index: usize,
    /// Outcome status of the solve attempt for this candidate.
    pub status: SolveStatus,
    /// Number of nodes expanded for this candidate.
    pub nodes_expanded: u32,
}

/// Result of a multi-candidate solve attempt via [`MoveSolver::solve_with_generator`].
#[derive(Debug)]
pub struct MultiSolveResult {
    /// The solve result from the winning candidate (or the last attempted).
    pub result: SolveResult,
    /// Index of the candidate that succeeded (`None` if all failed).
    pub candidate_index: Option<usize>,
    /// Total nodes expanded across all candidates.
    pub total_expansions: u32,
    /// Number of candidates actually attempted (excludes validation failures).
    pub candidates_tried: usize,
    /// Per-candidate attempt details for debugging.
    pub attempts: Vec<CandidateAttempt>,
}

impl MoveSolver {
    /// Solve using a target generator: generates candidates, validates each,
    /// and tries them in order with a shared expansion budget.
    ///
    /// Returns on the first successful solve, or the result of the last
    /// candidate if all fail or the budget runs out.
    #[allow(clippy::too_many_arguments)]
    pub fn solve_with_generator(
        &self,
        initial: impl IntoIterator<Item = (u32, LocationAddr)>,
        blocked: impl IntoIterator<Item = LocationAddr>,
        controls: &[u32],
        targets: &[u32],
        generator: &dyn TargetGenerator,
        max_expansions: Option<u32>,
        opts: &SolveOptions,
        entropy_opts: Option<&EntropyOptions>,
    ) -> Result<MultiSolveResult, ConfigError> {
        let initial_pairs: Vec<(u32, LocationAddr)> = initial.into_iter().collect();
        let blocked_locs: Vec<LocationAddr> = blocked.into_iter().collect();

        let ctx = TargetContext {
            placement: &initial_pairs,
            controls,
            targets,
            index: &self.index,
        };

        let candidates = generator.generate(&ctx);

        if candidates.is_empty() {
            let root = Config::new(initial_pairs.iter().copied())?;
            return Ok(MultiSolveResult {
                result: SolveResult::unsolvable(root),
                candidate_index: None,
                total_expansions: 0,
                candidates_tried: 0,
                attempts: Vec::new(),
            });
        }

        let mut total_expansions: u32 = 0;
        let mut remaining_budget = max_expansions;
        let mut last_result = None;
        let mut attempts = Vec::new();

        for (i, candidate) in candidates.iter().enumerate() {
            if validate_candidate(candidate, controls, targets, &self.index).is_err() {
                continue;
            }

            let result = self.solve(
                initial_pairs.iter().copied(),
                candidate.iter().copied(),
                blocked_locs.iter().copied(),
                remaining_budget,
                opts,
                entropy_opts,
            )?;

            total_expansions += result.nodes_expanded;
            attempts.push(CandidateAttempt {
                candidate_index: i,
                status: result.status,
                nodes_expanded: result.nodes_expanded,
            });

            if result.status == SolveStatus::Solved {
                return Ok(MultiSolveResult {
                    result,
                    candidate_index: Some(i),
                    total_expansions,
                    candidates_tried: attempts.len(),
                    attempts,
                });
            }

            if let Some(budget) = remaining_budget.as_mut() {
                *budget = budget.saturating_sub(result.nodes_expanded);
                if *budget == 0 {
                    return Ok(MultiSolveResult {
                        result,
                        candidate_index: None,
                        total_expansions,
                        candidates_tried: attempts.len(),
                        attempts,
                    });
                }
            }

            last_result = Some(result);
        }

        let result = last_result.unwrap_or_else(|| {
            let root =
                Config::new(initial_pairs.iter().copied()).expect("initial was valid on entry");
            SolveResult::unsolvable(root)
        });

        Ok(MultiSolveResult {
            result,
            candidate_index: None,
            total_expansions,
            candidates_tried: attempts.len(),
            attempts,
        })
    }

    /// Generate and validate candidate target configurations without solving.
    ///
    /// Useful for inspecting what a generator would produce.
    /// Returns only candidates that pass validation.
    pub fn generate_candidates(
        &self,
        initial: &[(u32, LocationAddr)],
        controls: &[u32],
        targets: &[u32],
        generator: &dyn TargetGenerator,
    ) -> Vec<Vec<(u32, LocationAddr)>> {
        let ctx = TargetContext {
            placement: initial,
            controls,
            targets,
            index: &self.index,
        };

        generator
            .generate(&ctx)
            .into_iter()
            .filter(|c| validate_candidate(c, controls, targets, &self.index).is_ok())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{example_arch_json, loc};

    /// Default test options: A*.
    fn default_opts() -> SolveOptions {
        SolveOptions::default()
    }

    #[test]
    fn solve_simple_one_step() {
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        let result = solver
            .solve(
                [(0, loc(0, 0))],
                [(0, loc(0, 5))],
                std::iter::empty(),
                Some(100),
                &default_opts(),
                None,
            )
            .unwrap();

        assert_eq!(result.status, SolveStatus::Solved);
        assert_eq!(result.cost, 1.0);
        assert_eq!(result.move_layers.len(), 1);
        assert_eq!(result.goal_config.location_of(0), Some(loc(0, 5)));
    }

    #[test]
    fn solve_already_at_target() {
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        let result = solver
            .solve(
                [(0, loc(0, 5))],
                [(0, loc(0, 5))],
                std::iter::empty(),
                Some(100),
                &default_opts(),
                None,
            )
            .unwrap();

        assert_eq!(result.status, SolveStatus::Solved);
        assert_eq!(result.cost, 0.0);
        assert!(result.move_layers.is_empty());
        assert_eq!(result.nodes_expanded, 0);
    }

    #[test]
    fn solve_cross_word() {
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        // Move qubit from word 0 site 5 to word 1 site 5 (one word bus hop).
        let result = solver
            .solve(
                [(0, loc(0, 5))],
                [(0, loc(1, 5))],
                std::iter::empty(),
                Some(100),
                &default_opts(),
                None,
            )
            .unwrap();

        assert_eq!(result.status, SolveStatus::Solved);
        assert_eq!(result.cost, 1.0);
        assert_eq!(result.move_layers.len(), 1);
        assert_eq!(result.goal_config.location_of(0), Some(loc(1, 5)));
    }

    #[test]
    fn solve_multi_step() {
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        // Word 0 site 0 → word 1 site 5: needs site bus + word bus = 2 steps.
        let result = solver
            .solve(
                [(0, loc(0, 0))],
                [(0, loc(1, 5))],
                std::iter::empty(),
                Some(1000),
                &default_opts(),
                None,
            )
            .unwrap();

        assert_eq!(result.status, SolveStatus::Solved);
        assert_eq!(result.cost, 2.0);
        assert_eq!(result.move_layers.len(), 2);
    }

    #[test]
    fn solve_no_solution() {
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        // Target a nonexistent location.
        let result = solver
            .solve(
                [(0, loc(0, 0))],
                [(0, loc(99, 99))],
                std::iter::empty(),
                Some(100),
                &default_opts(),
                None,
            )
            .unwrap();

        assert_ne!(result.status, SolveStatus::Solved);
        assert!(result.move_layers.is_empty());
    }

    #[test]
    fn solver_reusable() {
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        let opts = default_opts();

        let r1 = solver
            .solve(
                [(0, loc(0, 0))],
                [(0, loc(0, 5))],
                std::iter::empty(),
                Some(100),
                &opts,
                None,
            )
            .unwrap();

        let r2 = solver
            .solve(
                [(0, loc(0, 5))],
                [(0, loc(0, 0))],
                std::iter::empty(),
                Some(100),
                &opts,
                None,
            )
            .unwrap();

        assert_eq!(r1.cost, 1.0);
        assert_eq!(r2.cost, 1.0);
    }

    #[test]
    fn solve_with_blocked() {
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        // Qubit at site 0, target site 5, but site 5 is blocked.
        // Should find no solution (or a longer path if one exists).
        let result = solver
            .solve(
                [(0, loc(0, 0))],
                [(0, loc(0, 5))],
                [loc(0, 5)],
                Some(100),
                &default_opts(),
                None,
            )
            .unwrap();

        // Can't reach blocked destination.
        assert_ne!(result.status, SolveStatus::Solved);
        assert!(result.move_layers.is_empty());
    }

    #[test]
    fn solve_multiple_qubits() {
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        // Move two qubits from sites 0,1 to sites 5,6 (parallel site bus move).
        let result = solver
            .solve(
                [(0, loc(0, 0)), (1, loc(0, 1))],
                [(0, loc(0, 5)), (1, loc(0, 6))],
                std::iter::empty(),
                Some(1000),
                &default_opts(),
                None,
            )
            .unwrap();

        // Should find the parallel move in 1 step.
        assert_eq!(result.status, SolveStatus::Solved);
        assert_eq!(result.cost, 1.0);
        assert_eq!(result.goal_config.location_of(0), Some(loc(0, 5)));
        assert_eq!(result.goal_config.location_of(1), Some(loc(0, 6)));
    }

    #[test]
    fn cascade_finds_equal_or_better_than_ids() {
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        // Multi-step problem: word 0 site 0 → word 1 site 5.
        let ids_result = solver
            .solve(
                [(0, loc(0, 0))],
                [(0, loc(1, 5))],
                std::iter::empty(),
                Some(1000),
                &SolveOptions {
                    strategy: Strategy::Ids,
                    ..SolveOptions::default()
                },
                None,
            )
            .unwrap();

        let cascade_result = solver
            .solve(
                [(0, loc(0, 0))],
                [(0, loc(1, 5))],
                std::iter::empty(),
                Some(1000),
                &SolveOptions {
                    strategy: Strategy::Cascade {
                        inner: InnerStrategy::Ids,
                    },
                    ..SolveOptions::default()
                },
                None,
            )
            .unwrap();

        assert_eq!(ids_result.status, SolveStatus::Solved);
        assert_eq!(cascade_result.status, SolveStatus::Solved);
        assert!(cascade_result.cost <= ids_result.cost);
    }

    // ── solve_with_generator tests ──

    #[test]
    fn solve_with_generator_default_solves_cz() {
        use crate::placement::target_generator::DefaultTargetGenerator;

        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        // Qubit 0 at word 0 site 0, qubit 1 at word 1 site 0.
        // CZ pair: word 0 ↔ word 1. DefaultTargetGenerator should produce
        // a candidate where qubit 0 stays at word 0 (CZ partner of word 1).
        let result = solver
            .solve_with_generator(
                [(0, loc(0, 0)), (1, loc(1, 0))],
                std::iter::empty(),
                &[0],
                &[1],
                &DefaultTargetGenerator,
                Some(1000),
                &default_opts(),
                None,
            )
            .unwrap();

        assert_eq!(result.result.status, SolveStatus::Solved);
        assert_eq!(result.candidate_index, Some(0));
        assert_eq!(result.candidates_tried, 1);
        assert_eq!(result.attempts.len(), 1);
    }

    #[test]
    fn solve_with_generator_empty_candidates() {
        use crate::placement::target_generator::DefaultTargetGenerator;

        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        // Qubit 1 missing from placement — DefaultTargetGenerator returns empty.
        let result = solver
            .solve_with_generator(
                [(0, loc(0, 0))],
                std::iter::empty(),
                &[0],
                &[1],
                &DefaultTargetGenerator,
                Some(1000),
                &default_opts(),
                None,
            )
            .unwrap();

        assert_eq!(result.result.status, SolveStatus::Unsolvable);
        assert_eq!(result.candidate_index, None);
        assert_eq!(result.candidates_tried, 0);
        assert!(result.attempts.is_empty());
    }

    #[test]
    fn generate_candidates_returns_valid_only() {
        use crate::placement::target_generator::DefaultTargetGenerator;

        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        let initial = vec![(0, loc(0, 0)), (1, loc(1, 0))];
        let candidates = solver.generate_candidates(&initial, &[0], &[1], &DefaultTargetGenerator);
        assert_eq!(candidates.len(), 1);
    }

    #[test]
    fn entropy_strategy_can_collect_trace() {
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        let entropy_opts = EntropyOptions {
            w_t: 0.0,
            collect_entropy_trace: true,
            ..EntropyOptions::default()
        };
        let result = solver
            .solve(
                [(0, loc(0, 0))],
                [(0, loc(0, 5))],
                std::iter::empty(),
                Some(100),
                &SolveOptions {
                    strategy: Strategy::Entropy,
                    ..SolveOptions::default()
                },
                Some(&entropy_opts),
            )
            .unwrap();

        assert_eq!(result.status, SolveStatus::Solved);
        let trace = result
            .entropy_trace
            .as_ref()
            .expect("entropy trace should be populated");
        assert_eq!(trace.root_node_id, 0);
        assert!(!trace.steps.is_empty(), "trace should include step events");
    }

    // ── solve_entangling tests ──

    #[test]
    fn solve_entangling_finds_solution() {
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        let result = solver
            .solve_entangling(
                [(0, loc(0, 0)), (1, loc(1, 0))],
                &[(0, 1)],
                std::iter::empty(),
                Some(5000),
                &default_opts(),
                &EntanglingOptions::default(),
                &[],
            )
            .unwrap();

        assert_eq!(result.status, SolveStatus::Solved);
        // Verify goal config satisfies the entangling constraint.
        let arch: bloqade_lanes_bytecode_core::arch::types::ArchSpec =
            serde_json::from_str(example_arch_json()).unwrap();
        let eset = crate::ops::entangling::build_entangling_set(&arch);
        let loc_a = result.goal_config.location_of(0).unwrap().encode();
        let loc_b = result.goal_config.location_of(1).unwrap().encode();
        assert!(
            eset.contains(&(loc_a, loc_b)),
            "goal config should satisfy entangling constraint"
        );
    }

    #[test]
    fn solve_entangling_already_at_goal() {
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        // Qubits already at entangling positions.
        let result = solver
            .solve_entangling(
                [(0, loc(0, 5)), (1, loc(1, 5))],
                &[(0, 1)],
                std::iter::empty(),
                Some(100),
                &default_opts(),
                &EntanglingOptions::default(),
                &[],
            )
            .unwrap();

        assert_eq!(result.status, SolveStatus::Solved);
        assert_eq!(result.cost, 0.0);
        assert!(result.move_layers.is_empty());
    }

    #[test]
    fn solve_entangling_multiple_pairs() {
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        let result = solver
            .solve_entangling(
                [
                    (0, loc(0, 0)),
                    (1, loc(1, 0)),
                    (2, loc(0, 1)),
                    (3, loc(1, 1)),
                ],
                &[(0, 1), (2, 3)],
                std::iter::empty(),
                Some(10000),
                &default_opts(),
                &EntanglingOptions::default(),
                &[],
            )
            .unwrap();

        assert_eq!(result.status, SolveStatus::Solved);
        // Verify both pairs satisfy the constraint.
        let arch: bloqade_lanes_bytecode_core::arch::types::ArchSpec =
            serde_json::from_str(example_arch_json()).unwrap();
        let eset = crate::ops::entangling::build_entangling_set(&arch);
        for &(qa, qb) in &[(0u32, 1u32), (2, 3)] {
            let la = result.goal_config.location_of(qa).unwrap().encode();
            let lb = result.goal_config.location_of(qb).unwrap().encode();
            assert!(
                eset.contains(&(la, lb)),
                "pair ({qa}, {qb}) should be at entangling positions"
            );
        }
    }

    #[test]
    fn solve_entangling_spectator_qubits() {
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        // q0/q1 are a CZ pair, q2 is a spectator (not in any pair).
        let result = solver
            .solve_entangling(
                [(0, loc(0, 0)), (1, loc(1, 0)), (2, loc(0, 3))],
                &[(0, 1)],
                std::iter::empty(),
                Some(5000),
                &default_opts(),
                &EntanglingOptions::default(),
                &[],
            )
            .unwrap();

        assert_eq!(result.status, SolveStatus::Solved);
        // Spectator q2 should remain at its initial position.
        assert_eq!(result.goal_config.location_of(2), Some(loc(0, 3)));
    }

    #[test]
    fn solve_entangling_with_ids() {
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        let result = solver
            .solve_entangling(
                [(0, loc(0, 0)), (1, loc(1, 0))],
                &[(0, 1)],
                std::iter::empty(),
                Some(5000),
                &SolveOptions {
                    strategy: Strategy::Ids,
                    ..SolveOptions::default()
                },
                &EntanglingOptions::default(),
                &[],
            )
            .unwrap();

        assert_eq!(result.status, SolveStatus::Solved);
    }

    #[test]
    fn solve_entangling_with_cascade() {
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        let result = solver
            .solve_entangling(
                [(0, loc(0, 0)), (1, loc(1, 0))],
                &[(0, 1)],
                std::iter::empty(),
                Some(5000),
                &SolveOptions {
                    strategy: Strategy::Cascade {
                        inner: InnerStrategy::Ids,
                    },
                    ..SolveOptions::default()
                },
                &EntanglingOptions::default(),
                &[],
            )
            .unwrap();

        assert_eq!(result.status, SolveStatus::Solved);
    }
}
