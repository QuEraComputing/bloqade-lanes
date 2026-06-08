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

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;
use bloqade_lanes_bytecode_core::arch::types::ArchSpec;

use crate::placement::nohome::NoHomeOptions;
use crate::placement::target_generator::{TargetContext, TargetGenerator, validate_candidate};
use crate::primitives::config::ConfigError;
use crate::primitives::lane_index::LaneIndex;
use crate::search::engine::SearchEngine;
use crate::search::target_solver::solve_with_engine;

// Re-export the moved types under the original `solve::*` path so existing
// `use crate::search::solve::SolveStatus` imports keep resolving until the
// consumer migration completes.
pub use crate::search::options::{
    EntanglingOptions, EntropyOptions, InnerStrategy, SolveOptions, Strategy,
};
pub use crate::search::result::{SolveResult, SolveStatus};

/// Reusable move synthesis solver.
///
/// Constructed once per architecture — parses the arch spec JSON and builds
/// the lane index. Then [`solve`](MoveSolver::solve) can be called multiple
/// times with different initial/target placements.
///
/// Works for both physical and logical architectures (same interface,
/// different arch spec JSON).
///
/// **Internally a thin wrapper around [`SearchEngine`].** The new
/// `MoveSearch` / `TargetSolver` / `CzPlacement` composition layer (in
/// progress) consumes `Arc<SearchEngine>` directly; `MoveSolver`
/// remains as the legacy facade until the migration completes.
pub struct MoveSolver {
    engine: SearchEngine,
}

impl std::fmt::Debug for MoveSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MoveSolver")
            .field("engine", &self.engine)
            .finish_non_exhaustive()
    }
}

impl MoveSolver {
    /// Construct from an [`ArchSpec`] JSON string.
    ///
    /// Parses the JSON, builds the lane index (precomputes all lane lookups,
    /// endpoints, and positions).
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        Ok(Self {
            engine: SearchEngine::from_json(json)?,
        })
    }

    /// Construct from an existing [`LaneIndex`].
    pub fn from_index(index: LaneIndex) -> Self {
        Self {
            engine: SearchEngine::from_index(index),
        }
    }

    /// Construct from a borrowed [`ArchSpec`]. Avoids the JSON round-trip
    /// that callers holding a wrapper around an `ArchSpec` would otherwise
    /// pay to materialize an owned spec.
    pub fn from_arch_spec(arch_spec: &ArchSpec) -> Self {
        Self::from_index(LaneIndex::from_arch_spec(arch_spec))
    }

    /// Construct from an existing [`SearchEngine`]. Used by the new
    /// `MoveSearch` / `TargetSolver` composition layer to share an
    /// engine instance across the legacy facade.
    pub fn from_engine(engine: SearchEngine) -> Self {
        Self { engine }
    }

    /// Borrow the underlying [`SearchEngine`].
    pub fn engine(&self) -> &SearchEngine {
        &self.engine
    }

    /// Consume the solver and return the underlying [`SearchEngine`].
    pub fn into_engine(self) -> SearchEngine {
        self.engine
    }

    /// Access the underlying lane index.
    pub fn index(&self) -> &LaneIndex {
        self.engine.index()
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
        solve_with_engine(
            &self.engine,
            opts,
            entropy_opts,
            initial,
            target,
            blocked,
            max_expansions,
        )
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
        crate::placement::loose_goal::solve_loose_goal(
            &self.engine,
            opts,
            ent_opts,
            initial,
            cz_pairs,
            blocked,
            max_expansions,
            future_cz_layers,
        )
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
        crate::placement::receding_horizon::solve_receding_horizon(
            &self.engine,
            opts,
            ent_opts,
            rh_opts,
            initial,
            cz_pairs,
            blocked,
            max_expansions,
            future_cz_layers,
        )
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
        crate::placement::nohome::solve_nohome(
            &self.engine,
            opts,
            nohome_opts,
            initial,
            cz_pairs,
            blocked,
            max_expansions,
            future_cz_layers,
        )
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
        crate::placement::single_heuristic::solve_single_heuristic(
            &self.engine,
            opts,
            entropy_opts,
            generator,
            initial,
            controls,
            targets,
            blocked,
            max_expansions,
        )
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
            index: self.index(),
        };

        generator
            .generate(&ctx)
            .into_iter()
            .filter(|c| validate_candidate(c, controls, targets, self.index()).is_ok())
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
