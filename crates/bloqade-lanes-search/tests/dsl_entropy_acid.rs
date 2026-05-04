//! Acid test: `policies/reference/entropy.star` produces structurally
//! comparable results to `Strategy::Entropy` on deterministic placement
//! problems.
//!
//! ## Comparison specification (spec §9)
//!
//! ### goal_config (final qubit positions)
//! Both paths must reach the target configuration.
//! - `Strategy::Entropy` reaches `Solved` and its `goal_config` equals the
//!   declared target.
//! - `entropy.star` reaches `Solved` on most fixtures (candidate pipeline
//!   enabled by real `StarlarkConfig` from `graph.config` + working
//!   `is_goal`). Complex multi-hop fixtures may still fall back via
//!   `sequential_fallback`; in that case `goal_config` is still the target.
//!
//! ### move_layers.len()
//! Within ±3 of each other.  Observed diffs: tiny=0, medium=1, small=0.
//!
//! ### status
//! `Strategy::Entropy` → `Solved`.
//! `entropy.star` → `Solved` (tiny, medium) or `Fallback` (small).
//! Both `Solved` and `Fallback` are accepted; the key invariant is
//! `goal_config == target`.

use std::path::PathBuf;
use std::sync::Arc;

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;
use bloqade_lanes_bytecode_core::arch::types::ArchSpec;
use bloqade_lanes_dsl_core::sandbox::SandboxConfig;
use bloqade_lanes_search::LaneIndex;
use bloqade_lanes_search::move_policy_dsl::{
    PolicyOptions, PolicyResult, PolicyStatus, solve_with_policy,
};
use bloqade_lanes_search::solve::{MoveSolver, SolveOptions, SolveResult, SolveStatus, Strategy};

// ── helpers ──────────────────────────────────────────────────────────────────

/// Convenience constructor for `LocationAddr` (zone 0, word `w`, site `s`).
fn loc(word: u32, site: u32) -> LocationAddr {
    LocationAddr {
        zone_id: 0,
        word_id: word,
        site_id: site,
    }
}

/// Canonical sorted representation of a goal config for easy comparison.
fn canonical_config(cfg: &bloqade_lanes_search::Config) -> Vec<(u32, LocationAddr)> {
    let mut entries: Vec<(u32, LocationAddr)> = cfg.iter().collect();
    entries.sort_by_key(|&(q, _)| q);
    entries
}

/// Path to the reference `entropy.star` policy relative to the repo root.
///
/// Integration tests run from the repo root (workspace Cargo.toml directory),
/// so `CARGO_MANIFEST_DIR` for this crate is two levels below the repo root.
fn entropy_star_path() -> String {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/ dir")
        .parent()
        .expect("repo root")
        .join("policies/reference/entropy.star")
        .to_string_lossy()
        .into_owned()
}

/// Example two-word architecture JSON (same as `test_utils::example_arch_json`).
///
/// Zone-centric schema: 2 words × 10 sites each. Site-bus moves source
/// row (sites 0–4) to destination row (sites 5–9); word-bus moves between
/// words via sites 5–9.
fn example_arch_json() -> &'static str {
    r#"{
        "version": "2.0",
        "words": [
            { "sites": [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [0, 1], [1, 1], [2, 1], [3, 1], [4, 1]] },
            { "sites": [[0, 2], [1, 2], [2, 2], [3, 2], [4, 2], [0, 3], [1, 3], [2, 3], [3, 3], [4, 3]] }
        ],
        "zones": [
            {
                "grid": { "x_start": 1.0, "y_start": 2.5, "x_spacing": [2.0, 2.0, 2.0, 2.0], "y_spacing": [2.5, 7.5, 2.5] },
                "site_buses": [
                    { "src": [0, 1, 2, 3, 4], "dst": [5, 6, 7, 8, 9] }
                ],
                "word_buses": [
                    { "src": [0], "dst": [1] }
                ],
                "words_with_site_buses": [0, 1],
                "sites_with_word_buses": [5, 6, 7, 8, 9],
                "entangling_pairs": [[0, 1]]
            }
        ],
        "zone_buses": [],
        "modes": [
            { "name": "default", "zones": [0], "bitstring_order": [] }
        ]
    }"#
}

// ── Strategy::Entropy runner ──────────────────────────────────────────────────

/// Run `Strategy::Entropy` (Rust) on a problem and return the result.
///
/// Uses `restarts = 1` and `seed = 0` (no RNG variance) to get a
/// deterministic outcome.
fn run_strategy_entropy(
    arch_json: &str,
    initial: &[(u32, LocationAddr)],
    target: &[(u32, LocationAddr)],
    blocked: &[LocationAddr],
    max_expansions: u32,
) -> SolveResult {
    let solver = MoveSolver::from_json(arch_json).expect("arch parse");
    let opts = SolveOptions {
        strategy: Strategy::Entropy,
        restarts: 1, // single restart — no RNG-driven variance
        w_t: 0.0,    // hop-count only for determinism
        ..SolveOptions::default()
    };
    solver
        .solve(
            initial.iter().copied(),
            target.iter().copied(),
            blocked.iter().copied(),
            Some(max_expansions),
            &opts,
        )
        .expect("solve error")
}

// ── entropy.star runner ───────────────────────────────────────────────────────

/// Run `entropy.star` (DSL) on a problem and return the result.
fn run_entropy_star(
    arch_json: &str,
    initial: &[(u32, LocationAddr)],
    target: &[(u32, LocationAddr)],
    blocked: &[LocationAddr],
    max_expansions: u64,
) -> PolicyResult {
    let spec: ArchSpec = serde_json::from_str(arch_json).expect("arch parse");
    let index = Arc::new(LaneIndex::new(spec));
    let opts = PolicyOptions {
        policy_path: entropy_star_path(),
        policy_params: serde_json::json!({}),
        max_expansions,
        timeout_s: Some(30.0),
        sandbox: SandboxConfig::default(),
    };
    solve_with_policy(
        initial.iter().copied(),
        target.iter().copied(),
        blocked.iter().copied(),
        index,
        opts,
        &mut bloqade_lanes_search::move_policy_dsl::NoOpMoveObserver,
    )
    .expect("solve_with_policy error")
}

// ── assertions ────────────────────────────────────────────────────────────────

/// Core acid assertions shared across all fixtures.
///
/// Assertions:
/// 1. `Strategy::Entropy` must reach `Solved` (otherwise the fixture is broken).
/// 2. `entropy.star` must reach either `Solved` or `Fallback` (both are valid
///    in v1; anything else is a policy error).
/// 3. When both paths terminate, the `goal_config` must match the target
///    (correctness invariant — not a heuristic).
/// 4. If both paths produce a non-empty move sequence, their lengths must
///    be within the stated `n_layers_tolerance` of each other.
fn assert_acid(
    strategy_result: &SolveResult,
    policy_result: &PolicyResult,
    target: &[(u32, LocationAddr)],
    fixture_name: &str,
    n_layers_tolerance: i64,
) {
    println!("=== {fixture_name} ===");
    println!(
        "  Strategy::Entropy : status={:?}, n_layers={}, nodes_expanded={}",
        strategy_result.status,
        strategy_result.move_layers.len(),
        strategy_result.nodes_expanded
    );
    println!(
        "  entropy.star      : status={:?}, n_layers={}, nodes_expanded={}",
        policy_result.status,
        policy_result.move_layers.len(),
        policy_result.nodes_expanded
    );

    // ── 1. Rust side must solve ──────────────────────────────────────────────
    assert_eq!(
        strategy_result.status,
        SolveStatus::Solved,
        "[{fixture_name}] Strategy::Entropy did not solve: {:?}",
        strategy_result.status
    );

    // ── 2. DSL side must reach Solved or Fallback ───────────────────────────
    // entropy.star reaches Solved when the candidate pipeline finds a path.
    // On complex multi-hop problems the policy may still fall back via
    // sequential_fallback (Fallback); both outcomes are acceptable.
    let policy_ok = matches!(
        &policy_result.status,
        PolicyStatus::Solved | PolicyStatus::Fallback(_)
    );
    assert!(
        policy_ok,
        "[{fixture_name}] entropy.star hit an unexpected status: {:?}",
        policy_result.status
    );

    // ── 3. goal_config must equal target (correctness, not heuristic) ────────
    // Build canonical target from the test fixture.
    let mut expected_target: Vec<(u32, LocationAddr)> = target.to_vec();
    expected_target.sort_by_key(|&(q, _)| q);

    let strategy_goal = canonical_config(&strategy_result.goal_config);
    assert_eq!(
        strategy_goal, expected_target,
        "[{fixture_name}] Strategy::Entropy goal_config does not match target"
    );

    // For entropy.star: if the policy reached Solved, goal_config must match
    // target. If it reached Fallback, sequential_fallback should have produced
    // the target config.
    if !policy_result.move_layers.is_empty() {
        let policy_goal = canonical_config(&policy_result.goal_config);
        assert_eq!(
            policy_goal, expected_target,
            "[{fixture_name}] entropy.star goal_config does not match target"
        );
    }

    // ── 4. move_layers length tolerance ─────────────────────────────────────
    // Only check when both paths produce non-empty move layers (i.e., when the
    // problem actually requires moves, and entropy.star produced a path via
    // sequential_fallback).
    if !strategy_result.move_layers.is_empty() && !policy_result.move_layers.is_empty() {
        let diff = (strategy_result.move_layers.len() as i64
            - policy_result.move_layers.len() as i64)
            .abs();
        println!(
            "  n_layers diff     : |{} - {}| = {} (tolerance = {})",
            strategy_result.move_layers.len(),
            policy_result.move_layers.len(),
            diff,
            n_layers_tolerance,
        );
        assert!(
            diff <= n_layers_tolerance,
            "[{fixture_name}] move_layers length diff too large: strategy={}, policy={}, diff={}, tolerance={}",
            strategy_result.move_layers.len(),
            policy_result.move_layers.len(),
            diff,
            n_layers_tolerance,
        );
    } else {
        println!(
            "  n_layers diff     : skipped (strategy={}, policy={})",
            strategy_result.move_layers.len(),
            policy_result.move_layers.len(),
        );
    }
}

// ── Fixture 1: tiny — 1 qubit, 1 bus hop ──────────────────────────────────────

/// Fixture 1: single qubit, single-hop site-bus move.
///
/// Qubit 0 starts at word 0, site 0 (source row).
/// Target: word 0, site 5 (destination row).
/// One site-bus move is required.
///
/// This is the simplest non-trivial deterministic problem.
/// Expected: entropy.star now reaches Solved (graph.config returns a real
/// StarlarkConfig and is_goal works).  Both paths find the 1-hop optimal
/// solution.  n_layers tolerance tightened to ±3.
#[test]
fn entropy_star_matches_strategy_entropy_tiny() {
    let arch_json = example_arch_json();
    let initial = vec![(0u32, loc(0, 0))];
    let target = vec![(0u32, loc(0, 5))];
    let blocked: Vec<LocationAddr> = vec![];

    let strategy_result = run_strategy_entropy(arch_json, &initial, &target, &blocked, 1_000);
    let policy_result = run_entropy_star(arch_json, &initial, &target, &blocked, 1_000);

    assert_acid(
        &strategy_result,
        &policy_result,
        &target,
        "tiny: 1 qubit, 1 site-bus hop",
        // Tolerance tightened to ±3. Observed diff = 0.
        3,
    );
}

// ── Fixture 2: small — 1 qubit, 2 bus hops (cross-word) ───────────────────────

/// Fixture 2: single qubit, two-hop cross-word move.
///
/// Qubit 0 starts at word 0, site 0.
/// Target: word 1, site 5.
/// Path: site-bus (site 0 → site 5 in word 0), then word-bus (word 0 → word 1).
/// Two bus hops are required.
///
/// Strategy::Entropy finds the 2-layer optimal path.
/// entropy.star exhausts candidates (2-hop requires intermediate step that
/// doesn't score well under e_max=8 budget) and falls back via
/// sequential_fallback, which also produces a 2-hop BFS path.
/// Expected diff: 0 layers.  n_layers tolerance tightened to ±3.
#[test]
fn entropy_star_matches_strategy_entropy_small() {
    let arch_json = example_arch_json();
    let initial = vec![(0u32, loc(0, 0))];
    let target = vec![(0u32, loc(1, 5))];
    let blocked: Vec<LocationAddr> = vec![];

    let strategy_result = run_strategy_entropy(arch_json, &initial, &target, &blocked, 2_000);
    let policy_result = run_entropy_star(arch_json, &initial, &target, &blocked, 2_000);

    assert_acid(
        &strategy_result,
        &policy_result,
        &target,
        "small: 1 qubit, 2 hops (cross-word)",
        // Tolerance tightened to ±3. Observed diff = 0.
        3,
    );
}

// ── Fixture 3: medium — 2 qubits, parallel moves ──────────────────────────────

/// Fixture 3: two qubits, parallel site-bus move.
///
/// Qubit 0 at word 0, site 0.  Target: word 0, site 5.
/// Qubit 1 at word 0, site 1.  Target: word 0, site 6.
///
/// Both qubits share the same site-bus (bus 0: src=[0,1,2,3,4] dst=[5,6,7,8,9]).
/// Strategy::Entropy can move them in parallel (1 layer).
/// entropy.star now reaches Solved (graph.config + is_goal wired up):
///   - Finds a packed move that moves both qubits simultaneously → 2 layers.
///     Strategy::Entropy finds the 1-parallel-layer optimum; entropy.star finds
///     2 layers (sequential pack due to scoring).  Observed diff = 1.
///
/// n_layers tolerance tightened to ±3.
#[test]
fn entropy_star_matches_strategy_entropy_medium() {
    let arch_json = example_arch_json();
    let initial = vec![(0u32, loc(0, 0)), (1u32, loc(0, 1))];
    let target = vec![(0u32, loc(0, 5)), (1u32, loc(0, 6))];
    let blocked: Vec<LocationAddr> = vec![];

    let strategy_result = run_strategy_entropy(arch_json, &initial, &target, &blocked, 5_000);
    let policy_result = run_entropy_star(arch_json, &initial, &target, &blocked, 5_000);

    assert_acid(
        &strategy_result,
        &policy_result,
        &target,
        "medium: 2 qubits, parallel site-bus moves",
        // Tolerance tightened to ±3. Observed diff = 1.
        3,
    );
}
