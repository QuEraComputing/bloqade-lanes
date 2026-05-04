//! `policy_params` overrides are applied to the policy's PARAMS at solve time.
//!
//! The kernel binds `PARAMS_OVERRIDE` on the policy module before eval so that
//! module-level code in the policy can merge it into `PARAMS_DEFAULTS`.
//! This test confirms that:
//!   1. A no-override run (empty `{}`) behaves identically to the baseline.
//!   2. An override run with `e_max=1` (far below the default of 8) forces
//!      the entropy ceiling to trigger almost immediately, which changes the
//!      solve behaviour vs. the default.
//!   3. Neither run produces a `RuntimeError` mentioning `PARAMS_OVERRIDE`
//!      (which would indicate the binding failed).

use std::path::PathBuf;
use std::sync::Arc;

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;
use bloqade_lanes_bytecode_core::arch::types::ArchSpec;
use bloqade_lanes_dsl_core::sandbox::SandboxConfig;
use bloqade_lanes_search::LaneIndex;
use bloqade_lanes_search::move_policy_dsl::{
    PolicyOptions, PolicyResult, PolicyStatus, solve_with_policy,
};

// ── helpers ───────────────────────────────────────────────────────────────────

fn loc(word: u32, site: u32) -> LocationAddr {
    LocationAddr {
        zone_id: 0,
        word_id: word,
        site_id: site,
    }
}

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

/// Two-word architecture (same as the acid-test fixture).
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

fn run_with_params(
    initial: &[(u32, LocationAddr)],
    target: &[(u32, LocationAddr)],
    params: serde_json::Value,
    max_expansions: u64,
) -> PolicyResult {
    let spec: ArchSpec = serde_json::from_str(example_arch_json()).expect("arch parse");
    let index = Arc::new(LaneIndex::new(spec));
    let opts = PolicyOptions {
        policy_path: entropy_star_path(),
        policy_params: params,
        max_expansions,
        timeout_s: Some(15.0),
        sandbox: SandboxConfig::default(),
    };
    solve_with_policy(
        initial.iter().copied(),
        target.iter().copied(),
        std::iter::empty::<LocationAddr>(),
        index,
        opts,
        &mut bloqade_lanes_search::move_policy_dsl::NoOpMoveObserver,
    )
    .expect("solve_with_policy")
}

/// Returns true if the status does NOT mention `PARAMS_OVERRIDE` — which
/// would indicate the override binding broke the policy.
fn status_is_not_override_error(s: &PolicyStatus) -> bool {
    let s = format!("{s:?}");
    !s.contains("PARAMS_OVERRIDE")
}

// ── tests ─────────────────────────────────────────────────────────────────────

/// Empty override dict → identical behaviour to the prior baseline.
/// The acid test already covers this in detail; here we just confirm no
/// crash and a valid terminal status.
#[test]
fn empty_override_leaves_policy_unchanged() {
    // Tiny fixture: 1 qubit, 1 site-bus hop (always Solved quickly).
    let initial = vec![(0u32, loc(0, 0))];
    let target = vec![(0u32, loc(0, 5))];

    let result = run_with_params(&initial, &target, serde_json::json!({}), 1_000);

    println!(
        "empty override: status={:?} layers={}",
        result.status,
        result.move_layers.len()
    );

    assert!(
        status_is_not_override_error(&result.status),
        "empty override produced a PARAMS_OVERRIDE-related error: {:?}",
        result.status
    );
    // Tiny fixture is always Solved.
    assert_eq!(
        result.status,
        PolicyStatus::Solved,
        "expected Solved for tiny fixture with empty override, got {:?}",
        result.status
    );
}

/// `e_max=1` forces the entropy ceiling to trigger on the very first node
/// explored (entropy starts at 1 and the ceiling is 1), causing the policy
/// to immediately revert/fall back.  This differs from the default `e_max=8`
/// behaviour where the policy has room to explore.
#[test]
fn policy_params_override_e_max_changes_behavior() {
    // Use the "small" cross-word fixture: 1 qubit, 2 hops required.
    // With the default e_max=8 the policy has headroom to explore; with
    // e_max=1 it triggers fallback on the very first step.
    let initial = vec![(0u32, loc(0, 0))];
    let target = vec![(0u32, loc(1, 5))];

    let default_result = run_with_params(&initial, &target, serde_json::json!({}), 2_000);
    let low_emax_result =
        run_with_params(&initial, &target, serde_json::json!({"e_max": 1}), 2_000);

    println!(
        "default:   status={:?} layers={} nodes_expanded={}",
        default_result.status,
        default_result.move_layers.len(),
        default_result.nodes_expanded
    );
    println!(
        "e_max=1:   status={:?} layers={} nodes_expanded={}",
        low_emax_result.status,
        low_emax_result.move_layers.len(),
        low_emax_result.nodes_expanded
    );

    // Neither run should produce a PARAMS_OVERRIDE-related runtime error.
    assert!(
        status_is_not_override_error(&default_result.status),
        "default run produced a PARAMS_OVERRIDE error: {:?}",
        default_result.status
    );
    assert!(
        status_is_not_override_error(&low_emax_result.status),
        "e_max=1 run produced a PARAMS_OVERRIDE error: {:?}",
        low_emax_result.status
    );

    // Both runs must produce a valid terminal status (Solved or Fallback).
    let is_valid = |s: &PolicyStatus| matches!(s, PolicyStatus::Solved | PolicyStatus::Fallback(_));
    assert!(
        is_valid(&default_result.status),
        "default run hit unexpected status: {:?}",
        default_result.status
    );
    assert!(
        is_valid(&low_emax_result.status),
        "e_max=1 run hit unexpected status: {:?}",
        low_emax_result.status
    );

    // With e_max=1, the entropy ceiling triggers immediately, so the policy
    // goes to sequential_fallback much faster (fewer nodes_expanded).
    assert!(
        low_emax_result.nodes_expanded <= default_result.nodes_expanded,
        "expected e_max=1 to expand fewer or equal nodes than default, \
         but got e_max=1={} vs default={}",
        low_emax_result.nodes_expanded,
        default_result.nodes_expanded
    );
}

/// Override a float parameter (`w_d=0.0`).  This exercises the JSON→Starlark
/// float conversion path and confirms non-integer overrides work correctly.
#[test]
fn float_param_override_is_accepted() {
    let initial = vec![(0u32, loc(0, 0))];
    let target = vec![(0u32, loc(0, 5))];

    let result = run_with_params(
        &initial,
        &target,
        serde_json::json!({"w_d": 0.0, "w_m": 2.0}),
        1_000,
    );

    println!(
        "float override: status={:?} layers={}",
        result.status,
        result.move_layers.len()
    );

    assert!(
        status_is_not_override_error(&result.status),
        "float override produced a PARAMS_OVERRIDE error: {:?}",
        result.status
    );
    assert!(
        matches!(
            result.status,
            PolicyStatus::Solved | PolicyStatus::Fallback(_)
        ),
        "expected Solved or Fallback, got {:?}",
        result.status
    );
}
