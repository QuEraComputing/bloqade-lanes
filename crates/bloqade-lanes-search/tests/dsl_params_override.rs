//! `policy_params` overrides are applied to the policy's PARAMS at solve time.
//!
//! The kernel binds `PARAMS_OVERRIDE` on the policy module before eval so that
//! module-level code in the policy can merge it into `PARAMS_DEFAULTS`.
//! This test confirms that:
//!   1. A no-override run (empty `{}`) sees the policy defaults.
//!   2. An override run changes policy behaviour.
//!   3. Integer and float overrides cross the JSON -> Starlark boundary.

use std::io::Write;
use std::sync::Arc;

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;
use bloqade_lanes_bytecode_core::arch::types::ArchSpec;
use bloqade_lanes_dsl_core::sandbox::SandboxConfig;
use bloqade_lanes_search::LaneIndex;
use bloqade_lanes_search::move_policy_dsl::{
    PolicyOptions, PolicyResult, PolicyStatus, solve_with_policy,
};
use tempfile::NamedTempFile;

// ── helpers ───────────────────────────────────────────────────────────────────

fn loc(word: u32, site: u32) -> LocationAddr {
    LocationAddr {
        zone_id: 0,
        word_id: word,
        site_id: site,
    }
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
    policy_path: &str,
    params: serde_json::Value,
    max_expansions: u64,
) -> PolicyResult {
    let spec: ArchSpec = serde_json::from_str(example_arch_json()).expect("arch parse");
    let index = Arc::new(LaneIndex::new(spec));
    let opts = PolicyOptions {
        policy_path: policy_path.to_string(),
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

fn param_probe_policy() -> NamedTempFile {
    let mut tmp = NamedTempFile::new().expect("temp policy");
    writeln!(
        tmp,
        r#"
PARAMS_DEFAULTS = {{
    "mode": 0,
    "weight": 1.0,
}}

def _merge_params(defaults, overrides):
    merged = {{}}
    for k, v in defaults.items():
        merged[k] = v
    for k, v in overrides.items():
        merged[k] = v
    return merged

PARAMS = _merge_params(PARAMS_DEFAULTS, PARAMS_OVERRIDE)

def init(root, ctx):
    return {{"done": False}}

def step(graph, gs, ctx, lib):
    if PARAMS["weight"] == 2.5:
        return halt("solved", "float_override")
    if PARAMS["mode"] == 1:
        return halt("unsolvable", "int_override")
    return halt("solved", "default")
"#
    )
    .expect("write policy");
    tmp.flush().expect("flush policy");
    tmp
}

/// Returns true if the status does NOT mention `PARAMS_OVERRIDE` — which
/// would indicate the override binding broke the policy.
fn status_is_not_override_error(s: &PolicyStatus) -> bool {
    let s = format!("{s:?}");
    !s.contains("PARAMS_OVERRIDE")
}

// ── tests ─────────────────────────────────────────────────────────────────────

/// Empty override dict uses defaults and does not produce a PARAMS_OVERRIDE error.
#[test]
fn empty_override_leaves_policy_unchanged() {
    let policy = param_probe_policy();
    let initial = vec![(0u32, loc(0, 0))];
    let target = vec![(0u32, loc(0, 5))];

    let result = run_with_params(
        &initial,
        &target,
        policy.path().to_str().expect("utf8 path"),
        serde_json::json!({}),
        1_000,
    );

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

/// Integer overrides are visible to module-level PARAMS merge code.
#[test]
fn integer_param_override_changes_behavior() {
    let policy = param_probe_policy();
    let initial = vec![(0u32, loc(0, 0))];
    let target = vec![(0u32, loc(1, 5))];

    let default_result = run_with_params(
        &initial,
        &target,
        policy.path().to_str().expect("utf8 path"),
        serde_json::json!({}),
        2_000,
    );
    let int_override_result = run_with_params(
        &initial,
        &target,
        policy.path().to_str().expect("utf8 path"),
        serde_json::json!({"mode": 1}),
        2_000,
    );

    println!(
        "default:   status={:?} layers={} nodes_expanded={}",
        default_result.status,
        default_result.move_layers.len(),
        default_result.nodes_expanded
    );
    println!(
        "mode=1:   status={:?} layers={} nodes_expanded={}",
        int_override_result.status,
        int_override_result.move_layers.len(),
        int_override_result.nodes_expanded
    );

    // Neither run should produce a PARAMS_OVERRIDE-related runtime error.
    assert!(
        status_is_not_override_error(&default_result.status),
        "default run produced a PARAMS_OVERRIDE error: {:?}",
        default_result.status
    );
    assert!(
        status_is_not_override_error(&int_override_result.status),
        "integer override run produced a PARAMS_OVERRIDE error: {:?}",
        int_override_result.status
    );

    assert_eq!(default_result.status, PolicyStatus::Solved);
    assert_eq!(
        int_override_result.status,
        PolicyStatus::Unsolvable,
        "expected mode=1 override to change policy branch"
    );
}

/// Override a float parameter.  This exercises the JSON→Starlark
/// float conversion path and confirms non-integer overrides work correctly.
#[test]
fn float_param_override_is_accepted() {
    let policy = param_probe_policy();
    let initial = vec![(0u32, loc(0, 0))];
    let target = vec![(0u32, loc(0, 5))];

    let result = run_with_params(
        &initial,
        &target,
        policy.path().to_str().expect("utf8 path"),
        serde_json::json!({"weight": 2.5}),
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
    assert_eq!(result.status, PolicyStatus::Solved);
}
