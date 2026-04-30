//! `update_node_state` uses an additive schema: new keys encountered in any
//! patch are added to the allowed set.  `update_global_state` uses a strict
//! schema seeded from `init()`'s return value.
//!
//! Note: strict closed-world node-state enforcement is deferred to Task 19.
//! The reference `entropy.star` policy uses different node-state fields
//! (`entropy`, `tried`) on different code paths, which requires additive
//! schema semantics to work correctly.

use std::io::Write;
use std::sync::Arc;

use bloqade_lanes_bytecode_core::arch::types::ArchSpec;
use bloqade_lanes_dsl_core::sandbox::SandboxConfig;
use bloqade_lanes_search::lane_index::LaneIndex;
use bloqade_lanes_search::move_policy_dsl::{PolicyOptions, PolicyStatus, solve_with_policy};
use tempfile::NamedTempFile;

/// Minimal two-word arch JSON reused from kernel unit tests.
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

/// Node-state patches with new keys are accepted (additive schema).
///
/// The reference entropy.star policy uses both "entropy" and "tried" keys
/// in node-state patches on different code paths.  This test verifies that
/// a policy which patches different fields across ticks is NOT rejected.
#[test]
fn node_state_additive_schema_allows_new_keys() {
    // First update establishes {"entropy": int}.
    // Second update adds "tried" — must be accepted under additive semantics.
    let mut tmp = NamedTempFile::new().expect("temp file");
    writeln!(
        tmp,
        r#"
def init(root, ctx):
    return {{"stage": 0}}

def step(graph, gs, ctx, lib):
    if gs["stage"] == 0:
        return [
            update_node_state(graph.root, {{"entropy": 1}}),
            update_global_state({{"stage": 1}}),
        ]
    if gs["stage"] == 1:
        # New key "tried" — additive schema must accept this.
        return [
            update_node_state(graph.root, {{"tried": []}}),
            update_global_state({{"stage": 2}}),
        ]
    return halt("solved", "done")
"#
    )
    .expect("write");
    tmp.flush().expect("flush");
    let path = tmp.path().to_str().expect("path").to_owned();

    let spec: ArchSpec = serde_json::from_str(example_arch_json()).expect("parse arch");
    let index = Arc::new(LaneIndex::new(spec));

    let opts = PolicyOptions {
        policy_path: path,
        policy_params: serde_json::Value::Object(Default::default()),
        max_expansions: 100,
        timeout_s: Some(5.0),
        sandbox: SandboxConfig::default(),
    };

    let result = solve_with_policy(
        std::iter::empty(),
        std::iter::empty(),
        std::iter::empty(),
        index,
        opts,
    )
    .expect("solve");

    assert_eq!(
        result.status,
        PolicyStatus::Solved,
        "expected Solved (additive node-state schema), got {:?}",
        result.status
    );
}

/// Global-state patches with keys not in init()'s return value are rejected.
///
/// The global-state schema is seeded from init()'s return value at solve
/// start.  Any subsequent patch that introduces a key not present in init()'s
/// dict must halt with SchemaError.
#[test]
fn global_state_unknown_field_halts_with_schema_error() {
    // init() returns {"stage": 0}. Schema = {"stage"}.
    // The second update introduces "unknown_field" — must fail.
    let mut tmp = NamedTempFile::new().expect("temp file");
    writeln!(
        tmp,
        r#"
def init(root, ctx):
    return {{"stage": 0}}

def step(graph, gs, ctx, lib):
    if gs["stage"] == 0:
        return update_global_state({{"stage": 1}})
    if gs["stage"] == 1:
        # "unknown_field" not in init() dict — should fail schema check
        return update_global_state({{"unknown_field": 99}})
    return halt("error", "unreachable")
"#
    )
    .expect("write");
    tmp.flush().expect("flush");
    let path = tmp.path().to_str().expect("path").to_owned();

    let spec: ArchSpec = serde_json::from_str(example_arch_json()).expect("parse arch");
    let index = Arc::new(LaneIndex::new(spec));

    let opts = PolicyOptions {
        policy_path: path,
        policy_params: serde_json::Value::Object(Default::default()),
        max_expansions: 100,
        timeout_s: Some(5.0),
        sandbox: SandboxConfig::default(),
    };

    let result = solve_with_policy(
        std::iter::empty(),
        std::iter::empty(),
        std::iter::empty(),
        index,
        opts,
    )
    .expect("solve");

    match result.status {
        PolicyStatus::SchemaError(field) => {
            assert_eq!(field, "unknown_field");
        }
        other => panic!("expected SchemaError(unknown_field), got {other:?}"),
    }
}
