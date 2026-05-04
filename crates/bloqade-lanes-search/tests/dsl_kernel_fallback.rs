//! `invoke_builtin("sequential_fallback") + halt("fallback")` produces
//! a `Fallback` status.

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

#[test]
fn invoke_sequential_fallback_then_halt_returns_fallback_status() {
    let mut tmp = NamedTempFile::new().expect("temp file");
    writeln!(
        tmp,
        r#"
def init(root, ctx):
    return None

def step(graph, gs, ctx, lib):
    return [
        invoke_builtin("sequential_fallback", {{}}),
        halt("fallback", "test"),
    ]
"#
    )
    .expect("write");
    tmp.flush().expect("flush");
    let path = tmp.path().to_str().expect("path").to_owned();

    let spec: ArchSpec = serde_json::from_str(example_arch_json()).unwrap();
    let index = Arc::new(LaneIndex::new(spec));

    let opts = PolicyOptions {
        policy_path: path,
        policy_params: serde_json::Value::Object(Default::default()),
        max_expansions: 100,
        timeout_s: Some(5.0),
        sandbox: SandboxConfig::default(),
    };

    let result = solve_with_policy(
        std::iter::empty::<(u32, bloqade_lanes_bytecode_core::arch::addr::LocationAddr)>(),
        std::iter::empty::<(u32, bloqade_lanes_bytecode_core::arch::addr::LocationAddr)>(),
        std::iter::empty::<bloqade_lanes_bytecode_core::arch::addr::LocationAddr>(),
        index,
        opts,
        &mut bloqade_lanes_search::move_policy_dsl::NoOpMoveObserver,
    )
    .expect("solve");

    match result.status {
        PolicyStatus::Fallback(_) => {}
        other => panic!("expected Fallback(...), got {other:?}"),
    }
    assert!(result.move_layers.is_empty());
}
