//! After issuing an insert_child action, `graph.last_insert()` must
//! return a struct with the right shape on the *next* tick.

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
fn last_insert_reports_after_insert_child() {
    // A trivial policy that issues a single insert_child with an empty
    // moveset (empty list → zero lanes, which is aod_invalid because
    // there are no src qubits to resolve — but that still populates
    // last_insert). On the next tick the policy reads graph.last_insert()
    // and halts with "solved" if the side-channel is populated.

    let mut tmp = NamedTempFile::new().expect("temp file");
    writeln!(
        tmp,
        r#"
def init(root, ctx):
    return {{"stage": 0}}

def step(graph, gs, ctx, lib):
    if gs["stage"] == 0:
        return [
            insert_child(graph.root, []),
            update_global_state({{"stage": 1}}),
        ]
    if gs["stage"] == 1:
        outcome = graph.last_insert()
        if outcome == None:
            return halt("error", "no last_insert recorded")
        # Either is_new is True (success) or error is non-None (failure).
        # Both are acceptable: we just verify the side channel is populated.
        return halt("solved", "saw outcome")
    return halt("error", "unexpected stage")
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
        &mut bloqade_lanes_search::move_policy_dsl::NoOpMoveObserver,
    )
    .expect("solve");

    // The empty-moveset insert_child either becomes a no-op duplicate of
    // root (is_new=false) or aod_invalid (also recorded). Either way,
    // last_insert was populated and the policy reached the "saw outcome" branch.
    assert_eq!(
        result.status,
        PolicyStatus::Solved,
        "expected Solved (saw outcome), got {:?}",
        result.status
    );
}
