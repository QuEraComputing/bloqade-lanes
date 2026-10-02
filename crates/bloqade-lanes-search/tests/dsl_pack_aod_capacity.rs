//! `lib.pack_aod_rectangles` honours the architecture's AOD capacity, as
//! every other shot assembler does.

use std::sync::Arc;

use bloqade_lanes_bytecode_core::arch::AodCapacity;
use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;
use bloqade_lanes_bytecode_core::arch::types::ArchSpec;
use bloqade_lanes_dsl_core::sandbox::SandboxConfig;
use bloqade_lanes_search::dsl::move_policy_dsl::{PolicyOptions, PolicyStatus, solve_with_policy};
use bloqade_lanes_search::primitives::lane_index::LaneIndex;

/// The two-word example arch: one site bus maps sites 0..5 to 5..10.
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

fn loc(word_id: u32, site_id: u32) -> LocationAddr {
    LocationAddr {
        zone_id: 0,
        word_id,
        site_id,
    }
}

/// The widest shot the bundled A* policy returns for moving one row of three
/// atoms along the site bus, under `capacity`.
fn widest_shot(capacity: Option<AodCapacity>) -> usize {
    let spec: ArchSpec = serde_json::from_str(example_arch_json()).unwrap();
    let index = Arc::new(LaneIndex::new(spec.with_aod_capacity(capacity)));
    let policy = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../policies/autotune/candidate.star"
    );
    let opts = PolicyOptions {
        policy_path: policy.to_owned(),
        policy_params: serde_json::Value::Object(Default::default()),
        max_expansions: 1000,
        timeout_s: Some(30.0),
        sandbox: SandboxConfig::default(),
    };
    let result = solve_with_policy(
        (0..3).map(|i| (i, loc(0, i))),
        (0..3).map(|i| (i, loc(0, i + 5))),
        std::iter::empty(),
        index,
        opts,
        &mut bloqade_lanes_search::dsl::move_policy_dsl::NoOpMoveObserver,
    )
    .expect("solve");
    assert!(
        matches!(result.status, PolicyStatus::Solved),
        "{:?}",
        result.status
    );
    result
        .move_layers
        .iter()
        .map(|layer| layer.decode().len())
        .max()
        .unwrap_or(0)
}

#[test]
fn pack_aod_rectangles_honours_the_aod_capacity() {
    assert!(
        widest_shot(None) > 1,
        "uncapped, the policy should move atoms together"
    );
    assert_eq!(
        widest_shot(AodCapacity::new(1, 1)),
        1,
        "a 1x1 AOD cannot carry more than one atom per shot"
    );
}
