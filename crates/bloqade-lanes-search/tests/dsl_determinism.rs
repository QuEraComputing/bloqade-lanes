//! Determinism property test: running `entropy.star` twice on the same
//! fixture produces byte-identical SolveResults. Validates spec §8.

use std::path::PathBuf;
use std::sync::Arc;

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;
use bloqade_lanes_dsl_core::sandbox::SandboxConfig;
use bloqade_lanes_search::LaneIndex;
use bloqade_lanes_search::move_policy_dsl::{PolicyOptions, PolicyResult, solve_with_policy};

// ── helpers ──────────────────────────────────────────────────────────────────

/// Convenience constructor for `LocationAddr` (zone 0, word `w`, site `s`).
fn loc(word: u32, site: u32) -> LocationAddr {
    LocationAddr {
        zone_id: 0,
        word_id: word,
        site_id: site,
    }
}

/// Path to the reference `entropy.star` policy relative to the repo root.
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

/// Example two-word architecture JSON (same as in dsl_entropy_acid.rs).
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

/// Run `entropy.star` on a fixture and return the result.
fn run_entropy_star(
    arch_json: &str,
    initial: &[(u32, LocationAddr)],
    target: &[(u32, LocationAddr)],
    blocked: &[LocationAddr],
) -> PolicyResult {
    let spec = serde_json::from_str(arch_json).expect("arch parse");
    let index = Arc::new(LaneIndex::new(spec));
    let opts = PolicyOptions {
        policy_path: entropy_star_path(),
        policy_params: serde_json::json!({}),
        max_expansions: 1_000,
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

// ── Canonical form for determinism comparison ────────────────────────────────

/// Canonical-form representation of a PolicyResult for byte-identical
/// comparison across runs. This struct isolates the fields that should
/// be deterministic (status, structure, expansion count) and normalizes
/// them for comparison.
#[derive(Debug, PartialEq, Eq)]
struct CanonicalResult {
    status: String,
    n_layers: usize,
    move_lanes: Vec<Vec<u64>>,
    goal_config: Vec<(u32, u64)>,
    nodes_expanded: u32,
}

impl From<PolicyResult> for CanonicalResult {
    fn from(p: PolicyResult) -> Self {
        // Extract encoded lanes from each move layer.
        let move_lanes: Vec<Vec<u64>> = p
            .move_layers
            .iter()
            .map(|ms| ms.encoded_lanes().to_vec())
            .collect();

        // Normalize goal_config: encode all LocationAddrs and sort by qubit ID.
        let mut goal_config: Vec<(u32, u64)> =
            p.goal_config.iter().map(|(q, l)| (q, l.encode())).collect();
        goal_config.sort_by_key(|&(q, _)| q);

        Self {
            status: format!("{:?}", p.status),
            n_layers: p.move_layers.len(),
            move_lanes,
            goal_config,
            nodes_expanded: p.nodes_expanded,
        }
    }
}

// ── Fixture builders (copied from dsl_entropy_acid.rs) ───────────────────────

type Fixture = (
    &'static str,
    Vec<(u32, LocationAddr)>,
    Vec<(u32, LocationAddr)>,
    Vec<LocationAddr>,
);

/// Fixture 1: tiny — 1 qubit, 1 site-bus hop.
///
/// Qubit 0 starts at word 0, site 0 (source row).
/// Target: word 0, site 5 (destination row).
/// One site-bus move is required.
fn build_tiny_fixture() -> Fixture {
    (
        example_arch_json(),
        vec![(0u32, loc(0, 0))],
        vec![(0u32, loc(0, 5))],
        vec![],
    )
}

/// Fixture 2: small — 1 qubit, 2 bus hops (cross-word).
///
/// Qubit 0 starts at word 0, site 0.
/// Target: word 1, site 5.
/// Path: site-bus (site 0 → site 5 in word 0), then word-bus (word 0 → word 1).
/// Two bus hops are required.
fn build_small_fixture() -> Fixture {
    (
        example_arch_json(),
        vec![(0u32, loc(0, 0))],
        vec![(0u32, loc(1, 5))],
        vec![],
    )
}

/// Fixture 3: medium — 2 qubits, parallel moves.
///
/// Qubit 0 at word 0, site 0.  Target: word 0, site 5.
/// Qubit 1 at word 0, site 1.  Target: word 0, site 6.
fn build_medium_fixture() -> Fixture {
    (
        example_arch_json(),
        vec![(0u32, loc(0, 0)), (1u32, loc(0, 1))],
        vec![(0u32, loc(0, 5)), (1u32, loc(0, 6))],
        vec![],
    )
}

// ── Tests ────────────────────────────────────────────────────────────────────

/// Test: entropy.star is deterministic across runs on the tiny fixture.
#[test]
fn entropy_star_is_deterministic_tiny() {
    let (arch_json, initial, target, blocked) = build_tiny_fixture();

    let r1 = CanonicalResult::from(run_entropy_star(arch_json, &initial, &target, &blocked));
    let r2 = CanonicalResult::from(run_entropy_star(arch_json, &initial, &target, &blocked));

    assert_eq!(
        r1, r2,
        "entropy.star must be deterministic across runs (tiny fixture)"
    );
    println!("✓ Two runs produced byte-identical CanonicalResult (tiny): {r1:?}");
}

/// Test: entropy.star is deterministic across runs on the small fixture.
#[test]
fn entropy_star_is_deterministic_small() {
    let (arch_json, initial, target, blocked) = build_small_fixture();

    let r1 = CanonicalResult::from(run_entropy_star(arch_json, &initial, &target, &blocked));
    let r2 = CanonicalResult::from(run_entropy_star(arch_json, &initial, &target, &blocked));

    assert_eq!(
        r1, r2,
        "entropy.star must be deterministic across runs (small fixture)"
    );
    println!("✓ Two runs produced byte-identical CanonicalResult (small): {r1:?}");
}

/// Test: entropy.star is deterministic across runs on the medium fixture.
#[test]
fn entropy_star_is_deterministic_medium() {
    let (arch_json, initial, target, blocked) = build_medium_fixture();

    let r1 = CanonicalResult::from(run_entropy_star(arch_json, &initial, &target, &blocked));
    let r2 = CanonicalResult::from(run_entropy_star(arch_json, &initial, &target, &blocked));

    assert_eq!(
        r1, r2,
        "entropy.star must be deterministic across runs (medium fixture)"
    );
    println!("✓ Two runs produced byte-identical CanonicalResult (medium): {r1:?}");
}
