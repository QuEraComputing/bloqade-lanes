//! Shared test utilities for the search crate.

use bloqade_lanes_bytecode_core::arch::addr::{Direction, LaneAddr, LocationAddr, MoveType};

/// Create a [`LocationAddr`] from word and site IDs.
pub fn loc(word: u32, site: u32) -> LocationAddr {
    LocationAddr {
        zone_id: 0,
        word_id: word,
        site_id: site,
    }
}

/// Create a forward site-bus [`LaneAddr`] with specified word, site, and bus.
pub fn lane(word: u32, site: u32, bus: u32) -> LaneAddr {
    LaneAddr {
        direction: Direction::Forward,
        move_type: MoveType::SiteBus,
        zone_id: 0,
        word_id: word,
        site_id: site,
        bus_id: bus,
    }
}

/// Full three-word, two-zone architecture JSON for tests.
///
/// Loaded from `examples/arch/full.json`. Zone 0 has entangling pair [0, 1]
/// with 9 site buses and 1 word bus. Zone 1 is storage-only (no buses).
///
/// **P1 negative fixture.** Words 0–2 sit at identical physical coordinates
/// (they share site index pairs and there is one grid per zone), so distinct
/// lane sources in one bus group share a position. The spec passes
/// `validate()`, yet "cells on the source grid" is not well defined on it:
/// position-keyed maps keep one lane per cell, and the exhaustive generator's
/// precondition check must reject it before any enumeration runs.
#[allow(dead_code)]
pub fn full_arch_json() -> &'static str {
    include_str!("../../../examples/arch/full.json")
}

/// The example arch with site bus 0 rewired as a **conveyor chain**:
/// `0→1, 1→2, 2→3, 3→4` along one row of a word.
///
/// The destination set `{1,2,3,4}` overlaps the source set `{0,1,2,3}`, but
/// the relation is acyclic and endpoint-unique, so this is a legal bus
/// (issue #874) — and the only kind of spec on which the chain-handling paths
/// of the generators and the AOD grid layer are reachable (issue #866). The
/// shipped Gemini specs keep their endpoints disjoint, where a destination is
/// never a source of the same bus and the chain code is dead.
#[allow(dead_code)]
pub fn chain_arch_json() -> String {
    use bloqade_lanes_bytecode_core::arch::addr::SiteRef;
    use bloqade_lanes_bytecode_core::arch::types::ArchSpec;

    let mut spec: ArchSpec =
        serde_json::from_str(example_arch_json()).expect("example arch json parses");
    let bus = &mut spec.zones[0].site_buses[0];
    bus.src = (0..4).map(SiteRef).collect();
    bus.dst = (1..5).map(SiteRef).collect();
    let validation = spec.validate();
    assert!(
        validation.is_ok(),
        "conveyor-chain fixture must be a legal spec: {validation:?}"
    );
    serde_json::to_string(&spec).expect("spec serializes")
}

/// A conveyor chain with a **siding**: two words of three sites each, a chain
/// site bus (`0→1, 1→2`) in both, and a word bus joining them at every site.
///
/// The smallest spec on which a chain can be *admitted* at selection time and
/// still fail to assemble. Site `2` is a chain destination but not a chain
/// source, so an atom parked there cannot vacate on that bus group — a run of
/// three atoms therefore has a blocked head, every rectangle on the chain bus is
/// rejected, and the group emits nothing even though the leader's move scored
/// positive. The word bus is what makes the instance solvable anyway, and is the
/// escape route the generator has to fall back on (#910).
#[allow(dead_code)]
pub fn chain_with_siding_arch_json() -> &'static str {
    r#"{
        "version": "2.0",
        "words": [
            { "sites": [[0, 0], [1, 0], [2, 0]] },
            { "sites": [[0, 1], [1, 1], [2, 1]] }
        ],
        "zones": [
            {
                "grid": { "x_start": 0.0, "y_start": 0.0, "x_spacing": [2.0, 2.0], "y_spacing": [2.0] },
                "site_buses": [
                    { "src": [0, 1], "dst": [1, 2] }
                ],
                "word_buses": [
                    { "src": [0], "dst": [1] }
                ],
                "words_with_site_buses": [0, 1],
                "sites_with_word_buses": [0, 1, 2],
                "entangling_pairs": []
            }
        ],
        "zone_buses": [],
        "modes": [
            { "name": "default", "zones": [0], "bitstring_order": [] }
        ]
    }"#
}

/// Minimal two-zone architecture with a single inter-zone `zone_bus`.
///
/// Zone 0 ("gate") holds word 0 and zone 1 ("memory") holds word 1, each a
/// single-site word. A one-to-one `zone_bus` connects (zone 1, word 1) ->
/// (zone 0, word 0). There are no intra-zone buses, so the only edges in the
/// search graph come from the zone bus — making this arch a focused regression
/// fixture for inter-zone graph construction (issue #845).
#[allow(dead_code)]
pub fn two_zone_bus_arch_json() -> &'static str {
    r#"{
        "version": "2.0",
        "words": [
            { "sites": [[0, 0]] },
            { "sites": [[0, 0]] }
        ],
        "zones": [
            {
                "name": "gate",
                "grid": { "x_start": 0.0, "y_start": 0.0, "x_spacing": [], "y_spacing": [] },
                "site_buses": [],
                "word_buses": [],
                "words_with_site_buses": [],
                "sites_with_word_buses": [],
                "entangling_pairs": []
            },
            {
                "name": "memory",
                "grid": { "x_start": 0.0, "y_start": 10.0, "x_spacing": [], "y_spacing": [] },
                "site_buses": [],
                "word_buses": [],
                "words_with_site_buses": [],
                "sites_with_word_buses": [],
                "entangling_pairs": []
            }
        ],
        "zone_buses": [
            {
                "src": [{ "zone_id": 1, "word_id": 1 }],
                "dst": [{ "zone_id": 0, "word_id": 0 }]
            }
        ],
        "modes": [
            { "name": "default", "zones": [0, 1], "bitstring_order": [] }
        ]
    }"#
}

/// Two zones with **aligned columns** and one site bus each: the fixture on
/// which the one-bus-group rule (S3) can actually fire.
///
/// Both zones have a 2×2 grid with the same x positions (0, 2); zone 0 spans
/// y ∈ {0, 2} and zone 1 spans y ∈ {10, 12}, so their bounding boxes do not
/// overlap. Zone 0 owns word 0 and zone 1 owns word 1; each word's four sites
/// are the full 2×2 index grid, and each zone's site bus 0 lifts the bottom row
/// onto the top row (`0→2, 1→3`). The two buses share `bus_id = 0`, so a
/// rectangle built across zones — lanes `(zone 0, word 0, sites 0,1)` with
/// `(zone 1, word 1, sites 0,1)` — has source positions `{0,2} × {0,10}`, a
/// complete grid that passes the geometry check (S5) while mixing zones (S3).
/// A generator that groups on `(move_type, bus_id, direction)` across zones
/// emits exactly such shots.
#[allow(dead_code)]
pub fn two_zone_aligned_site_bus_arch_json() -> &'static str {
    r#"{
        "version": "2.0",
        "words": [
            { "sites": [[0, 0], [1, 0], [0, 1], [1, 1]] },
            { "sites": [[0, 0], [1, 0], [0, 1], [1, 1]] }
        ],
        "zones": [
            {
                "name": "lower",
                "grid": { "x_start": 0.0, "y_start": 0.0, "x_spacing": [2.0], "y_spacing": [2.0] },
                "site_buses": [
                    { "src": [0, 1], "dst": [2, 3] }
                ],
                "word_buses": [],
                "words_with_site_buses": [0],
                "sites_with_word_buses": [],
                "entangling_pairs": []
            },
            {
                "name": "upper",
                "grid": { "x_start": 0.0, "y_start": 10.0, "x_spacing": [2.0], "y_spacing": [2.0] },
                "site_buses": [
                    { "src": [0, 1], "dst": [2, 3] }
                ],
                "word_buses": [],
                "words_with_site_buses": [1],
                "sites_with_word_buses": [],
                "entangling_pairs": []
            }
        ],
        "zone_buses": [],
        "modes": [
            { "name": "default", "zones": [0, 1], "bitstring_order": [] }
        ]
    }"#
}

/// One site bus whose source→destination position map is **not separable**:
/// the P2 negative fixture.
///
/// One zone, a 4×2 grid (x ∈ {0,1,2,3}, y ∈ {0,1}), one word of eight sites.
/// Sites 0–3 are the source block `{0,1} × {0,1}` and sites 4–7 the
/// destination block `{2,3} × {0,1}`, but the bus pairs them with a twist:
/// `(0,0)→(2,0), (1,0)→(3,1), (0,1)→(2,1), (1,1)→(3,0)`. The full source and
/// destination sets are both rectangles, so build-time validation accepts the
/// spec, yet the source row `{(0,0),(1,0)}` maps to `{(2,0),(3,1)}`, which is
/// not a rectangle. A shot's sources can therefore form a complete grid while
/// its lane-address locations — what the validator's geometry check looks at
/// for a backward lane — do not.
#[allow(dead_code)]
pub fn non_separable_bus_arch_json() -> &'static str {
    r#"{
        "version": "2.0",
        "words": [
            { "sites": [[0, 0], [1, 0], [0, 1], [1, 1], [2, 0], [3, 1], [2, 1], [3, 0]] }
        ],
        "zones": [
            {
                "grid": { "x_start": 0.0, "y_start": 0.0, "x_spacing": [1.0, 1.0, 1.0], "y_spacing": [1.0] },
                "site_buses": [
                    { "src": [0, 1, 2, 3], "dst": [4, 5, 6, 7] }
                ],
                "word_buses": [],
                "words_with_site_buses": [0],
                "sites_with_word_buses": [],
                "entangling_pairs": []
            }
        ],
        "zone_buses": [],
        "modes": [
            { "name": "default", "zones": [0], "bitstring_order": [] }
        ]
    }"#
}

/// The forward site-bus lane `word 0, site 0 → site 5` on the example arch and
/// its reverse: the pair whose durations [`asymmetric_duration_arch_json`]
/// makes unequal.
#[allow(dead_code)]
pub fn asymmetric_duration_lane_pair() -> (LaneAddr, LaneAddr) {
    let forward = lane(0, 0, 0);
    let backward = LaneAddr {
        direction: Direction::Backward,
        ..forward
    };
    (forward, backward)
}

/// The example arch with transport paths under which one lane and its reverse
/// take **different** times.
///
/// The forward lane `word 0, site 0 → site 5` is a straight 2.5 µm hop from
/// `(1.0, 2.5)` to `(1.0, 5.0)`; its backward twin detours through `x = 6.0`
/// for 12.5 µm of travel. No other lane carries path data. Anything that
/// assumes a plan and its mirror image cost the same — the backwards-search
/// cost carry-over, a direction-symmetric objective — has to be tested on a
/// spec where they do not.
#[allow(dead_code)]
pub fn asymmetric_duration_arch_json() -> String {
    use bloqade_lanes_bytecode_core::arch::types::{ArchSpec, TransportPath};

    let mut spec: ArchSpec =
        serde_json::from_str(example_arch_json()).expect("example arch json parses");
    let (forward, backward) = asymmetric_duration_lane_pair();
    spec.paths = Some(vec![
        TransportPath {
            lane: forward.encode_u64(),
            waypoints: vec![[1.0, 2.5], [1.0, 5.0]],
        },
        TransportPath {
            lane: backward.encode_u64(),
            waypoints: vec![[1.0, 5.0], [6.0, 5.0], [6.0, 2.5], [1.0, 2.5]],
        },
    ]);
    let validation = spec.validate();
    assert!(
        validation.is_ok(),
        "asymmetric-duration fixture must be a legal spec: {validation:?}"
    );
    serde_json::to_string(&spec).expect("spec serializes")
}

/// Example two-word architecture JSON for tests.
///
/// Zone-centric schema: words at top level, zones own grids and buses.
///
/// Grid has 5 x-positions (x=1,3,5,7,9) and 4 y-positions (y=2.5,5.0,12.5,15.0).
/// Word 0 uses y-indices 0,1 (y=2.5,5.0); word 1 uses y-indices 2,3 (y=12.5,15.0).
/// Each word has 10 sites (5 source + 5 destination via site bus).
pub fn example_arch_json() -> &'static str {
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use bloqade_lanes_bytecode_core::arch::query::LaneGroupError;
    use bloqade_lanes_bytecode_core::arch::types::ArchSpec;

    use super::*;
    use crate::primitives::lane_index::LaneIndex;

    fn parse(json: &str) -> ArchSpec {
        serde_json::from_str(json).expect("fixture json parses")
    }

    /// Every fixture is a spec `validate()` accepts — including the two
    /// negatives, whose whole point is that build-time validation lets them
    /// through.
    #[test]
    fn every_fixture_is_a_valid_spec() {
        let fixtures: [(&str, String); 7] = [
            ("example", example_arch_json().to_string()),
            ("full (P1 negative)", full_arch_json().to_string()),
            ("chain", chain_arch_json()),
            (
                "chain with siding",
                chain_with_siding_arch_json().to_string(),
            ),
            ("two-zone bus", two_zone_bus_arch_json().to_string()),
            (
                "two-zone aligned site buses",
                two_zone_aligned_site_bus_arch_json().to_string(),
            ),
            (
                "non-separable bus (P2 negative)",
                non_separable_bus_arch_json().to_string(),
            ),
        ];
        for (name, json) in &fixtures {
            let spec = parse(json);
            assert!(
                spec.validate().is_ok(),
                "{name}: {:?}",
                spec.validate().err()
            );
        }
        // Built with its own validation assert; parse it once more here so a
        // serialization round-trip is covered too.
        assert!(parse(&asymmetric_duration_arch_json()).validate().is_ok());
    }

    /// The cross-zone lane set is a complete grid on positions (S5 passes)
    /// and mixes zones (S3 fails): exactly the shot the fixture exists to
    /// make expressible.
    #[test]
    fn two_zone_aligned_fixture_lets_s3_fire() {
        let spec = parse(two_zone_aligned_site_bus_arch_json());
        let cross_zone: Vec<LaneAddr> = [(0u32, 0u32), (1, 1)]
            .into_iter()
            .flat_map(|(zone_id, word_id)| {
                (0..2u32).map(move |site_id| LaneAddr {
                    direction: Direction::Forward,
                    move_type: MoveType::SiteBus,
                    zone_id,
                    word_id,
                    site_id,
                    bus_id: 0,
                })
            })
            .collect();
        for lane in &cross_zone {
            assert!(spec.check_lane(lane).is_empty(), "{lane:?} must be a lane");
        }
        assert!(
            spec.check_lane_group_geometry(&cross_zone).is_empty(),
            "the four sources sit at {{0,2}} × {{0,10}}, a complete grid"
        );
        let errors = spec.check_lanes(&cross_zone);
        assert!(
            errors.iter().any(|e| matches!(
                e,
                LaneGroupError::Inconsistent { message } if message.contains("zone_id mismatch")
            )),
            "check_lanes must reject the zone mix, got {errors:?}"
        );
        // Both zones' buses are registered under the same bus id, so a
        // zone-blind grouping sees them as one group.
        let index = LaneIndex::new(spec);
        assert_eq!(
            index
                .lanes_for_all_zones(MoveType::SiteBus, 0, Direction::Forward)
                .count(),
            4
        );
    }

    /// The bus carries rectangles to non-rectangles: the full source set is a
    /// grid, and so is the full destination set, but a source row maps to a
    /// diagonal pair.
    #[test]
    fn non_separable_fixture_breaks_p2() {
        let spec = parse(non_separable_bus_arch_json());
        let forward: Vec<LaneAddr> = (0..4).map(|site| lane(0, site, 0)).collect();
        assert!(spec.check_lane_group_geometry(&forward).is_empty());

        let dst_positions = |lanes: &[LaneAddr]| -> (BTreeSet<u64>, BTreeSet<u64>, usize) {
            let dsts: Vec<(f64, f64)> = lanes
                .iter()
                .map(|l| {
                    let (_, dst) = spec.lane_endpoints(l).expect("lane resolves");
                    spec.location_position(&dst)
                        .expect("destination has a position")
                })
                .collect();
            let xs: BTreeSet<u64> = dsts.iter().map(|p| p.0.to_bits()).collect();
            let ys: BTreeSet<u64> = dsts.iter().map(|p| p.1.to_bits()).collect();
            (xs, ys, dsts.len())
        };
        // Full set: destinations form the 2×2 block {2,3} × {0,1}.
        let (xs, ys, n) = dst_positions(&forward);
        assert_eq!((xs.len(), ys.len(), n), (2, 2, 4));
        // Bottom source row: two destinations spanning two x and two y — not
        // a 2×2 grid, so not a rectangle.
        let (xs, ys, n) = dst_positions(&forward[..2]);
        assert_eq!((xs.len(), ys.len(), n), (2, 2, 2));
    }

    #[test]
    fn asymmetric_fixture_prices_a_lane_and_its_reverse_differently() {
        let index = LaneIndex::new(parse(&asymmetric_duration_arch_json()));
        let (forward, backward) = asymmetric_duration_lane_pair();
        let fwd = index
            .lane_duration_us(&forward)
            .expect("forward lane has a path");
        let bwd = index
            .lane_duration_us(&backward)
            .expect("backward lane has a path");
        assert!(
            fwd > 0.0 && bwd > fwd,
            "forward {fwd} µs, backward {bwd} µs"
        );
        // The plain example arch prices neither.
        let plain = LaneIndex::new(parse(example_arch_json()));
        assert_eq!(plain.lane_duration_us(&forward), None);
    }
}
