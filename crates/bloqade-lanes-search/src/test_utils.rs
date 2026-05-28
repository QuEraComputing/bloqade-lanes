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

/// Create a forward site-bus [`LaneAddr`] with the given bus_id.
pub fn dummy_lane(id: u32) -> LaneAddr {
    LaneAddr {
        direction: Direction::Forward,
        move_type: MoveType::SiteBus,
        zone_id: 0,
        word_id: 0,
        site_id: id,
        bus_id: 0,
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
#[allow(dead_code)]
pub fn full_arch_json() -> &'static str {
    include_str!("../../../examples/arch/full.json")
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
