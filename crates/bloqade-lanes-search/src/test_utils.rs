//! Shared test utilities for the search crate.

use bloqade_lanes_bytecode_core::arch::addr::{Direction, LaneAddr, LocationAddr, MoveType};

/// Create a [`LocationAddr`] from word and site IDs.
pub fn loc(word: u32, site: u32) -> LocationAddr {
    LocationAddr {
        word_id: word,
        site_id: site,
    }
}

/// Create a forward site-bus [`LaneAddr`] with the given bus_id.
pub fn dummy_lane(id: u32) -> LaneAddr {
    LaneAddr {
        direction: Direction::Forward,
        move_type: MoveType::SiteBus,
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
        word_id: word,
        site_id: site,
        bus_id: bus,
    }
}

/// Example two-word architecture JSON for tests.
pub fn example_arch_json() -> &'static str {
    r#"{
        "version": "2.0",
        "geometry": {
            "sites_per_word": 10,
            "words": [
                {
                    "positions": { "x_start": 1.0, "y_start": 2.5, "x_spacing": [2.0, 2.0, 2.0, 2.0], "y_spacing": [2.5] },
                    "site_indices": [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [0, 1], [1, 1], [2, 1], [3, 1], [4, 1]]
                },
                {
                    "positions": { "x_start": 1.0, "y_start": 12.5, "x_spacing": [2.0, 2.0, 2.0, 2.0], "y_spacing": [2.5] },
                    "site_indices": [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [0, 1], [1, 1], [2, 1], [3, 1], [4, 1]]
                }
            ]
        },
        "buses": {
            "site_buses": [
                { "src": [0, 1, 2, 3, 4], "dst": [5, 6, 7, 8, 9] }
            ],
            "word_buses": [
                { "src": [0], "dst": [1] }
            ]
        },
        "words_with_site_buses": [0, 1],
        "sites_with_word_buses": [5, 6, 7, 8, 9],
        "zones": [
            { "words": [0, 1] }
        ],
        "entangling_zones": [[[0, 1]]],
        "blockade_radius": 2.0,
        "measurement_mode_zones": [0]
    }"#
}
