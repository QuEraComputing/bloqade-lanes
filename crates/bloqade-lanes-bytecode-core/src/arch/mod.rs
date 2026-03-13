pub mod addr;
pub mod query;
pub mod types;
pub mod validate;

pub use addr::{Direction, LaneAddr, LocationAddr, MoveType, ZoneAddr};
pub use query::ArchSpecLoadError;
pub use types::{ArchSpec, Bus, Buses, Geometry, Grid, TransportPath, Word, Zone};
pub use validate::ArchSpecError;

/// Example arch spec from the bytecode design doc, for use in tests.
#[cfg(test)]
pub(crate) fn example_arch_spec() -> ArchSpec {
    let json = r#"{
        "version": 1,
        "geometry": {
            "sites_per_word": 10,
            "words": [
                {
                    "grid": { "x_start": 1.0, "y_start": 2.5, "x_spacing": [2.0, 2.0, 2.0, 2.0], "y_spacing": [2.5] },
                    "sites": [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [0, 1], [1, 1], [2, 1], [3, 1], [4, 1]],
                    "cz_pairs": [[0, 5], [0, 6], [0, 7], [0, 8], [0, 9], [0, 0], [0, 1], [0, 2], [0, 3], [0, 4]]
                },
                {
                    "grid": { "x_start": 1.0, "y_start": 12.5, "x_spacing": [2.0, 2.0, 2.0, 2.0], "y_spacing": [2.5] },
                    "sites": [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [0, 1], [1, 1], [2, 1], [3, 1], [4, 1]],
                    "cz_pairs": [[1, 5], [1, 6], [1, 7], [1, 8], [1, 9], [1, 0], [1, 1], [1, 2], [1, 3], [1, 4]]
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
        "entangling_zones": [0],
        "measurement_mode_zones": [0],
        "paths": [
            {"lane": "0xC000000000010000", "waypoints": [[1.0, 12.5], [1.0, 7.5], [1.0, 2.5]]}
        ]
    }"#;
    serde_json::from_str(json).unwrap()
}
