use bloqade_lanes_bytecode::arch::ArchSpec;

const EXAMPLE_JSON: &str = r#"{
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

#[test]
fn load_and_validate_example() {
    let spec = ArchSpec::from_json_validated(EXAMPLE_JSON).unwrap();
    assert_eq!(spec.version, 1);
    assert_eq!(spec.geometry.sites_per_word, 10);
    assert_eq!(spec.geometry.words.len(), 2);
    assert_eq!(spec.buses.site_buses.len(), 1);
    assert_eq!(spec.buses.word_buses.len(), 1);
    assert_eq!(spec.zones.len(), 1);
}

#[test]
fn query_methods_on_loaded_spec() {
    let spec = ArchSpec::from_json_validated(EXAMPLE_JSON).unwrap();

    let word = spec.word_by_id(0).unwrap();
    assert_eq!(word.site_position(0), Some((1.0, 2.5)));

    let bus = spec.site_bus_by_id(0).unwrap();
    assert_eq!(bus.resolve_forward(2), Some(7));

    let wbus = spec.word_bus_by_id(0).unwrap();
    assert_eq!(wbus.resolve_forward(0), Some(1));

    let zone = spec.zone_by_id(0).unwrap();
    assert_eq!(zone.words, vec![0, 1]);
}
