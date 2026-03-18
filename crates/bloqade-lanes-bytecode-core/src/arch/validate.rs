//! Structural validation for [`ArchSpec`].
//!
//! Validates all structural rules in a single pass, collecting every error
//! rather than failing fast. See [`ArchSpec::validate`].

use std::collections::HashSet;

use thiserror::Error;

use super::types::ArchSpec;

/// Error variants for arch spec structural validation.
///
/// Each variant corresponds to a specific validation rule. Multiple errors
/// can be collected in a single validation pass.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum ArchSpecError {
    #[error("zone 0 must include all words: missing word IDs {missing:?}")]
    Zone0MissingWords { missing: Vec<u32> },

    #[error("measurement_mode_zones must not be empty")]
    MeasurementModeZonesEmpty,

    #[error("measurement_mode_zones[0] must be zone 0, got {got}")]
    MeasurementModeFirstNotZone0 { got: u32 },

    #[error("entangling_zones contains invalid zone ID {id}")]
    InvalidEntanglingZone { id: u32 },

    #[error("measurement_mode_zones contains invalid zone ID {id}")]
    InvalidMeasurementModeZone { id: u32 },

    #[error("word {word_id} has {got} sites, expected {expected} (sites_per_word)")]
    WrongSiteCount {
        word_id: u32,
        expected: u32,
        got: usize,
    },

    #[error("word {word_id}, site {site_idx}: x_idx {x_idx} out of range (grid has num_x={x_len})")]
    SiteXIndexOutOfRange {
        word_id: u32,
        site_idx: usize,
        x_idx: u32,
        x_len: usize,
    },

    #[error("word {word_id}, site {site_idx}: y_idx {y_idx} out of range (grid has num_y={y_len})")]
    SiteYIndexOutOfRange {
        word_id: u32,
        site_idx: usize,
        y_idx: u32,
        y_len: usize,
    },

    #[error("site_bus {bus_id}: src length ({src_len}) != dst length ({dst_len})")]
    SiteBusLengthMismatch {
        bus_id: u32,
        src_len: usize,
        dst_len: usize,
    },

    #[error("site_bus {bus_id}: src and dst overlap at site index {site_idx}")]
    SiteBusSrcDstOverlap { bus_id: u32, site_idx: u32 },

    #[error("site_bus {bus_id}: site index {site_idx} >= sites_per_word ({sites_per_word})")]
    SiteBusIndexOutOfRange {
        bus_id: u32,
        site_idx: u32,
        sites_per_word: u32,
    },

    #[error("word_bus {bus_id}: src length ({src_len}) != dst length ({dst_len})")]
    WordBusLengthMismatch {
        bus_id: u32,
        src_len: usize,
        dst_len: usize,
    },

    #[error("word_bus {bus_id}: invalid word ID {word_id}")]
    WordBusInvalidWordId { bus_id: u32, word_id: u32 },

    #[error("words_with_site_buses: invalid word ID {word_id}")]
    InvalidWordWithSiteBus { word_id: u32 },

    #[error("sites_with_word_buses: site index {site_idx} >= sites_per_word ({sites_per_word})")]
    InvalidSiteWithWordBus { site_idx: u32, sites_per_word: u32 },

    #[error("word {word_id} has {got} cz_pairs, expected {expected} (sites_per_word)")]
    WrongCzPairsCount {
        word_id: u32,
        expected: u32,
        got: usize,
    },

    #[error(
        "word {word_id} grid shape ({x_len}x{y_len}) differs from word 0 ({ref_x_len}x{ref_y_len})"
    )]
    InconsistentGridShape {
        word_id: u32,
        x_len: usize,
        y_len: usize,
        ref_x_len: usize,
        ref_y_len: usize,
    },

    #[error("word {word_id} grid contains non-finite value in {field}")]
    NonFiniteGridValue { word_id: u32, field: &'static str },

    #[error("paths[{index}]: waypoint contains non-finite coordinate")]
    NonFiniteWaypoint { index: usize },

    #[error("paths[{index}]: lane 0x{lane:016X} is invalid: {message}")]
    InvalidPathLane {
        index: usize,
        lane: u64,
        message: String,
    },

    #[error("paths[{index}]: lane 0x{lane:016X} has {count} waypoint(s), minimum is 2")]
    PathTooFewWaypoints {
        index: usize,
        lane: u64,
        count: usize,
    },

    #[error(
        "paths[{index}]: lane 0x{lane:016X} {endpoint} waypoint ({got_x}, {got_y}) does not match expected position ({expected_x}, {expected_y})"
    )]
    PathEndpointMismatch {
        index: usize,
        lane: u64,
        endpoint: &'static str,
        expected_x: f64,
        expected_y: f64,
        got_x: f64,
        got_y: f64,
    },
}

impl ArchSpec {
    /// Validate the arch spec against all structural rules.
    /// Collects all errors in one pass (not fail-fast).
    pub fn validate(&self) -> Result<(), Vec<ArchSpecError>> {
        let mut errors = Vec::new();

        let num_words = self.geometry.words.len() as u32;
        let num_zones = self.zones.len() as u32;
        let sites_per_word = self.geometry.sites_per_word;

        // Rule 1: Zone 0 must include all words
        check_zone0_includes_all_words(self, num_words, &mut errors);

        // Rule 2: measurement_mode_zones[0] must be zone 0
        check_measurement_mode_first_is_zone0(self, &mut errors);

        // Rule 3a: All entangling_zones IDs must be valid zone indices
        check_entangling_zones_valid(self, num_zones, &mut errors);

        // Rule 3b: All measurement_mode_zones IDs must be valid zone indices
        check_measurement_mode_zones_valid(self, num_zones, &mut errors);

        // Rules 4, 5a, 5b: Word site counts and grid index bounds
        check_word_sites(self, &mut errors);

        // Rules 6, 7, 8: Site bus validation
        check_site_buses(self, sites_per_word, &mut errors);

        // Rules 9, 10: Word bus validation
        check_word_buses(self, num_words, &mut errors);

        // Rule: All words must have the same grid shape
        check_consistent_grid_shape(self, &mut errors);

        // Rule 11: words_with_site_buses must be valid word indices
        check_words_with_site_buses(self, num_words, &mut errors);

        // Rule 12: sites_with_word_buses must be valid site indices
        check_sites_with_word_buses(self, sites_per_word, &mut errors);

        // Rule: path lane addresses must be valid
        check_path_lanes(self, &mut errors);

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

fn check_zone0_includes_all_words(
    spec: &ArchSpec,
    num_words: u32,
    errors: &mut Vec<ArchSpecError>,
) {
    if let Some(zone0) = spec.zones.first() {
        let zone0_words: HashSet<u32> = zone0.words.iter().copied().collect();
        let all_word_ids: HashSet<u32> = (0..num_words).collect();
        let missing: Vec<u32> = all_word_ids.difference(&zone0_words).copied().collect();
        if !missing.is_empty() {
            errors.push(ArchSpecError::Zone0MissingWords { missing });
        }
    }
}

fn check_measurement_mode_first_is_zone0(spec: &ArchSpec, errors: &mut Vec<ArchSpecError>) {
    if spec.measurement_mode_zones.is_empty() {
        errors.push(ArchSpecError::MeasurementModeZonesEmpty);
        return;
    }
    if spec.measurement_mode_zones[0] != 0 {
        errors.push(ArchSpecError::MeasurementModeFirstNotZone0 {
            got: spec.measurement_mode_zones[0],
        });
    }
}

fn check_entangling_zones_valid(spec: &ArchSpec, num_zones: u32, errors: &mut Vec<ArchSpecError>) {
    for &id in &spec.entangling_zones {
        if id >= num_zones {
            errors.push(ArchSpecError::InvalidEntanglingZone { id });
        }
    }
}

fn check_measurement_mode_zones_valid(
    spec: &ArchSpec,
    num_zones: u32,
    errors: &mut Vec<ArchSpecError>,
) {
    for &id in &spec.measurement_mode_zones {
        if id >= num_zones {
            errors.push(ArchSpecError::InvalidMeasurementModeZone { id });
        }
    }
}

fn check_word_sites(spec: &ArchSpec, errors: &mut Vec<ArchSpecError>) {
    let sites_per_word = spec.geometry.sites_per_word;
    for (word_id, word) in spec.geometry.words.iter().enumerate() {
        let word_id = word_id as u32;
        if let Err(field) = word.positions.check_finite() {
            errors.push(ArchSpecError::NonFiniteGridValue { word_id, field });
        }
        if word.site_indices.len() != sites_per_word as usize {
            errors.push(ArchSpecError::WrongSiteCount {
                word_id,
                expected: sites_per_word,
                got: word.site_indices.len(),
            });
        }
        if let Some(cz) = &word.has_cz
            && cz.len() != sites_per_word as usize
        {
            errors.push(ArchSpecError::WrongCzPairsCount {
                word_id,
                expected: sites_per_word,
                got: cz.len(),
            });
        }
        let x_len = word.positions.num_x();
        let y_len = word.positions.num_y();
        for (site_idx, site) in word.site_indices.iter().enumerate() {
            let x_idx = site[0];
            let y_idx = site[1];
            if x_idx as usize >= x_len {
                errors.push(ArchSpecError::SiteXIndexOutOfRange {
                    word_id,
                    site_idx,
                    x_idx,
                    x_len,
                });
            }
            if y_idx as usize >= y_len {
                errors.push(ArchSpecError::SiteYIndexOutOfRange {
                    word_id,
                    site_idx,
                    y_idx,
                    y_len,
                });
            }
        }
    }
}

fn check_site_buses(spec: &ArchSpec, sites_per_word: u32, errors: &mut Vec<ArchSpecError>) {
    for (bus_id, bus) in spec.buses.site_buses.iter().enumerate() {
        let bus_id = bus_id as u32;
        if bus.src.len() != bus.dst.len() {
            errors.push(ArchSpecError::SiteBusLengthMismatch {
                bus_id,
                src_len: bus.src.len(),
                dst_len: bus.dst.len(),
            });
        }
        for &idx in bus.src.iter().chain(bus.dst.iter()) {
            if idx >= sites_per_word {
                errors.push(ArchSpecError::SiteBusIndexOutOfRange {
                    bus_id,
                    site_idx: idx,
                    sites_per_word,
                });
            }
        }
        let src_set: HashSet<u32> = bus.src.iter().copied().collect();
        for &idx in &bus.dst {
            if src_set.contains(&idx) {
                errors.push(ArchSpecError::SiteBusSrcDstOverlap {
                    bus_id,
                    site_idx: idx,
                });
            }
        }
    }
}

fn check_word_buses(spec: &ArchSpec, num_words: u32, errors: &mut Vec<ArchSpecError>) {
    for (bus_id, bus) in spec.buses.word_buses.iter().enumerate() {
        let bus_id = bus_id as u32;
        if bus.src.len() != bus.dst.len() {
            errors.push(ArchSpecError::WordBusLengthMismatch {
                bus_id,
                src_len: bus.src.len(),
                dst_len: bus.dst.len(),
            });
        }
        for &wid in bus.src.iter().chain(bus.dst.iter()) {
            if wid >= num_words {
                errors.push(ArchSpecError::WordBusInvalidWordId {
                    bus_id,
                    word_id: wid,
                });
            }
        }
    }
}

fn check_words_with_site_buses(spec: &ArchSpec, num_words: u32, errors: &mut Vec<ArchSpecError>) {
    for &wid in &spec.words_with_site_buses {
        if wid >= num_words {
            errors.push(ArchSpecError::InvalidWordWithSiteBus { word_id: wid });
        }
    }
}

fn check_consistent_grid_shape(spec: &ArchSpec, errors: &mut Vec<ArchSpecError>) {
    if let Some(first) = spec.geometry.words.first() {
        let ref_x_len = first.positions.num_x();
        let ref_y_len = first.positions.num_y();
        for (idx, word) in spec.geometry.words.iter().enumerate().skip(1) {
            let x_len = word.positions.num_x();
            let y_len = word.positions.num_y();
            if x_len != ref_x_len || y_len != ref_y_len {
                errors.push(ArchSpecError::InconsistentGridShape {
                    word_id: idx as u32,
                    x_len,
                    y_len,
                    ref_x_len,
                    ref_y_len,
                });
            }
        }
    }
}

fn check_path_lanes(spec: &ArchSpec, errors: &mut Vec<ArchSpecError>) {
    if let Some(paths) = &spec.paths {
        for (index, path) in paths.iter().enumerate() {
            if !path.check_finite() {
                errors.push(ArchSpecError::NonFiniteWaypoint { index });
            }
            let lane = crate::arch::addr::LaneAddr::decode_u64(path.lane);
            let lane_errors = spec.check_lane(&lane);
            for message in lane_errors {
                errors.push(ArchSpecError::InvalidPathLane {
                    index,
                    lane: path.lane,
                    message,
                });
            }

            // Check minimum waypoint count
            if path.waypoints.len() < 2 {
                errors.push(ArchSpecError::PathTooFewWaypoints {
                    index,
                    lane: path.lane,
                    count: path.waypoints.len(),
                });
                continue; // can't check endpoints with < 2 waypoints
            }

            // Check that first/last waypoints match the lane's physical endpoints
            if let Some((src_loc, dst_loc)) = spec.lane_endpoints(&lane) {
                if let Some(src_pos) = spec.location_position(&src_loc) {
                    let first = path.waypoints.first().unwrap();
                    if first[0] != src_pos.0 || first[1] != src_pos.1 {
                        errors.push(ArchSpecError::PathEndpointMismatch {
                            index,
                            lane: path.lane,
                            endpoint: "first",
                            expected_x: src_pos.0,
                            expected_y: src_pos.1,
                            got_x: first[0],
                            got_y: first[1],
                        });
                    }
                }
                if let Some(dst_pos) = spec.location_position(&dst_loc) {
                    let last = path.waypoints.last().unwrap();
                    if last[0] != dst_pos.0 || last[1] != dst_pos.1 {
                        errors.push(ArchSpecError::PathEndpointMismatch {
                            index,
                            lane: path.lane,
                            endpoint: "last",
                            expected_x: dst_pos.0,
                            expected_y: dst_pos.1,
                            got_x: last[0],
                            got_y: last[1],
                        });
                    }
                }
            }
        }
    }
}

fn check_sites_with_word_buses(
    spec: &ArchSpec,
    sites_per_word: u32,
    errors: &mut Vec<ArchSpecError>,
) {
    for &idx in &spec.sites_with_word_buses {
        if idx >= sites_per_word {
            errors.push(ArchSpecError::InvalidSiteWithWordBus {
                site_idx: idx,
                sites_per_word,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::arch::example_arch_spec;

    use super::*;

    fn has_error<F: Fn(&ArchSpecError) -> bool>(errors: &[ArchSpecError], predicate: F) -> bool {
        errors.iter().any(predicate)
    }

    #[test]
    fn valid_spec_passes() {
        let spec = example_arch_spec();
        assert!(spec.validate().is_ok());
    }

    #[test]
    fn test_zone0_missing_words() {
        let mut spec = example_arch_spec();
        spec.zones[0].words = vec![0]; // remove word 1
        let errors = spec.validate().unwrap_err();
        assert!(has_error(&errors, |e| matches!(
            e,
            ArchSpecError::Zone0MissingWords { .. }
        )));
    }

    #[test]
    fn test_measurement_mode_first_not_zone0() {
        let mut spec = example_arch_spec();
        spec.measurement_mode_zones = vec![1];
        let errors = spec.validate().unwrap_err();
        assert!(has_error(&errors, |e| matches!(
            e,
            ArchSpecError::MeasurementModeFirstNotZone0 { got: 1 }
        )));
    }

    #[test]
    fn test_invalid_entangling_zone() {
        let mut spec = example_arch_spec();
        spec.entangling_zones.push(99);
        let errors = spec.validate().unwrap_err();
        assert!(has_error(&errors, |e| matches!(
            e,
            ArchSpecError::InvalidEntanglingZone { id: 99 }
        )));
    }

    #[test]
    fn test_invalid_measurement_mode_zone() {
        let mut spec = example_arch_spec();
        spec.measurement_mode_zones.push(99);
        let errors = spec.validate().unwrap_err();
        assert!(has_error(&errors, |e| matches!(
            e,
            ArchSpecError::InvalidMeasurementModeZone { id: 99 }
        )));
    }

    #[test]
    fn test_wrong_site_count() {
        let mut spec = example_arch_spec();
        spec.geometry.words[0].site_indices.pop();
        let errors = spec.validate().unwrap_err();
        assert!(has_error(&errors, |e| matches!(
            e,
            ArchSpecError::WrongSiteCount {
                word_id: 0,
                expected: 10,
                got: 9
            }
        )));
    }

    #[test]
    fn test_wrong_cz_pairs_count() {
        let mut spec = example_arch_spec();
        spec.geometry.words[0].has_cz.as_mut().unwrap().pop();
        let errors = spec.validate().unwrap_err();
        assert!(has_error(&errors, |e| matches!(
            e,
            ArchSpecError::WrongCzPairsCount {
                word_id: 0,
                expected: 10,
                got: 9
            }
        )));
    }

    #[test]
    fn test_site_x_index_out_of_range() {
        let mut spec = example_arch_spec();
        spec.geometry.words[0].site_indices[0] = [99, 0];
        let errors = spec.validate().unwrap_err();
        assert!(has_error(&errors, |e| matches!(
            e,
            ArchSpecError::SiteXIndexOutOfRange {
                word_id: 0,
                x_idx: 99,
                ..
            }
        )));
    }

    #[test]
    fn test_site_y_index_out_of_range() {
        let mut spec = example_arch_spec();
        spec.geometry.words[0].site_indices[0] = [0, 99];
        let errors = spec.validate().unwrap_err();
        assert!(has_error(&errors, |e| matches!(
            e,
            ArchSpecError::SiteYIndexOutOfRange {
                word_id: 0,
                y_idx: 99,
                ..
            }
        )));
    }

    #[test]
    fn test_site_bus_length_mismatch() {
        let mut spec = example_arch_spec();
        spec.buses.site_buses[0].dst.pop();
        let errors = spec.validate().unwrap_err();
        assert!(has_error(&errors, |e| matches!(
            e,
            ArchSpecError::SiteBusLengthMismatch { bus_id: 0, .. }
        )));
    }

    #[test]
    fn test_site_bus_overlap() {
        let mut spec = example_arch_spec();
        spec.buses.site_buses[0].src = vec![0, 1];
        spec.buses.site_buses[0].dst = vec![0, 2];
        let errors = spec.validate().unwrap_err();
        assert!(has_error(&errors, |e| matches!(
            e,
            ArchSpecError::SiteBusSrcDstOverlap {
                bus_id: 0,
                site_idx: 0
            }
        )));
    }

    #[test]
    fn test_site_bus_index_out_of_range() {
        let mut spec = example_arch_spec();
        spec.buses.site_buses[0].src[0] = 99;
        let errors = spec.validate().unwrap_err();
        assert!(has_error(&errors, |e| matches!(
            e,
            ArchSpecError::SiteBusIndexOutOfRange {
                bus_id: 0,
                site_idx: 99,
                ..
            }
        )));
    }

    #[test]
    fn test_word_bus_length_mismatch() {
        let mut spec = example_arch_spec();
        spec.buses.word_buses[0].dst.pop();
        let errors = spec.validate().unwrap_err();
        assert!(has_error(&errors, |e| matches!(
            e,
            ArchSpecError::WordBusLengthMismatch { bus_id: 0, .. }
        )));
    }

    #[test]
    fn test_word_bus_invalid_word_id() {
        let mut spec = example_arch_spec();
        spec.buses.word_buses[0].src = vec![99];
        let errors = spec.validate().unwrap_err();
        assert!(has_error(&errors, |e| matches!(
            e,
            ArchSpecError::WordBusInvalidWordId {
                bus_id: 0,
                word_id: 99
            }
        )));
    }

    #[test]
    fn test_invalid_word_with_site_bus() {
        let mut spec = example_arch_spec();
        spec.words_with_site_buses.push(99);
        let errors = spec.validate().unwrap_err();
        assert!(has_error(&errors, |e| matches!(
            e,
            ArchSpecError::InvalidWordWithSiteBus { word_id: 99 }
        )));
    }

    #[test]
    fn test_invalid_site_with_word_bus() {
        let mut spec = example_arch_spec();
        spec.sites_with_word_buses.push(99);
        let errors = spec.validate().unwrap_err();
        assert!(has_error(&errors, |e| matches!(
            e,
            ArchSpecError::InvalidSiteWithWordBus {
                site_idx: 99,
                sites_per_word: 10
            }
        )));
    }

    #[test]
    fn test_invalid_path_lane() {
        let mut spec = example_arch_spec();
        let bad_lane = crate::arch::addr::LaneAddr {
            direction: crate::arch::addr::Direction::Forward,
            move_type: crate::arch::addr::MoveType::SiteBus,
            word_id: 0,
            site_id: 0,
            bus_id: 99,
        };
        spec.paths = Some(vec![crate::arch::types::TransportPath {
            lane: {
                let (d0, d1) = bad_lane.encode();
                (d0 as u64) | ((d1 as u64) << 32)
            },
            waypoints: vec![[1.0, 2.0]],
        }]);
        let errors = spec.validate().unwrap_err();
        assert!(has_error(&errors, |e| matches!(
            e,
            ArchSpecError::InvalidPathLane {
                index: 0,
                lane: _,
                message: _,
            }
        )));
    }

    #[test]
    fn test_valid_path_lane() {
        let mut spec = example_arch_spec();
        let good_lane = crate::arch::addr::LaneAddr {
            direction: crate::arch::addr::Direction::Forward,
            move_type: crate::arch::addr::MoveType::SiteBus,
            word_id: 0,
            site_id: 0,
            bus_id: 0,
        };
        spec.paths = Some(vec![crate::arch::types::TransportPath {
            lane: {
                let (d0, d1) = good_lane.encode();
                (d0 as u64) | ((d1 as u64) << 32)
            },
            waypoints: vec![[1.0, 2.5], [1.0, 5.0]],
        }]);
        assert!(spec.validate().is_ok());
    }

    #[test]
    fn test_path_too_few_waypoints() {
        let mut spec = example_arch_spec();
        let lane = crate::arch::addr::LaneAddr {
            direction: crate::arch::addr::Direction::Forward,
            move_type: crate::arch::addr::MoveType::SiteBus,
            word_id: 0,
            site_id: 0,
            bus_id: 0,
        };
        spec.paths = Some(vec![crate::arch::types::TransportPath {
            lane: {
                let (d0, d1) = lane.encode();
                (d0 as u64) | ((d1 as u64) << 32)
            },
            waypoints: vec![[1.0, 2.5]],
        }]);
        let errors = spec.validate().unwrap_err();
        assert!(has_error(&errors, |e| matches!(
            e,
            ArchSpecError::PathTooFewWaypoints {
                index: 0,
                count: 1,
                ..
            }
        )));
    }

    #[test]
    fn test_path_endpoint_mismatch_first() {
        let mut spec = example_arch_spec();
        let lane = crate::arch::addr::LaneAddr {
            direction: crate::arch::addr::Direction::Forward,
            move_type: crate::arch::addr::MoveType::SiteBus,
            word_id: 0,
            site_id: 0,
            bus_id: 0,
        };
        spec.paths = Some(vec![crate::arch::types::TransportPath {
            lane: {
                let (d0, d1) = lane.encode();
                (d0 as u64) | ((d1 as u64) << 32)
            },
            waypoints: vec![[99.0, 99.0], [1.0, 5.0]],
        }]);
        let errors = spec.validate().unwrap_err();
        assert!(has_error(&errors, |e| matches!(
            e,
            ArchSpecError::PathEndpointMismatch {
                index: 0,
                endpoint: "first",
                ..
            }
        )));
    }

    #[test]
    fn test_path_endpoint_mismatch_last() {
        let mut spec = example_arch_spec();
        let lane = crate::arch::addr::LaneAddr {
            direction: crate::arch::addr::Direction::Forward,
            move_type: crate::arch::addr::MoveType::SiteBus,
            word_id: 0,
            site_id: 0,
            bus_id: 0,
        };
        spec.paths = Some(vec![crate::arch::types::TransportPath {
            lane: {
                let (d0, d1) = lane.encode();
                (d0 as u64) | ((d1 as u64) << 32)
            },
            waypoints: vec![[1.0, 2.5], [99.0, 99.0]],
        }]);
        let errors = spec.validate().unwrap_err();
        assert!(has_error(&errors, |e| matches!(
            e,
            ArchSpecError::PathEndpointMismatch {
                index: 0,
                endpoint: "last",
                ..
            }
        )));
    }

    #[test]
    fn test_inconsistent_grid_shape() {
        let mut spec = example_arch_spec();
        spec.geometry.words[1].positions.x_spacing = vec![2.0, 2.0];
        let errors = spec.validate().unwrap_err();
        assert!(has_error(&errors, |e| matches!(
            e,
            ArchSpecError::InconsistentGridShape {
                word_id: 1,
                x_len: 3,
                y_len: 2,
                ref_x_len: 5,
                ref_y_len: 2,
            }
        )));
    }

    #[test]
    fn multiple_errors_collected() {
        let mut spec = example_arch_spec();
        spec.zones[0].words = vec![0]; // missing word 1
        spec.measurement_mode_zones = vec![1]; // not zone 0
        spec.sites_with_word_buses.push(99); // invalid site index
        let errors = spec.validate().unwrap_err();
        assert!(
            errors.len() >= 3,
            "expected at least 3 errors, got {}",
            errors.len()
        );
    }
}
