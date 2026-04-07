//! Structural validation for [`ArchSpec`].
//!
//! Validates all structural rules in a single pass, collecting every error
//! rather than failing fast. See [`ArchSpec::validate`].

use std::collections::HashSet;

use thiserror::Error;

use super::types::ArchSpec;

/// Error categories for arch spec structural validation.
///
/// Each variant groups related validation checks. Multiple errors
/// can be collected in a single validation pass.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum ArchSpecError {
    /// Structural error (grid dimensions, word consistency, minimum counts).
    #[error("{0}")]
    Structure(String),

    /// Per-zone bus validation error (site/word buses, membership lists).
    #[error("{0}")]
    ZoneBus(String),

    /// Inter-zone bus validation error (zone bus entries).
    #[error("{0}")]
    InterZoneBus(String),

    /// Grid invariant error (bus src/dst not rectangular).
    #[error("{0}")]
    GridInvariant(String),

    /// Entangling zone pair validation error.
    #[error("{0}")]
    EntanglingPair(String),

    /// Mode validation error.
    #[error("{0}")]
    Mode(String),

    /// Transport path validation error.
    #[error("{0}")]
    Path(String),
}

impl ArchSpec {
    /// Validate the arch spec against all structural rules.
    /// Collects all errors in one pass (not fail-fast).
    pub fn validate(&self) -> Result<(), Vec<ArchSpecError>> {
        let mut errors = Vec::new();

        let num_words = self.words.len();
        let num_zones = self.zones.len();
        let sites_per_word = self.sites_per_word();

        // Structural invariants
        check_minimum_counts(self, &mut errors);
        check_uniform_grid_dimensions(self, &mut errors);
        check_uniform_word_site_counts(self, &mut errors);
        check_word_site_indices(self, &mut errors);

        // Per-zone bus validation
        for (zone_idx, zone) in self.zones.iter().enumerate() {
            check_zone_words_with_site_buses(zone_idx, zone, num_words, &mut errors);
            check_zone_sites_with_word_buses(zone_idx, zone, sites_per_word, &mut errors);
            check_zone_site_buses(zone_idx, zone, sites_per_word, &mut errors);
            check_zone_word_buses(zone_idx, zone, num_words, &mut errors);
        }

        // Inter-zone bus validation
        check_zone_buses(self, num_zones, num_words, &mut errors);

        // Entangling zone pair validation
        check_entangling_zone_pairs(self, num_zones, &mut errors);

        // Mode validation
        check_modes(self, num_zones, num_words, sites_per_word, &mut errors);

        // Path validation
        check_paths(self, num_zones, &mut errors);

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

/// At least one zone and one word must exist.
fn check_minimum_counts(spec: &ArchSpec, errors: &mut Vec<ArchSpecError>) {
    if spec.zones.is_empty() {
        errors.push(ArchSpecError::Structure(
            "at least one zone must exist".into(),
        ));
    }
    if spec.words.is_empty() {
        errors.push(ArchSpecError::Structure(
            "at least one word must exist".into(),
        ));
    }
}

/// All zones must have the same grid dimensions (same num_x and num_y).
fn check_uniform_grid_dimensions(spec: &ArchSpec, errors: &mut Vec<ArchSpecError>) {
    if let Some(first_zone) = spec.zones.first() {
        let ref_x = first_zone.grid.num_x();
        let ref_y = first_zone.grid.num_y();
        for (idx, zone) in spec.zones.iter().enumerate().skip(1) {
            let zx = zone.grid.num_x();
            let zy = zone.grid.num_y();
            if zx != ref_x || zy != ref_y {
                errors.push(ArchSpecError::Structure(format!(
                    "zone {} grid dimensions ({}x{}) differ from zone 0 ({}x{})",
                    idx, zx, zy, ref_x, ref_y
                )));
            }
        }
    }
}

/// All words must have the same number of sites.
fn check_uniform_word_site_counts(spec: &ArchSpec, errors: &mut Vec<ArchSpecError>) {
    if let Some(first_word) = spec.words.first() {
        let ref_count = first_word.sites.len();
        for (idx, word) in spec.words.iter().enumerate().skip(1) {
            if word.sites.len() != ref_count {
                errors.push(ArchSpecError::Structure(format!(
                    "word {} has {} sites, expected {} (same as word 0)",
                    idx,
                    word.sites.len(),
                    ref_count
                )));
            }
        }
    }
}

/// Word site indices must be within the grid dimensions of every zone.
/// For each site [x, y]: x < grid.num_x() and y < grid.num_y().
fn check_word_site_indices(spec: &ArchSpec, errors: &mut Vec<ArchSpecError>) {
    // Use zone 0's grid as the reference (uniform dimensions already checked).
    let (grid_x, grid_y) = match spec.zones.first() {
        Some(z) => (z.grid.num_x(), z.grid.num_y()),
        None => return, // no zones → nothing to check
    };

    for (word_idx, word) in spec.words.iter().enumerate() {
        for (site_idx, site) in word.sites.iter().enumerate() {
            let x = site[0] as usize;
            let y = site[1] as usize;
            if x >= grid_x {
                errors.push(ArchSpecError::Structure(format!(
                    "word {}, site {}: x index {} out of range (grid has {} x-positions)",
                    word_idx, site_idx, site[0], grid_x
                )));
            }
            if y >= grid_y {
                errors.push(ArchSpecError::Structure(format!(
                    "word {}, site {}: y index {} out of range (grid has {} y-positions)",
                    word_idx, site_idx, site[1], grid_y
                )));
            }
        }
    }
}

// --- Per-zone bus validation ---

use super::types::Zone;

/// `words_with_site_buses` entries must be < number of words.
fn check_zone_words_with_site_buses(
    zone_idx: usize,
    zone: &Zone,
    num_words: usize,
    errors: &mut Vec<ArchSpecError>,
) {
    for &wid in &zone.words_with_site_buses {
        if wid as usize >= num_words {
            errors.push(ArchSpecError::ZoneBus(format!(
                "zone {}: words_with_site_buses contains invalid word ID {}",
                zone_idx, wid
            )));
        }
    }
}

/// `sites_with_word_buses` entries must be valid site indices.
fn check_zone_sites_with_word_buses(
    zone_idx: usize,
    zone: &Zone,
    sites_per_word: usize,
    errors: &mut Vec<ArchSpecError>,
) {
    for &sid in &zone.sites_with_word_buses {
        if sid as usize >= sites_per_word {
            errors.push(ArchSpecError::ZoneBus(format!(
                "zone {}: sites_with_word_buses contains invalid site index {} (sites_per_word={})",
                zone_idx, sid, sites_per_word
            )));
        }
    }
}

/// Site bus src/dst must have same length and SiteRef values < sites_per_word.
fn check_zone_site_buses(
    zone_idx: usize,
    zone: &Zone,
    sites_per_word: usize,
    errors: &mut Vec<ArchSpecError>,
) {
    for (bus_idx, bus) in zone.site_buses.iter().enumerate() {
        if bus.src.len() != bus.dst.len() {
            errors.push(ArchSpecError::ZoneBus(format!(
                "zone {}, site_bus {}: src length ({}) != dst length ({})",
                zone_idx,
                bus_idx,
                bus.src.len(),
                bus.dst.len()
            )));
        }
        for (i, sref) in bus.src.iter().enumerate() {
            if sref.0 as usize >= sites_per_word {
                errors.push(ArchSpecError::ZoneBus(format!(
                    "zone {}, site_bus {}: src[{}] SiteRef({}) >= sites_per_word ({})",
                    zone_idx, bus_idx, i, sref.0, sites_per_word
                )));
            }
        }
        for (i, sref) in bus.dst.iter().enumerate() {
            if sref.0 as usize >= sites_per_word {
                errors.push(ArchSpecError::ZoneBus(format!(
                    "zone {}, site_bus {}: dst[{}] SiteRef({}) >= sites_per_word ({})",
                    zone_idx, bus_idx, i, sref.0, sites_per_word
                )));
            }
        }
    }
}

/// Word bus src/dst must have same length and WordRef values < number of words.
fn check_zone_word_buses(
    zone_idx: usize,
    zone: &Zone,
    num_words: usize,
    errors: &mut Vec<ArchSpecError>,
) {
    for (bus_idx, bus) in zone.word_buses.iter().enumerate() {
        if bus.src.len() != bus.dst.len() {
            errors.push(ArchSpecError::ZoneBus(format!(
                "zone {}, word_bus {}: src length ({}) != dst length ({})",
                zone_idx,
                bus_idx,
                bus.src.len(),
                bus.dst.len()
            )));
        }
        for (i, wref) in bus.src.iter().enumerate() {
            if wref.0 as usize >= num_words {
                errors.push(ArchSpecError::ZoneBus(format!(
                    "zone {}, word_bus {}: src[{}] WordRef({}) >= num_words ({})",
                    zone_idx, bus_idx, i, wref.0, num_words
                )));
            }
        }
        for (i, wref) in bus.dst.iter().enumerate() {
            if wref.0 as usize >= num_words {
                errors.push(ArchSpecError::ZoneBus(format!(
                    "zone {}, word_bus {}: dst[{}] WordRef({}) >= num_words ({})",
                    zone_idx, bus_idx, i, wref.0, num_words
                )));
            }
        }
    }
}

// --- Inter-zone bus validation ---

/// Zone bus entries must have valid zone_id and word_id, src/dst same length,
/// and every pair must cross a zone boundary.
fn check_zone_buses(
    spec: &ArchSpec,
    num_zones: usize,
    num_words: usize,
    errors: &mut Vec<ArchSpecError>,
) {
    for (bus_idx, bus) in spec.zone_buses.iter().enumerate() {
        if bus.src.len() != bus.dst.len() {
            errors.push(ArchSpecError::InterZoneBus(format!(
                "zone_bus {}: src length ({}) != dst length ({})",
                bus_idx,
                bus.src.len(),
                bus.dst.len()
            )));
        }

        // Validate all ZonedWordRef entries
        for (i, zwr) in bus.src.iter().enumerate() {
            if zwr.zone_id as usize >= num_zones {
                errors.push(ArchSpecError::InterZoneBus(format!(
                    "zone_bus {}: src[{}] zone_id {} >= num_zones ({})",
                    bus_idx, i, zwr.zone_id, num_zones
                )));
            }
            if zwr.word_id as usize >= num_words {
                errors.push(ArchSpecError::InterZoneBus(format!(
                    "zone_bus {}: src[{}] word_id {} >= num_words ({})",
                    bus_idx, i, zwr.word_id, num_words
                )));
            }
        }
        for (i, zwr) in bus.dst.iter().enumerate() {
            if zwr.zone_id as usize >= num_zones {
                errors.push(ArchSpecError::InterZoneBus(format!(
                    "zone_bus {}: dst[{}] zone_id {} >= num_zones ({})",
                    bus_idx, i, zwr.zone_id, num_zones
                )));
            }
            if zwr.word_id as usize >= num_words {
                errors.push(ArchSpecError::InterZoneBus(format!(
                    "zone_bus {}: dst[{}] word_id {} >= num_words ({})",
                    bus_idx, i, zwr.word_id, num_words
                )));
            }
        }

        // Every (src[i], dst[i]) pair must cross a zone boundary
        let pair_count = bus.src.len().min(bus.dst.len());
        for i in 0..pair_count {
            if bus.src[i].zone_id == bus.dst[i].zone_id {
                errors.push(ArchSpecError::InterZoneBus(format!(
                    "zone_bus {}: pair {} does not cross a zone boundary \
                     (src zone_id={}, dst zone_id={})",
                    bus_idx, i, bus.src[i].zone_id, bus.dst[i].zone_id
                )));
            }
        }
    }
}

// --- Entangling zone pair validation ---

/// Both zone indices must be valid and no duplicate pairs.
fn check_entangling_zone_pairs(
    spec: &ArchSpec,
    num_zones: usize,
    errors: &mut Vec<ArchSpecError>,
) {
    let mut seen: HashSet<[u32; 2]> = HashSet::new();
    for (idx, pair) in spec.entangling_zone_pairs.iter().enumerate() {
        let [a, b] = *pair;
        if a as usize >= num_zones {
            errors.push(ArchSpecError::EntanglingPair(format!(
                "entangling_zone_pairs[{}]: zone ID {} >= num_zones ({})",
                idx, a, num_zones
            )));
        }
        if b as usize >= num_zones {
            errors.push(ArchSpecError::EntanglingPair(format!(
                "entangling_zone_pairs[{}]: zone ID {} >= num_zones ({})",
                idx, b, num_zones
            )));
        }
        // Normalize pair order for duplicate detection
        let normalized = if a <= b { [a, b] } else { [b, a] };
        if !seen.insert(normalized) {
            errors.push(ArchSpecError::EntanglingPair(format!(
                "entangling_zone_pairs[{}]: duplicate pair [{}, {}]",
                idx, a, b
            )));
        }
    }
}

// --- Mode validation ---

/// Zone indices and bitstring_order entries must be valid.
fn check_modes(
    spec: &ArchSpec,
    num_zones: usize,
    num_words: usize,
    sites_per_word: usize,
    errors: &mut Vec<ArchSpecError>,
) {
    for (mode_idx, mode) in spec.modes.iter().enumerate() {
        for &zone_id in &mode.zones {
            if zone_id as usize >= num_zones {
                errors.push(ArchSpecError::Mode(format!(
                    "mode '{}' (index {}): zone ID {} >= num_zones ({})",
                    mode.name, mode_idx, zone_id, num_zones
                )));
            }
        }
        for (loc_idx, loc) in mode.bitstring_order.iter().enumerate() {
            if loc.zone_id as usize >= num_zones {
                errors.push(ArchSpecError::Mode(format!(
                    "mode '{}' (index {}): bitstring_order[{}] zone_id {} >= num_zones ({})",
                    mode.name, mode_idx, loc_idx, loc.zone_id, num_zones
                )));
            }
            if loc.word_id as usize >= num_words {
                errors.push(ArchSpecError::Mode(format!(
                    "mode '{}' (index {}): bitstring_order[{}] word_id {} >= num_words ({})",
                    mode.name, mode_idx, loc_idx, loc.word_id, num_words
                )));
            }
            if loc.site_id as usize >= sites_per_word {
                errors.push(ArchSpecError::Mode(format!(
                    "mode '{}' (index {}): bitstring_order[{}] site_id {} >= sites_per_word ({})",
                    mode.name, mode_idx, loc_idx, loc.site_id, sites_per_word
                )));
            }
        }
    }
}

// --- Path validation ---

/// If paths is Some, validate each path's waypoints and lane address.
fn check_paths(spec: &ArchSpec, num_zones: usize, errors: &mut Vec<ArchSpecError>) {
    if let Some(paths) = &spec.paths {
        for (idx, path) in paths.iter().enumerate() {
            if !path.check_finite() {
                errors.push(ArchSpecError::Path(format!(
                    "paths[{}]: waypoint contains non-finite coordinate",
                    idx
                )));
            }

            // Validate the lane address fields
            let lane = super::addr::LaneAddr::decode_u64(path.lane);
            if lane.zone_id as usize >= num_zones {
                errors.push(ArchSpecError::Path(format!(
                    "paths[{}]: lane 0x{:016X} has invalid zone_id {} (num_zones={})",
                    idx, path.lane, lane.zone_id, num_zones
                )));
            }

            if path.waypoints.len() < 2 {
                errors.push(ArchSpecError::Path(format!(
                    "paths[{}]: lane 0x{:016X} has {} waypoint(s), minimum is 2",
                    idx,
                    path.lane,
                    path.waypoints.len()
                )));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::addr::{SiteRef, WordRef, ZonedWordRef};
    use crate::arch::types::{Bus, Grid, Mode, Word, Zone};
    use crate::version::Version;

    /// Create a valid two-zone arch spec for testing.
    fn make_valid_two_zone_spec() -> ArchSpec {
        let grid0 = Grid::from_positions(&[0.0, 5.0, 10.0], &[0.0, 3.0]);
        let grid1 = Grid::from_positions(&[0.0, 7.5, 15.0], &[0.0, 4.0]);

        ArchSpec {
            version: Version::new(2, 0),
            words: vec![
                Word {
                    sites: vec![[0, 0], [0, 1]],
                },
                Word {
                    sites: vec![[1, 0], [1, 1]],
                },
            ],
            zones: vec![
                Zone {
                    grid: grid0,
                    site_buses: vec![Bus {
                        src: vec![SiteRef(0)],
                        dst: vec![SiteRef(1)],
                    }],
                    word_buses: vec![Bus {
                        src: vec![WordRef(0)],
                        dst: vec![WordRef(1)],
                    }],
                    words_with_site_buses: vec![0, 1],
                    sites_with_word_buses: vec![0],
                },
                Zone {
                    grid: grid1,
                    site_buses: vec![],
                    word_buses: vec![],
                    words_with_site_buses: vec![],
                    sites_with_word_buses: vec![],
                },
            ],
            zone_buses: vec![Bus {
                src: vec![ZonedWordRef {
                    zone_id: 0,
                    word_id: 0,
                }],
                dst: vec![ZonedWordRef {
                    zone_id: 1,
                    word_id: 0,
                }],
            }],
            entangling_zone_pairs: vec![[0, 1]],
            modes: vec![Mode {
                name: "full".to_string(),
                zones: vec![0, 1],
                bitstring_order: vec![],
            }],
            paths: None,
            feed_forward: false,
            atom_reloading: false,
        }
    }

    #[test]
    fn test_valid_two_zone_spec() {
        let spec = make_valid_two_zone_spec();
        assert!(spec.validate().is_ok());
    }

    #[test]
    fn test_validate_zones_must_have_same_grid_dimensions() {
        let mut spec = make_valid_two_zone_spec();
        // 4 x-points vs 3
        spec.zones[1].grid = Grid::from_positions(&[0.0, 1.0, 2.0, 3.0], &[0.0, 1.0]);
        assert!(matches!(
            spec.validate(),
            Err(ref errs) if errs.iter().any(|e| matches!(e, ArchSpecError::Structure(_)))
        ));
    }

    #[test]
    fn test_validate_site_bus_ref_out_of_range() {
        let mut spec = make_valid_two_zone_spec();
        spec.zones[0].site_buses = vec![Bus {
            src: vec![SiteRef(0)],
            dst: vec![SiteRef(999)],
        }];
        assert!(matches!(
            spec.validate(),
            Err(ref errs) if errs.iter().any(|e| matches!(e, ArchSpecError::ZoneBus(_)))
        ));
    }

    #[test]
    fn test_validate_zone_bus_must_cross_zones() {
        let mut spec = make_valid_two_zone_spec();
        spec.zone_buses = vec![Bus {
            src: vec![ZonedWordRef {
                zone_id: 0,
                word_id: 0,
            }],
            dst: vec![ZonedWordRef {
                zone_id: 0,
                word_id: 1,
            }], // same zone!
        }];
        assert!(matches!(
            spec.validate(),
            Err(ref errs) if errs.iter().any(|e| matches!(e, ArchSpecError::InterZoneBus(_)))
        ));
    }

    #[test]
    fn test_validate_entangling_zone_pair_invalid_zone() {
        let mut spec = make_valid_two_zone_spec();
        spec.entangling_zone_pairs = vec![[0, 99]];
        assert!(matches!(
            spec.validate(),
            Err(ref errs) if errs.iter().any(|e| matches!(e, ArchSpecError::EntanglingPair(_)))
        ));
    }

    #[test]
    fn test_validate_mode_invalid_zone() {
        let mut spec = make_valid_two_zone_spec();
        spec.modes = vec![Mode {
            name: "bad".to_string(),
            zones: vec![99],
            bitstring_order: vec![],
        }];
        assert!(matches!(
            spec.validate(),
            Err(ref errs) if errs.iter().any(|e| matches!(e, ArchSpecError::Mode(_)))
        ));
    }

    #[test]
    fn test_validate_no_zones() {
        let mut spec = make_valid_two_zone_spec();
        spec.zones = vec![];
        assert!(matches!(
            spec.validate(),
            Err(ref errs) if errs.iter().any(|e| matches!(e, ArchSpecError::Structure(msg) if msg.contains("zone")))
        ));
    }

    #[test]
    fn test_validate_no_words() {
        let mut spec = make_valid_two_zone_spec();
        spec.words = vec![];
        assert!(matches!(
            spec.validate(),
            Err(ref errs) if errs.iter().any(|e| matches!(e, ArchSpecError::Structure(msg) if msg.contains("word")))
        ));
    }

    #[test]
    fn test_validate_word_site_count_mismatch() {
        let mut spec = make_valid_two_zone_spec();
        spec.words[1].sites = vec![[0, 0]]; // 1 site vs 2
        assert!(matches!(
            spec.validate(),
            Err(ref errs) if errs.iter().any(|e| matches!(e, ArchSpecError::Structure(msg) if msg.contains("sites")))
        ));
    }

    #[test]
    fn test_validate_word_site_x_out_of_range() {
        let mut spec = make_valid_two_zone_spec();
        spec.words[0].sites[0] = [99, 0]; // x=99 but grid has 3 x-positions
        assert!(matches!(
            spec.validate(),
            Err(ref errs) if errs.iter().any(|e| matches!(e, ArchSpecError::Structure(msg) if msg.contains("x index")))
        ));
    }

    #[test]
    fn test_validate_word_site_y_out_of_range() {
        let mut spec = make_valid_two_zone_spec();
        spec.words[0].sites[0] = [0, 99]; // y=99 but grid has 2 y-positions
        assert!(matches!(
            spec.validate(),
            Err(ref errs) if errs.iter().any(|e| matches!(e, ArchSpecError::Structure(msg) if msg.contains("y index")))
        ));
    }

    #[test]
    fn test_validate_zone_words_with_site_buses_invalid() {
        let mut spec = make_valid_two_zone_spec();
        spec.zones[0].words_with_site_buses = vec![0, 99];
        assert!(matches!(
            spec.validate(),
            Err(ref errs) if errs.iter().any(|e| matches!(e, ArchSpecError::ZoneBus(msg) if msg.contains("words_with_site_buses")))
        ));
    }

    #[test]
    fn test_validate_zone_sites_with_word_buses_invalid() {
        let mut spec = make_valid_two_zone_spec();
        spec.zones[0].sites_with_word_buses = vec![99];
        assert!(matches!(
            spec.validate(),
            Err(ref errs) if errs.iter().any(|e| matches!(e, ArchSpecError::ZoneBus(msg) if msg.contains("sites_with_word_buses")))
        ));
    }

    #[test]
    fn test_validate_site_bus_length_mismatch() {
        let mut spec = make_valid_two_zone_spec();
        spec.zones[0].site_buses = vec![Bus {
            src: vec![SiteRef(0), SiteRef(1)],
            dst: vec![SiteRef(0)],
        }];
        assert!(matches!(
            spec.validate(),
            Err(ref errs) if errs.iter().any(|e| matches!(e, ArchSpecError::ZoneBus(msg) if msg.contains("src length")))
        ));
    }

    #[test]
    fn test_validate_word_bus_invalid_word_ref() {
        let mut spec = make_valid_two_zone_spec();
        spec.zones[0].word_buses = vec![Bus {
            src: vec![WordRef(0)],
            dst: vec![WordRef(99)],
        }];
        assert!(matches!(
            spec.validate(),
            Err(ref errs) if errs.iter().any(|e| matches!(e, ArchSpecError::ZoneBus(msg) if msg.contains("WordRef(99)")))
        ));
    }

    #[test]
    fn test_validate_zone_bus_invalid_zone_id() {
        let mut spec = make_valid_two_zone_spec();
        spec.zone_buses = vec![Bus {
            src: vec![ZonedWordRef {
                zone_id: 99,
                word_id: 0,
            }],
            dst: vec![ZonedWordRef {
                zone_id: 1,
                word_id: 0,
            }],
        }];
        assert!(matches!(
            spec.validate(),
            Err(ref errs) if errs.iter().any(|e| matches!(e, ArchSpecError::InterZoneBus(msg) if msg.contains("zone_id 99")))
        ));
    }

    #[test]
    fn test_validate_zone_bus_invalid_word_id() {
        let mut spec = make_valid_two_zone_spec();
        spec.zone_buses = vec![Bus {
            src: vec![ZonedWordRef {
                zone_id: 0,
                word_id: 99,
            }],
            dst: vec![ZonedWordRef {
                zone_id: 1,
                word_id: 0,
            }],
        }];
        assert!(matches!(
            spec.validate(),
            Err(ref errs) if errs.iter().any(|e| matches!(e, ArchSpecError::InterZoneBus(msg) if msg.contains("word_id 99")))
        ));
    }

    #[test]
    fn test_validate_zone_bus_length_mismatch() {
        let mut spec = make_valid_two_zone_spec();
        spec.zone_buses = vec![Bus {
            src: vec![
                ZonedWordRef {
                    zone_id: 0,
                    word_id: 0,
                },
                ZonedWordRef {
                    zone_id: 0,
                    word_id: 1,
                },
            ],
            dst: vec![ZonedWordRef {
                zone_id: 1,
                word_id: 0,
            }],
        }];
        assert!(matches!(
            spec.validate(),
            Err(ref errs) if errs.iter().any(|e| matches!(e, ArchSpecError::InterZoneBus(msg) if msg.contains("src length")))
        ));
    }

    #[test]
    fn test_validate_entangling_zone_pair_duplicate() {
        let mut spec = make_valid_two_zone_spec();
        spec.entangling_zone_pairs = vec![[0, 1], [1, 0]]; // same pair reversed
        assert!(matches!(
            spec.validate(),
            Err(ref errs) if errs.iter().any(|e| matches!(e, ArchSpecError::EntanglingPair(msg) if msg.contains("duplicate")))
        ));
    }

    #[test]
    fn test_validate_mode_bitstring_order_invalid() {
        use crate::arch::addr::LocationAddr;
        let mut spec = make_valid_two_zone_spec();
        spec.modes = vec![Mode {
            name: "bad_loc".to_string(),
            zones: vec![0],
            bitstring_order: vec![LocationAddr {
                zone_id: 99,
                word_id: 0,
                site_id: 0,
            }],
        }];
        assert!(matches!(
            spec.validate(),
            Err(ref errs) if errs.iter().any(|e| matches!(e, ArchSpecError::Mode(msg) if msg.contains("zone_id 99")))
        ));
    }

    #[test]
    fn test_validate_multiple_errors_collected() {
        let mut spec = make_valid_two_zone_spec();
        // Break multiple things
        spec.entangling_zone_pairs = vec![[0, 99]]; // bad zone
        spec.zones[0].words_with_site_buses = vec![99]; // bad word
        let errors = spec.validate().unwrap_err();
        assert!(
            errors.len() >= 2,
            "expected at least 2 errors, got {}",
            errors.len()
        );
    }

    #[test]
    fn test_validate_path_non_finite_waypoint() {
        let mut spec = make_valid_two_zone_spec();
        let lane = crate::arch::addr::LaneAddr {
            direction: crate::arch::addr::Direction::Forward,
            move_type: crate::arch::addr::MoveType::SiteBus,
            zone_id: 0,
            word_id: 0,
            site_id: 0,
            bus_id: 0,
        };
        spec.paths = Some(vec![crate::arch::types::TransportPath {
            lane: lane.encode_u64(),
            waypoints: vec![[f64::NAN, 0.0], [1.0, 2.0]],
        }]);
        assert!(matches!(
            spec.validate(),
            Err(ref errs) if errs.iter().any(|e| matches!(e, ArchSpecError::Path(msg) if msg.contains("non-finite")))
        ));
    }

    #[test]
    fn test_validate_path_too_few_waypoints() {
        let mut spec = make_valid_two_zone_spec();
        let lane = crate::arch::addr::LaneAddr {
            direction: crate::arch::addr::Direction::Forward,
            move_type: crate::arch::addr::MoveType::SiteBus,
            zone_id: 0,
            word_id: 0,
            site_id: 0,
            bus_id: 0,
        };
        spec.paths = Some(vec![crate::arch::types::TransportPath {
            lane: lane.encode_u64(),
            waypoints: vec![[1.0, 2.0]],
        }]);
        assert!(matches!(
            spec.validate(),
            Err(ref errs) if errs.iter().any(|e| matches!(e, ArchSpecError::Path(msg) if msg.contains("minimum is 2")))
        ));
    }

    #[test]
    fn test_validate_path_invalid_zone_id() {
        let mut spec = make_valid_two_zone_spec();
        let lane = crate::arch::addr::LaneAddr {
            direction: crate::arch::addr::Direction::Forward,
            move_type: crate::arch::addr::MoveType::SiteBus,
            zone_id: 99,
            word_id: 0,
            site_id: 0,
            bus_id: 0,
        };
        spec.paths = Some(vec![crate::arch::types::TransportPath {
            lane: lane.encode_u64(),
            waypoints: vec![[0.0, 0.0], [1.0, 2.0]],
        }]);
        assert!(matches!(
            spec.validate(),
            Err(ref errs) if errs.iter().any(|e| matches!(e, ArchSpecError::Path(msg) if msg.contains("zone_id")))
        ));
    }
}
