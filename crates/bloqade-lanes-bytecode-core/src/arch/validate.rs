//! Structural validation for [`ArchSpec`].
//!
//! Validates all structural rules in a single pass, collecting every error
//! rather than failing fast. See [`ArchSpec::validate`].

use std::collections::{BTreeMap, HashMap, HashSet};

use thiserror::Error;

use super::addr::LocationAddr;
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

    /// Bus src→dst relation contains a cycle.
    ///
    /// A bus is by definition a set of explicit edge transports executed
    /// simultaneously as one AOD operation — never a permutation. A chain
    /// (`x→y, y→z` with the final destination outside the source set) shifts
    /// atoms like a conveyor and is physically valid; a cycle
    /// (`x→y, y→z, z→x`, including the self-loop `x→x`) would rotate a
    /// fully-occupied set of atoms with no empty site, which AOD hardware
    /// cannot do (tones in a group share a direction and cannot cross).
    /// Cyclic buses are therefore rejected outright, while
    /// overlapping-but-acyclic buses remain legal.
    #[error("{0}")]
    CyclicBus(String),

    /// A bus describes no single AOD operation.
    ///
    /// An AOD traps at the Cartesian product of its x and y tones and sweeps
    /// each tone independently, so the lanes a bus fires together must be
    /// **separable and order-preserving**: an atom's landing x is decided by
    /// its column alone and its landing y by its row alone, and the sorted
    /// source tones land on the sorted destination tones, because two tones
    /// cannot cross, merge or split. The sources must also form a complete
    /// grid, since the AOD traps at every crossing of its tones. A bus that
    /// would send one row onto a diagonal, reverse the order of two columns,
    /// or start from a diagonal, is rejected.
    ///
    /// On a bus that passes, any rectangle of its sources lands on a
    /// rectangle, which is what lets the router build shots from source
    /// positions alone, and the full bus is one rectangle at each end.
    #[error("{0}")]
    AodTransport(String),

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
        check_non_negative_grid_spacings(self, &mut errors);
        check_uniform_grid_dimensions(self, &mut errors);
        check_uniform_word_site_counts(self, &mut errors);
        check_word_site_indices(self, &mut errors);

        // Per-zone bus and entangling pair validation
        for (zone_idx, zone) in self.zones.iter().enumerate() {
            check_zone_words_with_site_buses(zone_idx, zone, num_words, &mut errors);
            check_zone_sites_with_word_buses(zone_idx, zone, sites_per_word, &mut errors);
            check_zone_site_buses(zone_idx, zone, sites_per_word, &mut errors);
            check_zone_word_buses(zone_idx, zone, num_words, sites_per_word, &mut errors);
            check_zone_entangling_pairs(zone_idx, zone, num_words, &mut errors);
        }

        // Inter-zone bus validation
        check_zone_buses(self, num_zones, num_words, &mut errors);

        // Every bus must be one AOD operation
        check_aod_transports(self, sites_per_word, &mut errors);

        // Mode validation
        check_modes(self, num_zones, num_words, sites_per_word, &mut errors);

        // Path validation
        check_paths(self, num_zones, &mut errors);

        // Zone physical-space overlap
        check_zone_overlap(self, &mut errors);

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

/// All grid spacings must be non-negative. Negative spacings would produce
/// non-monotonic positions, breaking the `Grid::bounding_box()` invariant
/// and making position lookups ambiguous.
fn check_non_negative_grid_spacings(spec: &ArchSpec, errors: &mut Vec<ArchSpecError>) {
    for (idx, zone) in spec.zones.iter().enumerate() {
        if let Some(pos) = zone.grid.x_spacing.iter().position(|&v| v < 0.0) {
            errors.push(ArchSpecError::Structure(format!(
                "zone {} grid x_spacing[{}] is negative ({:.6})",
                idx, pos, zone.grid.x_spacing[pos]
            )));
        }
        if let Some(pos) = zone.grid.y_spacing.iter().position(|&v| v < 0.0) {
            errors.push(ArchSpecError::Structure(format!(
                "zone {} grid y_spacing[{}] is negative ({:.6})",
                idx, pos, zone.grid.y_spacing[pos]
            )));
        }
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

use super::types::{SiteMap, Zone};

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
        check_bus_relation(
            &bus.src,
            &bus.dst,
            &format!("zone {zone_idx}, site_bus {bus_idx}"),
            |s| s.0.to_string(),
            ArchSpecError::ZoneBus,
            errors,
        );
    }
}

/// Word bus src/dst must have same length and WordRef values < number of words,
/// and an optional `site_map` must be well formed.
fn check_zone_word_buses(
    zone_idx: usize,
    zone: &Zone,
    num_words: usize,
    sites_per_word: usize,
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
        check_bus_relation(
            &bus.src,
            &bus.dst,
            &format!("zone {zone_idx}, word_bus {bus_idx}"),
            |w| w.0.to_string(),
            ArchSpecError::ZoneBus,
            errors,
        );
        if let Some(map) = &bus.site_map {
            check_word_bus_site_map(zone_idx, bus_idx, zone, map, sites_per_word, errors);
        }
    }
}

/// A word bus's `site_map` obeys the same rules as a site bus, plus a
/// membership rule.
///
/// `src`/`dst` must have equal length, every entry must be `< sites_per_word`
/// and listed in the zone's `sites_with_word_buses` (the word-bus landing
/// pads), and the src→dst relation must be well formed exactly like a site
/// bus's ([`check_bus_relation`]): unique sources, unique destinations, and no
/// cycles — the site map is part of the same AOD operation, so a rotation of
/// sites is rejected even though the word relation already prevents a
/// `(word, site)` location from mapping onto itself.
fn check_word_bus_site_map(
    zone_idx: usize,
    bus_idx: usize,
    zone: &Zone,
    map: &SiteMap,
    sites_per_word: usize,
    errors: &mut Vec<ArchSpecError>,
) {
    let label = format!("zone {zone_idx}, word_bus {bus_idx} site_map");
    if map.src.len() != map.dst.len() {
        errors.push(ArchSpecError::ZoneBus(format!(
            "{label}: src length ({}) != dst length ({})",
            map.src.len(),
            map.dst.len()
        )));
    }
    for (side, refs) in [("src", &map.src), ("dst", &map.dst)] {
        for (i, sref) in refs.iter().enumerate() {
            if sref.0 as usize >= sites_per_word {
                errors.push(ArchSpecError::ZoneBus(format!(
                    "{label}: {side}[{i}] SiteRef({}) >= sites_per_word ({sites_per_word})",
                    sref.0
                )));
            } else if !zone.sites_with_word_buses.contains(&(sref.0 as u32)) {
                errors.push(ArchSpecError::ZoneBus(format!(
                    "{label}: {side}[{i}] site {} is not in zone {zone_idx} sites_with_word_buses",
                    sref.0
                )));
            }
        }
    }
    check_bus_relation(
        &map.src,
        &map.dst,
        &label,
        |s| s.0.to_string(),
        ArchSpecError::ZoneBus,
        errors,
    );
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

        check_bus_relation(
            &bus.src,
            &bus.dst,
            &format!("zone_bus {bus_idx}"),
            |z| format!("z{}w{}", z.zone_id, z.word_id),
            ArchSpecError::InterZoneBus,
            errors,
        );
    }
}

// --- AOD realizability ---

/// One lane's `(source, destination)` physical positions.
type TransportPair = ((f64, f64), (f64, f64));

/// One bus transport: one [`TransportPair`] per lane fired together.
type Transport = Vec<TransportPair>;

/// Every bus must be one AOD operation — see [`ArchSpecError::AodTransport`].
///
/// A transport is every lane that can fire in one shot, which is the key
/// `check_lane_group_consistency` groups on: a site or word bus is one
/// transport in its zone, and a zone bus is one transport per forward source
/// zone, since its lanes carry that zone's id. A bus with an endpoint whose
/// position does not resolve was already reported by the structural checks,
/// so it is skipped here rather than reported twice.
///
/// This is the rule the Python architecture builder enforced in
/// `check_aod_transport` for the specs it built, made stricter: the builder
/// let a bus's sources skip a tone crossing that held no atom, and here they
/// must cover every crossing. Specs loaded from JSON never went through the
/// builder, and a word bus `site_map` cannot be built there at all.
fn check_aod_transports(spec: &ArchSpec, sites_per_word: usize, errors: &mut Vec<ArchSpecError>) {
    let position = |zone_id: u32, word_id: u32, site_id: u32| {
        spec.location_position(&LocationAddr {
            zone_id,
            word_id,
            site_id,
        })
    };

    for (zone_idx, zone) in spec.zones.iter().enumerate() {
        let z = zone_idx as u32;
        for (bus_idx, bus) in zone.site_buses.iter().enumerate() {
            let transport: Option<Transport> = zone
                .words_with_site_buses
                .iter()
                .flat_map(|&w| {
                    bus.src
                        .iter()
                        .zip(&bus.dst)
                        .map(move |(s, d)| (w, u32::from(s.0), u32::from(d.0)))
                })
                .map(|(w, s, d)| Some((position(z, w, s)?, position(z, w, d)?)))
                .collect();
            report_aod_transport(
                transport,
                &format!("zone {zone_idx}, site_bus {bus_idx}"),
                errors,
            );
        }
        for (bus_idx, bus) in zone.word_buses.iter().enumerate() {
            let sites = zone.word_bus_site_pairs(bus_idx);
            let transport: Option<Transport> = bus
                .src
                .iter()
                .zip(&bus.dst)
                .flat_map(|(sw, dw)| {
                    sites
                        .iter()
                        .map(move |&(s, d)| (u32::from(sw.0), u32::from(dw.0), s, d))
                })
                .map(|(sw, dw, s, d)| Some((position(z, sw, s)?, position(z, dw, d)?)))
                .collect();
            report_aod_transport(
                transport,
                &format!("zone {zone_idx}, word_bus {bus_idx}"),
                errors,
            );
        }
    }

    for (bus_idx, bus) in spec.zone_buses.iter().enumerate() {
        let mut by_source_zone: BTreeMap<u8, Option<Transport>> = BTreeMap::new();
        for (s, d) in bus.src.iter().zip(&bus.dst) {
            let entry = by_source_zone
                .entry(s.zone_id)
                .or_insert_with(|| Some(Vec::new()));
            for site in 0..sites_per_word as u32 {
                let pair = position(s.zone_id.into(), s.word_id.into(), site).zip(position(
                    d.zone_id.into(),
                    d.word_id.into(),
                    site,
                ));
                match pair {
                    Some(p) => {
                        if let Some(transport) = entry {
                            transport.push(p);
                        }
                    }
                    None => *entry = None,
                }
            }
        }
        for (zone_id, transport) in by_source_zone {
            report_aod_transport(
                transport,
                &format!("zone_bus {bus_idx} (lanes from zone {zone_id})"),
                errors,
            );
        }
    }
}

fn report_aod_transport(
    transport: Option<Transport>,
    label: &str,
    errors: &mut Vec<ArchSpecError>,
) {
    if let Some(pairs) = transport
        && let Some(msg) = aod_transport_violation(&pairs)
    {
        errors.push(ArchSpecError::AodTransport(format!("{label}: {msg}")));
    }
}

/// `None` when `pairs` is one AOD operation: the sorted source tones map onto
/// the sorted destination tones, every atom lands where its own column and
/// row land, and the sources cover every crossing of their tones. Otherwise
/// a description of the first violation.
fn aod_transport_violation(pairs: &[TransportPair]) -> Option<String> {
    let tones = |pick: fn(&TransportPair) -> f64| -> Vec<f64> {
        let mut tones: Vec<f64> = pairs.iter().map(pick).collect();
        tones.sort_by(f64::total_cmp);
        tones.dedup_by(|a, b| a.to_bits() == b.to_bits());
        tones
    };
    let src_x = tones(|p| p.0.0);
    let src_y = tones(|p| p.0.1);
    let dst_x = tones(|p| p.1.0);
    let dst_y = tones(|p| p.1.1);

    if src_x.len() != dst_x.len() || src_y.len() != dst_y.len() {
        return Some(format!(
            "the bus spans a {}x{} grid of tones at its source but {}x{} at its \
             destination; an AOD cannot add, drop or merge a tone mid-transport",
            src_x.len(),
            src_y.len(),
            dst_x.len(),
            dst_y.len()
        ));
    }

    let tone_index = |tones: &[f64], v: f64| {
        tones
            .binary_search_by(|t| t.total_cmp(&v))
            .expect("every coordinate is one of its own side's tones")
    };
    for &((sx, sy), (dx, dy)) in pairs {
        let wx = dst_x[tone_index(&src_x, sx)];
        let wy = dst_y[tone_index(&src_y, sy)];
        if dx.to_bits() != wx.to_bits() || dy.to_bits() != wy.to_bits() {
            return Some(format!(
                "the atom at ({sx}, {sy}) is sent to ({dx}, {dy}), but its column and \
                 row land at ({wx}, {wy}); an AOD sweeps whole x and y tones, which \
                 cannot cross or trade places, so every atom must stay on its own tones"
            ));
        }
    }

    // Completeness: the AOD traps at every crossing of its tones, so every
    // crossing must be one of the bus's sources. (On a separable,
    // order-preserving bus the destinations are then a complete grid too.)
    let sources: HashSet<(u64, u64)> = pairs
        .iter()
        .map(|((x, y), _)| (x.to_bits(), y.to_bits()))
        .collect();
    if sources.len() != src_x.len() * src_y.len() {
        let missing = src_y
            .iter()
            .flat_map(|&y| src_x.iter().map(move |&x| (x, y)))
            .find(|(x, y)| !sources.contains(&(x.to_bits(), y.to_bits())))
            .expect("fewer sources than crossings leaves a crossing uncovered");
        return Some(format!(
            "the bus sources span {}x{} tones but cover only {} of their {} crossings \
             ({:?} is not a source); an AOD traps at every crossing of its tones, so \
             the sources of a bus must form a complete grid",
            src_x.len(),
            src_y.len(),
            sources.len(),
            src_x.len() * src_y.len(),
            missing
        ));
    }
    None
}

// --- Bus src→dst relation well-formedness ---

/// Check that a bus's src→dst relation is well-formed: unique sources,
/// unique destinations, and no cycles.
///
/// Duplicate sources are rejected because endpoint resolution is positional
/// first-match ([`super::types::Bus::resolve_forward`]), so a duplicated
/// source silently shadows every later pair. Duplicate destinations are
/// rejected because two simultaneous transports into one site cannot both
/// complete. Cycles (including self-loops) are rejected because a bus is a
/// set of explicit transports, never a permutation — see
/// [`ArchSpecError::CyclicBus`]. Overlapping-but-acyclic relations
/// (conveyor chains such as `0→1, 1→2`) are legal.
///
/// `dup_err` selects the error category for duplicate-endpoint errors
/// (`ZoneBus` for per-zone buses, `InterZoneBus` for zone buses); cycles
/// always report as [`ArchSpecError::CyclicBus`]. Pairs are zipped, so a
/// src/dst length mismatch (reported separately) truncates to the shorter
/// side here.
fn check_bus_relation<T>(
    src: &[T],
    dst: &[T],
    label: &str,
    fmt: impl Fn(&T) -> String,
    dup_err: impl Fn(String) -> ArchSpecError,
    errors: &mut Vec<ArchSpecError>,
) where
    T: Copy + Eq + std::hash::Hash,
{
    let mut first_src: HashMap<T, usize> = HashMap::new();
    for (i, s) in src.iter().enumerate() {
        if let Some(&j) = first_src.get(s) {
            errors.push(dup_err(format!(
                "{label}: duplicate src {} at indices {j} and {i} \
                 (first-match resolution silently shadows the later pair)",
                fmt(s)
            )));
        } else {
            first_src.insert(*s, i);
        }
    }

    let mut first_dst: HashMap<T, usize> = HashMap::new();
    for (i, d) in dst.iter().enumerate() {
        if let Some(&j) = first_dst.get(d) {
            errors.push(dup_err(format!(
                "{label}: duplicate dst {} at indices {j} and {i} \
                 (two simultaneous transports into one site cannot both complete)",
                fmt(d)
            )));
        } else {
            first_dst.insert(*d, i);
        }
    }

    // Cycle detection on the src→dst functional graph. With unique sources
    // each node has at most one successor; with duplicates the first pair
    // wins, matching first-match endpoint resolution.
    let mut next: HashMap<T, T> = HashMap::new();
    for (s, d) in src.iter().zip(dst.iter()) {
        next.entry(*s).or_insert(*d);
    }

    // 0 = unvisited (absent), 1 = on the current walk, 2 = finished.
    let mut state: HashMap<T, u8> = HashMap::new();
    for start in src {
        if state.get(start).copied() == Some(2) {
            continue;
        }
        let mut walk: Vec<T> = Vec::new();
        let mut cur = *start;
        loop {
            match state.get(&cur).copied() {
                Some(2) => break,
                Some(1) => {
                    let pos = walk
                        .iter()
                        .position(|n| *n == cur)
                        .expect("a node in state 1 is always on the current walk stack");
                    let cycle: Vec<String> = walk[pos..]
                        .iter()
                        .chain(std::iter::once(&cur))
                        .map(&fmt)
                        .collect();
                    errors.push(ArchSpecError::CyclicBus(format!(
                        "{label}: src→dst relation contains a cycle: {} \
                         (a bus is a set of explicit transports, never a rotation)",
                        cycle.join(" -> ")
                    )));
                    break;
                }
                _ => {
                    state.insert(cur, 1);
                    walk.push(cur);
                    match next.get(&cur) {
                        Some(n) => cur = *n,
                        None => break,
                    }
                }
            }
        }
        for n in walk {
            state.insert(n, 2);
        }
    }
}

// --- Per-zone entangling pair validation ---

/// Word indices in entangling_pairs must be valid, distinct, and no duplicate pairs.
fn check_zone_entangling_pairs(
    zone_idx: usize,
    zone: &super::types::Zone,
    num_words: usize,
    errors: &mut Vec<ArchSpecError>,
) {
    let mut seen: HashSet<[u32; 2]> = HashSet::new();
    // A word has at most one CZ partner per zone. The same word may still be
    // paired in other zones: the word template is spec-wide, pairs are per zone.
    let mut pair_of_word: HashMap<u32, usize> = HashMap::new();
    for (idx, pair) in zone.entangling_pairs.iter().enumerate() {
        let [a, b] = *pair;
        if a as usize >= num_words {
            errors.push(ArchSpecError::EntanglingPair(format!(
                "zone[{}].entangling_pairs[{}]: word ID {} >= num_words ({})",
                zone_idx, idx, a, num_words
            )));
        }
        if b as usize >= num_words {
            errors.push(ArchSpecError::EntanglingPair(format!(
                "zone[{}].entangling_pairs[{}]: word ID {} >= num_words ({})",
                zone_idx, idx, b, num_words
            )));
        }
        if a == b {
            errors.push(ArchSpecError::EntanglingPair(format!(
                "zone[{}].entangling_pairs[{}]: word paired with itself ({})",
                zone_idx, idx, a
            )));
        }
        let normalized = if a <= b { [a, b] } else { [b, a] };
        if !seen.insert(normalized) {
            errors.push(ArchSpecError::EntanglingPair(format!(
                "zone[{}].entangling_pairs[{}]: duplicate pair [{}, {}]",
                zone_idx, idx, a, b
            )));
            // Already reported as a duplicate; don't also flag its words.
            continue;
        }
        if a == b {
            continue;
        }
        for word in [a, b] {
            if let Some(&first) = pair_of_word.get(&word) {
                errors.push(ArchSpecError::EntanglingPair(format!(
                    "zone[{}].entangling_pairs[{}]: word {} is already paired in \
                     entangling_pairs[{}] (a word has at most one partner per zone)",
                    zone_idx, idx, word, first
                )));
            } else {
                pair_of_word.insert(word, idx);
            }
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

/// Verify that no two zones have overlapping bounding boxes in physical
/// (x, y) space. Overlap would produce ambiguous site positions and make
/// inter-zone path generation impossible (#463).
fn check_zone_overlap(spec: &ArchSpec, errors: &mut Vec<ArchSpecError>) {
    let boxes: Vec<(usize, (f64, f64, f64, f64))> = spec
        .zones
        .iter()
        .enumerate()
        .map(|(i, z)| (i, z.grid.bounding_box()))
        .collect();

    for i in 0..boxes.len() {
        for j in (i + 1)..boxes.len() {
            let (zi, (ax_min, ax_max, ay_min, ay_max)) = boxes[i];
            let (zj, (bx_min, bx_max, by_min, by_max)) = boxes[j];

            // Two axis-aligned rectangles overlap iff they overlap on
            // both axes independently.
            let x_overlap = ax_min < bx_max && bx_min < ax_max;
            let y_overlap = ay_min < by_max && by_min < ay_max;

            if x_overlap && y_overlap {
                errors.push(ArchSpecError::Structure(format!(
                    "zone {} and zone {} have overlapping bounding boxes: \
                     zone {} spans x=[{:.6}, {:.6}] y=[{:.6}, {:.6}], \
                     zone {} spans x=[{:.6}, {:.6}] y=[{:.6}, {:.6}]",
                    zi, zj, zi, ax_min, ax_max, ay_min, ay_max, zj, bx_min, bx_max, by_min, by_max,
                )));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::addr::{SiteRef, WordRef, ZonedWordRef};
    use crate::arch::types::{Bus, Grid, Mode, Word, WordBus, Zone};
    use crate::version::Version;

    /// Create a valid two-zone arch spec for testing.
    fn make_valid_two_zone_spec() -> ArchSpec {
        let grid0 = Grid::from_positions(&[0.0, 5.0, 10.0], &[0.0, 3.0]);
        // Zone 1 grid must not overlap zone 0 (x=[0,10], y=[0,3]).
        let grid1 = Grid::from_positions(&[20.0, 27.5, 35.0], &[0.0, 4.0]);

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
                    name: String::new(),
                    grid: grid0,
                    site_buses: vec![Bus {
                        src: vec![SiteRef(0)],
                        dst: vec![SiteRef(1)],
                    }],
                    word_buses: vec![WordBus {
                        src: vec![WordRef(0)],
                        dst: vec![WordRef(1)],
                        site_map: None,
                    }],
                    words_with_site_buses: vec![0, 1],
                    sites_with_word_buses: vec![0],
                    entangling_pairs: vec![[0, 1]],
                },
                Zone {
                    name: String::new(),
                    grid: grid1,
                    site_buses: vec![],
                    word_buses: vec![],
                    words_with_site_buses: vec![],
                    sites_with_word_buses: vec![],
                    entangling_pairs: vec![],
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
            modes: vec![Mode {
                name: "full".to_string(),
                zones: vec![0, 1],
                bitstring_order: vec![],
            }],
            paths: None,
            feed_forward: false,
            atom_reloading: false,
            blockade_radius: None,
            aod_capacity: None,
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

    /// Extend the fixture with a third word so chains (`0→1, 1→2`) are
    /// expressible in word buses.
    fn add_third_word(spec: &mut ArchSpec) {
        spec.words.push(Word {
            sites: vec![[2, 0], [2, 1]],
        });
        spec.zones[0].words_with_site_buses.push(2);
    }

    #[test]
    fn test_validate_overlapping_acyclic_bus_accepted() {
        let mut spec = make_valid_two_zone_spec();
        add_third_word(&mut spec);
        // Conveyor chain 0→1, 1→2: dst set overlaps src set, no cycle.
        spec.zones[0].word_buses = vec![WordBus {
            src: vec![WordRef(0), WordRef(1)],
            dst: vec![WordRef(1), WordRef(2)],
            site_map: None,
        }];
        assert!(spec.validate().is_ok());
    }

    #[test]
    fn test_validate_cyclic_site_bus_rejected() {
        let mut spec = make_valid_two_zone_spec();
        // 0→1, 1→0: a rotation.
        spec.zones[0].site_buses = vec![Bus {
            src: vec![SiteRef(0), SiteRef(1)],
            dst: vec![SiteRef(1), SiteRef(0)],
        }];
        assert!(matches!(
            spec.validate(),
            Err(ref errs) if errs.iter().any(|e| matches!(e, ArchSpecError::CyclicBus(_)))
        ));
    }

    #[test]
    fn test_validate_self_loop_site_bus_rejected() {
        let mut spec = make_valid_two_zone_spec();
        spec.zones[0].site_buses = vec![Bus {
            src: vec![SiteRef(0)],
            dst: vec![SiteRef(0)],
        }];
        assert!(matches!(
            spec.validate(),
            Err(ref errs) if errs.iter().any(|e| matches!(e, ArchSpecError::CyclicBus(_)))
        ));
    }

    #[test]
    fn test_validate_duplicate_bus_src_rejected() {
        let mut spec = make_valid_two_zone_spec();
        add_third_word(&mut spec);
        // Duplicate src 0; acyclic and dst-unique so only the src rule fires.
        spec.zones[0].word_buses = vec![WordBus {
            src: vec![WordRef(0), WordRef(0)],
            dst: vec![WordRef(1), WordRef(2)],
            site_map: None,
        }];
        assert!(matches!(
            spec.validate(),
            Err(ref errs) if errs.iter().any(
                |e| matches!(e, ArchSpecError::ZoneBus(msg) if msg.contains("duplicate src"))
            )
        ));
    }

    #[test]
    fn test_validate_duplicate_bus_dst_rejected() {
        let mut spec = make_valid_two_zone_spec();
        add_third_word(&mut spec);
        // Duplicate dst 1; acyclic and src-unique so only the dst rule fires.
        spec.zones[0].word_buses = vec![WordBus {
            src: vec![WordRef(0), WordRef(2)],
            dst: vec![WordRef(1), WordRef(1)],
            site_map: None,
        }];
        assert!(matches!(
            spec.validate(),
            Err(ref errs) if errs.iter().any(
                |e| matches!(e, ArchSpecError::ZoneBus(msg) if msg.contains("duplicate dst"))
            )
        ));
    }

    #[test]
    fn test_validate_cyclic_zone_bus_rejected() {
        let mut spec = make_valid_two_zone_spec();
        // A 2-cycle across zones: every pair crosses a zone boundary, so the
        // inter-zone rule passes; only the acyclicity rule can catch it.
        let z0 = ZonedWordRef {
            zone_id: 0,
            word_id: 0,
        };
        let z1 = ZonedWordRef {
            zone_id: 1,
            word_id: 0,
        };
        spec.zone_buses = vec![Bus {
            src: vec![z0, z1],
            dst: vec![z1, z0],
        }];
        assert!(matches!(
            spec.validate(),
            Err(ref errs) if errs.iter().any(|e| matches!(e, ArchSpecError::CyclicBus(_)))
        ));
    }

    #[test]
    fn test_validate_entangling_pair_invalid_word() {
        let mut spec = make_valid_two_zone_spec();
        spec.zones[0].entangling_pairs = vec![[0, 99]];
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

    fn site_mapped(src: &[u16], dst: &[u16]) -> ArchSpec {
        let mut spec = make_valid_two_zone_spec();
        spec.zones[0].sites_with_word_buses = vec![0, 1];
        spec.zones[0].word_buses = vec![WordBus::with_site_map(
            vec![WordRef(0)],
            vec![WordRef(1)],
            Bus {
                src: src.iter().copied().map(SiteRef).collect(),
                dst: dst.iter().copied().map(SiteRef).collect(),
            },
        )];
        spec
    }

    fn zone_bus_errors(spec: &ArchSpec) -> Vec<String> {
        match spec.validate() {
            Ok(()) => vec![],
            Err(errs) => errs.into_iter().map(|e| e.to_string()).collect(),
        }
    }

    #[test]
    fn test_validate_word_bus_site_map_valid() {
        assert!(site_mapped(&[0], &[1]).validate().is_ok());
        assert!(site_mapped(&[1], &[0]).validate().is_ok());
    }

    #[test]
    fn test_validate_word_bus_site_map_out_of_range() {
        let errs = zone_bus_errors(&site_mapped(&[0], &[9]));
        assert!(
            errs.iter()
                .any(|e| e.contains("site_map: dst[0] SiteRef(9)")),
            "{errs:?}"
        );
    }

    #[test]
    fn test_validate_word_bus_site_map_not_a_landing_pad() {
        let mut spec = site_mapped(&[0], &[1]);
        spec.zones[0].sites_with_word_buses = vec![0];
        let errs = zone_bus_errors(&spec);
        assert!(
            errs.iter()
                .any(|e| e.contains("site 1 is not in zone 0 sites_with_word_buses")),
            "{errs:?}"
        );
    }

    #[test]
    fn test_validate_word_bus_site_map_length_mismatch() {
        let errs = zone_bus_errors(&site_mapped(&[0, 1], &[1]));
        assert!(
            errs.iter()
                .any(|e| e.contains("site_map: src length (2) != dst length (1)")),
            "{errs:?}"
        );
    }

    #[test]
    fn test_validate_word_bus_site_map_duplicate_dst() {
        let errs = zone_bus_errors(&site_mapped(&[0, 1], &[1, 1]));
        assert!(
            errs.iter().any(|e| e.contains("site_map: duplicate dst 1")),
            "{errs:?}"
        );
    }

    #[test]
    fn test_validate_word_bus_site_map_cycle_rejected() {
        // A site permutation is a rotation, rejected like a cyclic site bus.
        let result = site_mapped(&[0, 1], &[1, 0]).validate();
        assert!(
            matches!(result, Err(ref errs) if errs.iter().any(|e| matches!(e, ArchSpecError::CyclicBus(msg) if msg.contains("site_map")))),
            "{result:?}"
        );
        let result = site_mapped(&[0], &[0]).validate();
        assert!(
            matches!(result, Err(ref errs) if errs.iter().any(|e| matches!(e, ArchSpecError::CyclicBus(_)))),
            "{result:?}"
        );
    }

    // --- AOD transport ---

    /// One zone on a 5 µm × 3 µm grid with `words.len()` words, each
    /// `nx × ny` sites laid out row-major (site `i` at column `i % nx`, row
    /// `i / nx`), placed side by side along x. Every site is a word-bus
    /// landing pad and every word has site buses; the buses start empty.
    fn aod_spec(num_words: u32, nx: u32, ny: u32) -> ArchSpec {
        let xs: Vec<f64> = (0..num_words * nx).map(|i| f64::from(i) * 5.0).collect();
        let ys: Vec<f64> = (0..ny).map(|j| f64::from(j) * 3.0).collect();
        let words = (0..num_words)
            .map(|w| Word {
                sites: (0..nx * ny).map(|i| [w * nx + i % nx, i / nx]).collect(),
            })
            .collect();
        ArchSpec {
            version: Version::new(2, 0),
            words,
            zones: vec![Zone {
                name: String::new(),
                grid: Grid::from_positions(&xs, &ys),
                site_buses: vec![],
                word_buses: vec![],
                words_with_site_buses: (0..num_words).collect(),
                sites_with_word_buses: (0..nx * ny).collect(),
                entangling_pairs: vec![],
            }],
            zone_buses: vec![],
            modes: vec![Mode {
                name: "all".to_string(),
                zones: vec![0],
                bitstring_order: vec![],
            }],
            paths: None,
            feed_forward: false,
            atom_reloading: false,
            blockade_radius: None,
            aod_capacity: None,
        }
    }

    fn site_bus(src: &[u16], dst: &[u16]) -> Bus<SiteRef> {
        Bus {
            src: src.iter().copied().map(SiteRef).collect(),
            dst: dst.iter().copied().map(SiteRef).collect(),
        }
    }

    fn aod_errors(spec: &ArchSpec) -> Vec<String> {
        match spec.validate() {
            Ok(()) => vec![],
            Err(errs) => errs
                .into_iter()
                .filter_map(|e| match e {
                    ArchSpecError::AodTransport(msg) => Some(msg),
                    _ => None,
                })
                .collect(),
        }
    }

    #[test]
    fn test_aod_site_bus_translation_and_conveyor_accepted() {
        let mut spec = aod_spec(1, 3, 2);
        // A diagonal translation of a whole row, and a conveyor chain 0→1→2.
        spec.zones[0].site_buses = vec![site_bus(&[0, 1], &[4, 5]), site_bus(&[0, 1], &[1, 2])];
        assert_eq!(spec.validate(), Ok(()));
    }

    #[test]
    fn test_aod_site_bus_row_onto_diagonal_rejected() {
        // Sites 0 and 1 share a row but land in different rows: one y tone
        // would have to split in two.
        let mut spec = aod_spec(1, 3, 2);
        spec.zones[0].site_buses = vec![site_bus(&[0, 1], &[2, 3])];
        let errs = aod_errors(&spec);
        assert_eq!(errs.len(), 1, "{errs:?}");
        assert!(errs[0].starts_with("zone 0, site_bus 0: "), "{errs:?}");
        assert!(
            errs[0].contains("2x1 grid of tones at its source but 2x2"),
            "{errs:?}"
        );
    }

    #[test]
    fn test_aod_site_bus_crossing_columns_rejected() {
        // Columns 0 and 1 land on columns 2 and 1: the x tones would cross.
        let mut spec = aod_spec(1, 3, 2);
        spec.zones[0].site_buses = vec![site_bus(&[0, 1], &[5, 4])];
        let errs = aod_errors(&spec);
        assert_eq!(errs.len(), 1, "{errs:?}");
        assert!(errs[0].contains("cannot cross"), "{errs:?}");
    }

    #[test]
    fn test_aod_site_bus_merging_rows_rejected() {
        // Rows 0 and 1 both land on row 0: two y tones would merge.
        let mut spec = aod_spec(1, 3, 2);
        spec.zones[0].site_buses = vec![site_bus(&[0, 4], &[1, 2])];
        let errs = aod_errors(&spec);
        assert_eq!(errs.len(), 1, "{errs:?}");
        assert!(errs[0].contains("cannot add, drop or merge"), "{errs:?}");
    }

    #[test]
    fn test_aod_sparse_site_bus_rejected() {
        // Sites 0 and 4 are a diagonal of a 3x2 word: a translation, but its
        // tones also cross at the unused sites 1 and 3.
        let mut spec = aod_spec(1, 3, 2);
        spec.zones[0].site_buses = vec![site_bus(&[0, 4], &[1, 5])];
        let errs = aod_errors(&spec);
        assert_eq!(errs.len(), 1, "{errs:?}");
        assert!(errs[0].contains("complete grid"), "{errs:?}");
    }

    #[test]
    fn test_aod_sparse_word_bus_sites_rejected() {
        let mut spec = aod_spec(2, 3, 2);
        // Without a map the bus runs on `sites_with_word_buses`, a diagonal.
        spec.zones[0].sites_with_word_buses = vec![0, 4];
        spec.zones[0].word_buses = vec![WordBus::new(vec![WordRef(0)], vec![WordRef(1)])];
        let errs = aod_errors(&spec);
        assert_eq!(errs.len(), 1, "{errs:?}");
        assert!(errs[0].contains("complete grid"), "{errs:?}");

        // With a full pad set, a diagonal map domain is just as sparse.
        spec.zones[0].sites_with_word_buses = (0..6).collect();
        spec.zones[0].word_buses[0].site_map = Some(site_bus(&[0, 4], &[1, 5]));
        let errs = aod_errors(&spec);
        assert_eq!(errs.len(), 1, "{errs:?}");
        assert!(errs[0].contains("complete grid"), "{errs:?}");

        // A whole row is a complete grid.
        spec.zones[0].word_buses[0].site_map = Some(site_bus(&[0, 1], &[4, 5]));
        assert_eq!(spec.validate(), Ok(()));
    }

    #[test]
    fn test_aod_site_map_follows_the_map() {
        let mut spec = aod_spec(2, 2, 2);
        // Bottom row of word 0 onto the top row of word 1: a translation.
        spec.zones[0].word_buses = vec![WordBus::with_site_map(
            vec![WordRef(0)],
            vec![WordRef(1)],
            site_bus(&[0, 1], &[2, 3]),
        )];
        assert_eq!(spec.validate(), Ok(()));

        // Bottom row onto a diagonal of word 1.
        spec.zones[0].word_buses[0].site_map = Some(site_bus(&[0, 1], &[1, 2]));
        let errs = aod_errors(&spec);
        assert_eq!(errs.len(), 1, "{errs:?}");
        assert!(errs[0].starts_with("zone 0, word_bus 0: "), "{errs:?}");
    }

    #[test]
    fn test_aod_site_map_twist_rejected() {
        // The 2x2 block {0, 1, 4, 5} lands on the 2x2 block {2, 3, 6, 7}, so
        // both ends are complete grids, but sites 0 and 1 share a row and land
        // in different rows. A destination-grid check alone accepts this.
        let mut spec = aod_spec(2, 4, 2);
        spec.zones[0].word_buses = vec![WordBus::with_site_map(
            vec![WordRef(0)],
            vec![WordRef(1)],
            site_bus(&[0, 1, 4, 5], &[2, 7, 6, 3]),
        )];
        let errs = aod_errors(&spec);
        assert_eq!(errs.len(), 1, "{errs:?}");
        assert!(errs[0].contains("cannot cross"), "{errs:?}");
    }

    #[test]
    fn test_aod_word_bus_word_pairs_must_not_cross() {
        // Words 0, 1 → 2, 1 is fine (a chain); 0, 1 → 2, 0 would send word 1's
        // column left past word 0's.
        let mut spec = aod_spec(3, 1, 2);
        spec.zones[0].word_buses = vec![WordBus::new(
            vec![WordRef(0), WordRef(1)],
            vec![WordRef(1), WordRef(2)],
        )];
        assert_eq!(spec.validate(), Ok(()));

        spec.zones[0].word_buses = vec![WordBus::new(
            vec![WordRef(0), WordRef(1)],
            vec![WordRef(2), WordRef(0)],
        )];
        let errs = aod_errors(&spec);
        assert_eq!(errs.len(), 1, "{errs:?}");
    }

    #[test]
    fn test_aod_zone_bus_crossing_rejected() {
        let mut spec = make_valid_two_zone_spec();
        // Zone 0 words 0, 1 sit at x = 0, 5; zone 1 words 1, 0 sit at
        // x = 27.5, 20, so the two columns would cross.
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
            dst: vec![
                ZonedWordRef {
                    zone_id: 1,
                    word_id: 1,
                },
                ZonedWordRef {
                    zone_id: 1,
                    word_id: 0,
                },
            ],
        }];
        let errs = aod_errors(&spec);
        assert_eq!(errs.len(), 1, "{errs:?}");
        assert!(
            errs[0].starts_with("zone_bus 0 (lanes from zone 0): "),
            "{errs:?}"
        );

        // In order, the same pairs are one transport.
        spec.zone_buses[0].dst.swap(0, 1);
        assert_eq!(spec.validate(), Ok(()));
    }

    #[test]
    fn test_validate_word_bus_invalid_word_ref() {
        let mut spec = make_valid_two_zone_spec();
        spec.zones[0].word_buses = vec![WordBus {
            src: vec![WordRef(0)],
            dst: vec![WordRef(99)],
            site_map: None,
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
    fn test_validate_entangling_pair_duplicate() {
        let mut spec = make_valid_two_zone_spec();
        spec.zones[0].entangling_pairs = vec![[0, 1], [1, 0]]; // same pair reversed
        assert!(matches!(
            spec.validate(),
            Err(ref errs) if errs.iter().any(|e| matches!(e, ArchSpecError::EntanglingPair(msg) if msg.contains("duplicate")))
        ));
    }

    /// `make_valid_two_zone_spec` with a third word, so overlapping pairs
    /// like `[[0, 1], [1, 2]]` can be expressed.
    fn make_valid_three_word_spec() -> ArchSpec {
        let mut spec = make_valid_two_zone_spec();
        spec.words.push(Word {
            sites: vec![[2, 0], [2, 1]],
        });
        spec
    }

    #[test]
    fn test_validate_entangling_pair_word_in_two_pairs() {
        let mut spec = make_valid_three_word_spec();
        assert!(spec.validate().is_ok());
        spec.zones[0].entangling_pairs = vec![[0, 1], [1, 2]];
        let errs = spec.validate().unwrap_err();
        let overlap: Vec<_> = errs
            .iter()
            .filter(|e| matches!(e, ArchSpecError::EntanglingPair(msg) if msg.contains("already paired")))
            .collect();
        assert_eq!(overlap.len(), 1, "{errs:?}");
        assert_eq!(
            overlap[0].to_string(),
            "zone[0].entangling_pairs[1]: word 1 is already paired in \
             entangling_pairs[0] (a word has at most one partner per zone)"
        );
    }

    #[test]
    fn test_validate_entangling_pair_reports_every_overlapping_word() {
        let mut spec = make_valid_three_word_spec();
        spec.zones[0].entangling_pairs = vec![[0, 1], [1, 2], [2, 0]];
        let errs = spec.validate().unwrap_err();
        let overlap = errs
            .iter()
            .filter(|e| matches!(e, ArchSpecError::EntanglingPair(msg) if msg.contains("already paired")))
            .count();
        // [1, 2] re-pairs word 1; [2, 0] re-pairs words 2 and 0.
        assert_eq!(overlap, 3, "{errs:?}");
    }

    #[test]
    fn test_validate_entangling_pair_same_word_in_different_zones() {
        let mut spec = make_valid_three_word_spec();
        spec.zones[0].entangling_pairs = vec![[0, 1]];
        spec.zones[1].entangling_pairs = vec![[1, 2]];
        assert!(spec.validate().is_ok());
    }

    #[test]
    fn test_validate_entangling_pair_duplicate_not_reported_as_overlap() {
        let mut spec = make_valid_two_zone_spec();
        spec.zones[0].entangling_pairs = vec![[0, 1], [1, 0]];
        let errs = spec.validate().unwrap_err();
        assert_eq!(errs.len(), 1, "{errs:?}");
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
        spec.zones[0].entangling_pairs = vec![[0, 99]]; // bad word
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

    // ── Zone overlap tests (#463) ──

    #[test]
    fn test_non_overlapping_zones_pass() {
        // The default fixture has non-overlapping zones (zone 0 at x=[0,10],
        // zone 1 at x=[20,35]).
        let spec = make_valid_two_zone_spec();
        assert!(spec.validate().is_ok());
    }

    #[test]
    fn test_overlapping_zones_rejected() {
        let mut spec = make_valid_two_zone_spec();
        // Move zone 1's grid to overlap zone 0 (x=[0,10], y=[0,3]).
        spec.zones[1].grid = Grid::from_positions(&[5.0, 12.5, 20.0], &[0.0, 4.0]);
        assert!(matches!(
            spec.validate(),
            Err(ref errs) if errs.iter().any(|e| matches!(e, ArchSpecError::Structure(msg) if msg.contains("overlapping bounding boxes")))
        ));
    }

    #[test]
    fn test_adjacent_zones_not_overlapping() {
        let mut spec = make_valid_two_zone_spec();
        // Zone 1 starts exactly where zone 0 ends (touching but not overlapping).
        spec.zones[1].grid = Grid::from_positions(&[10.0, 17.5, 25.0], &[0.0, 4.0]);
        assert!(spec.validate().is_ok());
    }

    #[test]
    fn test_shared_x_range_with_y_overlap_rejected() {
        let mut spec = make_valid_two_zone_spec();
        // Zones share an x range and have overlapping y ranges.
        spec.zones[1].grid = Grid::from_positions(&[0.0, 7.5, 15.0], &[1.0, 5.0]);
        // Zone 0: x=[0,10], y=[0,3]; Zone 1: x=[0,15], y=[1,5] → both axes overlap.
        assert!(matches!(
            spec.validate(),
            Err(ref errs) if errs.iter().any(|e| matches!(e, ArchSpecError::Structure(msg) if msg.contains("overlapping")))
        ));
    }

    #[test]
    fn test_single_zone_no_overlap_check() {
        // With a single zone, there's nothing to compare.
        let mut spec = make_valid_two_zone_spec();
        spec.zones.pop();
        spec.zone_buses.clear();
        spec.modes[0].zones = vec![0];
        assert!(spec.validate().is_ok());
    }
}
