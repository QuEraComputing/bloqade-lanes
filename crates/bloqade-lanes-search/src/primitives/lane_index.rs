//! Precomputed lane lookups from an architecture specification.
//!
//! [`LaneIndex`] builds all lane-related indexes once at construction time,
//! avoiding repeated computation during search. This is a direct port of
//! Python's `ConfigurationTree._build_lane_indexes()`.

use std::collections::HashMap;

use bloqade_lanes_bytecode_core::arch::addr::{Direction, LaneAddr, LocationAddr, MoveType};
use bloqade_lanes_bytecode_core::arch::types::ArchSpec;

use crate::primitives::bus_grid_maps::BusGridMaps;
use crate::primitives::ordering::TripletKey;

/// Precomputed lane lookups for an architecture.
///
/// Built once from an [`ArchSpec`] and reused across multiple searches.
/// Caches all lane addresses, their endpoints, and location positions.
///
/// `Clone` is implemented so callers can wrap the index in an `Arc` and
/// share it across closures (e.g. `LooseTargetGenerator`); the contents
/// are immutable hash maps so cloning is `O(n)` in entry count.
#[derive(Debug, Clone)]
pub struct LaneIndex {
    arch_spec: ArchSpec,
    /// (MoveType, bus_id, zone_id, Direction) → lanes for that triplet.
    lanes_by_triplet: HashMap<(MoveType, u32, u32, Direction), Vec<LaneAddr>>,
    /// (MoveType, bus_id, zone_id, Direction) → { encoded_src → LaneAddr }.
    lane_by_src: HashMap<(MoveType, u32, u32, Direction), HashMap<u64, LaneAddr>>,
    /// encoded_src → all outgoing lanes from that location.
    outgoing_by_src: HashMap<u64, Vec<LaneAddr>>,
    /// encoded LaneAddr (u64) → (src, dst) endpoints.
    endpoints: HashMap<u64, (LocationAddr, LocationAddr)>,
    /// encoded LocationAddr (u64) → (x, y) physical position.
    positions: HashMap<u64, (f64, f64)>,
    /// encoded LaneAddr (u64) → duration in microseconds (from transport paths).
    lane_durations: HashMap<u64, f64>,
    /// Fastest lane duration across all lanes with paths. `None` if no paths.
    fastest_lane_duration: Option<f64>,
    /// Precomputed AOD-grid lookup maps per [`TripletKey`]
    /// (`move_type, bus_id, direction`) bus group, spanning all zones.
    /// Occupancy-independent, so they are built once here and borrowed by every
    /// `BusGridContext` for that group (see [`BusGridMaps`]).
    bus_grid_maps: HashMap<TripletKey, BusGridMaps>,
}

impl LaneIndex {
    /// Build a lane index from a borrowed architecture specification.
    ///
    /// Clones the spec once internally — useful when the caller already
    /// holds a borrowed [`ArchSpec`] (e.g. through a wrapper) and would
    /// otherwise pay a JSON round-trip to materialize an owned copy.
    pub fn from_arch_spec(arch_spec: &ArchSpec) -> Self {
        Self::new(arch_spec.clone())
    }

    /// Build a lane index from an architecture specification.
    ///
    /// Iterates all zones and their buses, computes lane addresses,
    /// resolves endpoints, and lazily caches positions as endpoints
    /// are discovered.
    pub fn new(arch_spec: ArchSpec) -> Self {
        let mut lanes_by_triplet: HashMap<(MoveType, u32, u32, Direction), Vec<LaneAddr>> =
            HashMap::new();
        let mut lane_by_src: HashMap<(MoveType, u32, u32, Direction), HashMap<u64, LaneAddr>> =
            HashMap::new();
        let mut outgoing_by_src: HashMap<u64, Vec<LaneAddr>> = HashMap::new();
        let mut endpoints: HashMap<u64, (LocationAddr, LocationAddr)> = HashMap::new();
        let mut positions: HashMap<u64, (f64, f64)> = HashMap::new();

        // Helper: register a lane in all indexes and cache endpoint positions.
        let mut register_lane = |lane: LaneAddr,
                                 bus_id: u32,
                                 zone_id: u32,
                                 direction: Direction,
                                 mt: MoveType,
                                 positions: &mut HashMap<u64, (f64, f64)>,
                                 arch_spec: &ArchSpec| {
            if let Some((src, dst)) = arch_spec.lane_endpoints(&lane) {
                let encoded_lane = lane.encode_u64();
                endpoints.insert(encoded_lane, (src, dst));
                let src_enc = src.encode();
                let key = (mt, bus_id, zone_id, direction);
                lanes_by_triplet.entry(key).or_default().push(lane);
                lane_by_src.entry(key).or_default().insert(src_enc, lane);
                outgoing_by_src.entry(src_enc).or_default().push(lane);

                // Lazily cache positions for discovered endpoints.
                if let std::collections::hash_map::Entry::Vacant(e) = positions.entry(src.encode())
                    && let Some(pos) = arch_spec.location_position(&src)
                {
                    e.insert(pos);
                }
                if let std::collections::hash_map::Entry::Vacant(e) = positions.entry(dst.encode())
                    && let Some(pos) = arch_spec.location_position(&dst)
                {
                    e.insert(pos);
                }
            }
        };

        // Iterate zones; each zone owns its grid, site buses, and word buses.
        for (zone_idx, zone) in arch_spec.zones.iter().enumerate() {
            let zone_id = zone_idx as u32;

            // Site buses: iterate (bus_id, word_id, site_id).
            for (bus_idx, bus) in zone.site_buses.iter().enumerate() {
                let bus_id = bus_idx as u32;
                for direction in [Direction::Forward, Direction::Backward] {
                    for &word_id in &zone.words_with_site_buses {
                        for src_ref in &bus.src {
                            let site_id = src_ref.0 as u32;
                            let lane = LaneAddr {
                                move_type: MoveType::SiteBus,
                                zone_id,
                                word_id,
                                site_id,
                                bus_id,
                                direction,
                            };
                            register_lane(
                                lane,
                                bus_id,
                                zone_id,
                                direction,
                                MoveType::SiteBus,
                                &mut positions,
                                &arch_spec,
                            );
                        }
                    }
                }
            }

            // Word buses: iterate (bus_id, word_id from bus.src, site_id from zone.sites_with_word_buses).
            for (bus_idx, bus) in zone.word_buses.iter().enumerate() {
                let bus_id = bus_idx as u32;
                for direction in [Direction::Forward, Direction::Backward] {
                    for src_ref in &bus.src {
                        let word_id = src_ref.0 as u32;
                        for &site_id in &zone.sites_with_word_buses {
                            let lane = LaneAddr {
                                move_type: MoveType::WordBus,
                                zone_id,
                                word_id,
                                site_id,
                                bus_id,
                                direction,
                            };
                            register_lane(
                                lane,
                                bus_id,
                                zone_id,
                                direction,
                                MoveType::WordBus,
                                &mut positions,
                                &arch_spec,
                            );
                        }
                    }
                }
            }
        }

        // Zone buses: inter-zone word movement. Unlike site/word buses these
        // live on the spec itself (not per-zone), so they are registered after
        // the per-zone loop. Each `(src_ref, dst_ref)` pair moves a word across
        // a zone boundary; the destination zone/word is derived by
        // `lane_endpoints` from the zone-bus table, so we only encode the
        // forward source here. Mirrors Python's `PathFinder` zone-bus loop.
        //
        // Both the source and destination word sets of a zone bus are validated
        // to form AOD-compatible rectangles at arch-build time (see
        // `ArchBuilder.connect`), so the AOD rectangle builder can treat zone
        // buses exactly like intra-zone buses.
        let sites_per_word = arch_spec
            .words
            .iter()
            .map(|w| w.sites.len())
            .max()
            .unwrap_or(0) as u32;
        for (bus_idx, bus) in arch_spec.zone_buses.iter().enumerate() {
            let bus_id = bus_idx as u32;
            for direction in [Direction::Forward, Direction::Backward] {
                for src_ref in &bus.src {
                    let zone_id = src_ref.zone_id as u32;
                    let word_id = src_ref.word_id as u32;
                    for site_id in 0..sites_per_word {
                        let lane = LaneAddr {
                            move_type: MoveType::ZoneBus,
                            zone_id,
                            word_id,
                            site_id,
                            bus_id,
                            direction,
                        };
                        register_lane(
                            lane,
                            bus_id,
                            zone_id,
                            direction,
                            MoveType::ZoneBus,
                            &mut positions,
                            &arch_spec,
                        );
                    }
                }
            }
        }

        // Build lane duration cache from transport paths.
        let mut lane_durations: HashMap<u64, f64> = HashMap::new();
        let mut fastest: Option<f64> = None;
        if let Some(paths) = &arch_spec.paths {
            for tp in paths {
                let duration = compute_lane_duration_us(&tp.waypoints);
                if duration > 0.0 {
                    lane_durations.insert(tp.lane, duration);
                    fastest = Some(fastest.map_or(duration, |f: f64| f.min(duration)));
                }
            }
        }

        // Sort each outgoing-lane Vec by encoded lane address for deterministic
        // iteration order in score_moveset / mobility computation.
        for v in outgoing_by_src.values_mut() {
            v.sort_unstable_by_key(|lane| lane.encode_u64());
        }

        let mut index = Self {
            arch_spec,
            lanes_by_triplet,
            lane_by_src,
            outgoing_by_src,
            endpoints,
            positions,
            lane_durations,
            fastest_lane_duration: fastest,
            bus_grid_maps: HashMap::new(),
        };
        index.build_bus_grid_cache();
        index
    }

    /// Precompute the occupancy-independent AOD-grid maps for every bus group.
    ///
    /// Runs once at construction after all lane/endpoint/position indexes are
    /// populated. Each `BusGridContext` for the all-zones case then borrows
    /// the cached maps instead of rebuilding them (the entropy driver's hot
    /// path builds one context per bus-triplet group, many times per solve).
    fn build_bus_grid_cache(&mut self) {
        let groups: Vec<(MoveType, u32, Direction)> = self.bus_groups_no_zone().collect();
        let mut cache = HashMap::with_capacity(groups.len());
        for (mt, bus_id, dir) in groups {
            // `from_lanes` consumes the iterator directly — both borrows of
            // `self` are shared, so no intermediate `Vec` is needed.
            let maps =
                BusGridMaps::from_lanes(self, self.lanes_for_all_zones(mt, bus_id, dir).copied());
            cache.insert(TripletKey::new(mt, bus_id, dir), maps);
        }
        self.bus_grid_maps = cache;
    }

    /// Get the underlying architecture specification.
    pub fn arch_spec(&self) -> &ArchSpec {
        &self.arch_spec
    }

    /// Number of distinct locations in the architecture.
    pub fn num_locations(&self) -> usize {
        self.positions.len()
    }

    /// Iterate the encoded endpoints (src, dst) of every registered lane.
    ///
    /// Yields duplicates; callers dedupe. Unlike `positions`, this covers
    /// every registered endpoint regardless of whether its grid position
    /// resolves. Note it is still a strict subset of what `DistanceTable`
    /// interns: the distance table additionally interns isolated targets
    /// with no incident lanes (so `distance(t, t) = 0` works). Consumers
    /// that must answer for every distance-table location (e.g. the entropy
    /// `HeuristicTables`) handle those targets separately.
    pub(crate) fn lane_endpoint_encs(&self) -> impl Iterator<Item = u64> + '_ {
        self.endpoints
            .values()
            .flat_map(|(src, dst)| [src.encode(), dst.encode()])
    }

    /// Get all lanes for a `(move_type, bus_id, zone_id, direction)` triplet.
    pub fn lanes_for(
        &self,
        mt: MoveType,
        bus_id: u32,
        zone_id: u32,
        dir: Direction,
    ) -> &[LaneAddr] {
        self.lanes_by_triplet
            .get(&(mt, bus_id, zone_id, dir))
            .map_or(&[], |v| v.as_slice())
    }

    /// Get the lane originating from a specific source for a triplet.
    pub fn lane_for_source(
        &self,
        mt: MoveType,
        bus_id: u32,
        zone_id: u32,
        dir: Direction,
        src: LocationAddr,
    ) -> Option<LaneAddr> {
        self.lane_by_src
            .get(&(mt, bus_id, zone_id, dir))
            .and_then(|m| m.get(&src.encode()).copied())
    }

    /// Get all outgoing lanes from a location.
    pub fn outgoing_lanes(&self, src: LocationAddr) -> &[LaneAddr] {
        self.outgoing_by_src
            .get(&src.encode())
            .map_or(&[], |v| v.as_slice())
    }

    /// Get cached endpoints for a lane. Returns `None` if the lane is unknown.
    pub fn endpoints(&self, lane: &LaneAddr) -> Option<(LocationAddr, LocationAddr)> {
        self.endpoints.get(&lane.encode_u64()).copied()
    }

    /// Get cached physical position for a location.
    pub fn position(&self, loc: LocationAddr) -> Option<(f64, f64)> {
        self.positions.get(&loc.encode()).copied()
    }

    /// Iterate all `(move_type, bus_id, zone_id, direction)` bus groups that have lanes.
    pub fn bus_groups(&self) -> impl Iterator<Item = (MoveType, u32, u32, Direction)> + '_ {
        self.lanes_by_triplet.keys().copied()
    }

    /// Iterate all bus groups that have lanes.
    #[deprecated(note = "renamed to bus_groups() — triplets is misleading for 4-tuples")]
    pub fn triplets(&self) -> impl Iterator<Item = (MoveType, u32, u32, Direction)> + '_ {
        self.bus_groups()
    }

    /// Get all lanes for a bus across all zones.
    pub fn lanes_for_all_zones(
        &self,
        mt: MoveType,
        bus_id: u32,
        dir: Direction,
    ) -> impl Iterator<Item = &LaneAddr> + '_ {
        self.lanes_by_triplet
            .iter()
            .filter(move |&(&(m, b, _, d), _)| m == mt && b == bus_id && d == dir)
            .flat_map(|(_, lanes)| lanes.iter())
    }

    /// Iterate distinct `(move_type, bus_id, direction)` bus groups (ignoring zone).
    pub fn bus_groups_no_zone(&self) -> impl Iterator<Item = (MoveType, u32, Direction)> + '_ {
        let mut seen = std::collections::HashSet::new();
        self.lanes_by_triplet
            .keys()
            .filter_map(move |&(mt, bus_id, _zone_id, dir)| {
                if seen.insert((mt, bus_id, dir)) {
                    Some((mt, bus_id, dir))
                } else {
                    None
                }
            })
    }

    /// Get cached lane duration in microseconds. Returns `None` if the lane
    /// has no transport path data.
    pub fn lane_duration_us(&self, lane: &LaneAddr) -> Option<f64> {
        self.lane_durations.get(&lane.encode_u64()).copied()
    }

    /// Get the fastest (minimum) lane duration across all lanes with paths.
    /// Returns `None` if no lanes have path data.
    pub fn fastest_lane_duration_us(&self) -> Option<f64> {
        self.fastest_lane_duration
    }

    /// Borrow the precomputed all-zones AOD-grid maps for a bus group.
    ///
    /// Returns `None` if the group has no lanes. Used by
    /// [`BusGridContext::new`](crate::ops::aod_grid) to avoid rebuilding the
    /// occupancy-independent lookup maps on every call.
    pub(crate) fn bus_grid_maps(
        &self,
        mt: MoveType,
        bus_id: u32,
        dir: Direction,
    ) -> Option<&BusGridMaps> {
        self.bus_grid_maps.get(&TripletKey::new(mt, bus_id, dir))
    }
}

// ── FLAIR timing model ────────────────────────────────────────────

/// Constants from bloqade-flair's constant-jerk motion model.
const FLAIR_MAX_RAMP_US: f64 = 0.2;
const FLAIR_MAX_JERK_UM_PER_US3: f64 = 0.0004;
const FLAIR_MAX_ACCEL_UM_PER_US2: f64 = 0.0015;

/// Minimum duration (µs) for a constant-jerk move over `max_dist_um`.
///
/// Port of Python `MoveMetricCalculator._const_jerk_min_duration_us`.
fn const_jerk_min_duration_us(max_dist_um: f64) -> f64 {
    let max_dist_um = max_dist_um.abs();
    if max_dist_um < 1e-8 {
        return 0.0;
    }

    let t1 = FLAIR_MAX_ACCEL_UM_PER_US2 / FLAIR_MAX_JERK_UM_PER_US3;
    let a = FLAIR_MAX_JERK_UM_PER_US3 * t1;
    let b = 3.0 * FLAIR_MAX_JERK_UM_PER_US3 * t1 * t1;
    let c = 2.0 * FLAIR_MAX_JERK_UM_PER_US3 * t1 * t1 * t1 - max_dist_um;

    if c >= 0.0 {
        let t1_jerk = (max_dist_um / (2.0 * FLAIR_MAX_JERK_UM_PER_US3)).cbrt();
        return 4.0 * t1_jerk;
    }

    let discriminant = b * b - 4.0 * a * c;
    let t2 = (-b + discriminant.sqrt()) / (2.0 * a);
    4.0 * t1 + 2.0 * t2
}

/// Compute lane duration from waypoints: ramp + sum(segment durations) + ramp.
fn compute_lane_duration_us(waypoints: &[[f64; 2]]) -> f64 {
    if waypoints.len() <= 1 {
        return 0.0;
    }
    // Assumes unit amplitude for search-time cost estimation;
    // exact timing uses FLAIR bytecode values.
    let ramp = 1.0 / FLAIR_MAX_RAMP_US;
    let segment_sum: f64 = waypoints
        .windows(2)
        .map(|w| {
            let dx = w[1][0] - w[0][0];
            let dy = w[1][1] - w[0][1];
            let dist = (dx * dx + dy * dy).sqrt();
            const_jerk_min_duration_us(dist)
        })
        .sum();
    ramp + segment_sum + ramp
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::example_arch_json;

    fn make_index() -> LaneIndex {
        let spec: ArchSpec = serde_json::from_str(example_arch_json()).unwrap();
        LaneIndex::new(spec)
    }

    #[test]
    fn construction_succeeds() {
        let index = make_index();
        // Should have lanes for site bus and word bus
        assert!(!index.lanes_by_triplet.is_empty());
    }

    #[test]
    fn site_bus_forward_lanes() {
        let index = make_index();
        // Site bus 0, zone 0, forward: src=[0,1,2,3,4] in words [0,1] → 5*2=10 lanes
        let lanes = index.lanes_for(MoveType::SiteBus, 0, 0, Direction::Forward);
        assert_eq!(lanes.len(), 10);
    }

    #[test]
    fn zone_bus_edges_are_registered() {
        // Regression for #845: inter-zone `zone_buses` must appear as edges in
        // the search graph, mirroring Python's PathFinder.
        let spec: ArchSpec =
            serde_json::from_str(crate::test_utils::two_zone_bus_arch_json()).unwrap();
        let index = LaneIndex::new(spec);

        // The zone bus connects (zone 1, word 1, site 0) -> (zone 0, word 0, site 0).
        let src = LocationAddr {
            zone_id: 1,
            word_id: 1,
            site_id: 0,
        };
        let dst = LocationAddr {
            zone_id: 0,
            word_id: 0,
            site_id: 0,
        };

        // A forward ZoneBus lane should originate from the memory-zone source.
        let outgoing = index.outgoing_lanes(src);
        let zone_lane = outgoing
            .iter()
            .find(|l| l.move_type == MoveType::ZoneBus)
            .copied()
            .expect("zone bus lane must be present in outgoing edges");

        let (lsrc, ldst) = index.endpoints(&zone_lane).unwrap();
        assert_eq!(lsrc, src);
        assert_eq!(ldst, dst);

        // The reverse (backward) edge must also exist so the graph is traversable
        // in both directions, matching PathFinder's forward+reverse edge pair.
        let back = index.outgoing_lanes(dst);
        assert!(
            back.iter().any(|l| {
                l.move_type == MoveType::ZoneBus && index.endpoints(l).map(|(_, d)| d) == Some(src)
            }),
            "reverse zone bus edge (gate -> memory) must be present"
        );
    }

    #[test]
    fn word_bus_forward_lanes() {
        let index = make_index();
        // Word bus 0, zone 0, forward: src=[0], sites_with_word_buses=[5,6,7,8,9] → 5 lanes
        let lanes = index.lanes_for(MoveType::WordBus, 0, 0, Direction::Forward);
        assert_eq!(lanes.len(), 5);
    }

    #[test]
    fn endpoints_match_arch_spec() {
        let index = make_index();
        // Site bus 0, zone 0, forward, word 0, site 0 → should go to site 5
        let lane = LaneAddr {
            move_type: MoveType::SiteBus,
            zone_id: 0,
            word_id: 0,
            site_id: 0,
            bus_id: 0,
            direction: Direction::Forward,
        };
        let (src, dst) = index.endpoints(&lane).unwrap();
        assert_eq!(
            src,
            LocationAddr {
                zone_id: 0,
                word_id: 0,
                site_id: 0
            }
        );
        assert_eq!(
            dst,
            LocationAddr {
                zone_id: 0,
                word_id: 0,
                site_id: 5
            }
        );
    }

    #[test]
    fn position_cached() {
        let index = make_index();
        let loc = LocationAddr {
            zone_id: 0,
            word_id: 0,
            site_id: 0,
        };
        let pos = index.position(loc).unwrap();
        // Word 0, site 0 is at grid index [0,0] → x=1.0, y=2.5
        assert_eq!(pos, (1.0, 2.5));
    }

    #[test]
    fn lane_for_source_found() {
        let index = make_index();
        let src = LocationAddr {
            zone_id: 0,
            word_id: 0,
            site_id: 0,
        };
        let lane = index.lane_for_source(MoveType::SiteBus, 0, 0, Direction::Forward, src);
        assert!(lane.is_some());
        let lane = lane.unwrap();
        assert_eq!(lane.word_id, 0);
        assert_eq!(lane.site_id, 0);
    }

    #[test]
    fn lane_for_source_not_found() {
        let index = make_index();
        // Site 5 is a destination, not a source for forward
        let src = LocationAddr {
            zone_id: 0,
            word_id: 0,
            site_id: 5,
        };
        let lane = index.lane_for_source(MoveType::SiteBus, 0, 0, Direction::Forward, src);
        assert!(lane.is_none());
    }

    #[test]
    fn outgoing_lanes_nonempty() {
        let index = make_index();
        // Site 0 in word 0 is a site bus source (forward) and a backward destination source
        let src = LocationAddr {
            zone_id: 0,
            word_id: 0,
            site_id: 0,
        };
        let outgoing = index.outgoing_lanes(src);
        assert!(!outgoing.is_empty());
    }

    #[test]
    fn outgoing_lanes_empty_for_nonexistent() {
        let index = make_index();
        let src = LocationAddr {
            zone_id: 0,
            word_id: 99,
            site_id: 99,
        };
        assert!(index.outgoing_lanes(src).is_empty());
    }

    #[test]
    fn unknown_lane_returns_none() {
        let index = make_index();
        let lane = LaneAddr {
            move_type: MoveType::SiteBus,
            zone_id: 0,
            word_id: 99,
            site_id: 99,
            bus_id: 99,
            direction: Direction::Forward,
        };
        assert!(index.endpoints(&lane).is_none());
    }

    /// The load-bearing property for bidirectional search: flipping a lane's
    /// `Direction` swaps its endpoints, and the flipped lane is always
    /// registered in the index.
    ///
    /// Checked across every lane of the `example_arch_json` fixture, which
    /// declares `"zone_buses": []` — so `MoveType::ZoneBus` is **not**
    /// covered. That is the one lane kind where `zone_id`/`word_id` name a
    /// different zone than one of the lane's endpoints, i.e. exactly the case
    /// where the flip is least obviously an involution.
    #[test]
    fn flipping_direction_swaps_endpoints_for_every_lane() {
        use bloqade_lanes_dsl_core::primitives::move_set::MoveSet;

        let index = make_index();
        let mut checked = 0usize;

        for (mt, bus_id, zone_id, dir) in index.bus_groups().collect::<Vec<_>>() {
            for &lane in index.lanes_for(mt, bus_id, zone_id, dir) {
                let Some((src, dst)) = index.endpoints(&lane) else {
                    continue;
                };
                let inverted = MoveSet::new([lane]).inverse().decode();
                assert_eq!(inverted.len(), 1, "inverse must preserve lane count");
                let flipped = inverted[0];

                let (inv_src, inv_dst) = index
                    .endpoints(&flipped)
                    .expect("the flipped lane must be registered in the index");

                assert_eq!(
                    inv_src, dst,
                    "inverted source must be the forward destination"
                );
                assert_eq!(
                    inv_dst, src,
                    "inverted destination must be the forward source"
                );
                checked += 1;
            }
        }

        assert!(checked > 0, "the fixture spec must contain lanes to check");
    }
}
