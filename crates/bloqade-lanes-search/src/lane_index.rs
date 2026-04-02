//! Precomputed lane lookups from an architecture specification.
//!
//! [`LaneIndex`] builds all lane-related indexes once at construction time,
//! avoiding repeated computation during search. This is a direct port of
//! Python's `ConfigurationTree._build_lane_indexes()`.

use std::collections::HashMap;

use bloqade_lanes_bytecode_core::arch::addr::{Direction, LaneAddr, LocationAddr, MoveType};
use bloqade_lanes_bytecode_core::arch::types::ArchSpec;

/// Precomputed lane lookups for an architecture.
///
/// Built once from an [`ArchSpec`] and reused across multiple searches.
/// Caches all lane addresses, their endpoints, and location positions.
pub struct LaneIndex {
    arch_spec: ArchSpec,
    /// (MoveType, bus_id, Direction) → lanes for that triplet.
    lanes_by_triplet: HashMap<(MoveType, u32, Direction), Vec<LaneAddr>>,
    /// (MoveType, bus_id, Direction) → { encoded_src → LaneAddr }.
    lane_by_src: HashMap<(MoveType, u32, Direction), HashMap<u32, LaneAddr>>,
    /// encoded_src → all outgoing lanes from that location.
    outgoing_by_src: HashMap<u32, Vec<LaneAddr>>,
    /// encoded LaneAddr (u64) → (src, dst) endpoints.
    endpoints: HashMap<u64, (LocationAddr, LocationAddr)>,
    /// encoded LocationAddr (u32) → (x, y) physical position.
    positions: HashMap<u32, (f64, f64)>,
}

impl LaneIndex {
    /// Build a lane index from an architecture specification.
    ///
    /// Iterates all buses and directions, computes lane addresses,
    /// resolves endpoints, and caches positions.
    pub fn new(arch_spec: ArchSpec) -> Self {
        let mut lanes_by_triplet: HashMap<(MoveType, u32, Direction), Vec<LaneAddr>> =
            HashMap::new();
        let mut lane_by_src: HashMap<(MoveType, u32, Direction), HashMap<u32, LaneAddr>> =
            HashMap::new();
        let mut outgoing_by_src: HashMap<u32, Vec<LaneAddr>> = HashMap::new();
        let mut endpoints: HashMap<u64, (LocationAddr, LocationAddr)> = HashMap::new();
        let mut positions: HashMap<u32, (f64, f64)> = HashMap::new();

        // Cache all location positions.
        let num_words = arch_spec.geometry.words.len() as u32;
        let sites_per_word = arch_spec.geometry.sites_per_word;
        for word_id in 0..num_words {
            for site_id in 0..sites_per_word {
                let loc = LocationAddr { word_id, site_id };
                if let Some(pos) = arch_spec.location_position(&loc) {
                    positions.insert(loc.encode(), pos);
                }
            }
        }

        // Helper: register a lane in all indexes.
        let mut register_lane =
            |lane: LaneAddr, bus_id: u32, direction: Direction, mt: MoveType| {
                if let Some((src, dst)) = arch_spec.lane_endpoints(&lane) {
                    let encoded_lane = lane.encode_u64();
                    endpoints.insert(encoded_lane, (src, dst));
                    let src_enc = src.encode();
                    let key = (mt, bus_id, direction);
                    lanes_by_triplet.entry(key).or_default().push(lane);
                    lane_by_src.entry(key).or_default().insert(src_enc, lane);
                    outgoing_by_src.entry(src_enc).or_default().push(lane);
                }
            };

        // Site buses: iterate (bus_id, word_id, site_id).
        for (bus_id, bus) in arch_spec.buses.site_buses.iter().enumerate() {
            let bus_word_ids = bus
                .words
                .as_ref()
                .cloned()
                .unwrap_or_else(|| arch_spec.words_with_site_buses.clone());
            for direction in [Direction::Forward, Direction::Backward] {
                for &word_id in &bus_word_ids {
                    for &site_id in &bus.src {
                        let lane = LaneAddr {
                            move_type: MoveType::SiteBus,
                            word_id,
                            site_id,
                            bus_id: bus_id as u32,
                            direction,
                        };
                        register_lane(lane, bus_id as u32, direction, MoveType::SiteBus);
                    }
                }
            }
        }

        // Word buses: iterate (bus_id, word_id from bus.src, site_id from sites_with_word_buses).
        for (bus_id, bus) in arch_spec.buses.word_buses.iter().enumerate() {
            for direction in [Direction::Forward, Direction::Backward] {
                for &word_id in &bus.src {
                    for &site_id in &arch_spec.sites_with_word_buses {
                        let lane = LaneAddr {
                            move_type: MoveType::WordBus,
                            word_id,
                            site_id,
                            bus_id: bus_id as u32,
                            direction,
                        };
                        register_lane(lane, bus_id as u32, direction, MoveType::WordBus);
                    }
                }
            }
        }

        Self {
            arch_spec,
            lanes_by_triplet,
            lane_by_src,
            outgoing_by_src,
            endpoints,
            positions,
        }
    }

    /// Get the underlying architecture specification.
    pub fn arch_spec(&self) -> &ArchSpec {
        &self.arch_spec
    }

    /// Get all lanes for a `(move_type, bus_id, direction)` triplet.
    pub fn lanes_for(&self, mt: MoveType, bus_id: u32, dir: Direction) -> &[LaneAddr] {
        self.lanes_by_triplet
            .get(&(mt, bus_id, dir))
            .map_or(&[], |v| v.as_slice())
    }

    /// Get the lane originating from a specific source for a triplet.
    pub fn lane_for_source(
        &self,
        mt: MoveType,
        bus_id: u32,
        dir: Direction,
        src: LocationAddr,
    ) -> Option<LaneAddr> {
        self.lane_by_src
            .get(&(mt, bus_id, dir))
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

    /// Iterate all `(move_type, bus_id, direction)` triplets that have lanes.
    pub fn triplets(&self) -> impl Iterator<Item = (MoveType, u32, Direction)> + '_ {
        self.lanes_by_triplet.keys().copied()
    }

    /// Get the site buses from the architecture.
    pub fn site_buses(&self) -> &[bloqade_lanes_bytecode_core::arch::types::Bus] {
        &self.arch_spec.buses.site_buses
    }

    /// Get the word buses from the architecture.
    pub fn word_buses(&self) -> &[bloqade_lanes_bytecode_core::arch::types::Bus] {
        &self.arch_spec.buses.word_buses
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Same JSON as bloqade-lanes-bytecode-core's example_arch_spec.
    fn example_arch_json() -> &'static str {
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
            "measurement_mode_zones": [0],
            "paths": [
                {"lane": "0xC000000000000005", "waypoints": [[1.0, 15.0], [1.0, 10.0], [1.0, 5.0]]}
            ]
        }"#
    }

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
        // Site bus 0, forward: src=[0,1,2,3,4] in words [0,1] → 5*2=10 lanes
        let lanes = index.lanes_for(MoveType::SiteBus, 0, Direction::Forward);
        assert_eq!(lanes.len(), 10);
    }

    #[test]
    fn word_bus_forward_lanes() {
        let index = make_index();
        // Word bus 0, forward: src=[0], sites_with_word_buses=[5,6,7,8,9] → 5 lanes
        let lanes = index.lanes_for(MoveType::WordBus, 0, Direction::Forward);
        assert_eq!(lanes.len(), 5);
    }

    #[test]
    fn endpoints_match_arch_spec() {
        let index = make_index();
        // Site bus 0, forward, word 0, site 0 → should go to site 5
        let lane = LaneAddr {
            move_type: MoveType::SiteBus,
            word_id: 0,
            site_id: 0,
            bus_id: 0,
            direction: Direction::Forward,
        };
        let (src, dst) = index.endpoints(&lane).unwrap();
        assert_eq!(
            src,
            LocationAddr {
                word_id: 0,
                site_id: 0
            }
        );
        assert_eq!(
            dst,
            LocationAddr {
                word_id: 0,
                site_id: 5
            }
        );
    }

    #[test]
    fn position_cached() {
        let index = make_index();
        let loc = LocationAddr {
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
            word_id: 0,
            site_id: 0,
        };
        let lane = index.lane_for_source(MoveType::SiteBus, 0, Direction::Forward, src);
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
            word_id: 0,
            site_id: 5,
        };
        let lane = index.lane_for_source(MoveType::SiteBus, 0, Direction::Forward, src);
        assert!(lane.is_none());
    }

    #[test]
    fn outgoing_lanes_nonempty() {
        let index = make_index();
        // Site 0 in word 0 is a site bus source (forward) and a backward destination source
        let src = LocationAddr {
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
            word_id: 99,
            site_id: 99,
            bus_id: 99,
            direction: Direction::Forward,
        };
        assert!(index.endpoints(&lane).is_none());
    }
}
