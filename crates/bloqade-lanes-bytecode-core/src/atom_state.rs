//! Atom state tracking for qubit-to-location mappings.
//!
//! [`AtomStateData`] is an immutable state object that tracks where qubits
//! are located in the architecture as atoms move through transport lanes.
//! It is the core data structure used by the IR analysis pipeline to simulate
//! atom movement, detect collisions, and identify CZ gate pairings.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use crate::arch::addr::{LaneAddr, LocationAddr, ZoneAddr};
use crate::arch::types::ArchSpec;

/// Tracks qubit-to-location mappings as atoms move through the architecture.
///
/// This is an immutable value type: all mutation methods (`add_atoms`,
/// `apply_moves`) return a new instance rather than modifying in place.
///
/// The two primary maps (`locations_to_qubit` and `qubit_to_locations`) are
/// kept in sync as a bidirectional index. When a move causes two atoms to
/// occupy the same site, both are removed from the location maps and recorded
/// in `collision`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtomStateData {
    /// Reverse index: given a physical location, which qubit (if any) is there?
    pub locations_to_qubit: HashMap<LocationAddr, u32>,
    /// Forward index: given a qubit id, where is it currently located?
    pub qubit_to_locations: HashMap<u32, LocationAddr>,
    /// Cumulative record of qubits that have collided since this state was
    /// created (via constructors or `add_atoms`). Updated by `apply_moves` —
    /// new collisions are added to existing entries. Key is the moving qubit,
    /// value is the qubit it displaced. Collided qubits are removed from
    /// both location maps.
    pub collision: HashMap<u32, u32>,
    /// The lane each qubit used in the most recent `apply_moves`.
    /// Only populated for qubits that moved in the last step.
    pub prev_lanes: HashMap<u32, LaneAddr>,
    /// Cumulative number of moves each qubit has undergone across
    /// all `apply_moves` calls in the state's history.
    pub move_count: HashMap<u32, u32>,
}

impl Hash for AtomStateData {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash each field with a discriminant tag and length prefix to prevent
        // cross-field collisions (e.g. entries from one map aliasing another).
        fn hash_sorted_map<H: Hasher, K: Ord + Hash, V: Hash>(
            state: &mut H,
            tag: u8,
            entries: &mut [(K, V)],
        ) {
            tag.hash(state);
            entries.len().hash(state);
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            for (k, v) in entries.iter() {
                k.hash(state);
                v.hash(state);
            }
        }

        let mut loc_entries: Vec<_> = self
            .locations_to_qubit
            .iter()
            .map(|(k, v)| (k.encode(), *v))
            .collect();
        hash_sorted_map(state, 0, &mut loc_entries);

        let mut qubit_entries: Vec<_> = self
            .qubit_to_locations
            .iter()
            .map(|(k, v)| (*k, v.encode()))
            .collect();
        hash_sorted_map(state, 1, &mut qubit_entries);

        let mut collision_entries: Vec<_> = self.collision.iter().map(|(k, v)| (*k, *v)).collect();
        hash_sorted_map(state, 2, &mut collision_entries);

        let mut lane_entries: Vec<_> = self
            .prev_lanes
            .iter()
            .map(|(k, v)| (*k, v.encode_u64()))
            .collect();
        hash_sorted_map(state, 3, &mut lane_entries);

        let mut count_entries: Vec<_> = self.move_count.iter().map(|(k, v)| (*k, *v)).collect();
        hash_sorted_map(state, 4, &mut count_entries);
    }
}

impl AtomStateData {
    /// Create an empty state with no qubits or locations.
    pub fn new() -> Self {
        Self {
            locations_to_qubit: HashMap::new(),
            qubit_to_locations: HashMap::new(),
            collision: HashMap::new(),
            prev_lanes: HashMap::new(),
            move_count: HashMap::new(),
        }
    }

    /// Create a state from a list of `(qubit_id, location)` pairs.
    ///
    /// Builds both the forward (qubit → location) and reverse (location → qubit)
    /// maps. All other fields (collision, prev_lanes, move_count) are empty.
    pub fn from_locations(locations: &[(u32, LocationAddr)]) -> Self {
        let mut locations_to_qubit = HashMap::new();
        let mut qubit_to_locations = HashMap::new();

        for &(qubit, loc) in locations {
            qubit_to_locations.insert(qubit, loc);
            locations_to_qubit.insert(loc, qubit);
        }

        Self {
            locations_to_qubit,
            qubit_to_locations,
            collision: HashMap::new(),
            prev_lanes: HashMap::new(),
            move_count: HashMap::new(),
        }
    }

    /// Add atoms at new locations, returning a new state.
    ///
    /// Each `(qubit_id, location)` pair is added to the bidirectional maps.
    /// Returns `Err` if any qubit id already exists in this state or any
    /// location is already occupied by another qubit.
    ///
    /// The returned state inherits no collision, prev_lanes, or move_count
    /// data — those fields are reset to empty.
    pub fn add_atoms(&self, locations: &[(u32, LocationAddr)]) -> Result<Self, &'static str> {
        let mut qubit_to_locations = self.qubit_to_locations.clone();
        let mut locations_to_qubit = self.locations_to_qubit.clone();

        for &(qubit, loc) in locations {
            if qubit_to_locations.contains_key(&qubit) {
                return Err("Attempted to add atom that already exists");
            }
            if locations_to_qubit.contains_key(&loc) {
                return Err("Attempted to add atom to occupied location");
            }
            qubit_to_locations.insert(qubit, loc);
            locations_to_qubit.insert(loc, qubit);
        }

        Ok(Self {
            locations_to_qubit,
            qubit_to_locations,
            collision: HashMap::new(),
            prev_lanes: HashMap::new(),
            move_count: HashMap::new(),
        })
    }

    /// Apply a sequence of lane moves and return the resulting state.
    ///
    /// For each lane, resolves its source and destination locations via
    /// [`ArchSpec::lane_endpoints`]. If a qubit exists at the source, it is
    /// moved to the destination. If the destination is already occupied,
    /// both qubits are removed from the location maps and recorded in
    /// `collision`. Lanes whose source has no qubit are skipped.
    ///
    /// Returns `None` if any lane cannot be resolved to endpoints (invalid
    /// bus, word, or site). The `prev_lanes` field is reset to contain only
    /// the lanes used in this call; `move_count` is accumulated.
    pub fn apply_moves(&self, lanes: &[LaneAddr], arch_spec: &ArchSpec) -> Option<Self> {
        let mut qubit_to_locations = self.qubit_to_locations.clone();
        let mut locations_to_qubit = self.locations_to_qubit.clone();
        let mut collisions = self.collision.clone();
        let mut move_count = self.move_count.clone();
        let mut prev_lanes: HashMap<u32, LaneAddr> = HashMap::new();

        for lane in lanes {
            let (src, dst) = arch_spec.lane_endpoints(lane)?;

            let qubit = match locations_to_qubit.remove(&src) {
                Some(q) => q,
                None => continue,
            };

            *move_count.entry(qubit).or_insert(0) += 1;
            prev_lanes.insert(qubit, *lane);

            if let Some(other_qubit) = locations_to_qubit.remove(&dst) {
                qubit_to_locations.remove(&qubit);
                qubit_to_locations.remove(&other_qubit);
                collisions.insert(qubit, other_qubit);
            } else {
                qubit_to_locations.insert(qubit, dst);
                locations_to_qubit.insert(dst, qubit);
            }
        }

        Some(Self {
            locations_to_qubit,
            qubit_to_locations,
            prev_lanes,
            collision: collisions,
            move_count,
        })
    }

    /// Look up which qubit (if any) occupies the given location.
    pub fn get_qubit(&self, location: &LocationAddr) -> Option<u32> {
        self.locations_to_qubit.get(location).copied()
    }

    /// Find CZ gate control/target qubit pairings within a zone.
    ///
    /// Iterates over all qubits whose current location is in the given zone
    /// and checks whether the CZ pair site (via [`ArchSpec::get_blockaded_location`])
    /// is also occupied. If both sites are occupied, the qubits form a
    /// control/target pair. If the pair site is empty or doesn't exist, the
    /// qubit is unpaired.
    ///
    /// Returns `(controls, targets, unpaired)` where `controls[i]` and
    /// `targets[i]` are paired for CZ. Results are sorted by qubit id for
    /// deterministic ordering. Returns `None` if the zone id is invalid.
    pub fn get_qubit_pairing(
        &self,
        zone: &ZoneAddr,
        arch_spec: &ArchSpec,
    ) -> Option<(Vec<u32>, Vec<u32>, Vec<u32>)> {
        // In the zone-centric model, all zones share the same words.
        // Filter qubits by checking if their zone_id matches the requested zone.
        let _zone_data = arch_spec.zone_by_id(zone.zone_id)?;
        let zone_id = zone.zone_id;

        let mut controls = Vec::new();
        let mut targets = Vec::new();
        let mut unpaired = Vec::new();
        let mut visited = std::collections::HashSet::new();

        // Sort by qubit id for deterministic iteration order
        let mut sorted_qubits: Vec<_> = self.qubit_to_locations.iter().collect();
        sorted_qubits.sort_by_key(|(qubit, _)| **qubit);

        for (qubit, loc) in &sorted_qubits {
            let qubit = **qubit;
            let loc = **loc;
            if visited.contains(&qubit) {
                continue;
            }
            visited.insert(qubit);

            if loc.zone_id != zone_id {
                continue;
            }

            let blockaded = match arch_spec.get_cz_partner(&loc) {
                Some(b) => b,
                None => {
                    unpaired.push(qubit);
                    continue;
                }
            };

            let target_qubit = match self.get_qubit(&blockaded) {
                Some(t) => t,
                None => {
                    unpaired.push(qubit);
                    continue;
                }
            };

            controls.push(qubit);
            targets.push(target_qubit);
            visited.insert(target_qubit);
        }

        Some((controls, targets, unpaired))
    }
}

impl Default for AtomStateData {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::addr::{SiteRef, WordRef, ZonedWordRef};
    use crate::arch::types::{Bus, Grid, Mode, Word, Zone};
    use crate::version::Version;

    /// Build the same two-zone spec used by the arch module tests.
    /// Zone 0 has site bus 0 (site 0 -> site 1) and word bus 0 (word 0 -> word 1).
    /// Entangling pair: zones [0, 1].
    fn make_test_spec() -> crate::arch::ArchSpec {
        let grid0 = Grid::from_positions(&[0.0, 5.0, 10.0], &[0.0, 3.0]);
        let grid1 = Grid::from_positions(&[0.0, 7.5, 15.0], &[0.0, 4.0]);

        crate::arch::ArchSpec {
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
    fn new_state_is_empty() {
        let state = AtomStateData::new();
        assert!(state.locations_to_qubit.is_empty());
        assert!(state.qubit_to_locations.is_empty());
        assert!(state.collision.is_empty());
        assert!(state.prev_lanes.is_empty());
        assert!(state.move_count.is_empty());
    }

    #[test]
    fn from_locations_creates_bidirectional_map() {
        let locs = vec![
            (
                0,
                LocationAddr {
                    zone_id: 0,
                    word_id: 0,
                    site_id: 0,
                },
            ),
            (
                1,
                LocationAddr {
                    zone_id: 0,
                    word_id: 1,
                    site_id: 0,
                },
            ),
        ];
        let state = AtomStateData::from_locations(&locs);
        assert_eq!(
            state.get_qubit(&LocationAddr {
                zone_id: 0,
                word_id: 0,
                site_id: 0
            }),
            Some(0)
        );
        assert_eq!(
            state.get_qubit(&LocationAddr {
                zone_id: 0,
                word_id: 1,
                site_id: 0
            }),
            Some(1)
        );
    }

    #[test]
    fn add_atoms_succeeds_and_fields_match() {
        let state = AtomStateData::new();
        let loc0 = LocationAddr {
            zone_id: 0,
            word_id: 0,
            site_id: 0,
        };
        let loc1 = LocationAddr {
            zone_id: 0,
            word_id: 1,
            site_id: 0,
        };
        let new_state = state.add_atoms(&[(0, loc0), (1, loc1)]).unwrap();

        assert_eq!(new_state.qubit_to_locations.len(), 2);
        assert_eq!(new_state.qubit_to_locations[&0], loc0);
        assert_eq!(new_state.qubit_to_locations[&1], loc1);
        assert_eq!(new_state.locations_to_qubit[&loc0], 0);
        assert_eq!(new_state.locations_to_qubit[&loc1], 1);
        assert!(new_state.collision.is_empty());
        assert!(new_state.prev_lanes.is_empty());
        assert!(new_state.move_count.is_empty());
    }

    #[test]
    fn add_atoms_duplicate_qubit_fails() {
        let state = AtomStateData::from_locations(&[(
            0,
            LocationAddr {
                zone_id: 0,
                word_id: 0,
                site_id: 0,
            },
        )]);
        let result = state.add_atoms(&[(
            0,
            LocationAddr {
                zone_id: 0,
                word_id: 1,
                site_id: 0,
            },
        )]);
        assert!(result.is_err());
    }

    #[test]
    fn add_atoms_occupied_location_fails() {
        let state = AtomStateData::from_locations(&[(
            0,
            LocationAddr {
                zone_id: 0,
                word_id: 0,
                site_id: 0,
            },
        )]);
        let result = state.add_atoms(&[(
            1,
            LocationAddr {
                zone_id: 0,
                word_id: 0,
                site_id: 0,
            },
        )]);
        assert!(result.is_err());
    }

    #[test]
    fn apply_moves_basic() {
        let spec = make_test_spec();
        // Zone 0 site bus 0: site 0 -> site 1
        let state = AtomStateData::from_locations(&[
            (
                0,
                LocationAddr {
                    zone_id: 0,
                    word_id: 0,
                    site_id: 0,
                },
            ),
            (
                1,
                LocationAddr {
                    zone_id: 0,
                    word_id: 1,
                    site_id: 0,
                },
            ),
        ]);

        // Site bus 0 moves site 0 -> site 1 (forward) in zone 0
        let lane = LaneAddr {
            direction: crate::arch::addr::Direction::Forward,
            move_type: crate::arch::addr::MoveType::SiteBus,
            zone_id: 0,
            word_id: 0,
            site_id: 0,
            bus_id: 0,
        };

        let new_state = state.apply_moves(&[lane], &spec).unwrap();
        assert_eq!(
            new_state.get_qubit(&LocationAddr {
                zone_id: 0,
                word_id: 0,
                site_id: 1
            }),
            Some(0)
        );
        assert_eq!(
            new_state.get_qubit(&LocationAddr {
                zone_id: 0,
                word_id: 0,
                site_id: 0
            }),
            None
        );
        assert_eq!(*new_state.move_count.get(&0).unwrap(), 1);
    }

    #[test]
    fn apply_moves_collision() {
        let spec = make_test_spec();
        // Place qubit 0 at site 0 and qubit 1 at site 1 (the destination of site bus 0)
        let state = AtomStateData::from_locations(&[
            (
                0,
                LocationAddr {
                    zone_id: 0,
                    word_id: 0,
                    site_id: 0,
                },
            ),
            (
                1,
                LocationAddr {
                    zone_id: 0,
                    word_id: 0,
                    site_id: 1,
                },
            ),
        ]);

        let lane = LaneAddr {
            direction: crate::arch::addr::Direction::Forward,
            move_type: crate::arch::addr::MoveType::SiteBus,
            zone_id: 0,
            word_id: 0,
            site_id: 0,
            bus_id: 0,
        };

        let new_state = state.apply_moves(&[lane], &spec).unwrap();
        assert!(new_state.collision.contains_key(&0));
        assert_eq!(*new_state.collision.get(&0).unwrap(), 1);
        assert!(new_state.qubit_to_locations.is_empty());
    }

    #[test]
    fn apply_moves_verifies_all_fields() {
        let spec = make_test_spec();
        let loc_0_0 = LocationAddr {
            zone_id: 0,
            word_id: 0,
            site_id: 0,
        };
        let loc_0_1 = LocationAddr {
            zone_id: 0,
            word_id: 0,
            site_id: 1,
        };
        let loc_1_0 = LocationAddr {
            zone_id: 0,
            word_id: 1,
            site_id: 0,
        };
        let state = AtomStateData::from_locations(&[(0, loc_0_0), (1, loc_1_0)]);

        let lane = LaneAddr {
            direction: crate::arch::addr::Direction::Forward,
            move_type: crate::arch::addr::MoveType::SiteBus,
            zone_id: 0,
            word_id: 0,
            site_id: 0,
            bus_id: 0,
        };

        let new_state = state.apply_moves(&[lane], &spec).unwrap();

        // Qubit 0 moved from (0,0,0) to (0,0,1)
        assert_eq!(new_state.qubit_to_locations[&0], loc_0_1);
        assert_eq!(new_state.locations_to_qubit[&loc_0_1], 0);
        // Qubit 1 didn't move
        assert_eq!(new_state.qubit_to_locations[&1], loc_1_0);
        assert_eq!(new_state.locations_to_qubit[&loc_1_0], 1);
        // Old location is empty
        assert!(!new_state.locations_to_qubit.contains_key(&loc_0_0));
        // prev_lanes only has the moved qubit
        assert_eq!(new_state.prev_lanes.len(), 1);
        assert_eq!(new_state.prev_lanes[&0], lane);
        // move_count incremented
        assert_eq!(new_state.move_count[&0], 1);
        // No collision
        assert!(new_state.collision.is_empty());
    }

    #[test]
    fn apply_moves_collision_verifies_all_fields() {
        let spec = make_test_spec();
        let state = AtomStateData::from_locations(&[
            (
                0,
                LocationAddr {
                    zone_id: 0,
                    word_id: 0,
                    site_id: 0,
                },
            ),
            (
                1,
                LocationAddr {
                    zone_id: 0,
                    word_id: 0,
                    site_id: 1,
                },
            ),
        ]);

        let lane = LaneAddr {
            direction: crate::arch::addr::Direction::Forward,
            move_type: crate::arch::addr::MoveType::SiteBus,
            zone_id: 0,
            word_id: 0,
            site_id: 0,
            bus_id: 0,
        };

        let new_state = state.apply_moves(&[lane], &spec).unwrap();

        // Both qubits removed from location maps
        assert!(new_state.qubit_to_locations.is_empty());
        assert!(new_state.locations_to_qubit.is_empty());
        // Collision recorded
        assert_eq!(new_state.collision[&0], 1);
        // prev_lanes has the moving qubit's lane
        assert_eq!(new_state.prev_lanes[&0], lane);
        // move_count incremented for moving qubit
        assert_eq!(new_state.move_count[&0], 1);
    }

    #[test]
    fn apply_moves_skips_empty_source() {
        let spec = make_test_spec();
        // Only qubit at (0,1,0), no qubit at (0,0,0)
        let state = AtomStateData::from_locations(&[(
            1,
            LocationAddr {
                zone_id: 0,
                word_id: 1,
                site_id: 0,
            },
        )]);

        let lane = LaneAddr {
            direction: crate::arch::addr::Direction::Forward,
            move_type: crate::arch::addr::MoveType::SiteBus,
            zone_id: 0,
            word_id: 0,
            site_id: 0,
            bus_id: 0,
        };

        let new_state = state.apply_moves(&[lane], &spec).unwrap();
        // Nothing changed — lane source had no qubit
        assert_eq!(new_state.qubit_to_locations.len(), 1);
        assert!(new_state.prev_lanes.is_empty());
        assert!(new_state.move_count.is_empty());
    }

    #[test]
    fn apply_moves_invalid_lane_returns_none() {
        let spec = make_test_spec();
        let state = AtomStateData::from_locations(&[(
            0,
            LocationAddr {
                zone_id: 0,
                word_id: 0,
                site_id: 0,
            },
        )]);

        let bad_lane = LaneAddr {
            direction: crate::arch::addr::Direction::Forward,
            move_type: crate::arch::addr::MoveType::SiteBus,
            zone_id: 0,
            word_id: 0,
            site_id: 0,
            bus_id: 99, // invalid bus
        };

        assert!(state.apply_moves(&[bad_lane], &spec).is_none());
    }

    #[test]
    fn apply_moves_accumulates_move_count() {
        let spec = make_test_spec();
        let state = AtomStateData::from_locations(&[(
            0,
            LocationAddr {
                zone_id: 0,
                word_id: 0,
                site_id: 0,
            },
        )]);

        // Move forward: site 0 -> site 1
        let lane_fwd = LaneAddr {
            direction: crate::arch::addr::Direction::Forward,
            move_type: crate::arch::addr::MoveType::SiteBus,
            zone_id: 0,
            word_id: 0,
            site_id: 0,
            bus_id: 0,
        };
        let state2 = state.apply_moves(&[lane_fwd], &spec).unwrap();
        assert_eq!(state2.move_count[&0], 1);

        // Move backward: site 1 -> site 0
        // site_id is always the forward source (0), direction flips endpoints
        let lane_bwd = LaneAddr {
            direction: crate::arch::addr::Direction::Backward,
            move_type: crate::arch::addr::MoveType::SiteBus,
            zone_id: 0,
            word_id: 0,
            site_id: 0,
            bus_id: 0,
        };
        let state3 = state2.apply_moves(&[lane_bwd], &spec).unwrap();
        assert_eq!(state3.move_count[&0], 2);
    }

    #[test]
    fn get_qubit_empty_location() {
        let state = AtomStateData::from_locations(&[(
            0,
            LocationAddr {
                zone_id: 0,
                word_id: 0,
                site_id: 0,
            },
        )]);
        assert_eq!(
            state.get_qubit(&LocationAddr {
                zone_id: 0,
                word_id: 1,
                site_id: 0
            }),
            None
        );
    }

    #[test]
    fn get_qubit_pairing_all_unpaired() {
        let spec = make_test_spec();
        // Entangling pair: zones [0, 1]. CZ partner of (zone=0, w, s) is (zone=1, w, s).
        // Place qubits only in zone 0 — no matching occupancy in zone 1.
        let state = AtomStateData::from_locations(&[
            (
                0,
                LocationAddr {
                    zone_id: 0,
                    word_id: 0,
                    site_id: 0,
                },
            ),
            (
                1,
                LocationAddr {
                    zone_id: 0,
                    word_id: 0,
                    site_id: 1,
                },
            ),
        ]);

        let zone = ZoneAddr { zone_id: 0 };
        let (controls, targets, unpaired) = state.get_qubit_pairing(&zone, &spec).unwrap();

        assert!(controls.is_empty());
        assert!(targets.is_empty());
        assert_eq!(unpaired.len(), 2);
    }

    #[test]
    fn get_qubit_pairing_with_pairs() {
        let spec = make_test_spec();
        // CZ pairing is between zones: (zone 0, word w, site s) <-> (zone 1, word w, site s).
        // Place qubit 0 at (zone 0, word 0, site 0) and qubit 1 at (zone 1, word 0, site 0) -> paired.
        // Also place qubit 2 at (zone 0, word 0, site 1) without partner in zone 1 -> unpaired.
        let state = AtomStateData::from_locations(&[
            (
                0,
                LocationAddr {
                    zone_id: 0,
                    word_id: 0,
                    site_id: 0,
                },
            ),
            (
                1,
                LocationAddr {
                    zone_id: 1,
                    word_id: 0,
                    site_id: 0,
                },
            ),
            (
                2,
                LocationAddr {
                    zone_id: 0,
                    word_id: 0,
                    site_id: 1,
                },
            ),
        ]);

        let zone = ZoneAddr { zone_id: 0 };
        let (controls, targets, unpaired) = state.get_qubit_pairing(&zone, &spec).unwrap();

        // Qubits 0 and 1 should be paired (both at (word 0, site 0) across zones 0 and 1)
        assert_eq!(controls.len(), 1);
        assert_eq!(targets.len(), 1);
        use std::collections::HashSet;
        let control_set: HashSet<u32> = controls.iter().copied().collect();
        let target_set: HashSet<u32> = targets.iter().copied().collect();
        assert_eq!(control_set, HashSet::from([0]));
        assert_eq!(target_set, HashSet::from([1]));
        // Qubit 2 is unpaired (zone 0 word 0 site 1, partner zone 1 word 0 site 1 is empty)
        assert_eq!(unpaired, vec![2]);
    }

    #[test]
    fn get_qubit_pairing_invalid_zone() {
        let spec = make_test_spec();
        let state = AtomStateData::new();
        let zone = ZoneAddr { zone_id: 99 };
        assert!(state.get_qubit_pairing(&zone, &spec).is_none());
    }

    #[test]
    fn get_qubit_pairing_skips_qubits_outside_zone() {
        let spec = make_test_spec();
        // Zone 0 is queried. Place a qubit only in zone 0 — it has a CZ partner
        // zone (zone 1), but no qubit occupies the partner site.
        let state = AtomStateData::from_locations(&[(
            0,
            LocationAddr {
                zone_id: 0,
                word_id: 0,
                site_id: 0,
            },
        )]);

        // Use zone 0 — qubit at (0,0,0) is in zone but has no paired qubit in zone 1
        let zone = ZoneAddr { zone_id: 0 };
        let (controls, targets, unpaired) = state.get_qubit_pairing(&zone, &spec).unwrap();

        assert!(controls.is_empty());
        assert!(targets.is_empty());
        assert_eq!(unpaired, vec![0]);
    }

    #[test]
    fn default_is_empty() {
        let state = AtomStateData::default();
        assert!(state.locations_to_qubit.is_empty());
        assert!(state.qubit_to_locations.is_empty());
    }

    #[test]
    fn clone_produces_equal_state() {
        let state = AtomStateData::from_locations(&[
            (
                0,
                LocationAddr {
                    zone_id: 0,
                    word_id: 0,
                    site_id: 0,
                },
            ),
            (
                1,
                LocationAddr {
                    zone_id: 0,
                    word_id: 1,
                    site_id: 0,
                },
            ),
        ]);
        let cloned = state.clone();
        assert_eq!(state, cloned);
    }

    #[test]
    fn hash_is_deterministic() {
        use std::collections::hash_map::DefaultHasher;

        let state1 = AtomStateData::from_locations(&[
            (
                0,
                LocationAddr {
                    zone_id: 0,
                    word_id: 0,
                    site_id: 0,
                },
            ),
            (
                1,
                LocationAddr {
                    zone_id: 0,
                    word_id: 1,
                    site_id: 0,
                },
            ),
        ]);
        let state2 = AtomStateData::from_locations(&[
            (
                1,
                LocationAddr {
                    zone_id: 0,
                    word_id: 1,
                    site_id: 0,
                },
            ),
            (
                0,
                LocationAddr {
                    zone_id: 0,
                    word_id: 0,
                    site_id: 0,
                },
            ),
        ]);

        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();
        state1.hash(&mut h1);
        state2.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }
}
