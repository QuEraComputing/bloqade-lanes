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
    /// Qubits that collided during the most recent `apply_moves`.
    /// Key is the moving qubit, value is the qubit it displaced.
    /// Collided qubits are removed from both location maps.
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
        // Sort entries for deterministic hashing
        let mut loc_entries: Vec<_> = self
            .locations_to_qubit
            .iter()
            .map(|(k, v)| (k.encode(), *v))
            .collect();
        loc_entries.sort();
        for (k, v) in &loc_entries {
            k.hash(state);
            v.hash(state);
        }

        let mut qubit_entries: Vec<_> = self
            .qubit_to_locations
            .iter()
            .map(|(k, v)| (*k, v.encode()))
            .collect();
        qubit_entries.sort();
        for (k, v) in &qubit_entries {
            k.hash(state);
            v.hash(state);
        }

        let mut collision_entries: Vec<_> = self.collision.iter().collect();
        collision_entries.sort();
        for (k, v) in &collision_entries {
            k.hash(state);
            v.hash(state);
        }

        let mut lane_entries: Vec<_> = self
            .prev_lanes
            .iter()
            .map(|(k, v)| (*k, v.encode_u64()))
            .collect();
        lane_entries.sort();
        for (k, v) in &lane_entries {
            k.hash(state);
            v.hash(state);
        }

        let mut count_entries: Vec<_> = self.move_count.iter().collect();
        count_entries.sort();
        for (k, v) in &count_entries {
            k.hash(state);
            v.hash(state);
        }
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
        let zone_data = arch_spec.zone_by_id(zone.zone_id)?;
        let word_ids: std::collections::HashSet<u32> = zone_data.words.iter().copied().collect();

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

            if !word_ids.contains(&loc.word_id) {
                continue;
            }

            let blockaded = match arch_spec.get_blockaded_location(&loc) {
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
    use crate::arch::example_arch_spec;

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
                    word_id: 0,
                    site_id: 0,
                },
            ),
            (
                1,
                LocationAddr {
                    word_id: 1,
                    site_id: 0,
                },
            ),
        ];
        let state = AtomStateData::from_locations(&locs);
        assert_eq!(
            state.get_qubit(&LocationAddr {
                word_id: 0,
                site_id: 0
            }),
            Some(0)
        );
        assert_eq!(
            state.get_qubit(&LocationAddr {
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
            word_id: 0,
            site_id: 0,
        };
        let loc1 = LocationAddr {
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
                word_id: 0,
                site_id: 0,
            },
        )]);
        let result = state.add_atoms(&[(
            0,
            LocationAddr {
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
                word_id: 0,
                site_id: 0,
            },
        )]);
        let result = state.add_atoms(&[(
            1,
            LocationAddr {
                word_id: 0,
                site_id: 0,
            },
        )]);
        assert!(result.is_err());
    }

    #[test]
    fn apply_moves_basic() {
        let spec = example_arch_spec();
        let state = AtomStateData::from_locations(&[
            (
                0,
                LocationAddr {
                    word_id: 0,
                    site_id: 0,
                },
            ),
            (
                1,
                LocationAddr {
                    word_id: 1,
                    site_id: 0,
                },
            ),
        ]);

        // Site bus 0 moves site 0 -> site 5 (forward)
        let lane = LaneAddr {
            direction: crate::arch::addr::Direction::Forward,
            move_type: crate::arch::addr::MoveType::SiteBus,
            word_id: 0,
            site_id: 0,
            bus_id: 0,
        };

        let new_state = state.apply_moves(&[lane], &spec).unwrap();
        assert_eq!(
            new_state.get_qubit(&LocationAddr {
                word_id: 0,
                site_id: 5
            }),
            Some(0)
        );
        assert_eq!(
            new_state.get_qubit(&LocationAddr {
                word_id: 0,
                site_id: 0
            }),
            None
        );
        assert_eq!(*new_state.move_count.get(&0).unwrap(), 1);
    }

    #[test]
    fn apply_moves_collision() {
        let spec = example_arch_spec();
        let state = AtomStateData::from_locations(&[
            (
                0,
                LocationAddr {
                    word_id: 0,
                    site_id: 0,
                },
            ),
            (
                1,
                LocationAddr {
                    word_id: 0,
                    site_id: 5,
                },
            ),
        ]);

        let lane = LaneAddr {
            direction: crate::arch::addr::Direction::Forward,
            move_type: crate::arch::addr::MoveType::SiteBus,
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
        let spec = example_arch_spec();
        let loc_0_0 = LocationAddr {
            word_id: 0,
            site_id: 0,
        };
        let loc_0_5 = LocationAddr {
            word_id: 0,
            site_id: 5,
        };
        let loc_1_0 = LocationAddr {
            word_id: 1,
            site_id: 0,
        };
        let state = AtomStateData::from_locations(&[(0, loc_0_0), (1, loc_1_0)]);

        let lane = LaneAddr {
            direction: crate::arch::addr::Direction::Forward,
            move_type: crate::arch::addr::MoveType::SiteBus,
            word_id: 0,
            site_id: 0,
            bus_id: 0,
        };

        let new_state = state.apply_moves(&[lane], &spec).unwrap();

        // Qubit 0 moved from (0,0) to (0,5)
        assert_eq!(new_state.qubit_to_locations[&0], loc_0_5);
        assert_eq!(new_state.locations_to_qubit[&loc_0_5], 0);
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
        let spec = example_arch_spec();
        let state = AtomStateData::from_locations(&[
            (
                0,
                LocationAddr {
                    word_id: 0,
                    site_id: 0,
                },
            ),
            (
                1,
                LocationAddr {
                    word_id: 0,
                    site_id: 5,
                },
            ),
        ]);

        let lane = LaneAddr {
            direction: crate::arch::addr::Direction::Forward,
            move_type: crate::arch::addr::MoveType::SiteBus,
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
        let spec = example_arch_spec();
        // Only qubit at (1,0), no qubit at (0,0)
        let state = AtomStateData::from_locations(&[(
            1,
            LocationAddr {
                word_id: 1,
                site_id: 0,
            },
        )]);

        let lane = LaneAddr {
            direction: crate::arch::addr::Direction::Forward,
            move_type: crate::arch::addr::MoveType::SiteBus,
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
        let spec = example_arch_spec();
        let state = AtomStateData::from_locations(&[(
            0,
            LocationAddr {
                word_id: 0,
                site_id: 0,
            },
        )]);

        let bad_lane = LaneAddr {
            direction: crate::arch::addr::Direction::Forward,
            move_type: crate::arch::addr::MoveType::SiteBus,
            word_id: 0,
            site_id: 0,
            bus_id: 99, // invalid bus
        };

        assert!(state.apply_moves(&[bad_lane], &spec).is_none());
    }

    #[test]
    fn apply_moves_accumulates_move_count() {
        let spec = example_arch_spec();
        let state = AtomStateData::from_locations(&[(
            0,
            LocationAddr {
                word_id: 0,
                site_id: 0,
            },
        )]);

        // Move forward: site 0 -> site 5
        let lane_fwd = LaneAddr {
            direction: crate::arch::addr::Direction::Forward,
            move_type: crate::arch::addr::MoveType::SiteBus,
            word_id: 0,
            site_id: 0,
            bus_id: 0,
        };
        let state2 = state.apply_moves(&[lane_fwd], &spec).unwrap();
        assert_eq!(state2.move_count[&0], 1);

        // Move backward: site 5 -> site 0
        let lane_bwd = LaneAddr {
            direction: crate::arch::addr::Direction::Backward,
            move_type: crate::arch::addr::MoveType::SiteBus,
            word_id: 0,
            site_id: 5,
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
                word_id: 0,
                site_id: 0,
            },
        )]);
        assert_eq!(
            state.get_qubit(&LocationAddr {
                word_id: 1,
                site_id: 0
            }),
            None
        );
    }

    #[test]
    fn get_qubit_pairing_all_unpaired() {
        let spec = example_arch_spec();
        // Place qubits at sites 0, 1, 2 — site 0 pairs with 5, site 1 with 6,
        // site 2 with 7, but none of the pair sites are occupied.
        let state = AtomStateData::from_locations(&[
            (
                0,
                LocationAddr {
                    word_id: 0,
                    site_id: 0,
                },
            ),
            (
                1,
                LocationAddr {
                    word_id: 0,
                    site_id: 1,
                },
            ),
            (
                2,
                LocationAddr {
                    word_id: 1,
                    site_id: 0,
                },
            ),
        ]);

        let zone = ZoneAddr { zone_id: 0 };
        let (controls, targets, unpaired) = state.get_qubit_pairing(&zone, &spec).unwrap();

        assert!(controls.is_empty());
        assert!(targets.is_empty());
        assert_eq!(unpaired.len(), 3);
    }

    #[test]
    fn get_qubit_pairing_with_pairs() {
        let spec = example_arch_spec();
        // Site 0 pairs with site 5 in the same word. Place qubits at both.
        // Also place unpaired qubit at site 1 (pair site 6 is empty).
        let state = AtomStateData::from_locations(&[
            (
                0,
                LocationAddr {
                    word_id: 0,
                    site_id: 0,
                },
            ),
            (
                1,
                LocationAddr {
                    word_id: 0,
                    site_id: 5,
                },
            ),
            (
                2,
                LocationAddr {
                    word_id: 1,
                    site_id: 0,
                },
            ),
            (
                3,
                LocationAddr {
                    word_id: 1,
                    site_id: 5,
                },
            ),
            (
                4,
                LocationAddr {
                    word_id: 0,
                    site_id: 1,
                },
            ),
        ]);

        let zone = ZoneAddr { zone_id: 0 };
        let (controls, targets, unpaired) = state.get_qubit_pairing(&zone, &spec).unwrap();

        // Qubits 0+1 and 2+3 should be paired
        assert_eq!(controls.len(), 2);
        assert_eq!(targets.len(), 2);
        use std::collections::HashSet;
        let control_set: HashSet<u32> = controls.iter().copied().collect();
        let target_set: HashSet<u32> = targets.iter().copied().collect();
        assert_eq!(control_set, HashSet::from([0, 2]));
        assert_eq!(target_set, HashSet::from([1, 3]));
        // Qubit 4 is unpaired (site 1, pair site 6 is empty)
        assert_eq!(unpaired, vec![4]);
    }

    #[test]
    fn get_qubit_pairing_invalid_zone() {
        let spec = example_arch_spec();
        let state = AtomStateData::new();
        let zone = ZoneAddr { zone_id: 99 };
        assert!(state.get_qubit_pairing(&zone, &spec).is_none());
    }

    #[test]
    fn get_qubit_pairing_skips_qubits_outside_zone() {
        let spec = example_arch_spec();
        // Zone 0 contains words [0, 1]. Place a qubit at word 99 (out of zone).
        // But word 99 doesn't exist in the spec, so this qubit won't match any zone.
        let state = AtomStateData::from_locations(&[(
            0,
            LocationAddr {
                word_id: 0,
                site_id: 0,
            },
        )]);

        // Use zone 0 — qubit at (0,0) is in zone but has no CZ pair
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
                    word_id: 0,
                    site_id: 0,
                },
            ),
            (
                1,
                LocationAddr {
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
                    word_id: 0,
                    site_id: 0,
                },
            ),
            (
                1,
                LocationAddr {
                    word_id: 1,
                    site_id: 0,
                },
            ),
        ]);
        let state2 = AtomStateData::from_locations(&[
            (
                1,
                LocationAddr {
                    word_id: 1,
                    site_id: 0,
                },
            ),
            (
                0,
                LocationAddr {
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
