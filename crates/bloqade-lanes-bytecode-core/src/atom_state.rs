//! Atom state tracking for qubit-to-location mappings.
//!
//! [`AtomStateData`] is an immutable state object that tracks where qubits
//! are located in the architecture as atoms move through transport lanes.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use crate::arch::addr::{LaneAddr, LocationAddr, ZoneAddr};
use crate::arch::types::ArchSpec;

/// Tracks qubit-to-location mappings as atoms move through the architecture.
///
/// This is an immutable value type: mutation methods return a new instance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtomStateData {
    /// Mapping from location to qubit id.
    pub locations_to_qubit: HashMap<LocationAddr, u32>,
    /// Mapping from qubit id to its current location.
    pub qubit_to_locations: HashMap<u32, LocationAddr>,
    /// Mapping from qubit id to another qubit id it collided with.
    pub collision: HashMap<u32, u32>,
    /// Mapping from qubit id to the lane it took to reach this state.
    pub prev_lanes: HashMap<u32, LaneAddr>,
    /// Mapping from qubit id to number of moves.
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
    /// Create a new empty state.
    pub fn new() -> Self {
        Self {
            locations_to_qubit: HashMap::new(),
            qubit_to_locations: HashMap::new(),
            collision: HashMap::new(),
            prev_lanes: HashMap::new(),
            move_count: HashMap::new(),
        }
    }

    /// Create a state from a mapping of qubit ids to locations.
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

    /// Add atoms at new locations. Returns `Err` if any qubit already exists
    /// or any location is already occupied.
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

    /// Apply lane moves and return a new state.
    ///
    /// Returns `None` if a lane cannot be resolved to endpoints.
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

    /// Look up the qubit at a given location.
    pub fn get_qubit(&self, location: &LocationAddr) -> Option<u32> {
        self.locations_to_qubit.get(location).copied()
    }

    /// Find CZ control/target pairs in a zone.
    ///
    /// Returns `(controls, targets, unpaired)` where each control qubit
    /// has a corresponding target qubit at the blockaded location.
    /// Results are sorted by qubit id for deterministic ordering.
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
    fn add_atoms_succeeds() {
        let state = AtomStateData::new();
        let new_state = state
            .add_atoms(&[
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
            ])
            .unwrap();
        assert_eq!(new_state.qubit_to_locations.len(), 2);
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
