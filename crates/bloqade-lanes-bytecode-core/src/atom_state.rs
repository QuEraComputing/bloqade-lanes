//! Atom state tracking for qubit-to-location mappings.
//!
//! [`AtomStateData`] is an immutable state object that tracks where qubits
//! are located in the architecture as atoms move through transport lanes.
//! It is the core data structure used by the IR analysis pipeline to simulate
//! atom movement, detect collisions, and identify CZ gate pairings.

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

use thiserror::Error;

use crate::arch::addr::{LaneAddr, LocationAddr, ZoneAddr};
use crate::arch::query::LaneGroupError;
use crate::arch::types::ArchSpec;

/// A lane group that cannot execute against a given [`AtomStateData`].
///
/// Produced by [`AtomStateData::validate_moves`]. Static lane-group failures
/// (address validity, duplicates, consistency, membership, AOD geometry) are
/// delegated to [`ArchSpec::check_lanes`] and wrapped in [`Self::LaneGroup`];
/// the occupancy rules are this module's own.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum MoveValidationError {
    /// A lane could not be resolved to (src, dst) endpoints.
    #[error("lane {lane:?} cannot be resolved to endpoints")]
    UnresolvableLane {
        /// The unresolvable lane.
        lane: LaneAddr,
    },

    /// Static lane-group check failure from [`ArchSpec::check_lanes`].
    #[error("{0}")]
    LaneGroup(LaneGroupError),

    /// A lane's destination holds an atom that does not move in this group.
    ///
    /// The AOD trap site arrives at every lane's destination whether or not
    /// the lane carried an atom, so this is a fault for empty-source filler
    /// lanes just as for movers. An occupied destination is legal only when
    /// its occupant sits at the source of another lane in the same group
    /// (it vacates in the same simultaneous step).
    #[error(
        "lane {lane:?} targets {dst:?}, which is occupied by qubit {occupant} \
         that does not move in this group"
    )]
    DestinationOccupiedByStationaryAtom {
        /// The offending lane (mover or filler).
        lane: LaneAddr,
        /// The occupied destination.
        dst: LocationAddr,
        /// The stationary qubit at the destination.
        occupant: u32,
    },

    /// Two lanes in the group share a destination.
    ///
    /// Unreachable through a well-formed bus (destination-unique per
    /// [`ArchSpec::validate`]); kept as a defensive check for hand-built
    /// lane groups.
    #[error("lanes {first:?} and {second:?} share destination {dst:?}")]
    ContestedDestination {
        /// The shared destination.
        dst: LocationAddr,
        /// The lane that claimed the destination first.
        first: LaneAddr,
        /// The lane that collided with it.
        second: LaneAddr,
    },
}

/// A lane group proven executable against the [`AtomStateData`] it was
/// validated with.
///
/// Obtainable only from [`AtomStateData::validate_moves`], which makes
/// "apply without validating" unrepresentable: [`AtomStateData::apply_validated`]
/// takes this token and is total — no collision, no silent skip.
///
/// The token captures resolved mover assignments against the pre-move state,
/// so it must be applied to the same state it was validated against.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedMoves {
    /// Resolved movers: `(qubit, src, dst, lane)`, one entry per lane whose
    /// source held an atom at validation time.
    movers: Vec<(u32, LocationAddr, LocationAddr, LaneAddr)>,
}

impl ValidatedMoves {
    /// The resolved `(qubit, src, dst, lane)` mover assignments.
    pub fn movers(&self) -> &[(u32, LocationAddr, LocationAddr, LaneAddr)] {
        &self.movers
    }
}

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

    /// Resolve each lane against the pre-move state into `(qubit, src, dst,
    /// lane)` mover entries. Lanes whose source holds no atom contribute no
    /// entry; a source consumed by an earlier lane is not consumed again
    /// (first lane wins, matching first-match bus endpoint resolution).
    ///
    /// Returns `None` if any lane cannot be resolved to endpoints.
    fn resolve_movers(
        &self,
        lanes: &[LaneAddr],
        arch_spec: &ArchSpec,
    ) -> Option<Vec<(u32, LocationAddr, LocationAddr, LaneAddr)>> {
        let mut movers = Vec::with_capacity(lanes.len());
        let mut seen_srcs: HashSet<LocationAddr> = HashSet::new();
        for lane in lanes {
            let (src, dst) = arch_spec.lane_endpoints(lane)?;
            if !seen_srcs.insert(src) {
                continue;
            }
            if let Some(&qubit) = self.locations_to_qubit.get(&src) {
                movers.push((qubit, src, dst, *lane));
            }
        }
        Some(movers)
    }

    /// Apply a group of lane moves simultaneously and return the resulting
    /// state.
    ///
    /// All lanes execute as one AOD transport operation: every endpoint is
    /// resolved against the pre-move state, so for any
    /// [`check_lanes`](ArchSpec::check_lanes)-valid group the result is
    /// independent of lane order (distinct lanes sharing a source can only
    /// occur in invalid groups, and resolve first-in-slice-wins). A
    /// destination counts as free when its occupant moves in the same group —
    /// conveyor chains (`x→y, y→z`) are legal. Lanes whose source has no
    /// qubit are skipped.
    ///
    /// A qubit that lands on an atom which does *not* move in this group
    /// collides: both qubits are removed from the location maps and recorded
    /// in `collision`. This method never fails on collisions — use
    /// [`Self::validate_moves`] + [`Self::apply_validated`] to reject such
    /// groups up front instead.
    ///
    /// Returns `None` if any lane cannot be resolved to endpoints (invalid
    /// bus, word, or site). The `prev_lanes` field is reset to contain only
    /// the lanes used in this call; `move_count` is accumulated.
    pub fn apply_moves(&self, lanes: &[LaneAddr], arch_spec: &ArchSpec) -> Option<Self> {
        let mut movers = self.resolve_movers(lanes, arch_spec)?;
        // Deterministic landing order regardless of lane slice order; only
        // observable through which qubit a `collision` entry is keyed on
        // when two movers contest one destination (ill-formed bus).
        movers.sort_unstable_by_key(|&(qubit, ..)| qubit);

        let mut qubit_to_locations = self.qubit_to_locations.clone();
        let mut locations_to_qubit = self.locations_to_qubit.clone();
        let mut collisions = self.collision.clone();
        let mut move_count = self.move_count.clone();
        let mut prev_lanes: HashMap<u32, LaneAddr> = HashMap::new();

        // Phase 1: every mover vacates its source.
        for (qubit, src, _, _) in &movers {
            locations_to_qubit.remove(src);
            qubit_to_locations.remove(qubit);
        }

        // Phase 2: land, judging occupancy against the pre-move state.
        let mover_srcs: HashSet<LocationAddr> = movers.iter().map(|&(_, src, ..)| src).collect();
        for (qubit, _, dst, lane) in &movers {
            *move_count.entry(*qubit).or_insert(0) += 1;
            prev_lanes.insert(*qubit, *lane);

            // A pre-move occupant that is not itself a mover stays put:
            // both it and the arriving qubit are destroyed.
            if let Some(&stationary) = self.locations_to_qubit.get(dst)
                && !mover_srcs.contains(dst)
            {
                locations_to_qubit.remove(dst);
                qubit_to_locations.remove(&stationary);
                collisions.insert(*qubit, stationary);
                continue;
            }

            // Another mover already landed here (two lanes sharing a
            // destination — ill-formed bus): destroy both.
            if let Some(&other) = locations_to_qubit.get(dst) {
                locations_to_qubit.remove(dst);
                qubit_to_locations.remove(&other);
                collisions.insert(*qubit, other);
                continue;
            }

            qubit_to_locations.insert(*qubit, *dst);
            locations_to_qubit.insert(*dst, *qubit);
        }

        Some(Self {
            locations_to_qubit,
            qubit_to_locations,
            prev_lanes,
            collision: collisions,
            move_count,
        })
    }

    /// Validate that a lane group can execute against this state.
    ///
    /// This is the canonical executability check for a `move` group. It runs
    /// the static lane-group checks ([`ArchSpec::check_lanes`]) and the
    /// occupancy rules, all resolved against the pre-move state:
    ///
    /// - every occupied destination must be vacated by a lane in the same
    ///   group (the occupant sits at another lane's source) — this applies
    ///   uniformly to mover lanes *and* empty-source filler lanes, because
    ///   the AOD trap site arrives at every destination either way;
    /// - no two lanes may share a destination;
    /// - empty-source lanes whose destination is also free are legal no-ops
    ///   (AOD rectangle filler).
    ///
    /// Assumes `arch_spec` itself is valid (see [`ArchSpec::validate`]);
    /// with well-formed (acyclic, endpoint-unique) buses, a valid group can
    /// only be a set of independent transports or conveyor chains, never a
    /// rotation.
    ///
    /// All errors are collected in one pass. On success the returned
    /// [`ValidatedMoves`] token feeds [`Self::apply_validated`]; it is tied
    /// to this state and must not be applied to any other.
    pub fn validate_moves(
        &self,
        lanes: &[LaneAddr],
        arch_spec: &ArchSpec,
    ) -> Result<ValidatedMoves, Vec<MoveValidationError>> {
        let mut errors: Vec<MoveValidationError> = arch_spec
            .check_lanes(lanes)
            .into_iter()
            .map(MoveValidationError::LaneGroup)
            .collect();

        // `lane_endpoints` fails exactly when `check_lane` already reported
        // `InvalidLane`, so only surface `UnresolvableLane` when it would
        // otherwise go unreported (a true shouldn't-happen).
        let invalid_lane_reported = errors.iter().any(|e| {
            matches!(
                e,
                MoveValidationError::LaneGroup(LaneGroupError::InvalidLane { .. })
            )
        });

        // Duplicate lane addresses are reported by `check_lanes`; dedup here
        // so a repeated lane doesn't also self-report as a contested
        // destination or double-report an occupied one.
        let mut seen_lanes: HashSet<u64> = HashSet::new();
        let mut resolved: Vec<(LaneAddr, LocationAddr, LocationAddr)> =
            Vec::with_capacity(lanes.len());
        for lane in lanes {
            if !seen_lanes.insert(lane.encode_u64()) {
                continue;
            }
            match arch_spec.lane_endpoints(lane) {
                Some((src, dst)) => resolved.push((*lane, src, dst)),
                None if !invalid_lane_reported => {
                    errors.push(MoveValidationError::UnresolvableLane { lane: *lane })
                }
                None => {}
            }
        }

        let mover_srcs: HashSet<LocationAddr> = resolved
            .iter()
            .filter(|(_, src, _)| self.locations_to_qubit.contains_key(src))
            .map(|&(_, src, _)| src)
            .collect();

        let mut claimed_dsts: HashMap<LocationAddr, LaneAddr> = HashMap::new();
        for &(lane, _, dst) in &resolved {
            if let Some(&occupant) = self.locations_to_qubit.get(&dst)
                && !mover_srcs.contains(&dst)
            {
                errors.push(MoveValidationError::DestinationOccupiedByStationaryAtom {
                    lane,
                    dst,
                    occupant,
                });
            }
            if let Some(&first) = claimed_dsts.get(&dst) {
                errors.push(MoveValidationError::ContestedDestination {
                    dst,
                    first,
                    second: lane,
                });
            } else {
                claimed_dsts.insert(dst, lane);
            }
        }

        if !errors.is_empty() {
            return Err(errors);
        }

        let movers = self
            .resolve_movers(lanes, arch_spec)
            .expect("all lanes resolved above");
        Ok(ValidatedMoves { movers })
    }

    /// Apply a validated lane group and return the resulting state.
    ///
    /// Total on its input: [`Self::validate_moves`] has already ruled out
    /// every collision, so no atom is ever destroyed or silently skipped
    /// here. The `prev_lanes` field is reset to the movers of this call;
    /// `move_count` is accumulated; `collision` is carried over unchanged.
    ///
    /// The token must have been produced by `validate_moves` on this same
    /// state (debug-asserted).
    pub fn apply_validated(&self, moves: &ValidatedMoves) -> Self {
        let mut qubit_to_locations = self.qubit_to_locations.clone();
        let mut locations_to_qubit = self.locations_to_qubit.clone();
        let mut move_count = self.move_count.clone();
        let mut prev_lanes: HashMap<u32, LaneAddr> = HashMap::new();

        // Guard against a stale token (validated against a different state):
        // every mover's source must still hold the validated qubit, and every
        // destination must still be free or vacated by this same group —
        // otherwise the insert below would silently desync the bidirectional
        // maps.
        #[cfg(debug_assertions)]
        {
            let mover_srcs: HashSet<LocationAddr> =
                moves.movers.iter().map(|&(_, src, _, _)| src).collect();
            for (qubit, src, dst, _) in &moves.movers {
                debug_assert_eq!(
                    self.locations_to_qubit.get(src),
                    Some(qubit),
                    "ValidatedMoves applied to a state it was not validated against"
                );
                debug_assert!(
                    !self.locations_to_qubit.contains_key(dst) || mover_srcs.contains(dst),
                    "ValidatedMoves applied to a state where destination {dst:?} \
                     is occupied by an atom that does not move in this group"
                );
            }
        }

        for (qubit, src, _, _) in &moves.movers {
            locations_to_qubit.remove(src);
            qubit_to_locations.remove(qubit);
        }

        for (qubit, _, dst, lane) in &moves.movers {
            *move_count.entry(*qubit).or_insert(0) += 1;
            prev_lanes.insert(*qubit, *lane);
            qubit_to_locations.insert(*qubit, *dst);
            locations_to_qubit.insert(*dst, *qubit);
        }

        Self {
            locations_to_qubit,
            qubit_to_locations,
            prev_lanes,
            collision: self.collision.clone(),
            move_count,
        }
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
                    name: String::new(),
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
        // Zone 0 entangling_pairs: [[0, 1]] — word 0 paired with word 1.
        // Place both qubits in word 0 only — no qubit in word 1, so all unpaired.
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
        // Zone 0 entangling_pairs: [[0, 1]] — word 0 paired with word 1.
        // Place qubit 0 at (zone 0, word 0, site 0) and qubit 1 at (zone 0, word 1, site 0)
        // -> paired (same zone, partner words, same site).
        // Place qubit 2 at (zone 0, word 0, site 1) without partner at (zone 0, word 1, site 1)
        // -> unpaired.
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

        // Qubits 0 and 1 should be paired (word 0 and word 1 at site 0 in zone 0)
        assert_eq!(controls.len(), 1);
        assert_eq!(targets.len(), 1);
        use std::collections::HashSet;
        let control_set: HashSet<u32> = controls.iter().copied().collect();
        let target_set: HashSet<u32> = targets.iter().copied().collect();
        assert_eq!(control_set, HashSet::from([0]));
        assert_eq!(target_set, HashSet::from([1]));
        // Qubit 2 is unpaired (zone 0 word 0 site 1, partner word 1 site 1 is empty)
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
        // Zone 0 entangling_pairs: [[0, 1]] — word 0 paired with word 1.
        // Place a qubit only at word 0 — partner word 1 has no qubit.
        let state = AtomStateData::from_locations(&[(
            0,
            LocationAddr {
                zone_id: 0,
                word_id: 0,
                site_id: 0,
            },
        )]);

        // Use zone 0 — qubit at (0,0,0), partner at (0,1,0) is empty
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

    // --- Simultaneous semantics: chains, order-independence, validate/apply ---

    /// A `validate()`-clean spec with an overlapping (acyclic) word bus:
    /// bus 0 maps words 0→1, 1→2, 2→3 — a conveyor chain.
    fn make_chain_spec() -> crate::arch::ArchSpec {
        let grid0 = Grid::from_positions(&[0.0, 5.0, 10.0, 15.0], &[0.0, 3.0]);
        let grid1 = Grid::from_positions(&[30.0, 35.0, 40.0, 45.0], &[0.0, 4.0]);

        let spec = crate::arch::ArchSpec {
            version: Version::new(2, 0),
            words: (0..4u32)
                .map(|w| Word {
                    sites: vec![[w, 0], [w, 1]],
                })
                .collect(),
            zones: vec![
                Zone {
                    name: String::new(),
                    grid: grid0,
                    site_buses: vec![],
                    word_buses: vec![Bus {
                        src: vec![WordRef(0), WordRef(1), WordRef(2)],
                        dst: vec![WordRef(1), WordRef(2), WordRef(3)],
                    }],
                    words_with_site_buses: vec![],
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
            zone_buses: vec![],
            modes: vec![Mode {
                name: "full".to_string(),
                zones: vec![0, 1],
                bitstring_order: vec![],
            }],
            paths: None,
            feed_forward: false,
            atom_reloading: false,
            blockade_radius: None,
        };
        assert!(
            spec.validate().is_ok(),
            "overlapping acyclic buses must be legal: {:?}",
            spec.validate()
        );
        spec
    }

    /// Forward lane on chain-spec word bus 0, sourced at `word_id`.
    fn chain_lane(word_id: u32) -> LaneAddr {
        LaneAddr {
            direction: crate::arch::addr::Direction::Forward,
            move_type: crate::arch::addr::MoveType::WordBus,
            zone_id: 0,
            word_id,
            site_id: 0,
            bus_id: 0,
        }
    }

    fn word_loc(word_id: u32) -> LocationAddr {
        LocationAddr {
            zone_id: 0,
            word_id,
            site_id: 0,
        }
    }

    #[test]
    fn apply_moves_chain_succeeds_in_any_lane_order() {
        let spec = make_chain_spec();
        // Atoms at words 0 and 1; word 2 empty: conveyor shift 0→1→2.
        let state = AtomStateData::from_locations(&[(0, word_loc(0)), (1, word_loc(1))]);
        let lanes = [chain_lane(0), chain_lane(1)];

        for order in [[0usize, 1], [1, 0]] {
            let slice = [lanes[order[0]], lanes[order[1]]];
            let result = state.apply_moves(&slice, &spec).unwrap();
            assert!(
                result.collision.is_empty(),
                "chain must not collide (order {order:?})"
            );
            assert_eq!(result.qubit_to_locations[&0], word_loc(1));
            assert_eq!(result.qubit_to_locations[&1], word_loc(2));
            assert_eq!(result.locations_to_qubit.len(), 2);
            assert_eq!(result.move_count[&0], 1);
            assert_eq!(result.move_count[&1], 1);
        }
    }

    #[test]
    fn apply_moves_is_invariant_under_lane_permutation() {
        let spec = make_chain_spec();
        // Full chain: atoms at words 0, 1, 2 shift to 1, 2, 3.
        let state =
            AtomStateData::from_locations(&[(0, word_loc(0)), (1, word_loc(1)), (2, word_loc(2))]);
        let lanes = [chain_lane(0), chain_lane(1), chain_lane(2)];

        let reference = state.apply_moves(&lanes, &spec).unwrap();
        assert!(reference.collision.is_empty());
        assert_eq!(reference.qubit_to_locations[&2], word_loc(3));

        for perm in [
            [0usize, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ] {
            let slice = [lanes[perm[0]], lanes[perm[1]], lanes[perm[2]]];
            let permuted = state.apply_moves(&slice, &spec).unwrap();
            assert_eq!(
                permuted, reference,
                "lane order {perm:?} changed the result"
            );
        }
    }

    #[test]
    fn apply_moves_stationary_collision_is_order_independent() {
        let spec = make_chain_spec();
        // Atom at word 3 has no outgoing lane in the group: qubit 1 lands on
        // a stationary atom regardless of lane order.
        let state =
            AtomStateData::from_locations(&[(0, word_loc(1)), (1, word_loc(2)), (2, word_loc(3))]);
        let lanes = [chain_lane(1), chain_lane(2)];

        let reference = state.apply_moves(&lanes, &spec).unwrap();
        assert_eq!(reference.collision, HashMap::from([(1, 2)]));
        // Qubit 0 still completes its move; the collided pair is destroyed.
        assert_eq!(reference.qubit_to_locations[&0], word_loc(2));
        assert_eq!(reference.qubit_to_locations.len(), 1);

        let reversed = state.apply_moves(&[lanes[1], lanes[0]], &spec).unwrap();
        assert_eq!(reversed, reference);
    }

    #[test]
    fn validate_moves_accepts_chain_and_apply_validated_matches() {
        let spec = make_chain_spec();
        let state = AtomStateData::from_locations(&[(0, word_loc(0)), (1, word_loc(1))]);
        let lanes = [chain_lane(0), chain_lane(1)];

        let validated = state.validate_moves(&lanes, &spec).expect("chain is valid");
        let via_token = state.apply_validated(&validated);
        let via_legacy = state.apply_moves(&lanes, &spec).unwrap();
        assert_eq!(via_token, via_legacy);
        assert!(via_token.collision.is_empty());
    }

    #[test]
    fn validate_moves_rejects_mover_onto_stationary_atom() {
        let spec = make_chain_spec();
        // Word 3 is occupied but has no lane in the group.
        let state = AtomStateData::from_locations(&[(0, word_loc(2)), (1, word_loc(3))]);
        let lanes = [chain_lane(2)];

        let errors = state.validate_moves(&lanes, &spec).unwrap_err();
        assert!(errors.iter().any(|e| matches!(
            e,
            MoveValidationError::DestinationOccupiedByStationaryAtom { occupant: 1, .. }
        )));
    }

    #[test]
    fn validate_moves_rejects_filler_onto_stationary_atom() {
        let spec = make_chain_spec();
        // Word 1 is empty (filler lane) but its destination word 2 holds a
        // stationary atom: the trap site still arrives there.
        let state = AtomStateData::from_locations(&[(0, word_loc(2))]);
        let lanes = [chain_lane(1)];

        let errors = state.validate_moves(&lanes, &spec).unwrap_err();
        assert!(errors.iter().any(|e| matches!(
            e,
            MoveValidationError::DestinationOccupiedByStationaryAtom { occupant: 0, .. }
        )));
        // Legacy apply skips the filler silently — pinned so the contrast
        // between the two APIs stays intentional.
        let legacy = state.apply_moves(&lanes, &spec).unwrap();
        assert_eq!(legacy.qubit_to_locations[&0], word_loc(2));
        assert!(legacy.collision.is_empty());
    }

    #[test]
    fn validate_moves_duplicate_lane_reports_duplicate_only() {
        let spec = make_chain_spec();
        let state = AtomStateData::from_locations(&[(0, word_loc(0))]);
        let lanes = [chain_lane(0), chain_lane(0)];

        let errors = state.validate_moves(&lanes, &spec).unwrap_err();
        assert!(errors.iter().any(|e| matches!(
            e,
            MoveValidationError::LaneGroup(LaneGroupError::DuplicateAddress { .. })
        )));
        // The repeated lane must not also self-report as a contested
        // destination or an unresolvable lane.
        assert!(
            !errors
                .iter()
                .any(|e| matches!(e, MoveValidationError::ContestedDestination { .. }))
        );
        assert!(
            !errors
                .iter()
                .any(|e| matches!(e, MoveValidationError::UnresolvableLane { .. }))
        );
    }

    #[test]
    fn validate_moves_reports_contested_destination() {
        // An ill-formed bus with a duplicated destination: 0→1 and 2→1.
        // `validate_moves` never sees `ArchSpec::validate()`, so it must
        // catch the contested landing itself.
        let mut spec = make_chain_spec();
        spec.zones[0].word_buses[0] = Bus {
            src: vec![WordRef(0), WordRef(2)],
            dst: vec![WordRef(1), WordRef(1)],
        };
        let state = AtomStateData::from_locations(&[(0, word_loc(0)), (1, word_loc(2))]);
        let lanes = [chain_lane(0), chain_lane(2)];

        let errors = state.validate_moves(&lanes, &spec).unwrap_err();
        assert!(errors.iter().any(|e| matches!(
            e,
            MoveValidationError::ContestedDestination { first, second, .. }
                if first != second
        )));
    }

    #[test]
    fn validate_moves_invalid_lane_reports_single_error() {
        let spec = make_chain_spec();
        let state = AtomStateData::from_locations(&[(0, word_loc(0))]);
        // bus_id 7 does not exist: `check_lanes` reports InvalidLane, and
        // the redundant UnresolvableLane must be suppressed.
        let mut bad = chain_lane(0);
        bad.bus_id = 7;

        let errors = state.validate_moves(&[bad], &spec).unwrap_err();
        assert_eq!(errors.len(), 1, "expected exactly one error: {errors:?}");
        assert!(matches!(
            errors[0],
            MoveValidationError::LaneGroup(LaneGroupError::InvalidLane { .. })
        ));
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "is occupied by an atom that does not move")]
    fn apply_validated_stale_token_panics_in_debug() {
        let spec = make_chain_spec();
        let state = AtomStateData::from_locations(&[(0, word_loc(0))]);
        let validated = state
            .validate_moves(&[chain_lane(0)], &spec)
            .expect("valid against the original state");

        // The destination becomes occupied by a stationary atom after
        // validation; applying the stale token must trip the debug guard.
        let later = state.add_atoms(&[(5, word_loc(1))]).unwrap();
        let _ = later.apply_validated(&validated);
    }

    #[test]
    fn validate_moves_accepts_filler_onto_vacated_site() {
        let spec = make_chain_spec();
        // Word 1 empty, word 2 occupied by a mover: the filler lane 1→2
        // points at a site vacated in the same step.
        let state = AtomStateData::from_locations(&[(0, word_loc(2))]);
        let lanes = [chain_lane(1), chain_lane(2)];

        let validated = state
            .validate_moves(&lanes, &spec)
            .expect("filler is valid");
        let result = state.apply_validated(&validated);
        assert_eq!(result.qubit_to_locations[&0], word_loc(3));
        assert_eq!(result.locations_to_qubit.len(), 1);
    }
}
