//! Compact, canonical configuration of qubit positions.
//!
//! A [`Config`] maps qubit IDs to physical locations. It is stored as a
//! sorted `Vec<(u32, u64)>` where each entry is `(qubit_id, encoded_location)`.
//! Sorting by qubit ID makes the representation canonical (order-independent)
//! and enables deterministic hashing.

use std::fmt;
use std::hash::{Hash, Hasher};

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;

/// Error returned when a [`Config`] cannot be constructed.
#[derive(Debug, Clone)]
pub struct ConfigError {
    /// The qubit ID that appeared more than once.
    pub duplicate_qubit_id: u32,
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "duplicate qubit_id: {}", self.duplicate_qubit_id)
    }
}

impl std::error::Error for ConfigError {}

/// Compute a hash for a sorted entries slice.
///
/// Uses FNV-1a for speed (entries are pre-sorted, so no ordering concern).
fn hash_entries(entries: &[(u32, u64)]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325; // FNV offset basis
    for &(qid, loc) in entries {
        h ^= qid as u64;
        h = h.wrapping_mul(0x100000001b3); // FNV prime
        h ^= loc;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Compact, canonical configuration of qubit positions.
///
/// Internally stored as a `Vec<(qubit_id, encoded_location)>` sorted by
/// `qubit_id`. For 20 qubits this is 240 bytes — far smaller than a
/// `HashMap<u32, LocationAddr>`.
///
/// The hash is cached at construction time to avoid re-hashing on every
/// transposition table lookup.
///
/// This is an immutable value type. Use [`with_moves`](Config::with_moves)
/// to derive a new configuration with some qubits relocated.
#[derive(Debug, Clone, Eq)]
pub struct Config {
    /// Sorted by qubit_id. Invariant: no duplicate qubit_ids.
    entries: Vec<(u32, u64)>,
    /// Cached hash of `entries`, computed once at construction.
    cached_hash: u64,
}

impl Config {
    /// Create a configuration from `(qubit_id, location)` pairs.
    ///
    /// The pairs may be in any order; they will be sorted by qubit ID.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] if any qubit ID appears more than once.
    pub fn new(pairs: impl IntoIterator<Item = (u32, LocationAddr)>) -> Result<Self, ConfigError> {
        let mut entries: Vec<(u32, u64)> = pairs
            .into_iter()
            .map(|(qid, loc)| (qid, loc.encode()))
            .collect();
        entries.sort_unstable_by_key(|&(qid, _)| qid);

        // Check for duplicates (adjacent after sort).
        for window in entries.windows(2) {
            if window[0].0 == window[1].0 {
                return Err(ConfigError {
                    duplicate_qubit_id: window[0].0,
                });
            }
        }

        let cached_hash = hash_entries(&entries);
        Ok(Self {
            entries,
            cached_hash,
        })
    }

    /// Number of qubits in this configuration.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the configuration contains no qubits.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Look up the location of a qubit. O(log n) via binary search.
    pub fn location_of(&self, qubit_id: u32) -> Option<LocationAddr> {
        self.entries
            .binary_search_by_key(&qubit_id, |&(qid, _)| qid)
            .ok()
            .map(|i| LocationAddr::decode(self.entries[i].1))
    }

    /// Find which qubit occupies a location, if any. O(n) scan.
    pub fn qubit_at(&self, loc: LocationAddr) -> Option<u32> {
        let encoded = loc.encode();
        self.entries
            .iter()
            .find(|&&(_, eloc)| eloc == encoded)
            .map(|&(qid, _)| qid)
    }

    /// Check whether a location is occupied by any qubit. O(n) scan.
    pub fn is_occupied(&self, loc: LocationAddr) -> bool {
        let encoded = loc.encode();
        self.entries.iter().any(|&(_, eloc)| eloc == encoded)
    }

    /// Build a reverse lookup: encoded_location → qubit_id.
    ///
    /// Use this when you need many `qubit_at` lookups on the same config
    /// (e.g., in the expander inner loop) to avoid O(n) per lookup.
    pub fn location_to_qubit_map(&self) -> std::collections::HashMap<u64, u32> {
        self.entries
            .iter()
            .map(|&(qid, eloc)| (eloc, qid))
            .collect()
    }

    /// Iterate over `(qubit_id, LocationAddr)` pairs in qubit ID order.
    pub fn iter(&self) -> impl Iterator<Item = (u32, LocationAddr)> + '_ {
        self.entries
            .iter()
            .map(|&(qid, eloc)| (qid, LocationAddr::decode(eloc)))
    }

    /// Derive a new configuration with some qubits moved to new locations.
    ///
    /// Each `(qubit_id, new_location)` in `moves` relocates that qubit.
    /// Qubits not mentioned in `moves` keep their current location.
    /// Moves for qubit IDs not in this configuration are silently ignored.
    ///
    /// This does **not** validate moves (no collision or architecture checks).
    ///
    /// Complexity: O(n) clone + O(m log n) binary searches, where n is the
    /// number of qubits and m is the number of moves. No re-sort needed
    /// because qubit IDs (the sort key) do not change.
    pub fn with_moves(&self, moves: &[(u32, LocationAddr)]) -> Self {
        let mut entries = self.entries.clone();
        for &(qid, loc) in moves {
            if let Ok(i) = entries.binary_search_by_key(&qid, |&(q, _)| q) {
                entries[i].1 = loc.encode();
            }
        }
        let cached_hash = hash_entries(&entries);
        Self {
            entries,
            cached_hash,
        }
    }

    /// Borrow the raw sorted entries for direct inspection.
    pub fn as_entries(&self) -> &[(u32, u64)] {
        &self.entries
    }
}

impl PartialEq for Config {
    fn eq(&self, other: &Self) -> bool {
        self.entries == other.entries
    }
}

impl Hash for Config {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.cached_hash);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::loc;

    #[test]
    fn canonical_ordering() {
        let a = Config::new([(0, loc(0, 1)), (1, loc(0, 2)), (2, loc(1, 0))]).unwrap();
        let b = Config::new([(2, loc(1, 0)), (0, loc(0, 1)), (1, loc(0, 2))]).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn hash_determinism() {
        use std::hash::DefaultHasher;

        let a = Config::new([(0, loc(0, 1)), (1, loc(0, 2))]).unwrap();
        let b = Config::new([(1, loc(0, 2)), (0, loc(0, 1))]).unwrap();

        let hash_of = |c: &Config| {
            let mut h = DefaultHasher::new();
            c.hash(&mut h);
            h.finish()
        };
        assert_eq!(hash_of(&a), hash_of(&b));
    }

    #[test]
    fn location_of_found() {
        let c = Config::new([(0, loc(1, 2)), (5, loc(3, 4))]).unwrap();
        assert_eq!(c.location_of(0), Some(loc(1, 2)));
        assert_eq!(c.location_of(5), Some(loc(3, 4)));
    }

    #[test]
    fn location_of_not_found() {
        let c = Config::new([(0, loc(1, 2))]).unwrap();
        assert_eq!(c.location_of(99), None);
    }

    #[test]
    fn qubit_at_found() {
        let c = Config::new([(7, loc(1, 2))]).unwrap();
        assert_eq!(c.qubit_at(loc(1, 2)), Some(7));
    }

    #[test]
    fn qubit_at_not_found() {
        let c = Config::new([(7, loc(1, 2))]).unwrap();
        assert_eq!(c.qubit_at(loc(0, 0)), None);
    }

    #[test]
    fn is_occupied_positive() {
        let c = Config::new([(0, loc(1, 2))]).unwrap();
        assert!(c.is_occupied(loc(1, 2)));
    }

    #[test]
    fn is_occupied_negative() {
        let c = Config::new([(0, loc(1, 2))]).unwrap();
        assert!(!c.is_occupied(loc(0, 0)));
    }

    #[test]
    fn with_moves_updates_moved_qubits() {
        let c = Config::new([(0, loc(0, 0)), (1, loc(0, 1)), (2, loc(0, 2))]).unwrap();
        let moved = c.with_moves(&[(1, loc(1, 1))]);

        assert_eq!(moved.location_of(0), Some(loc(0, 0))); // unchanged
        assert_eq!(moved.location_of(1), Some(loc(1, 1))); // moved
        assert_eq!(moved.location_of(2), Some(loc(0, 2))); // unchanged
    }

    #[test]
    fn with_moves_preserves_sorted_invariant() {
        let c = Config::new([(0, loc(0, 0)), (5, loc(0, 1)), (10, loc(0, 2))]).unwrap();
        let moved = c.with_moves(&[(5, loc(2, 2)), (10, loc(3, 3))]);

        let entries = moved.as_entries();
        for window in entries.windows(2) {
            assert!(window[0].0 < window[1].0, "entries not sorted by qubit_id");
        }
    }

    #[test]
    fn with_moves_ignores_unknown_qubits() {
        let c = Config::new([(0, loc(0, 0))]).unwrap();
        let moved = c.with_moves(&[(99, loc(1, 1))]);
        assert_eq!(moved, c);
    }

    #[test]
    fn duplicate_qubit_id_returns_error() {
        let err = Config::new([(0, loc(0, 0)), (0, loc(0, 1))]).unwrap_err();
        assert_eq!(err.duplicate_qubit_id, 0);
    }

    #[test]
    fn len_and_is_empty() {
        let empty = Config::new(std::iter::empty()).unwrap();
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);

        let c = Config::new([(0, loc(0, 0)), (1, loc(0, 1))]).unwrap();
        assert!(!c.is_empty());
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn iter_yields_decoded_pairs() {
        let c = Config::new([(2, loc(1, 0)), (0, loc(0, 1))]).unwrap();
        let pairs: Vec<_> = c.iter().collect();
        // Should be in qubit_id order
        assert_eq!(pairs, vec![(0, loc(0, 1)), (2, loc(1, 0))]);
    }
}
