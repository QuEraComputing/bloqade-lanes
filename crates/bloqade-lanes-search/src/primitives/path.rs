//! Lane-graph path finding primitives.
//!
//! Hosts the occupancy-respecting BFS used by:
//!
//! - [`crate::drivers::entropy::entropy_search`] inside the sequential
//!   fallback pass (per-qubit shortest-path routing),
//! - [`crate::generators::greedy`] when seeding initial candidate moves,
//! - [`crate::placement::nohome`] when assigning return locations after
//!   the main route completes.
//!
//! Previously lived inside `drivers/entropy.rs`, which forced
//! `generators/greedy.rs` and `placement/nohome.rs` to reach into the
//! entropy engine for a primitive that is conceptually independent of
//! the entropy driver (one of the §4 findings in the
//! `bloqade-lanes-search` review).

use std::collections::{HashMap, HashSet, VecDeque};

use bloqade_lanes_bytecode_core::arch::addr::{LaneAddr, LocationAddr};

use crate::primitives::lane_index::LaneIndex;

/// Find the shortest lane path from `from` to `to`, avoiding any
/// destination encoded in `occupied`.
///
/// Returns `Some(vec![])` when `from == to`, `Some(path)` when a path
/// exists, and `None` otherwise. The path is a sequence of
/// [`LaneAddr`]s in execution order (first lane first).
///
/// BFS over [`LaneIndex::outgoing_lanes`] with a parent-pointer map for
/// O(V) memory rather than O(V × path-length).
pub fn find_path_occupied(
    from: LocationAddr,
    to: LocationAddr,
    occupied: &HashSet<u64>,
    index: &LaneIndex,
) -> Option<Vec<LaneAddr>> {
    let from_enc = from.encode();
    let to_enc = to.encode();
    if from_enc == to_enc {
        return Some(Vec::new());
    }

    let mut visited: HashSet<u64> = HashSet::new();
    visited.insert(from_enc);

    let mut parent: HashMap<u64, (u64, LaneAddr)> = HashMap::new();
    let mut queue: VecDeque<u64> = VecDeque::new();
    queue.push_back(from_enc);

    while let Some(current_enc) = queue.pop_front() {
        let current = LocationAddr::decode(current_enc);
        for &lane in index.outgoing_lanes(current) {
            let Some((_, dst)) = index.endpoints(&lane) else {
                continue;
            };
            let dst_enc = dst.encode();
            if occupied.contains(&dst_enc) || visited.contains(&dst_enc) {
                continue;
            }
            visited.insert(dst_enc);
            parent.insert(dst_enc, (current_enc, lane));
            if dst_enc == to_enc {
                // Reconstruct path by walking parent pointers backwards.
                let mut path = Vec::new();
                let mut cur = to_enc;
                while let Some(&(prev, lane)) = parent.get(&cur) {
                    path.push(lane);
                    cur = prev;
                }
                path.reverse();
                return Some(path);
            }
            queue.push_back(dst_enc);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{full_arch_json, loc};

    fn make_index() -> LaneIndex {
        LaneIndex::new(serde_json::from_str(full_arch_json()).unwrap())
    }

    #[test]
    fn find_path_occupied_basic() {
        let index = make_index();
        let path = find_path_occupied(loc(0, 0), loc(0, 5), &HashSet::new(), &index);
        assert!(path.is_some());
        assert!(!path.unwrap().is_empty());
    }

    #[test]
    fn find_path_occupied_respects_blocked() {
        let index = make_index();
        let blocked: HashSet<u64> = [loc(0, 5).encode()].into_iter().collect();
        let path = find_path_occupied(loc(0, 0), loc(0, 5), &blocked, &index);
        assert!(path.is_none());
    }

    #[test]
    fn find_path_occupied_source_equals_target() {
        let index = make_index();
        let path = find_path_occupied(loc(0, 0), loc(0, 0), &HashSet::new(), &index);
        assert_eq!(path, Some(Vec::new()));
    }
}
