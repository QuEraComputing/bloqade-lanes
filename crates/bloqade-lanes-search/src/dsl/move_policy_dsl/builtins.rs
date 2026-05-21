//! Move-DSL builtins invoked via `actions.invoke_builtin(name, **args)`.
//!
//! v1 implements `sequential_fallback`: greedy per-qubit BFS routing
//! from an initial config to a target config.

use std::collections::{HashMap, HashSet, VecDeque};

use bloqade_lanes_bytecode_core::arch::addr::{LaneAddr, LocationAddr};

use crate::config::Config;
use crate::graph::MoveSet;
use crate::lane_index::LaneIndex;

/// Greedy per-qubit BFS routing.
///
/// For each qubit not at its target, find a path through the lane graph
/// avoiding occupied / blocked locations, and emit one `MoveSet` per hop.
/// If a qubit can't reach its target, return the partial path.
pub(super) fn sequential_fallback(
    initial: &Config,
    target: &Config,
    index: &LaneIndex,
    blocked: &HashSet<u64>,
) -> Vec<MoveSet> {
    let mut path: Vec<MoveSet> = Vec::new();
    let mut working = initial.clone();

    let qubit_ids: Vec<u32> = working.iter().map(|(q, _)| q).collect();
    for qid in qubit_ids {
        let Some(current) = working.location_of(qid) else {
            continue;
        };
        let Some(t) = target.location_of(qid) else {
            continue;
        };
        if current == t {
            continue;
        }

        let occupied: HashSet<u64> = working
            .iter()
            .filter(|(q, _)| *q != qid)
            .map(|(_, l)| l.encode())
            .chain(blocked.iter().copied())
            .collect();

        if let Some(hop_lanes) = bfs_path(current, t, &occupied, index) {
            for lane in hop_lanes {
                let Some((src, dst)) = index.endpoints(&lane) else {
                    break;
                };
                if working.location_of(qid) != Some(src) {
                    break;
                }
                working = working.with_moves(&[(qid, dst)]);
                path.push(MoveSet::new([lane]));
            }
        }
    }

    path
}

fn bfs_path(
    start: LocationAddr,
    goal: LocationAddr,
    occupied: &HashSet<u64>,
    index: &LaneIndex,
) -> Option<Vec<LaneAddr>> {
    if start == goal {
        return Some(Vec::new());
    }
    let start_enc = start.encode();
    let goal_enc = goal.encode();

    let mut visited: HashMap<u64, (u64, LaneAddr)> = HashMap::new();
    let mut queue: VecDeque<LocationAddr> = VecDeque::new();
    queue.push_back(start);

    while let Some(loc) = queue.pop_front() {
        for &lane in index.outgoing_lanes(loc) {
            let Some((_, dst)) = index.endpoints(&lane) else {
                continue;
            };
            let dst_enc = dst.encode();
            if dst_enc == start_enc {
                continue;
            }
            if dst_enc != goal_enc && occupied.contains(&dst_enc) {
                continue;
            }
            if visited.contains_key(&dst_enc) {
                continue;
            }
            visited.insert(dst_enc, (loc.encode(), lane));
            if dst_enc == goal_enc {
                let mut lanes_rev: Vec<LaneAddr> = Vec::new();
                let mut cursor = goal_enc;
                while cursor != start_enc {
                    let (parent_enc, lane) = *visited
                        .get(&cursor)
                        .expect("BFS reconstruction: cursor must be in visited");
                    lanes_rev.push(lane);
                    cursor = parent_enc;
                }
                lanes_rev.reverse();
                return Some(lanes_rev);
            }
            queue.push_back(dst);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::example_arch_json;
    use bloqade_lanes_bytecode_core::arch::types::ArchSpec;

    #[test]
    fn empty_config_returns_empty_path() {
        let initial = Config::new(std::iter::empty()).unwrap();
        let target = Config::new(std::iter::empty()).unwrap();
        let arch: ArchSpec = serde_json::from_str(example_arch_json()).unwrap();
        let index = LaneIndex::new(arch);
        let blocked = HashSet::new();
        let path = sequential_fallback(&initial, &target, &index, &blocked);
        assert!(path.is_empty());
    }

    #[test]
    fn already_at_target_returns_empty_path() {
        let loc = LocationAddr {
            zone_id: 0,
            word_id: 0,
            site_id: 0,
        };
        let initial = Config::new([(0u32, loc)]).unwrap();
        let target = initial.clone();
        let arch: ArchSpec = serde_json::from_str(example_arch_json()).unwrap();
        let index = LaneIndex::new(arch);
        let blocked = HashSet::new();
        let path = sequential_fallback(&initial, &target, &index, &blocked);
        assert!(path.is_empty());
    }
}
