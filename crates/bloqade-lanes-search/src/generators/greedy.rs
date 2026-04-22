//! Greedy shortest-path move generator.
//!
//! For each unresolved qubit, computes the shortest path to its target
//! while routing around all other atoms (but not itself) and takes the
//! first lane. Those first-step lanes are grouped by
//! `(move_type, bus_id, direction)` (zone-independent) and partitioned into
//! AOD-compatible rectangular grids via [`BusGridContext`].

use std::collections::HashMap;
use std::collections::HashSet;

use bloqade_lanes_bytecode_core::arch::addr::{Direction, LaneAddr, LocationAddr, MoveType};

use crate::aod_grid::BusGridContext;
use crate::config::Config;
use crate::context::{MoveCandidate, SearchContext, SearchState};
use crate::entropy::find_path_occupied;
use crate::graph::{MoveSet, NodeId};
use crate::traits::MoveGenerator;

/// Greedy shortest-path move generator.
///
/// Routes each unresolved qubit along the shortest path to its target,
/// takes the first lane, groups by bus parameters, and builds
/// AOD-compatible rectangular grids.
pub struct GreedyGenerator;

impl MoveGenerator for GreedyGenerator {
    fn generate(
        &self,
        config: &Config,
        _node_id: NodeId,
        ctx: &SearchContext,
        _state: &mut SearchState,
        out: &mut Vec<MoveCandidate>,
    ) {
        let index = ctx.index;

        // 1. Build occupied set: blocked ∪ {all current qubit locations}.
        let mut occupied: HashSet<u64> = ctx.blocked.clone();
        for (_qid, loc) in config.iter() {
            occupied.insert(loc.encode());
        }

        // 2. For each unresolved qubit, find shortest path and record first lane.
        let mut first_lanes: Vec<(u32, LaneAddr)> = Vec::new();

        for &(qid, target_enc) in ctx.targets {
            let Some(current_loc) = config.location_of(qid) else {
                continue;
            };
            let current_enc = current_loc.encode();
            if current_enc == target_enc {
                continue; // already at target
            }

            let target_loc = LocationAddr::decode(target_enc);

            // Route around others but not self.
            occupied.remove(&current_enc);
            let path = find_path_occupied(current_loc, target_loc, &occupied, index);
            occupied.insert(current_enc);

            if let Some(lanes) = path
                && let Some(&first) = lanes.first()
            {
                first_lanes.push((qid, first));
            }
        }

        if first_lanes.is_empty() {
            return;
        }

        // 3. Group first lanes by (move_type, bus_id, direction) — zone-independent.
        // Use Vec<(src_enc, lane_enc)> to preserve insertion order for AOD clustering.
        let mut groups: HashMap<(MoveType, u32, Direction), Vec<(u64, u64)>> = HashMap::new();
        let mut seen_per_group: HashMap<(MoveType, u32, Direction), HashSet<u64>> = HashMap::new();

        // Build reverse lookup for resolving qubit → source location.
        let loc_to_qubit = config.location_to_qubit_map();

        for &(_qid, lane) in &first_lanes {
            let key = (lane.move_type, lane.bus_id, lane.direction);
            let Some((src, _dst)) = index.endpoints(&lane) else {
                continue;
            };
            let src_enc = src.encode();
            let lane_enc = lane.encode_u64();
            let seen = seen_per_group.entry(key).or_default();
            if seen.insert(src_enc) {
                groups.entry(key).or_default().push((src_enc, lane_enc));
            }
        }

        // 4. For each group, build AOD grids and emit candidates.
        for ((mt, bus_id, dir), entries) in &groups {
            let grid_ctx = BusGridContext::new(index, *mt, *bus_id, None, *dir, &occupied);
            let grids = grid_ctx.build_aod_grids(entries);

            for grid in grids {
                if grid.is_empty() {
                    continue;
                }

                // Resolve each lane in the grid to (qubit, dst).
                let mut moves: Vec<(u32, LocationAddr)> = Vec::new();
                for &lane_enc in &grid {
                    let lane = LaneAddr::decode_u64(lane_enc);
                    let Some((src, dst)) = index.endpoints(&lane) else {
                        continue;
                    };
                    let Some(&qid) = loc_to_qubit.get(&src.encode()) else {
                        continue;
                    };
                    moves.push((qid, dst));
                }

                if moves.is_empty() {
                    continue;
                }

                let new_config = config.with_moves(&moves);
                out.push(MoveCandidate {
                    move_set: MoveSet::from_encoded(grid),
                    new_config,
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::SearchState;
    use crate::heuristic::DistanceTable;
    use crate::lane_index::LaneIndex;
    use crate::test_utils::{example_arch_json, loc};
    use bloqade_lanes_bytecode_core::arch::types::ArchSpec;

    #[test]
    fn greedy_generator_produces_candidates() {
        let spec: ArchSpec = serde_json::from_str(example_arch_json()).unwrap();
        let index = LaneIndex::new(spec);

        // q0 at site (0,0), target at site (0,5) — a site-bus destination.
        let target_loc = loc(0, 5);
        let target_enc: Vec<(u32, u64)> = vec![(0, target_loc.encode())];
        let locs: Vec<u64> = target_enc.iter().map(|&(_, l)| l).collect();
        let table = DistanceTable::new(&locs, &index);
        let blocked = HashSet::new();
        let ctx = SearchContext {
            index: &index,
            dist_table: &table,
            blocked: &blocked,
            targets: &target_enc,
        };
        let mut state = SearchState::default();

        let config = Config::new([(0, loc(0, 0))]).unwrap();
        let mut out = Vec::new();
        GreedyGenerator.generate(&config, NodeId(0), &ctx, &mut state, &mut out);

        assert!(!out.is_empty(), "should produce at least one candidate");

        // Verify q0 moved closer to target (or arrived).
        let start_loc = loc(0, 0);
        let t_enc = target_loc.encode();
        let start_dist = table.distance(start_loc.encode(), t_enc).unwrap();
        for candidate in &out {
            let new_loc = candidate.new_config.location_of(0).unwrap();
            let new_dist = table.distance(new_loc.encode(), t_enc).unwrap_or(0);
            assert!(
                new_dist <= start_dist,
                "q0 should move closer to (or stay at same distance from) target; \
                 start_dist={start_dist}, new_dist={new_dist}"
            );
        }
    }

    #[test]
    fn greedy_generator_skips_resolved_qubits() {
        let spec: ArchSpec = serde_json::from_str(example_arch_json()).unwrap();
        let index = LaneIndex::new(spec);

        // q0 already at target.
        let target_loc = loc(0, 0);
        let target_enc: Vec<(u32, u64)> = vec![(0, target_loc.encode())];
        let locs: Vec<u64> = target_enc.iter().map(|&(_, l)| l).collect();
        let table = DistanceTable::new(&locs, &index);
        let blocked = HashSet::new();
        let ctx = SearchContext {
            index: &index,
            dist_table: &table,
            blocked: &blocked,
            targets: &target_enc,
        };
        let mut state = SearchState::default();

        let config = Config::new([(0, loc(0, 0))]).unwrap();
        let mut out = Vec::new();
        GreedyGenerator.generate(&config, NodeId(0), &ctx, &mut state, &mut out);

        assert!(out.is_empty(), "no candidates when qubit already at target");
    }
}
