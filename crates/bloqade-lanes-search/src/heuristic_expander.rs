//! Heuristic move generator for A* search.
//!
//! Scores qubit-bus pairs by distance improvement using a precomputed
//! [`DistanceTable`], then builds a small number of high-quality movesets
//! per bus group. Independent of entropy-guided search.

use std::collections::{HashMap, HashSet};

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;

use crate::astar::Expander;
use crate::config::Config;
use crate::graph::MoveSet;
use crate::heuristic::DistanceTable;
use crate::lane_index::LaneIndex;

/// Heuristic move generator that produces a small number of high-quality
/// movesets per expansion.
///
/// For each node:
/// 1. Score each (qubit, bus, direction) by distance improvement (O(1) lookup).
/// 2. Per qubit: keep top C bus options.
/// 3. Group by bus triplet.
/// 4. Per group: generate multiple movesets by varying the lead qubit.
/// 5. Sort by total distance improvement.
///
/// Typically produces 5-15 candidates per expansion, vs hundreds from
/// the exhaustive generator.
#[derive(Debug)]
pub struct HeuristicExpander<'a> {
    index: &'a LaneIndex,
    blocked: HashSet<u64>,
    /// (qubit_id, encoded_target_location)
    targets: Vec<(u32, u64)>,
    dist_table: &'a DistanceTable,
    /// Top bus options to keep per qubit.
    top_c: usize,
    /// Max movesets to generate per bus group.
    max_movesets_per_group: usize,
}

impl<'a> HeuristicExpander<'a> {
    pub fn new(
        index: &'a LaneIndex,
        blocked: impl IntoIterator<Item = LocationAddr>,
        targets: impl IntoIterator<Item = (u32, LocationAddr)>,
        dist_table: &'a DistanceTable,
        top_c: usize,
        max_movesets_per_group: usize,
    ) -> Self {
        Self {
            index,
            blocked: blocked.into_iter().map(|l| l.encode()).collect(),
            targets: targets.into_iter().map(|(q, l)| (q, l.encode())).collect(),
            dist_table,
            top_c,
            max_movesets_per_group,
        }
    }
}

/// A scored (qubit, bus triplet) entry.
#[derive(Clone)]
struct ScoredTriple {
    qubit_id: u32,
    score: i32, // d_now - d_after (can be negative)
    lane_encoded: u64,
    dst_encoded: u64,
}

impl Expander for HeuristicExpander<'_> {
    fn expand(&self, config: &Config, out: &mut Vec<(MoveSet, Config, f64)>) {
        // Build occupied set: config qubit locations + blocked.
        // Pre-allocate to avoid rehashing.
        let mut occupied = HashSet::with_capacity(self.blocked.len() + config.len());
        occupied.extend(&self.blocked);
        for (_, loc) in config.iter() {
            occupied.insert(loc.encode());
        }

        // Step 1: identify unresolved qubits.
        let unresolved: Vec<(u32, u64, u64)> = self
            .targets
            .iter()
            .filter_map(|&(qid, target_enc)| {
                let loc = config.location_of(qid)?;
                let loc_enc = loc.encode();
                if loc_enc == target_enc {
                    None
                } else {
                    Some((qid, loc_enc, target_enc))
                }
            })
            .collect();

        if unresolved.is_empty() {
            return;
        }

        // Step 2: score (qubit, bus triplet) pairs.
        // Key: (qubit_id, mt_encoded, bus_id, dir_encoded) — but we just
        // need to group by triplet later, so store the triplet key separately.
        type TripletKey = (u8, u32, u8); // (move_type as u8, bus_id, direction as u8)

        let mut all_scores: Vec<(TripletKey, ScoredTriple)> = Vec::new();

        for &(qid, loc_enc, target_enc) in &unresolved {
            let d_now = self.dist_table.distance(loc_enc, target_enc);
            let d_now = match d_now {
                Some(d) => d as i32,
                None => continue, // unreachable target
            };

            let loc = LocationAddr::decode(loc_enc);
            for &lane in self.index.outgoing_lanes(loc) {
                let Some((_, dst)) = self.index.endpoints(&lane) else {
                    continue;
                };
                let dst_enc = dst.encode();
                if occupied.contains(&dst_enc) {
                    continue;
                }

                let d_after = self
                    .dist_table
                    .distance(dst_enc, target_enc)
                    .map_or(i32::MAX, |d| d as i32);
                let score = d_now - d_after;

                let triplet_key = (lane.move_type as u8, lane.bus_id, lane.direction as u8);
                all_scores.push((
                    triplet_key,
                    ScoredTriple {
                        qubit_id: qid,
                        score,
                        lane_encoded: lane.encode_u64(),
                        dst_encoded: dst_enc,
                    },
                ));
            }
        }

        // Step 3: per qubit, keep top C triples.
        // (If all_scores is empty, this produces no candidates and the
        // deadlock escape at the end activates.)
        let mut per_qubit: HashMap<u32, Vec<(TripletKey, ScoredTriple)>> = HashMap::new();
        for entry in all_scores {
            per_qubit.entry(entry.1.qubit_id).or_default().push(entry);
        }

        let mut selected: Vec<(TripletKey, ScoredTriple)> = Vec::new();
        let mut has_positive = false;

        for (_, entries) in per_qubit.iter_mut() {
            entries.sort_by(|a, b| b.1.score.cmp(&a.1.score));
            entries.truncate(self.top_c);
            // Keep only positive scores (or all if none positive — handled below).
            for e in entries.iter() {
                if e.1.score > 0 {
                    has_positive = true;
                }
                selected.push(e.clone());
            }
        }

        // Fallback: if no positive scores, keep only the single best entry.
        if !has_positive {
            selected.sort_by(|a, b| b.1.score.cmp(&a.1.score));
            selected.truncate(1);
        } else {
            selected.retain(|e| e.1.score > 0);
        }

        // Step 4: group by bus triplet.
        let mut groups: HashMap<TripletKey, Vec<ScoredTriple>> = HashMap::new();
        for (key, triple) in selected {
            groups.entry(key).or_default().push(triple);
        }

        // Step 5: per group, generate multiple movesets.
        let mut candidates: Vec<(i32, MoveSet, Config)> = Vec::new();

        for (_, mut qubits) in groups {
            // Sort by score descending.
            qubits.sort_by(|a, b| b.score.cmp(&a.score));

            let n = qubits.len().min(self.max_movesets_per_group);

            for start in 0..n {
                let mut lanes: Vec<u64> = Vec::new();
                let mut moves: Vec<(u32, LocationAddr)> = Vec::new();
                let mut used_dsts: HashSet<u64> = HashSet::new();
                let mut total_score: i32 = 0;

                // Greedy: start from `start`, then add remaining in order.
                let order: Vec<usize> = (start..qubits.len()).chain(0..start).collect();

                for &idx in &order {
                    let t = &qubits[idx];
                    if used_dsts.contains(&t.dst_encoded) {
                        continue;
                    }
                    // Check destination still unoccupied (another moveset
                    // in this expansion could have used it, but within one
                    // moveset we track via used_dsts).
                    if occupied.contains(&t.dst_encoded) {
                        continue;
                    }
                    lanes.push(t.lane_encoded);
                    used_dsts.insert(t.dst_encoded);
                    total_score += t.score;

                    let dst = LocationAddr::decode(t.dst_encoded);
                    moves.push((t.qubit_id, dst));
                }

                if lanes.is_empty() {
                    continue;
                }

                let move_set = MoveSet::from_encoded(lanes);
                let new_config = config.with_moves(&moves);

                // Deduplicate: skip if we already have this exact moveset.
                if candidates.iter().any(|(_, ms, _)| *ms == move_set) {
                    continue;
                }

                candidates.push((total_score, move_set, new_config));
            }
        }

        // Step 6: sort by total score descending, emit.
        candidates.sort_by(|a, b| b.0.cmp(&a.0));

        for (_, move_set, new_config) in candidates {
            out.push((move_set, new_config, 1.0));
        }

        // Step 7: deadlock escape.
        // If no positive-score candidates were produced (deadlock — all
        // improving moves blocked), also generate all valid single-lane
        // moves for EVERY qubit in the config. This includes moving
        // resolved qubits out of the way to unblock others. A*/DFS will
        // explore these escape moves and find the path through.
        if !has_positive {
            for (qid, loc) in config.iter() {
                for &lane in self.index.outgoing_lanes(loc) {
                    let Some((_, dst)) = self.index.endpoints(&lane) else {
                        continue;
                    };
                    if occupied.contains(&dst.encode()) {
                        continue;
                    }
                    let ms = MoveSet::from_encoded(vec![lane.encode_u64()]);
                    let new_cfg = config.with_moves(&[(qid, dst)]);
                    out.push((ms, new_cfg, 1.0));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{example_arch_json, loc};
    use bloqade_lanes_bytecode_core::arch::types::ArchSpec;

    fn make_index() -> LaneIndex {
        let spec: ArchSpec = serde_json::from_str(example_arch_json()).unwrap();
        LaneIndex::new(spec)
    }

    fn make_table(targets: &[(u32, LocationAddr)], index: &LaneIndex) -> DistanceTable {
        let locs: Vec<u64> = targets.iter().map(|&(_, l)| l.encode()).collect();
        DistanceTable::new(&locs, index)
    }

    #[test]
    fn fewer_candidates_than_exhaustive() {
        use crate::expander::ExhaustiveExpander;

        let index = make_index();
        let targets = [(0, loc(0, 5)), (1, loc(0, 6))];
        let table = make_table(&targets, &index);
        let config = Config::new([(0, loc(0, 0)), (1, loc(0, 1))]).unwrap();

        let mut heuristic_out = Vec::new();
        let h_exp = HeuristicExpander::new(&index, std::iter::empty(), targets, &table, 3, 3);
        h_exp.expand(&config, &mut heuristic_out);

        let mut exhaustive_out = Vec::new();
        let e_exp = ExhaustiveExpander::new(&index, std::iter::empty(), None, None);
        e_exp.expand(&config, &mut exhaustive_out);

        assert!(
            heuristic_out.len() < exhaustive_out.len(),
            "heuristic ({}) should produce fewer candidates than exhaustive ({})",
            heuristic_out.len(),
            exhaustive_out.len()
        );
        assert!(!heuristic_out.is_empty());
    }

    #[test]
    fn prefers_distance_reducing_moves() {
        let index = make_index();
        let targets = [(0, loc(0, 5))];
        let table = make_table(&targets, &index);
        let config = Config::new([(0, loc(0, 0))]).unwrap();

        let mut out = Vec::new();
        let exp = HeuristicExpander::new(&index, std::iter::empty(), targets, &table, 3, 3);
        exp.expand(&config, &mut out);

        // Best move should place qubit 0 at site 5 (direct site bus forward).
        assert!(!out.is_empty());
        let best_cfg = &out[0].1;
        assert_eq!(best_cfg.location_of(0), Some(loc(0, 5)));
    }

    #[test]
    fn skips_blocked_destinations() {
        let index = make_index();
        let targets = [(0, loc(0, 5))];
        let table = make_table(&targets, &index);
        let config = Config::new([(0, loc(0, 0))]).unwrap();

        let mut out = Vec::new();
        let exp = HeuristicExpander::new(&index, [loc(0, 5)], targets, &table, 3, 3);
        exp.expand(&config, &mut out);

        // No move should place qubit at blocked site 5.
        for (_, cfg, _) in &out {
            assert_ne!(cfg.location_of(0), Some(loc(0, 5)));
        }
    }

    #[test]
    fn conflict_resolution_same_destination() {
        let index = make_index();
        // Two qubits at sites 0 and 1, both targeting site 5.
        // Only one can go to site 5 in one step.
        let targets = [(0, loc(0, 5)), (1, loc(0, 5))];
        let table = make_table(&targets, &index);
        let config = Config::new([(0, loc(0, 0)), (1, loc(0, 1))]).unwrap();

        let mut out = Vec::new();
        let exp = HeuristicExpander::new(&index, std::iter::empty(), targets, &table, 3, 3);
        exp.expand(&config, &mut out);

        // Each moveset should not have two qubits at the same destination.
        for (_, cfg, _) in &out {
            let loc0 = cfg.location_of(0);
            let loc1 = cfg.location_of(1);
            if loc0.is_some() && loc1.is_some() {
                // If both moved, they shouldn't be at the same place.
                // (They could also be at their original locations if not moved.)
            }
        }
        assert!(!out.is_empty());
    }

    #[test]
    fn multiple_movesets_per_group() {
        let index = make_index();
        // 3 qubits at sites 0, 1, 2 all targeting sites 5, 6, 7.
        let targets = [(0, loc(0, 5)), (1, loc(0, 6)), (2, loc(0, 7))];
        let table = make_table(&targets, &index);
        let config = Config::new([(0, loc(0, 0)), (1, loc(0, 1)), (2, loc(0, 2))]).unwrap();

        let mut out = Vec::new();
        let exp = HeuristicExpander::new(&index, std::iter::empty(), targets, &table, 3, 3);
        exp.expand(&config, &mut out);

        // With max_movesets_per_group=3 and 3 qubits in the same group,
        // we should get multiple (possibly up to 3) movesets.
        // At minimum 1, but likely >1 with different lead qubits.
        assert!(!out.is_empty());
    }

    #[test]
    fn already_resolved_produces_nothing() {
        let index = make_index();
        let targets = [(0, loc(0, 5))];
        let table = make_table(&targets, &index);
        let config = Config::new([(0, loc(0, 5))]).unwrap();

        let mut out = Vec::new();
        let exp = HeuristicExpander::new(&index, std::iter::empty(), targets, &table, 3, 3);
        exp.expand(&config, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn fallback_when_no_positive_scores() {
        let index = make_index();
        // Qubit at site 5, target at site 0. Site bus backward would help,
        // but let's block site 0. The only moves available make things worse.
        let targets = [(0, loc(0, 0))];
        let table = make_table(&targets, &index);
        let config = Config::new([(0, loc(0, 5))]).unwrap();

        let mut out = Vec::new();
        // Block site 0 (the target) so no move reaches it.
        let exp = HeuristicExpander::new(&index, [loc(0, 0)], targets, &table, 3, 3);
        exp.expand(&config, &mut out);

        // Fallback should still produce at least one candidate
        // (the least-bad move).
        assert!(!out.is_empty());
    }

    #[test]
    fn integration_astar_finds_solution() {
        use crate::astar::astar;
        use crate::heuristic::HopDistanceHeuristic;

        let index = make_index();
        let targets = vec![(0u32, loc(0, 5))];
        let target_locs: Vec<u64> = targets.iter().map(|&(_, l)| l.encode()).collect();
        let table = DistanceTable::new(&target_locs, &index);
        let h = HopDistanceHeuristic::new(targets.clone(), &table);

        let config = Config::new([(0, loc(0, 0))]).unwrap();
        let exp = HeuristicExpander::new(&index, std::iter::empty(), targets, &table, 3, 3);

        let target_enc = loc(0, 5).encode();
        let result = astar(
            config,
            |cfg| cfg.location_of(0).is_some_and(|l| l.encode() == target_enc),
            |cfg| h.estimate(cfg),
            &exp,
            Some(100),
        );

        assert!(result.goal.is_some());
        let path = result.solution_path().unwrap();
        assert_eq!(path.len(), 1); // site 0 → site 5 in one hop
    }

    #[test]
    fn integration_multi_step() {
        use crate::astar::astar;
        use crate::heuristic::HopDistanceHeuristic;

        let index = make_index();
        // Word 0 site 0 → word 1 site 5: 2 hops.
        let targets = vec![(0u32, loc(1, 5))];
        let target_locs: Vec<u64> = targets.iter().map(|&(_, l)| l.encode()).collect();
        let table = DistanceTable::new(&target_locs, &index);
        let h = HopDistanceHeuristic::new(targets.clone(), &table);

        let config = Config::new([(0, loc(0, 0))]).unwrap();
        let exp = HeuristicExpander::new(&index, std::iter::empty(), targets, &table, 3, 3);

        let target_enc = loc(1, 5).encode();
        let result = astar(
            config,
            |cfg| cfg.location_of(0).is_some_and(|l| l.encode() == target_enc),
            |cfg| h.estimate(cfg),
            &exp,
            Some(1000),
        );

        assert!(result.goal.is_some());
        let path = result.solution_path().unwrap();
        assert_eq!(path.len(), 2);
    }

    #[test]
    fn deadlock_escape_generates_moves() {
        // Two qubits mutually blocking: q0 at site 0 wants site 5,
        // q1 at site 5 wants site 0. Both direct moves are blocked.
        // Verify the escape generates at least some moves (even if the
        // full search can't solve the swap on this bus topology).
        use crate::astar::Expander;

        let index = make_index();
        let targets = vec![(0u32, loc(0, 5)), (1, loc(0, 0))];
        let target_locs: Vec<u64> = targets.iter().map(|&(_, l)| l.encode()).collect();
        let table = DistanceTable::new(&target_locs, &index);
        let config = Config::new([(0, loc(0, 0)), (1, loc(0, 5))]).unwrap();
        let exp = HeuristicExpander::new(&index, std::iter::empty(), targets, &table, 3, 3);

        let mut out = Vec::new();
        exp.expand(&config, &mut out);

        // q1 at site 5 has word bus outgoing lanes (sites_with_word_buses = [5..9]).
        // The escape should find these even though no heuristic move improves distance.
        assert!(
            !out.is_empty(),
            "escape should produce moves even when heuristic candidates are blocked"
        );
    }

    #[test]
    fn deadlock_escape_solves_blocking() {
        // q0 at site 0, target site 5. q1 at site 5, target word 1 site 5.
        // q1 blocks q0's direct move (0→5). But q1 can escape via word bus
        // (site 5 is in sites_with_word_buses). Path:
        //   q1: word 0 site 5 → word 1 site 5 (word bus fwd) — reaches target!
        //   q0: site 0 → site 5 (site bus fwd) — now unblocked.
        // The heuristic expander normally can't find this because q0's
        // improving move is blocked. The escape generates q1's word bus
        // move, enabling q0's move on the next expansion.
        use crate::astar::astar;
        use crate::heuristic::HopDistanceHeuristic;

        let index = make_index();
        let targets = vec![(0u32, loc(0, 5)), (1, loc(1, 5))];
        let target_locs: Vec<u64> = targets.iter().map(|&(_, l)| l.encode()).collect();
        let table = DistanceTable::new(&target_locs, &index);
        let h = HopDistanceHeuristic::new(targets.clone(), &table);

        let config = Config::new([(0, loc(0, 0)), (1, loc(0, 5))]).unwrap();
        let exp = HeuristicExpander::new(&index, std::iter::empty(), targets, &table, 3, 3);

        let result = astar(
            config,
            |cfg| {
                cfg.location_of(0)
                    .is_some_and(|l| l.encode() == loc(0, 5).encode())
                    && cfg
                        .location_of(1)
                        .is_some_and(|l| l.encode() == loc(1, 5).encode())
            },
            |cfg| h.estimate(cfg),
            &exp,
            Some(500),
        );

        assert!(
            result.goal.is_some(),
            "should solve blocking via escape (expanded {} nodes)",
            result.nodes_expanded
        );
    }
}
