//! Admissible heuristics and precomputed distance tables for A* search.
//!
//! [`DistanceTable`] precomputes BFS lane-hop distances from target locations
//! and is shared between the heuristic and the move generator.
//!
//! [`MisplacedHeuristic`] is a simple count-based heuristic.
//! [`HopDistanceHeuristic`] uses the distance table for a tighter bound.

use std::collections::{HashMap, VecDeque};

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;

use crate::config::Config;
use crate::lane_index::LaneIndex;

// ── DistanceTable ───────────────────────────────────────────────────

/// Precomputed minimum lane-hop distances from every reachable location
/// to each target location.
///
/// Built once via BFS on the reversed lane graph (ignoring occupancy).
/// Shared between the heuristic and the heuristic move generator.
#[derive(Debug)]
pub struct DistanceTable {
    /// encoded_target → { encoded_location → min hops to target }
    distance_to: HashMap<u64, HashMap<u64, u32>>,
}

impl DistanceTable {
    /// Build a distance table by running BFS from each unique target
    /// location on the reversed lane graph.
    pub fn new(target_locations: &[u64], index: &LaneIndex) -> Self {
        // Deduplicate targets.
        let unique_targets: Vec<u64> = {
            let mut v = target_locations.to_vec();
            v.sort_unstable();
            v.dedup();
            v
        };

        // Build reverse adjacency: dst → [src, ...].
        let mut reverse_adj: HashMap<u64, Vec<u64>> = HashMap::new();
        for (mt, bus_id, zone_id, dir) in index.bus_groups() {
            for &lane in index.lanes_for(mt, bus_id, zone_id, dir) {
                if let Some((src, dst)) = index.endpoints(&lane) {
                    reverse_adj
                        .entry(dst.encode())
                        .or_default()
                        .push(src.encode());
                }
            }
        }

        // BFS from each target on reversed edges.
        let mut distance_to: HashMap<u64, HashMap<u64, u32>> = HashMap::new();
        for &target_enc in &unique_targets {
            let mut dist: HashMap<u64, u32> = HashMap::new();
            let mut queue: VecDeque<u64> = VecDeque::new();
            dist.insert(target_enc, 0);
            queue.push_back(target_enc);

            while let Some(current) = queue.pop_front() {
                let current_dist = dist[&current];
                if let Some(preds) = reverse_adj.get(&current) {
                    for &pred in preds {
                        if let std::collections::hash_map::Entry::Vacant(e) = dist.entry(pred) {
                            e.insert(current_dist + 1);
                            queue.push_back(pred);
                        }
                    }
                }
            }
            distance_to.insert(target_enc, dist);
        }

        Self { distance_to }
    }

    /// O(1) lookup: minimum lane hops from `from_encoded` to `to_target_encoded`.
    ///
    /// Returns `None` if the target is unknown or the source is unreachable.
    pub fn distance(&self, from_encoded: u64, to_target_encoded: u64) -> Option<u32> {
        self.distance_to
            .get(&to_target_encoded)?
            .get(&from_encoded)
            .copied()
    }
}

// ── MisplacedHeuristic ──────────────────────────────────────────────

/// Simple heuristic: returns 1.0 if any qubit is not at target, else 0.0.
///
/// Admissible but provides very weak guidance.
pub struct MisplacedHeuristic {
    targets: Vec<(u32, u64)>,
}

impl MisplacedHeuristic {
    pub fn new(target: impl IntoIterator<Item = (u32, LocationAddr)>) -> Self {
        Self {
            targets: target.into_iter().map(|(q, l)| (q, l.encode())).collect(),
        }
    }

    pub fn estimate(&self, config: &Config) -> f64 {
        for &(qid, target_enc) in &self.targets {
            if let Some(loc) = config.location_of(qid) {
                if loc.encode() != target_enc {
                    return 1.0;
                }
            } else {
                return f64::INFINITY;
            }
        }
        0.0
    }
}

// ── HopDistanceHeuristic ────────────────────────────────────────────

/// Heuristic based on precomputed lane-hop distances.
///
/// Returns the max hop distance over all qubits — admissible because
/// the worst-case qubit needs at least that many steps.
pub struct HopDistanceHeuristic<'a> {
    targets: Vec<(u32, u64)>,
    table: &'a DistanceTable,
}

impl<'a> HopDistanceHeuristic<'a> {
    /// Create from target placement and a precomputed distance table.
    pub fn new(
        target: impl IntoIterator<Item = (u32, LocationAddr)>,
        table: &'a DistanceTable,
    ) -> Self {
        Self {
            targets: target.into_iter().map(|(q, l)| (q, l.encode())).collect(),
            table,
        }
    }

    /// Estimate cost-to-go: max hop distance over all qubits.
    ///
    /// Admissible — the worst-case qubit needs at least this many steps.
    /// Use for A* where admissibility matters.
    pub fn estimate_max(&self, config: &Config) -> f64 {
        let mut max_dist: u32 = 0;
        for &(qid, target_enc) in &self.targets {
            let Some(loc) = config.location_of(qid) else {
                return f64::INFINITY;
            };
            let loc_enc = loc.encode();
            if loc_enc == target_enc {
                continue;
            }
            let Some(d) = self.table.distance(loc_enc, target_enc) else {
                return f64::INFINITY;
            };
            max_dist = max_dist.max(d);
        }
        max_dist as f64
    }

    /// Estimate cost-to-go: sum of hop distances over all qubits.
    ///
    /// Not admissible (overestimates because of bus parallelism), but gives
    /// much better ordering for IDS/DFS — distinguishes "1 qubit far, rest done"
    /// from "many qubits all far".
    pub fn estimate_sum(&self, config: &Config) -> f64 {
        let mut total: u32 = 0;
        for &(qid, target_enc) in &self.targets {
            let Some(loc) = config.location_of(qid) else {
                return f64::INFINITY;
            };
            let loc_enc = loc.encode();
            if loc_enc == target_enc {
                continue;
            }
            let Some(d) = self.table.distance(loc_enc, target_enc) else {
                return f64::INFINITY;
            };
            total += d;
        }
        total as f64
    }

    /// Alias for [`estimate_max`] — backward compatibility.
    pub fn estimate(&self, config: &Config) -> f64 {
        self.estimate_max(config)
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
        let target_locs: Vec<u64> = targets.iter().map(|&(_, l)| l.encode()).collect();
        DistanceTable::new(&target_locs, index)
    }

    // ── DistanceTable ──

    #[test]
    fn distance_to_self_is_zero() {
        let index = make_index();
        let table = make_table(&[(0, loc(0, 5))], &index);
        assert_eq!(
            table.distance(loc(0, 5).encode(), loc(0, 5).encode()),
            Some(0)
        );
    }

    #[test]
    fn distance_one_hop() {
        let index = make_index();
        let table = make_table(&[(0, loc(0, 5))], &index);
        // Site 0 → site 5 = 1 hop via site bus forward.
        assert_eq!(
            table.distance(loc(0, 0).encode(), loc(0, 5).encode()),
            Some(1)
        );
    }

    #[test]
    fn distance_unknown_target() {
        let index = make_index();
        let table = make_table(&[(0, loc(0, 5))], &index);
        assert_eq!(
            table.distance(loc(0, 0).encode(), loc(99, 99).encode()),
            None
        );
    }

    // ── MisplacedHeuristic ──

    #[test]
    fn misplaced_all_at_target() {
        let h = MisplacedHeuristic::new([(0, loc(0, 0))]);
        let config = Config::new([(0, loc(0, 0))]).unwrap();
        assert_eq!(h.estimate(&config), 0.0);
    }

    #[test]
    fn misplaced_one_off() {
        let h = MisplacedHeuristic::new([(0, loc(0, 5))]);
        let config = Config::new([(0, loc(0, 0))]).unwrap();
        assert_eq!(h.estimate(&config), 1.0);
    }

    #[test]
    fn misplaced_missing_qubit() {
        let h = MisplacedHeuristic::new([(99, loc(0, 0))]);
        let config = Config::new([(0, loc(0, 0))]).unwrap();
        assert_eq!(h.estimate(&config), f64::INFINITY);
    }

    // ── HopDistanceHeuristic ──

    #[test]
    fn hop_at_target_returns_zero() {
        let index = make_index();
        let table = make_table(&[(0, loc(0, 5))], &index);
        let h = HopDistanceHeuristic::new([(0, loc(0, 5))], &table);
        let config = Config::new([(0, loc(0, 5))]).unwrap();
        assert_eq!(h.estimate(&config), 0.0);
    }

    #[test]
    fn hop_one_step_away() {
        let index = make_index();
        let table = make_table(&[(0, loc(0, 5))], &index);
        let h = HopDistanceHeuristic::new([(0, loc(0, 5))], &table);
        let config = Config::new([(0, loc(0, 0))]).unwrap();
        assert_eq!(h.estimate(&config), 1.0);
    }

    #[test]
    fn hop_cross_word() {
        let index = make_index();
        let table = make_table(&[(0, loc(1, 5))], &index);
        let h = HopDistanceHeuristic::new([(0, loc(1, 5))], &table);
        let config = Config::new([(0, loc(0, 5))]).unwrap();
        assert_eq!(h.estimate(&config), 1.0);
    }

    #[test]
    fn hop_two_steps() {
        let index = make_index();
        let table = make_table(&[(0, loc(1, 5))], &index);
        let h = HopDistanceHeuristic::new([(0, loc(1, 5))], &table);
        let config = Config::new([(0, loc(0, 0))]).unwrap();
        assert_eq!(h.estimate(&config), 2.0);
    }

    #[test]
    fn hop_max_over_qubits() {
        let index = make_index();
        let targets = [(0, loc(0, 5)), (1, loc(1, 5))];
        let table = make_table(&targets, &index);
        let h = HopDistanceHeuristic::new(targets, &table);
        let config = Config::new([(0, loc(0, 0)), (1, loc(0, 0))]).unwrap();
        assert_eq!(h.estimate(&config), 2.0);
    }

    #[test]
    fn hop_unreachable_returns_infinity() {
        let index = make_index();
        let table = make_table(&[(0, loc(99, 99))], &index);
        let h = HopDistanceHeuristic::new([(0, loc(99, 99))], &table);
        let config = Config::new([(0, loc(0, 0))]).unwrap();
        assert_eq!(h.estimate(&config), f64::INFINITY);
    }

    #[test]
    fn hop_three_steps_via_site_word_site() {
        let index = make_index();
        let table = make_table(&[(0, loc(1, 0))], &index);
        let h = HopDistanceHeuristic::new([(0, loc(1, 0))], &table);
        let config = Config::new([(0, loc(0, 0))]).unwrap();
        assert_eq!(h.estimate(&config), 3.0);
    }

    #[test]
    fn hop_admissible_vs_actual() {
        use crate::astar::astar;
        use crate::expander::ExhaustiveExpander;

        let index = make_index();
        let targets = vec![(0u32, loc(1, 5))];
        let table = make_table(&targets, &index);
        let h = HopDistanceHeuristic::new(targets, &table);
        let config = Config::new([(0, loc(0, 0))]).unwrap();

        let estimate = h.estimate(&config);

        let expander = ExhaustiveExpander::new(&index, std::iter::empty(), None, None);
        let target_enc = loc(1, 5).encode();
        let result = astar(
            config,
            |cfg| cfg.location_of(0).is_some_and(|l| l.encode() == target_enc),
            |cfg| h.estimate(cfg),
            &expander,
            Some(1000),
        );

        assert!(result.goal.is_some());
        let actual_cost = result.graph.g_score(result.goal.unwrap());
        assert!(
            estimate <= actual_cost,
            "heuristic {estimate} should not exceed actual cost {actual_cost}"
        );
    }
}
