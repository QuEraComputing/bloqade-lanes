//! Admissible heuristics for A* search over qubit configurations.
//!
//! Provides [`MisplacedHeuristic`] (count of misplaced qubits) and
//! [`HopDistanceHeuristic`] (max lane-hop distance to target, precomputed
//! via BFS on the lane graph).

use std::collections::{HashMap, VecDeque};

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;

use crate::config::Config;
use crate::lane_index::LaneIndex;

/// Simple heuristic: count of qubits not at their target location.
///
/// Admissible for unit-cost search (each step moves at least one qubit
/// closer, so the true cost is at least the number of misplaced qubits
/// divided by the max parallelism — but even just the count of misplaced
/// qubits is a very weak lower bound of 1 if any are misplaced).
///
/// Very cheap to compute but provides weak guidance.
pub struct MisplacedHeuristic {
    /// (qubit_id, encoded_target_location)
    targets: Vec<(u32, u32)>,
}

impl MisplacedHeuristic {
    /// Create from target placement.
    pub fn new(target: impl IntoIterator<Item = (u32, LocationAddr)>) -> Self {
        Self {
            targets: target.into_iter().map(|(q, l)| (q, l.encode())).collect(),
        }
    }

    /// Estimate cost-to-go: number of misplaced qubits (0 or 1 minimum).
    ///
    /// Returns 0.0 if all qubits are at target, else 1.0.
    /// This is admissible: if any qubit is misplaced, at least one more
    /// move step is needed.
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

/// Heuristic based on precomputed lane-hop distances.
///
/// For each target location, precomputes the minimum number of lane hops
/// from every reachable location (BFS on the lane graph, ignoring occupancy).
/// The heuristic returns the max hop distance over all qubits, which is
/// admissible: the worst-case qubit needs at least that many steps.
pub struct HopDistanceHeuristic {
    /// (qubit_id, encoded_target_location)
    targets: Vec<(u32, u32)>,
    /// Precomputed: encoded_target → { encoded_location → min hops }.
    /// BFS from each target, reversed edges (so we get distance TO target).
    distance_to: HashMap<u32, HashMap<u32, u32>>,
}

impl HopDistanceHeuristic {
    /// Create from target placement and a lane index.
    ///
    /// Runs BFS from each unique target location on the reversed lane graph
    /// to compute minimum hop distances.
    pub fn new(target: impl IntoIterator<Item = (u32, LocationAddr)>, index: &LaneIndex) -> Self {
        let targets: Vec<(u32, u32)> = target.into_iter().map(|(q, l)| (q, l.encode())).collect();

        // Collect unique target locations.
        let unique_targets: Vec<u32> = {
            let mut v: Vec<u32> = targets.iter().map(|&(_, t)| t).collect();
            v.sort_unstable();
            v.dedup();
            v
        };

        // Build reverse adjacency: for each location, which locations can
        // reach it in one lane hop? (If lane goes src→dst, reverse is dst→src.)
        let mut reverse_adj: HashMap<u32, Vec<u32>> = HashMap::new();
        for (mt, bus_id, dir) in index.triplets() {
            for &lane in index.lanes_for(mt, bus_id, dir) {
                if let Some((src, dst)) = index.endpoints(&lane) {
                    reverse_adj
                        .entry(dst.encode())
                        .or_default()
                        .push(src.encode());
                }
            }
        }

        // BFS from each target location on reversed edges.
        let mut distance_to: HashMap<u32, HashMap<u32, u32>> = HashMap::new();
        for &target_enc in &unique_targets {
            let mut dist: HashMap<u32, u32> = HashMap::new();
            let mut queue: VecDeque<u32> = VecDeque::new();
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

        Self {
            targets,
            distance_to,
        }
    }

    /// Estimate cost-to-go: max hop distance over all qubits.
    ///
    /// Returns 0.0 if all qubits are at target, `f64::INFINITY` if any
    /// qubit cannot reach its target.
    pub fn estimate(&self, config: &Config) -> f64 {
        let mut max_dist: u32 = 0;
        for &(qid, target_enc) in &self.targets {
            let Some(loc) = config.location_of(qid) else {
                return f64::INFINITY;
            };
            let loc_enc = loc.encode();
            if loc_enc == target_enc {
                continue;
            }
            let Some(dist_map) = self.distance_to.get(&target_enc) else {
                return f64::INFINITY;
            };
            let Some(&d) = dist_map.get(&loc_enc) else {
                return f64::INFINITY;
            };
            max_dist = max_dist.max(d);
        }
        max_dist as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bloqade_lanes_bytecode_core::arch::types::ArchSpec;

    fn example_arch_json() -> &'static str {
        r#"{
            "version": "2.0",
            "geometry": {
                "sites_per_word": 10,
                "words": [
                    {
                        "positions": { "x_start": 1.0, "y_start": 2.5, "x_spacing": [2.0, 2.0, 2.0, 2.0], "y_spacing": [2.5] },
                        "site_indices": [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [0, 1], [1, 1], [2, 1], [3, 1], [4, 1]]
                    },
                    {
                        "positions": { "x_start": 1.0, "y_start": 12.5, "x_spacing": [2.0, 2.0, 2.0, 2.0], "y_spacing": [2.5] },
                        "site_indices": [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [0, 1], [1, 1], [2, 1], [3, 1], [4, 1]]
                    }
                ]
            },
            "buses": {
                "site_buses": [
                    { "src": [0, 1, 2, 3, 4], "dst": [5, 6, 7, 8, 9] }
                ],
                "word_buses": [
                    { "src": [0], "dst": [1] }
                ]
            },
            "words_with_site_buses": [0, 1],
            "sites_with_word_buses": [5, 6, 7, 8, 9],
            "zones": [
                { "words": [0, 1] }
            ],
            "entangling_zones": [[[0, 1]]],
            "blockade_radius": 2.0,
            "measurement_mode_zones": [0]
        }"#
    }

    fn loc(word: u32, site: u32) -> LocationAddr {
        LocationAddr {
            word_id: word,
            site_id: site,
        }
    }

    fn make_index() -> LaneIndex {
        let spec: ArchSpec = serde_json::from_str(example_arch_json()).unwrap();
        LaneIndex::new(spec)
    }

    // ── MisplacedHeuristic ──

    #[test]
    fn misplaced_all_at_target() {
        let h = MisplacedHeuristic::new([(0, loc(0, 0))]);
        let config = Config::new([(0, loc(0, 0))]);
        assert_eq!(h.estimate(&config), 0.0);
    }

    #[test]
    fn misplaced_one_off() {
        let h = MisplacedHeuristic::new([(0, loc(0, 5))]);
        let config = Config::new([(0, loc(0, 0))]);
        assert_eq!(h.estimate(&config), 1.0);
    }

    #[test]
    fn misplaced_missing_qubit() {
        let h = MisplacedHeuristic::new([(99, loc(0, 0))]);
        let config = Config::new([(0, loc(0, 0))]);
        assert_eq!(h.estimate(&config), f64::INFINITY);
    }

    // ── HopDistanceHeuristic ──

    #[test]
    fn hop_at_target_returns_zero() {
        let index = make_index();
        let h = HopDistanceHeuristic::new([(0, loc(0, 5))], &index);
        let config = Config::new([(0, loc(0, 5))]);
        assert_eq!(h.estimate(&config), 0.0);
    }

    #[test]
    fn hop_one_step_away() {
        let index = make_index();
        // Site 0 → site 5 is one site bus forward hop.
        let h = HopDistanceHeuristic::new([(0, loc(0, 5))], &index);
        let config = Config::new([(0, loc(0, 0))]);
        assert_eq!(h.estimate(&config), 1.0);
    }

    #[test]
    fn hop_cross_word() {
        let index = make_index();
        // Word 0 site 5 → word 1 site 5: one word bus forward hop.
        let h = HopDistanceHeuristic::new([(0, loc(1, 5))], &index);
        let config = Config::new([(0, loc(0, 5))]);
        assert_eq!(h.estimate(&config), 1.0);
    }

    #[test]
    fn hop_two_steps() {
        let index = make_index();
        // Word 0 site 0 → word 1 site 5:
        //   site 0 → site 5 (site bus forward) → word 1 site 5 (word bus forward)
        // = 2 hops.
        let h = HopDistanceHeuristic::new([(0, loc(1, 5))], &index);
        let config = Config::new([(0, loc(0, 0))]);
        assert_eq!(h.estimate(&config), 2.0);
    }

    #[test]
    fn hop_max_over_qubits() {
        let index = make_index();
        // Qubit 0: 1 hop away, qubit 1: 2 hops away → max = 2.
        let h = HopDistanceHeuristic::new([(0, loc(0, 5)), (1, loc(1, 5))], &index);
        let config = Config::new([(0, loc(0, 0)), (1, loc(0, 0))]);
        // qubit 0: site 0 → site 5 = 1 hop
        // qubit 1: site 0 → site 5 → word 1 site 5 = 2 hops
        let est = h.estimate(&config);
        assert_eq!(est, 2.0);
    }

    #[test]
    fn hop_unreachable_returns_infinity() {
        let index = make_index();
        // Target a location that doesn't exist in the architecture.
        let h = HopDistanceHeuristic::new([(0, loc(99, 99))], &index);
        let config = Config::new([(0, loc(0, 0))]);
        assert_eq!(h.estimate(&config), f64::INFINITY);
    }

    #[test]
    fn hop_three_steps_via_site_word_site() {
        let index = make_index();
        // Word 0 site 0 → word 1 site 0:
        //   site 0 → site 5 (site bus fwd), word 0 → word 1 (word bus fwd),
        //   site 5 → site 0 (site bus bwd) = 3 hops.
        let h = HopDistanceHeuristic::new([(0, loc(1, 0))], &index);
        let config = Config::new([(0, loc(0, 0))]);
        assert_eq!(h.estimate(&config), 3.0);
    }

    #[test]
    fn hop_admissible_vs_actual() {
        // Verify the heuristic never overestimates by comparing with
        // actual A* solution cost.
        use crate::astar::astar;
        use crate::expander::ExhaustiveExpander;

        let index = make_index();
        let target = vec![(0u32, loc(1, 5))];
        let h = HopDistanceHeuristic::new(target.clone(), &index);
        let config = Config::new([(0, loc(0, 0))]);

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
