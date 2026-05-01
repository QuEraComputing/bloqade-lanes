//! Admissible heuristics and precomputed distance tables for A* search.
//!
//! [`DistanceTable`] precomputes BFS lane-hop distances from target locations
//! and is shared between the heuristic and the move generator.
//!
//! [`MisplacedHeuristic`] is a simple count-based heuristic.
//! [`HopDistanceHeuristic`] uses the distance table for a tighter bound.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, VecDeque};

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
    /// Flattened (from, to) → distance for faster single-probe lookup.
    flat_distance: HashMap<(u64, u64), u32>,
    /// Optional time-weighted distances: encoded_target → { encoded_location → min time (µs) }
    time_distance_to: Option<HashMap<u64, HashMap<u64, f64>>>,
    /// Fastest lane duration across all lanes (for normalization).
    fastest_lane_us: Option<f64>,
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

        // Build flattened index for single-probe lookups.
        let mut flat_distance = HashMap::with_capacity(distance_to.values().map(|m| m.len()).sum());
        for (&target, sources) in &distance_to {
            for (&source, &dist) in sources {
                flat_distance.insert((source, target), dist);
            }
        }

        Self {
            distance_to,
            flat_distance,
            time_distance_to: None,
            fastest_lane_us: None,
        }
    }

    /// Also compute time-weighted distances using Dijkstra with lane durations.
    ///
    /// Only call when `w_t > 0`. Skips if the index has no lane duration data.
    pub fn with_time_distances(mut self, index: &LaneIndex) -> Self {
        let fastest = match index.fastest_lane_duration_us() {
            Some(f) => f,
            None => return self, // no path data — fall back to hop-count only
        };
        self.fastest_lane_us = Some(fastest);

        // Build reverse weighted adjacency: dst_enc → [(src_enc, duration_us)]
        let mut reverse_adj: HashMap<u64, Vec<(u64, f64)>> = HashMap::new();
        for (mt, bus_id, zone_id, dir) in index.bus_groups() {
            for &lane in index.lanes_for(mt, bus_id, zone_id, dir) {
                if let Some((src, dst)) = index.endpoints(&lane)
                    && let Some(dur) = index.lane_duration_us(&lane)
                {
                    reverse_adj
                        .entry(dst.encode())
                        .or_default()
                        .push((src.encode(), dur));
                }
            }
        }

        // Dijkstra from each target on reversed weighted edges.
        let targets: Vec<u64> = self.distance_to.keys().copied().collect();
        let mut time_dist_to: HashMap<u64, HashMap<u64, f64>> = HashMap::new();

        for target_enc in targets {
            let mut dist: HashMap<u64, f64> = HashMap::new();
            let mut heap = BinaryHeap::new();
            dist.insert(target_enc, 0.0);
            heap.push(DijkstraEntry {
                cost: 0.0,
                node: target_enc,
            });

            while let Some(entry) = heap.pop() {
                if entry.cost > *dist.get(&entry.node).unwrap_or(&f64::MAX) {
                    continue;
                }
                if let Some(preds) = reverse_adj.get(&entry.node) {
                    for &(pred, dur) in preds {
                        let new_cost = entry.cost + dur;
                        if new_cost < *dist.get(&pred).unwrap_or(&f64::MAX) {
                            dist.insert(pred, new_cost);
                            heap.push(DijkstraEntry {
                                cost: new_cost,
                                node: pred,
                            });
                        }
                    }
                }
            }
            time_dist_to.insert(target_enc, dist);
        }

        self.time_distance_to = Some(time_dist_to);
        self
    }

    /// O(1) lookup: minimum lane hops from `from_encoded` to `to_target_encoded`.
    ///
    /// Returns `None` if the target is unknown or the source is unreachable.
    pub fn distance(&self, from_encoded: u64, to_target_encoded: u64) -> Option<u32> {
        self.flat_distance
            .get(&(from_encoded, to_target_encoded))
            .copied()
    }

    /// O(1) lookup: minimum time (µs) from `from_encoded` to `to_target_encoded`.
    ///
    /// Returns `None` if time distances weren't computed or location is unreachable.
    pub fn time_distance(&self, from_encoded: u64, to_target_encoded: u64) -> Option<f64> {
        self.time_distance_to
            .as_ref()?
            .get(&to_target_encoded)?
            .get(&from_encoded)
            .copied()
    }

    /// Fastest lane duration for normalization. `None` if no time data.
    pub fn fastest_lane_us(&self) -> Option<f64> {
        self.fastest_lane_us
    }

    /// Iterate over all `(source_encoded, hop_distance)` pairs for a given target.
    ///
    /// Calls `f(source_encoded, distance)` for every reachable source.
    /// Does nothing if `target_encoded` is unknown.
    pub fn for_each_source(&self, target_encoded: u64, mut f: impl FnMut(u64, u32)) {
        if let Some(sources) = self.distance_to.get(&target_encoded) {
            for (&src, &d) in sources {
                f(src, d);
            }
        }
    }
}

/// Dijkstra priority queue entry (min-heap by cost).
struct DijkstraEntry {
    cost: f64,
    node: u64,
}

impl PartialEq for DijkstraEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cost.total_cmp(&other.cost) == Ordering::Equal
    }
}

impl Eq for DijkstraEntry {}

impl Ord for DijkstraEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse for min-heap.
        other.cost.total_cmp(&self.cost)
    }
}

impl PartialOrd for DijkstraEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
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
    #[deprecated(note = "use estimate_max() or estimate_sum() directly")]
    pub fn estimate(&self, config: &Config) -> f64 {
        self.estimate_max(config)
    }
}

// ── PairDistanceHeuristic ──────────────────────────────────────────

/// Heuristic for loose-goal search based on per-word-pair minimum distances.
///
/// For each CZ pair `(qa, qb)`, finds the cheapest entangling word pair
/// using precomputed [`WordPairDistances`]. The cost for a word pair is
/// `max(min_dist_to_word_a[qa_loc], min_dist_to_word_b[qb_loc])`, capturing
/// that both qubits must reach the same word pair.
///
/// Per-node cost is O(pairs × word_pairs), with word_pairs typically 1-4.
pub struct PairDistanceHeuristic<'a> {
    pairs: Vec<(u32, u32)>,
    word_pair_dists: &'a crate::entangling::WordPairDistances,
}

impl<'a> PairDistanceHeuristic<'a> {
    /// Create from CZ pairs and precomputed word-pair distances.
    pub fn new(
        pairs: &[(u32, u32)],
        word_pair_dists: &'a crate::entangling::WordPairDistances,
    ) -> Self {
        Self {
            pairs: pairs.to_vec(),
            word_pair_dists,
        }
    }

    /// Admissible heuristic: max over all pairs of min achievable cost.
    ///
    /// For each CZ pair, tries both qubit-to-word assignments on each word
    /// pair and takes the best. Then returns the max across all CZ pairs.
    pub fn estimate_max(&self, config: &Config) -> f64 {
        let mut max_pair_cost: u32 = 0;
        for &(qa, qb) in &self.pairs {
            let cost = self.min_pair_cost(qa, qb, config);
            if cost == u32::MAX {
                return f64::INFINITY;
            }
            max_pair_cost = max_pair_cost.max(cost);
        }
        max_pair_cost as f64
    }

    /// Sum heuristic: sum over all pairs of min achievable cost.
    ///
    /// Not admissible (overestimates due to bus parallelism), but gives
    /// better ordering for IDS/DFS.
    pub fn estimate_sum(&self, config: &Config) -> f64 {
        let mut total: u32 = 0;
        for &(qa, qb) in &self.pairs {
            let cost = self.min_pair_cost(qa, qb, config);
            if cost == u32::MAX {
                return f64::INFINITY;
            }
            total = total.saturating_add(cost);
        }
        total as f64
    }

    /// Min cost for a single CZ pair across all word pairs and both assignments.
    fn min_pair_cost(&self, qa: u32, qb: u32, config: &Config) -> u32 {
        let Some(loc_a) = config.location_of(qa) else {
            return u32::MAX;
        };
        let Some(loc_b) = config.location_of(qb) else {
            return u32::MAX;
        };
        let a_enc = loc_a.encode();
        let b_enc = loc_b.encode();

        let mut min_cost = u32::MAX;
        for (_wp, dist_a, dist_b) in self.word_pair_dists.iter() {
            // Assignment 1: qa → word_a, qb → word_b
            let d_a1 = dist_a.get(&a_enc).copied().unwrap_or(u32::MAX);
            let d_b1 = dist_b.get(&b_enc).copied().unwrap_or(u32::MAX);
            let cost1 = d_a1.max(d_b1);
            min_cost = min_cost.min(cost1);

            // Assignment 2: qa → word_b, qb → word_a
            let d_a2 = dist_b.get(&a_enc).copied().unwrap_or(u32::MAX);
            let d_b2 = dist_a.get(&b_enc).copied().unwrap_or(u32::MAX);
            let cost2 = d_a2.max(d_b2);
            min_cost = min_cost.min(cost2);
        }
        min_cost
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observer::NoOpObserver;
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
        assert_eq!(h.estimate_max(&config), 0.0);
    }

    #[test]
    fn hop_one_step_away() {
        let index = make_index();
        let table = make_table(&[(0, loc(0, 5))], &index);
        let h = HopDistanceHeuristic::new([(0, loc(0, 5))], &table);
        let config = Config::new([(0, loc(0, 0))]).unwrap();
        assert_eq!(h.estimate_max(&config), 1.0);
    }

    #[test]
    fn hop_cross_word() {
        let index = make_index();
        let table = make_table(&[(0, loc(1, 5))], &index);
        let h = HopDistanceHeuristic::new([(0, loc(1, 5))], &table);
        let config = Config::new([(0, loc(0, 5))]).unwrap();
        assert_eq!(h.estimate_max(&config), 1.0);
    }

    #[test]
    fn hop_two_steps() {
        let index = make_index();
        let table = make_table(&[(0, loc(1, 5))], &index);
        let h = HopDistanceHeuristic::new([(0, loc(1, 5))], &table);
        let config = Config::new([(0, loc(0, 0))]).unwrap();
        assert_eq!(h.estimate_max(&config), 2.0);
    }

    #[test]
    fn hop_max_over_qubits() {
        let index = make_index();
        let targets = [(0, loc(0, 5)), (1, loc(1, 5))];
        let table = make_table(&targets, &index);
        let h = HopDistanceHeuristic::new(targets, &table);
        let config = Config::new([(0, loc(0, 0)), (1, loc(0, 0))]).unwrap();
        assert_eq!(h.estimate_max(&config), 2.0);
    }

    #[test]
    fn hop_unreachable_returns_infinity() {
        let index = make_index();
        let table = make_table(&[(0, loc(99, 99))], &index);
        let h = HopDistanceHeuristic::new([(0, loc(99, 99))], &table);
        let config = Config::new([(0, loc(0, 0))]).unwrap();
        assert_eq!(h.estimate_max(&config), f64::INFINITY);
    }

    #[test]
    fn hop_three_steps_via_site_word_site() {
        let index = make_index();
        let table = make_table(&[(0, loc(1, 0))], &index);
        let h = HopDistanceHeuristic::new([(0, loc(1, 0))], &table);
        let config = Config::new([(0, loc(0, 0))]).unwrap();
        assert_eq!(h.estimate_max(&config), 3.0);
    }

    #[test]
    fn hop_admissible_vs_actual() {
        use std::collections::HashSet;

        use crate::context::{SearchContext, SearchState};
        use crate::cost::UniformCost;
        use crate::frontier::{self, PriorityFrontier};
        use crate::generators::exhaustive::ExhaustiveGenerator;
        use crate::goals::AllAtTarget;
        use crate::scorers::DistanceScorer;

        let index = make_index();
        let targets = vec![(0u32, loc(1, 5))];
        let table = make_table(&targets, &index);
        let h = HopDistanceHeuristic::new(targets.clone(), &table);
        let config = Config::new([(0, loc(0, 0))]).unwrap();

        let estimate = h.estimate_max(&config);

        let target_encoded: Vec<(u32, u64)> =
            targets.iter().map(|&(q, l)| (q, l.encode())).collect();
        let target_locs: Vec<u64> = targets.iter().map(|&(_, l)| l.encode()).collect();
        let dist_table = DistanceTable::new(&target_locs, &index);
        let blocked = HashSet::new();
        let ctx = SearchContext {
            index: &index,
            dist_table: &dist_table,
            blocked: &blocked,
            targets: &target_encoded,
            cz_pairs: None,
        };

        let generator = ExhaustiveGenerator::new(None, None);
        let scorer = DistanceScorer;
        let cost = UniformCost;
        let goal = AllAtTarget::new(&target_encoded);
        let mut f = PriorityFrontier::astar(|cfg: &Config| h.estimate_max(cfg), 1.0);

        let result = frontier::run_search(
            config,
            &generator,
            &scorer,
            &cost,
            &goal,
            &mut f,
            &ctx,
            &mut SearchState::default(),
            &mut NoOpObserver,
            Some(1000),
            None,
        );

        assert!(result.goal.is_some());
        let actual_cost = result.graph.g_score(result.goal.unwrap());
        assert!(
            estimate <= actual_cost,
            "heuristic {estimate} should not exceed actual cost {actual_cost}"
        );
    }

    // ── PairDistanceHeuristic ──

    fn make_pair_heuristic(
        index: &LaneIndex,
    ) -> (DistanceTable, crate::entangling::WordPairDistances) {
        let arch = index.arch_spec();
        let locs = crate::entangling::all_entangling_locations(arch);
        let dist_table = DistanceTable::new(&locs, index);
        let word_pairs = crate::entangling::enumerate_word_pairs(arch);
        let wpd =
            crate::entangling::WordPairDistances::from_dist_table(&word_pairs, arch, &dist_table);
        (dist_table, wpd)
    }

    #[test]
    fn pair_heuristic_zero_at_goal() {
        let index = make_index();
        let (_dt, wpd) = make_pair_heuristic(&index);
        let h = PairDistanceHeuristic::new(&[(0, 1)], &wpd);
        // Both qubits already at valid entangling positions.
        let config = Config::new([(0, loc(0, 5)), (1, loc(1, 5))]).unwrap();
        assert_eq!(h.estimate_max(&config), 0.0);
    }

    #[test]
    fn pair_heuristic_nonzero_away_from_goal() {
        let index = make_index();
        let (_dt, wpd) = make_pair_heuristic(&index);
        let h = PairDistanceHeuristic::new(&[(0, 1)], &wpd);
        // q0 on word 0 site 0, q1 on word 0 site 0 — same word, need to
        // move q1 to word 1 (site bus + word bus).
        let config = Config::new([(0, loc(0, 0)), (1, loc(0, 1))]).unwrap();
        assert!(
            h.estimate_max(&config) >= 1.0,
            "both on same word should need at least 1 hop: {}",
            h.estimate_max(&config)
        );
    }

    #[test]
    fn pair_heuristic_missing_qubit() {
        let index = make_index();
        let (_dt, wpd) = make_pair_heuristic(&index);
        let h = PairDistanceHeuristic::new(&[(0, 1)], &wpd);
        let config = Config::new([(0, loc(0, 5))]).unwrap();
        assert_eq!(h.estimate_max(&config), f64::INFINITY);
    }

    #[test]
    fn pair_heuristic_sum_ge_max() {
        let index = make_index();
        let (_dt, wpd) = make_pair_heuristic(&index);
        let h = PairDistanceHeuristic::new(&[(0, 1), (2, 3)], &wpd);
        let config = Config::new([
            (0, loc(0, 0)),
            (1, loc(1, 0)),
            (2, loc(0, 1)),
            (3, loc(1, 1)),
        ])
        .unwrap();
        assert!(h.estimate_sum(&config) >= h.estimate_max(&config));
    }

    #[test]
    fn pair_heuristic_admissible() {
        // Verify h <= actual cost by solving with A*.
        use crate::context::{SearchContext, SearchState};
        use crate::cost::UniformCost;
        use crate::frontier::{self, PriorityFrontier};
        use crate::generators::exhaustive::ExhaustiveGenerator;
        use crate::goals::EntanglingConstraintGoal;
        use crate::scorers::DistanceScorer;
        use std::collections::HashSet;

        let index = make_index();
        let arch = index.arch_spec();
        let eset = crate::entangling::build_entangling_set(arch);
        let locs = crate::entangling::all_entangling_locations(arch);
        let dist_table = DistanceTable::new(&locs, &index);
        let word_pairs = crate::entangling::enumerate_word_pairs(arch);
        let wpd =
            crate::entangling::WordPairDistances::from_dist_table(&word_pairs, arch, &dist_table);

        let cz_pairs = [(0u32, 1u32)];
        let h = PairDistanceHeuristic::new(&cz_pairs, &wpd);
        let config = Config::new([(0, loc(0, 0)), (1, loc(1, 0))]).unwrap();
        let estimate = h.estimate_max(&config);

        // Solve with A* to get actual cost.
        let goal = EntanglingConstraintGoal::new(&cz_pairs, eset);
        let target_encoded = crate::entangling::greedy_assign_pairs(
            &cz_pairs,
            &config,
            arch,
            &dist_table,
            0,
            None,
            0.0,
            0.0,
        );
        let blocked = HashSet::new();
        let ctx = SearchContext {
            index: &index,
            dist_table: &dist_table,
            blocked: &blocked,
            targets: &target_encoded,
            cz_pairs: None,
        };
        let generator = ExhaustiveGenerator::new(None, None);
        let scorer = DistanceScorer;
        let cost_fn = UniformCost;
        let mut f = PriorityFrontier::astar(|cfg: &Config| h.estimate_max(cfg), 1.0);

        let result = frontier::run_search(
            config,
            &generator,
            &scorer,
            &cost_fn,
            &goal,
            &mut f,
            &ctx,
            &mut SearchState::default(),
            &mut NoOpObserver,
            Some(5000),
            None,
        );

        assert!(result.goal.is_some(), "should find a solution");
        let actual_cost = result.graph.g_score(result.goal.unwrap());
        assert!(
            estimate <= actual_cost,
            "pair heuristic {estimate} should not exceed actual cost {actual_cost}"
        );
    }
}
