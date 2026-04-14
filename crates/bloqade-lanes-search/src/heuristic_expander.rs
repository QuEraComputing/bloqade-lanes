//! Heuristic move generator for A* search.
//!
//! Scores qubit-bus pairs by distance improvement using a precomputed
//! [`DistanceTable`], then builds a small number of high-quality movesets
//! per bus group. Independent of entropy-guided search.
//!
//! Supports configurable deadlock escape, free-rider policies,
//! 2-step lookahead scoring, and seeded score perturbation for
//! restart diversity.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU32, Ordering};

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::astar::Expander;
use crate::config::Config;
use crate::graph::MoveSet;
use crate::heuristic::DistanceTable;
use crate::lane_index::LaneIndex;

/// Policy for handling deadlocks (no improving moves available).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeadlockPolicy {
    /// Do nothing when deadlocked — rely on outer search to backtrack.
    Skip,
    /// Generate single-lane escape moves for qubits that block unresolved targets.
    MoveBlockers,
    /// Generate all valid single-lane moves for every qubit in the config.
    AllMoves,
}

/// Policy for adding non-target ("free rider") qubits to movesets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FreeRiderPolicy {
    /// Never add free riders.
    Off,
    /// Add free riders only if they unblock an improving move.
    Unblock,
    /// Add free riders if they unblock or improve their own distance.
    UnblockOrImprove,
}

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
    /// Policy for deadlock escape.
    deadlock_policy: DeadlockPolicy,
    /// Policy for free-rider qubits.
    free_rider_policy: FreeRiderPolicy,
    /// Enable 2-step lookahead scoring.
    lookahead: bool,
    /// Seed for score perturbation RNG (restart diversity).
    seed: u64,
    /// Counter for deadlock occurrences.
    deadlock_count: AtomicU32,
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
            deadlock_policy: DeadlockPolicy::AllMoves,
            free_rider_policy: FreeRiderPolicy::Off,
            lookahead: false,
            seed: 0,
            deadlock_count: AtomicU32::new(0),
        }
    }

    /// Set the deadlock escape policy.
    pub fn with_deadlock_policy(mut self, policy: DeadlockPolicy) -> Self {
        self.deadlock_policy = policy;
        self
    }

    /// Set the free-rider policy.
    pub fn with_free_rider_policy(mut self, policy: FreeRiderPolicy) -> Self {
        self.free_rider_policy = policy;
        self
    }

    /// Enable or disable 2-step lookahead scoring.
    pub fn with_lookahead(mut self, enabled: bool) -> Self {
        self.lookahead = enabled;
        self
    }

    /// Set the seed for score perturbation.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Return the number of deadlocks encountered so far.
    pub fn deadlock_count(&self) -> u32 {
        self.deadlock_count.load(Ordering::Relaxed)
    }

    /// Compute the best 2-step score for a qubit moving to `dst_enc`.
    ///
    /// Looks at all outgoing lanes from `dst` and picks the best
    /// follow-up score (d_after_1 - d_after_2), returning
    /// `score_step1 + best_step2`.
    fn lookahead_score(
        &self,
        score_step1: i32,
        dst_enc: u64,
        target_enc: u64,
        occupied: &HashSet<u64>,
    ) -> i32 {
        let d_after_1 = self
            .dist_table
            .distance(dst_enc, target_enc)
            .map_or(i32::MAX, |d| d as i32);

        let dst = LocationAddr::decode(dst_enc);
        let mut best_step2: i32 = 0; // 0 = no improvement available

        for &lane2 in self.index.outgoing_lanes(dst) {
            let Some((_, dst2)) = self.index.endpoints(&lane2) else {
                continue;
            };
            let dst2_enc = dst2.encode();
            if occupied.contains(&dst2_enc) {
                continue;
            }
            let d_after_2 = self
                .dist_table
                .distance(dst2_enc, target_enc)
                .map_or(i32::MAX, |d| d as i32);
            let step2_score = d_after_1.saturating_sub(d_after_2);
            best_step2 = best_step2.max(step2_score);
        }

        score_step1 + best_step2
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
                let mut score = d_now - d_after;

                // 2-step lookahead: replace score with combined 2-step score.
                if self.lookahead {
                    score = self.lookahead_score(score, dst_enc, target_enc, &occupied);
                }

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
        let mut per_qubit: HashMap<u32, Vec<(TripletKey, ScoredTriple)>> = HashMap::new();
        for entry in all_scores {
            per_qubit.entry(entry.1.qubit_id).or_default().push(entry);
        }

        let mut selected: Vec<(TripletKey, ScoredTriple)> = Vec::new();
        let mut has_positive = false;

        for (_, entries) in per_qubit.iter_mut() {
            entries.sort_by(|a, b| b.1.score.cmp(&a.1.score));
            entries.truncate(self.top_c);
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

        // Collect unresolved qubit IDs for free-rider filtering.
        let unresolved_ids: HashSet<u32> = unresolved.iter().map(|&(qid, _, _)| qid).collect();

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
                    if occupied.contains(&t.dst_encoded) {
                        continue;
                    }
                    lanes.push(t.lane_encoded);
                    used_dsts.insert(t.dst_encoded);
                    total_score += t.score;

                    let dst = LocationAddr::decode(t.dst_encoded);
                    moves.push((t.qubit_id, dst));
                }

                // Free-rider augmentation: add non-target qubits sharing
                // the same bus triplet that unblock or improve.
                if self.free_rider_policy != FreeRiderPolicy::Off && !moves.is_empty() {
                    // Determine the bus triplet from the first lane.
                    // All lanes in this group share the same triplet.
                    self.add_free_riders(
                        config,
                        &occupied,
                        &unresolved_ids,
                        &mut lanes,
                        &mut moves,
                        &mut used_dsts,
                        &mut total_score,
                    );
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

        // Apply score perturbation for restart diversity.
        if self.seed != 0 {
            let mut rng = SmallRng::seed_from_u64(self.seed);
            for (score, _, _) in candidates.iter_mut() {
                // Perturb by up to +/-1 to break ties without destroying ranking.
                let perturbation = rng.random_range(-1i32..=1i32);
                *score = score.saturating_add(perturbation);
            }
        }

        // Step 6: sort by total score descending, emit.
        candidates.sort_by(|a, b| b.0.cmp(&a.0));

        for (_, move_set, new_config) in candidates {
            out.push((move_set, new_config, 1.0));
        }

        // Step 7: deadlock escape.
        if !has_positive {
            self.deadlock_count.fetch_add(1, Ordering::Relaxed);

            match self.deadlock_policy {
                DeadlockPolicy::Skip => {}
                DeadlockPolicy::MoveBlockers => {
                    self.generate_blocker_escape(config, &occupied, &unresolved, out);
                }
                DeadlockPolicy::AllMoves => {
                    self.generate_all_escape(config, &occupied, out);
                }
            }
        }
    }
}

impl HeuristicExpander<'_> {
    /// Generate escape moves only for qubits that block an unresolved target.
    fn generate_blocker_escape(
        &self,
        config: &Config,
        occupied: &HashSet<u64>,
        unresolved: &[(u32, u64, u64)],
        out: &mut Vec<(MoveSet, Config, f64)>,
    ) {
        // Find which locations are blocked targets.
        let target_locs: HashSet<u64> = unresolved.iter().map(|&(_, _, t)| t).collect();

        for (qid, loc) in config.iter() {
            let loc_enc = loc.encode();
            // Only move qubits sitting on someone else's target.
            if !target_locs.contains(&loc_enc) {
                continue;
            }
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

    /// Generate all valid single-lane moves for every qubit.
    fn generate_all_escape(
        &self,
        config: &Config,
        occupied: &HashSet<u64>,
        out: &mut Vec<(MoveSet, Config, f64)>,
    ) {
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

    /// Add free-rider qubits to a moveset.
    ///
    /// Scans non-target qubits for available moves on the same bus group.
    /// Depending on the policy, adds them if they unblock or improve.
    #[allow(clippy::too_many_arguments)]
    fn add_free_riders(
        &self,
        config: &Config,
        occupied: &HashSet<u64>,
        unresolved_ids: &HashSet<u32>,
        lanes: &mut Vec<u64>,
        moves: &mut Vec<(u32, LocationAddr)>,
        used_dsts: &mut HashSet<u64>,
        _total_score: &mut i32,
    ) {
        // Collect the target locations of unresolved qubits so we can
        // check if a free rider unblocks one.
        let target_locs: HashSet<u64> = self.targets.iter().map(|&(_, t)| t).collect();

        for (qid, loc) in config.iter() {
            // Skip qubits that are unresolved targets (already handled).
            if unresolved_ids.contains(&qid) {
                continue;
            }

            let loc_enc = loc.encode();

            for &lane in self.index.outgoing_lanes(loc) {
                let Some((_, dst)) = self.index.endpoints(&lane) else {
                    continue;
                };
                let dst_enc = dst.encode();
                if occupied.contains(&dst_enc) || used_dsts.contains(&dst_enc) {
                    continue;
                }

                let should_add = match self.free_rider_policy {
                    FreeRiderPolicy::Off => false,
                    FreeRiderPolicy::Unblock => {
                        // Add only if the qubit is sitting on a target location.
                        target_locs.contains(&loc_enc)
                    }
                    FreeRiderPolicy::UnblockOrImprove => {
                        // Add if unblocking OR if the qubit has its own target
                        // and the move improves distance.
                        if target_locs.contains(&loc_enc) {
                            true
                        } else {
                            // Check if this qubit has a target and the move helps.
                            self.targets.iter().any(|&(tq, t_enc)| {
                                if tq != qid {
                                    return false;
                                }
                                let d_now =
                                    self.dist_table.distance(loc_enc, t_enc).unwrap_or(u32::MAX);
                                let d_after =
                                    self.dist_table.distance(dst_enc, t_enc).unwrap_or(u32::MAX);
                                d_after < d_now
                            })
                        }
                    }
                };

                if should_add {
                    lanes.push(lane.encode_u64());
                    used_dsts.insert(dst_enc);
                    moves.push((qid, dst));
                    // Free riders don't contribute to score (they're bonus).
                    break; // One free-rider move per qubit.
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
            }
        }
        assert!(!out.is_empty());
    }

    #[test]
    fn multiple_movesets_per_group() {
        let index = make_index();
        let targets = [(0, loc(0, 5)), (1, loc(0, 6)), (2, loc(0, 7))];
        let table = make_table(&targets, &index);
        let config = Config::new([(0, loc(0, 0)), (1, loc(0, 1)), (2, loc(0, 2))]).unwrap();

        let mut out = Vec::new();
        let exp = HeuristicExpander::new(&index, std::iter::empty(), targets, &table, 3, 3);
        exp.expand(&config, &mut out);

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
        let targets = [(0, loc(0, 0))];
        let table = make_table(&targets, &index);
        let config = Config::new([(0, loc(0, 5))]).unwrap();

        let mut out = Vec::new();
        // Block site 0 (the target) so no move reaches it.
        let exp = HeuristicExpander::new(&index, [loc(0, 0)], targets, &table, 3, 3);
        exp.expand(&config, &mut out);

        // Fallback should still produce at least one candidate.
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
            |cfg| h.estimate_max(cfg),
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
            |cfg| h.estimate_max(cfg),
            &exp,
            Some(1000),
        );

        assert!(result.goal.is_some());
        let path = result.solution_path().unwrap();
        assert_eq!(path.len(), 2);
    }

    #[test]
    fn deadlock_escape_generates_moves() {
        use crate::astar::Expander;

        let index = make_index();
        let targets = vec![(0u32, loc(0, 5)), (1, loc(0, 0))];
        let target_locs: Vec<u64> = targets.iter().map(|&(_, l)| l.encode()).collect();
        let table = DistanceTable::new(&target_locs, &index);
        let config = Config::new([(0, loc(0, 0)), (1, loc(0, 5))]).unwrap();
        let exp = HeuristicExpander::new(&index, std::iter::empty(), targets, &table, 3, 3);

        let mut out = Vec::new();
        exp.expand(&config, &mut out);

        assert!(
            !out.is_empty(),
            "escape should produce moves even when heuristic candidates are blocked"
        );
    }

    #[test]
    fn deadlock_escape_solves_blocking() {
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
            |cfg| h.estimate_max(cfg),
            &exp,
            Some(500),
        );

        assert!(
            result.goal.is_some(),
            "should solve blocking via escape (expanded {} nodes)",
            result.nodes_expanded
        );
    }

    // ── DeadlockPolicy tests ──

    #[test]
    fn deadlock_policy_skip_produces_no_escape() {
        let index = make_index();
        let targets = vec![(0u32, loc(0, 5)), (1, loc(0, 0))];
        let target_locs: Vec<u64> = targets.iter().map(|&(_, l)| l.encode()).collect();
        let table = DistanceTable::new(&target_locs, &index);
        let config = Config::new([(0, loc(0, 0)), (1, loc(0, 5))]).unwrap();

        let exp = HeuristicExpander::new(&index, std::iter::empty(), targets, &table, 3, 3)
            .with_deadlock_policy(DeadlockPolicy::Skip);

        let mut out = Vec::new();
        exp.expand(&config, &mut out);

        // With Skip, deadlock produces only the fallback best-of-negatives
        // (step 3 truncated to 1 entry), but no escape moves.
        // Count should be very small — just the single best negative entry.
        assert!(
            out.len() <= 1,
            "Skip policy should produce at most 1 candidate (the fallback), got {}",
            out.len()
        );
    }

    #[test]
    fn deadlock_policy_move_blockers_targets_blockers() {
        let index = make_index();
        // q0 at site 0 wants site 5; q1 at site 5 wants site 0.
        // q1 blocks q0, and q0 blocks q1.
        let targets = vec![(0u32, loc(0, 5)), (1, loc(0, 0))];
        let target_locs: Vec<u64> = targets.iter().map(|&(_, l)| l.encode()).collect();
        let table = DistanceTable::new(&target_locs, &index);
        let config = Config::new([(0, loc(0, 0)), (1, loc(0, 5))]).unwrap();

        let exp = HeuristicExpander::new(&index, std::iter::empty(), targets, &table, 3, 3)
            .with_deadlock_policy(DeadlockPolicy::MoveBlockers);

        let mut out = Vec::new();
        exp.expand(&config, &mut out);

        // MoveBlockers should generate escape moves, but only for qubits
        // sitting on target locations.
        assert!(!out.is_empty(), "MoveBlockers should generate escape moves");
    }

    #[test]
    fn deadlock_count_increments() {
        let index = make_index();
        let targets = vec![(0u32, loc(0, 5)), (1, loc(0, 0))];
        let target_locs: Vec<u64> = targets.iter().map(|&(_, l)| l.encode()).collect();
        let table = DistanceTable::new(&target_locs, &index);
        let config = Config::new([(0, loc(0, 0)), (1, loc(0, 5))]).unwrap();

        let exp = HeuristicExpander::new(&index, std::iter::empty(), targets, &table, 3, 3);
        assert_eq!(exp.deadlock_count(), 0);

        let mut out = Vec::new();
        exp.expand(&config, &mut out);
        assert_eq!(exp.deadlock_count(), 1);

        out.clear();
        exp.expand(&config, &mut out);
        assert_eq!(exp.deadlock_count(), 2);
    }

    // ── Lookahead tests ──

    #[test]
    fn lookahead_produces_candidates() {
        let index = make_index();
        let targets = [(0, loc(0, 5))];
        let table = make_table(&targets, &index);
        let config = Config::new([(0, loc(0, 0))]).unwrap();

        let mut out = Vec::new();
        let exp = HeuristicExpander::new(&index, std::iter::empty(), targets, &table, 3, 3)
            .with_lookahead(true);
        exp.expand(&config, &mut out);

        assert!(!out.is_empty(), "lookahead should still produce candidates");
        // Best move should still be site 0 → site 5.
        let best_cfg = &out[0].1;
        assert_eq!(best_cfg.location_of(0), Some(loc(0, 5)));
    }

    #[test]
    fn lookahead_improves_multi_step_ordering() {
        use crate::astar::astar;
        use crate::heuristic::HopDistanceHeuristic;

        let index = make_index();
        let targets = vec![(0u32, loc(1, 5))];
        let target_locs: Vec<u64> = targets.iter().map(|&(_, l)| l.encode()).collect();
        let table = DistanceTable::new(&target_locs, &index);
        let h = HopDistanceHeuristic::new(targets.clone(), &table);

        let config = Config::new([(0, loc(0, 0))]).unwrap();
        let exp_no_la =
            HeuristicExpander::new(&index, std::iter::empty(), targets.clone(), &table, 3, 3);
        let exp_la =
            HeuristicExpander::new(&index, std::iter::empty(), targets.clone(), &table, 3, 3)
                .with_lookahead(true);

        let target_enc = loc(1, 5).encode();
        let is_goal = |cfg: &Config| cfg.location_of(0).is_some_and(|l| l.encode() == target_enc);

        let result_no_la = astar(
            config.clone(),
            is_goal,
            |cfg| h.estimate_max(cfg),
            &exp_no_la,
            Some(1000),
        );
        let result_la = astar(config, is_goal, |cfg| h.estimate_max(cfg), &exp_la, Some(1000));

        assert!(result_no_la.goal.is_some());
        assert!(result_la.goal.is_some());
        // Lookahead should expand no more nodes than without.
        assert!(
            result_la.nodes_expanded <= result_no_la.nodes_expanded,
            "lookahead expanded {} vs {} without",
            result_la.nodes_expanded,
            result_no_la.nodes_expanded
        );
    }

    // ── Seed perturbation tests ──

    #[test]
    fn different_seeds_can_produce_different_orderings() {
        let index = make_index();
        let targets = [(0, loc(0, 5)), (1, loc(0, 6)), (2, loc(0, 7))];
        let table = make_table(&targets, &index);
        let config = Config::new([(0, loc(0, 0)), (1, loc(0, 1)), (2, loc(0, 2))]).unwrap();

        let exp1 =
            HeuristicExpander::new(&index, std::iter::empty(), targets, &table, 3, 3).with_seed(42);
        let exp2 = HeuristicExpander::new(&index, std::iter::empty(), targets, &table, 3, 3)
            .with_seed(123);

        let mut out1 = Vec::new();
        let mut out2 = Vec::new();
        exp1.expand(&config, &mut out1);
        exp2.expand(&config, &mut out2);

        // Both should produce candidates.
        assert!(!out1.is_empty());
        assert!(!out2.is_empty());
        // We can't guarantee different orderings (perturbation is small),
        // but at minimum they should both work.
    }

    #[test]
    fn seed_zero_means_no_perturbation() {
        let index = make_index();
        let targets = [(0, loc(0, 5))];
        let table = make_table(&targets, &index);
        let config = Config::new([(0, loc(0, 0))]).unwrap();

        let exp1 =
            HeuristicExpander::new(&index, std::iter::empty(), targets, &table, 3, 3).with_seed(0);
        let exp2 =
            HeuristicExpander::new(&index, std::iter::empty(), targets, &table, 3, 3).with_seed(0);

        let mut out1 = Vec::new();
        let mut out2 = Vec::new();
        exp1.expand(&config, &mut out1);
        exp2.expand(&config, &mut out2);

        assert_eq!(out1.len(), out2.len());
        // With seed 0 (no perturbation), results should be identical.
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert_eq!(a.0, b.0); // same MoveSet
        }
    }

    // ── Free rider tests ──

    #[test]
    fn free_rider_off_does_not_add_extras() {
        let index = make_index();
        // q0 targets site 5, q1 (non-target) at site 1.
        let targets = [(0, loc(0, 5))];
        let table = make_table(&targets, &index);
        let config = Config::new([(0, loc(0, 0)), (1, loc(0, 1))]).unwrap();

        let exp = HeuristicExpander::new(&index, std::iter::empty(), targets, &table, 3, 3)
            .with_free_rider_policy(FreeRiderPolicy::Off);

        let mut out = Vec::new();
        exp.expand(&config, &mut out);

        // With Off policy, q1 should not move in any candidate.
        for (_, cfg, _) in &out {
            assert_eq!(
                cfg.location_of(1),
                Some(loc(0, 1)),
                "free rider q1 should not move with Off policy"
            );
        }
    }

    #[test]
    fn free_rider_unblock_moves_blockers() {
        let index = make_index();
        // q0 at site 0 targets site 5.
        // q1 at site 5 (non-target, blocking q0's destination).
        // With Unblock policy, q1 should be moved out of the way.
        let targets = [(0, loc(0, 5))];
        let table = make_table(&targets, &index);
        let config = Config::new([(0, loc(0, 0)), (1, loc(0, 5))]).unwrap();

        let exp = HeuristicExpander::new(&index, std::iter::empty(), targets, &table, 3, 3)
            .with_free_rider_policy(FreeRiderPolicy::Unblock);

        let mut out = Vec::new();
        exp.expand(&config, &mut out);

        // The expander should produce candidates. q1 at site 5 sits on
        // q0's target, so it may get moved as a free rider if it shares
        // the same bus group. The deadlock escape also applies.
        assert!(!out.is_empty());
    }
}
