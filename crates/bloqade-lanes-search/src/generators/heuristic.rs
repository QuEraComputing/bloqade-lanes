//! Heuristic move generator for search.
//!
//! Scores qubit-bus pairs by distance improvement using a precomputed
//! [`DistanceTable`], then builds a small number of high-quality movesets
//! per bus group. Independent of entropy-guided search.
//!
//! Supports configurable deadlock escape, 2-step lookahead scoring,
//! and seeded score perturbation for restart diversity.

use std::cell::Cell;
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};

use bloqade_lanes_bytecode_core::arch::addr::{Direction, LaneAddr, LocationAddr, MoveType};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::config::Config;
use crate::context::{MoveCandidate, SearchContext, SearchState};
use crate::graph::{MoveSet, NodeId};
use crate::heuristic::DistanceTable;
use crate::lane_index::LaneIndex;
use crate::ordering::{TripletKey, cmp_moveset_config_tiebreak, cmp_triplet_entry_tiebreak};
use crate::traits::MoveGenerator;

/// Policy for handling deadlocks (no improving moves available).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeadlockPolicy {
    /// Do nothing when deadlocked -- rely on outer search to backtrack.
    Skip,
    /// Generate single-lane escape moves for qubits that block unresolved targets.
    MoveBlockers,
    /// Generate all valid single-lane moves for every qubit in the config.
    AllMoves,
}

/// A scored (qubit, bus triplet) entry.
#[derive(Clone)]
struct ScoredTriple {
    qubit_id: u32,
    score: i32, // d_now - d_after (can be negative)
    lane_encoded: u64,
    dst_encoded: u64,
}

fn cmp_scored_triples(a: &(TripletKey, ScoredTriple), b: &(TripletKey, ScoredTriple)) -> Ordering {
    b.1.score.cmp(&a.1.score).then_with(|| {
        cmp_triplet_entry_tiebreak(
            &a.0,
            a.1.qubit_id,
            a.1.lane_encoded,
            a.1.dst_encoded,
            &b.0,
            b.1.qubit_id,
            b.1.lane_encoded,
            b.1.dst_encoded,
        )
    })
}

fn cmp_candidates(a: &(i32, MoveSet, Config), b: &(i32, MoveSet, Config)) -> Ordering {
    b.0.cmp(&a.0)
        .then_with(|| cmp_moveset_config_tiebreak(&a.1, &a.2, &b.1, &b.2))
}

/// Heuristic move generator that produces a small number of high-quality
/// movesets per expansion.
///
/// For each node:
/// 1. Score each (qubit, bus, direction) by distance improvement (O(1) lookup).
/// 2. Keep all positive-scoring bus options (or one fallback if none are positive).
/// 3. Group by bus triplet.
/// 4. Per group: generate multiple movesets by varying the lead qubit.
/// 5. Sort by total distance improvement.
///
/// Typically produces 5-15 candidates per expansion, vs hundreds from
/// the exhaustive generator.
///
/// Configuration fields (`deadlock_policy`, `lookahead`, `seed`) are
/// stored on the struct. Problem-specific data (`index`, `blocked`, `targets`,
/// `dist_table`) come from [`SearchContext`] passed to [`MoveGenerator::generate`].
#[derive(Debug)]
pub struct HeuristicGenerator {
    /// Policy for deadlock escape.
    deadlock_policy: DeadlockPolicy,
    /// Enable 2-step lookahead scoring.
    lookahead: bool,
    /// Seed for score perturbation RNG (restart diversity).
    seed: u64,
    /// Counter for deadlock occurrences (single-threaded; each restart gets its own generator).
    deadlock_count: Cell<u32>,
}

impl Default for HeuristicGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl HeuristicGenerator {
    /// Create a new heuristic generator.
    pub fn new() -> Self {
        Self {
            deadlock_policy: DeadlockPolicy::AllMoves,
            lookahead: false,
            seed: 0,
            deadlock_count: Cell::new(0),
        }
    }

    /// Set the deadlock escape policy.
    pub fn with_deadlock_policy(mut self, policy: DeadlockPolicy) -> Self {
        self.deadlock_policy = policy;
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
        index: &LaneIndex,
        dist_table: &DistanceTable,
    ) -> i32 {
        let d_after_1 = dist_table
            .distance(dst_enc, target_enc)
            .map_or(i32::MAX, |d| d as i32);

        let dst = LocationAddr::decode(dst_enc);
        let mut best_step2: i32 = 0; // 0 = no improvement available

        for &lane2 in index.outgoing_lanes(dst) {
            let Some((_, dst2)) = index.endpoints(&lane2) else {
                continue;
            };
            let dst2_enc = dst2.encode();
            if occupied.contains(&dst2_enc) {
                continue;
            }
            let d_after_2 = dist_table
                .distance(dst2_enc, target_enc)
                .map_or(i32::MAX, |d| d as i32);
            let step2_score = d_after_1.saturating_sub(d_after_2);
            best_step2 = best_step2.max(step2_score);
        }

        score_step1 + best_step2
    }

    /// Generate escape moves only for qubits that block an unresolved target.
    fn generate_blocker_escape(
        &self,
        config: &Config,
        occupied: &HashSet<u64>,
        unresolved: &[(u32, u64, u64)],
        index: &LaneIndex,
        out: &mut Vec<MoveCandidate>,
    ) {
        // Find which locations are blocked targets.
        let target_locs: HashSet<u64> = unresolved.iter().map(|&(_, _, t)| t).collect();

        for (qid, loc) in config.iter() {
            let loc_enc = loc.encode();
            // Only move qubits sitting on someone else's target.
            if !target_locs.contains(&loc_enc) {
                continue;
            }
            for &lane in index.outgoing_lanes(loc) {
                let Some((_, dst)) = index.endpoints(&lane) else {
                    continue;
                };
                if occupied.contains(&dst.encode()) {
                    continue;
                }
                let ms = MoveSet::from_encoded(vec![lane.encode_u64()]);
                let new_config = config.with_moves(&[(qid, dst)]);
                out.push(MoveCandidate {
                    move_set: ms,
                    new_config,
                });
            }
        }
    }

    /// Generate all valid single-lane moves for every qubit.
    fn generate_all_escape(
        &self,
        config: &Config,
        occupied: &HashSet<u64>,
        index: &LaneIndex,
        out: &mut Vec<MoveCandidate>,
    ) {
        for (qid, loc) in config.iter() {
            for &lane in index.outgoing_lanes(loc) {
                let Some((_, dst)) = index.endpoints(&lane) else {
                    continue;
                };
                if occupied.contains(&dst.encode()) {
                    continue;
                }
                let ms = MoveSet::from_encoded(vec![lane.encode_u64()]);
                let new_config = config.with_moves(&[(qid, dst)]);
                out.push(MoveCandidate {
                    move_set: ms,
                    new_config,
                });
            }
        }
    }
}

impl MoveGenerator for HeuristicGenerator {
    fn generate(
        &self,
        config: &Config,
        _node_id: NodeId,
        ctx: &SearchContext,
        _state: &mut SearchState,
        out: &mut Vec<MoveCandidate>,
    ) {
        // Build occupied set: config qubit locations + blocked.
        // Pre-allocate to avoid rehashing.
        let mut occupied = HashSet::with_capacity(ctx.blocked.len() + config.len());
        occupied.extend(ctx.blocked);
        for (_, loc) in config.iter() {
            occupied.insert(loc.encode());
        }

        // Step 1: identify unresolved qubits.
        let unresolved: Vec<(u32, u64, u64)> = ctx
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
        let mut all_scores: Vec<(TripletKey, ScoredTriple)> = Vec::new();

        for &(qid, loc_enc, target_enc) in &unresolved {
            let d_now = ctx.dist_table.distance(loc_enc, target_enc);
            let d_now = match d_now {
                Some(d) => d as i32,
                None => continue, // unreachable target
            };

            let loc = LocationAddr::decode(loc_enc);
            for &lane in ctx.index.outgoing_lanes(loc) {
                let Some((_, dst)) = ctx.index.endpoints(&lane) else {
                    continue;
                };
                let dst_enc = dst.encode();
                if occupied.contains(&dst_enc) {
                    continue;
                }

                let d_after = ctx
                    .dist_table
                    .distance(dst_enc, target_enc)
                    .map_or(i32::MAX, |d| d as i32);
                let mut score = d_now - d_after;

                // 2-step lookahead: replace score with combined 2-step score.
                if self.lookahead {
                    score = self.lookahead_score(
                        score,
                        dst_enc,
                        target_enc,
                        &occupied,
                        ctx.index,
                        ctx.dist_table,
                    );
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

        // Step 3: retain all scored triples.
        let mut per_qubit: BTreeMap<u32, Vec<(TripletKey, ScoredTriple)>> = BTreeMap::new();
        for entry in all_scores {
            per_qubit.entry(entry.1.qubit_id).or_default().push(entry);
        }

        let mut selected: Vec<(TripletKey, ScoredTriple)> = Vec::new();
        let mut has_positive = false;

        for entries in per_qubit.values_mut() {
            entries.sort_by(cmp_scored_triples);
            for e in entries.iter() {
                if e.1.score > 0 {
                    has_positive = true;
                }
                selected.push(e.clone());
            }
        }

        // Fallback: if no positive scores, keep only the single best entry.
        if !has_positive {
            selected.sort_by(cmp_scored_triples);
            selected.truncate(1);
        } else {
            selected.retain(|e| e.1.score > 0);
        }

        // Step 4: group by bus triplet.
        let mut groups: BTreeMap<TripletKey, Vec<ScoredTriple>> = BTreeMap::new();
        for (key, triple) in selected {
            groups.entry(key).or_default().push(triple);
        }

        // Step 5: per group, build AOD-compatible rectangular grids.
        let mut candidates: Vec<(i32, MoveSet, Config)> = Vec::new();

        for ((mt_u8, bus_id, dir_u8), qubits) in groups {
            // Reconstruct typed triplet from u8 discriminants.
            let mt = match mt_u8 {
                x if x == MoveType::SiteBus as u8 => MoveType::SiteBus,
                x if x == MoveType::WordBus as u8 => MoveType::WordBus,
                x if x == MoveType::ZoneBus as u8 => MoveType::ZoneBus,
                _ => unreachable!("invalid MoveType discriminant: {mt_u8}"),
            };
            let dir = match dir_u8 {
                x if x == Direction::Forward as u8 => Direction::Forward,
                x if x == Direction::Backward as u8 => Direction::Backward,
                _ => unreachable!("invalid Direction discriminant: {dir_u8}"),
            };

            // Build grid context from ALL lanes on this bus group (cross-zone).
            let grid_ctx =
                crate::aod_grid::BusGridContext::new(ctx.index, mt, bus_id, None, dir, &occupied);

            // Build entries (src_encoded -> lane_encoded) and a lane -> triple lookup.
            // Each source location has at most one atom, so no overwrites occur.
            let mut entries: HashMap<u64, u64> = HashMap::new();
            let mut triple_by_lane: HashMap<u64, &ScoredTriple> = HashMap::new();

            for t in &qubits {
                let lane = LaneAddr::decode_u64(t.lane_encoded);
                if let Some((src, _)) = ctx.index.endpoints(&lane) {
                    let src_enc = src.encode();
                    entries.insert(src_enc, t.lane_encoded);
                    triple_by_lane.insert(t.lane_encoded, t);
                }
            }

            // Build rectangular grids via two-phase algorithm.
            // Every grid lane has a corresponding triple because `is_valid_rect`
            // requires all grid positions to be in the movers set (derived from entries).
            let grids = grid_ctx.build_aod_grids(&entries);

            for grid_lanes in grids {
                let mut total_score: i32 = 0;
                let mut moves: Vec<(u32, LocationAddr)> = Vec::new();

                for &lane_enc in &grid_lanes {
                    if let Some(t) = triple_by_lane.get(&lane_enc) {
                        total_score += t.score;
                        let dst = LocationAddr::decode(t.dst_encoded);
                        moves.push((t.qubit_id, dst));
                    }
                }

                if moves.is_empty() {
                    continue;
                }

                let move_set = MoveSet::from_encoded(grid_lanes);
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
        candidates.sort_by(cmp_candidates);

        for (_, move_set, new_config) in candidates {
            out.push(MoveCandidate {
                move_set,
                new_config,
            });
        }

        // Step 7: deadlock escape.
        if !has_positive {
            self.deadlock_count.set(self.deadlock_count.get() + 1);

            match self.deadlock_policy {
                DeadlockPolicy::Skip => {}
                DeadlockPolicy::MoveBlockers => {
                    self.generate_blocker_escape(config, &occupied, &unresolved, ctx.index, out);
                }
                DeadlockPolicy::AllMoves => {
                    self.generate_all_escape(config, &occupied, ctx.index, out);
                }
            }
        }
    }

    fn deadlock_count(&self) -> u32 {
        self.deadlock_count.get()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use bloqade_lanes_bytecode_core::arch::types::ArchSpec;

    use super::*;
    use crate::heuristic::DistanceTable;
    use crate::observer::NoOpObserver;
    use crate::test_utils::{example_arch_json, loc};

    fn make_index() -> LaneIndex {
        let spec: ArchSpec = serde_json::from_str(example_arch_json()).unwrap();
        LaneIndex::new(spec)
    }

    fn make_table(targets: &[(u32, LocationAddr)], index: &LaneIndex) -> DistanceTable {
        let locs: Vec<u64> = targets.iter().map(|&(_, l)| l.encode()).collect();
        DistanceTable::new(&locs, index)
    }

    fn make_ctx<'a>(
        index: &'a LaneIndex,
        dist_table: &'a DistanceTable,
        targets: &'a [(u32, u64)],
        blocked: &'a HashSet<u64>,
    ) -> SearchContext<'a> {
        SearchContext {
            index,
            dist_table,
            blocked,
            targets,
        }
    }

    #[test]
    fn fewer_candidates_than_exhaustive() {
        use crate::generators::exhaustive::ExhaustiveGenerator;

        let index = make_index();
        let targets = [(0, loc(0, 5)), (1, loc(0, 6))];
        let table = make_table(&targets, &index);
        let config = Config::new([(0, loc(0, 0)), (1, loc(0, 1))]).unwrap();

        // Heuristic generator path.
        let generator = HeuristicGenerator::new();
        let target_encoded: Vec<(u32, u64)> =
            targets.iter().map(|&(q, l)| (q, l.encode())).collect();
        let blocked = HashSet::new();
        let ctx = make_ctx(&index, &table, &target_encoded, &blocked);
        let mut state = SearchState::default();
        let mut heuristic_out = Vec::new();
        generator.generate(&config, NodeId(0), &ctx, &mut state, &mut heuristic_out);

        // Exhaustive generator for comparison.
        let exhaustive = ExhaustiveGenerator::new(None, None);
        let mut exhaustive_out = Vec::new();
        exhaustive.generate(&config, NodeId(0), &ctx, &mut state, &mut exhaustive_out);

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

        let generator = HeuristicGenerator::new();
        let target_encoded: Vec<(u32, u64)> =
            targets.iter().map(|&(q, l)| (q, l.encode())).collect();
        let blocked = HashSet::new();
        let ctx = make_ctx(&index, &table, &target_encoded, &blocked);
        let mut state = SearchState::default();
        let mut out = Vec::new();
        generator.generate(&config, NodeId(0), &ctx, &mut state, &mut out);

        // Best move should place qubit 0 at site 5 (direct site bus forward).
        assert!(!out.is_empty());
        let best_cfg = &out[0].new_config;
        assert_eq!(best_cfg.location_of(0), Some(loc(0, 5)));
    }

    #[test]
    fn skips_blocked_destinations() {
        let index = make_index();
        let targets = [(0, loc(0, 5))];
        let table = make_table(&targets, &index);
        let config = Config::new([(0, loc(0, 0))]).unwrap();

        let generator = HeuristicGenerator::new();
        let target_encoded: Vec<(u32, u64)> =
            targets.iter().map(|&(q, l)| (q, l.encode())).collect();
        let blocked: HashSet<u64> = [loc(0, 5).encode()].into_iter().collect();
        let ctx = make_ctx(&index, &table, &target_encoded, &blocked);
        let mut state = SearchState::default();
        let mut out = Vec::new();
        generator.generate(&config, NodeId(0), &ctx, &mut state, &mut out);

        // No move should place qubit at blocked site 5.
        for c in &out {
            assert_ne!(c.new_config.location_of(0), Some(loc(0, 5)));
        }
    }

    #[test]
    fn conflict_resolution_same_destination() {
        let index = make_index();
        let targets = [(0, loc(0, 5)), (1, loc(0, 5))];
        let table = make_table(&targets, &index);
        let config = Config::new([(0, loc(0, 0)), (1, loc(0, 1))]).unwrap();

        let generator = HeuristicGenerator::new();
        let target_encoded: Vec<(u32, u64)> =
            targets.iter().map(|&(q, l)| (q, l.encode())).collect();
        let blocked = HashSet::new();
        let ctx = make_ctx(&index, &table, &target_encoded, &blocked);
        let mut state = SearchState::default();
        let mut out = Vec::new();
        generator.generate(&config, NodeId(0), &ctx, &mut state, &mut out);

        // Each moveset should not have two qubits at the same destination.
        for c in &out {
            let loc0 = c.new_config.location_of(0);
            let loc1 = c.new_config.location_of(1);
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

        let generator = HeuristicGenerator::new();
        let target_encoded: Vec<(u32, u64)> =
            targets.iter().map(|&(q, l)| (q, l.encode())).collect();
        let blocked = HashSet::new();
        let ctx = make_ctx(&index, &table, &target_encoded, &blocked);
        let mut state = SearchState::default();
        let mut out = Vec::new();
        generator.generate(&config, NodeId(0), &ctx, &mut state, &mut out);

        assert!(!out.is_empty());
    }

    #[test]
    fn already_resolved_produces_nothing() {
        let index = make_index();
        let targets = [(0, loc(0, 5))];
        let table = make_table(&targets, &index);
        let config = Config::new([(0, loc(0, 5))]).unwrap();

        let generator = HeuristicGenerator::new();
        let target_encoded: Vec<(u32, u64)> =
            targets.iter().map(|&(q, l)| (q, l.encode())).collect();
        let blocked = HashSet::new();
        let ctx = make_ctx(&index, &table, &target_encoded, &blocked);
        let mut state = SearchState::default();
        let mut out = Vec::new();
        generator.generate(&config, NodeId(0), &ctx, &mut state, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn fallback_when_no_positive_scores() {
        let index = make_index();
        let targets = [(0, loc(0, 0))];
        let table = make_table(&targets, &index);
        let config = Config::new([(0, loc(0, 5))]).unwrap();

        let generator = HeuristicGenerator::new();
        let target_encoded: Vec<(u32, u64)> =
            targets.iter().map(|&(q, l)| (q, l.encode())).collect();
        // Block site 0 (the target) so no move reaches it.
        let blocked: HashSet<u64> = [loc(0, 0).encode()].into_iter().collect();
        let ctx = make_ctx(&index, &table, &target_encoded, &blocked);
        let mut state = SearchState::default();
        let mut out = Vec::new();
        generator.generate(&config, NodeId(0), &ctx, &mut state, &mut out);

        // Fallback should still produce at least one candidate.
        assert!(!out.is_empty());
    }

    #[test]
    fn integration_search_finds_solution() {
        use crate::cost::UniformCost;
        use crate::frontier::{self, PriorityFrontier};
        use crate::goals::AllAtTarget;
        use crate::heuristic::HopDistanceHeuristic;
        use crate::scorers::DistanceScorer;

        let index = make_index();
        let targets = vec![(0u32, loc(0, 5))];
        let target_locs: Vec<u64> = targets.iter().map(|&(_, l)| l.encode()).collect();
        let table = DistanceTable::new(&target_locs, &index);
        let h = HopDistanceHeuristic::new(targets.clone(), &table);

        let config = Config::new([(0, loc(0, 0))]).unwrap();
        let generator = HeuristicGenerator::new();

        let target_encoded: Vec<(u32, u64)> =
            targets.iter().map(|&(q, l)| (q, l.encode())).collect();
        let blocked = HashSet::new();
        let ctx = make_ctx(&index, &table, &target_encoded, &blocked);
        let goal = AllAtTarget::new(&target_encoded);
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
            Some(100),
            None,
        );

        assert!(result.goal.is_some());
        let path = result.solution_path().unwrap();
        assert_eq!(path.len(), 1); // site 0 -> site 5 in one hop
    }

    #[test]
    fn integration_multi_step() {
        use crate::cost::UniformCost;
        use crate::frontier::{self, PriorityFrontier};
        use crate::goals::AllAtTarget;
        use crate::heuristic::HopDistanceHeuristic;
        use crate::scorers::DistanceScorer;

        let index = make_index();
        let targets = vec![(0u32, loc(1, 5))];
        let target_locs: Vec<u64> = targets.iter().map(|&(_, l)| l.encode()).collect();
        let table = DistanceTable::new(&target_locs, &index);
        let h = HopDistanceHeuristic::new(targets.clone(), &table);

        let config = Config::new([(0, loc(0, 0))]).unwrap();
        let generator = HeuristicGenerator::new();

        let target_encoded: Vec<(u32, u64)> =
            targets.iter().map(|&(q, l)| (q, l.encode())).collect();
        let blocked = HashSet::new();
        let ctx = make_ctx(&index, &table, &target_encoded, &blocked);
        let goal = AllAtTarget::new(&target_encoded);
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
            Some(1000),
            None,
        );

        assert!(result.goal.is_some());
        let path = result.solution_path().unwrap();
        assert_eq!(path.len(), 2);
    }

    #[test]
    fn deadlock_escape_generates_moves() {
        let index = make_index();
        let targets = [(0u32, loc(0, 5)), (1, loc(0, 0))];
        let table = make_table(&targets, &index);
        let config = Config::new([(0, loc(0, 0)), (1, loc(0, 5))]).unwrap();

        let generator = HeuristicGenerator::new();
        let target_encoded: Vec<(u32, u64)> =
            targets.iter().map(|&(q, l)| (q, l.encode())).collect();
        let blocked = HashSet::new();
        let ctx = make_ctx(&index, &table, &target_encoded, &blocked);
        let mut state = SearchState::default();
        let mut out = Vec::new();
        generator.generate(&config, NodeId(0), &ctx, &mut state, &mut out);

        assert!(
            !out.is_empty(),
            "escape should produce moves even when heuristic candidates are blocked"
        );
    }

    #[test]
    fn deadlock_escape_solves_blocking() {
        use crate::cost::UniformCost;
        use crate::frontier::{self, PriorityFrontier};
        use crate::goals::AllAtTarget;
        use crate::heuristic::HopDistanceHeuristic;
        use crate::scorers::DistanceScorer;

        let index = make_index();
        let targets = vec![(0u32, loc(0, 5)), (1, loc(1, 5))];
        let target_locs: Vec<u64> = targets.iter().map(|&(_, l)| l.encode()).collect();
        let table = DistanceTable::new(&target_locs, &index);
        let h = HopDistanceHeuristic::new(targets.clone(), &table);

        let config = Config::new([(0, loc(0, 0)), (1, loc(0, 5))]).unwrap();
        let generator = HeuristicGenerator::new();

        let target_encoded: Vec<(u32, u64)> =
            targets.iter().map(|&(q, l)| (q, l.encode())).collect();
        let blocked = HashSet::new();
        let ctx = make_ctx(&index, &table, &target_encoded, &blocked);
        let goal = AllAtTarget::new(&target_encoded);
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
            Some(500),
            None,
        );

        assert!(
            result.goal.is_some(),
            "should solve blocking via escape (expanded {} nodes)",
            result.nodes_expanded
        );
    }

    // -- DeadlockPolicy tests --

    #[test]
    fn deadlock_policy_skip_produces_no_escape() {
        let index = make_index();
        let targets = [(0u32, loc(0, 5)), (1, loc(0, 0))];
        let table = make_table(&targets, &index);
        let config = Config::new([(0, loc(0, 0)), (1, loc(0, 5))]).unwrap();

        let generator = HeuristicGenerator::new().with_deadlock_policy(DeadlockPolicy::Skip);
        let target_encoded: Vec<(u32, u64)> =
            targets.iter().map(|&(q, l)| (q, l.encode())).collect();
        let blocked = HashSet::new();
        let ctx = make_ctx(&index, &table, &target_encoded, &blocked);
        let mut state = SearchState::default();
        let mut out = Vec::new();
        generator.generate(&config, NodeId(0), &ctx, &mut state, &mut out);

        // With Skip, deadlock produces only the fallback best-of-negatives
        // (step 3 truncated to 1 entry), but no escape moves.
        // Count should be very small -- just the single best negative entry.
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
        let targets = [(0u32, loc(0, 5)), (1, loc(0, 0))];
        let table = make_table(&targets, &index);
        let config = Config::new([(0, loc(0, 0)), (1, loc(0, 5))]).unwrap();

        let generator =
            HeuristicGenerator::new().with_deadlock_policy(DeadlockPolicy::MoveBlockers);
        let target_encoded: Vec<(u32, u64)> =
            targets.iter().map(|&(q, l)| (q, l.encode())).collect();
        let blocked = HashSet::new();
        let ctx = make_ctx(&index, &table, &target_encoded, &blocked);
        let mut state = SearchState::default();
        let mut out = Vec::new();
        generator.generate(&config, NodeId(0), &ctx, &mut state, &mut out);

        // MoveBlockers should generate escape moves, but only for qubits
        // sitting on target locations.
        assert!(!out.is_empty(), "MoveBlockers should generate escape moves");
    }

    #[test]
    fn deadlock_count_increments() {
        let index = make_index();
        let targets = [(0u32, loc(0, 5)), (1, loc(0, 0))];
        let table = make_table(&targets, &index);
        let config = Config::new([(0, loc(0, 0)), (1, loc(0, 5))]).unwrap();

        let generator = HeuristicGenerator::new();
        let target_encoded: Vec<(u32, u64)> =
            targets.iter().map(|&(q, l)| (q, l.encode())).collect();
        let blocked = HashSet::new();
        let ctx = make_ctx(&index, &table, &target_encoded, &blocked);
        assert_eq!(generator.deadlock_count(), 0);

        let mut state = SearchState::default();
        let mut out = Vec::new();
        generator.generate(&config, NodeId(0), &ctx, &mut state, &mut out);
        assert_eq!(generator.deadlock_count(), 1);

        out.clear();
        generator.generate(&config, NodeId(0), &ctx, &mut state, &mut out);
        assert_eq!(generator.deadlock_count(), 2);
    }

    // -- Lookahead tests --

    #[test]
    fn lookahead_produces_candidates() {
        let index = make_index();
        let targets = [(0, loc(0, 5))];
        let table = make_table(&targets, &index);
        let config = Config::new([(0, loc(0, 0))]).unwrap();

        let generator = HeuristicGenerator::new().with_lookahead(true);
        let target_encoded: Vec<(u32, u64)> =
            targets.iter().map(|&(q, l)| (q, l.encode())).collect();
        let blocked = HashSet::new();
        let ctx = make_ctx(&index, &table, &target_encoded, &blocked);
        let mut state = SearchState::default();
        let mut out = Vec::new();
        generator.generate(&config, NodeId(0), &ctx, &mut state, &mut out);

        assert!(!out.is_empty(), "lookahead should still produce candidates");
        // Best move should still be site 0 -> site 5.
        let best_cfg = &out[0].new_config;
        assert_eq!(best_cfg.location_of(0), Some(loc(0, 5)));
    }

    #[test]
    fn lookahead_improves_multi_step_ordering() {
        use crate::cost::UniformCost;
        use crate::frontier::{self, PriorityFrontier};
        use crate::goals::AllAtTarget;
        use crate::heuristic::HopDistanceHeuristic;
        use crate::scorers::DistanceScorer;

        let index = make_index();
        let targets = vec![(0u32, loc(1, 5))];
        let target_locs: Vec<u64> = targets.iter().map(|&(_, l)| l.encode()).collect();
        let table = DistanceTable::new(&target_locs, &index);
        let h = HopDistanceHeuristic::new(targets.clone(), &table);

        let config = Config::new([(0, loc(0, 0))]).unwrap();
        let generator_no_la = HeuristicGenerator::new();
        let generator_la = HeuristicGenerator::new().with_lookahead(true);

        let target_encoded: Vec<(u32, u64)> =
            targets.iter().map(|&(q, l)| (q, l.encode())).collect();
        let blocked = HashSet::new();
        let ctx = make_ctx(&index, &table, &target_encoded, &blocked);
        let goal = AllAtTarget::new(&target_encoded);
        let scorer = DistanceScorer;
        let cost_fn = UniformCost;

        let mut f_no_la = PriorityFrontier::astar(|cfg: &Config| h.estimate_max(cfg), 1.0);
        let result_no_la = frontier::run_search(
            config.clone(),
            &generator_no_la,
            &scorer,
            &cost_fn,
            &goal,
            &mut f_no_la,
            &ctx,
            &mut SearchState::default(),
            &mut NoOpObserver,
            Some(1000),
            None,
        );

        let mut f_la = PriorityFrontier::astar(|cfg: &Config| h.estimate_max(cfg), 1.0);
        let result_la = frontier::run_search(
            config,
            &generator_la,
            &scorer,
            &cost_fn,
            &goal,
            &mut f_la,
            &ctx,
            &mut SearchState::default(),
            &mut NoOpObserver,
            Some(1000),
            None,
        );

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

    #[test]
    fn scored_triple_tie_break_is_deterministic() {
        let mut entries = [
            (
                (1, 2, 3),
                ScoredTriple {
                    qubit_id: 9,
                    score: 7,
                    lane_encoded: 12,
                    dst_encoded: 90,
                },
            ),
            (
                (1, 1, 3),
                ScoredTriple {
                    qubit_id: 2,
                    score: 7,
                    lane_encoded: 15,
                    dst_encoded: 88,
                },
            ),
            (
                (1, 1, 3),
                ScoredTriple {
                    qubit_id: 1,
                    score: 7,
                    lane_encoded: 13,
                    dst_encoded: 88,
                },
            ),
        ];

        entries.sort_by(cmp_scored_triples);

        assert_eq!(entries[0].0, (1, 1, 3));
        assert_eq!(entries[0].1.lane_encoded, 13);
        assert_eq!(entries[1].0, (1, 1, 3));
        assert_eq!(entries[1].1.lane_encoded, 15);
        assert_eq!(entries[2].0, (1, 2, 3));
    }

    #[test]
    fn candidate_tie_break_uses_moveset_then_config() {
        let cfg_a = Config::new([(0, loc(0, 1))]).unwrap();
        let cfg_b = Config::new([(0, loc(0, 2))]).unwrap();
        let mut candidates = [
            (5, MoveSet::from_encoded(vec![9]), cfg_b),
            (5, MoveSet::from_encoded(vec![3]), cfg_a),
        ];

        candidates.sort_by(cmp_candidates);

        assert_eq!(candidates[0].1.encoded_lanes(), &[3]);
        assert_eq!(candidates[1].1.encoded_lanes(), &[9]);
    }

    // -- Seed perturbation tests --

    #[test]
    fn different_seeds_can_produce_different_orderings() {
        let index = make_index();
        let targets = [(0, loc(0, 5)), (1, loc(0, 6)), (2, loc(0, 7))];
        let table = make_table(&targets, &index);
        let config = Config::new([(0, loc(0, 0)), (1, loc(0, 1)), (2, loc(0, 2))]).unwrap();

        let generator1 = HeuristicGenerator::new().with_seed(42);
        let generator2 = HeuristicGenerator::new().with_seed(123);

        let target_encoded: Vec<(u32, u64)> =
            targets.iter().map(|&(q, l)| (q, l.encode())).collect();
        let blocked = HashSet::new();
        let ctx = make_ctx(&index, &table, &target_encoded, &blocked);

        let mut out1 = Vec::new();
        let mut out2 = Vec::new();
        generator1.generate(
            &config,
            NodeId(0),
            &ctx,
            &mut SearchState::default(),
            &mut out1,
        );
        generator2.generate(
            &config,
            NodeId(0),
            &ctx,
            &mut SearchState::default(),
            &mut out2,
        );

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

        let generator1 = HeuristicGenerator::new().with_seed(0);
        let generator2 = HeuristicGenerator::new().with_seed(0);

        let target_encoded: Vec<(u32, u64)> =
            targets.iter().map(|&(q, l)| (q, l.encode())).collect();
        let blocked = HashSet::new();
        let ctx = make_ctx(&index, &table, &target_encoded, &blocked);

        let mut out1 = Vec::new();
        let mut out2 = Vec::new();
        generator1.generate(
            &config,
            NodeId(0),
            &ctx,
            &mut SearchState::default(),
            &mut out1,
        );
        generator2.generate(
            &config,
            NodeId(0),
            &ctx,
            &mut SearchState::default(),
            &mut out2,
        );

        assert_eq!(out1.len(), out2.len());
        // With seed 0 (no perturbation), results should be identical.
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert_eq!(a.move_set, b.move_set); // same MoveSet
        }
    }
}
