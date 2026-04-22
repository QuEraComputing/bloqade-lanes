//! Entropy-guided search for move synthesis.
//!
//! Port of the Python `EntropyGuidedSearch` algorithm. Walks a single path
//! down the search tree, using per-node entropy to shift scoring from
//! distance-focused (low entropy) to mobility-focused (high entropy).
//! Backtracks by walking parent pointers when entropy exceeds a threshold,
//! and falls back to greedy single-qubit routing when fully stuck.
//!
//! Self-contained module: provides its own [`solve`] entry point that builds
//! all required infrastructure internally. Can be removed by deleting this
//! file and the one-line references in `lib.rs`, `solve.rs`, and the Python
//! bindings.

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};

use bloqade_lanes_bytecode_core::arch::addr::{Direction, LaneAddr, LocationAddr, MoveType};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::aod_grid::BusGridContext;
use crate::astar::SearchResult;
use crate::config::Config;
use crate::context::SearchContext;
use crate::graph::{MoveSet, NodeId, SearchGraph};
use crate::heuristic::DistanceTable;
use crate::lane_index::LaneIndex;
use crate::ordering::{
    TripletKey, cmp_moveset_config_tiebreak, cmp_qubit_lane_dst_tiebreak,
    cmp_triplet_entry_tiebreak,
};
use crate::traits::Goal;

// ── Parameters ─────────────────────────────────────────────────────

/// Tunable parameters for entropy-guided search.
/// Mirrors the Python `SearchParams` dataclass.
#[derive(Debug, Clone)]
pub struct EntropyParams {
    // Per-qubit-bus scoring.
    pub w_d: f64,
    pub w_m: f64,
    // Moveset scoring.
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
    // Search control.
    pub max_candidates: usize,
    pub reversion_steps: u32,
    pub delta_e: u32,
    pub e_max: u32,
    pub max_goal_candidates: usize,
    // Expander settings.
    pub max_movesets_per_group: usize,
    /// Time-distance blend weight (0.0 = hop-count only, 1.0 = time only).
    pub w_t: f64,
}

impl Default for EntropyParams {
    fn default() -> Self {
        Self {
            // Synced with Python SearchParams defaults (commit 9b470b3).
            w_d: 0.95,
            w_m: 0.8,
            alpha: 80.0,
            beta: 3.0,
            gamma: 3.1,
            max_candidates: 4,
            reversion_steps: 1,
            delta_e: 1,
            e_max: 4,
            max_goal_candidates: 3,
            max_movesets_per_group: 3,
            w_t: 0.95,
        }
    }
}

// ── Per-node state ─────────────────────────────────────────────────

#[derive(Debug)]
struct EntropyState {
    entropy: u32,
    candidates_tried: usize,
    candidate_cache: Vec<(MoveSet, Config, f64)>,
    /// Encoded lane vecs of movesets already attempted from this node.
    tried_moves: HashSet<Vec<u64>>,
    /// Encoded lane vecs of movesets that failed (collision/transposition).
    /// Skipped on retry to avoid repeating known failures.
    failed_candidates: HashSet<Vec<u64>>,
    /// Number of actually-created children (is_new=true from graph.insert).
    n_children: usize,
}

struct ScoredEntry {
    qubit_id: u32,
    score: f64,
    lane_encoded: u64,
    dst_encoded: u64,
}

fn cmp_scored_entries(a: &(TripletKey, ScoredEntry), b: &(TripletKey, ScoredEntry)) -> Ordering {
    b.1.score.total_cmp(&a.1.score).then_with(|| {
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

fn cmp_group_entries(a: &ScoredEntry, b: &ScoredEntry) -> Ordering {
    b.score.total_cmp(&a.score).then_with(|| {
        cmp_qubit_lane_dst_tiebreak(
            a.qubit_id,
            a.lane_encoded,
            a.dst_encoded,
            b.qubit_id,
            b.lane_encoded,
            b.dst_encoded,
        )
    })
}

fn cmp_scored_candidates(a: &(f64, MoveSet, Config), b: &(f64, MoveSet, Config)) -> Ordering {
    b.0.total_cmp(&a.0)
        .then_with(|| cmp_moveset_config_tiebreak(&a.1, &a.2, &b.1, &b.2))
}

impl Default for EntropyState {
    fn default() -> Self {
        Self {
            entropy: 1,
            candidates_tried: 0,
            candidate_cache: Vec::new(),
            tried_moves: HashSet::new(),
            failed_candidates: HashSet::new(),
            n_children: 0,
        }
    }
}

// ── Candidate generation (entropy-weighted) ────────────────────────

/// Score and generate ranked candidate movesets with entropy-weighted scoring.
///
/// Mirrors the Python `HeuristicMoveGenerator.generate()` + `CandidateScorer`.
#[allow(clippy::too_many_arguments)]
/// Blend hop-count and time-weighted distance.
///
/// Returns `(1 - w_t) * hop_dist + w_t * (time_dist / fastest_lane)`.
/// Falls back to hop-count if time data is unavailable.
fn blended_distance(
    hop_dist: f64,
    from_enc: u64,
    target_enc: u64,
    w_t: f64,
    dist_table: &DistanceTable,
) -> f64 {
    if w_t <= 0.0 {
        return hop_dist;
    }
    let Some(time_d) = dist_table.time_distance(from_enc, target_enc) else {
        return hop_dist;
    };
    let Some(fastest) = dist_table.fastest_lane_us() else {
        return hop_dist;
    };
    let normalized_time_d = time_d / fastest;
    (1.0 - w_t) * hop_dist + w_t * normalized_time_d
}

#[allow(clippy::too_many_arguments)]
pub fn generate_candidates(
    config: &Config,
    entropy: u32,
    params: &EntropyParams,
    ctx: &SearchContext,
    seed: u64,
) -> Vec<(MoveSet, Config, f64)> {
    assert!(
        params.max_movesets_per_group > 0,
        "max_movesets_per_group must be > 0"
    );

    let index = ctx.index;
    let dist_table = ctx.dist_table;
    let targets = ctx.targets;
    let blocked = ctx.blocked;
    let mut rng = if seed != 0 {
        Some(SmallRng::seed_from_u64(
            seed ^ {
                let mut h = std::hash::DefaultHasher::new();
                config.hash(&mut h);
                h.finish()
            } ^ (entropy as u64),
        ))
    } else {
        None
    };
    let e_eff = entropy.min(params.e_max) as f64;

    // Build occupied set.
    let mut occupied = HashSet::with_capacity(blocked.len() + config.len());
    occupied.extend(blocked);
    for (_, loc) in config.iter() {
        occupied.insert(loc.encode());
    }

    // Step 1: identify unresolved qubits.
    let unresolved: Vec<(u32, u64, u64)> = targets
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
        return Vec::new();
    }

    let mut raw_deltas: Vec<(TripletKey, u32, f64, f64, u64, u64)> = Vec::new();
    // Collect (triplet, qid, delta_d, delta_m, lane_enc, dst_enc).

    for &(qid, loc_enc, target_enc) in &unresolved {
        let d_now = match dist_table.distance(loc_enc, target_enc) {
            Some(d) => blended_distance(d as f64, loc_enc, target_enc, params.w_t, dist_table),
            None => continue,
        };
        let m_now = {
            let loc = LocationAddr::decode(loc_enc);
            let mut m = 0.0_f64;
            for &lane in index.outgoing_lanes(loc) {
                let Some((_, dst)) = index.endpoints(&lane) else {
                    continue;
                };
                let dst_e = dst.encode();
                if occupied.contains(&dst_e) {
                    continue;
                }
                let d = dist_table
                    .distance(dst_e, target_enc)
                    .map_or(f64::MAX, |d| {
                        blended_distance(d as f64, dst_e, target_enc, params.w_t, dist_table)
                    });
                if d < f64::MAX {
                    m += 1.0 / (1.0 + d);
                }
            }
            m
        };

        let loc = LocationAddr::decode(loc_enc);
        for &lane in index.outgoing_lanes(loc) {
            let Some((_, dst)) = index.endpoints(&lane) else {
                continue;
            };
            let dst_enc = dst.encode();
            if occupied.contains(&dst_enc) {
                continue;
            }
            let d_after = dist_table
                .distance(dst_enc, target_enc)
                .map_or(f64::MAX, |d| {
                    blended_distance(d as f64, dst_enc, target_enc, params.w_t, dist_table)
                });

            // Mobility accumulation in a single outgoing-lanes pass.
            let mut m_after = 0.0_f64;
            for &next_lane in index.outgoing_lanes(dst) {
                let Some((_, next_dst)) = index.endpoints(&next_lane) else {
                    continue;
                };
                let enc = next_dst.encode();
                if occupied.contains(&enc) {
                    continue;
                }
                // Distance-weighted mobility: closer destinations count more.
                let d_to_target = dist_table.distance(enc, target_enc).map_or(f64::MAX, |d| {
                    blended_distance(d as f64, enc, target_enc, params.w_t, dist_table)
                });
                if d_to_target < f64::MAX {
                    m_after += 1.0 / (1.0 + d_to_target);
                }
            }
            let delta_d = d_now - d_after;
            let delta_m = m_after - m_now;

            let triplet_key = (lane.move_type as u8, lane.bus_id, lane.direction as u8);
            raw_deltas.push((
                triplet_key,
                qid,
                delta_d,
                delta_m,
                lane.encode_u64(),
                dst_enc,
            ));
        }
    }

    if raw_deltas.is_empty() {
        return Vec::new();
    }

    // Normalize deltas.
    let d_ref = raw_deltas
        .iter()
        .map(|(_, _, dd, _, _, _)| dd.abs())
        .fold(1.0_f64, f64::max);
    let m_ref = raw_deltas
        .iter()
        .map(|(_, _, _, dm, _, _)| dm.abs())
        .fold(1.0_f64, f64::max);

    debug_assert!(d_ref >= 1.0, "d_ref must be >= 1.0 (fold seed)");
    debug_assert!(m_ref >= 1.0, "m_ref must be >= 1.0 (fold seed)");

    // Apply entropy-weighted formula and build scored entries.
    let all_scores: Vec<(TripletKey, ScoredEntry)> = raw_deltas
        .into_iter()
        .map(|(key, qid, delta_d, delta_m, lane_enc, dst_enc)| {
            let d_hat = delta_d / d_ref;
            let m_hat = delta_m / m_ref;
            let perturbation = rng.as_mut().map_or(0.0, |r| r.random_range(-0.5..0.5));
            let score = (params.w_d / e_eff) * d_hat + params.w_m * e_eff * m_hat + perturbation;
            (
                key,
                ScoredEntry {
                    qubit_id: qid,
                    score,
                    lane_encoded: lane_enc,
                    dst_encoded: dst_enc,
                },
            )
        })
        .collect();

    // Step 3: keep all positive-scoring entries (Python parity).
    // Save the best overall entry as a singleton fallback in case Phase D produces nothing.
    //
    // Gate alignment with Python `_best_singleton_fallback`:
    //   - Python skips Phase D when no valid_entries exist (no positives), and calls
    //     `_best_singleton_fallback` directly.
    //   - Python also calls `_best_singleton_fallback` after Phase D if scored_candidates
    //     is empty (i.e. all rectangles were rejected).
    //   - Rust mirrors this: when no positives, take only the best entry into Phase D
    //     (equivalent to the Python singleton bypass — a 1-entry AOD grid is always a
    //     singleton). The singleton_fallback saved here covers the has_positive=true case
    //     where Phase D might exceptionally produce nothing.
    let has_positive = all_scores.iter().any(|e| e.1.score > 0.0);

    // Best entry across all scores (used as singleton fallback if Phase D yields nothing).
    // `min_by` on the descending comparator yields the entry with the HIGHEST score
    // (the same "max" that Python computes in `_best_singleton_fallback`).
    let singleton_fallback: Option<(TripletKey, ScoredEntry)> = all_scores
        .iter()
        .min_by(|a, b| cmp_scored_entries(a, b))
        .map(|(key, entry)| {
            (
                *key,
                ScoredEntry {
                    qubit_id: entry.qubit_id,
                    score: entry.score,
                    lane_encoded: entry.lane_encoded,
                    dst_encoded: entry.dst_encoded,
                },
            )
        });

    let selected: Vec<(TripletKey, ScoredEntry)> = if has_positive {
        all_scores.into_iter().filter(|e| e.1.score > 0.0).collect()
    } else {
        // No positive entries: send only the single best entry into Phase D.
        // This matches Python, which skips Phase D entirely and calls
        // `_best_singleton_fallback`. A 1-entry AOD grid is always a singleton,
        // so the result is equivalent. The singleton_fallback gate below provides
        // the safety net for the has_positive=true path.
        all_scores
            .into_iter()
            .min_by(cmp_scored_entries)
            .into_iter()
            .collect()
    };

    // Step 4: group by bus triplet.
    let mut groups: BTreeMap<TripletKey, Vec<ScoredEntry>> = BTreeMap::new();
    for (key, entry) in selected {
        groups.entry(key).or_default().push(entry);
    }

    // Step 5: per group, build AOD-compatible rectangular grids.
    let mut candidates: Vec<(f64, MoveSet, Config)> = Vec::new();

    for ((mt_u8, bus_id, dir_u8), mut qubits) in groups {
        qubits.sort_by(cmp_group_entries);
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

        let grid_ctx = BusGridContext::new(ctx.index, mt, bus_id, None, dir, &occupied);

        // Build entries in the same order as Python's
        // `CandidateScorer.score_rectangle_bus_candidates`: iterate over
        // the bus's lanes (lane-index order across all zones) and keep the
        // ones that appear in `qubits` (the scored, valid-entry set).
        // This matches Python's `valid_entries` dict iteration order, which
        // `greedy_init` consumes verbatim. Sorting by score or by src_enc
        // would produce different AOD clusters — see parity notes in
        // `aod_grid::greedy_init`.
        let mut entry_by_lane: HashMap<u64, &ScoredEntry> = HashMap::new();
        for t in &qubits {
            entry_by_lane.insert(t.lane_encoded, t);
        }
        let mut entries_ordered: Vec<(u64, u64)> = Vec::with_capacity(qubits.len());
        let mut entries_seen: HashSet<u64> = HashSet::new();
        // Iterate zones in numerical order and concatenate lanes —
        // mirrors Python's `ConfigurationTree.lanes_for(zone_id=None)`.
        for zone_id in 0..ctx.index.num_zones() {
            for &lane in ctx.index.lanes_for(mt, bus_id, zone_id, dir) {
                let lane_enc = lane.encode_u64();
                if entry_by_lane.contains_key(&lane_enc)
                    && let Some((src, _)) = ctx.index.endpoints(&lane)
                {
                    let src_enc = src.encode();
                    if entries_seen.insert(src_enc) {
                        entries_ordered.push((src_enc, lane_enc));
                    }
                }
            }
        }

        let grids = grid_ctx.build_aod_grids(&entries_ordered);
        let mut group_candidates: Vec<(f64, MoveSet, Config)> = Vec::new();
        for grid_lanes in grids {
            let mut total_score = 0.0;
            let mut moves: Vec<(u32, LocationAddr)> = Vec::new();

            for &lane_enc in &grid_lanes {
                if let Some(t) = entry_by_lane.get(&lane_enc) {
                    total_score += t.score;
                    moves.push((t.qubit_id, LocationAddr::decode(t.dst_encoded)));
                }
            }

            if moves.is_empty() {
                continue;
            }

            let move_set = MoveSet::from_encoded(grid_lanes);
            let new_config = config.with_moves(&moves);
            if group_candidates
                .iter()
                .any(|(_, existing, _)| *existing == move_set)
            {
                continue;
            }
            group_candidates.push((total_score, move_set, new_config));
        }

        group_candidates.sort_by(cmp_scored_candidates);
        group_candidates.truncate(params.max_movesets_per_group);

        for candidate in group_candidates {
            if candidates
                .iter()
                .any(|(_, existing, _)| existing == &candidate.1)
            {
                continue;
            }
            candidates.push(candidate);
        }
    }

    // Step 6: score each moveset with alpha/beta/gamma + perturbation, sort descending.
    let mut scored: Vec<(f64, MoveSet, Config)> = candidates
        .into_iter()
        .map(|(_raw_score, ms, new_cfg)| {
            let ms_score = score_moveset(config, &new_cfg, &occupied, ctx, params);
            let ms_perturbation = rng.as_mut().map_or(0.0, |r| r.random_range(-0.5..0.5));
            (ms_score + ms_perturbation, ms, new_cfg)
        })
        .collect();
    scored.sort_by(cmp_scored_candidates);

    // Step 7: singleton fallback — mirrors Python `_best_singleton_fallback`.
    // If Phase D produced no candidates, emit the single best-scoring lane as a
    // singleton moveset directly (bypassing AOD rectangle enumeration, exactly
    // as Python does).  Gate fires on `scored.is_empty()` (AFTER Phase D),
    // matching Python's `if not scored_candidates:` guard.
    if scored.is_empty() {
        if let Some((_, fallback)) = singleton_fallback {
            let lane = LaneAddr::decode_u64(fallback.lane_encoded);
            if let Some((_, dst)) = ctx.index.endpoints(&lane) {
                let dst_enc = dst.encode();
                // Destination must still be unoccupied (guarded in Phase A, but
                // double-check for correctness).
                if !occupied.contains(&dst_enc) {
                    let move_set = MoveSet::new([lane]);
                    let new_config =
                        config.with_moves(&[(fallback.qubit_id, LocationAddr::decode(dst_enc))]);
                    return vec![(move_set, new_config, 1.0)];
                }
            }
        }
        return Vec::new();
    }

    scored
        .into_iter()
        .map(|(_, ms, cfg)| (ms, cfg, 1.0))
        .collect()
}

/// Score a moveset: `alpha * distance_progress + beta * arrived + gamma * mobility_gain`.
pub fn score_moveset(
    old_config: &Config,
    new_config: &Config,
    occupied: &HashSet<u64>,
    ctx: &SearchContext,
    params: &EntropyParams,
) -> f64 {
    let targets = ctx.targets;
    let dist_table = ctx.dist_table;
    let blocked = ctx.blocked;
    let index = ctx.index;
    let mut new_occupied: HashSet<u64> = new_config.iter().map(|(_, loc)| loc.encode()).collect();
    new_occupied.extend(blocked);

    let mut distance_progress = 0.0_f64;
    let mut arrived = 0.0_f64;
    let mut mobility_before = 0.0_f64;
    let mut mobility_after = 0.0_f64;

    for &(qid, target_enc) in targets {
        let Some(old_loc) = old_config.location_of(qid) else {
            continue;
        };
        let Some(new_loc) = new_config.location_of(qid) else {
            continue;
        };
        if old_loc == new_loc {
            continue; // didn't move
        }

        let d_before = dist_table
            .distance(old_loc.encode(), target_enc)
            .map_or(0.0, |d| {
                blended_distance(
                    d as f64,
                    old_loc.encode(),
                    target_enc,
                    params.w_t,
                    dist_table,
                )
            });
        let d_after = dist_table
            .distance(new_loc.encode(), target_enc)
            .map_or(0.0, |d| {
                blended_distance(
                    d as f64,
                    new_loc.encode(),
                    target_enc,
                    params.w_t,
                    dist_table,
                )
            });
        // TODO(parity): penalise backward moves instead of clipping. Matches
        // Python `scoring.py:score_moveset`; change both sides together.
        distance_progress += (d_before - d_after).max(0.0);

        if new_loc.encode() == target_enc {
            arrived += 1.0;
        }

        // Distance-weighted mobility: closer destinations count more.
        for &lane in index.outgoing_lanes(old_loc) {
            let Some((_, dst)) = index.endpoints(&lane) else {
                continue;
            };
            let dst_enc = dst.encode();
            if occupied.contains(&dst_enc) {
                continue;
            }
            let d = dist_table
                .distance(dst_enc, target_enc)
                .map_or(f64::MAX, |d| {
                    blended_distance(d as f64, dst_enc, target_enc, params.w_t, dist_table)
                });
            if d < f64::MAX {
                mobility_before += 1.0 / (1.0 + d);
            }
        }
        for &lane in index.outgoing_lanes(new_loc) {
            let Some((_, dst)) = index.endpoints(&lane) else {
                continue;
            };
            let dst_enc = dst.encode();
            if new_occupied.contains(&dst_enc) {
                continue;
            }
            let d = dist_table
                .distance(dst_enc, target_enc)
                .map_or(f64::MAX, |d| {
                    blended_distance(d as f64, dst_enc, target_enc, params.w_t, dist_table)
                });
            if d < f64::MAX {
                mobility_after += 1.0 / (1.0 + d);
            }
        }
    }

    params.alpha * distance_progress
        + params.beta * arrived
        + params.gamma * (mobility_after - mobility_before)
}

// ── BFS path-finding with occupancy ────────────────────────────────

/// Find shortest lane path from `from` to `to`, avoiding `occupied` locations.
pub(crate) fn find_path_occupied(
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

    // BFS with parent-pointer map: O(V) memory instead of O(V×L).
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

    None // no path found
}

// ── Sequential fallback ────────────────────────────────────────────

/// Greedy sequential fallback: move each unresolved qubit along its shortest path.
fn sequential_fallback(
    graph: &mut SearchGraph,
    start: NodeId,
    ctx: &SearchContext,
    goal: &impl Goal,
) -> (Option<NodeId>, u32) {
    let targets = ctx.targets;
    let index = ctx.index;
    let blocked = ctx.blocked;
    let mut current = start;
    let mut nodes_expanded: u32 = 0;

    // Identify unresolved qubits.
    let config = graph.config(current).clone();
    let unresolved: Vec<(u32, u64)> = targets
        .iter()
        .filter_map(|&(qid, target_enc)| {
            let loc = config.location_of(qid)?;
            if loc.encode() == target_enc {
                None
            } else {
                Some((qid, target_enc))
            }
        })
        .collect();

    for (qid, target_enc) in unresolved {
        let cfg = graph.config(current).clone();
        let Some(current_loc) = cfg.location_of(qid) else {
            continue;
        };
        let target_loc = LocationAddr::decode(target_enc);

        if current_loc == target_loc {
            continue;
        }

        // Build occupied set: all other qubits + blocked.
        let mut occ = blocked.clone();
        for (other_qid, loc) in cfg.iter() {
            if other_qid != qid {
                occ.insert(loc.encode());
            }
        }

        let Some(path) = find_path_occupied(current_loc, target_loc, &occ, index) else {
            return (None, nodes_expanded);
        };

        for lane in path {
            let Some((src, dst)) = index.endpoints(&lane) else {
                return (None, nodes_expanded);
            };
            let move_set = MoveSet::new([lane]);
            let cur_config = graph.config(current).clone();

            // Find which qubit is at src.
            let Some(moving_qid) = cur_config.qubit_at(src) else {
                return (None, nodes_expanded);
            };

            let new_config = cur_config.with_moves(&[(moving_qid, dst)]);
            let new_g = graph.g_score(current) + 1.0;
            let (child_id, _) = graph.insert_branch_local(current, move_set, new_config, new_g);
            nodes_expanded += 1;
            current = child_id;
        }
    }

    if goal.is_goal(graph.config(current)) {
        (Some(current), nodes_expanded)
    } else {
        (None, nodes_expanded)
    }
}

// ── Resume buffer ──────────────────────────────────────────────────

/// A bounded buffer of scored resume candidates.
///
/// Mirrors Python's `_resume_buffer` / `_buffer_insert` / `_buffer_pop_best`
/// in `EntropyGuidedSearch`.  The sort key is `(score, depth, order)` —
/// all ascending — so `max` picks the best (highest score, then deepest node,
/// then most-recently inserted on tie).  Eviction: a new entry replaces the
/// worst existing entry **only if strictly better** (Python's `<=` guard means
/// "do not insert on tie").
struct ResumeEntry {
    node: NodeId,
    /// Incoming moveset score for this node.
    score: f64,
    /// Graph depth of this node.
    depth: u32,
    /// Monotonically increasing insertion counter — later insertion wins ties.
    order: u64,
}

impl ResumeEntry {
    /// Comparable key: `(score, depth, order)`.
    ///
    /// All fields ascending: higher score > lower score, deeper > shallower,
    /// larger order > smaller order.  `f64` is wrapped in `OrderedFloat` via
    /// a manual total order: NaN sorts last (treated as +∞ for eviction
    /// purposes, which is safe because scores from `score_moveset` are finite).
    fn sort_key(&self) -> (i64, u32, u64) {
        // Map f64 → i64 bits with a total order: negative values sort low,
        // positive sort high, NaN maps to i64::MAX.
        let bits = self.score.to_bits() as i64;
        let ordered = if self.score.is_nan() {
            i64::MAX
        } else if self.score < 0.0 {
            // Flip sign bit so negative floats sort below zero.
            bits ^ i64::MIN
        } else {
            bits
        };
        (ordered, self.depth, self.order)
    }
}

struct ResumeBuffer {
    entries: Vec<ResumeEntry>,
    capacity: usize,
    insert_counter: u64,
}

impl ResumeBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            capacity,
            insert_counter: 0,
        }
    }

    /// Insert `node` with the given `score` and `depth`.
    ///
    /// If the buffer is not full, append unconditionally.
    /// If full, evict the entry with the lowest sort key — but ONLY if the new
    /// entry is STRICTLY better (mirrors Python's `<= → return` guard).
    fn insert(&mut self, node: NodeId, score: f64, depth: u32) {
        if self.capacity == 0 {
            return;
        }

        let order = self.insert_counter;
        self.insert_counter += 1;

        let candidate = ResumeEntry {
            node,
            score,
            depth,
            order,
        };

        if self.entries.len() < self.capacity {
            self.entries.push(candidate);
            return;
        }

        // Find the worst entry (minimum sort key).
        let worst_idx = self
            .entries
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.sort_key().cmp(&b.sort_key()))
            .map(|(i, _)| i)
            .expect("entries non-empty");

        // Only replace if new candidate is STRICTLY better than the worst.
        // On a tie (candidate key == worst key), keep the old entry (Python parity).
        if candidate.sort_key() > self.entries[worst_idx].sort_key() {
            self.entries[worst_idx] = candidate;
        }
    }

    /// Remove any entry pointing to `node`.
    fn discard(&mut self, node: NodeId) {
        self.entries.retain(|e| e.node != node);
    }

    /// Pop and return the best entry (highest sort key), or `None` if empty.
    fn pop_best(&mut self) -> Option<NodeId> {
        if self.entries.is_empty() {
            return None;
        }
        let best_idx = self
            .entries
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.sort_key().cmp(&b.sort_key()))
            .map(|(i, _)| i)
            .expect("entries non-empty");
        Some(self.entries.swap_remove(best_idx).node)
    }
}

// ── Main search loop ───────────────────────────────────────────────

/// Run entropy-guided search.
///
/// This is a single-path DFS with entropy-based backtracking, NOT a
/// standard frontier-based search. See module docs for algorithm details.
pub fn entropy_search(
    root: Config,
    goal: &impl Goal,
    params: &EntropyParams,
    ctx: &SearchContext,
    max_expansions: Option<u32>,
    max_depth: Option<u32>,
    seed: u64,
) -> SearchResult {
    // Early check.
    if goal.is_goal(&root) {
        let graph = SearchGraph::new(root);
        return SearchResult {
            goal: Some(graph.root()),
            nodes_expanded: 0,
            max_depth_reached: 0,
            graph,
        };
    }

    let mut graph = SearchGraph::new(root);
    let root_id = graph.root();
    let mut entropy_map: HashMap<NodeId, EntropyState> = HashMap::new();
    let mut current = root_id;
    let mut nodes_expanded: u32 = 0;
    let mut max_depth_seen: u32 = 0;
    let mut found_goals: Vec<NodeId> = Vec::new();
    let mut hit_max_depth = false;

    // Resume buffer: mirrors Python's `_resume_buffer`.  Used only when
    // `max_goal_candidates > 1` to track the best non-goal candidates to
    // resume from after finding a goal, matching Python's `_buffer_insert` /
    // `_buffer_pop_best` control flow.
    let use_resume_buffer = params.max_goal_candidates > 1;
    let mut resume_buffer = ResumeBuffer::new(params.max_goal_candidates);

    // Safety cap: hard iteration limit prevents infinite loops when
    // max_expansions is None and the search gets stuck in reversion cycles.
    let hard_limit = max_expansions.unwrap_or(ctx.index.num_locations() as u32 * 10);
    let mut iterations: u32 = 0;

    loop {
        iterations += 1;
        if nodes_expanded >= hard_limit || iterations >= hard_limit * 2 {
            break;
        }

        let es = entropy_map.entry(current).or_default();

        // Force entropy at depth limit.
        if let Some(max_d) = max_depth
            && graph.depth(current) >= max_d
        {
            hit_max_depth = true;
            es.entropy = params.e_max;
        }

        // REVERSION: entropy too high.
        if es.entropy >= params.e_max {
            let mut ancestor = current;
            for _ in 0..params.reversion_steps {
                if let Some(parent) = graph.parent(ancestor) {
                    ancestor = parent;
                } else {
                    break;
                }
            }

            let ancestor_es = entropy_map.entry(ancestor).or_default();
            if ancestor == root_id && ancestor_es.entropy >= params.e_max {
                // Root saturated: exit loop.  Sequential fallback is only invoked
                // below if `max_expansions` was reached or `max_depth` was hit —
                // matches Python's `EntropyGuidedSearch.run` post-loop guard.
                break;
            }

            ancestor_es.entropy += params.delta_e;
            current = ancestor;
            continue;
        }

        // CANDIDATE SELECTION.
        let candidate = get_next_candidate(&mut entropy_map, current, params, ctx, &graph, seed);

        let Some((move_set, new_config, cost)) = candidate else {
            // No candidates available — bump entropy.
            entropy_map.entry(current).or_default().entropy += params.delta_e;
            continue;
        };

        // Record as tried.
        let es = entropy_map.entry(current).or_default();
        let move_key = move_set.encoded_lanes().to_vec();
        es.tried_moves.insert(move_key.clone());
        es.candidates_tried += 1;

        // Compute moveset score before consuming new_config (used for resume buffer).
        // Mirrors Python `_state_move_score` → `CandidateScorer.score_moveset`,
        // whose `occupied` is `node.occupied_locations | tree.blocked_locations`
        // — i.e. qubit locations PLUS the tree's blocked set.  Omitting blocked
        // here yields a systematically higher mobility_before and therefore a
        // lower score, which changes the resume-buffer pop order on trotter-like
        // workloads.
        let child_score = if use_resume_buffer {
            let old_config = graph.config(current);
            let mut occupied: HashSet<u64> =
                old_config.iter().map(|(_, loc)| loc.encode()).collect();
            occupied.extend(ctx.blocked);
            score_moveset(old_config, &new_config, &occupied, ctx, params)
        } else {
            0.0
        };

        // Insert into graph using branch-local (ancestor-only) transposition
        // semantics, mirroring Python's `ConfigurationTree._ancestor_with_config_key`.
        // Using the global `seen` transposition table here would reject configs
        // reached via sibling branches — e.g. after `_buffer_pop_best` jumps to
        // a different resume node — which Python treats as valid new children.
        let new_g = graph.g_score(current) + cost;
        let (child_id, is_new) = graph.insert_branch_local(current, move_set, new_config, new_g);

        if !is_new {
            // Transposition: config seen at equal or better cost.
            let es = entropy_map.entry(current).or_default();
            es.failed_candidates.insert(move_key.clone());
            es.entropy += params.delta_e;
            continue;
        }

        // Track that a new child was created from this node.
        entropy_map.entry(current).or_default().n_children += 1;
        nodes_expanded += 1;
        let child_depth = graph.depth(child_id);
        max_depth_seen = max_depth_seen.max(child_depth);

        // Buffer insert: mirrors Python's `if max_goal_candidates > 1: _buffer_insert(child)`.
        if use_resume_buffer {
            resume_buffer.insert(child_id, child_score, child_depth);
        }

        if goal.is_goal(graph.config(child_id)) {
            found_goals.push(child_id);
            if found_goals.len() >= params.max_goal_candidates {
                break;
            }
            if use_resume_buffer {
                // Discard the goal from the buffer, then pop the best remaining
                // candidate.  Fall back to root if buffer is empty.
                // Mirrors Python: `_buffer_discard(child); current = _buffer_pop_best() or root`.
                resume_buffer.discard(child_id);
                current = resume_buffer.pop_best().unwrap_or(root_id);
            } else {
                // Single-goal mode: walk up to first branching ancestor.
                current = cutoff_ancestor(child_id, &graph, &entropy_map);
            }
            continue;
        }

        current = child_id; // descend
    }

    // Post-loop sequential fallback — matches Python's guard:
    //   if max_expansions reached OR max_depth hit.
    // Python runs fallback from root, then combines with found_goals.
    let exhausted_budget = max_expansions.is_some_and(|max| nodes_expanded >= max);
    if exhausted_budget || hit_max_depth {
        let (goal_id, fb_expanded) = sequential_fallback(&mut graph, root_id, ctx, goal);
        nodes_expanded += fb_expanded;
        if let Some(gid) = goal_id {
            found_goals.push(gid);
        }
    }

    // Return shallowest goal, tie-break by lexicographic move program for
    // Python parity (`_goal_sort_key = (depth, encoded_program)`).
    let best = found_goals.into_iter().min_by(|&a, &b| {
        let da = graph.depth(a);
        let db = graph.depth(b);
        match da.cmp(&db) {
            std::cmp::Ordering::Equal => {
                let pa = graph.reconstruct_path(a);
                let pb = graph.reconstruct_path(b);
                let ea: Vec<Vec<u64>> = pa
                    .iter()
                    .map(|ms| {
                        let mut v = ms.encoded_lanes().to_vec();
                        v.sort();
                        v
                    })
                    .collect();
                let eb: Vec<Vec<u64>> = pb
                    .iter()
                    .map(|ms| {
                        let mut v = ms.encoded_lanes().to_vec();
                        v.sort();
                        v
                    })
                    .collect();
                ea.cmp(&eb)
            }
            other => other,
        }
    });
    SearchResult {
        goal: best,
        nodes_expanded,
        max_depth_reached: max_depth_seen,
        graph,
    }
}

/// Populate `es.candidate_cache` from a fresh `generate_candidates` call and
/// reset `candidates_tried` to 0.  Mirrors Python's `_refresh_candidate_cache`.
fn fill_candidate_cache(
    es: &mut EntropyState,
    config: &Config,
    params: &EntropyParams,
    ctx: &SearchContext,
    seed: u64,
) {
    es.candidate_cache = generate_candidates(config, es.entropy, params, ctx, seed);
    es.candidates_tried = 0;
}

/// Scan forward from `es.candidates_tried`, skipping entries already in
/// `tried_moves` or `failed_candidates`.  Returns the first valid candidate
/// without consuming it (i.e. `candidates_tried` is left pointing at the
/// returned entry).  Mirrors the Python inner while-loop in `_get_next_candidate`.
fn peek_cache(es: &mut EntropyState) -> Option<(MoveSet, Config, f64)> {
    while es.candidates_tried < es.candidate_cache.len() {
        let (ref ms, ref cfg, cost) = es.candidate_cache[es.candidates_tried];
        let move_key = ms.encoded_lanes().to_vec();
        if !es.tried_moves.contains(&move_key) && !es.failed_candidates.contains(&move_key) {
            return Some((ms.clone(), cfg.clone(), cost));
        }
        es.candidates_tried += 1;
    }
    None
}

/// Get the next untried candidate from the cache, regenerating if needed.
///
/// Mirrors Python's `_get_next_candidate` two-pass shape:
///   1. If the cache has been consumed up to `max_candidates` or is empty,
///      refill it (first refresh).
///   2. Scan forward for the first valid entry.
///   3. If that scan came up empty, refill once more (second refresh, Python
///      parity safety net) and scan again.
///
/// In the current Rust implementation the second refresh is a no-op because
/// `generate_candidates` is pure — re-calling it with unchanged state returns
/// the same result.  The second pass is kept for shape alignment with Python,
/// which uses a lazy iterator and may yield fresh candidates on the second
/// `_refresh_candidate_cache` call.
fn get_next_candidate(
    entropy_map: &mut HashMap<NodeId, EntropyState>,
    node_id: NodeId,
    params: &EntropyParams,
    ctx: &SearchContext,
    graph: &SearchGraph,
    seed: u64,
) -> Option<(MoveSet, Config, f64)> {
    let config = graph.config(node_id).clone();
    let es = entropy_map.entry(node_id).or_default();

    // First pass: refill if the cache has been consumed up to max_candidates or
    // is empty.
    if es.candidates_tried >= params.max_candidates || es.candidate_cache.is_empty() {
        fill_candidate_cache(es, &config, params, ctx, seed);
    }
    if let Some(c) = peek_cache(es) {
        return Some(c);
    }

    // Second pass: Python-parity safety net.  If the first pass's cache came
    // back fully filtered, refill once more and try again.  In the current Rust
    // impl this is a no-op because generate_candidates is pure and the filter
    // state is unchanged — kept for control-flow shape alignment with Python.
    fill_candidate_cache(es, &config, params, ctx, seed);
    peek_cache(es)
}

/// Walk up from goal to find the first branching ancestor.
///
/// Matches Python's `_cutoff_ancestor`: walks up and stops at the first
/// node whose parent has more than one child (branch point).
fn cutoff_ancestor(
    goal_id: NodeId,
    graph: &SearchGraph,
    entropy_map: &HashMap<NodeId, EntropyState>,
) -> NodeId {
    let mut ancestor = goal_id;
    while let Some(parent) = graph.parent(ancestor) {
        let parent_children = entropy_map.get(&parent).map_or(0, |es| es.n_children);
        if parent_children > 1 {
            return ancestor;
        }
        ancestor = parent;
    }
    ancestor
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{example_arch_json, loc};

    fn make_index() -> LaneIndex {
        let spec: bloqade_lanes_bytecode_core::arch::types::ArchSpec =
            serde_json::from_str(example_arch_json()).unwrap();
        LaneIndex::new(spec)
    }

    /// Helper: run entropy search with minimal setup.
    fn run_entropy(
        initial: impl IntoIterator<Item = (u32, LocationAddr)>,
        target: impl IntoIterator<Item = (u32, LocationAddr)>,
        max_expansions: Option<u32>,
    ) -> SearchResult {
        let index = make_index();
        let root = Config::new(initial).unwrap();
        let target_pairs: Vec<(u32, LocationAddr)> = target.into_iter().collect();
        let target_encoded: Vec<(u32, u64)> =
            target_pairs.iter().map(|&(q, l)| (q, l.encode())).collect();
        let target_locs: Vec<u64> = target_encoded.iter().map(|&(_, l)| l).collect();
        let dist_table = DistanceTable::new(&target_locs, &index);
        let blocked = HashSet::new();
        let goal = crate::goals::AllAtTarget::new(&target_encoded);
        let ctx = SearchContext {
            index: &index,
            dist_table: &dist_table,
            blocked: &blocked,
            targets: &target_encoded,
        };
        entropy_search(
            root,
            &goal,
            &EntropyParams::default(),
            &ctx,
            max_expansions,
            None,
            0,
        )
    }

    #[test]
    fn solve_simple_one_step() {
        let r = run_entropy([(0, loc(0, 0))], [(0, loc(0, 5))], Some(100));
        assert!(r.goal.is_some());
        assert_eq!(
            r.graph.config(r.goal.unwrap()).location_of(0),
            Some(loc(0, 5))
        );
    }

    #[test]
    fn solve_already_at_target() {
        let r = run_entropy([(0, loc(0, 5))], [(0, loc(0, 5))], Some(100));
        assert!(r.goal.is_some());
        assert_eq!(r.nodes_expanded, 0);
    }

    #[test]
    fn solve_cross_word() {
        let r = run_entropy([(0, loc(0, 5))], [(0, loc(1, 5))], Some(100));
        assert!(r.goal.is_some());
        assert_eq!(
            r.graph.config(r.goal.unwrap()).location_of(0),
            Some(loc(1, 5))
        );
    }

    #[test]
    fn solve_multi_step() {
        let r = run_entropy([(0, loc(0, 0))], [(0, loc(1, 5))], Some(1000));
        assert!(r.goal.is_some());
        assert!(r.solution_path().unwrap().len() >= 2);
    }

    #[test]
    fn budget_exceeded_returns_no_goal() {
        let r = run_entropy([(0, loc(0, 0))], [(0, loc(99, 99))], Some(10));
        assert!(r.goal.is_none());
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
    fn scored_entry_tie_break_is_deterministic() {
        let mut entries = vec![
            (
                (1, 2, 1),
                ScoredEntry {
                    qubit_id: 8,
                    score: 3.0,
                    lane_encoded: 19,
                    dst_encoded: 40,
                },
            ),
            (
                (1, 1, 1),
                ScoredEntry {
                    qubit_id: 4,
                    score: 3.0,
                    lane_encoded: 12,
                    dst_encoded: 40,
                },
            ),
            (
                (1, 1, 1),
                ScoredEntry {
                    qubit_id: 4,
                    score: 3.0,
                    lane_encoded: 10,
                    dst_encoded: 40,
                },
            ),
        ];

        entries.sort_by(cmp_scored_entries);

        assert_eq!(entries[0].0, (1, 1, 1));
        assert_eq!(entries[0].1.lane_encoded, 10);
        assert_eq!(entries[1].0, (1, 1, 1));
        assert_eq!(entries[1].1.lane_encoded, 12);
        assert_eq!(entries[2].0, (1, 2, 1));
    }

    #[test]
    fn generate_candidates_seed_zero_tie_fallback_is_stable() {
        let index = make_index();
        let config = Config::new([(0, loc(0, 0))]).unwrap();
        let target_encoded = vec![(0u32, loc(0, 5).encode())];
        let target_locs: Vec<u64> = target_encoded.iter().map(|&(_, enc)| enc).collect();
        let dist_table = DistanceTable::new(&target_locs, &index);
        let blocked = HashSet::new();
        let ctx = SearchContext {
            index: &index,
            dist_table: &dist_table,
            blocked: &blocked,
            targets: &target_encoded,
        };
        let params = EntropyParams {
            w_d: 0.0,
            w_m: 0.0,
            max_movesets_per_group: 8,
            ..EntropyParams::default()
        };

        let out1 = generate_candidates(&config, 1, &params, &ctx, 0);
        let out2 = generate_candidates(&config, 1, &params, &ctx, 0);

        assert!(!out1.is_empty());
        assert_eq!(out1.len(), out2.len());
        for ((ms_a, cfg_a, _), (ms_b, cfg_b, _)) in out1.iter().zip(out2.iter()) {
            assert_eq!(ms_a, ms_b);
            assert_eq!(cfg_a.as_entries(), cfg_b.as_entries());
        }
    }

    #[test]
    #[should_panic(expected = "max_movesets_per_group must be > 0")]
    fn generate_candidates_rejects_zero_movesets_per_group() {
        let index = make_index();
        let config = Config::new([(0, loc(0, 0))]).unwrap();
        let target_encoded = vec![(0u32, loc(0, 5).encode())];
        let target_locs: Vec<u64> = target_encoded.iter().map(|&(_, enc)| enc).collect();
        let dist_table = DistanceTable::new(&target_locs, &index);
        let blocked = HashSet::new();
        let ctx = SearchContext {
            index: &index,
            dist_table: &dist_table,
            blocked: &blocked,
            targets: &target_encoded,
        };
        let params = EntropyParams {
            max_movesets_per_group: 0,
            ..EntropyParams::default()
        };

        let _ = generate_candidates(&config, 1, &params, &ctx, 0);
    }

    #[test]
    fn generate_candidates_emit_aod_rectangles() {
        let index = make_index();
        let config = Config::new([(0, loc(0, 0)), (1, loc(0, 1))]).unwrap();
        let target_encoded = vec![(0u32, loc(0, 5).encode()), (1u32, loc(0, 6).encode())];
        let target_locs: Vec<u64> = target_encoded.iter().map(|&(_, enc)| enc).collect();
        let dist_table = DistanceTable::new(&target_locs, &index);
        let blocked = HashSet::new();
        let ctx = SearchContext {
            index: &index,
            dist_table: &dist_table,
            blocked: &blocked,
            targets: &target_encoded,
        };
        let params = EntropyParams {
            max_movesets_per_group: 4,
            ..EntropyParams::default()
        };

        let out = generate_candidates(&config, 1, &params, &ctx, 0);
        assert!(!out.is_empty());

        let mut occupied = HashSet::new();
        for (_, loc) in config.iter() {
            occupied.insert(loc.encode());
        }

        for (moveset, _, _) in out {
            let lanes = moveset.decode();
            if lanes.is_empty() {
                continue;
            }
            let first = lanes[0];
            let grid_ctx = BusGridContext::new(
                &index,
                first.move_type,
                first.bus_id,
                None,
                first.direction,
                &occupied,
            );

            let mut entries: Vec<(u64, u64)> = Vec::new();
            for lane in &lanes {
                assert_eq!(lane.move_type, first.move_type);
                assert_eq!(lane.bus_id, first.bus_id);
                assert_eq!(lane.direction, first.direction);
                let (src, _) = index.endpoints(lane).expect("lane endpoints must exist");
                entries.push((src.encode(), lane.encode_u64()));
            }

            let grids = grid_ctx.build_aod_grids(&entries);
            let expected = moveset.encoded_lanes().to_vec();
            assert!(
                grids.into_iter().any(|grid| {
                    let candidate = MoveSet::from_encoded(grid);
                    candidate.encoded_lanes() == expected.as_slice()
                }),
                "moveset must be directly reproducible via AOD grid builder"
            );
        }
    }

    /// Verify two-pass shape of `get_next_candidate`.
    ///
    /// Directly pre-populates `candidate_cache` to avoid relying on
    /// `generate_candidates` producing a specific count (the example arch
    /// yields only 1 candidate per single-qubit move).
    ///
    /// **Scenario A – first pass skips a failed entry via `peek_cache`:**
    /// Two synthetic candidates are placed in the cache.  The first is
    /// pre-seeded into `failed_candidates`.  `peek_cache` scans past it and
    /// returns the second.  The second refresh is never reached.
    ///
    /// **Scenario B – all candidates failed → second refresh is a no-op:**
    /// Both cache entries are marked failed.  Both `peek_cache` calls return
    /// `None`.  Because `generate_candidates` is pure, the second
    /// `fill_candidate_cache` produces the same set → still `None`.  This
    /// confirms the second-refresh path is **vacuous** in the current Rust
    /// implementation.
    ///
    /// The test pins the two-pass control-flow shape for Python parity.
    #[test]
    fn get_next_candidate_calls_fill_twice_when_first_pass_empty() {
        use crate::test_utils::dummy_lane;

        let index = make_index();
        // Single qubit: minimal setup; cache will be pre-populated manually.
        let config = Config::new([(0, loc(0, 0))]).unwrap();
        let target_encoded = vec![(0u32, loc(0, 5).encode())];
        let target_locs: Vec<u64> = target_encoded.iter().map(|&(_, enc)| enc).collect();
        let dist_table = DistanceTable::new(&target_locs, &index);
        let blocked = HashSet::new();
        let ctx = SearchContext {
            index: &index,
            dist_table: &dist_table,
            blocked: &blocked,
            targets: &target_encoded,
        };
        // Large max_candidates so the "candidates_tried >= max_candidates"
        // trigger does NOT fire — we rely on cache-empty trigger only.
        let params = EntropyParams {
            max_candidates: 100,
            ..EntropyParams::default()
        };

        let graph = SearchGraph::new(config.clone());
        let root = graph.root();

        // Build two synthetic cache entries using distinct dummy lanes so their
        // encoded_lanes keys are different.
        let ms_a = MoveSet::new([dummy_lane(0)]);
        let ms_b = MoveSet::new([dummy_lane(1)]);
        let key_a = ms_a.encoded_lanes().to_vec();
        let key_b = ms_b.encoded_lanes().to_vec();
        let entry_a = (ms_a.clone(), config.clone(), 1.0);
        let entry_b = (ms_b.clone(), config.clone(), 1.0);

        // ── Scenario A: first entry failed → second is returned ──
        let mut entropy_map: HashMap<NodeId, EntropyState> = HashMap::new();
        {
            let es = entropy_map.entry(root).or_default();
            // Pre-populate the cache so fill_candidate_cache need not run.
            // (Cache is NOT empty, and candidates_tried=0 < max_candidates=100,
            // so the initial trigger will not fire — but we still test
            // peek_cache's forward-scan behavior.)
            // For the trigger to fire we clear the cache, which is the
            // "cache is empty" branch.  Leave it empty here so the first
            // fill_candidate_cache call populates it from generate_candidates.
            // Instead, manually seed failed_candidates with whatever key
            // generate_candidates would produce first, which we don't know.
            // Use an artificial pre-populated cache instead:
            es.candidate_cache = vec![entry_a.clone(), entry_b.clone()];
            es.candidates_tried = 0; // cache is NOT empty; trigger skipped
            es.failed_candidates.insert(key_a.clone());
        }
        // Cache is not empty AND candidates_tried < max_candidates → no refill.
        // peek_cache skips key_a (failed) and returns key_b.
        let got = get_next_candidate(&mut entropy_map, root, &params, &ctx, &graph, 0);
        assert!(
            got.is_some(),
            "scenario A: peek_cache should skip the failed entry and return the second"
        );
        assert_ne!(
            got.unwrap().0.encoded_lanes().to_vec(),
            key_a,
            "returned candidate must not be the one marked failed"
        );

        // ── Scenario B: all candidates failed → second refresh is a no-op ──
        {
            let es = entropy_map.entry(root).or_default();
            es.candidate_cache.clear(); // force the "cache is empty" trigger
            es.candidates_tried = 0;
            es.failed_candidates.clear();
            // Mark whatever generate_candidates returns as failed.
            let generated = generate_candidates(&config, 1, &params, &ctx, 0);
            for (ms, _, _) in &generated {
                es.failed_candidates.insert(ms.encoded_lanes().to_vec());
            }
            // Also mark our synthetic keys so even if synthetic entries sneak
            // in they're covered.
            es.failed_candidates.insert(key_a.clone());
            es.failed_candidates.insert(key_b.clone());
        }
        let got2 = get_next_candidate(&mut entropy_map, root, &params, &ctx, &graph, 0);
        assert!(
            got2.is_none(),
            "scenario B: pure generate_candidates returns the same set on second refresh → \
             both passes return None (second refresh is vacuous)"
        );
    }

    /// Pin Python's `_buffer_insert` tie-break semantics in `ResumeBuffer`.
    ///
    /// Python's eviction guard:
    ///   `if self._resume_sort_key(candidate) <= self._resume_sort_key(lowest): return`
    ///
    /// This means: only evict when the new candidate is **strictly better**.
    /// On a tie (same score, depth, and order would differ but score+depth are equal),
    /// the old entry is kept because the new entry's insertion order is always higher,
    /// so the new key (`(score, depth, new_order)`) is NEVER ≤ the old key's
    /// (`(score, depth, old_order)`) — i.e. on equal score+depth, the NEWER entry
    /// always wins.
    ///
    /// More precisely, with capacity=1:
    /// - Insert A (order=0): buffer = [A]
    /// - Insert B with same (score, depth) (order=1): B's key (score, depth, 1) >
    ///   A's key (score, depth, 0) → B evicts A → buffer = [B]
    /// - Pop best returns B.
    ///
    /// This test verifies that Rust matches Python's "later insertion wins on score+depth tie".
    #[test]
    fn resume_buffer_tie_matches_python() {
        // Python _resume_sort_key = (score, depth, order) ascending.
        // max() picks best; on equal score+depth, later order (larger) wins.
        // So with capacity 1: inserting B after A with same score+depth → B wins.
        let node_a = NodeId(0);
        let node_b = NodeId(1);
        let score = 5.0_f64;
        let depth = 3_u32;

        let mut buf = ResumeBuffer::new(1);

        // Insert A first (order=0).
        buf.insert(node_a, score, depth);
        assert_eq!(buf.entries.len(), 1);
        assert_eq!(
            buf.entries[0].node, node_a,
            "A should be in buffer after first insert"
        );

        // Insert B with same score and depth (order=1).
        // B's sort key = (score, depth, 1) > A's sort key = (score, depth, 0).
        // Python: candidate_key > lowest_key → evict A, insert B.
        buf.insert(node_b, score, depth);
        assert_eq!(buf.entries.len(), 1);
        assert_eq!(
            buf.entries[0].node, node_b,
            "B should evict A on equal score+depth because B has higher insertion order (later wins)"
        );

        // Pop best returns B.
        let popped = buf.pop_best();
        assert_eq!(
            popped,
            Some(node_b),
            "pop_best should return B (later insertion)"
        );
        assert!(buf.entries.is_empty());
    }

    /// Verify singleton-fallback gate is placed AFTER Phase D, matching Python parity.
    ///
    /// Python's `_best_singleton_fallback` fires under `if not scored_candidates` —
    /// i.e., when Phase D (rectangle enumeration) produces nothing.  Rust mirrors
    /// this with a `scored.is_empty()` gate after Step 6, and separately saves the
    /// best overall entry as `singleton_fallback` before filtering.
    ///
    /// Scenario A — all-zero scores (no-positive branch), Phase D still succeeds:
    ///   With `w_d = 0, w_m = 0`, every per-entry score is 0.0 (not positive).
    ///   `has_positive = false` → `selected` = single best entry → Phase D produces
    ///   a singleton rectangle.  `scored` is non-empty → singleton_fallback NOT
    ///   invoked.  Output must be non-empty.
    ///
    /// Scenario B — already at target, no unresolved qubits:
    ///   Phase A returns early with `unresolved.is_empty()`.  Neither Phase D nor
    ///   the singleton fallback fires.  Output must be empty.
    #[test]
    fn singleton_fallback_gate_fires_after_phase_d() {
        let index = make_index();

        // Scenario A: all-zero scores — no-positive path, Phase D still produces a candidate.
        {
            let config = Config::new([(0, loc(0, 0))]).unwrap();
            let target_encoded = vec![(0u32, loc(0, 5).encode())];
            let target_locs: Vec<u64> = target_encoded.iter().map(|&(_, enc)| enc).collect();
            let dist_table = DistanceTable::new(&target_locs, &index);
            let blocked = HashSet::new();
            let ctx = SearchContext {
                index: &index,
                dist_table: &dist_table,
                blocked: &blocked,
                targets: &target_encoded,
            };
            // w_d = 0, w_m = 0 → all scores = 0.0, has_positive = false.
            // `selected` = best single entry → Phase D produces singleton → non-empty output.
            // singleton_fallback gate (scored.is_empty()) does NOT fire here.
            let params = EntropyParams {
                w_d: 0.0,
                w_m: 0.0,
                max_movesets_per_group: 4,
                ..EntropyParams::default()
            };
            let out = generate_candidates(&config, 1, &params, &ctx, 0);
            assert!(
                !out.is_empty(),
                "no-positive path must still yield candidates when Phase D succeeds"
            );
            // Must be a singleton moveset (single best entry → 1-entry Phase D).
            assert_eq!(
                out[0].0.encoded_lanes().len(),
                1,
                "no-positive path: Phase D must produce a singleton moveset from the single best entry"
            );
        }

        // Scenario B: qubit already at target (Phase A returns empty) → no candidates.
        // Neither Phase D nor the singleton fallback fires.
        {
            let config = Config::new([(0, loc(0, 5))]).unwrap();
            let target_encoded = vec![(0u32, loc(0, 5).encode())];
            let target_locs: Vec<u64> = target_encoded.iter().map(|&(_, enc)| enc).collect();
            let dist_table = DistanceTable::new(&target_locs, &index);
            let blocked = HashSet::new();
            let ctx = SearchContext {
                index: &index,
                dist_table: &dist_table,
                blocked: &blocked,
                targets: &target_encoded,
            };
            let params = EntropyParams::default();
            let out = generate_candidates(&config, 1, &params, &ctx, 0);
            assert!(
                out.is_empty(),
                "already-at-target qubit must yield no candidates (Phase A gate)"
            );
        }
    }

    /// Verify that a strictly worse candidate does NOT evict a better one.
    #[test]
    fn resume_buffer_worse_does_not_evict() {
        let node_a = NodeId(0);
        let node_b = NodeId(1);

        let mut buf = ResumeBuffer::new(1);
        buf.insert(node_a, 10.0, 5); // A: higher score
        buf.insert(node_b, 5.0, 5); // B: lower score → should not evict A

        assert_eq!(buf.entries.len(), 1);
        assert_eq!(
            buf.entries[0].node, node_a,
            "A (better score) should remain"
        );
    }

    /// Verify that a strictly better candidate DOES evict a worse one.
    #[test]
    fn resume_buffer_better_evicts_worse() {
        let node_a = NodeId(0);
        let node_b = NodeId(1);

        let mut buf = ResumeBuffer::new(1);
        buf.insert(node_a, 5.0, 3); // A: lower score
        buf.insert(node_b, 10.0, 3); // B: higher score → should evict A

        assert_eq!(buf.entries.len(), 1);
        assert_eq!(
            buf.entries[0].node, node_b,
            "B (better score) should evict A"
        );
    }
}
