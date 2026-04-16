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

use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};

use bloqade_lanes_bytecode_core::arch::addr::{LaneAddr, LocationAddr};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::astar::SearchResult;
use crate::config::Config;
use crate::context::SearchContext;
use crate::graph::{MoveSet, NodeId, SearchGraph};
use crate::heuristic::DistanceTable;
use crate::lane_index::LaneIndex;
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
    pub top_c: usize,
    pub max_candidates: usize,
    pub reversion_steps: u32,
    pub delta_e: u32,
    pub e_max: u32,
    pub max_goal_candidates: usize,
    // Expander settings.
    pub max_movesets_per_group: usize,
    /// Enable 2-step lookahead scoring.
    pub lookahead: bool,
    /// Time-distance blend weight (0.0 = hop-count only, 1.0 = time only).
    pub w_t: f64,
}

impl Default for EntropyParams {
    fn default() -> Self {
        Self {
            w_d: 1.0,
            w_m: 0.3,
            alpha: 100.0,
            beta: 2.0,
            gamma: 0.5,
            top_c: 3,
            max_candidates: 2,
            reversion_steps: 1,
            delta_e: 1,
            e_max: 4,
            max_goal_candidates: 2,
            max_movesets_per_group: 3,
            lookahead: false,
            w_t: 0.05,
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
    /// Number of actually-created children (is_new=true from graph.insert).
    n_children: usize,
}

impl Default for EntropyState {
    fn default() -> Self {
        Self {
            entropy: 1,
            candidates_tried: 0,
            candidate_cache: Vec::new(),
            tried_moves: HashSet::new(),
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
pub(crate) fn generate_candidates(
    config: &Config,
    entropy: u32,
    params: &EntropyParams,
    ctx: &SearchContext,
    seed: u64,
) -> Vec<(MoveSet, Config, f64)> {
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

    // Step 2: score (qubit, bus triplet) pairs with entropy weighting.
    type TripletKey = (u8, u32, u8);

    struct ScoredEntry {
        qubit_id: u32,
        score: f64,
        lane_encoded: u64,
        dst_encoded: u64,
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
            index
                .outgoing_lanes(loc)
                .iter()
                .filter(|lane| {
                    index
                        .endpoints(lane)
                        .is_some_and(|(_, dst)| !occupied.contains(&dst.encode()))
                })
                .count() as f64
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

            // Combined lookahead + mobility in a single outgoing-lanes pass.
            let mut best_d2 = d_after;
            let mut m_after = 0.0_f64;
            for &next_lane in index.outgoing_lanes(dst) {
                let Some((_, next_dst)) = index.endpoints(&next_lane) else {
                    continue;
                };
                let enc = next_dst.encode();
                if occupied.contains(&enc) {
                    continue;
                }
                m_after += 1.0;
                if params.lookahead
                    && let Some(d) = dist_table.distance(enc, target_enc)
                {
                    let d_blended =
                        blended_distance(d as f64, enc, target_enc, params.w_t, dist_table);
                    best_d2 = best_d2.min(d_blended);
                }
            }
            let effective_d_after = if params.lookahead { best_d2 } else { d_after };
            let delta_d = d_now - effective_d_after;
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
    let mut all_scores: Vec<(TripletKey, ScoredEntry)> = raw_deltas
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

    // Step 3: per qubit, keep top C.
    let mut per_qubit: HashMap<u32, Vec<(TripletKey, ScoredEntry)>> = HashMap::new();
    for entry in all_scores.drain(..) {
        per_qubit.entry(entry.1.qubit_id).or_default().push(entry);
    }

    let mut selected: Vec<(TripletKey, ScoredEntry)> = Vec::new();
    let mut has_positive = false;

    for entries in per_qubit.values_mut() {
        entries.sort_by(|a, b| b.1.score.total_cmp(&a.1.score));
        entries.truncate(params.top_c);
        for e in entries.drain(..) {
            if e.1.score > 0.0 {
                has_positive = true;
            }
            selected.push(e);
        }
    }

    if !has_positive {
        selected.sort_by(|a, b| b.1.score.total_cmp(&a.1.score));
        selected.truncate(1);
    } else {
        selected.retain(|e| e.1.score > 0.0);
    }

    // Step 4: group by bus triplet.
    let mut groups: HashMap<TripletKey, Vec<ScoredEntry>> = HashMap::new();
    for (key, entry) in selected {
        groups.entry(key).or_default().push(entry);
    }

    // Step 5: per group, generate movesets (greedy by score, rotating start).
    let mut candidates: Vec<(f64, MoveSet, Config)> = Vec::new();

    for (_key, mut qubits) in groups {
        qubits.sort_by(|a, b| b.score.total_cmp(&a.score));
        let n = qubits.len().min(params.max_movesets_per_group);

        for start in 0..n {
            let mut lanes: Vec<u64> = Vec::new();
            let mut moves: Vec<(u32, LocationAddr)> = Vec::new();
            let mut used_dsts: HashSet<u64> = HashSet::new();
            let mut total_score: f64 = 0.0;

            let order: Vec<usize> = (start..qubits.len()).chain(0..start).collect();
            for &idx in &order {
                let t = &qubits[idx];
                if used_dsts.contains(&t.dst_encoded) || occupied.contains(&t.dst_encoded) {
                    continue;
                }
                lanes.push(t.lane_encoded);
                used_dsts.insert(t.dst_encoded);
                total_score += t.score;
                moves.push((t.qubit_id, LocationAddr::decode(t.dst_encoded)));
            }

            if lanes.is_empty() {
                continue;
            }

            let move_set = MoveSet::from_encoded(lanes);
            if candidates.iter().any(|(_, ms, _)| *ms == move_set) {
                continue;
            }
            let new_config = config.with_moves(&moves);
            candidates.push((total_score, move_set, new_config));
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
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));

    scored
        .into_iter()
        .map(|(_, ms, cfg)| (ms, cfg, 1.0))
        .collect()
}

/// Score a moveset: `alpha * distance_progress + beta * arrived + gamma * mobility_gain`.
pub(crate) fn score_moveset(
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
        distance_progress += (d_before - d_after).max(0.0);

        if new_loc.encode() == target_enc {
            arrived += 1.0;
        }

        mobility_before += index
            .outgoing_lanes(old_loc)
            .iter()
            .filter(|l| {
                index
                    .endpoints(l)
                    .is_some_and(|(_, d)| !occupied.contains(&d.encode()))
            })
            .count() as f64;
        mobility_after += index
            .outgoing_lanes(new_loc)
            .iter()
            .filter(|l| {
                index
                    .endpoints(l)
                    .is_some_and(|(_, d)| !new_occupied.contains(&d.encode()))
            })
            .count() as f64;
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
            let (child_id, _) = graph.insert(current, move_set, new_config, new_g);
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
                // Sequential fallback from root.
                let (goal_id, fb_expanded) = sequential_fallback(&mut graph, root_id, ctx, goal);
                nodes_expanded += fb_expanded;
                if let Some(gid) = goal_id {
                    found_goals.push(gid);
                }
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
        es.tried_moves.insert(move_key);
        es.candidates_tried += 1;

        // Insert into graph.
        let new_g = graph.g_score(current) + cost;
        let (child_id, is_new) = graph.insert(current, move_set, new_config, new_g);

        if !is_new {
            // Transposition: config seen at equal or better cost.
            entropy_map.entry(current).or_default().entropy += params.delta_e;
            continue;
        }

        // Track that a new child was created from this node.
        entropy_map.entry(current).or_default().n_children += 1;
        nodes_expanded += 1;
        max_depth_seen = max_depth_seen.max(graph.depth(child_id));

        if goal.is_goal(graph.config(child_id)) {
            found_goals.push(child_id);
            if found_goals.len() >= params.max_goal_candidates {
                break;
            }
            // cutoff_ancestor: walk up to first branching node.
            current = cutoff_ancestor(child_id, &graph, &entropy_map);
            continue;
        }

        current = child_id; // descend
    }

    // Return shallowest goal.
    let best = found_goals.into_iter().min_by_key(|&id| graph.depth(id));
    SearchResult {
        goal: best,
        nodes_expanded,
        max_depth_reached: max_depth_seen,
        graph,
    }
}

/// Get the next untried candidate from the cache, regenerating if needed.
fn get_next_candidate(
    entropy_map: &mut HashMap<NodeId, EntropyState>,
    node_id: NodeId,
    params: &EntropyParams,
    ctx: &SearchContext,
    graph: &SearchGraph,
    seed: u64,
) -> Option<(MoveSet, Config, f64)> {
    let config = graph.config(node_id);
    let es = entropy_map.entry(node_id).or_default();

    // Regenerate if we've exhausted max_candidates from current cache.
    if es.candidates_tried >= params.max_candidates || es.candidate_cache.is_empty() {
        es.candidate_cache = generate_candidates(config, es.entropy, params, ctx, seed);
        es.candidates_tried = 0;
    }

    // Find first untried candidate.
    while es.candidates_tried < es.candidate_cache.len() {
        let (ref ms, ref cfg, cost) = es.candidate_cache[es.candidates_tried];
        let move_key = ms.encoded_lanes().to_vec();
        if !es.tried_moves.contains(&move_key) {
            let result = (ms.clone(), cfg.clone(), cost);
            return Some(result);
        }
        es.candidates_tried += 1;
    }

    // All cached candidates already tried — regenerate and try again.
    es.candidate_cache = generate_candidates(config, es.entropy, params, ctx, seed);
    es.candidates_tried = 0;

    while es.candidates_tried < es.candidate_cache.len() {
        let (ref ms, ref cfg, cost) = es.candidate_cache[es.candidates_tried];
        let move_key = ms.encoded_lanes().to_vec();
        if !es.tried_moves.contains(&move_key) {
            let result = (ms.clone(), cfg.clone(), cost);
            return Some(result);
        }
        es.candidates_tried += 1;
    }

    None // all candidates exhausted
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
}
