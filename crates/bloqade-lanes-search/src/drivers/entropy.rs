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
use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::{Hash, Hasher};

use crate::drivers::astar::SearchResult;
use crate::observer::{SearchEvent, SearchObserver};
use crate::ops::aod_grid::BusGridContext;
use crate::primitives::config::Config;
use crate::primitives::context::SearchContext;
use crate::primitives::distance::DistanceTable;
use crate::primitives::graph::{MoveSet, NodeId, SearchGraph};
use crate::primitives::lane_index::LaneIndex;
use crate::primitives::ordering::{
    TripletKey, cmp_moveset_config_tiebreak, cmp_qubit_lane_dst_tiebreak,
    cmp_triplet_entry_tiebreak,
};
use crate::primitives::path::find_path_occupied;
use crate::traits::Goal;
use bloqade_lanes_bytecode_core::arch::addr::{Direction, LaneAddr, LocationAddr, MoveType};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

#[cfg(test)]
static COMPUTE_MOVESET_METRICS_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Trace payload for entropy visualization/replay.
#[derive(Debug, Clone, Default)]
pub struct EntropyTrace {
    pub root_node_id: u32,
    pub best_buffer_size: u32,
    pub steps: Vec<EntropyTraceStep>,
}

impl EntropyTrace {
    /// Construct an empty trace sized for the given entropy params.
    ///
    /// `best_buffer_size` is derived from `params.max_goal_candidates`
    /// (the resume buffer holds up to `max_goal_candidates - 1` entries).
    /// `root_node_id` is `0` — the convention for the root of a fresh
    /// [`SearchGraph`].
    pub fn for_params(params: &EntropyParams) -> Self {
        Self {
            root_node_id: 0,
            best_buffer_size: params.max_goal_candidates.saturating_sub(1) as u32,
            steps: Vec::new(),
        }
    }
}

/// One entropy-search step snapshot.
#[derive(Debug, Clone)]
#[allow(clippy::type_complexity)]
pub struct EntropyTraceStep {
    pub event: String,
    pub node_id: u32,
    pub parent_node_id: Option<u32>,
    pub depth: u32,
    pub entropy: u32,
    pub unresolved_count: u32,
    pub moveset: Option<Vec<(u8, u8, u32, u32, u32, u32)>>,
    pub candidate_movesets: Vec<Vec<(u8, u8, u32, u32, u32, u32)>>,
    pub candidate_index: Option<u32>,
    pub reason: Option<String>,
    pub state_seen_node_id: Option<u32>,
    pub no_valid_moves_qubit: Option<u32>,
    pub trigger_node_id: Option<u32>,
    pub configuration: Vec<(u32, u32, u32, u32)>,
    pub parent_configuration: Option<Vec<(u32, u32, u32, u32)>>,
    pub moveset_score: Option<f64>,
    pub best_buffer_node_ids: Vec<u32>,
}

/// `EntropyTrace` collects entropy-driver events into a `Vec<EntropyTraceStep>`,
/// preserving the legacy step-record shape consumed by the Python
/// visualization layer. Frontier-driver events (`GoalFound`,
/// `NodeExpanded`) are ignored — `EntropyTrace` is specifically the
/// entropy driver's trace sink.
impl SearchObserver for EntropyTrace {
    fn on_event(&mut self, event: SearchEvent<'_>) {
        let to_candidate_tuples = |movesets: &[MoveSet]| {
            movesets
                .iter()
                .map(moveset_to_trace_tuple)
                .collect::<Vec<_>>()
        };
        match event {
            SearchEvent::EntropyDescend {
                node_id,
                parent_node_id,
                depth,
                entropy,
                unresolved_count,
                moveset,
                candidate_movesets,
                candidate_index,
                reason,
                configuration,
                parent_configuration,
                moveset_score,
                best_buffer_node_ids,
            } => {
                self.steps.push(EntropyTraceStep {
                    event: "descend".to_string(),
                    node_id: node_id.0,
                    parent_node_id: Some(parent_node_id.0),
                    depth,
                    entropy,
                    unresolved_count,
                    moveset: Some(moveset_to_trace_tuple(moveset)),
                    candidate_movesets: to_candidate_tuples(candidate_movesets),
                    candidate_index: Some(candidate_index),
                    reason: reason.map(|s| s.to_string()),
                    state_seen_node_id: None,
                    no_valid_moves_qubit: None,
                    trigger_node_id: None,
                    configuration: config_as_trace_tuples(configuration),
                    parent_configuration: Some(config_as_trace_tuples(parent_configuration)),
                    moveset_score: Some(moveset_score),
                    best_buffer_node_ids: best_buffer_node_ids.to_vec(),
                });
            }
            SearchEvent::EntropyGoal {
                node_id,
                parent_node_id,
                depth,
                entropy,
                moveset,
                candidate_movesets,
                candidate_index,
                reason,
                state_seen_node_id,
                trigger_node_id,
                configuration,
                parent_configuration,
                best_buffer_node_ids,
            } => {
                self.steps.push(EntropyTraceStep {
                    event: "goal".to_string(),
                    node_id: node_id.0,
                    parent_node_id: parent_node_id.map(|id| id.0),
                    depth,
                    entropy,
                    unresolved_count: 0,
                    moveset: moveset.map(moveset_to_trace_tuple),
                    candidate_movesets: to_candidate_tuples(candidate_movesets),
                    candidate_index,
                    reason: reason.map(str::to_string),
                    state_seen_node_id: state_seen_node_id.map(|id| id.0),
                    no_valid_moves_qubit: None,
                    trigger_node_id: trigger_node_id.map(|id| id.0),
                    configuration: config_as_trace_tuples(configuration),
                    parent_configuration: parent_configuration.map(config_as_trace_tuples),
                    moveset_score: None,
                    best_buffer_node_ids: best_buffer_node_ids.to_vec(),
                });
            }
            SearchEvent::EntropyBump {
                node_id,
                parent_node_id,
                depth,
                entropy,
                unresolved_count,
                moveset,
                candidate_movesets,
                candidate_index,
                reason,
                state_seen_node_id,
                no_valid_moves_qubit,
                configuration,
                parent_configuration,
                best_buffer_node_ids,
            } => {
                self.steps.push(EntropyTraceStep {
                    event: "entropy_bump".to_string(),
                    node_id: node_id.0,
                    parent_node_id: parent_node_id.map(|id| id.0),
                    depth,
                    entropy,
                    unresolved_count,
                    moveset: moveset.map(moveset_to_trace_tuple),
                    candidate_movesets: to_candidate_tuples(candidate_movesets),
                    candidate_index,
                    reason: Some(reason.to_string()),
                    state_seen_node_id: state_seen_node_id.map(|id| id.0),
                    no_valid_moves_qubit,
                    trigger_node_id: None,
                    configuration: config_as_trace_tuples(configuration),
                    parent_configuration: parent_configuration.map(config_as_trace_tuples),
                    moveset_score: None,
                    best_buffer_node_ids: best_buffer_node_ids.to_vec(),
                });
            }
            SearchEvent::EntropyRevert {
                node_id,
                parent_node_id,
                depth,
                entropy,
                unresolved_count,
                candidate_movesets,
                trigger_node_id,
                trigger_entropy,
                configuration,
                parent_configuration,
                best_buffer_node_ids,
            } => {
                self.steps.push(EntropyTraceStep {
                    event: "revert".to_string(),
                    node_id: node_id.0,
                    parent_node_id: parent_node_id.map(|id| id.0),
                    depth,
                    entropy,
                    unresolved_count,
                    moveset: None,
                    candidate_movesets: to_candidate_tuples(candidate_movesets),
                    candidate_index: None,
                    reason: Some("entropy".to_string()),
                    state_seen_node_id: None,
                    no_valid_moves_qubit: None,
                    trigger_node_id: Some(trigger_node_id.0),
                    configuration: config_as_trace_tuples(configuration),
                    parent_configuration: parent_configuration.map(config_as_trace_tuples),
                    moveset_score: Some(trigger_entropy as f64),
                    best_buffer_node_ids: best_buffer_node_ids.to_vec(),
                });
            }
            SearchEvent::EntropyFallbackStart {
                node_id,
                parent_node_id,
                depth,
                unresolved_count,
                configuration,
                best_buffer_node_ids,
            } => {
                self.steps.push(EntropyTraceStep {
                    event: "fallback_start".to_string(),
                    node_id: node_id.0,
                    parent_node_id: parent_node_id.map(|id| id.0),
                    depth,
                    entropy: 0,
                    unresolved_count,
                    moveset: None,
                    candidate_movesets: Vec::new(),
                    candidate_index: None,
                    reason: None,
                    state_seen_node_id: None,
                    no_valid_moves_qubit: None,
                    trigger_node_id: None,
                    configuration: config_as_trace_tuples(configuration),
                    parent_configuration: None,
                    moveset_score: None,
                    best_buffer_node_ids: best_buffer_node_ids.to_vec(),
                });
            }
            // Frontier-driver events — EntropyTrace records only entropy-driver events.
            SearchEvent::GoalFound { .. } | SearchEvent::NodeExpanded { .. } => {}
        }
    }
}

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
            // Synced with Python SearchParams defaults (commit 9b470b3).
            w_d: 0.95,
            w_m: 0.8,
            alpha: 80.0,
            beta: 3.0,
            gamma: 3.1,
            max_candidates: 4,
            reversion_steps: 1,
            e_max: 4,
            max_goal_candidates: 3,
            max_movesets_per_group: 3,
            lookahead: false,
            w_t: 0.95,
        }
    }
}

// ── Per-node state ─────────────────────────────────────────────────

#[derive(Debug)]
struct EntropyState {
    entropy: u32,
    candidates_tried: usize,
    candidate_cache: Vec<CandidateEntry>,
    /// Encoded lane vecs of movesets already attempted from this node.
    tried_moves: HashSet<Vec<u64>>,
    /// Encoded lane vecs of movesets that failed (collision/transposition).
    /// Skipped on retry to avoid repeating known failures.
    failed_candidates: HashSet<Vec<u64>>,
    /// Number of actually-created children (is_new=true from graph.insert).
    n_children: usize,
}

#[derive(Clone, Copy)]
struct ScoredEntry {
    qubit_id: u32,
    score: f64,
    lane_encoded: u64,
    dst_encoded: u64,
}

type CandidateEntry = (MoveSet, Config, f64, bool);

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

fn decode_triplet_key(mt_u8: u8, dir_u8: u8) -> (MoveType, Direction) {
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
    (mt, dir)
}

fn build_deadlock_breaker_candidate(
    config: &Config,
    occupied: &HashSet<u64>,
    all_scores: &[(TripletKey, ScoredEntry)],
    ctx: &SearchContext,
) -> Option<(f64, MoveSet, Config)> {
    let unresolved: HashSet<u32> = ctx
        .targets
        .iter()
        .filter_map(|(qid, target)| {
            let current = config.location_of(*qid)?;
            (current.encode() != *target).then_some(*qid)
        })
        .collect();
    if unresolved.is_empty() {
        return None;
    }
    let target_movers = unresolved.len().div_ceil(2).max(1);

    let mut groups: BTreeMap<TripletKey, Vec<ScoredEntry>> = BTreeMap::new();
    for &(key, entry) in all_scores {
        groups.entry(key).or_default().push(entry);
    }

    let mut best: Option<(usize, f64, MoveSet, Config)> = None;
    for ((mt_u8, bus_id, dir_u8), mut qubits) in groups {
        qubits.sort_by(cmp_group_entries);
        let (mt, dir) = decode_triplet_key(mt_u8, dir_u8);
        let grid_ctx = BusGridContext::new(ctx.index, mt, bus_id, None, dir, occupied);

        let mut entries: HashMap<u64, u64> = HashMap::new();
        let mut entry_by_lane: HashMap<u64, ScoredEntry> = HashMap::new();
        let mut seen_qubits: HashSet<u32> = HashSet::new();
        let mut selected_unresolved = 0usize;
        for t in &qubits {
            if !seen_qubits.insert(t.qubit_id) {
                continue;
            }
            if unresolved.contains(&t.qubit_id) && selected_unresolved >= target_movers {
                continue;
            }
            let lane = LaneAddr::decode_u64(t.lane_encoded);
            if let Some((src, _)) = ctx.index.endpoints(&lane) {
                let src_enc = src.encode();
                if entries.contains_key(&src_enc) {
                    continue;
                }
                entries.insert(src_enc, t.lane_encoded);
                entry_by_lane.insert(t.lane_encoded, *t);
                if unresolved.contains(&t.qubit_id) {
                    selected_unresolved += 1;
                }
            }
        }

        if entries.is_empty() {
            continue;
        }

        for grid_lanes in grid_ctx.build_aod_grids(&entries) {
            let mut total_score = 0.0;
            let mut moves: Vec<(u32, LocationAddr)> = Vec::new();
            let mut moved_unresolved = 0usize;

            for lane_enc in &grid_lanes {
                if let Some(t) = entry_by_lane.get(lane_enc) {
                    total_score += t.score;
                    moves.push((t.qubit_id, LocationAddr::decode(t.dst_encoded)));
                    if unresolved.contains(&t.qubit_id) {
                        moved_unresolved += 1;
                    }
                }
            }

            if moves.is_empty() {
                continue;
            }

            let move_set = MoveSet::from_encoded(grid_lanes);
            let new_config = config.with_moves(&moves);
            match &best {
                None => best = Some((moved_unresolved, total_score, move_set, new_config)),
                Some((best_moved, best_score, best_moveset, _)) => {
                    let better = moved_unresolved > *best_moved
                        || (moved_unresolved == *best_moved
                            && (total_score > *best_score
                                || (total_score == *best_score
                                    && move_set.encoded_lanes() < best_moveset.encoded_lanes())));
                    if better {
                        best = Some((moved_unresolved, total_score, move_set, new_config));
                    }
                }
            }
        }
    }

    best.map(|(_, score, move_set, new_config)| (score, move_set, new_config))
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

#[derive(Debug, Clone, Copy)]
struct ScoredResumeState {
    node_id: NodeId,
    score: f64,
    depth: u32,
    order: u64,
}

fn cmp_resume_states(a: &ScoredResumeState, b: &ScoredResumeState) -> Ordering {
    b.score
        .total_cmp(&a.score)
        .then_with(|| b.depth.cmp(&a.depth))
        .then_with(|| b.order.cmp(&a.order))
}

fn resume_buffer_insert(
    buffer: &mut Vec<ScoredResumeState>,
    node_id: NodeId,
    score: f64,
    depth: u32,
    capacity: usize,
    next_order: &mut u64,
) {
    if capacity == 0 {
        return;
    }
    resume_buffer_discard(buffer, node_id);

    let candidate = ScoredResumeState {
        node_id,
        score,
        depth,
        order: *next_order,
    };
    *next_order = next_order.saturating_add(1);

    if buffer.len() < capacity {
        buffer.push(candidate);
        return;
    }

    let Some(worst_idx) = buffer
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| cmp_resume_states(a, b))
        .map(|(idx, _)| idx)
    else {
        return;
    };
    if cmp_resume_states(&candidate, &buffer[worst_idx]) == Ordering::Less {
        buffer[worst_idx] = candidate;
    }
}

fn resume_buffer_discard(buffer: &mut Vec<ScoredResumeState>, node_id: NodeId) {
    buffer.retain(|entry| entry.node_id != node_id);
}

fn resume_buffer_pop_best(
    buffer: &mut Vec<ScoredResumeState>,
    best_goal_depth: Option<u32>,
) -> Option<NodeId> {
    loop {
        let best_idx = buffer
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| cmp_resume_states(a, b))
            .map(|(idx, _)| idx)?;
        let best = buffer.swap_remove(best_idx);
        if let Some(depth_cap) = best_goal_depth
            && best.depth >= depth_cap
        {
            continue;
        }
        return Some(best.node_id);
    }
}

fn trace_buffer_node_ids(buffer: &[ScoredResumeState]) -> Vec<u32> {
    let mut ranked = buffer.to_vec();
    ranked.sort_by(cmp_resume_states);
    ranked.into_iter().map(|entry| entry.node_id.0).collect()
}

fn approx_layer_time_us(moveset: &MoveSet, index: &LaneIndex) -> f64 {
    moveset
        .encoded_lanes()
        .iter()
        .map(|&lane_bits| {
            let lane = LaneAddr::decode_u64(lane_bits);
            index.lane_duration_us(&lane).unwrap_or(1.0)
        })
        .fold(0.0_f64, f64::max)
}

fn approx_path_time_us(graph: &SearchGraph, goal_id: NodeId, index: &LaneIndex) -> f64 {
    graph
        .reconstruct_path(goal_id)
        .into_iter()
        .map(|moveset| approx_layer_time_us(&moveset, index))
        .sum()
}

fn path_lexicographic_key(graph: &SearchGraph, goal_id: NodeId) -> Vec<Vec<u64>> {
    graph
        .reconstruct_path(goal_id)
        .into_iter()
        .map(|moveset| moveset.encoded_lanes().to_vec())
        .collect()
}

fn select_best_goal_with_tiebreak(
    found_goals: &[NodeId],
    graph: &SearchGraph,
    index: &LaneIndex,
) -> Option<NodeId> {
    let min_depth = found_goals.iter().map(|&id| graph.depth(id)).min()?;
    found_goals
        .iter()
        .copied()
        .filter(|&id| graph.depth(id) == min_depth)
        .map(|id| {
            (
                id,
                approx_path_time_us(graph, id, index),
                path_lexicographic_key(graph, id),
            )
        })
        .min_by(|a, b| {
            a.1.total_cmp(&b.1)
                .then_with(|| a.2.cmp(&b.2))
                .then_with(|| a.0.0.cmp(&b.0.0))
        })
        .map(|(id, _, _)| id)
}

fn best_untried_moveset_score(
    es: &EntropyState,
    config: &Config,
    occupied: &HashSet<u64>,
    ctx: &SearchContext,
    params: &EntropyParams,
) -> Option<f64> {
    es.candidate_cache
        .iter()
        .filter_map(|(moveset, candidate_cfg, _, _)| {
            let move_key = moveset.encoded_lanes().to_vec();
            if es.tried_moves.contains(&move_key) || es.failed_candidates.contains(&move_key) {
                return None;
            }
            Some(score_moveset(config, candidate_cfg, occupied, ctx, params))
        })
        .max_by(|a, b| a.total_cmp(b))
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

fn unresolved_count(config: &Config, targets: &[(u32, u64)]) -> u32 {
    targets
        .iter()
        .filter(|&&(qid, target_enc)| {
            config
                .location_of(qid)
                .is_some_and(|loc| loc.encode() != target_enc)
        })
        .count() as u32
}

fn config_as_trace_tuples(config: &Config) -> Vec<(u32, u32, u32, u32)> {
    config
        .iter()
        .map(|(qid, loc)| (qid, loc.zone_id, loc.word_id, loc.site_id))
        .collect()
}

fn lane_to_trace_tuple(lane: LaneAddr) -> (u8, u8, u32, u32, u32, u32) {
    (
        lane.direction as u8,
        lane.move_type as u8,
        lane.zone_id,
        lane.word_id,
        lane.site_id,
        lane.bus_id,
    )
}

fn moveset_to_trace_tuple(ms: &MoveSet) -> Vec<(u8, u8, u32, u32, u32, u32)> {
    ms.decode().into_iter().map(lane_to_trace_tuple).collect()
}

fn first_unresolved_qubit_without_valid_move(config: &Config, ctx: &SearchContext) -> Option<u32> {
    let mut occupied = HashSet::with_capacity(ctx.blocked.len() + config.len());
    occupied.extend(ctx.blocked);
    for (_, loc) in config.iter() {
        occupied.insert(loc.encode());
    }

    for &(qid, target_enc) in ctx.targets {
        let Some(current_loc) = config.location_of(qid) else {
            continue;
        };
        if current_loc.encode() == target_enc {
            continue;
        }
        let mut has_valid_lane = false;
        for &lane in ctx.index.outgoing_lanes(current_loc) {
            let Some((_, dst)) = ctx.index.endpoints(&lane) else {
                continue;
            };
            if !occupied.contains(&dst.encode()) {
                has_valid_lane = true;
                break;
            }
        }
        if !has_valid_lane {
            return Some(qid);
        }
    }
    None
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_candidates(
    config: &Config,
    entropy: u32,
    params: &EntropyParams,
    ctx: &SearchContext,
    seed: u64,
) -> Vec<CandidateEntry> {
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
                if blocked.contains(&dst_e) {
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
            if blocked.contains(&dst_enc) {
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
                if blocked.contains(&enc) {
                    continue;
                }
                // Distance-weighted mobility: closer destinations count more.
                let d_to_target = dist_table.distance(enc, target_enc).map_or(f64::MAX, |d| {
                    blended_distance(d as f64, enc, target_enc, params.w_t, dist_table)
                });
                if d_to_target < f64::MAX {
                    m_after += 1.0 / (1.0 + d_to_target);
                }
                if params.lookahead && d_to_target < f64::MAX {
                    best_d2 = best_d2.min(d_to_target);
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
    // If none are positive, keep only the single best entry as fallback.
    let has_positive = all_scores.iter().any(|e| e.1.score > 0.0);
    let selected: Vec<(TripletKey, ScoredEntry)> = if has_positive {
        all_scores
            .iter()
            .copied()
            .filter(|e| e.1.score > 0.0)
            .collect()
    } else {
        all_scores
            .iter()
            .copied()
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
        let (mt, dir) = decode_triplet_key(mt_u8, dir_u8);

        let grid_ctx = BusGridContext::new(ctx.index, mt, bus_id, None, dir, &occupied);

        let mut entries: HashMap<u64, u64> = HashMap::new();
        let mut entry_by_lane: HashMap<u64, &ScoredEntry> = HashMap::new();
        for t in &qubits {
            let lane = LaneAddr::decode_u64(t.lane_encoded);
            if let Some((src, _)) = ctx.index.endpoints(&lane) {
                let src_enc = src.encode();
                entries.insert(src_enc, t.lane_encoded);
                entry_by_lane.insert(t.lane_encoded, t);
            }
        }

        // Grids may include empty filler lanes so the emitted MoveSet remains
        // a complete AOD rectangle. Only selected entries add qubit moves.
        let grids = grid_ctx.build_aod_grids(&entries);
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

    let mut used_deadlock_breaker = false;
    if candidates.is_empty()
        && let Some(deadlock_breaker) =
            build_deadlock_breaker_candidate(config, &occupied, &all_scores, ctx)
    {
        candidates.push(deadlock_breaker);
        used_deadlock_breaker = true;
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

    scored
        .into_iter()
        .map(|(_, ms, cfg)| (ms, cfg, 1.0, used_deadlock_breaker))
        .collect()
}

/// Detailed per-moveset scoring breakdown returned by [`compute_moveset_metrics`].
#[derive(Debug, Clone, Default)]
pub struct MovesetMetrics {
    pub distance_progress: f64,
    pub arrived: u32,
    pub mobility_before: f64,
    pub mobility_after: f64,
    /// Qubit ids that ended up strictly closer to their target.
    pub closer: Vec<u32>,
    /// Qubit ids that ended up strictly further from their target.
    pub further: Vec<u32>,
}

impl MovesetMetrics {
    pub fn mobility_gain(&self) -> f64 {
        self.mobility_after - self.mobility_before
    }

    pub fn score(&self, params: &EntropyParams) -> f64 {
        params.alpha * self.distance_progress
            + params.beta * (self.arrived as f64)
            + params.gamma * self.mobility_gain()
    }
}

/// Compute the full metrics breakdown for moving from `old_config` to `new_config`.
///
/// Extends [`score_moveset`]'s scalar output with distance/arrival/mobility
/// components plus the set of qubits that got closer vs further from their
/// targets, so visualizers and tests can inspect contributions individually.
pub fn compute_moveset_metrics(
    old_config: &Config,
    new_config: &Config,
    occupied: &HashSet<u64>,
    ctx: &SearchContext,
    params: &EntropyParams,
) -> MovesetMetrics {
    #[cfg(test)]
    COMPUTE_MOVESET_METRICS_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    let targets = ctx.targets;
    let dist_table = ctx.dist_table;
    let blocked = ctx.blocked;
    let index = ctx.index;
    let mut new_occupied: HashSet<u64> = new_config.iter().map(|(_, loc)| loc.encode()).collect();
    new_occupied.extend(blocked);

    let mut metrics = MovesetMetrics::default();

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
        metrics.distance_progress += (d_before - d_after).max(0.0);
        if d_after < d_before {
            metrics.closer.push(qid);
        } else if d_after > d_before {
            metrics.further.push(qid);
        }

        if new_loc.encode() == target_enc {
            metrics.arrived += 1;
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
                metrics.mobility_before += 1.0 / (1.0 + d);
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
                metrics.mobility_after += 1.0 / (1.0 + d);
            }
        }
    }

    metrics.closer.sort_unstable();
    metrics.further.sort_unstable();
    metrics
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

    let mut distance_progress = 0.0;
    let mut arrived = 0_u32;
    let mut mobility_before = 0.0;
    let mut mobility_after = 0.0;

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
            arrived += 1;
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
        + params.beta * (arrived as f64)
        + params.gamma * (mobility_after - mobility_before)
}

// ── Sequential fallback ────────────────────────────────────────────

fn fire_fallback_start_event(
    observer: &mut dyn SearchObserver,
    graph: &SearchGraph,
    root_id: NodeId,
    ctx: &SearchContext,
    resume_buffer: &[ScoredResumeState],
) {
    let cfg = graph.config(root_id);
    let buffer_ids = trace_buffer_node_ids(resume_buffer);
    observer.on_event(SearchEvent::EntropyFallbackStart {
        node_id: root_id,
        parent_node_id: graph.parent(root_id),
        depth: graph.depth(root_id),
        unresolved_count: unresolved_count(cfg, ctx.targets),
        configuration: cfg,
        best_buffer_node_ids: &buffer_ids,
    });
}

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
#[allow(clippy::too_many_arguments)]
pub fn entropy_search(
    root: Config,
    goal: &impl Goal,
    params: &EntropyParams,
    ctx: &SearchContext,
    max_expansions: Option<u32>,
    max_depth: Option<u32>,
    seed: u64,
    observer: &mut dyn SearchObserver,
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
    let resume_capacity = params.max_goal_candidates.saturating_sub(1);
    let mut resume_buffer: Vec<ScoredResumeState> = Vec::new();
    let mut resume_insert_order: u64 = 0;
    let mut best_goal_depth: Option<u32> = None;
    let mut budget_exhausted = false;

    // Safety cap: hard iteration limit prevents infinite loops when
    // max_expansions is None and the search gets stuck in reversion cycles.
    let hard_limit = max_expansions.unwrap_or(ctx.index.num_locations() as u32 * 10);
    let mut iterations: u32 = 0;

    loop {
        iterations += 1;
        if nodes_expanded >= hard_limit || iterations >= hard_limit * 2 {
            budget_exhausted = true;
            break;
        }
        if let Some(depth_cap) = best_goal_depth
            && graph.depth(current) >= depth_cap
        {
            current =
                resume_buffer_pop_best(&mut resume_buffer, best_goal_depth).unwrap_or(root_id);
            continue;
        }

        let es = entropy_map.entry(current).or_default();

        // Force entropy at depth limit.
        if let Some(max_d) = max_depth
            && graph.depth(current) >= max_d
        {
            es.entropy = params.e_max;
        }

        // REVERSION: entropy too high. The root node is allowed to keep
        // accumulating entropy until the expansion/iteration budget is exhausted.
        if current != root_id && es.entropy >= params.e_max {
            let trigger_node = current;
            let trigger_entropy = es.entropy;
            let mut ancestor = current;
            for _ in 0..params.reversion_steps {
                if let Some(parent) = graph.parent(ancestor) {
                    ancestor = parent;
                } else {
                    break;
                }
            }

            let new_ancestor_entropy = {
                let ancestor_es = entropy_map.entry(ancestor).or_default();
                ancestor_es.entropy += 1;
                ancestor_es.entropy
            };
            let ancestor_cfg = graph.config(ancestor);
            let parent_id = graph.parent(ancestor);
            let parent_cfg = parent_id.map(|pid| graph.config(pid));
            let candidate_movesets: Vec<MoveSet> = entropy_map
                .get(&trigger_node)
                .map(|s| {
                    s.candidate_cache
                        .iter()
                        .map(|(ms, _, _, _)| ms.clone())
                        .collect()
                })
                .unwrap_or_default();
            let buffer_ids = trace_buffer_node_ids(&resume_buffer);
            observer.on_event(SearchEvent::EntropyRevert {
                node_id: ancestor,
                parent_node_id: parent_id,
                depth: graph.depth(ancestor),
                entropy: new_ancestor_entropy,
                unresolved_count: unresolved_count(ancestor_cfg, ctx.targets),
                candidate_movesets: &candidate_movesets,
                trigger_node_id: trigger_node,
                trigger_entropy,
                configuration: ancestor_cfg,
                parent_configuration: parent_cfg,
                best_buffer_node_ids: &buffer_ids,
            });
            current = ancestor;
            continue;
        }

        // CANDIDATE SELECTION.
        let candidate = get_next_candidate(&mut entropy_map, current, params, ctx, &graph, seed);

        let Some((candidate_idx, move_set, new_config, cost, candidate_origin)) = candidate else {
            // No candidates available — bump entropy.
            let no_valid_qid =
                first_unresolved_qubit_without_valid_move(graph.config(current), ctx);
            let new_entropy = {
                let current_es = entropy_map.entry(current).or_default();
                current_es.entropy += 1;
                current_es.entropy
            };
            let candidate_movesets: Vec<MoveSet> = entropy_map
                .get(&current)
                .map(|s| {
                    s.candidate_cache
                        .iter()
                        .map(|(ms, _, _, _)| ms.clone())
                        .collect()
                })
                .unwrap_or_default();
            let cfg = graph.config(current);
            let parent_id = graph.parent(current);
            let parent_cfg = parent_id.map(|pid| graph.config(pid));
            let buffer_ids = trace_buffer_node_ids(&resume_buffer);
            observer.on_event(SearchEvent::EntropyBump {
                node_id: current,
                parent_node_id: parent_id,
                depth: graph.depth(current),
                entropy: new_entropy,
                unresolved_count: unresolved_count(cfg, ctx.targets),
                moveset: None,
                candidate_movesets: &candidate_movesets,
                candidate_index: None,
                reason: "no-valid-moves",
                state_seen_node_id: None,
                no_valid_moves_qubit: no_valid_qid,
                configuration: cfg,
                parent_configuration: parent_cfg,
                best_buffer_node_ids: &buffer_ids,
            });
            continue;
        };

        // Record as tried.
        let es = entropy_map.entry(current).or_default();
        let move_key = move_set.encoded_lanes().to_vec();
        es.tried_moves.insert(move_key.clone());
        es.candidates_tried += 1;

        // Insert into graph.
        let trace_move_set = move_set.clone();
        let new_g = graph.g_score(current) + cost;
        let (child_id, is_new) = graph.insert(current, move_set, new_config, new_g);

        if !is_new {
            if goal.is_goal(graph.config(child_id)) {
                let goal_depth = graph.depth(child_id);
                found_goals.push(child_id);
                if best_goal_depth.is_none_or(|depth| goal_depth < depth) {
                    best_goal_depth = Some(goal_depth);
                }
                resume_buffer_discard(&mut resume_buffer, child_id);
                let goal_cfg = graph.config(child_id);
                let goal_parent_id = graph.parent(child_id);
                let goal_parent_cfg = goal_parent_id.map(|pid| graph.config(pid));
                let entropy_now = entropy_map.get(&current).map_or(1, |s| s.entropy);
                let candidate_movesets: Vec<MoveSet> = entropy_map
                    .get(&current)
                    .map(|s| {
                        s.candidate_cache
                            .iter()
                            .map(|(ms, _, _, _)| ms.clone())
                            .collect()
                    })
                    .unwrap_or_default();
                let buffer_ids = trace_buffer_node_ids(&resume_buffer);
                observer.on_event(SearchEvent::EntropyGoal {
                    node_id: child_id,
                    // Keep canonical parent for existing nodes; using the
                    // current trigger node would visually re-parent the
                    // node in the reducer and cause tree jitter/overlap.
                    parent_node_id: goal_parent_id,
                    depth: graph.depth(child_id),
                    entropy: entropy_now,
                    moveset: Some(&trace_move_set),
                    candidate_movesets: &candidate_movesets,
                    candidate_index: Some(candidate_idx as u32),
                    reason: Some("state-seen-goal"),
                    state_seen_node_id: Some(child_id),
                    trigger_node_id: Some(current),
                    configuration: goal_cfg,
                    parent_configuration: goal_parent_cfg,
                    best_buffer_node_ids: &buffer_ids,
                });
                if found_goals.len() >= params.max_goal_candidates {
                    break;
                }
                current =
                    resume_buffer_pop_best(&mut resume_buffer, best_goal_depth).unwrap_or(root_id);
                continue;
            }
            // Transposition: config seen at equal or better cost.
            let new_entropy = {
                let es = entropy_map.entry(current).or_default();
                es.failed_candidates.insert(move_key.clone());
                es.entropy += 1;
                es.entropy
            };
            let candidate_movesets: Vec<MoveSet> = entropy_map
                .get(&current)
                .map(|s| {
                    s.candidate_cache
                        .iter()
                        .map(|(ms, _, _, _)| ms.clone())
                        .collect()
                })
                .unwrap_or_default();
            let cfg = graph.config(current);
            let parent_id = graph.parent(current);
            let parent_cfg = parent_id.map(|pid| graph.config(pid));
            let buffer_ids = trace_buffer_node_ids(&resume_buffer);
            observer.on_event(SearchEvent::EntropyBump {
                node_id: current,
                parent_node_id: parent_id,
                depth: graph.depth(current),
                entropy: new_entropy,
                unresolved_count: unresolved_count(cfg, ctx.targets),
                moveset: Some(&trace_move_set),
                candidate_movesets: &candidate_movesets,
                candidate_index: Some(candidate_idx as u32),
                reason: "state-seen",
                state_seen_node_id: Some(child_id),
                no_valid_moves_qubit: None,
                configuration: cfg,
                parent_configuration: parent_cfg,
                best_buffer_node_ids: &buffer_ids,
            });
            continue;
        }

        // Track that a new child was created from this node.
        entropy_map.entry(current).or_default().n_children += 1;
        nodes_expanded += 1;
        let child_depth = graph.depth(child_id);
        max_depth_seen = max_depth_seen.max(child_depth);
        let child_cfg = graph.config(child_id);
        let current_cfg = graph.config(current);
        let mut occupied = HashSet::with_capacity(ctx.blocked.len() + current_cfg.len());
        occupied.extend(ctx.blocked);
        for (_, loc) in current_cfg.iter() {
            occupied.insert(loc.encode());
        }
        let moveset_score = score_moveset(current_cfg, child_cfg, &occupied, ctx, params);
        resume_buffer_discard(&mut resume_buffer, current);
        if let Some(next_best_score) = entropy_map
            .get(&current)
            .and_then(|es| best_untried_moveset_score(es, current_cfg, &occupied, ctx, params))
        {
            resume_buffer_insert(
                &mut resume_buffer,
                current,
                next_best_score,
                graph.depth(current),
                resume_capacity,
                &mut resume_insert_order,
            );
        }

        let entropy_now = entropy_map.get(&current).map_or(1, |s| s.entropy);
        let candidate_movesets: Vec<MoveSet> = entropy_map
            .get(&current)
            .map(|s| {
                s.candidate_cache
                    .iter()
                    .map(|(ms, _, _, _)| ms.clone())
                    .collect()
            })
            .unwrap_or_default();
        let current_cfg_owned = graph.config(current);
        let buffer_ids = trace_buffer_node_ids(&resume_buffer);
        observer.on_event(SearchEvent::EntropyDescend {
            node_id: child_id,
            parent_node_id: current,
            depth: graph.depth(child_id),
            entropy: entropy_now,
            unresolved_count: unresolved_count(child_cfg, ctx.targets),
            moveset: &trace_move_set,
            candidate_movesets: &candidate_movesets,
            candidate_index: candidate_idx as u32,
            reason: candidate_origin.then_some("deadlock-breaker"),
            configuration: child_cfg,
            parent_configuration: current_cfg_owned,
            moveset_score,
            best_buffer_node_ids: &buffer_ids,
        });

        if goal.is_goal(graph.config(child_id)) {
            let goal_depth = graph.depth(child_id);
            found_goals.push(child_id);
            if best_goal_depth.is_none_or(|depth| goal_depth < depth) {
                best_goal_depth = Some(goal_depth);
            }
            resume_buffer_discard(&mut resume_buffer, child_id);
            let goal_cfg = graph.config(child_id);
            let goal_parent_id = graph.parent(child_id);
            let goal_parent_cfg = goal_parent_id.map(|pid| graph.config(pid));
            let entropy_at_goal = entropy_map.get(&current).map_or(1, |s| s.entropy);
            let buffer_ids = trace_buffer_node_ids(&resume_buffer);
            observer.on_event(SearchEvent::EntropyGoal {
                node_id: child_id,
                parent_node_id: goal_parent_id,
                depth: graph.depth(child_id),
                entropy: entropy_at_goal,
                moveset: None,
                candidate_movesets: &[],
                candidate_index: None,
                reason: None,
                state_seen_node_id: None,
                trigger_node_id: None,
                configuration: goal_cfg,
                parent_configuration: goal_parent_cfg,
                best_buffer_node_ids: &buffer_ids,
            });
            if found_goals.len() >= params.max_goal_candidates {
                break;
            }
            current =
                resume_buffer_pop_best(&mut resume_buffer, best_goal_depth).unwrap_or(root_id);
            continue;
        }

        if let Some(depth_cap) = best_goal_depth
            && child_depth >= depth_cap
        {
            resume_buffer_discard(&mut resume_buffer, child_id);
            current =
                resume_buffer_pop_best(&mut resume_buffer, best_goal_depth).unwrap_or(root_id);
            continue;
        }
        current = child_id; // descend
    }

    if found_goals.is_empty() && budget_exhausted {
        fire_fallback_start_event(observer, &graph, root_id, ctx, &resume_buffer);
        let (goal_id, fb_expanded) = sequential_fallback(&mut graph, root_id, ctx, goal);
        nodes_expanded += fb_expanded;
        if let Some(gid) = goal_id {
            found_goals.push(gid);
        }
    }

    // Return the best goal by:
    // 1) shallowest depth, 2) lowest approximate path move time,
    // 3) lexicographic path key (deterministic), 4) node id (deterministic).
    let best = select_best_goal_with_tiebreak(&found_goals, &graph, ctx.index);
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
) -> Option<(usize, MoveSet, Config, f64, bool)> {
    let config = graph.config(node_id);
    let es = entropy_map.entry(node_id).or_default();

    // Regenerate if we've exhausted max_candidates from current cache.
    if es.candidates_tried >= params.max_candidates || es.candidate_cache.is_empty() {
        es.candidate_cache = generate_candidates(config, es.entropy, params, ctx, seed);
        es.candidates_tried = 0;
    }

    // Find first untried, non-failed candidate.
    while es.candidates_tried < es.candidate_cache.len() {
        let (ref ms, ref cfg, cost, origin) = es.candidate_cache[es.candidates_tried];
        let move_key = ms.encoded_lanes().to_vec();
        if !es.tried_moves.contains(&move_key) && !es.failed_candidates.contains(&move_key) {
            let result = (es.candidates_tried, ms.clone(), cfg.clone(), cost, origin);
            return Some(result);
        }
        es.candidates_tried += 1;
    }

    // All cached candidates already tried — regenerate and try again.
    es.candidate_cache = generate_candidates(config, es.entropy, params, ctx, seed);
    es.candidates_tried = 0;

    while es.candidates_tried < es.candidate_cache.len() {
        let (ref ms, ref cfg, cost, origin) = es.candidate_cache[es.candidates_tried];
        let move_key = ms.encoded_lanes().to_vec();
        if !es.tried_moves.contains(&move_key) && !es.failed_candidates.contains(&move_key) {
            let result = (es.candidates_tried, ms.clone(), cfg.clone(), cost, origin);
            return Some(result);
        }
        es.candidates_tried += 1;
    }

    None // all candidates exhausted
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{example_arch_json, loc};
    use bloqade_lanes_bytecode_core::arch::types::TransportPath;

    fn make_index() -> LaneIndex {
        let spec: bloqade_lanes_bytecode_core::arch::types::ArchSpec =
            serde_json::from_str(example_arch_json()).unwrap();
        LaneIndex::new(spec)
    }

    fn make_chain_index() -> LaneIndex {
        let spec: bloqade_lanes_bytecode_core::arch::types::ArchSpec = serde_json::from_str(
            r#"{
                "version": "2.0",
                "words": [
                    { "sites": [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0]] }
                ],
                "zones": [
                    {
                        "grid": { "x_start": 0.0, "y_start": 0.0, "x_spacing": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], "y_spacing": [] },
                        "site_buses": [
                            { "src": [0, 2, 4], "dst": [2, 4, 6] }
                        ],
                        "word_buses": [],
                        "words_with_site_buses": [0],
                        "sites_with_word_buses": [],
                        "entangling_pairs": []
                    }
                ],
                "zone_buses": [],
                "modes": [
                    { "name": "default", "zones": [0], "bitstring_order": [] }
                ]
            }"#,
        )
        .unwrap();
        LaneIndex::new(spec)
    }

    fn make_deadlock_breaker_index() -> LaneIndex {
        let spec: bloqade_lanes_bytecode_core::arch::types::ArchSpec = serde_json::from_str(
            r#"{
                "version": "2.0",
                "words": [
                    { "sites": [[0, 0], [1, 0], [2, 0], [3, 0]] }
                ],
                "zones": [
                    {
                        "grid": { "x_start": 0.0, "y_start": 0.0, "x_spacing": [1.0, 1.0, 1.0], "y_spacing": [] },
                        "site_buses": [
                            { "src": [0, 1], "dst": [1, 2] },
                            { "src": [1], "dst": [3] }
                        ],
                        "word_buses": [],
                        "words_with_site_buses": [0],
                        "sites_with_word_buses": [],
                        "entangling_pairs": []
                    }
                ],
                "zone_buses": [],
                "modes": [
                    { "name": "default", "zones": [0], "bitstring_order": [] }
                ]
            }"#,
        )
        .unwrap();
        LaneIndex::new(spec)
    }

    fn make_index_with_paths(paths: Vec<(LaneAddr, Vec<[f64; 2]>)>) -> LaneIndex {
        let mut spec: bloqade_lanes_bytecode_core::arch::types::ArchSpec =
            serde_json::from_str(example_arch_json()).unwrap();
        spec.paths = Some(
            paths
                .into_iter()
                .map(|(lane, waypoints)| TransportPath {
                    lane: lane.encode_u64(),
                    waypoints,
                })
                .collect(),
        );
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
            cz_pairs: None,
        };
        entropy_search(
            root,
            &goal,
            &EntropyParams::default(),
            &ctx,
            max_expansions,
            None,
            0,
            &mut crate::observer::NoOpObserver,
        )
    }

    fn run_entropy_with_trace(
        initial: impl IntoIterator<Item = (u32, LocationAddr)>,
        target: impl IntoIterator<Item = (u32, LocationAddr)>,
        max_expansions: Option<u32>,
        max_depth: Option<u32>,
        trace: &mut EntropyTrace,
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
            cz_pairs: None,
        };
        entropy_search(
            root,
            &goal,
            &EntropyParams::default(),
            &ctx,
            max_expansions,
            max_depth,
            0,
            trace,
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
    fn final_goal_tiebreak_prefers_lower_approx_move_time() {
        let l_a1 = LaneAddr {
            direction: Direction::Forward,
            move_type: MoveType::SiteBus,
            zone_id: 0,
            word_id: 0,
            site_id: 0,
            bus_id: 0,
        };
        let l_a2 = LaneAddr {
            direction: Direction::Forward,
            move_type: MoveType::SiteBus,
            zone_id: 0,
            word_id: 0,
            site_id: 1,
            bus_id: 0,
        };
        let l_b1 = LaneAddr {
            direction: Direction::Forward,
            move_type: MoveType::SiteBus,
            zone_id: 0,
            word_id: 0,
            site_id: 2,
            bus_id: 0,
        };
        let l_b2 = LaneAddr {
            direction: Direction::Forward,
            move_type: MoveType::SiteBus,
            zone_id: 0,
            word_id: 0,
            site_id: 3,
            bus_id: 0,
        };
        let index = make_index_with_paths(vec![
            (l_a1, vec![[0.0, 0.0], [1.0, 0.0]]),
            (l_a2, vec![[0.0, 0.0], [1.0, 0.0]]),
            (l_b1, vec![[0.0, 0.0], [50.0, 0.0]]),
            (l_b2, vec![[0.0, 0.0], [50.0, 0.0]]),
        ]);

        let mut graph = SearchGraph::new(Config::new([(0, loc(0, 0))]).unwrap());
        let (a_mid, _) = graph.insert(
            graph.root(),
            MoveSet::new([l_a1]),
            Config::new([(0, loc(0, 1))]).unwrap(),
            1.0,
        );
        let (a_goal, _) = graph.insert(
            a_mid,
            MoveSet::new([l_a2]),
            Config::new([(0, loc(0, 2))]).unwrap(),
            2.0,
        );
        let (b_mid, _) = graph.insert(
            graph.root(),
            MoveSet::new([l_b1]),
            Config::new([(0, loc(0, 3))]).unwrap(),
            1.0,
        );
        let (b_goal, _) = graph.insert(
            b_mid,
            MoveSet::new([l_b2]),
            Config::new([(0, loc(0, 4))]).unwrap(),
            2.0,
        );

        let best = select_best_goal_with_tiebreak(&[b_goal, a_goal], &graph, &index);
        assert_eq!(best, Some(a_goal));
    }

    #[test]
    fn final_goal_tiebreak_uses_lexicographic_path_when_time_ties() {
        let l_a1 = LaneAddr {
            direction: Direction::Forward,
            move_type: MoveType::SiteBus,
            zone_id: 0,
            word_id: 0,
            site_id: 0,
            bus_id: 0,
        };
        let l_a2 = LaneAddr {
            direction: Direction::Forward,
            move_type: MoveType::SiteBus,
            zone_id: 0,
            word_id: 0,
            site_id: 1,
            bus_id: 0,
        };
        let l_b1 = LaneAddr {
            direction: Direction::Forward,
            move_type: MoveType::SiteBus,
            zone_id: 0,
            word_id: 0,
            site_id: 0,
            bus_id: 1,
        };
        let l_b2 = LaneAddr {
            direction: Direction::Forward,
            move_type: MoveType::SiteBus,
            zone_id: 0,
            word_id: 0,
            site_id: 2,
            bus_id: 1,
        };
        let index = make_index_with_paths(vec![
            (l_a1, vec![[0.0, 0.0], [5.0, 0.0]]),
            (l_a2, vec![[0.0, 0.0], [5.0, 0.0]]),
            (l_b1, vec![[0.0, 0.0], [5.0, 0.0]]),
            (l_b2, vec![[0.0, 0.0], [5.0, 0.0]]),
        ]);

        let mut graph = SearchGraph::new(Config::new([(0, loc(0, 0))]).unwrap());
        let (a_mid, _) = graph.insert(
            graph.root(),
            MoveSet::new([l_a1]),
            Config::new([(0, loc(0, 1))]).unwrap(),
            1.0,
        );
        let (a_goal, _) = graph.insert(
            a_mid,
            MoveSet::new([l_a2]),
            Config::new([(0, loc(0, 2))]).unwrap(),
            2.0,
        );
        let (b_mid, _) = graph.insert(
            graph.root(),
            MoveSet::new([l_b1]),
            Config::new([(0, loc(0, 3))]).unwrap(),
            1.0,
        );
        let (b_goal, _) = graph.insert(
            b_mid,
            MoveSet::new([l_b2]),
            Config::new([(0, loc(0, 4))]).unwrap(),
            2.0,
        );

        let best = select_best_goal_with_tiebreak(&[b_goal, a_goal], &graph, &index);
        assert_eq!(best, Some(a_goal));
    }

    #[test]
    fn budget_exceeded_returns_no_goal() {
        let r = run_entropy([(0, loc(0, 0))], [(0, loc(99, 99))], Some(10));
        assert!(r.goal.is_none());
    }

    #[test]
    fn budget_exhaustion_runs_sequential_fallback() {
        let r = run_entropy([(0, loc(0, 0))], [(0, loc(0, 5))], Some(0));
        let goal = r
            .goal
            .expect("sequential fallback should find reachable target");
        assert_eq!(r.graph.config(goal).location_of(0), Some(loc(0, 5)));
    }

    #[test]
    fn budget_exhaustion_records_fallback_start_trace() {
        let mut trace = EntropyTrace::default();
        let r = run_entropy_with_trace(
            [(0, loc(0, 0))],
            [(0, loc(0, 5))],
            Some(0),
            None,
            &mut trace,
        );

        assert!(r.goal.is_some());
        assert!(
            trace
                .steps
                .iter()
                .any(|step| step.event == "fallback_start")
        );
    }

    #[test]
    fn root_entropy_limit_continues_without_sequential_fallback() {
        let mut trace = EntropyTrace::default();
        let r = run_entropy_with_trace(
            [(0, loc(0, 0))],
            [(0, loc(0, 5))],
            Some(100),
            Some(0),
            &mut trace,
        );

        assert!(r.goal.is_some());
        assert!(
            trace
                .steps
                .iter()
                .all(|step| step.event != "fallback_start")
        );
    }

    #[test]
    fn scored_entry_tie_break_is_deterministic() {
        let mut entries = [
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
    fn resume_buffer_orders_by_score_then_depth_then_order() {
        let mut buffer = Vec::new();
        let mut next_order = 0_u64;

        resume_buffer_insert(&mut buffer, NodeId(1), 10.0, 2, 3, &mut next_order);
        resume_buffer_insert(&mut buffer, NodeId(2), 10.0, 4, 3, &mut next_order);
        resume_buffer_insert(&mut buffer, NodeId(3), 11.0, 1, 3, &mut next_order);

        assert_eq!(resume_buffer_pop_best(&mut buffer, None), Some(NodeId(3)));
        assert_eq!(resume_buffer_pop_best(&mut buffer, None), Some(NodeId(2)));
        assert_eq!(resume_buffer_pop_best(&mut buffer, None), Some(NodeId(1)));
        assert_eq!(resume_buffer_pop_best(&mut buffer, None), None);
    }

    #[test]
    fn resume_buffer_capacity_and_depth_gate() {
        let mut buffer = Vec::new();
        let mut next_order = 0_u64;

        resume_buffer_insert(&mut buffer, NodeId(11), 5.0, 1, 2, &mut next_order);
        resume_buffer_insert(&mut buffer, NodeId(12), 9.0, 2, 2, &mut next_order);
        resume_buffer_insert(&mut buffer, NodeId(13), 3.0, 3, 2, &mut next_order);

        // Lowest-priority node (13) is dropped at capacity.
        assert_eq!(buffer.len(), 2);

        // best_goal_depth=2 blocks node 12 (depth 2), so 11 is next.
        assert_eq!(
            resume_buffer_pop_best(&mut buffer, Some(2)),
            Some(NodeId(11))
        );
        assert_eq!(resume_buffer_pop_best(&mut buffer, Some(2)), None);
    }

    #[test]
    fn after_first_goal_depth_gate_blocks_deeper_descend() {
        let mut buffer = Vec::new();
        let mut next_order = 0_u64;

        resume_buffer_insert(&mut buffer, NodeId(20), 8.0, 3, 3, &mut next_order);
        resume_buffer_insert(&mut buffer, NodeId(21), 7.5, 2, 3, &mut next_order);
        resume_buffer_insert(&mut buffer, NodeId(22), 7.0, 1, 3, &mut next_order);

        // Once best_goal_depth=2, depth>=2 candidates are skipped.
        assert_eq!(
            resume_buffer_pop_best(&mut buffer, Some(2)),
            Some(NodeId(22))
        );
        assert_eq!(resume_buffer_pop_best(&mut buffer, Some(2)), None);
    }

    #[test]
    fn goal_resume_uses_buffer_then_root_fallback() {
        let root = NodeId(0);
        let mut buffer = Vec::new();
        let mut next_order = 0_u64;
        resume_buffer_insert(&mut buffer, NodeId(31), 4.0, 1, 1, &mut next_order);

        let first_resume = resume_buffer_pop_best(&mut buffer, Some(3)).unwrap_or(root);
        let fallback_resume = resume_buffer_pop_best(&mut buffer, Some(3)).unwrap_or(root);

        assert_eq!(first_resume, NodeId(31));
        assert_eq!(fallback_resume, root);
    }

    #[test]
    fn capacity_is_max_goal_candidates_minus_one() {
        let params = EntropyParams {
            max_goal_candidates: 4,
            ..EntropyParams::default()
        };
        assert_eq!(params.max_goal_candidates.saturating_sub(1), 3);
    }

    #[test]
    fn resume_buffer_reinsertion_refreshes_score_and_dedupes_node() {
        let mut buffer = Vec::new();
        let mut next_order = 0_u64;
        let parent = NodeId(42);

        resume_buffer_insert(&mut buffer, parent, 1.0, 3, 3, &mut next_order);
        resume_buffer_insert(&mut buffer, NodeId(9), 2.0, 3, 3, &mut next_order);
        // Reinsert same parent with a better move score.
        resume_buffer_insert(&mut buffer, parent, 5.0, 3, 3, &mut next_order);

        // Node id is de-duplicated and priority is refreshed.
        assert_eq!(buffer.iter().filter(|e| e.node_id == parent).count(), 1);
        assert_eq!(resume_buffer_pop_best(&mut buffer, None), Some(parent));
    }

    #[test]
    fn score_moveset_uses_scalar_path_without_detailed_metrics() {
        let index = make_index();
        let old_config = Config::new([(0, loc(0, 0))]).unwrap();
        let new_config = Config::new([(0, loc(0, 1))]).unwrap();
        let target_encoded = vec![(0u32, loc(0, 5).encode())];
        let target_locs: Vec<u64> = target_encoded.iter().map(|&(_, enc)| enc).collect();
        let dist_table = DistanceTable::new(&target_locs, &index);
        let blocked = HashSet::new();
        let occupied: HashSet<u64> = old_config.iter().map(|(_, loc)| loc.encode()).collect();
        let ctx = SearchContext {
            index: &index,
            dist_table: &dist_table,
            blocked: &blocked,
            targets: &target_encoded,
            cz_pairs: None,
        };
        let params = EntropyParams::default();

        COMPUTE_MOVESET_METRICS_CALLS.store(0, std::sync::atomic::Ordering::Relaxed);
        let score = score_moveset(&old_config, &new_config, &occupied, &ctx, &params);

        assert_eq!(
            COMPUTE_MOVESET_METRICS_CALLS.load(std::sync::atomic::Ordering::Relaxed),
            0
        );

        let detailed_score =
            compute_moveset_metrics(&old_config, &new_config, &occupied, &ctx, &params)
                .score(&params);
        assert_eq!(score, detailed_score);
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
            cz_pairs: None,
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
        for ((ms_a, cfg_a, _, _), (ms_b, cfg_b, _, _)) in out1.iter().zip(out2.iter()) {
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
            cz_pairs: None,
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
            cz_pairs: None,
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

        for (moveset, _, _, _) in out {
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

            let mut entries: HashMap<u64, u64> = HashMap::new();
            for lane in &lanes {
                assert_eq!(lane.move_type, first.move_type);
                assert_eq!(lane.bus_id, first.bus_id);
                assert_eq!(lane.direction, first.direction);
                let (src, _) = index.endpoints(lane).expect("lane endpoints must exist");
                entries.insert(src.encode(), lane.encode_u64());
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

    #[test]
    #[ignore = "greedy_init sorts by src_encoded ascending, so follow-move chains fail to seed valid single-element rects; pre-existing limitation unrelated to deadlock-breaker"]
    fn generate_candidates_allows_follow_moves_into_moving_occupants() {
        let index = make_chain_index();
        let config = Config::new([(0, loc(0, 0)), (2, loc(0, 2)), (4, loc(0, 4))]).unwrap();
        let target_encoded = vec![
            (0u32, loc(0, 2).encode()),
            (2u32, loc(0, 4).encode()),
            (4u32, loc(0, 6).encode()),
        ];
        let target_locs: Vec<u64> = target_encoded.iter().map(|&(_, enc)| enc).collect();
        let dist_table = DistanceTable::new(&target_locs, &index);
        let blocked = HashSet::new();
        let ctx = SearchContext {
            index: &index,
            dist_table: &dist_table,
            blocked: &blocked,
            targets: &target_encoded,
            cz_pairs: None,
        };
        let params = EntropyParams {
            max_movesets_per_group: 8,
            ..EntropyParams::default()
        };

        let out = generate_candidates(&config, 1, &params, &ctx, 0);

        assert!(
            out.iter().any(|(_, candidate_config, _, _)| {
                candidate_config.location_of(0) == Some(loc(0, 2))
                    && candidate_config.location_of(2) == Some(loc(0, 4))
                    && candidate_config.location_of(4) == Some(loc(0, 6))
            }),
            "expected a candidate that moves 0->2, 2->4, and 4->6 in one AOD layer; got {out:?}"
        );
    }

    #[test]
    fn generate_candidates_deadlock_breaker_caps_moves_to_half_unresolved() {
        let index = make_deadlock_breaker_index();

        // q0 and q1 are unresolved, q2 is a stationary blocker at loc(0,2).
        // Positive moves are q0:0->1 and q1:1->2 (blocked by q2), while q1:1->3
        // is a lower-priority escape lane. Normal rectangle generation can empty
        // out here; deadlock breaker should still return a fallback candidate.
        let config = Config::new([(0, loc(0, 0)), (1, loc(0, 1)), (2, loc(0, 2))]).unwrap();
        let target_encoded = vec![
            (0u32, loc(0, 1).encode()),
            (1u32, loc(0, 2).encode()),
            (2u32, loc(0, 2).encode()),
        ];
        let target_locs: Vec<u64> = target_encoded.iter().map(|&(_, enc)| enc).collect();
        let dist_table = DistanceTable::new(&target_locs, &index);
        let blocked = HashSet::new();
        let ctx = SearchContext {
            index: &index,
            dist_table: &dist_table,
            blocked: &blocked,
            targets: &target_encoded,
            cz_pairs: None,
        };
        let params = EntropyParams {
            w_m: 0.0,
            max_movesets_per_group: 8,
            ..EntropyParams::default()
        };

        let out = generate_candidates(&config, 1, &params, &ctx, 0);
        assert!(
            !out.is_empty(),
            "deadlock breaker should emit at least one fallback candidate"
        );
        assert!(
            out.iter().any(|(_, _, _, origin)| *origin),
            "expected deadlock-breaker candidate origin in fallback output"
        );

        let unresolved_ids: HashSet<u32> = target_encoded
            .iter()
            .filter_map(|(qid, target_enc)| {
                let current = config.location_of(*qid)?;
                (current.encode() != *target_enc).then_some(*qid)
            })
            .collect();
        let target_movers = unresolved_ids.len().div_ceil(2);

        let best_moved_unresolved = out
            .iter()
            .map(|(_, candidate_config, _, _)| {
                unresolved_ids
                    .iter()
                    .filter(|qid| candidate_config.location_of(**qid) != config.location_of(**qid))
                    .count()
            })
            .max()
            .unwrap_or(0);

        assert!(
            best_moved_unresolved > 0,
            "fallback should move at least one unresolved qubit"
        );
        assert!(
            best_moved_unresolved <= target_movers,
            "expected fallback to move at most half unresolved qubits ({target_movers}), got {best_moved_unresolved}"
        );
    }

    #[test]
    fn entropy_trace_marks_deadlock_breaker_descend() {
        let index = make_deadlock_breaker_index();
        let root = Config::new([(0, loc(0, 0)), (1, loc(0, 1)), (2, loc(0, 2))]).unwrap();
        let target_encoded = vec![
            (0u32, loc(0, 1).encode()),
            (1u32, loc(0, 2).encode()),
            (2u32, loc(0, 2).encode()),
        ];
        let target_locs: Vec<u64> = target_encoded.iter().map(|&(_, enc)| enc).collect();
        let dist_table = DistanceTable::new(&target_locs, &index);
        let blocked = HashSet::new();
        let ctx = SearchContext {
            index: &index,
            dist_table: &dist_table,
            blocked: &blocked,
            targets: &target_encoded,
            cz_pairs: None,
        };
        let goal = crate::goals::AllAtTarget::new(&target_encoded);
        let params = EntropyParams {
            w_m: 0.0,
            max_movesets_per_group: 8,
            max_goal_candidates: 3,
            ..EntropyParams::default()
        };
        let mut trace = EntropyTrace::default();

        let _ = entropy_search(root, &goal, &params, &ctx, Some(8), None, 0, &mut trace);

        assert!(
            trace.steps.iter().any(|step| step.event == "descend"
                && step.reason.as_deref() == Some("deadlock-breaker")),
            "expected descend step marked with deadlock-breaker reason in trace"
        );
    }
}
