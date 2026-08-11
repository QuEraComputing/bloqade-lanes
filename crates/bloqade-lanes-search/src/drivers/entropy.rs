//! Entropy-guided search for move synthesis.
//!
//! Port of the Python `EntropyGuidedSearch` algorithm. Walks a single path
//! down the search tree, using per-node entropy to shift scoring from
//! distance-focused (low entropy) to mobility-focused (high entropy).
//! Backtracks by walking parent pointers when entropy exceeds a threshold,
//! and falls back to greedy single-qubit routing when fully stuck.
//!
//! Mostly self-contained: [`entropy_search`] builds all required
//! infrastructure internally. Removing the driver touches this file plus its
//! references in `lib.rs`, the solver dispatch (`search/restarts.rs`,
//! `search/target_solver.rs`, `placement/loose_goal.rs`), the
//! [`BlendedColumnCache`] field on `search/engine.rs`, and the Python
//! bindings.

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::bounds::{CompletionBound, NoBound};
use crate::cost::UniformCost;
use crate::drivers::result::SearchResult;
use crate::feasibility::graph::LaneGraph;
use crate::observer::{SearchEvent, SearchObserver};
use crate::ops::aod_grid::{BusGridContext, ChainLink, close_chain_entries};
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
use crate::push_rotate::{DEFAULT_MOVE_BUDGET, plan as push_rotate_plan};
use crate::traits::{Goal, Objective};
use bloqade_lanes_bytecode_core::arch::addr::{LaneAddr, LocationAddr};
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
    // Generator settings.
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

/// A generated candidate cached on its node.
///
/// `score` is the unperturbed level-2 moveset score, cached at generation
/// time so the resume buffer can rank untried candidates without re-running
/// `score_moveset` on every descend.
///
/// Deliberately carries **no cost**: candidate generation is the entropy
/// heuristic's business, and cost belongs to the [`Objective`]. The driver
/// prices a candidate through `objective.edge_cost` at insert time, so `g` has
/// exactly one source of truth.
#[derive(Debug, Clone)]
pub(crate) struct CandidateEntry {
    pub(crate) move_set: MoveSet,
    pub(crate) new_config: Config,
    pub(crate) deadlock_breaker: bool,
    pub(crate) score: f64,
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

/// Turn the conveyor followers [`close_chain_entries`] pulled in into scored
/// entries, so their moves get recorded in the candidate's config.
///
/// The score is `0.0` deliberately. A follower is not a chosen move — it is
/// what the chosen move costs — and these entry scores only order candidates
/// *within* a bus group before the `max_movesets_per_group` truncation. The
/// authoritative ranking is [`score_moveset`], which reads the whole
/// old-config → new-config delta and therefore already prices every follower's
/// displacement, arrival, and mobility change. Synthesizing an entry score here
/// would double-count it.
fn chain_scored_entries(links: &[ChainLink]) -> impl Iterator<Item = ScoredEntry> + '_ {
    links.iter().map(|link| ScoredEntry {
        qubit_id: link.qubit_id,
        score: 0.0,
        lane_encoded: link.lane_encoded,
        dst_encoded: link.dst_encoded,
    })
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
    for (
        TripletKey {
            move_type: mt,
            bus_id,
            direction: dir,
        },
        mut qubits,
    ) in groups
    {
        qubits.sort_by(cmp_group_entries);
        let grid_ctx = BusGridContext::new(ctx.index, mt, bus_id, None, dir, occupied);

        let mut entries: HashMap<u64, u64> = HashMap::new();
        let mut entry_by_lane: HashMap<u64, ScoredEntry> = HashMap::new();
        let mut seed_lanes: Vec<u64> = Vec::new();
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
                seed_lanes.push(t.lane_encoded);
                if unresolved.contains(&t.qubit_id) {
                    selected_unresolved += 1;
                }
            }
        }

        if entries.is_empty() {
            continue;
        }

        // Conveyor followers ride along (#910). They are exempt from the
        // `target_movers` cap on purpose: the cap limits how many atoms this
        // breaker *chooses* to move, and a follower is not a choice — without
        // it the mover ahead has nowhere to go and the group emits nothing.
        for entry in chain_scored_entries(&close_chain_entries(
            &mut entries,
            &seed_lanes,
            occupied,
            config,
            ctx.index,
        )) {
            entry_by_lane.insert(entry.lane_encoded, entry);
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
    /// `g + h` for this node: accumulated objective cost plus the completion
    /// bound, fixed at insertion time because both are functions of the node's
    /// configuration alone. Used *only* by the incumbent gate in
    /// [`resume_buffer_pop_best`] — never by the ordering below. Equals `g`
    /// exactly when bounding is disabled.
    f: f64,
    depth: u32,
    order: u64,
}

fn cmp_resume_states(a: &ScoredResumeState, b: &ScoredResumeState) -> Ordering {
    b.score
        .total_cmp(&a.score)
        .then_with(|| b.depth.cmp(&a.depth))
        .then_with(|| b.order.cmp(&a.order))
}

#[allow(clippy::too_many_arguments)]
fn resume_buffer_insert(
    buffer: &mut Vec<ScoredResumeState>,
    node_id: NodeId,
    score: f64,
    f: f64,
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
        f,
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

/// Pop the highest-priority resumable node, skipping any that the incumbent
/// already dominates.
///
/// `best_cost` is the incumbent objective cost `C`. A buffered node whose
/// `g + h` reaches `C` cannot lead to a strictly cheaper solution, so it is
/// dropped rather than returned. Ordering is untouched by the gate — the
/// priority comparison stays `(score, depth, order)`.
fn resume_buffer_pop_best(
    buffer: &mut Vec<ScoredResumeState>,
    best_cost: Option<f64>,
) -> Option<NodeId> {
    loop {
        let best_idx = buffer
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| cmp_resume_states(a, b))
            .map(|(idx, _)| idx)?;
        let best = buffer.swap_remove(best_idx);
        if let Some(cost_cap) = best_cost
            && best.f >= cost_cap
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

/// Pick the best goal: lowest objective cost, then the deterministic
/// tiebreaks.
///
/// The primary key is `g_score` — the accumulated [`Objective`] cost — not
/// depth. Under [`UniformCost`](crate::cost::UniformCost) the two are equal for
/// every node, so this selects exactly what the previous min-depth rule did.
/// Move time stays a *tiebreak* among equal-cost goals rather than being folded
/// into the objective.
fn select_best_goal_with_tiebreak(
    found_goals: &[NodeId],
    graph: &SearchGraph,
    index: &LaneIndex,
) -> Option<NodeId> {
    let min_cost = found_goals
        .iter()
        .map(|&id| graph.g_score(id))
        .min_by(f64::total_cmp)?;
    found_goals
        .iter()
        .copied()
        .filter(|&id| graph.g_score(id).total_cmp(&min_cost) == Ordering::Equal)
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

fn best_untried_moveset_score(es: &EntropyState) -> Option<f64> {
    es.candidate_cache
        .iter()
        .filter_map(|entry| {
            let move_key = entry.move_set.encoded_lanes();
            if es.tried_moves.contains(move_key) || es.failed_candidates.contains(move_key) {
                return None;
            }
            // Cached at generation time; identical to re-running
            // `score_moveset` because the score depends only on the node's
            // config (fixed for this cache) and per-solve context.
            Some(entry.score)
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

/// Static (occupancy-independent) mobility of `loc_enc` toward `target_enc`:
/// sum of `1/(1+d)` over outgoing lanes whose destination is unblocked and can
/// reach the target. A pure function of `(location, target)` for a fixed solve
/// — atom positions never enter, only the static `blocked` set — which is what
/// makes [`HeuristicTables`] a valid memoization.
fn static_mobility(loc_enc: u64, target_enc: u64, ctx: &SearchContext, w_t: f64) -> f64 {
    let loc = LocationAddr::decode(loc_enc);
    let mut m = 0.0_f64;
    for &lane in ctx.index.outgoing_lanes(loc) {
        let Some((_, dst)) = ctx.index.endpoints(&lane) else {
            continue;
        };
        let dst_e = dst.encode();
        if ctx.blocked.contains(&dst_e) {
            continue;
        }
        let d = ctx
            .dist_table
            .distance(dst_e, target_enc)
            .map_or(f64::MAX, |d| {
                blended_distance(d as f64, dst_e, target_enc, w_t, ctx.dist_table)
            });
        if d < f64::MAX {
            m += 1.0 / (1.0 + d);
        }
    }
    m
}

/// Minimum blended distance to `target_enc` over the unblocked successors of
/// `loc_enc`. Returns `f64::MAX` when no successor reaches the target. Pure
/// per solve, like [`static_mobility`]. Used for 2-step lookahead.
fn min_successor_distance(loc_enc: u64, target_enc: u64, ctx: &SearchContext, w_t: f64) -> f64 {
    let loc = LocationAddr::decode(loc_enc);
    let mut best = f64::MAX;
    for &lane in ctx.index.outgoing_lanes(loc) {
        let Some((_, dst)) = ctx.index.endpoints(&lane) else {
            continue;
        };
        let dst_e = dst.encode();
        if ctx.blocked.contains(&dst_e) {
            continue;
        }
        let d = ctx
            .dist_table
            .distance(dst_e, target_enc)
            .map_or(f64::MAX, |d| {
                blended_distance(d as f64, dst_e, target_enc, w_t, ctx.dist_table)
            });
        if d < f64::MAX {
            best = best.min(d);
        }
    }
    best
}

/// The `w_t`-independent inputs to a blended-distance column: one hop count
/// and one normalized time per interned location.
///
/// [`blended_distance`] mixes these as
/// `(1 - w_t) * hop + w_t * (time / fastest_lane)`. Both terms are pure
/// architecture data, so storing them unmixed lets any `w_t` be derived at
/// fill time from a single cached column — see [`Self::blend`].
struct DistanceColumn {
    /// Hop distance per interned row. `u32::MAX` = unreachable, matching
    /// `DistanceTable::distance` returning `None`.
    hops: Vec<u32>,
    /// `time / fastest_lane` per interned row. `f64::NAN` = no time distance
    /// for that pair (a real value is always finite and non-negative:
    /// durations are positive and the fastest lane is arch-global), which
    /// makes [`Self::blend`] fall back to the hop count exactly as
    /// [`blended_distance`] does.
    norm_times: Vec<f64>,
}

impl DistanceColumn {
    /// Reconstruct one blended distance. Mirrors [`blended_distance`]'s
    /// branches and arithmetic exactly, so a value derived here is
    /// bit-identical to one computed directly from the distance table.
    #[inline]
    fn blend(&self, row: usize, w_t: f64) -> f64 {
        let hop = self.hops[row];
        if hop == u32::MAX {
            return f64::MAX; // unreachable — the direct path leaves MAX here
        }
        let hop_dist = hop as f64;
        if w_t <= 0.0 {
            return hop_dist;
        }
        let normalized_time_d = self.norm_times[row];
        if normalized_time_d.is_nan() {
            return hop_dist;
        }
        (1.0 - w_t) * hop_dist + w_t * normalized_time_d
    }
}

/// Engine-lifetime cache of per-target distance columns, keyed by target
/// location.
///
/// A column is a pure function of the architecture alone — `DistanceTable`
/// ignores occupancy and `blocked` entirely — so columns computed during one
/// solve are valid for every later solve on the same engine. The physical
/// pipeline issues one solve per candidate target layout per CZ layer, and
/// the target *locations* (entangling slots, home sites) recur heavily across
/// those solves; caching the columns turns the per-solve `d_blend` fill from
/// `n_loc × n_targets` distance evaluations into a arithmetic pass over
/// cached rows after first touch.
///
/// Columns store the *unmixed* [`DistanceColumn`] components rather than
/// blended values, so `w_t` — a continuous weight, not an enum — stays out of
/// the key: one column set serves every weight, and the memory bound below
/// holds regardless of how many weights an engine sees.
///
/// Lives in [`SearchEngine`](crate::search::engine::SearchEngine) behind a
/// `OnceLock`; shared by every `TargetSolver` cloned from that engine.
///
/// Memory: 12 bytes per interned location per column (a `u32` hop and an
/// `f64` normalized time), and at most one column per target location ever
/// solved for, so the cache is bounded by `n_locations²  × 12` bytes with no
/// eviction needed — about 300 KB fully populated on the physical Gemini
/// spec's 160 locations.
pub(crate) struct BlendedColumnCache {
    /// Shared location interning — every lane endpoint, fixed per arch.
    /// Tables built through this cache adopt it so cached columns can be
    /// read by row index.
    interner: Arc<HashMap<u64, u32>>,
    /// `target_enc` → column indexed by `interner` rows.
    cols: std::sync::RwLock<ColumnMap>,
}

/// Cached distance columns keyed by target location.
type ColumnMap = HashMap<u64, Arc<DistanceColumn>>;

impl BlendedColumnCache {
    pub(crate) fn new(index: &LaneIndex) -> Self {
        let mut interner: HashMap<u64, u32> = HashMap::with_capacity(index.num_locations());
        for loc_enc in index.lane_endpoint_encs() {
            let next = interner.len() as u32;
            interner.entry(loc_enc).or_insert(next);
        }
        Self {
            interner: Arc::new(interner),
            cols: std::sync::RwLock::new(HashMap::new()),
        }
    }
}

/// Per-solve memoization of the occupancy-independent parts of the level-1
/// scoring heuristic: blended distance, static mobility, and lookahead
/// minima, all keyed `(location, target)`.
///
/// Everything here is a pure function of `(targets, blocked, w_t)` — fixed
/// for one `entropy_search` call — computed with the same formulas, gates,
/// and successor iteration order as the fallback path's free functions
/// ([`blended_distance`], [`static_mobility`], [`min_successor_distance`]),
/// so table reads are bit-identical to direct computation and search
/// behavior is unchanged. The `generate_candidates_tables_match_direct_computation`
/// test enforces this.
///
/// Sizing: `n_locations × n_targets` f64 entries per table (~tens of KB on
/// the physical Gemini spec); build cost is comparable to a single node
/// expansion, and [`Self::build_cached`] amortizes the distance columns
/// across solves via [`BlendedColumnCache`].
pub(crate) struct HeuristicTables {
    loc_idx: Arc<HashMap<u64, u32>>,
    /// Per-solve overlay rows for targets absent from the endpoint interner:
    /// `DistanceTable` interns isolated targets (no incident lanes) so
    /// `distance(t, t) = 0` works, and these rows mirror that, keeping table
    /// lookups aligned with the direct path on every location the distance
    /// table can answer for. Empty on typical architectures.
    extra_idx: HashMap<u64, u32>,
    target_col: HashMap<u64, u32>,
    n_targets: usize,
    /// The `w_t` the tables were built with; must match the consuming
    /// `EntropyParams::w_t` (debug-asserted at the read sites).
    w_t: f64,
    /// Blended distance, `f64::MAX` = unreachable.
    d_blend: Vec<f64>,
    /// Static (blocked-only) mobility.
    mobility: Vec<f64>,
    /// Min blended distance over unblocked successors (`f64::MAX` = none).
    /// Only built when lookahead scoring is enabled.
    lookahead_min: Option<Vec<f64>>,
}

impl HeuristicTables {
    /// Build without a cross-solve cache (tests, bench, trait paths).
    pub(crate) fn build(ctx: &SearchContext, w_t: f64, lookahead: bool) -> Self {
        Self::build_inner(ctx, w_t, lookahead, None)
    }

    /// Build reusing engine-cached blended columns where available.
    pub(crate) fn build_cached(
        ctx: &SearchContext,
        w_t: f64,
        lookahead: bool,
        cache: &BlendedColumnCache,
    ) -> Self {
        Self::build_inner(ctx, w_t, lookahead, Some(cache))
    }

    fn build_inner(
        ctx: &SearchContext,
        w_t: f64,
        lookahead: bool,
        cache: Option<&BlendedColumnCache>,
    ) -> Self {
        let mut target_col: HashMap<u64, u32> = HashMap::new();
        for &(_, t_enc) in ctx.targets {
            let next = target_col.len() as u32;
            target_col.entry(t_enc).or_insert(next);
        }
        let n_targets = target_col.len();

        // A `w_t > 0` solve whose distance table carries no time data blends
        // nothing — `blended_distance` silently degrades to hop counts. The
        // cache tolerates it (such columns are never stored, see
        // `time_definitive` below), but the solve itself is misconfigured.
        debug_assert!(
            w_t <= 0.0
                || ctx.index.fastest_lane_duration_us().is_none()
                || ctx.dist_table.fastest_lane_us().is_some(),
            "w_t > 0 requires a DistanceTable built with_time_distances when \
             the arch has lane durations"
        );

        // Interned rows cover every lane endpoint — with a cache, adopt its
        // interner so cached columns copy by row. `DistanceTable` additionally
        // interns isolated targets (no incident lanes); give those per-solve
        // overlay rows below so lookups never miss where the direct path hits.
        let loc_idx: Arc<HashMap<u64, u32>> = match cache {
            Some(c) => c.interner.clone(),
            None => {
                let mut m: HashMap<u64, u32> = HashMap::with_capacity(ctx.index.num_locations());
                for loc_enc in ctx.index.lane_endpoint_encs() {
                    let next = m.len() as u32;
                    m.entry(loc_enc).or_insert(next);
                }
                Arc::new(m)
            }
        };
        let n_loc = loc_idx.len();
        let mut extra_idx: HashMap<u64, u32> = HashMap::new();
        for &t_enc in target_col.keys() {
            if !loc_idx.contains_key(&t_enc) {
                let next = (n_loc + extra_idx.len()) as u32;
                extra_idx.entry(t_enc).or_insert(next);
            }
        }
        let n_rows = n_loc + extra_idx.len();

        // A column's time component is trustworthy for later solves only if
        // this table actually carries time distances, or the arch has no lane
        // durations at all so no table ever could. Otherwise the all-NaN
        // component we'd compute here is an artifact of *this* table and must
        // not be cached, or a later `w_t > 0` solve would silently read hop
        // counts. This makes the invariant structural rather than asserted.
        let time_definitive = ctx.dist_table.fastest_lane_us().is_some()
            || ctx.index.fastest_lane_duration_us().is_none();

        let mut d_blend = vec![f64::MAX; n_rows * n_targets];
        for (&t_enc, &tc) in &target_col {
            let cached = cache.and_then(|c| c.cols.read().expect("poisoned").get(&t_enc).cloned());
            let column = match cached {
                Some(column) => column,
                None => {
                    // Compute the column's `w_t`-independent components from
                    // this solve's distance table. `DistanceTable` ignores
                    // occupancy/blocked and shortest distances are unique, so
                    // a column computed now is bit-identical for any later
                    // solve sharing the target location — at any `w_t`.
                    let fastest = ctx.dist_table.fastest_lane_us();
                    let mut hops = vec![u32::MAX; n_loc];
                    let mut norm_times = vec![f64::NAN; n_loc];
                    for (&loc_enc, &li) in loc_idx.iter() {
                        if let Some(d) = ctx.dist_table.distance(loc_enc, t_enc) {
                            hops[li as usize] = d;
                        }
                        if let (Some(time_d), Some(f)) =
                            (ctx.dist_table.time_distance(loc_enc, t_enc), fastest)
                        {
                            norm_times[li as usize] = time_d / f;
                        }
                    }
                    let column = Arc::new(DistanceColumn { hops, norm_times });
                    if let Some(c) = cache
                        && time_definitive
                    {
                        c.cols
                            .write()
                            .expect("poisoned")
                            .entry(t_enc)
                            .or_insert_with(|| column.clone());
                    }
                    column
                }
            };
            for li in 0..n_loc {
                d_blend[li * n_targets + tc as usize] = column.blend(li, w_t);
            }
            // Overlay rows (isolated targets) are never in cached columns —
            // fill them directly, matching what the direct path computes.
            for (&loc_enc, &li) in &extra_idx {
                if let Some(d) = ctx.dist_table.distance(loc_enc, t_enc) {
                    d_blend[li as usize * n_targets + tc as usize] =
                        blended_distance(d as f64, loc_enc, t_enc, w_t, ctx.dist_table);
                }
            }
        }

        // Mobility and lookahead are folds over each location's unblocked
        // successors' blended distances — read them from the freshly built
        // `d_blend` rows rather than re-deriving each through the distance
        // tables (`d_blend` stores exactly what [`static_mobility`] /
        // [`min_successor_distance`] would recompute, in the same successor
        // order, so the folds are bit-identical). Successor resolution is
        // hoisted out of the target loop; this matters because the physical
        // pipeline issues many short solves and each pays the build.
        // Overlay rows keep the defaults (mobility 0.0, lookahead MAX):
        // isolated targets have no outgoing lanes, matching the direct path.
        let mut mobility = vec![0.0_f64; n_rows * n_targets];
        let mut lookahead_min = lookahead.then(|| vec![f64::MAX; n_rows * n_targets]);
        let mut successor_rows: Vec<u32> = Vec::new();
        for (&loc_enc, &li) in loc_idx.iter() {
            successor_rows.clear();
            for &lane in ctx.index.outgoing_lanes(LocationAddr::decode(loc_enc)) {
                let Some((_, dst)) = ctx.index.endpoints(&lane) else {
                    continue;
                };
                let dst_e = dst.encode();
                if ctx.blocked.contains(&dst_e) {
                    continue;
                }
                // Every lane endpoint is interned above, so the lookup holds.
                successor_rows.push(loc_idx[&dst_e]);
            }

            let row = li as usize * n_targets;
            for tc in 0..n_targets {
                let mut m = 0.0_f64;
                let mut best = f64::MAX;
                for &dst_li in &successor_rows {
                    let d = d_blend[dst_li as usize * n_targets + tc];
                    if d < f64::MAX {
                        m += 1.0 / (1.0 + d);
                        best = best.min(d);
                    }
                }
                mobility[row + tc] = m;
                if let Some(la) = &mut lookahead_min {
                    la[row + tc] = best;
                }
            }
        }

        Self {
            loc_idx,
            extra_idx,
            target_col,
            n_targets,
            w_t,
            d_blend,
            mobility,
            lookahead_min,
        }
    }

    /// Resolve a target's column index. `None` if the location was never a
    /// target of this solve.
    #[inline]
    fn col(&self, target_enc: u64) -> Option<usize> {
        self.target_col.get(&target_enc).map(|&c| c as usize)
    }

    /// Resolve a location's row index: interned lane endpoints first, then
    /// the per-solve overlay rows for isolated targets.
    #[inline]
    fn row(&self, loc_enc: u64) -> Option<u32> {
        self.loc_idx
            .get(&loc_enc)
            .or_else(|| self.extra_idx.get(&loc_enc))
            .copied()
    }

    /// Blended distance by pre-resolved column. `f64::MAX` = unreachable or
    /// unknown location.
    #[inline]
    fn d(&self, loc_enc: u64, col: usize) -> f64 {
        match self.row(loc_enc) {
            Some(li) => self.d_blend[li as usize * self.n_targets + col],
            None => f64::MAX,
        }
    }

    /// Static mobility by pre-resolved column. `0.0` for unknown locations
    /// (no outgoing lanes).
    #[inline]
    fn m(&self, loc_enc: u64, col: usize) -> f64 {
        match self.row(loc_enc) {
            Some(li) => self.mobility[li as usize * self.n_targets + col],
            None => 0.0,
        }
    }

    /// Whether the lookahead table was built. Callers needing lookahead with
    /// tables built without it must fall back to direct computation.
    #[inline]
    fn has_lookahead(&self) -> bool {
        self.lookahead_min.is_some()
    }

    /// Lookahead minimum by pre-resolved column. `f64::MAX` for unknown
    /// locations or when no successor reaches the target. Only meaningful
    /// when [`Self::has_lookahead`] — guard at the call site.
    #[inline]
    fn la(&self, loc_enc: u64, col: usize) -> f64 {
        match (&self.lookahead_min, self.row(loc_enc)) {
            (Some(la), Some(li)) => la[li as usize * self.n_targets + col],
            _ => f64::MAX,
        }
    }
}

/// Guard the by-convention coupling between prebuilt tables and the params
/// consuming them: a `w_t` mismatch would silently blend distances with the
/// build-time weight while the fallback path uses the runtime weight.
#[inline]
fn debug_assert_tables_match(tables: Option<&HeuristicTables>, params: &EntropyParams) {
    if let Some(tb) = tables {
        debug_assert!(
            tb.w_t.to_bits() == params.w_t.to_bits(),
            "HeuristicTables built with w_t={} but consumed with w_t={}",
            tb.w_t,
            params.w_t
        );
    }
}

/// Branch-and-bound test: can `node` still lead to something strictly better
/// than the incumbent?
///
/// Returns `true` when the branch is provably not worth exploring, either
/// because the bound proves no completion exists at all (`h = +∞`) or because
/// `g + h` already reaches the incumbent cost `C`. Ties are cut: an equal-cost
/// completion adds nothing once one is held.
///
/// `h` is the *unweighted* admissible estimate. Ordering elsewhere in this
/// driver may be perturbed or reweighted; a pruning decision may not be.
///
/// With a [`TRIVIAL`](CompletionBound::TRIVIAL) bound the `h` term folds to a
/// constant `0.0` at monomorphization, leaving exactly the `g >= C` test this
/// generalizes — the bound-disabled path is the same code, not merely
/// equivalent code.
#[inline]
fn is_pruned<B: CompletionBound>(
    graph: &SearchGraph,
    node: NodeId,
    best_cost: Option<f64>,
    bound: &B,
) -> bool {
    let h = if B::TRIVIAL {
        0.0
    } else {
        bound.estimate(graph.config(node))
    };
    if h.is_infinite() {
        return true; // infeasible regardless of any incumbent
    }
    best_cost.is_some_and(|cost_cap| graph.g_score(node) + h >= cost_cap)
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
    tables: Option<&HeuristicTables>,
) -> Vec<CandidateEntry> {
    assert!(
        params.max_movesets_per_group > 0,
        "max_movesets_per_group must be > 0"
    );
    debug_assert_tables_match(tables, params);

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
    //
    // The per-(location, target) quantities below — blended distance, static
    // mobility, lookahead minimum — are pure per solve (they filter on
    // `blocked` only, never on atom positions). With `tables` present they
    // are flat-array reads; the fallback evaluates the same free functions
    // directly, producing bit-identical values.

    for &(qid, loc_enc, target_enc) in &unresolved {
        let tcol = tables.and_then(|tb| tb.col(target_enc));
        let d_of = |enc: u64| -> f64 {
            match (tables, tcol) {
                (Some(tb), Some(c)) => tb.d(enc, c),
                _ => dist_table.distance(enc, target_enc).map_or(f64::MAX, |d| {
                    blended_distance(d as f64, enc, target_enc, params.w_t, dist_table)
                }),
            }
        };
        let m_of = |enc: u64| -> f64 {
            match (tables, tcol) {
                (Some(tb), Some(c)) => tb.m(enc, c),
                _ => static_mobility(enc, target_enc, ctx, params.w_t),
            }
        };
        let la_of = |enc: u64| -> f64 {
            match (tables, tcol) {
                // Tables built without lookahead (params mismatch) must not
                // silently mask the term — fall back to direct computation.
                (Some(tb), Some(c)) if tb.has_lookahead() => tb.la(enc, c),
                _ => min_successor_distance(enc, target_enc, ctx, params.w_t),
            }
        };

        let d_now = d_of(loc_enc);
        if d_now == f64::MAX {
            continue;
        }
        let m_now = m_of(loc_enc);

        let loc = LocationAddr::decode(loc_enc);
        for &lane in index.outgoing_lanes(loc) {
            let Some((_, dst)) = index.endpoints(&lane) else {
                continue;
            };
            let dst_enc = dst.encode();
            if blocked.contains(&dst_enc) {
                continue;
            }
            let d_after = d_of(dst_enc);
            let m_after = m_of(dst_enc);
            let effective_d_after = if params.lookahead {
                d_after.min(la_of(dst_enc))
            } else {
                d_after
            };
            let delta_d = d_now - effective_d_after;
            let delta_m = m_after - m_now;

            let triplet_key = TripletKey::new(lane.move_type, lane.bus_id, lane.direction);
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

    for (
        TripletKey {
            move_type: mt,
            bus_id,
            direction: dir,
        },
        mut qubits,
    ) in groups
    {
        qubits.sort_by(cmp_group_entries);

        let grid_ctx = BusGridContext::new(ctx.index, mt, bus_id, None, dir, &occupied);

        let mut entries: HashMap<u64, u64> = HashMap::new();
        let mut entry_by_lane: HashMap<u64, ScoredEntry> = HashMap::new();
        let mut seed_lanes: Vec<u64> = Vec::with_capacity(qubits.len());
        for t in &qubits {
            let lane = LaneAddr::decode_u64(t.lane_encoded);
            if let Some((src, _)) = ctx.index.endpoints(&lane) {
                let src_enc = src.encode();
                entries.insert(src_enc, t.lane_encoded);
                entry_by_lane.insert(t.lane_encoded, *t);
                seed_lanes.push(t.lane_encoded);
            }
        }

        // Co-select the conveyor followers a selected mover has to displace, to
        // the end of the chain (#910). Without them the leader's rectangle is
        // unexecutable and a packed block yields no candidate at all.
        for entry in chain_scored_entries(&close_chain_entries(
            &mut entries,
            &seed_lanes,
            &occupied,
            config,
            ctx.index,
        )) {
            entry_by_lane.insert(entry.lane_encoded, entry);
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
    // The unperturbed score rides along so it can be cached on the entry.
    let mut scored: Vec<(f64, f64, MoveSet, Config)> = candidates
        .into_iter()
        .map(|(_raw_score, ms, new_cfg)| {
            let ms_score = score_moveset(config, &new_cfg, &occupied, ctx, params, tables);
            let ms_perturbation = rng.as_mut().map_or(0.0, |r| r.random_range(-0.5..0.5));
            (ms_score + ms_perturbation, ms_score, ms, new_cfg)
        })
        .collect();
    scored.sort_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then_with(|| cmp_moveset_config_tiebreak(&a.2, &a.3, &b.2, &b.3))
    });

    scored
        .into_iter()
        .map(|(_, score, move_set, new_config)| CandidateEntry {
            move_set,
            new_config,
            deadlock_breaker: used_deadlock_breaker,
            score,
        })
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
///
/// Unlike the level-1 generator mobility, the mobility terms here filter on
/// full occupancy (atoms included), so only the distance evaluations are
/// table-backed; the occupancy checks stay at call time.
pub(crate) fn score_moveset(
    old_config: &Config,
    new_config: &Config,
    occupied: &HashSet<u64>,
    ctx: &SearchContext,
    params: &EntropyParams,
    tables: Option<&HeuristicTables>,
) -> f64 {
    debug_assert_tables_match(tables, params);
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

        // Resolve the target column once per qubit — `d_of` runs per
        // successor in the mobility loops, and re-resolving the column
        // there doubles the hash probes in the hottest loop of the solve.
        let tcol = tables.and_then(|tb| tb.col(target_enc));
        let d_of = |enc: u64| -> Option<f64> {
            if let (Some(tb), Some(c)) = (tables, tcol) {
                let d = tb.d(enc, c);
                (d < f64::MAX).then_some(d)
            } else {
                // No tables, or a target missing from this table's columns
                // (tables built from a different target set): fall back to
                // direct computation, matching `generate_candidates`.
                dist_table
                    .distance(enc, target_enc)
                    .map(|d| blended_distance(d as f64, enc, target_enc, params.w_t, dist_table))
            }
        };

        let d_before = d_of(old_loc.encode()).unwrap_or(0.0);
        let d_after = d_of(new_loc.encode()).unwrap_or(0.0);
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
            if let Some(d) = d_of(dst_enc) {
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
            if let Some(d) = d_of(dst_enc) {
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
    if !observer.wants_events() {
        return;
    }
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

/// Budget-exhaustion fallback: Push and Rotate first, greedy sequential as
/// the out-of-regime tertiary.
///
/// The planner is complete for instances with ≥ 2 empties per moving
/// component and can displace atoms out of each other's way, so it solves
/// everything the greedy router can and more (permutations, congestion).
/// The greedy router's residual value is narrow and worth stating exactly:
/// it cannot enter an occupied vertex, so at `m = 0` it never succeeds, and
/// at `m = 1` it can only realize direct slides into the unique hole (any
/// longer path would need a second empty). That single-hole slide is the
/// one case the planner refuses (`TooFewEmpty`) that remains solvable here,
/// so keeping the tertiary makes this a strict behavioural superset of the
/// old greedy-only fallback.
fn budget_exhaustion_fallback(
    graph: &mut SearchGraph,
    start: NodeId,
    ctx: &SearchContext,
    goal: &impl Goal,
    objective: &impl Objective,
) -> (Option<NodeId>, u32) {
    let config = graph.config(start).clone();
    let lane_graph = LaneGraph::build(ctx.index, ctx.blocked);

    // Any location off the carved graph makes the instance inexpressible for
    // the planner; let the greedy router report what it can.
    let initial_v: Option<Vec<(u32, usize)>> = config
        .iter()
        .map(|(q, loc)| lane_graph.vertex_of(loc.encode()).map(|v| (q, v)))
        .collect();
    let target_v: Option<Vec<(u32, usize)>> = ctx
        .targets
        .iter()
        .map(|&(q, enc)| lane_graph.vertex_of(enc).map(|v| (q, v)))
        .collect();
    let (Some(initial_v), Some(target_v)) = (initial_v, target_v) else {
        return sequential_fallback(graph, start, ctx, goal, objective);
    };

    let Ok(p) = push_rotate_plan(
        ctx.index,
        &lane_graph,
        &initial_v,
        &target_v,
        DEFAULT_MOVE_BUDGET,
    ) else {
        return sequential_fallback(graph, start, ctx, goal, objective);
    };

    // Graft the plan onto the search graph as a chain of single-lane moves —
    // the same currency the greedy router emits.
    let mut current = start;
    let mut nodes_expanded: u32 = 0;
    for mv in &p.moves {
        let src = LocationAddr::decode(lane_graph.location_of(mv.from));
        let dst_enc = lane_graph.location_of(mv.to);
        // Any lane realizing this edge works; one exists because the plan
        // only steps along lane-graph adjacencies. Take the smallest
        // encoding so the choice is deterministic.
        let Some(lane) = ctx
            .index
            .outgoing_lanes(src)
            .iter()
            .filter(|l| {
                ctx.index
                    .endpoints(l)
                    .is_some_and(|(_, d)| d.encode() == dst_enc)
            })
            .min_by_key(|l| l.encode_u64())
            .copied()
        else {
            return (None, nodes_expanded);
        };

        let cur_config = graph.config(current).clone();
        let Some(moving_qid) = cur_config.qubit_at(src) else {
            return (None, nodes_expanded);
        };
        let new_config = cur_config.with_moves(&[(moving_qid, LocationAddr::decode(dst_enc))]);
        let move_set = MoveSet::new([lane]);
        let new_g =
            graph.g_score(current) + objective.edge_cost(&move_set, &cur_config, &new_config);
        let (child_id, _) = graph.insert(current, move_set, new_config, new_g);
        nodes_expanded += 1;
        current = child_id;
    }

    if goal.is_goal(graph.config(current)) {
        (Some(current), nodes_expanded)
    } else {
        (None, nodes_expanded)
    }
}

/// Greedy sequential fallback: move each unresolved qubit along its shortest path.
fn sequential_fallback(
    graph: &mut SearchGraph,
    start: NodeId,
    ctx: &SearchContext,
    goal: &impl Goal,
    objective: &impl Objective,
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
            let new_g =
                graph.g_score(current) + objective.edge_cost(&move_set, &cur_config, &new_config);
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

/// Run entropy-guided search under the default objective
/// ([`UniformCost`] — minimize moveset count).
///
/// This is a single-path DFS with entropy-based backtracking, NOT a
/// standard frontier-based search. See module docs for algorithm details.
///
/// Use [`entropy_search_with_objective`] to search under a different
/// [`Objective`].
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
    entropy_search_with_objective(
        root,
        goal,
        params,
        ctx,
        max_expansions,
        max_depth,
        seed,
        observer,
        &UniformCost,
    )
}

/// [`entropy_search`] under an explicit [`Objective`].
///
/// The objective is the single source of truth for `g`: it prices every shot
/// the driver appends and therefore defines what the incumbent comparison
/// means. Swapping it requires no other change to the driver.
#[allow(clippy::too_many_arguments)]
pub fn entropy_search_with_objective<O>(
    root: Config,
    goal: &impl Goal,
    params: &EntropyParams,
    ctx: &SearchContext,
    max_expansions: Option<u32>,
    max_depth: Option<u32>,
    seed: u64,
    observer: &mut dyn SearchObserver,
    objective: &O,
) -> SearchResult
where
    O: Objective,
{
    entropy_search_with_bound(
        root,
        goal,
        params,
        ctx,
        max_expansions,
        max_depth,
        seed,
        observer,
        objective,
        &NoBound::for_objective(objective),
    )
}

/// [`entropy_search`] under an explicit [`Objective`] and completion bound.
///
/// The bound must be admissible for `objective` — see [`CompletionBound`].
/// Pass [`NoBound`] to disable pruning entirely.
#[allow(clippy::too_many_arguments)]
pub fn entropy_search_with_bound<O, B>(
    root: Config,
    goal: &impl Goal,
    params: &EntropyParams,
    ctx: &SearchContext,
    max_expansions: Option<u32>,
    max_depth: Option<u32>,
    seed: u64,
    observer: &mut dyn SearchObserver,
    objective: &O,
    bound: &B,
) -> SearchResult
where
    O: Objective,
    B: CompletionBound<Obj = O>,
{
    entropy_search_with_tables(
        root,
        goal,
        params,
        ctx,
        max_expansions,
        max_depth,
        seed,
        observer,
        None,
        objective,
        bound,
    )
}

/// [`entropy_search`] with an optionally prebuilt [`HeuristicTables`].
///
/// The solver dispatch builds the tables once per solve (through the
/// engine's [`BlendedColumnCache`]) and shares them across restarts;
/// `None` builds them internally, preserving the public entry point's
/// behavior for tests, benches, and direct callers.
#[allow(clippy::too_many_arguments)]
pub(crate) fn entropy_search_with_tables<O, B>(
    root: Config,
    goal: &impl Goal,
    params: &EntropyParams,
    ctx: &SearchContext,
    max_expansions: Option<u32>,
    max_depth: Option<u32>,
    seed: u64,
    observer: &mut dyn SearchObserver,
    tables: Option<&HeuristicTables>,
    objective: &O,
    bound: &B,
) -> SearchResult
where
    O: Objective,
    B: CompletionBound<Obj = O>,
{
    // Admissibility is relative to an objective: pruning with a bound built
    // against a *different* objective instance silently discards correct
    // solutions. The associated type makes the objective *types* agree at
    // compile time; this catches same-type/different-parameter instances.
    // Once per solve, so it is a hard assert rather than a debug one — a wrong
    // prune is a correctness failure, not a performance one.
    assert_eq!(
        bound.objective_id(),
        objective.id(),
        "completion bound was built against a different objective instance \
         than the one accumulating g"
    );

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

    // Per-solve memoization of the occupancy-independent heuristic terms.
    // Build cost is roughly one node expansion; every expansion reads it.
    let owned_tables;
    let tables = match tables {
        Some(t) => t,
        None => {
            owned_tables = HeuristicTables::build(ctx, params.w_t, params.lookahead);
            &owned_tables
        }
    };

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
    // Incumbent: the lowest objective cost among goals found so far. A node
    // whose accumulated `g` already reaches it cannot lead anywhere strictly
    // cheaper, so the three gates below drop it. Under `UniformCost`
    // `g == depth` for every node, making this identical to the depth cap it
    // replaces.
    let mut best_cost: Option<f64> = None;
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
        if is_pruned(&graph, current, best_cost, bound) {
            if let Some(next) = resume_buffer_pop_best(&mut resume_buffer, best_cost) {
                current = next;
            } else if is_pruned(&graph, root_id, best_cost, bound) {
                // Nothing buffered, and even the root cannot beat the
                // incumbent (or is infeasible) — restarting from it would spin
                // to the iteration cap. Unreachable with bounding disabled:
                // `g(root) == 0 < C` for any incumbent, since a goal at the
                // root returns before this loop.
                break;
            } else {
                current = root_id;
            }
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
            if observer.wants_events() {
                let ancestor_cfg = graph.config(ancestor);
                let parent_id = graph.parent(ancestor);
                let parent_cfg = parent_id.map(|pid| graph.config(pid));
                let candidate_movesets: Vec<MoveSet> = entropy_map
                    .get(&trigger_node)
                    .map(|s| {
                        s.candidate_cache
                            .iter()
                            .map(|entry| entry.move_set.clone())
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
            }
            current = ancestor;
            continue;
        }

        // CANDIDATE SELECTION.
        let candidate = get_next_candidate(
            &mut entropy_map,
            current,
            params,
            ctx,
            &graph,
            seed,
            Some(tables),
        );

        let Some((candidate_idx, move_set, new_config, candidate_origin)) = candidate else {
            // No candidates available — bump entropy.
            let new_entropy = {
                let current_es = entropy_map.entry(current).or_default();
                current_es.entropy += 1;
                current_es.entropy
            };
            if observer.wants_events() {
                let no_valid_qid =
                    first_unresolved_qubit_without_valid_move(graph.config(current), ctx);
                let candidate_movesets: Vec<MoveSet> = entropy_map
                    .get(&current)
                    .map(|s| {
                        s.candidate_cache
                            .iter()
                            .map(|entry| entry.move_set.clone())
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
            }
            continue;
        };

        // Record as tried.
        let es = entropy_map.entry(current).or_default();
        let move_key = move_set.encoded_lanes().to_vec();
        es.tried_moves.insert(move_key.clone());
        es.candidates_tried += 1;

        // Insert into graph. The objective prices the shot — the only place
        // `g` grows in this driver besides the fallbacks.
        let trace_move_set = move_set.clone();
        let new_g = graph.g_score(current)
            + objective.edge_cost(&move_set, graph.config(current), &new_config);
        let (child_id, is_new) = graph.insert(current, move_set, new_config, new_g);

        if !is_new {
            if goal.is_goal(graph.config(child_id)) {
                let goal_cost = graph.g_score(child_id);
                found_goals.push(child_id);
                if best_cost.is_none_or(|cost| goal_cost < cost) {
                    best_cost = Some(goal_cost);
                }
                resume_buffer_discard(&mut resume_buffer, child_id);
                if observer.wants_events() {
                    let goal_cfg = graph.config(child_id);
                    let goal_parent_id = graph.parent(child_id);
                    let goal_parent_cfg = goal_parent_id.map(|pid| graph.config(pid));
                    let entropy_now = entropy_map.get(&current).map_or(1, |s| s.entropy);
                    let candidate_movesets: Vec<MoveSet> = entropy_map
                        .get(&current)
                        .map(|s| {
                            s.candidate_cache
                                .iter()
                                .map(|entry| entry.move_set.clone())
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
                }
                if found_goals.len() >= params.max_goal_candidates {
                    break;
                }
                current = resume_buffer_pop_best(&mut resume_buffer, best_cost).unwrap_or(root_id);
                continue;
            }
            // Transposition: config seen at equal or better cost.
            let new_entropy = {
                let es = entropy_map.entry(current).or_default();
                es.failed_candidates.insert(move_key.clone());
                es.entropy += 1;
                es.entropy
            };
            if observer.wants_events() {
                let candidate_movesets: Vec<MoveSet> = entropy_map
                    .get(&current)
                    .map(|s| {
                        s.candidate_cache
                            .iter()
                            .map(|entry| entry.move_set.clone())
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
            }
            continue;
        }

        // Track that a new child was created from this node.
        entropy_map.entry(current).or_default().n_children += 1;
        nodes_expanded += 1;
        let child_depth = graph.depth(child_id);
        max_depth_seen = max_depth_seen.max(child_depth);
        let child_cfg = graph.config(child_id);
        let current_cfg = graph.config(current);
        resume_buffer_discard(&mut resume_buffer, current);
        if let Some(next_best_score) = entropy_map
            .get(&current)
            .and_then(best_untried_moveset_score)
        {
            let f = graph.g_score(current)
                + if B::TRIVIAL {
                    0.0
                } else {
                    bound.estimate(graph.config(current))
                };
            resume_buffer_insert(
                &mut resume_buffer,
                current,
                next_best_score,
                f,
                graph.depth(current),
                resume_capacity,
                &mut resume_insert_order,
            );
        }

        if observer.wants_events() {
            let mut occupied = HashSet::with_capacity(ctx.blocked.len() + current_cfg.len());
            occupied.extend(ctx.blocked);
            for (_, loc) in current_cfg.iter() {
                occupied.insert(loc.encode());
            }
            let moveset_score =
                score_moveset(current_cfg, child_cfg, &occupied, ctx, params, Some(tables));
            let entropy_now = entropy_map.get(&current).map_or(1, |s| s.entropy);
            let candidate_movesets: Vec<MoveSet> = entropy_map
                .get(&current)
                .map(|s| {
                    s.candidate_cache
                        .iter()
                        .map(|entry| entry.move_set.clone())
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
        }

        if goal.is_goal(graph.config(child_id)) {
            let goal_cost = graph.g_score(child_id);
            found_goals.push(child_id);
            if best_cost.is_none_or(|cost| goal_cost < cost) {
                best_cost = Some(goal_cost);
            }
            resume_buffer_discard(&mut resume_buffer, child_id);
            if observer.wants_events() {
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
            }
            if found_goals.len() >= params.max_goal_candidates {
                break;
            }
            current = resume_buffer_pop_best(&mut resume_buffer, best_cost).unwrap_or(root_id);
            continue;
        }

        if is_pruned(&graph, child_id, best_cost, bound) {
            resume_buffer_discard(&mut resume_buffer, child_id);
            current = resume_buffer_pop_best(&mut resume_buffer, best_cost).unwrap_or(root_id);
            continue;
        }
        current = child_id; // descend
    }

    if found_goals.is_empty() && budget_exhausted {
        fire_fallback_start_event(observer, &graph, root_id, ctx, &resume_buffer);
        let (goal_id, fb_expanded) =
            budget_exhaustion_fallback(&mut graph, root_id, ctx, goal, objective);
        nodes_expanded += fb_expanded;
        if let Some(gid) = goal_id {
            found_goals.push(gid);
        }
    }

    // Return the best goal by:
    // 1) lowest objective cost, 2) lowest approximate path move time,
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
#[allow(clippy::too_many_arguments)]
fn get_next_candidate(
    entropy_map: &mut HashMap<NodeId, EntropyState>,
    node_id: NodeId,
    params: &EntropyParams,
    ctx: &SearchContext,
    graph: &SearchGraph,
    seed: u64,
    tables: Option<&HeuristicTables>,
) -> Option<(usize, MoveSet, Config, bool)> {
    let config = graph.config(node_id);
    let es = entropy_map.entry(node_id).or_default();

    // Regenerate if we've exhausted max_candidates from current cache.
    let mut regenerated =
        if es.candidates_tried >= params.max_candidates || es.candidate_cache.is_empty() {
            es.candidate_cache = generate_candidates(config, es.entropy, params, ctx, seed, tables);
            es.candidates_tried = 0;
            true
        } else {
            false
        };

    loop {
        // Find first untried, non-failed candidate.
        while es.candidates_tried < es.candidate_cache.len() {
            let entry = &es.candidate_cache[es.candidates_tried];
            let move_key = entry.move_set.encoded_lanes();
            if !es.tried_moves.contains(move_key) && !es.failed_candidates.contains(move_key) {
                let result = (
                    es.candidates_tried,
                    entry.move_set.clone(),
                    entry.new_config.clone(),
                    entry.deadlock_breaker,
                );
                return Some(result);
            }
            es.candidates_tried += 1;
        }

        // All cached candidates already tried. If we generated this cache
        // during this call, regenerating again is pointless:
        // `generate_candidates` is a pure function of
        // `(config, entropy, params, ctx, seed)` and none of those changed,
        // so a second call yields an identical cache with the same
        // all-tried outcome. Only regenerate once, when the cache we just
        // scanned was stale (carried over from a lower entropy).
        if regenerated {
            return None;
        }
        es.candidate_cache = generate_candidates(config, es.entropy, params, ctx, seed, tables);
        es.candidates_tried = 0;
        regenerated = true;
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{example_arch_json, loc};
    use crate::traits::CostFn;
    use bloqade_lanes_bytecode_core::arch::addr::{Direction, MoveType};
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
    fn budget_exhaustion_runs_fallback() {
        let r = run_entropy([(0, loc(0, 0))], [(0, loc(0, 5))], Some(0));
        let goal = r.goal.expect("fallback should find reachable target");
        assert_eq!(r.graph.config(goal).location_of(0), Some(loc(0, 5)));
    }

    /// Interference the greedy sequential fallback cannot handle: qubit 0's
    /// only route runs through qubit 1's current site, so a path avoiding
    /// every occupied vertex does not exist — the atom in the way must be
    /// displaced ahead, which is exactly what Push and Rotate does. (The
    /// example arch's components are 4-vertex paths `(0,0)-(0,5)-(1,5)-(1,0)`,
    /// so the targets preserve the atoms' order — a true swap would be
    /// genuinely unsolvable here.)
    #[test]
    fn budget_exhaustion_fallback_solves_interference() {
        let r = run_entropy(
            [(0, loc(0, 0)), (1, loc(0, 5))],
            [(0, loc(1, 5)), (1, loc(1, 0))],
            Some(0),
        );
        let goal = r
            .goal
            .expect("the planner fallback should displace the blocker");
        assert_eq!(r.graph.config(goal).location_of(0), Some(loc(1, 5)));
        assert_eq!(r.graph.config(goal).location_of(1), Some(loc(1, 0)));
    }

    /// The greedy tertiary's one residual case: a component with a single
    /// hole where an unresolved qubit slides directly into it. `m = 1` is
    /// outside the planner's regime (`TooFewEmpty`), and the greedy router
    /// can do nothing more than this — it cannot enter occupied vertices,
    /// so at `m = 1` only direct slides into the hole are realizable.
    #[test]
    fn budget_exhaustion_fallback_out_of_regime_hole_slide() {
        // Component (0,0)-(0,5)-(1,5)-(1,0), three atoms, hole at (1,0).
        let r = run_entropy(
            [(0, loc(0, 0)), (1, loc(0, 5)), (2, loc(1, 5))],
            [(2, loc(1, 0))],
            Some(0),
        );
        let goal = r
            .goal
            .expect("the greedy tertiary should slide into the hole");
        assert_eq!(r.graph.config(goal).location_of(2), Some(loc(1, 0)));
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
        let key_bus1 = TripletKey::new(MoveType::WordBus, 1, Direction::Backward);
        let key_bus2 = TripletKey::new(MoveType::WordBus, 2, Direction::Backward);
        let mut entries = [
            (
                key_bus2,
                ScoredEntry {
                    qubit_id: 8,
                    score: 3.0,
                    lane_encoded: 19,
                    dst_encoded: 40,
                },
            ),
            (
                key_bus1,
                ScoredEntry {
                    qubit_id: 4,
                    score: 3.0,
                    lane_encoded: 12,
                    dst_encoded: 40,
                },
            ),
            (
                key_bus1,
                ScoredEntry {
                    qubit_id: 4,
                    score: 3.0,
                    lane_encoded: 10,
                    dst_encoded: 40,
                },
            ),
        ];

        entries.sort_by(cmp_scored_entries);

        assert_eq!(entries[0].0, key_bus1);
        assert_eq!(entries[0].1.lane_encoded, 10);
        assert_eq!(entries[1].0, key_bus1);
        assert_eq!(entries[1].1.lane_encoded, 12);
        assert_eq!(entries[2].0, key_bus2);
    }

    #[test]
    fn resume_buffer_orders_by_score_then_depth_then_order() {
        let mut buffer = Vec::new();
        let mut next_order = 0_u64;

        resume_buffer_insert(&mut buffer, NodeId(1), 10.0, 2.0, 2, 3, &mut next_order);
        resume_buffer_insert(&mut buffer, NodeId(2), 10.0, 4.0, 4, 3, &mut next_order);
        resume_buffer_insert(&mut buffer, NodeId(3), 11.0, 1.0, 1, 3, &mut next_order);

        assert_eq!(resume_buffer_pop_best(&mut buffer, None), Some(NodeId(3)));
        assert_eq!(resume_buffer_pop_best(&mut buffer, None), Some(NodeId(2)));
        assert_eq!(resume_buffer_pop_best(&mut buffer, None), Some(NodeId(1)));
        assert_eq!(resume_buffer_pop_best(&mut buffer, None), None);
    }

    #[test]
    fn resume_buffer_capacity_and_depth_gate() {
        let mut buffer = Vec::new();
        let mut next_order = 0_u64;

        resume_buffer_insert(&mut buffer, NodeId(11), 5.0, 1.0, 1, 2, &mut next_order);
        resume_buffer_insert(&mut buffer, NodeId(12), 9.0, 2.0, 2, 2, &mut next_order);
        resume_buffer_insert(&mut buffer, NodeId(13), 3.0, 3.0, 3, 2, &mut next_order);

        // Lowest-priority node (13) is dropped at capacity.
        assert_eq!(buffer.len(), 2);

        // An incumbent cost of 2 blocks node 12 (g = 2), so 11 is next.
        assert_eq!(
            resume_buffer_pop_best(&mut buffer, Some(2.0)),
            Some(NodeId(11))
        );
        assert_eq!(resume_buffer_pop_best(&mut buffer, Some(2.0)), None);
    }

    #[test]
    fn after_first_goal_depth_gate_blocks_deeper_descend() {
        let mut buffer = Vec::new();
        let mut next_order = 0_u64;

        resume_buffer_insert(&mut buffer, NodeId(20), 8.0, 3.0, 3, 3, &mut next_order);
        resume_buffer_insert(&mut buffer, NodeId(21), 7.5, 2.0, 2, 3, &mut next_order);
        resume_buffer_insert(&mut buffer, NodeId(22), 7.0, 1.0, 1, 3, &mut next_order);

        // Once the incumbent cost is 2, candidates with g >= 2 are skipped.
        assert_eq!(
            resume_buffer_pop_best(&mut buffer, Some(2.0)),
            Some(NodeId(22))
        );
        assert_eq!(resume_buffer_pop_best(&mut buffer, Some(2.0)), None);
    }

    #[test]
    fn goal_resume_uses_buffer_then_root_fallback() {
        let root = NodeId(0);
        let mut buffer = Vec::new();
        let mut next_order = 0_u64;
        resume_buffer_insert(&mut buffer, NodeId(31), 4.0, 1.0, 1, 1, &mut next_order);

        let first_resume = resume_buffer_pop_best(&mut buffer, Some(3.0)).unwrap_or(root);
        let fallback_resume = resume_buffer_pop_best(&mut buffer, Some(3.0)).unwrap_or(root);

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

        resume_buffer_insert(&mut buffer, parent, 1.0, 3.0, 3, 3, &mut next_order);
        resume_buffer_insert(&mut buffer, NodeId(9), 2.0, 3.0, 3, 3, &mut next_order);
        // Reinsert same parent with a better move score.
        resume_buffer_insert(&mut buffer, parent, 5.0, 3.0, 3, 3, &mut next_order);

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
        let score = score_moveset(&old_config, &new_config, &occupied, &ctx, &params, None);

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

        let out1 = generate_candidates(&config, 1, &params, &ctx, 0, None);
        let out2 = generate_candidates(&config, 1, &params, &ctx, 0, None);

        assert!(!out1.is_empty());
        assert_eq!(out1.len(), out2.len());
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert_eq!(a.move_set, b.move_set);
            assert_eq!(a.new_config.as_entries(), b.new_config.as_entries());
        }
    }

    /// The per-solve tables are a memoization of the direct computation:
    /// candidates, ordering, and cached scores must be bit-identical with
    /// and without them. Exercises blocked locations, multiple qubits, and
    /// lookahead so all three tables (distance, mobility, lookahead) are hit.
    #[test]
    fn generate_candidates_tables_match_direct_computation() {
        let index = make_index();
        let config = Config::new([(0, loc(0, 0)), (1, loc(0, 1)), (2, loc(1, 0))]).unwrap();
        let target_encoded = vec![
            (0u32, loc(1, 5).encode()),
            (1u32, loc(0, 6).encode()),
            (2u32, loc(0, 5).encode()),
        ];
        let target_locs: Vec<u64> = target_encoded.iter().map(|&(_, enc)| enc).collect();
        let dist_table = DistanceTable::new(&target_locs, &index).with_time_distances(&index);
        let blocked: HashSet<u64> = [loc(0, 7).encode()].into_iter().collect();
        let ctx = SearchContext {
            index: &index,
            dist_table: &dist_table,
            blocked: &blocked,
            targets: &target_encoded,
            cz_pairs: None,
        };

        for lookahead in [false, true] {
            let params = EntropyParams {
                lookahead,
                max_movesets_per_group: 8,
                ..EntropyParams::default()
            };
            let tables = HeuristicTables::build(&ctx, params.w_t, params.lookahead);

            for entropy in [1u32, 3] {
                let direct = generate_candidates(&config, entropy, &params, &ctx, 0, None);
                let tabled = generate_candidates(&config, entropy, &params, &ctx, 0, Some(&tables));
                assert_eq!(direct.len(), tabled.len(), "lookahead={lookahead}");
                for (a, b) in direct.iter().zip(tabled.iter()) {
                    assert_eq!(a.move_set, b.move_set);
                    assert_eq!(a.new_config.as_entries(), b.new_config.as_entries());
                    assert_eq!(a.score.to_bits(), b.score.to_bits());
                    assert_eq!(a.deadlock_breaker, b.deadlock_breaker);
                }
            }

            let occupied: HashSet<u64> = config
                .iter()
                .map(|(_, l)| l.encode())
                .chain(blocked.iter().copied())
                .collect();
            for entry in generate_candidates(&config, 1, &params, &ctx, 0, None) {
                let s_direct =
                    score_moveset(&config, &entry.new_config, &occupied, &ctx, &params, None);
                let s_tabled = score_moveset(
                    &config,
                    &entry.new_config,
                    &occupied,
                    &ctx,
                    &params,
                    Some(&tables),
                );
                assert_eq!(s_direct.to_bits(), s_tabled.to_bits());
                assert_eq!(s_direct.to_bits(), entry.score.to_bits());
            }
        }
    }

    /// Cached-column builds must be bit-identical to uncached builds — on a
    /// cold cache (columns computed and inserted), on a warm cache (columns
    /// reused), and across solves with different target sets sharing target
    /// locations (the CZ-layer candidate-loop pattern the cache exists for).
    #[test]
    fn heuristic_tables_blended_cache_matches_uncached() {
        let index = make_index();
        let cache = BlendedColumnCache::new(&index);
        let blocked: HashSet<u64> = [loc(0, 7).encode()].into_iter().collect();
        let config = Config::new([(0, loc(0, 0)), (1, loc(0, 1))]).unwrap();
        let params = EntropyParams::default();

        // Two "solves": overlapping but distinct target location sets.
        let target_sets: Vec<Vec<(u32, u64)>> = vec![
            vec![(0, loc(1, 5).encode()), (1, loc(0, 6).encode())],
            vec![(0, loc(0, 6).encode()), (1, loc(0, 5).encode())],
        ];

        for targets in &target_sets {
            let target_locs: Vec<u64> = targets.iter().map(|&(_, enc)| enc).collect();
            let dist_table = DistanceTable::new(&target_locs, &index).with_time_distances(&index);
            let ctx = SearchContext {
                index: &index,
                dist_table: &dist_table,
                blocked: &blocked,
                targets,
                cz_pairs: None,
            };

            let uncached = HeuristicTables::build(&ctx, params.w_t, params.lookahead);
            // Build twice: first pass mixes cold and warm columns, second is
            // fully warm. Both must match the uncached build bit-for-bit.
            for _ in 0..2 {
                let cached =
                    HeuristicTables::build_cached(&ctx, params.w_t, params.lookahead, &cache);
                let direct = generate_candidates(&config, 1, &params, &ctx, 0, Some(&uncached));
                let via_cache = generate_candidates(&config, 1, &params, &ctx, 0, Some(&cached));
                assert_eq!(direct.len(), via_cache.len());
                for (a, b) in direct.iter().zip(via_cache.iter()) {
                    assert_eq!(a.move_set, b.move_set);
                    assert_eq!(a.score.to_bits(), b.score.to_bits());
                }
            }
        }
    }

    /// Columns cache the `w_t`-independent components, so one cache entry per
    /// target serves every weight: a cache warmed at one `w_t` must yield
    /// bit-identical tables at a different `w_t`, and must still produce the
    /// weight-specific values (not the warmed weight's).
    #[test]
    fn heuristic_tables_blended_cache_serves_multiple_w_t() {
        let index = make_index();
        let cache = BlendedColumnCache::new(&index);
        let targets = vec![(0u32, loc(1, 5).encode()), (1u32, loc(0, 6).encode())];
        let target_locs: Vec<u64> = targets.iter().map(|&(_, enc)| enc).collect();
        let dist_table = DistanceTable::new(&target_locs, &index).with_time_distances(&index);
        let blocked = HashSet::new();
        let ctx = SearchContext {
            index: &index,
            dist_table: &dist_table,
            blocked: &blocked,
            targets: &targets,
            cz_pairs: None,
        };
        let config = Config::new([(0, loc(0, 0)), (1, loc(0, 1))]).unwrap();

        // Warm the cache at one weight, then serve two others from it.
        let _warm = HeuristicTables::build_cached(&ctx, 0.05, false, &cache);
        for w_t in [0.0, 0.5, 1.0] {
            let params = EntropyParams {
                w_t,
                max_movesets_per_group: 8,
                ..EntropyParams::default()
            };
            let uncached = HeuristicTables::build(&ctx, w_t, params.lookahead);
            let cached = HeuristicTables::build_cached(&ctx, w_t, params.lookahead, &cache);

            let direct = generate_candidates(&config, 1, &params, &ctx, 0, Some(&uncached));
            let via_cache = generate_candidates(&config, 1, &params, &ctx, 0, Some(&cached));
            assert_eq!(direct.len(), via_cache.len(), "w_t={w_t}");
            for (a, b) in direct.iter().zip(via_cache.iter()) {
                assert_eq!(a.move_set, b.move_set, "w_t={w_t}");
                assert_eq!(a.score.to_bits(), b.score.to_bits(), "w_t={w_t}");
            }
        }

        // One entry per target location regardless of how many weights ran.
        assert_eq!(cache.cols.read().unwrap().len(), target_locs.len());
    }

    /// Targets with no incident lanes are interned by `DistanceTable`
    /// (`distance(t, t) = 0`) but absent from the lane-endpoint interner;
    /// the tables must give them overlay rows so `d(t, col)` matches the
    /// direct path (blended 0.0, not the unknown-location MAX sentinel) —
    /// on both the uncached and cached builds.
    #[test]
    fn heuristic_tables_intern_isolated_targets() {
        let index = make_index();
        let iso = LocationAddr {
            zone_id: 0,
            word_id: 9,
            site_id: 9,
        };
        let iso_enc = iso.encode();
        assert!(
            index.outgoing_lanes(iso).is_empty(),
            "test premise: the isolated target must have no lanes"
        );

        let targets = vec![(0u32, loc(0, 5).encode()), (1u32, iso_enc)];
        let target_locs: Vec<u64> = targets.iter().map(|&(_, enc)| enc).collect();
        let dist_table = DistanceTable::new(&target_locs, &index).with_time_distances(&index);
        let blocked = HashSet::new();
        let ctx = SearchContext {
            index: &index,
            dist_table: &dist_table,
            blocked: &blocked,
            targets: &targets,
            cz_pairs: None,
        };
        let w_t = EntropyParams::default().w_t;

        let expected = blended_distance(0.0, iso_enc, iso_enc, w_t, &dist_table);
        let cache = BlendedColumnCache::new(&index);
        for tables in [
            HeuristicTables::build(&ctx, w_t, false),
            HeuristicTables::build_cached(&ctx, w_t, false, &cache),
        ] {
            let col = tables.col(iso_enc).expect("isolated target has a column");
            assert_eq!(
                tables.d(iso_enc, col).to_bits(),
                expected.to_bits(),
                "table distance for the isolated target must match the direct path"
            );
            assert_eq!(tables.m(iso_enc, col), 0.0);
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

        let _ = generate_candidates(&config, 1, &params, &ctx, 0, None);
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

        let out = generate_candidates(&config, 1, &params, &ctx, 0, None);
        assert!(!out.is_empty());

        let mut occupied = HashSet::new();
        for (_, loc) in config.iter() {
            occupied.insert(loc.encode());
        }

        for entry in out {
            let lanes = entry.move_set.decode();
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
            let expected = entry.move_set.encoded_lanes().to_vec();
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

        let out = generate_candidates(&config, 1, &params, &ctx, 0, None);

        assert!(
            out.iter().any(|entry| {
                let candidate_config = &entry.new_config;
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

        let out = generate_candidates(&config, 1, &params, &ctx, 0, None);
        assert!(
            !out.is_empty(),
            "deadlock breaker should emit at least one fallback candidate"
        );
        assert!(
            out.iter().any(|entry| entry.deadlock_breaker),
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
            .map(|entry| {
                let candidate_config = &entry.new_config;
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

    // ── Objective: the single source of truth for `g` ───────────────

    /// Pins the audited semantics of `g`: under [`UniformCost`] every shot
    /// costs exactly `1.0`, so a node's accumulated objective cost equals its
    /// tree depth.
    ///
    /// This equality is load-bearing, not incidental. The incumbent gate
    /// compares `g` against the best goal's cost where it used to compare
    /// depths, and `Strategy::Cascade` turns a solution cost into a depth
    /// budget — both are only equivalent to their previous behaviour while
    /// this holds. A change that priced shots differently by default would
    /// break here first.
    #[test]
    fn g_score_equals_depth_under_uniform_cost() {
        let result = run_entropy([(0, loc(0, 0))], [(0, loc(1, 5))], Some(200));
        let goal = result.goal.expect("instance should solve");
        assert!(result.graph.depth(goal) > 0, "goal must not be the root");

        // Every node, not just the solution path: the two budget-exhaustion
        // fallbacks grow `g` as well, and they must price shots through the
        // objective rather than adding a literal 1.0.
        for raw in 0..result.graph.len() {
            let id = NodeId(raw as u32);
            assert_eq!(
                result.graph.g_score(id),
                f64::from(result.graph.depth(id)),
                "node {raw}: g must equal depth under UniformCost"
            );
        }
    }

    /// [`UniformCost`] honours the framework's C2/C3 contract: shot costs are
    /// non-negative, and every shot costs at least the weight of each lane it
    /// contains (with equality here, which is what makes the derived bound the
    /// tightest one available for this objective).
    ///
    /// Step 3 generalizes this into a reusable
    /// `assert_objective_bound_contract` helper every objective is checked
    /// with; pinning the default objective is cheap now.
    #[test]
    fn uniform_cost_objective_honours_the_lane_floor() {
        let index = make_index();
        let objective = UniformCost;
        let config = Config::new([(0, loc(0, 0))]).unwrap();

        assert_eq!(objective.min_shot_cost(), 1.0);
        assert_eq!(objective.id(), objective.id(), "id must be stable");

        let mut lanes_checked = 0_usize;
        for (mt, bus_id, zone_id, dir) in index.bus_groups() {
            for &lane in index.lanes_for(mt, bus_id, zone_id, dir) {
                let shot = MoveSet::new([lane]);
                let cost = objective.edge_cost(&shot, &config, &config);
                assert!(cost >= 0.0, "C2 violated for {lane:?}: {cost}");
                assert!(
                    cost >= objective.lane_weight(lane),
                    "C3 violated for {lane:?}: shot cost {cost} < lane weight {}",
                    objective.lane_weight(lane)
                );
                lanes_checked += 1;
            }
        }
        assert!(lanes_checked > 0, "arch fixture should expose lanes");
    }

    /// The objective is swappable at the driver seam: `entropy_search` and
    /// `entropy_search_with_objective(.., &UniformCost)` are the same search.
    ///
    /// Guards the delegation, so the default entry point cannot drift onto a
    /// different objective than the one the audit and benchmarks assume.
    #[test]
    fn default_entry_point_matches_explicit_uniform_cost_objective() {
        let index = make_index();
        let target_encoded = vec![(0u32, loc(1, 5).encode())];
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
        let params = EntropyParams::default();
        let root = Config::new([(0, loc(0, 0))]).unwrap();

        let implicit = entropy_search(
            root.clone(),
            &goal,
            &params,
            &ctx,
            Some(200),
            None,
            0,
            &mut crate::observer::NoOpObserver,
        );
        let explicit = entropy_search_with_objective(
            root,
            &goal,
            &params,
            &ctx,
            Some(200),
            None,
            0,
            &mut crate::observer::NoOpObserver,
            &UniformCost,
        );

        assert_eq!(implicit.nodes_expanded, explicit.nodes_expanded);
        assert_eq!(implicit.max_depth_reached, explicit.max_depth_reached);
        assert_eq!(
            implicit.solution_path().map(|p| p.len()),
            explicit.solution_path().map(|p| p.len())
        );
        assert_eq!(
            implicit.goal.map(|g| implicit.graph.g_score(g)),
            explicit.goal.map(|g| explicit.graph.g_score(g))
        );
    }

    /// Swapping the objective requires no driver change, and `g` really is
    /// that objective's cost rather than a moveset count in disguise.
    ///
    /// The assertion that matters is the independent recomputation: the
    /// solution's `g_score` must equal the sum of `edge_cost` over the emitted
    /// move layers. If the driver had kept any inlined `+ 1.0`, `g` would come
    /// out as the layer count instead.
    #[test]
    fn driver_accumulates_g_under_a_swapped_objective() {
        use crate::cost::WeightedDuration;

        let index = make_index();
        let target_encoded = vec![(0u32, loc(1, 0).encode())];
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
        let root = Config::new([(0, loc(0, 0))]).unwrap();
        let objective = WeightedDuration::new(&index, 10.0);

        let result = entropy_search_with_objective(
            root.clone(),
            &goal,
            &EntropyParams::default(),
            &ctx,
            Some(200),
            None,
            0,
            &mut crate::observer::NoOpObserver,
            &objective,
        );

        let goal_id = result.goal.expect("instance should solve");
        let layers = result.solution_path().expect("solved run has a path");
        assert!(!layers.is_empty());

        // `WeightedDuration::edge_cost` is configuration-independent, so the
        // endpoints passed here do not affect the recomputation.
        let expected: f64 = layers
            .iter()
            .map(|ms| objective.edge_cost(ms, &root, &root))
            .sum();
        let actual = result.graph.g_score(goal_id);
        assert!(
            (actual - expected).abs() < 1e-9,
            "g {actual} should equal the summed objective cost {expected}"
        );

        // And it is genuinely not the moveset count: every shot costs
        // `1 + dur/tau` with a positive duration term.
        let depth = f64::from(result.graph.depth(goal_id));
        assert!(
            actual > depth,
            "weighted g {actual} should exceed the layer count {depth}"
        );
    }

    // ── Completion bound (branch-and-bound pruning) ─────────────────

    /// Run the driver twice on one instance, with and without `h0`.
    fn run_bounded_and_unbounded(
        initial: impl IntoIterator<Item = (u32, LocationAddr)>,
        target: impl IntoIterator<Item = (u32, LocationAddr)>,
        blocked_locs: &[LocationAddr],
        max_expansions: Option<u32>,
    ) -> (SearchResult, SearchResult) {
        let index = make_index();
        let target_encoded: Vec<(u32, u64)> =
            target.into_iter().map(|(q, l)| (q, l.encode())).collect();
        let target_locs: Vec<u64> = target_encoded.iter().map(|&(_, l)| l).collect();
        let dist_table = DistanceTable::new(&target_locs, &index);
        let blocked: HashSet<u64> = blocked_locs.iter().map(|l| l.encode()).collect();
        let goal = crate::goals::AllAtTarget::new(&target_encoded);
        let ctx = SearchContext {
            index: &index,
            dist_table: &dist_table,
            blocked: &blocked,
            targets: &target_encoded,
            cz_pairs: None,
        };
        let params = EntropyParams::default();
        let root = Config::new(initial).unwrap();

        let unbounded = entropy_search_with_bound(
            root.clone(),
            &goal,
            &params,
            &ctx,
            max_expansions,
            None,
            0,
            &mut crate::observer::NoOpObserver,
            &UniformCost,
            &NoBound::for_objective(&UniformCost),
        );
        let bound = crate::bounds::WeightedDistanceBound::new(
            &UniformCost,
            &target_encoded,
            &index,
            &blocked,
        );
        let bounded = entropy_search_with_bound(
            root,
            &goal,
            &params,
            &ctx,
            max_expansions,
            None,
            0,
            &mut crate::observer::NoOpObserver,
            &UniformCost,
            &bound,
        );
        (unbounded, bounded)
    }

    /// An unreachable target makes `h0 = +∞`, an infeasibility proof
    /// independent of any incumbent, so the bounded run stops instead of
    /// grinding to the iteration cap.
    ///
    /// This is the pre-registered behaviour change of enabling the bound:
    /// unroutable instances stop early and so report `Unsolvable` rather than
    /// `BudgetExceeded`. Sound, because both budget-exhaustion fallbacks also
    /// carve out `blocked` and could not have reached the target either.
    #[test]
    fn infeasible_instance_is_cut_immediately_when_bounded() {
        let (unbounded, bounded) = run_bounded_and_unbounded(
            [(0, loc(0, 0))],
            [(0, loc(99, 99))], // not a location in the fixture at all
            &[],
            Some(64),
        );

        assert!(unbounded.goal.is_none() && bounded.goal.is_none());
        assert_eq!(
            bounded.nodes_expanded, 0,
            "an infeasible root should be cut before any expansion"
        );
    }

    /// Enabling the bound must never worsen the answer: pruning only removes
    /// branches that provably cannot contain a strictly cheaper solution.
    ///
    /// Note what is deliberately *not* asserted: that bounding expands fewer
    /// nodes. It often expands more, and that is not a defect. This driver
    /// stops once it has collected `max_goal_candidates` goals; with ties
    /// pruned, every goal after the first must be *strictly* cheaper, which is
    /// far rarer, so the search keeps going — frequently until it has proven
    /// no better solution exists. That is strictly more work than the
    /// unbounded run performs, but it is also a stronger result, and it is why
    /// bounding sometimes returns a cheaper plan. Measured node counts move in
    /// both directions; see the step-5 instrumentation.
    #[test]
    fn bound_preserves_solution_cost() {
        /// `(initial placement, target placement)` for one instance.
        type Instance = (Vec<(u32, LocationAddr)>, Vec<(u32, LocationAddr)>);

        let cases: [Instance; 3] = [
            (vec![(0, loc(0, 0))], vec![(0, loc(1, 0))]),
            (
                vec![(0, loc(0, 0)), (1, loc(0, 5))],
                vec![(0, loc(1, 0)), (1, loc(1, 5))],
            ),
            (
                vec![(0, loc(0, 0)), (1, loc(1, 0))],
                vec![(0, loc(1, 0)), (1, loc(0, 0))],
            ),
        ];

        for (initial, target) in cases {
            let (unbounded, bounded) =
                run_bounded_and_unbounded(initial.clone(), target.clone(), &[], Some(400));

            match (unbounded.goal, bounded.goal) {
                (Some(u), Some(b)) => {
                    let uc = unbounded.graph.g_score(u);
                    let bc = bounded.graph.g_score(b);
                    assert!(
                        bc <= uc,
                        "bounded cost {bc} must not exceed unbounded {uc} for {initial:?}"
                    );
                }
                (None, None) => {}
                (u, b) => panic!("solvability disagreed: unbounded={u:?} bounded={b:?}"),
            }
        }
    }

    /// Pruning with a bound built against a different objective *instance*
    /// would silently discard correct solutions. Both sides are
    /// `WeightedDuration` here, so the associated type cannot tell them apart
    /// and only the driver-entry id check can.
    #[test]
    #[should_panic(expected = "different objective instance")]
    fn driver_rejects_a_bound_from_another_objective_instance() {
        use crate::cost::WeightedDuration;

        let index = make_index();
        let target_encoded = vec![(0u32, loc(1, 0).encode())];
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

        let accumulating = WeightedDuration::new(&index, 1.0);
        let bound_from_other = crate::bounds::WeightedDistanceBound::new(
            &WeightedDuration::new(&index, 10.0),
            &target_encoded,
            &index,
            &blocked,
        );

        let _ = entropy_search_with_bound(
            Config::new([(0, loc(0, 0))]).unwrap(),
            &goal,
            &EntropyParams::default(),
            &ctx,
            Some(50),
            None,
            0,
            &mut crate::observer::NoOpObserver,
            &accumulating,
            &bound_from_other,
        );
    }
}

#[cfg(test)]
mod chain_assembly {
    use super::*;
    use crate::primitives::distance::DistanceTable;
    use crate::test_utils::loc;
    use bloqade_lanes_bytecode_core::arch::types::ArchSpec;
    use std::collections::HashSet;

    /// The entropy driver serialized conveyor chains before #887: it emitted
    /// only the follower's single-lane move because rectangle growth discarded
    /// the leader. With growth repairing the chain, both hops ride in one
    /// candidate — and it outranks the follower-only option, since its score
    /// is the sum of both entries'.
    #[test]
    fn entropy_generates_a_chain_candidate() {
        let spec: ArchSpec = serde_json::from_str(&crate::test_utils::chain_arch_json()).unwrap();
        let index = LaneIndex::new(spec);
        let config = Config::new([(0, loc(0, 0)), (1, loc(0, 1))]).unwrap();
        let target_encoded = vec![(0u32, loc(0, 1).encode()), (1u32, loc(0, 2).encode())];
        let target_locs: Vec<u64> = target_encoded.iter().map(|&(_, e)| e).collect();
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
            max_movesets_per_group: 16,
            ..EntropyParams::default()
        };

        let out = generate_candidates(&config, 1, &params, &ctx, 0, None);
        let chain = out.iter().find(|c| {
            c.new_config.location_of(0) == Some(loc(0, 1))
                && c.new_config.location_of(1) == Some(loc(0, 2))
        });
        assert!(
            chain.is_some(),
            "entropy must offer the one-shot chain; got {:?}",
            out.iter()
                .map(|c| (
                    c.move_set.len(),
                    c.new_config.location_of(0).map(|l| l.site_id),
                    c.new_config.location_of(1).map(|l| l.site_id),
                ))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            chain.expect("checked above").move_set.len(),
            2,
            "the chain must be a single two-lane operation"
        );
    }
}
