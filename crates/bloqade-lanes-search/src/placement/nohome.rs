//! No-home return assignment for two-phase placement.
//!
//! After a CZ layer, atoms sit at CZ-staging positions. Instead of returning
//! them to their *original* home sites, this module finds an optimal
//! assignment of displaced qubits to *any* available home site, minimising a
//! blend of immediate routing cost and future CZ-partner proximity.
//!
//! The assignment is solved with the Hungarian algorithm (reused from
//! [`crate::ops::entangling`]), making it polynomial and architecture-general.

use std::collections::{HashMap, HashSet};

use bloqade_lanes_bytecode_core::arch::addr::{Direction, LocationAddr, MoveType};

use crate::ops::entangling;
use crate::primitives::config::Config;
use crate::primitives::distance::DistanceTable;
use crate::primitives::lane_index::LaneIndex;
use crate::primitives::path::find_path_occupied;

/// Lane-signature triple: identifies a parallelisable bus group.
type LaneSig = (MoveType, u32, Direction);

/// Per-edge scoring entry in `candidate_return_layouts`:
/// `(score, hole_index, hop_cost, sig_set)`.
type EdgeScore = (f64, usize, u32, HashSet<LaneSig>);

/// A CZ pair with each qubit's current location:
/// `((control, location), (target, location))`.
type StagePair = ((u32, LocationAddr), (u32, LocationAddr));

/// A candidate CZ-staging target: every qubit's location.
type CzTarget = Vec<(u32, LocationAddr)>;

// ── Options ───────────────────────────────────────────────────────

/// How the CZ phase chooses, for each pair, which qubit moves.
///
/// A pair whose qubits both have a CZ partner can be staged two ways: the
/// control moves to the target's partner site, or the target moves to the
/// control's. Over a stage's `k` such pairs that is up to `2^k` candidate
/// targets. Pairs without a partner go to a free entangling slot under every
/// variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MoverSelection {
    /// A fixed per-pair rule, with no comparison: the control moves if both
    /// qubits share a word, else the target if it sits on a home site, else
    /// the control. One routing solve.
    Rule,
    /// Plan every candidate with Push and Rotate, which is fast and always
    /// finishes but is not shortest, and route the candidate with the
    /// shortest plan (the next, on failure). The plan's length is an upper
    /// bound on the candidate's cost, but not always a good predictor of the
    /// router's, so the rule's candidate is routed too and the ranked pick is
    /// kept only if it takes fewer layers: never worse than [`Self::Rule`].
    /// Two routing solves, or one when the rule's candidate is ranked first.
    #[default]
    Ranked,
    /// Route every candidate and keep the one with the fewest move layers,
    /// the rule's on ties. The most thorough, and one routing solve per
    /// candidate.
    RouteAll,
}

/// Tuning knobs for the no-home return assignment.
#[derive(Debug, Clone)]
pub struct NoHomeOptions {
    /// Discount factor for future CZ layer weights (default 0.85).
    pub gamma: f64,
    /// Blend weight: how much future proximity matters relative to
    /// immediate path cost (default 0.5).
    pub lambda_lookahead: f64,
    /// Maximum candidate holes per returner for cost-matrix pruning
    /// (default 8).
    pub k_candidates: usize,
    /// Number of bus-reward variant assignments to generate (default 6).
    /// Each variant rewards edges sharing a high-coverage lane signature,
    /// biasing the assignment toward layouts with parallel routing.
    pub top_bus_signatures: usize,
    /// Per-edge hop-count discount applied to edges using a top signature
    /// when building bus-reward variant cost matrices (default 1).
    pub bus_reward_rho: u32,
    /// How the CZ phase picks which qubit of each pair moves (default
    /// [`MoverSelection::Ranked`]).
    pub mover_selection: MoverSelection,
    /// Most candidate targets [`MoverSelection::Ranked`] and
    /// [`MoverSelection::RouteAll`] compare (default 64). A stage with more
    /// mover assignments than this compares the rule's, every single-pair
    /// flip of it, and a seeded sample of the rest.
    pub max_mover_candidates: usize,
}

impl Default for NoHomeOptions {
    fn default() -> Self {
        Self {
            gamma: 0.85,
            lambda_lookahead: 0.5,
            k_candidates: 8,
            top_bus_signatures: 6,
            bus_reward_rho: 1,
            mover_selection: MoverSelection::default(),
            max_mover_candidates: 64,
        }
    }
}

/// The verdict for a CZ phase in which no candidate routed.
///
/// It describes the stage, not any one candidate's target:
/// [`SolveStatus::BudgetExceeded`] if any candidate ran out of budget, since
/// more budget may solve it; otherwise [`SolveStatus::Unsolvable`], carrying a
/// no-plan proof only when `exhaustive` (every way to stage the pairs was
/// routed) and every one of them proved it (`all_proven`).
fn failed_stage_verdict(
    any_budget: bool,
    all_proven: bool,
    exhaustive: bool,
) -> (SolveStatus, Termination) {
    if any_budget {
        (SolveStatus::BudgetExceeded, Termination::Budget)
    } else {
        (
            SolveStatus::Unsolvable,
            Termination::Exhausted {
                proof: all_proven && exhaustive,
            },
        )
    }
}

/// Whether [`mover_assignments`] returns every assignment of `k` pairs under
/// `cap`, rather than a sample.
fn enumerates_all(k: usize, cap: usize) -> bool {
    k < usize::BITS as usize && (1usize << k) <= cap.max(1)
}

/// Mover assignments to compare, the rule's first: `true` moves the pair's
/// target, `false` its control.
///
/// Every assignment when there are at most `cap`; otherwise the rule's, every
/// single-pair flip of it, then distinct assignments drawn from a fixed seed,
/// up to `cap` (or fewer, when the draws keep repeating).
fn mover_assignments(rule: &[bool], cap: usize) -> Vec<Vec<bool>> {
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    let k = rule.len();
    let cap = cap.max(1);
    if enumerates_all(k, cap) {
        let mut all = vec![rule.to_vec()];
        all.extend(
            (0..1usize << k)
                .map(|bits| (0..k).map(|i| bits >> i & 1 == 1).collect::<Vec<_>>())
                .filter(|a| a != rule),
        );
        return all;
    }
    let mut seen: HashSet<Vec<bool>> = HashSet::new();
    let mut out = Vec::with_capacity(cap);
    let mut push = |a: Vec<bool>, out: &mut Vec<Vec<bool>>| {
        if out.len() < cap && seen.insert(a.clone()) {
            out.push(a);
        }
    };
    push(rule.to_vec(), &mut out);
    for i in 0..k {
        let mut flip = rule.to_vec();
        flip[i] = !flip[i];
        push(flip, &mut out);
    }
    let mut rng = SmallRng::seed_from_u64(0x4E0_40E5);
    for _ in 0..cap.saturating_mul(8) {
        if out.len() >= cap {
            break;
        }
        push((0..k).map(|_| rng.random_bool(0.5)).collect(), &mut out);
    }
    out
}

// ── Helpers ───────────────────────────────────────────────────────

/// Compute gamma-decayed partner weights from future CZ layers.
///
/// Returns `qubit_id → { partner_id → accumulated_weight }`.
pub fn partner_weights(
    future_cz_layers: &[Vec<(u32, u32)>],
    gamma: f64,
) -> HashMap<u32, HashMap<u32, f64>> {
    let mut weights: HashMap<u32, HashMap<u32, f64>> = HashMap::new();
    for (depth, layer) in future_cz_layers.iter().enumerate() {
        let w = gamma.powi(depth as i32);
        for &(c, t) in layer {
            *weights.entry(c).or_default().entry(t).or_insert(0.0) += w;
            *weights.entry(t).or_default().entry(c).or_insert(0.0) += w;
        }
    }
    weights
}

/// Greedy nearest-home assignment: each non-home qubit gets the closest
/// available home site by hop distance.
///
/// Returns `(qubit_id, encoded_home_location)` pairs for returners only.
pub fn nearest_home_layout(
    config: &Config,
    home_set: &HashSet<u64>,
    holes: &[u64],
    dist_table: &DistanceTable,
) -> Vec<(u32, u64)> {
    let mut available: Vec<u64> = holes.to_vec();
    let mut assignments = Vec::new();

    for (qid, loc) in config.iter() {
        let loc_enc = loc.encode();
        if home_set.contains(&loc_enc) {
            continue; // already home
        }
        if available.is_empty() {
            break; // no holes left
        }
        // Pick hole with smallest hop distance to current position.
        let best_idx = available
            .iter()
            .enumerate()
            .min_by_key(|&(_, &hole)| dist_table.distance(loc_enc, hole).unwrap_or(u32::MAX))
            .map(|(i, _)| i)
            .unwrap();
        let best_hole = available.swap_remove(best_idx);
        assignments.push((qid, best_hole));
    }
    assignments
}

/// Compute optimal return-layout candidates using the Hungarian algorithm.
///
/// Returns up to `1 + opts.top_bus_signatures` distinct candidate layouts.
/// Each layout is a full qubit→location mapping (all qubits, not just
/// returners) as `(qubit_id, LocationAddr)`.
///
/// Algorithm:
/// 1. For each (returner, hole) edge: BFS-shortest-path, hop-cost, lane
///    signatures used by the path, lookahead-blended score.
/// 2. Top-K hole pruning per returner.
/// 3. Run plain Hungarian on the baseline cost matrix → 1 candidate.
/// 4. For each top-`top_bus_signatures` lane signature ranked by coverage
///    (number of returners whose pruned edges include it), build a cost
///    matrix where edges using that signature get a `bus_reward_rho` hop
///    discount; run Hungarian → up to N more candidates.
/// 5. Deduplicate and return.
#[allow(clippy::too_many_arguments)]
pub fn candidate_return_layouts(
    config: &Config,
    home_set: &HashSet<u64>,
    holes: &[u64],
    dist_table: &DistanceTable,
    index: &LaneIndex,
    pw: &HashMap<u32, HashMap<u32, f64>>,
    opts: &NoHomeOptions,
) -> Vec<Vec<(u32, LocationAddr)>> {
    // Identify returners: qubits not at home positions.
    let returners: Vec<(u32, u64)> = config
        .iter()
        .map(|(qid, loc)| (qid, loc.encode()))
        .filter(|(_, loc)| !home_set.contains(loc))
        .collect();

    if returners.is_empty() {
        // Everyone is home already — return current layout as-is.
        return vec![config.iter().collect()];
    }
    if holes.len() < returners.len() {
        // Not enough holes — fall back to nearest-home greedy.
        let greedy = nearest_home_layout(config, home_set, holes, dist_table);
        return vec![build_full_layout(config, home_set, &greedy)];
    }

    // Build reference positions from greedy baseline (for lookahead cost).
    let greedy_assignments = nearest_home_layout(config, home_set, holes, dist_table);
    let mut reference: HashMap<u32, u64> = HashMap::new();
    for (qid, loc) in config.iter() {
        let enc = loc.encode();
        if home_set.contains(&enc) {
            reference.insert(qid, enc);
        }
    }
    for &(qid, loc) in &greedy_assignments {
        reference.insert(qid, loc);
    }

    // ── Score each (returner, hole) edge: hop-cost, lookahead, sigs ──
    // For each returner, collect its scored edges. Each entry is
    // (score, hidx, hop_cost, sig_set). Paths come from BFS so the hop
    // count = path.len() = `dist_table.distance` for the same arch (both
    // are unweighted shortest paths).
    let empty_blocked: HashSet<u64> = HashSet::new();
    let n_returners = returners.len();
    let mut scored_per_returner: Vec<Vec<EdgeScore>> = Vec::with_capacity(n_returners);

    for &(qid, src_enc) in &returners {
        let src_loc = LocationAddr::decode(src_enc);
        let mut scored: Vec<EdgeScore> = Vec::with_capacity(holes.len());
        for (hidx, &hole) in holes.iter().enumerate() {
            let dst_loc = LocationAddr::decode(hole);
            let path = match find_path_occupied(src_loc, dst_loc, &empty_blocked, index) {
                Some(p) => p,
                None => continue, // unreachable
            };
            let hop_cost = path.len() as u32;

            // Lookahead penalty against reference partners. Skip pairs the
            // dist_table can't reach — `u32::MAX as f64` would saturate the
            // score and dominate the cost matrix, masking the real ranking.
            let mut future_delta = 0.0;
            if let Some(partners) = pw.get(&qid) {
                for (&pid, &weight) in partners {
                    if let Some(&ref_pos) = reference.get(&pid)
                        && let Some(d) = dist_table.distance(hole, ref_pos)
                    {
                        future_delta += weight * d as f64;
                    }
                }
            }
            let score = hop_cost as f64 + opts.lambda_lookahead * future_delta;

            // Lane-signature set for this path.
            let sigs: HashSet<LaneSig> = path
                .iter()
                .map(|lane| (lane.move_type, lane.bus_id, lane.direction))
                .collect();

            scored.push((score, hidx, hop_cost, sigs));
        }
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(opts.k_candidates);
        scored_per_returner.push(scored);
    }

    // ── Pruned hole universe ──
    let mut active_holes: Vec<usize> = scored_per_returner
        .iter()
        .flat_map(|s| s.iter().map(|&(_, hidx, _, _)| hidx))
        .collect();
    active_holes.sort_unstable();
    active_holes.dedup();

    let n_active = active_holes.len();
    if n_active == 0 || n_active < n_returners {
        return vec![build_full_layout(config, home_set, &greedy_assignments)];
    }

    let hole_to_compact: HashMap<usize, usize> = active_holes
        .iter()
        .enumerate()
        .map(|(ci, &hi)| (hi, ci))
        .collect();

    // ── Build baseline cost matrix (scaled u32) ──
    const SCALE: f64 = 1000.0;
    let large_cost: u32 = 1_000_000_000;
    let mut base_cost = vec![large_cost; n_returners * n_active];

    for (ridx, scored) in scored_per_returner.iter().enumerate() {
        for &(score, hidx, _, _) in scored {
            if let Some(&cidx) = hole_to_compact.get(&hidx) {
                base_cost[ridx * n_active + cidx] = (score * SCALE) as u32;
            }
        }
    }

    // ── Run baseline Hungarian — first candidate ──
    let mut all_assignments: Vec<Vec<usize>> = Vec::new();
    let baseline_assign = entangling::hungarian(&base_cost, n_returners, n_active);
    all_assignments.push(baseline_assign);

    // ── Rank lane signatures by coverage across pruned edges ──
    if opts.top_bus_signatures > 0 {
        // sig_coverage[sig] = set of returner indices whose pruned edges include sig.
        let mut sig_coverage: HashMap<LaneSig, HashSet<usize>> = HashMap::new();
        for (ridx, scored) in scored_per_returner.iter().enumerate() {
            for (_, _, _, sigs) in scored {
                for &sig in sigs {
                    sig_coverage.entry(sig).or_default().insert(ridx);
                }
            }
        }

        // Rank signatures by coverage (ties broken deterministically by sig itself).
        let mut ranked: Vec<(LaneSig, usize)> = sig_coverage
            .into_iter()
            .map(|(sig, set)| (sig, set.len()))
            .collect();
        ranked.sort_by(|a, b| {
            b.1.cmp(&a.1)
                .then_with(|| (a.0.0 as u8).cmp(&(b.0.0 as u8)))
                .then_with(|| a.0.1.cmp(&b.0.1))
                .then_with(|| (a.0.2 as u8).cmp(&(b.0.2 as u8)))
        });
        ranked.truncate(opts.top_bus_signatures);

        // ── Build a reward-modified cost matrix per top signature ──
        let reward_scaled = (opts.bus_reward_rho as f64 * SCALE) as u32;
        for (sig, _) in &ranked {
            if reward_scaled == 0 {
                break;
            }
            let mut cost = base_cost.clone();
            for (ridx, scored) in scored_per_returner.iter().enumerate() {
                for (_, hidx, _, sigs) in scored {
                    if sigs.contains(sig)
                        && let Some(&cidx) = hole_to_compact.get(hidx)
                    {
                        let idx = ridx * n_active + cidx;
                        cost[idx] = cost[idx].saturating_sub(reward_scaled);
                    }
                }
            }
            let assign = entangling::hungarian(&cost, n_returners, n_active);
            // Dedup: skip if identical to any previously-collected assignment.
            if !all_assignments.iter().any(|a| a == &assign) {
                all_assignments.push(assign);
            }
        }
    }

    // ── Materialise candidate layouts ──
    let mut candidates: Vec<Vec<(u32, LocationAddr)>> = Vec::with_capacity(all_assignments.len());
    for assignment in &all_assignments {
        let mut returner_assigns: Vec<(u32, u64)> = Vec::with_capacity(n_returners);
        for (ridx, &compact_col) in assignment.iter().enumerate() {
            let original_hidx = active_holes[compact_col];
            let (qid, _) = returners[ridx];
            returner_assigns.push((qid, holes[original_hidx]));
        }
        candidates.push(build_full_layout(config, home_set, &returner_assigns));
    }
    candidates
}

/// Build a full `(qubit_id, LocationAddr)` layout from a config plus
/// returner assignments (encoded).
fn build_full_layout(
    config: &Config,
    home_set: &HashSet<u64>,
    returner_assignments: &[(u32, u64)],
) -> Vec<(u32, LocationAddr)> {
    let assign_map: HashMap<u32, u64> = returner_assignments.iter().copied().collect();
    config
        .iter()
        .map(|(qid, loc)| {
            let enc = loc.encode();
            if home_set.contains(&enc) {
                (qid, loc) // already home — keep
            } else {
                let target_enc = assign_map.get(&qid).unwrap_or(&enc);
                (qid, LocationAddr::decode(*target_enc))
            }
        })
        .collect()
}

/// Place CZ pairs on free entangling slots.
///
/// This is Phase 2's fallback for pairs the per-pair rule cannot place,
/// because a qubit sits where [`ArchSpec::get_cz_partner`] has no partner.
/// A slot is one site index of an entangling word pair: the same site on
/// both words, in the pair's zone. It is free when neither half is in
/// `claimed`, so a pair placed there collides with no other atom and has no
/// third atom beside it. The Hungarian chooses the slots that minimise total
/// hop distance, with each slot taking its cheaper orientation.
///
/// Entangling pairs may overlap (`[0, 1]` and `[1, 2]` both pass
/// validation), so two slots can share a half, and the Hungarian does not
/// know that. The search crate assumes one CZ partner per location (see
/// [`entangling::build_partner_map`]), so rather than repair such an
/// assignment this reports the stage unplaceable: it never returns
/// colliding targets.
///
/// Returns both qubits' targets for every pair, or `None` when some pair
/// cannot be placed: there are fewer free slots than pairs, none that both
/// of its qubits can reach, or the chosen slots share a half.
fn assign_free_slots(
    pairs: &[StagePair],
    claimed: &HashSet<u64>,
    index: &LaneIndex,
    dist_table: &DistanceTable,
) -> Option<Vec<(u32, LocationAddr)>> {
    let sites_per_word = index.sites_per_word() as u32;
    let slots: Vec<(LocationAddr, LocationAddr)> = entangling::enumerate_word_pairs(index)
        .into_iter()
        .flat_map(|wp| {
            (0..sites_per_word).map(move |site_id| {
                let at = |word_id| LocationAddr {
                    zone_id: wp.zone_id,
                    word_id,
                    site_id,
                };
                (at(wp.word_a), at(wp.word_b))
            })
        })
        .filter(|(a, b)| !claimed.contains(&a.encode()) && !claimed.contains(&b.encode()))
        .collect();
    let n_slots = slots.len();
    if n_slots < pairs.len() {
        return None;
    }

    // Unreachable halves cost `BIG`, so any cell at or above it is unusable.
    const BIG: u32 = u32::MAX / 4;
    let hops = |from: LocationAddr, to: LocationAddr| {
        dist_table
            .distance(from.encode(), to.encode())
            .unwrap_or(BIG)
    };
    let mut costs = vec![BIG; pairs.len() * n_slots];
    // `crossed[i]`: the control takes half `b` and the target half `a`.
    let mut crossed = vec![false; costs.len()];
    for (row, &((_, c_loc), (_, t_loc))) in pairs.iter().enumerate() {
        for (col, &(a, b)) in slots.iter().enumerate() {
            let straight_cost = hops(c_loc, a).saturating_add(hops(t_loc, b));
            let crossed_cost = hops(c_loc, b).saturating_add(hops(t_loc, a));
            let idx = row * n_slots + col;
            costs[idx] = straight_cost.min(crossed_cost);
            crossed[idx] = crossed_cost < straight_cost;
        }
    }

    let assignment = entangling::hungarian(&costs, pairs.len(), n_slots);
    let mut taken: HashSet<u64> = HashSet::with_capacity(2 * pairs.len());
    let mut targets = Vec::with_capacity(2 * pairs.len());
    for (row, &col) in assignment.iter().enumerate() {
        let idx = row * n_slots + col;
        if costs[idx] >= BIG {
            return None;
        }
        let (a, b) = slots[col];
        if !taken.insert(a.encode()) || !taken.insert(b.encode()) {
            return None; // overlapping entangling pairs: slots share a half
        }
        let ((c, _), (t, _)) = pairs[row];
        let (c_dst, t_dst) = if crossed[idx] { (b, a) } else { (a, b) };
        targets.push((c, c_dst));
        targets.push((t, t_dst));
    }
    Some(targets)
}

// ── CzPlacement composition ─────────────────────────────────────────────

use crate::drivers::result::Termination;
use crate::placement::cz_placement::{CzPlacement, CzStage, PlacementBudget, PlacementResult};
use crate::primitives::config::ConfigError;
use crate::search::engine::SearchEngine;
use crate::search::move_search::MoveSearch;
use crate::search::options::{SolveOptions, Strategy};
use crate::search::result::{SolveResult, SolveStatus};
use crate::search::target_solver::solve_with_engine;
use std::sync::Arc;

/// Two-phase no-home CZ placement.
///
/// Composes `Arc<SearchEngine> + MoveSearch + NoHomeOptions`. Phase 1
/// generates `1 + nohome_opts.top_bus_signatures` candidate home
/// layouts (Hungarian with lane-signature reward variants), routes
/// each via [`solve_with_engine`], and keeps the candidate whose
/// return-routing produces the fewest move layers. Phase 2 stages each pair
/// by moving one qubit to the other's CZ partner site, and
/// [`NoHomeOptions::mover_selection`] picks which:
///
/// - [`MoverSelection::Rule`]: a fixed per-pair rule, routed once.
/// - [`MoverSelection::Ranked`] (the default): every candidate mover
///   assignment (up to [`NoHomeOptions::max_mover_candidates`]) is planned
///   with Push and Rotate; the best-ranked candidate and the rule's are
///   routed, and the ranked pick is kept only if it takes fewer layers.
/// - [`MoverSelection::RouteAll`]: every candidate is routed and the one
///   with the fewest layers kept.
///
/// A pair with a qubit that has no CZ partner (e.g. in a storage zone with
/// no entangling pairs) goes to a free entangling slot instead. If some pair
/// cannot be placed at all, the result is [`SolveStatus::Unsolvable`]. When
/// no candidate routes, the result is [`SolveStatus::BudgetExceeded`] if any
/// ran out of budget, and carries a no-plan proof only when every way to
/// stage the pairs was routed and each proved it.
pub struct NoHomeCzPlacement {
    engine: Arc<SearchEngine>,
    search: MoveSearch,
    nohome_options: NoHomeOptions,
}

impl NoHomeCzPlacement {
    /// Build a `NoHomeCzPlacement` from its three composing pieces.
    pub fn new(
        engine: Arc<SearchEngine>,
        search: MoveSearch,
        nohome_options: NoHomeOptions,
    ) -> Self {
        Self {
            engine,
            search,
            nohome_options,
        }
    }

    /// Borrow the underlying engine.
    pub fn engine(&self) -> &Arc<SearchEngine> {
        &self.engine
    }

    /// Borrow the search configuration.
    pub fn search(&self) -> &MoveSearch {
        &self.search
    }

    /// Borrow the nohome-options bundle.
    pub fn nohome_options(&self) -> &NoHomeOptions {
        &self.nohome_options
    }
}

impl CzPlacement for NoHomeCzPlacement {
    /// `budget.max_expansions` caps each routing solve separately: every
    /// return-phase candidate and the CZ phase each get the full cap (and,
    /// inside a solve, each restart does).
    fn place(
        &self,
        stage: &CzStage<'_>,
        budget: &PlacementBudget,
    ) -> Result<PlacementResult, ConfigError> {
        solve_nohome(
            &self.engine,
            &self.search.options,
            &self.nohome_options,
            stage.initial.iter().copied(),
            stage.pairs,
            stage.blocked.iter().copied(),
            budget.max_expansions,
            stage.future_layers,
        )
        .map(PlacementResult::single)
    }
}

/// Shared implementation backing [`NoHomeCzPlacement`]'s [`CzPlacement::place`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_nohome(
    engine: &SearchEngine,
    opts: &SolveOptions,
    nohome_opts: &NoHomeOptions,
    initial: impl IntoIterator<Item = (u32, LocationAddr)>,
    cz_pairs: &[(u32, u32)],
    blocked: impl IntoIterator<Item = LocationAddr>,
    max_expansions: Option<u32>,
    future_cz_layers: &[Vec<(u32, u32)>],
) -> Result<SolveResult, ConfigError> {
    // Both phases route under entangling-style contention.
    let upgraded_opts = opts.upgraded_for_entangling();
    let opts = &upgraded_opts;

    let root = Config::new(initial)?;
    let blocked_locs: Vec<LocationAddr> = blocked.into_iter().collect();
    let nh_cache = engine.nohome_cache();
    let index = engine.index();

    let has_returners = root
        .iter()
        .any(|(_, loc)| !nh_cache.home_set.contains(&loc.encode()));

    let blocked_set: HashSet<u64> = blocked_locs.iter().map(|l| l.encode()).collect();

    // Helper: the candidate CZ-staging targets for Phase 2 from a config, the
    // rule's first.
    //
    // A pair whose qubits both have a CZ partner is staged by moving one
    // qubit to the other's partner site; the rule picks which, and the other
    // candidates (under `Ranked` and `RouteAll`) vary that choice. A pair
    // with a qubit anywhere else (e.g. a storage zone with no entangling
    // pairs) has no such move, so it goes to a free entangling slot instead.
    // `None` means the rule's own target cannot be placed at all. A
    // non-rule candidate that cannot be placed, or that puts two qubits on
    // one location, is dropped.
    let cz_target_candidates = |from: &Config| -> Option<(Vec<CzTarget>, bool)> {
        let mut options: Vec<((u32, LocationAddr), (u32, LocationAddr))> = Vec::new();
        let mut rule: Vec<bool> = Vec::new();
        let mut unpartnered: Vec<StagePair> = Vec::new();
        for &(c, t) in cz_pairs {
            let c_addr = from.location_of(c)?;
            let t_addr = from.location_of(t)?;
            let (Some(c_dst), Some(t_dst)) = (index.cz_partner(&t_addr), index.cz_partner(&c_addr))
            else {
                unpartnered.push(((c, c_addr), (t, t_addr)));
                continue;
            };
            options.push(((c, c_dst), (t, t_dst)));
            rule.push(c_addr.word_id != t_addr.word_id && index.is_home_position(&t_addr));
        }

        // Whether the candidates cover every way to stage the pairs: every
        // mover assignment, and no pair left to a free-slot assignment that
        // offers only one layout.
        let complete =
            unpartnered.is_empty() && enumerates_all(rule.len(), nohome_opts.max_mover_candidates);
        let assignments = match nohome_opts.mover_selection {
            MoverSelection::Rule => vec![rule],
            MoverSelection::Ranked | MoverSelection::RouteAll => {
                mover_assignments(&rule, nohome_opts.max_mover_candidates)
            }
        };

        let mut candidates = Vec::with_capacity(assignments.len());
        for (n, assignment) in assignments.iter().enumerate() {
            let mut chosen: HashMap<u32, LocationAddr> = HashMap::with_capacity(cz_pairs.len());
            for (&moves_target, &(move_c, move_t)) in assignment.iter().zip(&options) {
                let (qid, dst) = if moves_target { move_t } else { move_c };
                chosen.insert(qid, dst);
            }

            if !unpartnered.is_empty() {
                // A slot is taken if another atom ends on either half, or
                // either half is blocked.
                let slotted: HashSet<u32> = unpartnered
                    .iter()
                    .flat_map(|&((c, _), (t, _))| [c, t])
                    .collect();
                let mut claimed = blocked_set.clone();
                claimed.extend(
                    from.iter()
                        .filter(|(qid, _)| !slotted.contains(qid))
                        .map(|(qid, loc)| chosen.get(&qid).copied().unwrap_or(loc).encode()),
                );
                let dist_table = &engine.entangling_cache().dist_table;
                match assign_free_slots(&unpartnered, &claimed, index, dist_table) {
                    Some(slots) => chosen.extend(slots),
                    None if n == 0 => return None,
                    None => continue,
                }
            }

            let target: Vec<(u32, LocationAddr)> = from
                .iter()
                .map(|(qid, loc)| (qid, chosen.get(&qid).copied().unwrap_or(loc)))
                .collect();
            if n > 0 {
                let mut ends = HashSet::with_capacity(target.len());
                if !target.iter().all(|(_, loc)| ends.insert(loc.encode())) {
                    continue;
                }
            }
            candidates.push(target);
        }
        Some((candidates, complete))
    };

    // Helper: route Phase 2 from `from` to one of `candidates` (the rule's
    // first), chosen per `mover_selection`. Under `Rule` this is exactly one
    // routing solve to the rule's target. Otherwise the result is the chosen
    // candidate's, with the search counters of every routing solve summed in.
    // When none routes, the result describes the stage rather than the rule's
    // target alone: `BudgetExceeded` if any candidate ran out of budget, else
    // `Unsolvable`, and a no-plan proof only when `complete` candidates were
    // all routed and all proved it.
    let route_cz_phase = |from: &Config,
                          candidates: Vec<CzTarget>,
                          complete: bool|
     -> Result<SolveResult, ConfigError> {
        let route = |target: &[(u32, LocationAddr)]| {
            solve_with_engine(
                engine,
                opts,
                None,
                from.iter(),
                target.iter().copied(),
                blocked_locs.iter().copied(),
                max_expansions,
            )
        };
        // The routing order, the rule's candidate always first, and whether
        // one routed candidate is enough.
        let (order, first_solve_wins): (Vec<usize>, bool) = match nohome_opts.mover_selection {
            MoverSelection::Rule => return route(&candidates[0]),
            MoverSelection::RouteAll => ((0..candidates.len()).collect(), false),
            MoverSelection::Ranked => {
                let plan_opts = SolveOptions {
                    strategy: Strategy::PushRotate,
                    backwards_search: false,
                    ..opts.clone()
                };
                let mut planned: Vec<(usize, usize)> = Vec::new();
                for (i, target) in candidates.iter().enumerate() {
                    let plan = solve_with_engine(
                        engine,
                        &plan_opts,
                        None,
                        from.iter(),
                        target.iter().copied(),
                        blocked_locs.iter().copied(),
                        None,
                    )?;
                    if plan.status == SolveStatus::Solved {
                        planned.push((plan.move_layers.len(), i));
                    }
                }
                planned.sort_unstable();
                // When Push and Rotate ranks the rule's candidate first,
                // routing it alone is enough. Otherwise the rule's candidate
                // is routed too, so the ranked pick is kept only if it routes
                // in fewer layers: ranking never does worse than the rule.
                let rule_ranked_first = planned.first().is_some_and(|&(_, i)| i == 0);
                let mut order = vec![0];
                order.extend(planned.into_iter().map(|(_, i)| i).filter(|&i| i != 0));
                (order, rule_ranked_first)
            }
        };

        let mut expanded: u32 = 0;
        let mut generated: u32 = 0;
        let mut best: Option<SolveResult> = None;
        let mut rule_result: Option<SolveResult> = None;
        let mut routed = 0usize;
        let mut any_budget = false;
        let mut all_proven = true;
        for i in order {
            let result = route(&candidates[i])?;
            routed += 1;
            expanded = expanded.saturating_add(result.nodes_expanded);
            generated = generated.saturating_add(result.nodes_generated);
            if result.status != SolveStatus::Solved {
                any_budget |= result.status == SolveStatus::BudgetExceeded;
                all_proven &= result.proven();
            }
            if result.status == SolveStatus::Solved {
                // Strictly fewer layers: a tie keeps the earlier candidate,
                // and the rule's is routed first.
                if best
                    .as_ref()
                    .is_none_or(|b| result.move_layers.len() < b.move_layers.len())
                {
                    best = Some(result);
                }
                // `Ranked` stops at its ranked pick: the first routed
                // candidate other than the rule's, or the rule's itself when
                // it was ranked first.
                if nohome_opts.mover_selection == MoverSelection::Ranked
                    && (i != 0 || first_solve_wins)
                {
                    break;
                }
            } else if i == 0 {
                rule_result = Some(result);
            }
        }
        let mut result = match best {
            Some(solved) => solved,
            None => {
                // The rule's candidate is always routed first, so when
                // nothing solved, its result is set; it keeps the stage's
                // starting configuration and the rule's partial.
                let rule = rule_result.expect("the rule's candidate is always routed");
                let (status, termination) = failed_stage_verdict(
                    any_budget,
                    all_proven,
                    complete && routed == candidates.len(),
                );
                SolveResult {
                    status,
                    termination,
                    ..rule
                }
            }
        };
        result.nodes_expanded = expanded;
        result.nodes_generated = generated;
        Ok(result)
    };

    if !has_returners {
        // Skip the return phase — go directly to fixed-target entangling.
        let Some((cz_targets, complete)) = cz_target_candidates(&root) else {
            return Ok(SolveResult::unsolvable(root));
        };
        return route_cz_phase(&root, cz_targets, complete);
    }

    let occupied_set: HashSet<u64> = root.iter().map(|(_, loc)| loc.encode()).collect();
    let holes: Vec<u64> = nh_cache
        .home_locs
        .iter()
        .filter(|l| !occupied_set.contains(l) && !blocked_set.contains(l))
        .copied()
        .collect();

    let pw = partner_weights(future_cz_layers, nohome_opts.gamma);

    let candidates = candidate_return_layouts(
        &root,
        &nh_cache.home_set,
        &holes,
        &nh_cache.dist_table,
        engine.index(),
        &pw,
        nohome_opts,
    );

    // Phase 1: route every candidate's return layout.
    let mut total_expanded: u32 = 0;
    let mut total_generated: u32 = 0;
    let mut best_p1: Option<SolveResult> = None;
    let mut p1_saw_budget_exceeded = false;
    for candidate in &candidates {
        let return_target: Vec<(u32, LocationAddr)> = candidate.clone();
        let return_result = solve_with_engine(
            engine,
            opts,
            None,
            root.iter(),
            return_target,
            blocked_locs.iter().copied(),
            max_expansions,
        )?;

        total_expanded += return_result.nodes_expanded;
        total_generated = total_generated.saturating_add(return_result.nodes_generated);

        match return_result.status {
            SolveStatus::Solved => {
                if best_p1
                    .as_ref()
                    .is_none_or(|b| return_result.move_layers.len() < b.move_layers.len())
                {
                    best_p1 = Some(return_result);
                }
            }
            SolveStatus::BudgetExceeded => p1_saw_budget_exceeded = true,
            SolveStatus::Unsolvable => {}
        }
    }

    let Some(return_result) = best_p1 else {
        let status = if p1_saw_budget_exceeded {
            SolveStatus::BudgetExceeded
        } else {
            SolveStatus::Unsolvable
        };
        let mut unsolved = SolveResult::unsolved(status, root, total_expanded, 0);
        unsolved.nodes_generated = total_generated;
        return Ok(unsolved);
    };

    // Phase 2: pick the CZ-staging target per `mover_selection`, and route it.
    let Some((cz_targets, complete)) = cz_target_candidates(&return_result.goal_config) else {
        let mut unsolved = SolveResult::unsolved(
            SolveStatus::Unsolvable,
            root,
            total_expanded,
            return_result.deadlocks,
        );
        unsolved.nodes_generated = total_generated;
        return Ok(unsolved);
    };
    let entangling_result = route_cz_phase(&return_result.goal_config, cz_targets, complete)?;

    total_expanded += entangling_result.nodes_expanded;
    total_generated = total_generated.saturating_add(entangling_result.nodes_generated);

    if entangling_result.status == SolveStatus::Solved {
        let total_cost = return_result.cost + entangling_result.cost;
        let mut combined_layers = return_result.move_layers;
        combined_layers.extend(entangling_result.move_layers);
        let mut solved = SolveResult::solved(
            entangling_result.goal_config,
            combined_layers,
            total_cost,
            total_expanded,
            return_result.deadlocks + entangling_result.deadlocks,
        );
        solved.nodes_generated = total_generated;
        return Ok(solved);
    }

    let mut unsolved = SolveResult::unsolved(
        entangling_result.status,
        root,
        total_expanded,
        return_result.deadlocks + entangling_result.deadlocks,
    );
    unsolved.nodes_generated = total_generated;
    Ok(unsolved)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::lane_index::LaneIndex;
    use crate::search::result::SolveStatus;
    use crate::test_utils::{example_arch_json, loc, storage_gate_arch_json};
    use bloqade_lanes_bytecode_core::arch::types::ArchSpec;

    fn make_parts() -> (ArchSpec, LaneIndex) {
        let json = example_arch_json();
        let arch: ArchSpec = serde_json::from_str(json).unwrap();
        let index = LaneIndex::new(arch.clone());
        (arch, index)
    }

    #[test]
    fn mover_assignments_enumerates_all_when_under_the_cap() {
        let rule = [true, false, true];
        let all = mover_assignments(&rule, 8);
        assert_eq!(all.len(), 8);
        assert_eq!(all[0], rule);
        let distinct: HashSet<_> = all.iter().cloned().collect();
        assert_eq!(distinct.len(), 8);
    }

    #[test]
    fn mover_assignments_samples_over_the_cap() {
        let rule = vec![false; 10];
        let some = mover_assignments(&rule, 20);
        assert_eq!(some.len(), 20);
        assert_eq!(some[0], rule);
        // Every single-pair flip comes right after the rule.
        for (i, flip) in some[1..=10].iter().enumerate() {
            assert!(flip.iter().enumerate().all(|(j, &b)| b == (j == i)));
        }
        let distinct: HashSet<_> = some.iter().cloned().collect();
        assert_eq!(distinct.len(), 20);
        // Deterministic.
        assert_eq!(some, mover_assignments(&rule, 20));
        // A cap below the flips keeps the rule and the first flips.
        assert_eq!(mover_assignments(&rule, 3).len(), 3);
    }

    #[test]
    fn test_home_sites_nonempty() {
        let (arch, index) = make_parts();
        let sites = entangling::home_sites(&index);
        assert!(!sites.is_empty(), "should have at least one home site");
        let home_words: HashSet<u32> = arch.left_cz_word_ids().into_iter().collect();
        for &enc in &sites {
            let addr = LocationAddr::decode(enc);
            assert!(
                home_words.contains(&addr.word_id),
                "home site word_id={} not in home words",
                addr.word_id
            );
        }
    }

    /// Neither zone has entangling pairs, so every location in every zone is
    /// home — in particular `(zone 1, word 1)`, the storage end of the zone
    /// bus. The old owner-zone lookup put word 1 in zone 0 only, so an atom
    /// parked in storage looked like a returner and NoHome ran a return phase
    /// towards a hole in the wrong zone.
    #[test]
    fn test_home_sites_span_every_zone() {
        let arch: ArchSpec =
            serde_json::from_str(crate::test_utils::two_zone_bus_arch_json()).unwrap();
        let sites: HashSet<LocationAddr> = entangling::home_sites(&LaneIndex::new(arch))
            .into_iter()
            .map(LocationAddr::decode)
            .collect();
        let at = |zone_id, word_id| LocationAddr {
            zone_id,
            word_id,
            site_id: 0,
        };
        let expected: HashSet<LocationAddr> = [at(0, 0), at(0, 1), at(1, 0), at(1, 1)]
            .into_iter()
            .collect();
        assert_eq!(sites, expected);
    }

    #[test]
    fn test_partner_weights_gamma_decay() {
        let layers = vec![vec![(0, 1), (2, 3)], vec![(0, 2)]];
        let pw = partner_weights(&layers, 0.5);

        // Depth 0: weight=1.0, depth 1: weight=0.5
        assert!((pw[&0][&1] - 1.0).abs() < 1e-9);
        assert!((pw[&0][&2] - 0.5).abs() < 1e-9);
        assert!((pw[&2][&3] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_nearest_home_assigns_all_returners() {
        let (_, index) = make_parts();
        let home_locs = entangling::home_sites(&index);
        let home_set: HashSet<u64> = home_locs.iter().copied().collect();

        // Place qubits at non-home locations (CZ staging).
        let mut non_home = Vec::new();
        for (mt, bus_id, zone_id, dir) in index.bus_groups() {
            for &lane in index.lanes_for(mt, bus_id, zone_id, dir) {
                if let Some((src, _)) = index.endpoints(&lane) {
                    let enc = src.encode();
                    if !home_set.contains(&enc) && !non_home.contains(&enc) {
                        non_home.push(enc);
                    }
                }
            }
        }
        if non_home.len() < 2 {
            return; // skip if arch has no staging sites
        }

        let config = Config::new(vec![
            (0, LocationAddr::decode(non_home[0])),
            (1, LocationAddr::decode(non_home[1])),
        ])
        .unwrap();

        let dist = DistanceTable::new(&home_locs, &index);
        let assignments = nearest_home_layout(&config, &home_set, &home_locs, &dist);
        assert_eq!(assignments.len(), 2, "both qubits should be assigned");
        for &(_, loc) in &assignments {
            assert!(home_set.contains(&loc));
        }
        // No duplicates.
        let assigned_locs: HashSet<u64> = assignments.iter().map(|&(_, l)| l).collect();
        assert_eq!(assigned_locs.len(), 2);
    }

    #[test]
    fn test_candidate_layouts_all_home_is_identity() {
        let (_, index) = make_parts();
        let home_locs = entangling::home_sites(&index);
        let home_set: HashSet<u64> = home_locs.iter().copied().collect();

        // Place qubits at home — should get identity layout back.
        if home_locs.len() < 2 {
            return;
        }
        let config = Config::new(vec![
            (0, LocationAddr::decode(home_locs[0])),
            (1, LocationAddr::decode(home_locs[1])),
        ])
        .unwrap();

        let holes: Vec<u64> = home_locs
            .iter()
            .filter(|l| !config.is_occupied(LocationAddr::decode(**l)))
            .copied()
            .collect();
        let pw = HashMap::new();
        let dist = DistanceTable::new(&home_locs, &index);
        let candidates = candidate_return_layouts(
            &config,
            &home_set,
            &holes,
            &dist,
            &index,
            &pw,
            &NoHomeOptions::default(),
        );
        assert_eq!(candidates.len(), 1);
        // Layout should match original.
        for &(qid, loc) in &candidates[0] {
            assert_eq!(loc, config.location_of(qid).unwrap());
        }
    }

    /// End-to-end smoke: `NoHomeCzPlacement::place` runs the full
    /// two-phase pipeline (return assignment + entangling routing) and reaches
    /// a deterministic terminal verdict without erroring. The toy
    /// `example_arch_json` lacks the distinct home/staging zones the no-home
    /// strategy targets, so this scenario is `Unsolvable`; real-arch solvable
    /// coverage lives in the Python integration tests and benchmarks.
    #[test]
    fn nohome_placement_runs_two_phase_pipeline() {
        let engine = Arc::new(SearchEngine::from_json(example_arch_json()).unwrap());
        let search = MoveSearch::new(SolveOptions::default(), Default::default());
        let placement = NoHomeCzPlacement::new(engine, search, NoHomeOptions::default());

        // Initial: qubit 0 at a non-home position (forces the return phase),
        // qubit 1 already at home.
        let initial = [(0u32, loc(0, 5)), (1u32, loc(0, 1))];
        let cz_pairs = [(0u32, 1u32)];
        let blocked: [LocationAddr; 0] = [];

        let result = placement
            .place(
                &CzStage::new(&initial, &cz_pairs, &blocked),
                &PlacementBudget::new(Some(5000)),
            )
            .map(|placed| placed.result)
            .unwrap();

        assert_eq!(result.status, SolveStatus::Unsolvable);
    }

    /// One CZ stage on the example arch under `selection`, routed with
    /// `strategy` and a `budget` per solve.
    fn place_example(
        initial: &[(u32, LocationAddr)],
        selection: MoverSelection,
        strategy: Strategy,
        budget: u32,
    ) -> SolveResult {
        let engine = Arc::new(SearchEngine::from_json(example_arch_json()).unwrap());
        let search = MoveSearch::new(
            SolveOptions {
                strategy,
                ..SolveOptions::default()
            },
            Default::default(),
        );
        let options = NoHomeOptions {
            mover_selection: selection,
            ..NoHomeOptions::default()
        };
        let blocked: [LocationAddr; 0] = [];
        NoHomeCzPlacement::new(engine, search, options)
            .place(
                &CzStage::new(initial, &[(0, 1)], &blocked),
                &PlacementBudget::new(Some(budget)),
            )
            .unwrap()
            .result
    }

    /// When no candidate routes, the verdict describes the stage, not any one
    /// candidate's target: one candidate out of budget makes the stage
    /// `BudgetExceeded`, however the others failed, and a no-plan proof needs
    /// every way to stage the pairs routed and proved.
    #[test]
    fn a_failed_stage_verdict_describes_the_stage() {
        use SolveStatus::{BudgetExceeded, Unsolvable};
        let budget = (BudgetExceeded, Termination::Budget);
        let unproven = (Unsolvable, Termination::Exhausted { proof: false });
        let proven = (Unsolvable, Termination::Exhausted { proof: true });
        // (any_budget, all_proven, exhaustive) -> verdict
        assert_eq!(failed_stage_verdict(true, true, true), budget);
        assert_eq!(failed_stage_verdict(true, false, false), budget);
        assert_eq!(failed_stage_verdict(false, true, true), proven);
        assert_eq!(failed_stage_verdict(false, true, false), unproven);
        assert_eq!(failed_stage_verdict(false, false, true), unproven);
    }

    /// A no-plan proof survives only when every way to stage the pair was
    /// routed and each proved it. Sites 0 and 1 can never pair on the
    /// example arch, and Push and Rotate proves each candidate unroutable:
    /// `RouteAll` routes both and keeps the proof; `Ranked` routes only the
    /// rule's (Push and Rotate plans neither), so it drops it.
    #[test]
    fn a_no_plan_proof_needs_every_candidate() {
        let initial = [(0u32, loc(0, 0)), (1u32, loc(0, 1))];
        let all = place_example(
            &initial,
            MoverSelection::RouteAll,
            Strategy::PushRotate,
            5000,
        );
        assert_eq!(all.status, SolveStatus::Unsolvable);
        assert!(all.proven(), "{:?}", all.termination);

        let ranked = place_example(&initial, MoverSelection::Ranked, Strategy::PushRotate, 5000);
        assert_eq!(ranked.status, SolveStatus::Unsolvable);
        assert!(!ranked.proven(), "{:?}", ranked.termination);
    }

    fn zloc(zone_id: u32, word_id: u32, site_id: u32) -> LocationAddr {
        LocationAddr {
            zone_id,
            word_id,
            site_id,
        }
    }

    fn solve_stage(
        arch_json: &str,
        initial: &[(u32, LocationAddr)],
        cz_pairs: &[(u32, u32)],
    ) -> (Arc<SearchEngine>, SolveResult) {
        let engine = Arc::new(SearchEngine::from_json_validated(arch_json).unwrap());
        let search = MoveSearch::new(SolveOptions::default(), Default::default());
        let placement = NoHomeCzPlacement::new(engine.clone(), search, NoHomeOptions::default());
        let blocked: [LocationAddr; 0] = [];
        let result = placement
            .place(
                &CzStage::new(initial, cz_pairs, &blocked),
                &PlacementBudget::new(Some(5000)),
            )
            .map(|placed| placed.result)
            .unwrap();
        (engine, result)
    }

    /// A pair in a zone with no entangling pairs has no per-pair move, so it
    /// must be staged in the gate zone. It used to be skipped: every target
    /// defaulted to the current site and the stage "solved" in zero layers
    /// with the pair still apart in storage. Covers both qubits in storage;
    /// one in storage with its partner on home gate word 1; and one in
    /// storage with its partner on non-home gate word 2, which runs Phase 1
    /// first, so the fallback resolves from the returned layout.
    #[test]
    fn nohome_stages_a_pair_from_a_pairless_zone() {
        let starts = [
            [(0u32, zloc(0, 0, 0)), (1u32, zloc(0, 0, 1))],
            [(0u32, zloc(0, 0, 0)), (1u32, zloc(1, 1, 1))],
            [(0u32, zloc(0, 0, 0)), (1u32, zloc(1, 2, 1))],
        ];
        for initial in starts {
            let (engine, result) = solve_stage(storage_gate_arch_json(), &initial, &[(0, 1)]);

            assert_eq!(result.status, SolveStatus::Solved, "from {initial:?}");
            assert!(!result.move_layers.is_empty(), "from {initial:?}");
            let c = result.goal_config.location_of(0).unwrap();
            let t = result.goal_config.location_of(1).unwrap();
            assert_eq!(
                engine.index().cz_partner(&t),
                Some(c),
                "from {initial:?}: pair ends at {c:?} and {t:?}, not on CZ partner sites",
            );
        }
    }

    /// A pair that cannot be staged makes the stage unsolvable, not solved in
    /// place. Two ways: spectators on gate word 1 claim a half of every slot,
    /// and a pair names a qubit the placement does not hold.
    #[test]
    fn nohome_is_unsolvable_when_a_pair_cannot_be_staged() {
        let in_storage = [(0u32, zloc(0, 0, 0)), (1u32, zloc(0, 0, 1))];
        let spectators = [(2u32, zloc(1, 1, 0)), (3u32, zloc(1, 1, 1))];
        let full_gate: Vec<_> = in_storage.iter().chain(&spectators).copied().collect();

        let (_, gate_full) = solve_stage(storage_gate_arch_json(), &full_gate, &[(0, 1)]);
        assert_eq!(gate_full.status, SolveStatus::Unsolvable);
        assert!(gate_full.move_layers.is_empty());

        let (_, missing) = solve_stage(storage_gate_arch_json(), &in_storage, &[(0, 9)]);
        assert_eq!(missing.status, SolveStatus::Unsolvable);
        assert!(missing.move_layers.is_empty());
    }

    /// The same verdict when the slots fill only after Phase 1: the
    /// spectator on non-home word 2 returns to the last free home site,
    /// (1, 1, 1), and with (1, 1, 0) already taken no slot is left free.
    #[test]
    fn nohome_is_unsolvable_when_phase_1_fills_the_slots() {
        let initial = [
            (0u32, zloc(0, 0, 0)),
            (1u32, zloc(0, 0, 1)),
            (2u32, zloc(1, 1, 0)),
            (3u32, zloc(1, 2, 1)),
        ];
        let (_, result) = solve_stage(storage_gate_arch_json(), &initial, &[(0, 1)]);

        assert_eq!(result.status, SolveStatus::Unsolvable);
        assert!(result.move_layers.is_empty());
        assert!(result.nodes_expanded > 0, "Phase 1 should have routed");
    }

    /// The storage/gate spec with a third gate word and **overlapping**
    /// entangling pairs `[1, 2]` and `[2, 3]`, which validation allows.
    fn overlapping_pairs_arch_json() -> String {
        let mut spec: serde_json::Value = serde_json::from_str(storage_gate_arch_json()).unwrap();
        spec["words"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({ "sites": [[0, 3], [1, 3]] }));
        for zone in spec["zones"].as_array_mut().unwrap() {
            zone["grid"]["y_spacing"] = serde_json::json!([2.0, 2.0, 2.0]);
        }
        let gate = &mut spec["zones"][1];
        gate["word_buses"] = serde_json::json!([
            { "src": [1], "dst": [2] },
            { "src": [2], "dst": [3] }
        ]);
        gate["words_with_site_buses"] = serde_json::json!([1, 2, 3]);
        gate["entangling_pairs"] = serde_json::json!([[1, 2], [2, 3]]);
        spec.to_string()
    }

    /// With overlapping entangling pairs two slots can share a half. Here the
    /// cheapest assignment puts `[1, 2]` and `[2, 3]` both at site 0, sharing
    /// (1, 2, 0), so the stage is unplaceable rather than given colliding
    /// targets. The distance table is built directly: the engine's entangling
    /// cache debug-asserts one partner per location, which this spec breaks.
    #[test]
    fn free_slots_reject_a_shared_half_across_overlapping_pairs() {
        let engine = SearchEngine::from_json_validated(&overlapping_pairs_arch_json()).unwrap();
        let index = engine.index();
        let dist_table = DistanceTable::new(&entangling::all_entangling_locations(index), index);
        let pairs = [
            ((0u32, zloc(0, 0, 0)), (1u32, zloc(0, 0, 0))),
            ((2u32, zloc(0, 0, 0)), (3u32, zloc(0, 0, 0))),
        ];
        // Take site 1 of `[1, 2]` so the collision is the cheapest answer.
        let claimed = HashSet::from([zloc(1, 1, 1).encode()]);

        assert!(assign_free_slots(&pairs, &claimed, index, &dist_table).is_none());

        // With `[2, 3]` at site 0 taken too, the only disjoint choice is left.
        let claimed = HashSet::from([zloc(1, 1, 1).encode(), zloc(1, 3, 0).encode()]);
        let targets = assign_free_slots(&pairs, &claimed, index, &dist_table).unwrap();
        let distinct: HashSet<LocationAddr> = targets.iter().map(|&(_, l)| l).collect();
        assert_eq!(distinct.len(), 4, "colliding targets: {targets:?}");
    }
}
