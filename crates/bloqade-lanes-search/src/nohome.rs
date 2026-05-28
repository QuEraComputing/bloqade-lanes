//! No-home return assignment for two-phase placement.
//!
//! After a CZ layer, atoms sit at CZ-staging positions. Instead of returning
//! them to their *original* home sites, this module finds an optimal
//! assignment of displaced qubits to *any* available home site, minimising a
//! blend of immediate routing cost and future CZ-partner proximity.
//!
//! The assignment is solved with the Hungarian algorithm (reused from
//! [`crate::entangling`]), making it polynomial and architecture-general.

use std::collections::{HashMap, HashSet};

use bloqade_lanes_bytecode_core::arch::addr::{Direction, LocationAddr, MoveType};
use bloqade_lanes_bytecode_core::arch::types::ArchSpec;

use crate::config::Config;
use crate::entangling;
use crate::entropy::find_path_occupied;
use crate::heuristic::DistanceTable;
use crate::lane_index::LaneIndex;

/// Lane-signature triple: identifies a parallelisable bus group.
type LaneSig = (MoveType, u32, Direction);

/// Per-edge scoring entry in `candidate_return_layouts`:
/// `(score, hole_index, hop_cost, sig_set)`.
type EdgeScore = (f64, usize, u32, HashSet<LaneSig>);

// ── Options ───────────────────────────────────────────────────────

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
}

impl Default for NoHomeOptions {
    fn default() -> Self {
        Self {
            gamma: 0.85,
            lambda_lookahead: 0.5,
            k_candidates: 8,
            top_bus_signatures: 6,
            bus_reward_rho: 1,
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────

/// Enumerate all home-site locations (encoded) from the architecture.
///
/// A home site is any `(zone_id, word_id, site_id)` where `word_id` is
/// in [`ArchSpec::left_cz_word_ids`].
pub fn home_sites(arch: &ArchSpec) -> Vec<u64> {
    let home_words = arch.left_cz_word_ids();
    let word_zone = arch.word_zone_map();
    let sites_per_word = arch.sites_per_word() as u32;
    let mut result = Vec::new();
    for &word_id in &home_words {
        let zone_id = *word_zone.get(&word_id).unwrap_or(&0);
        for site_id in 0..sites_per_word {
            result.push(
                LocationAddr {
                    zone_id,
                    word_id,
                    site_id,
                }
                .encode(),
            );
        }
    }
    result
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lane_index::LaneIndex;
    use crate::test_utils::example_arch_json;

    fn make_parts() -> (ArchSpec, LaneIndex) {
        let json = example_arch_json();
        let arch: ArchSpec = serde_json::from_str(json).unwrap();
        let index = LaneIndex::new(arch.clone());
        (arch, index)
    }

    #[test]
    fn test_home_sites_nonempty() {
        let (arch, _) = make_parts();
        let sites = home_sites(&arch);
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
        let (arch, index) = make_parts();
        let home_locs = home_sites(&arch);
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
        let (arch, index) = make_parts();
        let home_locs = home_sites(&arch);
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
}
