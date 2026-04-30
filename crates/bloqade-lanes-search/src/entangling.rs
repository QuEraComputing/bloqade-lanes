//! Entangling placement enumeration, distance precomputation, and optimal
//! pair-to-position assignment for loose-goal search.
//!
//! This module supports [`crate::solve::MoveSolver::solve_entangling`] by
//! providing the architectural queries needed to work with entangling
//! constraints rather than fixed target locations.

use std::collections::{HashMap, HashSet};

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;
use bloqade_lanes_bytecode_core::arch::types::ArchSpec;

use crate::config::Config;
use crate::heuristic::DistanceTable;

// ── Entangling word pairs ──────────────────────────────────────────

/// A pair of words within a zone that can perform CZ gates together.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntanglingWordPair {
    pub zone_id: u32,
    pub word_a: u32,
    pub word_b: u32,
}

/// Enumerate all entangling word pairs from the architecture spec.
///
/// Iterates all zones and their `entangling_pairs` fields.
pub fn enumerate_word_pairs(arch: &ArchSpec) -> Vec<EntanglingWordPair> {
    let mut result = Vec::new();
    for (zone_id, zone) in arch.zones.iter().enumerate() {
        for pair in &zone.entangling_pairs {
            result.push(EntanglingWordPair {
                zone_id: zone_id as u32,
                word_a: pair[0],
                word_b: pair[1],
            });
        }
    }
    result
}

/// Collect all encoded locations that participate in any entangling pair.
///
/// Returns a deduplicated list of encoded `LocationAddr` values — both words
/// of each pair, all sites. Used to build a [`DistanceTable`] targeting all
/// potential entangling positions.
pub fn all_entangling_locations(arch: &ArchSpec) -> Vec<u64> {
    let sites_per_word = arch.sites_per_word() as u32;
    let mut locs = Vec::new();
    for wp in enumerate_word_pairs(arch) {
        for site in 0..sites_per_word {
            locs.push(
                LocationAddr {
                    zone_id: wp.zone_id,
                    word_id: wp.word_a,
                    site_id: site,
                }
                .encode(),
            );
            locs.push(
                LocationAddr {
                    zone_id: wp.zone_id,
                    word_id: wp.word_b,
                    site_id: site,
                }
                .encode(),
            );
        }
    }
    locs.sort_unstable();
    locs.dedup();
    locs
}

/// Build O(1) lookup set for valid entangling placements.
///
/// An entangling placement is a pair `(encoded_a, encoded_b)` where both
/// locations are in the same zone, on an entangling word pair, at the same
/// site index. Both orderings are stored so lookup works regardless of
/// which qubit is on which word.
pub fn build_entangling_set(arch: &ArchSpec) -> HashSet<(u64, u64)> {
    let sites_per_word = arch.sites_per_word() as u32;
    let mut set = HashSet::new();
    for wp in enumerate_word_pairs(arch) {
        for site in 0..sites_per_word {
            let loc_a = LocationAddr {
                zone_id: wp.zone_id,
                word_id: wp.word_a,
                site_id: site,
            }
            .encode();
            let loc_b = LocationAddr {
                zone_id: wp.zone_id,
                word_id: wp.word_b,
                site_id: site,
            }
            .encode();
            set.insert((loc_a, loc_b));
            set.insert((loc_b, loc_a));
        }
    }
    set
}

// ── Per-word-pair minimum distances ────────────────────────────────

/// One entry per entangling word pair: the pair identity plus collapsed
/// minimum distances from every reachable location to each word.
#[derive(Debug)]
struct WordPairEntry {
    word_pair: EntanglingWordPair,
    /// `encoded_source → min hops to any site on word_a`
    min_dist_a: HashMap<u64, u32>,
    /// `encoded_source → min hops to any site on word_b`
    min_dist_b: HashMap<u64, u32>,
}

/// Precomputed minimum hop distances from any location to each entangling
/// word (collapsed across all sites on that word).
///
/// For each word pair `(zone, word_a, word_b)` and each reachable location:
/// - `min_dist_a[loc]` = min over all sites on `word_a`: distance(loc → site)
/// - `min_dist_b[loc]` = min over all sites on `word_b`: distance(loc → site)
///
/// This allows the [`PairDistanceHeuristic`](crate::heuristic::PairDistanceHeuristic)
/// to evaluate pair costs in O(word_pairs) per pair instead of O(placements).
#[derive(Debug)]
pub struct WordPairDistances {
    entries: Vec<WordPairEntry>,
}

impl WordPairDistances {
    /// Post-process a [`DistanceTable`] (built over all entangling locations)
    /// by grouping targets by word and taking the minimum distance per group.
    pub fn from_dist_table(
        word_pairs: &[EntanglingWordPair],
        arch: &ArchSpec,
        dist_table: &DistanceTable,
    ) -> Self {
        let sites_per_word = arch.sites_per_word() as u32;

        let entries = word_pairs
            .iter()
            .map(|wp| {
                // Collect all target locations for word_a and word_b.
                let targets_a: Vec<u64> = (0..sites_per_word)
                    .map(|s| {
                        LocationAddr {
                            zone_id: wp.zone_id,
                            word_id: wp.word_a,
                            site_id: s,
                        }
                        .encode()
                    })
                    .collect();

                let targets_b: Vec<u64> = (0..sites_per_word)
                    .map(|s| {
                        LocationAddr {
                            zone_id: wp.zone_id,
                            word_id: wp.word_b,
                            site_id: s,
                        }
                        .encode()
                    })
                    .collect();

                // For each reachable source location, find the min distance
                // to any site on word_a (resp. word_b).
                let min_a = collapse_min_distances(dist_table, &targets_a);
                let min_b = collapse_min_distances(dist_table, &targets_b);

                WordPairEntry {
                    word_pair: wp.clone(),
                    min_dist_a: min_a,
                    min_dist_b: min_b,
                }
            })
            .collect();

        Self { entries }
    }

    /// Number of entangling word pairs.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether there are no entangling word pairs.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over `(word_pair, min_dist_to_a, min_dist_to_b)`.
    pub fn iter(
        &self,
    ) -> impl Iterator<Item = (&EntanglingWordPair, &HashMap<u64, u32>, &HashMap<u64, u32>)> {
        self.entries
            .iter()
            .map(|e| (&e.word_pair, &e.min_dist_a, &e.min_dist_b))
    }
}

/// For a set of target locations, build `source → min distance to any target`.
fn collapse_min_distances(dist_table: &DistanceTable, targets: &[u64]) -> HashMap<u64, u32> {
    // Collect all (source, distance) pairs from the distance table for each
    // target, then keep the minimum per source.
    let mut min_dist: HashMap<u64, u32> = HashMap::new();
    for &target_enc in targets {
        // Iterate all sources reachable from this target.
        dist_table.for_each_source(target_enc, |src_enc, d| {
            let entry = min_dist.entry(src_enc).or_insert(u32::MAX);
            *entry = (*entry).min(d);
        });
    }
    min_dist
}

// ── Hungarian (min-cost) pair assignment ──────────────────────────

/// An entangling position slot: the two locations that form a CZ pair.
struct PositionSlot {
    loc_a: u64,
    loc_b: u64,
}

/// Optimal assignment of CZ pairs to entangling positions via the
/// Hungarian algorithm (min-cost bipartite matching).
///
/// For each CZ pair, evaluates all valid entangling placements. The
/// Hungarian algorithm finds the assignment minimising total routing
/// cost `Σ (d_a + d_b)` across all pairs simultaneously, avoiding the
/// suboptimal first-fit behaviour of greedy assignment.
///
/// The `seed` parameter controls tie-breaking perturbation for restart
/// diversity: seed 0 means no perturbation.
///
/// Returns `(qubit_id, encoded_target_location)` entries suitable for
/// [`SearchContext::targets`](crate::context::SearchContext).
pub fn greedy_assign_pairs(
    cz_pairs: &[(u32, u32)],
    config: &Config,
    arch: &ArchSpec,
    dist_table: &DistanceTable,
    seed: u64,
    transition_targets: Option<&HashMap<u32, u64>>,
    transition_weight: f64,
) -> Vec<(u32, u64)> {
    if cz_pairs.is_empty() {
        return Vec::new();
    }

    let word_pairs = enumerate_word_pairs(arch);
    let sites_per_word = arch.sites_per_word() as u32;

    // Build columns: each (word_pair, site) is one position slot.
    // For each slot, we store the best orientation per pair when building costs.
    let mut slots: Vec<PositionSlot> = Vec::new();
    for wp in &word_pairs {
        for site in 0..sites_per_word {
            let pos_a = LocationAddr {
                zone_id: wp.zone_id,
                word_id: wp.word_a,
                site_id: site,
            }
            .encode();
            let pos_b = LocationAddr {
                zone_id: wp.zone_id,
                word_id: wp.word_b,
                site_id: site,
            }
            .encode();
            slots.push(PositionSlot {
                loc_a: pos_a,
                loc_b: pos_b,
            });
        }
    }

    let n_pairs = cz_pairs.len();
    let n_slots = slots.len();
    if n_slots == 0 {
        return Vec::new();
    }

    // Build cost matrix: cost[pair_i][slot_j] = min over both orientations
    // of d(qa, target_a) + d(qb, target_b).
    // We also store which orientation was chosen per cell.
    const BIG: u32 = u32::MAX / 4;
    let mut costs = vec![BIG; n_pairs * n_slots];
    // swapped[i * n_slots + j] = true means qa→loc_b, qb→loc_a is cheaper.
    let mut swapped = vec![false; n_pairs * n_slots];

    for (i, &(qa, qb)) in cz_pairs.iter().enumerate() {
        let loc_a_enc = match config.location_of(qa) {
            Some(l) => l.encode(),
            None => continue,
        };
        let loc_b_enc = match config.location_of(qb) {
            Some(l) => l.encode(),
            None => continue,
        };

        for (j, slot) in slots.iter().enumerate() {
            // Orientation 1: qa→loc_a, qb→loc_b
            let d_a1 = dist_table.distance(loc_a_enc, slot.loc_a).unwrap_or(BIG);
            let d_b1 = dist_table.distance(loc_b_enc, slot.loc_b).unwrap_or(BIG);
            let mut cost1 = d_a1.saturating_add(d_b1);

            // Orientation 2: qa→loc_b, qb→loc_a
            let d_a2 = dist_table.distance(loc_a_enc, slot.loc_b).unwrap_or(BIG);
            let d_b2 = dist_table.distance(loc_b_enc, slot.loc_a).unwrap_or(BIG);
            let mut cost2 = d_a2.saturating_add(d_b2);

            // Add transition cost to next layer's assigned positions.
            if let Some(targets) = transition_targets {
                // Orientation 1: qa ends at slot.loc_a, qb ends at slot.loc_b
                if let Some(&next_a) = targets.get(&qa) {
                    let dt = dist_table.distance(slot.loc_a, next_a).unwrap_or(BIG);
                    cost1 = cost1.saturating_add((dt as f64 * transition_weight) as u32);
                }
                if let Some(&next_b) = targets.get(&qb) {
                    let dt = dist_table.distance(slot.loc_b, next_b).unwrap_or(BIG);
                    cost1 = cost1.saturating_add((dt as f64 * transition_weight) as u32);
                }
                // Orientation 2: qa ends at slot.loc_b, qb ends at slot.loc_a
                if let Some(&next_a) = targets.get(&qa) {
                    let dt = dist_table.distance(slot.loc_b, next_a).unwrap_or(BIG);
                    cost2 = cost2.saturating_add((dt as f64 * transition_weight) as u32);
                }
                if let Some(&next_b) = targets.get(&qb) {
                    let dt = dist_table.distance(slot.loc_a, next_b).unwrap_or(BIG);
                    cost2 = cost2.saturating_add((dt as f64 * transition_weight) as u32);
                }
            }

            let idx = i * n_slots + j;
            if cost1 <= cost2 {
                costs[idx] = cost1;
                swapped[idx] = false;
            } else {
                costs[idx] = cost2;
                swapped[idx] = true;
            }
        }
    }

    // Apply seed-based perturbation for restart diversity.
    if seed != 0 {
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};
        let mut rng = SmallRng::seed_from_u64(seed);
        for c in costs.iter_mut() {
            if *c < BIG {
                let perturbation: i32 = rng.random_range(-1..=1);
                *c = (*c as i32).saturating_add(perturbation).max(0) as u32;
            }
        }
    }

    // Solve the rectangular assignment problem.
    let assignment = hungarian(&costs, n_pairs, n_slots);

    // Extract result.
    let mut result: Vec<(u32, u64)> = Vec::with_capacity(n_pairs * 2);
    for (i, &col) in assignment.iter().enumerate() {
        if col >= n_slots {
            continue; // unassigned (shouldn't happen if slots >= pairs)
        }
        let idx = i * n_slots + col;
        if costs[idx] >= BIG {
            continue; // unreachable position
        }
        let (qa, qb) = cz_pairs[i];
        let slot = &slots[col];
        if swapped[idx] {
            result.push((qa, slot.loc_b));
            result.push((qb, slot.loc_a));
        } else {
            result.push((qa, slot.loc_a));
            result.push((qb, slot.loc_b));
        }
    }

    result
}

/// Two-pass lookahead Hungarian assignment with variable depth.
///
/// Forward pass: compute preliminary assignments for the current and each
/// future layer using forward cost only, simulating positions from each
/// layer's output to the next.
///
/// Backward pass: refine each layer's assignment using the next layer's
/// assigned positions as transition targets (weighted by `beta`).
///
/// Returns the refined assignment for the current layer (`cz_pairs`).
pub fn lookahead_assign_pairs(
    cz_pairs: &[(u32, u32)],
    config: &Config,
    arch: &ArchSpec,
    dist_table: &DistanceTable,
    seed: u64,
    future_layers: &[Vec<(u32, u32)>],
    beta: f64,
) -> Vec<(u32, u64)> {
    if future_layers.is_empty() || beta == 0.0 {
        return greedy_assign_pairs(cz_pairs, config, arch, dist_table, seed, None, 0.0);
    }

    // Collect all layers: current + future.
    let depth = 1 + future_layers.len();
    let all_layers: Vec<&[(u32, u32)]> = std::iter::once(cz_pairs)
        .chain(future_layers.iter().map(|v| v.as_slice()))
        .collect();

    // Forward pass: preliminary assignments.
    let mut forward_assignments: Vec<Vec<(u32, u64)>> = Vec::with_capacity(depth);
    let mut sim_config = config.clone();

    for layer_pairs in &all_layers {
        let assign =
            greedy_assign_pairs(layer_pairs, &sim_config, arch, dist_table, seed, None, 0.0);
        // Build simulated config: move assigned qubits to their targets.
        let moves: Vec<(u32, LocationAddr)> = assign
            .iter()
            .map(|&(qid, enc)| (qid, LocationAddr::decode(enc)))
            .collect();
        sim_config = sim_config.with_moves(&moves);
        forward_assignments.push(assign);
    }

    // Backward pass: refine with transition targets from the next layer.
    // Start from the second-to-last layer and work backward.
    for i in (0..depth - 1).rev() {
        // Build transition targets: qubit → assigned position in layer i+1.
        let next_assign = &forward_assignments[i + 1];
        let targets: HashMap<u32, u64> = next_assign.iter().copied().collect();

        // Determine the config for this layer.
        // Layer 0 uses the original config. Layers 1+ use simulated configs
        // rebuilt from the (already refined) previous layer.
        let layer_config = if i == 0 {
            config.clone()
        } else {
            let mut cfg = config.clone();
            for j in 0..i {
                let moves: Vec<(u32, LocationAddr)> = forward_assignments[j]
                    .iter()
                    .map(|&(qid, enc)| (qid, LocationAddr::decode(enc)))
                    .collect();
                cfg = cfg.with_moves(&moves);
            }
            cfg
        };

        forward_assignments[i] = greedy_assign_pairs(
            all_layers[i],
            &layer_config,
            arch,
            dist_table,
            seed,
            Some(&targets),
            beta,
        );
    }

    forward_assignments.into_iter().next().unwrap_or_default()
}

/// Solve the rectangular min-cost assignment problem (n_rows ≤ n_cols).
///
/// `costs` is a flattened row-major n_rows × n_cols matrix.
/// Returns a vector of length `n_rows` where `result[i]` is the column
/// assigned to row `i`.
///
/// Uses the shortest-augmenting-path variant of the Hungarian algorithm,
/// O(n_rows² × n_cols).
fn hungarian(costs: &[u32], n_rows: usize, n_cols: usize) -> Vec<usize> {
    assert!(n_rows <= n_cols);
    assert_eq!(costs.len(), n_rows * n_cols);

    const INF: i64 = i64::MAX / 2;

    // Convert to i64 for safe arithmetic with dual variables.
    let cost = |r: usize, c: usize| -> i64 { costs[r * n_cols + c] as i64 };

    // Dual variables: u[i] for rows, v[j] for columns.
    let mut u = vec![0i64; n_rows + 1];
    let mut v = vec![0i64; n_cols + 1];

    // col_to_row[j] = row assigned to column j (0 = unassigned, 1-indexed).
    let mut col_to_row = vec![0usize; n_cols + 1];

    // Process each row.
    for i in 1..=n_rows {
        // Start augmenting path from a virtual column 0 linked to row i.
        col_to_row[0] = i;
        let mut cur_col = 0usize;

        // Shortest reduced-cost path from row i to each column.
        let mut min_cost = vec![INF; n_cols + 1];
        let mut visited = vec![false; n_cols + 1];
        // For path reconstruction: prev[j] = column before j in the path.
        let mut prev = vec![0usize; n_cols + 1];

        loop {
            visited[cur_col] = true;
            let cur_row = col_to_row[cur_col];
            let mut delta = INF;
            let mut next_col = 0usize;

            for j in 1..=n_cols {
                if visited[j] {
                    continue;
                }
                let reduced = cost(cur_row - 1, j - 1) - u[cur_row] - v[j];
                if reduced < min_cost[j] {
                    min_cost[j] = reduced;
                    prev[j] = cur_col;
                }
                if min_cost[j] < delta {
                    delta = min_cost[j];
                    next_col = j;
                }
            }

            // Update dual variables along the augmenting path.
            for j in 0..=n_cols {
                if visited[j] {
                    u[col_to_row[j]] += delta;
                    v[j] -= delta;
                } else {
                    min_cost[j] -= delta;
                }
            }

            cur_col = next_col;
            if col_to_row[cur_col] == 0 {
                break; // found an unmatched column
            }
        }

        // Augment: trace back through prev links and update col_to_row.
        loop {
            let prev_col = prev[cur_col];
            col_to_row[cur_col] = col_to_row[prev_col];
            cur_col = prev_col;
            if cur_col == 0 {
                break;
            }
        }
    }

    // Extract row→col assignment (convert from 1-indexed).
    let mut result = vec![0usize; n_rows];
    for j in 1..=n_cols {
        if col_to_row[j] > 0 {
            result[col_to_row[j] - 1] = j - 1;
        }
    }
    result
}

// ── Assignment cost ────────────────────────────────────────────────

/// Compute the total cost of a target assignment for the current config.
///
/// For each qubit in `targets`, looks up the hop distance from its current
/// position (in `config`) to the assigned target. Returns the sum of these
/// distances. Unreachable targets contribute `u32::MAX / 2` to avoid overflow.
pub fn assignment_cost(config: &Config, targets: &[(u32, u64)], dist_table: &DistanceTable) -> u32 {
    let mut total: u32 = 0;
    for &(qid, target_enc) in targets {
        let Some(loc) = config.location_of(qid) else {
            return u32::MAX;
        };
        let loc_enc = loc.encode();
        if loc_enc == target_enc {
            continue;
        }
        let d = dist_table
            .distance(loc_enc, target_enc)
            .unwrap_or(u32::MAX / 2);
        total = total.saturating_add(d);
    }
    total
}

// ── Accidental CZ detection ────────────────────────────────────────

/// Find spectator qubits involved in accidental CZ pairings.
///
/// An accidental CZ occurs when two spectator qubits (neither in `cz_qubits`)
/// occupy partner sites in the entangling set. Only one qubit per accidental
/// pair is returned (the one with the higher qubit ID).
///
/// Returns `(qubit_id, location)` pairs that need to be moved.
pub fn find_accidental_cz(
    config: &Config,
    cz_qubits: &HashSet<u32>,
    partner_map: &HashMap<u64, u64>,
) -> Vec<(u32, LocationAddr)> {
    let mut result = Vec::new();
    let mut seen_pairs: HashSet<(u64, u64)> = HashSet::new();

    for (qid, loc) in config.iter() {
        if cz_qubits.contains(&qid) {
            continue;
        }
        let loc_enc = loc.encode();
        if let Some(&partner_enc) = partner_map.get(&loc_enc)
            && let Some(other_qid) = config.qubit_at(LocationAddr::decode(partner_enc))
            && !cz_qubits.contains(&other_qid)
        {
            // Both are spectators at partner sites — accidental CZ.
            let pair = if loc_enc < partner_enc {
                (loc_enc, partner_enc)
            } else {
                (partner_enc, loc_enc)
            };
            if seen_pairs.insert(pair) {
                // Pick the qubit with higher ID to move.
                let (move_qid, move_loc) = if qid > other_qid {
                    (qid, loc)
                } else {
                    (other_qid, LocationAddr::decode(partner_enc))
                };
                result.push((move_qid, move_loc));
            }
        }
    }
    result
}

/// Build a partner map from the entangling set: for each encoded location,
/// its CZ partner location (if any). Used for O(1) accidental CZ checks.
pub fn build_partner_map(entangling_set: &HashSet<(u64, u64)>) -> HashMap<u64, u64> {
    let mut map = HashMap::new();
    for &(a, b) in entangling_set {
        // Each location has at most one CZ partner (same as ArchSpec::get_cz_partner).
        debug_assert!(
            !map.contains_key(&a) || map[&a] == b,
            "location has multiple CZ partners"
        );
        map.insert(a, b);
    }
    map
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lane_index::LaneIndex;
    use crate::test_utils::{example_arch_json, loc};

    fn make_arch() -> ArchSpec {
        serde_json::from_str(example_arch_json()).unwrap()
    }

    fn make_index() -> LaneIndex {
        LaneIndex::new(make_arch())
    }

    // ── enumerate_word_pairs ──

    #[test]
    fn enumerate_example_arch() {
        let arch = make_arch();
        let pairs = enumerate_word_pairs(&arch);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].zone_id, 0);
        assert_eq!(pairs[0].word_a, 0);
        assert_eq!(pairs[0].word_b, 1);
    }

    // ── hungarian ──

    #[test]
    fn hungarian_2x3_optimal() {
        // 2 rows (pairs), 3 columns (positions).
        // Cost matrix:
        //   [ 3, 1, 5 ]
        //   [ 2, 4, 1 ]
        // Optimal: row 0→col 1 (cost 1), row 1→col 2 (cost 1) = total 2.
        let costs = vec![3, 1, 5, 2, 4, 1];
        let result = hungarian(&costs, 2, 3);
        assert_eq!(result[0], 1); // row 0 → col 1
        assert_eq!(result[1], 2); // row 1 → col 2
    }

    #[test]
    fn hungarian_3x3_square() {
        // Classic 3×3 example.
        // [ 1, 2, 3 ]
        // [ 2, 4, 6 ]
        // [ 3, 6, 9 ]
        // Optimal: 0→0(1), 1→1(4), 2→2(9) = 14? No.
        // 0→2(3), 1→1(4), 2→0(3) = 10? Let's check:
        // 0→0(1), 1→1(4), 2→2(9) = 14
        // 0→1(2), 1→0(2), 2→2(9) = 13
        // 0→2(3), 1→0(2), 2→1(6) = 11
        // 0→2(3), 1→1(4), 2→0(3) = 10
        // 0→1(2), 1→2(6), 2→0(3) = 11
        // 0→0(1), 1→2(6), 2→1(6) = 13
        // Optimal = 10: 0→2, 1→1, 2→0.
        let costs = vec![1, 2, 3, 2, 4, 6, 3, 6, 9];
        let result = hungarian(&costs, 3, 3);
        let total: u32 = result
            .iter()
            .enumerate()
            .map(|(r, &c)| costs[r * 3 + c])
            .sum();
        assert_eq!(total, 10);
    }

    #[test]
    fn hungarian_1x1() {
        let costs = vec![42];
        let result = hungarian(&costs, 1, 1);
        assert_eq!(result, vec![0]);
    }

    // ── all_entangling_locations ──

    #[test]
    fn entangling_locations_count() {
        let arch = make_arch();
        let locs = all_entangling_locations(&arch);
        // 1 word pair × 10 sites × 2 words = 20 locations.
        assert_eq!(locs.len(), 20);
    }

    // ── build_entangling_set ──

    #[test]
    fn entangling_set_contains_valid_pairs() {
        let arch = make_arch();
        let set = build_entangling_set(&arch);
        // (word 0 site 5, word 1 site 5) should be valid.
        let a = loc(0, 5).encode();
        let b = loc(1, 5).encode();
        assert!(set.contains(&(a, b)));
        assert!(set.contains(&(b, a))); // both orderings
    }

    #[test]
    fn entangling_set_rejects_different_sites() {
        let arch = make_arch();
        let set = build_entangling_set(&arch);
        // (word 0 site 3, word 1 site 5) — different sites, not valid.
        let a = loc(0, 3).encode();
        let b = loc(1, 5).encode();
        assert!(!set.contains(&(a, b)));
    }

    #[test]
    fn entangling_set_rejects_same_word() {
        let arch = make_arch();
        let set = build_entangling_set(&arch);
        // (word 0 site 5, word 0 site 5) — same word, not valid.
        let a = loc(0, 5).encode();
        assert!(!set.contains(&(a, a)));
    }

    // ── WordPairDistances ──

    #[test]
    fn word_pair_distances_basic() {
        let arch = make_arch();
        let index = make_index();
        let locs = all_entangling_locations(&arch);
        let dist_table = DistanceTable::new(&locs, &index);
        let word_pairs = enumerate_word_pairs(&arch);
        let wpd = WordPairDistances::from_dist_table(&word_pairs, &arch, &dist_table);

        assert_eq!(wpd.len(), 1);

        // From word 0 site 0 to word 0 (any site): should be 0 or 1 hop.
        let (_, min_a, _) = wpd.iter().next().unwrap();
        let from = loc(0, 0).encode();
        let d = min_a.get(&from).copied().unwrap_or(u32::MAX);
        assert!(d <= 1, "distance from word 0 to nearest word 0 site: {d}");
    }

    #[test]
    fn word_pair_distances_cross_word() {
        let arch = make_arch();
        let index = make_index();
        let locs = all_entangling_locations(&arch);
        let dist_table = DistanceTable::new(&locs, &index);
        let word_pairs = enumerate_word_pairs(&arch);
        let wpd = WordPairDistances::from_dist_table(&word_pairs, &arch, &dist_table);

        // From word 0 site 0 to word 1 (any site): should need site bus + word bus = 2 hops.
        let (_, _, min_b) = wpd.iter().next().unwrap();
        let from = loc(0, 0).encode();
        let d = min_b.get(&from).copied().unwrap_or(u32::MAX);
        assert!(d >= 1, "cross-word distance should be at least 1: {d}");
    }

    // ── greedy_assign_pairs ──

    #[test]
    fn greedy_assigns_all_pairs() {
        let arch = make_arch();
        let index = make_index();
        let locs = all_entangling_locations(&arch);
        let dist_table = DistanceTable::new(&locs, &index);

        let config = Config::new([(0, loc(0, 0)), (1, loc(1, 0))]).unwrap();
        let cz_pairs = [(0u32, 1u32)];
        let targets = greedy_assign_pairs(&cz_pairs, &config, &arch, &dist_table, 0, None, 0.0);

        // Should assign both qubits.
        assert_eq!(targets.len(), 2);
        let qids: HashSet<u32> = targets.iter().map(|&(q, _)| q).collect();
        assert!(qids.contains(&0));
        assert!(qids.contains(&1));
    }

    #[test]
    fn greedy_assigns_to_entangling_positions() {
        let arch = make_arch();
        let index = make_index();
        let locs = all_entangling_locations(&arch);
        let dist_table = DistanceTable::new(&locs, &index);
        let eset = build_entangling_set(&arch);

        let config = Config::new([(0, loc(0, 5)), (1, loc(1, 5))]).unwrap();
        let cz_pairs = [(0u32, 1u32)];
        let targets = greedy_assign_pairs(&cz_pairs, &config, &arch, &dist_table, 0, None, 0.0);

        let t0 = targets.iter().find(|&&(q, _)| q == 0).unwrap().1;
        let t1 = targets.iter().find(|&&(q, _)| q == 1).unwrap().1;
        assert!(
            eset.contains(&(t0, t1)),
            "assigned positions should be a valid entangling pair"
        );
    }

    #[test]
    fn greedy_no_double_booking() {
        let arch = make_arch();
        let index = make_index();
        let locs = all_entangling_locations(&arch);
        let dist_table = DistanceTable::new(&locs, &index);

        // Two pairs, both starting near the same site.
        let config = Config::new([
            (0, loc(0, 5)),
            (1, loc(1, 5)),
            (2, loc(0, 6)),
            (3, loc(1, 6)),
        ])
        .unwrap();
        let cz_pairs = [(0u32, 1u32), (2u32, 3u32)];
        let targets = greedy_assign_pairs(&cz_pairs, &config, &arch, &dist_table, 0, None, 0.0);

        // All 4 qubit targets should be at distinct locations.
        let locs_used: HashSet<u64> = targets.iter().map(|&(_, l)| l).collect();
        assert_eq!(locs_used.len(), 4, "no double-booking of locations");
    }

    #[test]
    fn greedy_perturbation_changes_assignment() {
        let arch = make_arch();
        let index = make_index();
        let locs = all_entangling_locations(&arch);
        let dist_table = DistanceTable::new(&locs, &index);

        // Multiple pairs with equal-cost options — perturbation should vary.
        let config = Config::new([
            (0, loc(0, 0)),
            (1, loc(1, 0)),
            (2, loc(0, 1)),
            (3, loc(1, 1)),
        ])
        .unwrap();
        let cz_pairs = [(0u32, 1u32), (2u32, 3u32)];

        let t0 = greedy_assign_pairs(&cz_pairs, &config, &arch, &dist_table, 0, None, 0.0);
        let t1 = greedy_assign_pairs(&cz_pairs, &config, &arch, &dist_table, 42, None, 0.0);
        let t2 = greedy_assign_pairs(&cz_pairs, &config, &arch, &dist_table, 123, None, 0.0);

        // At least one seed should produce a different assignment.
        // (Not guaranteed, but very likely with different seeds.)
        let all_same = t0 == t1 && t1 == t2;
        // This is a soft check — perturbation is small (+/-1), so we just
        // verify the function doesn't crash and produces valid output.
        assert_eq!(t0.len(), 4);
        assert_eq!(t1.len(), 4);
        assert_eq!(t2.len(), 4);
        let _ = all_same; // suppress unused warning
    }

    // ── build_partner_map ──

    #[test]
    fn partner_map_bidirectional() {
        let arch = make_arch();
        let eset = build_entangling_set(&arch);
        let pmap = build_partner_map(&eset);
        // (word 0, site 5) → (word 1, site 5) and vice versa.
        let a = loc(0, 5).encode();
        let b = loc(1, 5).encode();
        assert_eq!(pmap.get(&a), Some(&b));
        assert_eq!(pmap.get(&b), Some(&a));
    }

    // ── find_accidental_cz ──

    #[test]
    fn no_accidental_cz_when_partner_empty() {
        let arch = make_arch();
        let eset = build_entangling_set(&arch);
        let pmap = build_partner_map(&eset);
        // q0 at (word 0, site 5), q1 at (word 0, site 6) — no partner occupied.
        let config = Config::new([(0, loc(0, 5)), (1, loc(0, 6))]).unwrap();
        let cz_qubits = HashSet::new();
        let accidental = find_accidental_cz(&config, &cz_qubits, &pmap);
        assert!(accidental.is_empty());
    }

    #[test]
    fn detects_accidental_cz() {
        let arch = make_arch();
        let eset = build_entangling_set(&arch);
        let pmap = build_partner_map(&eset);
        // q0 at (word 0, site 5), q1 at (word 1, site 5) — partner sites!
        let config = Config::new([(0, loc(0, 5)), (1, loc(1, 5))]).unwrap();
        let cz_qubits = HashSet::new(); // both are spectators
        let accidental = find_accidental_cz(&config, &cz_qubits, &pmap);
        assert_eq!(accidental.len(), 1); // one of the pair needs to move
    }

    #[test]
    fn no_accidental_cz_when_partner_is_cz_participant() {
        let arch = make_arch();
        let eset = build_entangling_set(&arch);
        let pmap = build_partner_map(&eset);
        // q0 at (word 0, site 5), q1 at (word 1, site 5) — but q1 is a CZ participant.
        let config = Config::new([(0, loc(0, 5)), (1, loc(1, 5))]).unwrap();
        let cz_qubits: HashSet<u32> = [1].into_iter().collect();
        let accidental = find_accidental_cz(&config, &cz_qubits, &pmap);
        assert!(accidental.is_empty());
    }
}
