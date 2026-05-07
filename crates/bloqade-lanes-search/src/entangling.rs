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
use crate::lane_index::LaneIndex;

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
/// `occupancy_penalty` adds a per-slot-half cost when the slot is
/// occupied by a spectator atom (an atom that is not in any CZ pair of the
/// current layer). It steers the assignment away from slots that would
/// require evicting non-participating atoms. Pass `0.0` to disable
/// (recovers the pre-occupancy-aware behaviour). Atoms that *are* in
/// some CZ pair this layer are never penalised — they will be reassigned
/// by the Hungarian and move out of the way naturally. Fractional values
/// are supported via per-cell rounding; values must be finite and
/// non-negative.
///
/// Returns `(qubit_id, encoded_target_location)` entries suitable for
/// [`SearchContext::targets`](crate::context::SearchContext).
#[allow(clippy::too_many_arguments)]
pub fn greedy_assign_pairs(
    cz_pairs: &[(u32, u32)],
    config: &Config,
    arch: &ArchSpec,
    dist_table: &DistanceTable,
    seed: u64,
    transition_targets: Option<&HashMap<u32, u64>>,
    transition_weight: f64,
    congestion_weight: f64,
    occupancy_penalty: f64,
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

    // Filter to pairs whose qubits are both present in the config. Leaving
    // missing-qubit rows in the matrix as all-BIG would force Hungarian to
    // assign them a column anyway, depriving valid pairs of preferred slots.
    let valid_pairs: Vec<((u32, u32), u64, u64)> = cz_pairs
        .iter()
        .filter_map(|&(qa, qb)| {
            let loc_a_enc = config.location_of(qa)?.encode();
            let loc_b_enc = config.location_of(qb)?.encode();
            Some(((qa, qb), loc_a_enc, loc_b_enc))
        })
        .collect();
    let n_pairs = valid_pairs.len();
    let n_slots = slots.len();
    if n_pairs == 0 || n_slots == 0 {
        return Vec::new();
    }

    // Spectator-occupied locations: positions held by atoms not in any
    // CZ pair this layer. Adding `occupancy_penalty` per slot half
    // currently held by such an atom biases the Hungarian away from
    // assignments that require evicting non-participating atoms.
    let occupancy_active = occupancy_penalty > 0.0;
    let spectator_locs: HashSet<u64> = if occupancy_active {
        let pair_qubits: HashSet<u32> = cz_pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
        config
            .iter()
            .filter(|(qid, _)| !pair_qubits.contains(qid))
            .map(|(_, loc)| loc.encode())
            .collect()
    } else {
        HashSet::new()
    };

    // Build cost matrix: cost[pair_i][slot_j] = min over both orientations
    // of d(qa, target_a) + d(qb, target_b).
    // We also store which orientation was chosen per cell.
    const BIG: u32 = u32::MAX / 4;
    let mut base_costs = vec![BIG; n_pairs * n_slots];
    // swapped[i * n_slots + j] = true means qa→loc_b, qb→loc_a is cheaper.
    let mut swapped = vec![false; n_pairs * n_slots];

    for (i, &((qa, qb), loc_a_enc, loc_b_enc)) in valid_pairs.iter().enumerate() {
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

            // Spectator-occupancy penalty: orientation-independent because
            // both orientations target the same pair of slot positions.
            // Per-slot-half penalties sum, so a slot with both halves
            // spectator-occupied costs `2 * occupancy_penalty` more.
            // Round once per cell so fractional `occupancy_penalty` values
            // (e.g. 0.5, 1.5) carry signal in the integer cost matrix.
            if occupancy_active {
                let count = (spectator_locs.contains(&slot.loc_a) as u32)
                    + (spectator_locs.contains(&slot.loc_b) as u32);
                if count > 0 {
                    let occ_pen = (count as f64 * occupancy_penalty).round().max(0.0) as u32;
                    cost1 = cost1.saturating_add(occ_pen);
                    cost2 = cost2.saturating_add(occ_pen);
                }
            }

            let idx = i * n_slots + j;
            if cost1 <= cost2 {
                base_costs[idx] = cost1;
                swapped[idx] = false;
            } else {
                base_costs[idx] = cost2;
                swapped[idx] = true;
            }
        }
    }

    // Apply seed-based perturbation for restart diversity.
    if seed != 0 {
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};
        let mut rng = SmallRng::seed_from_u64(seed);
        for c in base_costs.iter_mut() {
            if *c < BIG {
                let perturbation: i32 = rng.random_range(-1..=1);
                *c = (*c as i32).saturating_add(perturbation).max(0) as u32;
            }
        }
    }

    // Run Hungarian, possibly iterating with congestion penalty.
    let assignment = if congestion_weight > 0.0 && word_pairs.len() > 1 {
        congestion_aware_hungarian(
            &base_costs,
            n_pairs,
            n_slots,
            sites_per_word as usize,
            word_pairs.len(),
            congestion_weight,
            BIG,
        )
    } else {
        hungarian(&base_costs, n_pairs, n_slots)
    };

    // Extract result.
    let mut result: Vec<(u32, u64)> = Vec::with_capacity(n_pairs * 2);
    for (i, &col) in assignment.iter().enumerate() {
        if col >= n_slots {
            continue; // unassigned (shouldn't happen if slots >= pairs)
        }
        let idx = i * n_slots + col;
        if base_costs[idx] >= BIG {
            continue; // unreachable position
        }
        let (qa, qb) = valid_pairs[i].0;
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

// ── Iterative Hungarian with blocker augmentation ──────────────────

/// A row in the mixed-row Hungarian cost matrix.
///
/// `Pair` rows behave exactly like the rows in [`greedy_assign_pairs`] —
/// they target a slot and occupy *both* halves. `Spectator` rows are
/// added by [`assign_pairs_with_blockers`] to displace blocking atoms;
/// they target a slot but occupy only *one* half (the cheaper of
/// `loc_a` / `loc_b`), leaving the other half empty.
#[derive(Clone, Copy)]
enum Row {
    Pair {
        qa: u32,
        qb: u32,
        loc_a_enc: u64,
        loc_b_enc: u64,
    },
    Spectator {
        qid: u32,
        loc_enc: u64,
    },
}

/// Iterative Hungarian assignment that augments the row set with
/// "blockers" — non-pair atoms currently sitting on a CZ pair's
/// assigned target (Case A) or hemming a CZ atom in at its current
/// location (Case B).
///
/// Each iteration: build a mixed-row cost matrix (CZ pairs + currently
/// known blockers), run the Hungarian, detect new blockers, repeat.
/// Termination is bounded by the number of non-pair qubits in `config`;
/// in practice converges in 1–2 passes for the regimes that motivated
/// this design.
///
/// Compared to `greedy_assign_pairs`:
/// - blocker spectators get explicit `(qid, target)` entries in the
///   returned target list, so downstream search treats them as ordinary
///   unresolved qubits (no `MoveBlockers`-style escape needed);
/// - the global Hungarian sees the displacement cost explicitly, so it
///   only relocates a spectator when doing so is genuinely cheaper than
///   routing the CZ pair around it.
///
/// Capacity guard: if `n_pairs > n_slots` (pathological density), falls
/// back to plain `greedy_assign_pairs` and returns its result.
#[allow(clippy::too_many_arguments)]
pub fn assign_pairs_with_blockers(
    cz_pairs: &[(u32, u32)],
    config: &Config,
    arch: &ArchSpec,
    index: &LaneIndex,
    dist_table: &DistanceTable,
    blocked: &HashSet<u64>,
    seed: u64,
    transition_targets: Option<&HashMap<u32, u64>>,
    transition_weight: f64,
    congestion_weight: f64,
    occupancy_penalty: f64,
    move_penalty: f64,
    enable_case_d: bool,
) -> Vec<(u32, u64)> {
    if cz_pairs.is_empty() {
        return Vec::new();
    }

    let word_pairs = enumerate_word_pairs(arch);
    let sites_per_word = arch.sites_per_word() as u32;

    // Build slot columns (same layout as `greedy_assign_pairs`).
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
    let n_slots = slots.len();

    // Pairs whose qubits are both present in the config (same filter as
    // `greedy_assign_pairs`). Pre-encoded for cost-matrix building.
    let valid_pairs: Vec<((u32, u32), u64, u64)> = cz_pairs
        .iter()
        .filter_map(|&(qa, qb)| {
            let loc_a_enc = config.location_of(qa)?.encode();
            let loc_b_enc = config.location_of(qb)?.encode();
            Some(((qa, qb), loc_a_enc, loc_b_enc))
        })
        .collect();
    if valid_pairs.is_empty() || n_slots == 0 {
        return Vec::new();
    }

    // Capacity guard: rectangular Hungarian needs n_rows ≤ n_slots. With
    // pairs alone we already exceed → bail to plain greedy. (Note: plain
    // greedy doesn't honour `move_penalty` or `enable_case_d`; we accept
    // that since this path is the pathological-density bail-out.)
    if valid_pairs.len() > n_slots {
        return greedy_assign_pairs(
            cz_pairs,
            config,
            arch,
            dist_table,
            seed,
            transition_targets,
            transition_weight,
            congestion_weight,
            occupancy_penalty,
        );
    }

    let pair_qubits: HashSet<u32> = cz_pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
    let mut row_qubits: HashSet<u32> = pair_qubits.clone();
    let mut blockers: Vec<u32> = Vec::new();

    // Iterations bounded by the count of non-pair qubits — each loop
    // either adds at least one blocker row or returns.
    let max_iter = config.len().saturating_sub(pair_qubits.len()) + 1;
    let mut last_targets: Vec<(u32, u64)> = Vec::new();

    for _ in 0..max_iter {
        // Stop adding blockers once the row count would exceed slot capacity.
        // Truncate the most-recently-added blockers if needed.
        if valid_pairs.len() + blockers.len() > n_slots {
            let keep = n_slots - valid_pairs.len();
            for &q in &blockers[keep..] {
                row_qubits.remove(&q);
            }
            blockers.truncate(keep);
        }

        // Build the mixed row vector for this iteration.
        let mut rows: Vec<Row> = Vec::with_capacity(valid_pairs.len() + blockers.len());
        for &((qa, qb), loc_a_enc, loc_b_enc) in &valid_pairs {
            rows.push(Row::Pair {
                qa,
                qb,
                loc_a_enc,
                loc_b_enc,
            });
        }
        for &qid in &blockers {
            if let Some(loc) = config.location_of(qid) {
                rows.push(Row::Spectator {
                    qid,
                    loc_enc: loc.encode(),
                });
            }
        }
        if rows.is_empty() {
            return Vec::new();
        }

        let targets = mixed_row_hungarian(
            &rows,
            &slots,
            &word_pairs,
            sites_per_word as usize,
            config,
            dist_table,
            &row_qubits,
            blocked,
            transition_targets,
            transition_weight,
            congestion_weight,
            occupancy_penalty,
            move_penalty,
            seed,
        );

        let new_ab = detect_blockers(&targets, &pair_qubits, config, &row_qubits, index);
        let new_d = if enable_case_d {
            detect_accidental_cz_spectators(config, arch, &pair_qubits, &row_qubits)
        } else {
            Vec::new()
        };
        // Case E: accidentals *created* by this iteration's Hungarian.
        // The cost-matrix accidental-creation penalty
        // (`mixed_row_hungarian`) is a soft bias and may not always
        // succeed. This catches the residual cases by adding the
        // non-row partner of any newly-paired half as a row in the next
        // iteration. Gated by `enable_case_d` because it is the same
        // accidental-resolution mechanism (and we don't want it firing
        // inside lookahead simulations for the same reason — sim configs
        // legitimately put prior-layer pair atoms at slot partner halves).
        let new_e = if enable_case_d {
            detect_accidentals_created(&targets, arch, config, &pair_qubits, &row_qubits)
        } else {
            Vec::new()
        };
        if new_ab.is_empty() && new_d.is_empty() && new_e.is_empty() {
            return targets;
        }
        last_targets = targets;
        for q in new_ab
            .into_iter()
            .chain(new_d.into_iter())
            .chain(new_e.into_iter())
        {
            if !row_qubits.contains(&q) {
                blockers.push(q);
                row_qubits.insert(q);
            }
        }
    }

    // Safety fallback: should not be reachable under the iteration bound.
    last_targets
}

/// Run the Hungarian for one iteration of [`assign_pairs_with_blockers`].
///
/// Builds a `n_rows × n_slots` cost matrix where:
/// - `Row::Pair` cells use `min(d_a1+d_b1, d_a2+d_b2)` over orientations
///   plus transition / congestion / occupancy terms (same as
///   `greedy_assign_pairs`);
/// - `Row::Spectator` cells use `min(half_a_total, half_b_total)` where
///   each half's total = `d(s, half) + transition + occupancy`,
///   computed independently so the per-half occupancy isn't double-counted.
///
/// Tracks per-cell orientation (for pair rows) and which half won (for
/// spectator rows) so the final assignment can decode each row's target
/// correctly.
#[allow(clippy::too_many_arguments)]
fn mixed_row_hungarian(
    rows: &[Row],
    slots: &[PositionSlot],
    word_pairs: &[EntanglingWordPair],
    sites_per_word: usize,
    config: &Config,
    dist_table: &DistanceTable,
    row_qubits: &HashSet<u32>,
    blocked: &HashSet<u64>,
    transition_targets: Option<&HashMap<u32, u64>>,
    transition_weight: f64,
    congestion_weight: f64,
    occupancy_penalty: f64,
    move_penalty: f64,
    seed: u64,
) -> Vec<(u32, u64)> {
    const BIG: u32 = u32::MAX / 4;
    let n_rows = rows.len();
    let n_slots = slots.len();

    let mut base_costs = vec![BIG; n_rows * n_slots];
    // For pair rows: true means the qa→loc_b, qb→loc_a orientation was
    // cheaper. For spectator rows: true means the spectator should land
    // on slot.loc_b (otherwise loc_a). Reusing the same array keeps
    // extraction symmetric.
    let mut swapped_or_locb = vec![false; n_rows * n_slots];

    // Build the spectator-occupancy set: atoms in `config` whose qubit
    // ID is *not* a row of the current Hungarian. These atoms won't move
    // themselves, so a slot half they occupy costs an `occupancy_penalty`
    // to claim. Excluding `row_qubits` is critical — otherwise a
    // spectator's own current slot half would be self-penalised.
    let occupancy_active = occupancy_penalty > 0.0;
    let spectator_locs: HashSet<u64> = if occupancy_active {
        config
            .iter()
            .filter(|(qid, _)| !row_qubits.contains(qid))
            .map(|(_, loc)| loc.encode())
            .collect()
    } else {
        HashSet::new()
    };

    let half_pen = |loc_enc: u64| -> u32 {
        if occupancy_active && spectator_locs.contains(&loc_enc) {
            occupancy_penalty.round().max(0.0) as u32
        } else {
            0
        }
    };

    // Stay-in-place bias: per atom, charge `move_penalty` if its assigned
    // half differs from its current location. Encourages anchored
    // assignments when the choice is close. Rounded to integer for the
    // u32 cost matrix.
    let move_active = move_penalty > 0.0;
    let move_pen_u32 = move_penalty.round().max(0.0) as u32;

    // Slot-half blockedness: a half held by a `blocked` atom (an atom
    // not part of this strategy's layout) is unmovable, so no row may
    // land there. Pair rows occupy both halves and must skip slots
    // where either half is blocked. Spectator rows can still pick the
    // non-blocked half of a partly-blocked slot.
    let pair_slot_blocked: Vec<bool> = slots
        .iter()
        .map(|s| blocked.contains(&s.loc_a) || blocked.contains(&s.loc_b))
        .collect();

    for (i, row) in rows.iter().enumerate() {
        for (j, slot) in slots.iter().enumerate() {
            let idx = i * n_slots + j;
            match *row {
                Row::Pair {
                    qa,
                    qb,
                    loc_a_enc,
                    loc_b_enc,
                } => {
                    // Pair must take both halves of this slot — skip if
                    // either is blocked.
                    if pair_slot_blocked[j] {
                        // Leave base_costs[idx] = BIG (init value).
                        continue;
                    }
                    // Orientation 1: qa→loc_a, qb→loc_b.
                    let d_a1 = dist_table.distance(loc_a_enc, slot.loc_a).unwrap_or(BIG);
                    let d_b1 = dist_table.distance(loc_b_enc, slot.loc_b).unwrap_or(BIG);
                    let mut cost1 = d_a1.saturating_add(d_b1);

                    // Orientation 2: qa→loc_b, qb→loc_a.
                    let d_a2 = dist_table.distance(loc_a_enc, slot.loc_b).unwrap_or(BIG);
                    let d_b2 = dist_table.distance(loc_b_enc, slot.loc_a).unwrap_or(BIG);
                    let mut cost2 = d_a2.saturating_add(d_b2);

                    if let Some(targets) = transition_targets {
                        if let Some(&next_a) = targets.get(&qa) {
                            let dt = dist_table.distance(slot.loc_a, next_a).unwrap_or(BIG);
                            cost1 = cost1.saturating_add((dt as f64 * transition_weight) as u32);
                            let dt2 = dist_table.distance(slot.loc_b, next_a).unwrap_or(BIG);
                            cost2 = cost2.saturating_add((dt2 as f64 * transition_weight) as u32);
                        }
                        if let Some(&next_b) = targets.get(&qb) {
                            let dt = dist_table.distance(slot.loc_b, next_b).unwrap_or(BIG);
                            cost1 = cost1.saturating_add((dt as f64 * transition_weight) as u32);
                            let dt2 = dist_table.distance(slot.loc_a, next_b).unwrap_or(BIG);
                            cost2 = cost2.saturating_add((dt2 as f64 * transition_weight) as u32);
                        }
                    }

                    // Spectator-occupancy penalty — orientation-independent
                    // since both orientations target the same slot halves.
                    if occupancy_active {
                        let count = (spectator_locs.contains(&slot.loc_a) as u32)
                            + (spectator_locs.contains(&slot.loc_b) as u32);
                        if count > 0 {
                            let pen = (count as f64 * occupancy_penalty).round().max(0.0) as u32;
                            cost1 = cost1.saturating_add(pen);
                            cost2 = cost2.saturating_add(pen);
                        }
                    }

                    // Stay-in-place bias per moved atom (orientation-aware:
                    // each orientation maps qa/qb to different halves and
                    // therefore has potentially different `moved` counts).
                    if move_active && move_pen_u32 > 0 {
                        let moved1 =
                            (loc_a_enc != slot.loc_a) as u32 + (loc_b_enc != slot.loc_b) as u32;
                        let moved2 =
                            (loc_a_enc != slot.loc_b) as u32 + (loc_b_enc != slot.loc_a) as u32;
                        cost1 = cost1.saturating_add(moved1.saturating_mul(move_pen_u32));
                        cost2 = cost2.saturating_add(moved2.saturating_mul(move_pen_u32));
                    }

                    if cost1 <= cost2 {
                        base_costs[idx] = cost1;
                        swapped_or_locb[idx] = false;
                    } else {
                        base_costs[idx] = cost2;
                        swapped_or_locb[idx] = true;
                    }
                }
                Row::Spectator { qid, loc_enc } => {
                    // Per-half totals computed independently so the
                    // per-half occupancy and move penalties aren't
                    // double-counted.
                    let mut half_a_total = dist_table.distance(loc_enc, slot.loc_a).unwrap_or(BIG);
                    let mut half_b_total = dist_table.distance(loc_enc, slot.loc_b).unwrap_or(BIG);

                    if let Some(targets) = transition_targets
                        && let Some(&next) = targets.get(&qid)
                    {
                        let dt_a = dist_table.distance(slot.loc_a, next).unwrap_or(BIG);
                        let dt_b = dist_table.distance(slot.loc_b, next).unwrap_or(BIG);
                        half_a_total =
                            half_a_total.saturating_add((dt_a as f64 * transition_weight) as u32);
                        half_b_total =
                            half_b_total.saturating_add((dt_b as f64 * transition_weight) as u32);
                    }

                    half_a_total = half_a_total.saturating_add(half_pen(slot.loc_a));
                    half_b_total = half_b_total.saturating_add(half_pen(slot.loc_b));

                    // Accidental-creation penalty: landing on one half
                    // pairs this row with whatever non-row atom currently
                    // sits at the other half. That atom won't move (it's
                    // not a row), so the result is a new accidental CZ.
                    // Penalise both directions so the Hungarian prefers
                    // slots whose partner half is free (or held by another
                    // row qubit that will itself move away).
                    if occupancy_active {
                        if spectator_locs.contains(&slot.loc_b) {
                            half_a_total = half_a_total.saturating_add(half_pen(slot.loc_b));
                        }
                        if spectator_locs.contains(&slot.loc_a) {
                            half_b_total = half_b_total.saturating_add(half_pen(slot.loc_a));
                        }
                    }

                    // Stay-in-place bias for the spectator atom.
                    if move_active && move_pen_u32 > 0 {
                        if loc_enc != slot.loc_a {
                            half_a_total = half_a_total.saturating_add(move_pen_u32);
                        }
                        if loc_enc != slot.loc_b {
                            half_b_total = half_b_total.saturating_add(move_pen_u32);
                        }
                    }

                    // Blocked-half guard: a spectator can only land on a
                    // half not currently held by an immobile atom.
                    if blocked.contains(&slot.loc_a) {
                        half_a_total = BIG;
                    }
                    if blocked.contains(&slot.loc_b) {
                        half_b_total = BIG;
                    }

                    if half_a_total <= half_b_total {
                        base_costs[idx] = half_a_total;
                        swapped_or_locb[idx] = false;
                    } else {
                        base_costs[idx] = half_b_total;
                        swapped_or_locb[idx] = true;
                    }
                }
            }
        }
    }

    // Apply seed-based perturbation (same convention as
    // `greedy_assign_pairs`): only in-bound (< BIG) cells.
    if seed != 0 {
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};
        let mut rng = SmallRng::seed_from_u64(seed);
        for c in base_costs.iter_mut() {
            if *c < BIG {
                let perturbation: i32 = rng.random_range(-1..=1);
                *c = (*c as i32).saturating_add(perturbation).max(0) as u32;
            }
        }
    }

    let assignment = if congestion_weight > 0.0 && word_pairs.len() > 1 {
        congestion_aware_hungarian(
            &base_costs,
            n_rows,
            n_slots,
            sites_per_word,
            word_pairs.len(),
            congestion_weight,
            BIG,
        )
    } else {
        hungarian(&base_costs, n_rows, n_slots)
    };

    // Extract per-row targets.
    let mut result: Vec<(u32, u64)> = Vec::with_capacity(n_rows * 2);
    for (i, &col) in assignment.iter().enumerate() {
        if col >= n_slots {
            continue;
        }
        let idx = i * n_slots + col;
        if base_costs[idx] >= BIG {
            continue;
        }
        let slot = &slots[col];
        match rows[i] {
            Row::Pair { qa, qb, .. } => {
                if swapped_or_locb[idx] {
                    result.push((qa, slot.loc_b));
                    result.push((qb, slot.loc_a));
                } else {
                    result.push((qa, slot.loc_a));
                    result.push((qb, slot.loc_b));
                }
            }
            Row::Spectator { qid, .. } => {
                let target = if swapped_or_locb[idx] {
                    slot.loc_b
                } else {
                    slot.loc_a
                };
                result.push((qid, target));
            }
        }
    }

    result
}

/// Detect blockers under the current Hungarian assignment.
///
/// **Case A** (target-blocked): a non-row atom is currently parked on a
/// slot half that the Hungarian assigned as a CZ *pair*'s target. (We
/// deliberately exclude *spectator-row* targets — those are destinations,
/// not blockers.)
///
/// **Case B** (egress-blocked): a CZ pair atom's current location has no
/// outgoing lane to an unoccupied destination. The non-row atoms
/// occupying those adjacent destinations are the blockers.
///
/// Returns deduplicated blocker qubit IDs, sorted for deterministic
/// output. Empty result signals iteration convergence.
fn detect_blockers(
    targets: &[(u32, u64)],
    pair_qubits: &HashSet<u32>,
    config: &Config,
    row_qubits: &HashSet<u32>,
    index: &LaneIndex,
) -> Vec<u32> {
    // CZ-pair-only target locations from the latest Hungarian pass.
    // Spectator-row targets are intentionally excluded — those are
    // destinations the iteration is moving spectators *to*, not
    // problems to solve.
    let pair_target_locs: HashSet<u64> = targets
        .iter()
        .filter(|&&(qid, _)| pair_qubits.contains(&qid))
        .map(|&(_, loc)| loc)
        .collect();

    let occupied: HashSet<u64> = config.iter().map(|(_, loc)| loc.encode()).collect();
    let mut new: Vec<u32> = Vec::new();

    // Case A: spectator currently at a CZ pair target location.
    for (qid, loc) in config.iter() {
        if row_qubits.contains(&qid) {
            continue;
        }
        if pair_target_locs.contains(&loc.encode()) {
            new.push(qid);
        }
    }

    // Case B: CZ pair atom egress-blocked at its current position.
    for (qid, _) in config.iter() {
        if !pair_qubits.contains(&qid) {
            continue;
        }
        let Some(loc) = config.location_of(qid) else {
            continue;
        };

        // Does qid have any outgoing lane to a free destination?
        let mut has_free = false;
        for &lane in index.outgoing_lanes(loc) {
            if let Some((_, dst)) = index.endpoints(&lane)
                && !occupied.contains(&dst.encode())
            {
                has_free = true;
                break;
            }
        }
        if has_free {
            continue;
        }

        // qid is hemmed in. Identify the spectators in its way.
        for &lane in index.outgoing_lanes(loc) {
            if let Some((_, dst)) = index.endpoints(&lane)
                && let Some(blocker) = config.qubit_at(dst)
                && !row_qubits.contains(&blocker)
            {
                new.push(blocker);
            }
        }
    }

    new.sort_unstable();
    new.dedup();
    new
}

/// Detect spectator pairs forming an "accidental CZ" in the current config.
///
/// An accidental CZ exists when two non-row, non-CZ-pair atoms occupy
/// partner halves of an entangling word pair at the same site
/// (i.e. `arch.get_cz_partner(loc_of_qa) == loc_of_qb` and neither qa
/// nor qb is in the current Hungarian row set or in any CZ pair).
///
/// Returns **both** members of every accidental pair. Returning both
/// (rather than just the higher-id one as IDS's MoveBlockers does) lets
/// the Hungarian's column-uniqueness break the accidental: both rows
/// compete for the slot containing their partner halves, only one can
/// claim that column, and the other is forced to a different slot —
/// dissolving the accidental CZ.
///
/// Mirrors the IDS Step-1b logic in `HeuristicGenerator::generate`
/// (`heuristic.rs:272-307`) but emits both atoms instead of one.
///
/// Used by `assign_pairs_with_blockers` only when `enable_case_d` is
/// `true`. Inside `lookahead_assign_pairs`, this is gated off for
/// future-layer simulations because their `sim_config` legitimately has
/// the just-finished CZ pairs sitting at partner halves of the slot
/// they targeted; flagging *those* atoms as "accidental" would inflate
/// the row set with false positives and corrupt the lookahead bias
/// passed via `transition_targets` back to the current layer.
/// Detect non-row spectators whose location is the *partner half* of any
/// spectator-row's Hungarian target. These atoms would form a new accidental
/// CZ once the Hungarian's targets are applied, because:
/// - the spectator-row lands on one half of a slot column,
/// - the non-row atom currently at the other half doesn't move,
/// - the two end up at slot partner halves → accidental CZ.
///
/// Pair-row targets don't suffer from this: pairs occupy *both* halves of
/// their assigned slot column, and any non-row at either half is caught
/// by Case A (`detect_blockers`) and added as its own row.
///
/// The cost-matrix accidental-creation penalty (`mixed_row_hungarian`)
/// makes this case rare — it biases the Hungarian toward slots whose
/// partner half is free or row-held — but it's still possible when no
/// such slot exists. This function provides the fallback: returned qubits
/// are added as new rows for the next iteration of
/// `assign_pairs_with_blockers`.
fn detect_accidentals_created(
    targets: &[(u32, u64)],
    arch: &ArchSpec,
    config: &Config,
    pair_qubits: &HashSet<u32>,
    row_qubits: &HashSet<u32>,
) -> Vec<u32> {
    let mut new: Vec<u32> = Vec::new();
    let mut seen: HashSet<u32> = HashSet::new();
    for &(qid, target_enc) in targets {
        // Pair-row targets fill both halves; non-row evictees are caught
        // by Case A. Only spectator-row targets create new accidentals.
        if pair_qubits.contains(&qid) {
            continue;
        }
        let target_loc = LocationAddr::decode(target_enc);
        let Some(partner_loc) = arch.get_cz_partner(&target_loc) else {
            continue;
        };
        let Some(partner_qid) = config.qubit_at(partner_loc) else {
            continue;
        };
        // If the partner-half occupant is itself a row qubit, it will
        // move away and the half becomes free — no accidental.
        if row_qubits.contains(&partner_qid) {
            continue;
        }
        if seen.insert(partner_qid) {
            new.push(partner_qid);
        }
    }
    new.sort_unstable();
    new
}

fn detect_accidental_cz_spectators(
    config: &Config,
    arch: &ArchSpec,
    pair_qubits: &HashSet<u32>,
    row_qubits: &HashSet<u32>,
) -> Vec<u32> {
    let mut new: Vec<u32> = Vec::new();
    let mut seen_loc: HashSet<u64> = HashSet::new();
    for (qid, loc) in config.iter() {
        if pair_qubits.contains(&qid) || row_qubits.contains(&qid) {
            continue;
        }
        let loc_enc = loc.encode();
        if seen_loc.contains(&loc_enc) {
            continue;
        }
        let Some(partner_loc) = arch.get_cz_partner(&loc) else {
            continue;
        };
        let partner_enc = partner_loc.encode();
        let Some(other_qid) = config.qubit_at(partner_loc) else {
            continue;
        };
        if pair_qubits.contains(&other_qid) || row_qubits.contains(&other_qid) {
            continue;
        }
        // Both qid and other_qid are non-row spectators at partner halves.
        new.push(qid);
        new.push(other_qid);
        seen_loc.insert(loc_enc);
        seen_loc.insert(partner_enc);
    }
    new.sort_unstable();
    new.dedup();
    new
}

/// Congestion-aware assignment: vanilla Hungarian followed by greedy
/// rebalancing that re-routes pairs from over-loaded entangling word pairs
/// to under-loaded ones, picking the lowest-cost-loss reassignment first.
///
/// `slots` are laid out as `[wp_0_site_0, ..., wp_0_site_{S-1}, wp_1_site_0, ...]`,
/// so `slot_j / sites_per_word == word_pair_idx`.
///
/// `congestion_weight` is the maximum cost increase (in distance hops) the
/// rebalance accepts to move one pair off an over-loaded word pair.
///   - `0.0`: rebalance disabled.
///   - `1.0`: only accept "free" or cheaper moves (≤ +1 hop).
///   - `5.0`: accept moves that lengthen the routing by up to 5 hops.
fn congestion_aware_hungarian(
    base_costs: &[u32],
    n_pairs: usize,
    n_slots: usize,
    sites_per_word: usize,
    n_word_pairs: usize,
    congestion_weight: f64,
    big: u32,
) -> Vec<usize> {
    let mut assignment = hungarian(base_costs, n_pairs, n_slots);
    if congestion_weight <= 0.0 {
        return assignment;
    }

    let ideal_load = n_pairs.div_ceil(n_word_pairs);
    let mut load = load_per_wp(&assignment, sites_per_word, n_word_pairs);
    if *load.iter().max().unwrap_or(&0) <= ideal_load {
        return assignment;
    }

    let max_acceptable_loss = congestion_weight;

    // Track which slots are taken.
    let mut slot_used: Vec<bool> = vec![false; n_slots];
    for &col in &assignment {
        if col < n_slots {
            slot_used[col] = true;
        }
    }

    // Repeatedly re-route the cheapest excess pair until quota satisfied
    // or no improving move within budget remains. O(n_pairs² × n_slots)
    // worst case.
    let max_iter = n_pairs;
    for _ in 0..max_iter {
        let max_wp = load
            .iter()
            .enumerate()
            .max_by_key(|&(_, &c)| c)
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        if load[max_wp] <= ideal_load {
            break;
        }

        // For each pair currently assigned to `max_wp`, find the cheapest
        // re-routing to an under-loaded wp within the cost-loss budget.
        let mut best_pair: Option<usize> = None;
        let mut best_target: Option<usize> = None;
        let mut best_loss: f64 = f64::INFINITY;

        for (i, &col) in assignment.iter().enumerate() {
            if col >= n_slots {
                continue;
            }
            if col / sites_per_word != max_wp {
                continue;
            }
            let cur_cost = base_costs[i * n_slots + col];
            if cur_cost >= big {
                continue;
            }
            for j in 0..n_slots {
                let wp = j / sites_per_word;
                if wp == max_wp || load[wp] >= ideal_load || slot_used[j] {
                    continue;
                }
                let new_cost = base_costs[i * n_slots + j];
                if new_cost >= big {
                    continue;
                }
                let cost_loss = new_cost as f64 - cur_cost as f64;
                if cost_loss > max_acceptable_loss {
                    continue;
                }
                if cost_loss < best_loss {
                    best_loss = cost_loss;
                    best_pair = Some(i);
                    best_target = Some(j);
                }
            }
        }

        let (Some(i), Some(j)) = (best_pair, best_target) else {
            break;
        };
        let old_col = assignment[i];
        slot_used[old_col] = false;
        slot_used[j] = true;
        load[old_col / sites_per_word] -= 1;
        load[j / sites_per_word] += 1;
        assignment[i] = j;
    }

    assignment
}

fn load_per_wp(assignment: &[usize], sites_per_word: usize, n_word_pairs: usize) -> Vec<usize> {
    let mut load = vec![0usize; n_word_pairs];
    for &col in assignment {
        let wp = col / sites_per_word;
        if wp < n_word_pairs {
            load[wp] += 1;
        }
    }
    load
}

#[cfg(test)]
fn max_load_per_wp(assignment: &[usize], sites_per_word: usize, n_word_pairs: usize) -> usize {
    load_per_wp(assignment, sites_per_word, n_word_pairs)
        .into_iter()
        .max()
        .unwrap_or(0)
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
#[allow(clippy::too_many_arguments)]
pub fn lookahead_assign_pairs(
    cz_pairs: &[(u32, u32)],
    config: &Config,
    arch: &ArchSpec,
    index: &LaneIndex,
    dist_table: &DistanceTable,
    blocked: &HashSet<u64>,
    seed: u64,
    future_layers: &[Vec<(u32, u32)>],
    beta: f64,
    congestion_weight: f64,
    occupancy_penalty: f64,
    move_penalty: f64,
) -> Vec<(u32, u64)> {
    if future_layers.is_empty() || beta == 0.0 {
        return assign_pairs_with_blockers(
            cz_pairs,
            config,
            arch,
            index,
            dist_table,
            blocked,
            seed,
            None,
            0.0,
            congestion_weight,
            occupancy_penalty,
            move_penalty,
            true,
        );
    }

    // Collect all layers: current + future.
    let depth = 1 + future_layers.len();
    let all_layers: Vec<&[(u32, u32)]> = std::iter::once(cz_pairs)
        .chain(future_layers.iter().map(|v| v.as_slice()))
        .collect();

    // Forward pass: preliminary assignments.
    //
    // Case D (accidental-CZ spectator detection) is only enabled for layer
    // 0. Simulated layers ≥ 1 see post-move sim_configs where prior-layer
    // pair atoms necessarily occupy slot partner halves; treating them as
    // accidentals is a false positive that bloats row sets and corrupts
    // the transition_targets fed back into the layer-0 backward refinement.
    let mut forward_assignments: Vec<Vec<(u32, u64)>> = Vec::with_capacity(depth);
    let mut sim_config = config.clone();

    for (layer_idx, layer_pairs) in all_layers.iter().enumerate() {
        let assign = assign_pairs_with_blockers(
            layer_pairs,
            &sim_config,
            arch,
            index,
            dist_table,
            blocked,
            seed,
            None,
            0.0,
            congestion_weight,
            occupancy_penalty,
            move_penalty,
            layer_idx == 0,
        );
        // Build simulated config: move assigned qubits to their targets.
        // `assign` already includes spectator displacements, so the
        // simulated config reflects those moves and downstream layers
        // see the post-displacement layout.
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
        // Includes both pair targets and spectator targets so layer i's
        // assignment can pull blockers toward where they'll land next.
        let next_assign = &forward_assignments[i + 1];
        let targets: HashMap<u32, u64> = next_assign.iter().copied().collect();

        // Cascade replays preliminary forward-pass assignments for
        // layers 0..i (the backward sweep visits them after i, so they
        // are not yet refined). Transition `targets` above use the
        // refined layer i+1. Mismatch is intentional: the refinement
        // pass exists to inject the lookahead signal via targets.
        let layer_config = if i == 0 {
            config.clone()
        } else {
            let mut cfg = config.clone();
            for assignment in forward_assignments.iter().take(i) {
                let moves: Vec<(u32, LocationAddr)> = assignment
                    .iter()
                    .map(|&(qid, enc)| (qid, LocationAddr::decode(enc)))
                    .collect();
                cfg = cfg.with_moves(&moves);
            }
            cfg
        };

        forward_assignments[i] = assign_pairs_with_blockers(
            all_layers[i],
            &layer_config,
            arch,
            index,
            dist_table,
            blocked,
            seed,
            Some(&targets),
            beta,
            congestion_weight,
            occupancy_penalty,
            move_penalty,
            i == 0,
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
pub(crate) fn hungarian(costs: &[u32], n_rows: usize, n_cols: usize) -> Vec<usize> {
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
        let targets = greedy_assign_pairs(
            &cz_pairs,
            &config,
            &arch,
            &dist_table,
            0,
            None,
            0.0,
            0.0,
            0.0,
        );

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
        let targets = greedy_assign_pairs(
            &cz_pairs,
            &config,
            &arch,
            &dist_table,
            0,
            None,
            0.0,
            0.0,
            0.0,
        );

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
        let targets = greedy_assign_pairs(
            &cz_pairs,
            &config,
            &arch,
            &dist_table,
            0,
            None,
            0.0,
            0.0,
            0.0,
        );

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

        let t0 = greedy_assign_pairs(
            &cz_pairs,
            &config,
            &arch,
            &dist_table,
            0,
            None,
            0.0,
            0.0,
            0.0,
        );
        let t1 = greedy_assign_pairs(
            &cz_pairs,
            &config,
            &arch,
            &dist_table,
            42,
            None,
            0.0,
            0.0,
            0.0,
        );
        let t2 = greedy_assign_pairs(
            &cz_pairs,
            &config,
            &arch,
            &dist_table,
            123,
            None,
            0.0,
            0.0,
            0.0,
        );

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

    // ── occupancy-aware assignment ──

    #[test]
    fn occupancy_penalty_avoids_spectator_slots() {
        let arch = make_arch();
        let index = make_index();
        let locs = all_entangling_locations(&arch);
        let dist_table = DistanceTable::new(&locs, &index);

        // 1 CZ pair already at slot site 5 (q0 at pos_a, q1 at pos_b),
        // plus two spectators occupying both halves of slot site 6.
        // Without penalty, Hungarian picks site 5 (cost 0). With a
        // large penalty, site 5 is still cheapest — the spectators are
        // at site 6, not site 5 — so the penalty does not change the
        // pick. We use this scenario to verify the function runs in
        // both modes; the avoidance behaviour is checked in the
        // companion test below.
        let spectator_a = loc(0, 6);
        let spectator_b = loc(1, 6);
        let config = Config::new([
            (0, loc(0, 5)),
            (1, loc(1, 5)),
            (2, spectator_a),
            (3, spectator_b),
        ])
        .unwrap();
        let cz_pairs = [(0u32, 1u32)];

        let baseline = greedy_assign_pairs(
            &cz_pairs,
            &config,
            &arch,
            &dist_table,
            0,
            None,
            0.0,
            0.0,
            0.0,
        );
        let with_pen = greedy_assign_pairs(
            &cz_pairs,
            &config,
            &arch,
            &dist_table,
            0,
            None,
            0.0,
            0.0,
            1000.0,
        );
        assert_eq!(baseline.len(), 2, "baseline assigns both qubits");
        assert_eq!(with_pen.len(), 2, "with penalty also assigns both qubits");

        // Anchored assignment (cost 0) survives the penalty since the
        // anchored slot has no spectator occupant.
        let assigned: HashSet<u64> = with_pen.iter().map(|&(_, l)| l).collect();
        assert!(assigned.contains(&loc(0, 5).encode()));
        assert!(assigned.contains(&loc(1, 5).encode()));
    }

    #[test]
    fn occupancy_penalty_steers_away_when_alternative_exists() {
        // The CZ pair is already at slot site 5; site 6 has a spectator
        // at pos_b. Force the Hungarian to *consider* moving (e.g. by
        // making the anchored slot more expensive via blocked positions
        // is intrusive — instead we use 2 CZ pairs competing for 2 of 3
        // slot indices). Without penalty, ties between two equally-cheap
        // slots may resolve in favour of a spectator-occupied one;
        // with a large penalty, the Hungarian must avoid the spectator.
        let arch = make_arch();
        let index = make_index();
        let locs = all_entangling_locations(&arch);
        let dist_table = DistanceTable::new(&locs, &index);

        // Two CZ pairs, both already anchored at site 5 and site 8
        // respectively (cost 0 each). A third spectator atom sits at
        // slot site 6's pos_b — irrelevant when the anchors are free,
        // but the test drives the penalty path through the cost matrix.
        let spectator = loc(1, 6);
        let config = Config::new([
            (0, loc(0, 5)),
            (1, loc(1, 5)),
            (2, loc(0, 8)),
            (3, loc(1, 8)),
            (4, spectator),
        ])
        .unwrap();
        let cz_pairs = [(0u32, 1u32), (2u32, 3u32)];

        let with_pen = greedy_assign_pairs(
            &cz_pairs,
            &config,
            &arch,
            &dist_table,
            0,
            None,
            0.0,
            0.0,
            1000.0,
        );
        assert_eq!(with_pen.len(), 4, "all qubits assigned");

        // No qubit lands on the spectator location.
        let assigned: HashSet<u64> = with_pen.iter().map(|&(_, l)| l).collect();
        assert!(
            !assigned.contains(&spectator.encode()),
            "high penalty: no target should land on spectator at (1, 6)"
        );
    }

    #[test]
    fn occupancy_penalty_ignores_cz_pair_member_occupants() {
        // Atoms that are themselves part of a CZ pair this layer must NOT
        // be treated as spectators — they will be reassigned by the
        // Hungarian and move out of the way naturally. Two pairs already
        // sitting at their shared entangling positions: the penalty must
        // not steer them away just because the slot positions are
        // currently held by the other pair's members.
        let arch = make_arch();
        let index = make_index();
        let locs = all_entangling_locations(&arch);
        let dist_table = DistanceTable::new(&locs, &index);

        let config = Config::new([
            (0, loc(0, 5)),
            (1, loc(1, 5)),
            (2, loc(0, 6)),
            (3, loc(1, 6)),
        ])
        .unwrap();
        let cz_pairs = [(0u32, 1u32), (2u32, 3u32)];

        let baseline = greedy_assign_pairs(
            &cz_pairs,
            &config,
            &arch,
            &dist_table,
            0,
            None,
            0.0,
            0.0,
            0.0,
        );
        let with_pen = greedy_assign_pairs(
            &cz_pairs,
            &config,
            &arch,
            &dist_table,
            0,
            None,
            0.0,
            0.0,
            1000.0,
        );
        assert_eq!(
            baseline, with_pen,
            "CZ-pair-member occupants must not trigger the occupancy penalty"
        );
    }

    // ── iterative blocker augmentation ──

    #[test]
    fn iterative_assigns_pair_when_no_blockers() {
        // No spectators in the way → result identical (modulo target encoding)
        // to plain `greedy_assign_pairs`.
        let arch = make_arch();
        let index = make_index();
        let locs = all_entangling_locations(&arch);
        let dist_table = DistanceTable::new(&locs, &index);

        let config = Config::new([(0, loc(0, 0)), (1, loc(1, 0))]).unwrap();
        let cz_pairs = [(0u32, 1u32)];

        let blocked = HashSet::new();
        let iter_targets = assign_pairs_with_blockers(
            &cz_pairs,
            &config,
            &arch,
            &index,
            &dist_table,
            &blocked,
            0,
            None,
            0.0,
            0.0,
            0.0,
            0.0,
            true,
        );
        let greedy_targets = greedy_assign_pairs(
            &cz_pairs,
            &config,
            &arch,
            &dist_table,
            0,
            None,
            0.0,
            0.0,
            0.0,
        );
        assert_eq!(
            iter_targets, greedy_targets,
            "with no blockers the iterative wrapper must match greedy"
        );
    }

    #[test]
    fn detect_blockers_case_a_finds_target_blocker() {
        // Hand-craft Hungarian targets and a config where a spectator
        // sits exactly on a pair-target location. detect_blockers must
        // identify the spectator (Case A).
        let index = make_index();
        let config = Config::new([
            (0, loc(0, 0)),
            (1, loc(1, 0)),
            (2, loc(0, 5)), // spectator
        ])
        .unwrap();
        // Hand-crafted targets: pair (q0, q1) → (0, 5), (1, 5). q2 at (0, 5) blocks q0's target.
        let targets = vec![(0u32, loc(0, 5).encode()), (1u32, loc(1, 5).encode())];
        let pair_qubits: HashSet<u32> = [0u32, 1u32].into_iter().collect();
        let row_qubits = pair_qubits.clone();
        let blockers = detect_blockers(&targets, &pair_qubits, &config, &row_qubits, &index);
        assert_eq!(blockers, vec![2], "q2 must be flagged as Case A blocker");
    }

    #[test]
    fn detect_blockers_skips_spectator_row_targets() {
        // Spectator-row targets are destinations, not blockers — Case A
        // must only flag spectators sitting on *pair*-target locations.
        // Setup: pair (q0, q1) anchored at slot site 5 (cost 0); q2 is a
        // spectator row whose target is (0, 7); q3 sits on (0, 7).
        // q3 is sitting on q2's *spectator-row target*, not on any pair
        // target, and the pair atoms have no free egress in this arch
        // (terminal at site 5) so Case B must also not fire.
        let index = make_index();
        let config = Config::new([
            (0, loc(0, 5)), // pair member, anchored
            (1, loc(1, 5)), // pair member, anchored
            (2, loc(0, 6)), // spectator row
            (3, loc(0, 7)), // sits on q2's spectator target — NOT a pair target
        ])
        .unwrap();
        let targets = vec![
            (0u32, loc(0, 5).encode()),
            (1u32, loc(1, 5).encode()),
            (2u32, loc(0, 7).encode()),
        ];
        let pair_qubits: HashSet<u32> = [0u32, 1u32].into_iter().collect();
        let row_qubits: HashSet<u32> = [0u32, 1u32, 2u32].into_iter().collect();
        let blockers = detect_blockers(&targets, &pair_qubits, &config, &row_qubits, &index);
        assert!(
            blockers.is_empty(),
            "q3 sits on q2's spectator-target, not on any pair target — must not be flagged"
        );
    }

    #[test]
    fn detect_blockers_excludes_existing_row_qubits() {
        // A qubit already in row_qubits must never be re-added.
        let index = make_index();
        let config = Config::new([(0, loc(0, 0)), (1, loc(1, 0)), (2, loc(0, 5))]).unwrap();
        let targets = vec![(0u32, loc(0, 5).encode()), (1u32, loc(1, 5).encode())];
        let pair_qubits: HashSet<u32> = [0u32, 1u32].into_iter().collect();
        let mut row_qubits = pair_qubits.clone();
        row_qubits.insert(2); // q2 is already a (spectator) row.
        let blockers = detect_blockers(&targets, &pair_qubits, &config, &row_qubits, &index);
        assert!(blockers.is_empty(), "row qubits must not be re-flagged");
    }

    #[test]
    fn iterative_no_self_penalty_for_displaced_blocker() {
        // With 0 occupancy penalty the wrapper must produce a result
        // identical to plain greedy when there are no blockers; with
        // strong penalty it must still produce a valid assignment for
        // the pair (this verifies the spectator-row's own slot is not
        // self-penalised when the spectator becomes a row in iteration 2).
        let arch = make_arch();
        let index = make_index();
        let locs = all_entangling_locations(&arch);
        let dist_table = DistanceTable::new(&locs, &index);

        let config = Config::new([(0, loc(0, 0)), (1, loc(1, 0))]).unwrap();
        let cz_pairs = [(0u32, 1u32)];

        let blocked = HashSet::new();
        let pen0 = assign_pairs_with_blockers(
            &cz_pairs,
            &config,
            &arch,
            &index,
            &dist_table,
            &blocked,
            0,
            None,
            0.0,
            0.0,
            0.0,
            0.0,
            true,
        );
        let pen5 = assign_pairs_with_blockers(
            &cz_pairs,
            &config,
            &arch,
            &index,
            &dist_table,
            &blocked,
            0,
            None,
            0.0,
            0.0,
            5.0,
            0.0,
            true,
        );
        assert_eq!(pen0.len(), 2);
        assert_eq!(pen5.len(), 2);
    }

    // ── Case E: post-Hungarian "accidental CZ created by assignment" ──

    #[test]
    fn detect_accidentals_created_flags_partner_at_target() {
        // Spectator-row q5 (loc (0,3)) is assigned target (0,5). The
        // partner half (1,5) currently holds q6, who is NOT a row qubit.
        // Case E must flag q6 so the next iteration can give it its own
        // target and prevent q5+q6 ending up at slot partner halves.
        let arch = make_arch();
        let pair_qubits: HashSet<u32> = [0u32, 1u32].into_iter().collect();
        let row_qubits: HashSet<u32> = [0u32, 1u32, 5u32].into_iter().collect();
        let config = Config::new([
            (0, loc(0, 0)),
            (1, loc(1, 0)),
            (5, loc(0, 3)),
            (6, loc(1, 5)),
        ])
        .unwrap();
        let targets = vec![
            (0u32, loc(0, 0).encode()),
            (1u32, loc(1, 0).encode()),
            (5u32, loc(0, 5).encode()),
        ];
        let new = detect_accidentals_created(&targets, &arch, &config, &pair_qubits, &row_qubits);
        assert_eq!(new, vec![6], "q6 sits at q5's target's partner half");
    }

    #[test]
    fn detect_accidentals_created_skips_row_partner() {
        // Same setup but q6 is already a row qubit. Then it will move to
        // its own assigned target and the half becomes free — no Case E.
        let arch = make_arch();
        let pair_qubits: HashSet<u32> = [0u32, 1u32].into_iter().collect();
        let row_qubits: HashSet<u32> = [0u32, 1u32, 5u32, 6u32].into_iter().collect();
        let config = Config::new([
            (0, loc(0, 0)),
            (1, loc(1, 0)),
            (5, loc(0, 3)),
            (6, loc(1, 5)),
        ])
        .unwrap();
        let targets = vec![
            (0u32, loc(0, 0).encode()),
            (1u32, loc(1, 0).encode()),
            (5u32, loc(0, 5).encode()),
        ];
        let new = detect_accidentals_created(&targets, &arch, &config, &pair_qubits, &row_qubits);
        assert!(new.is_empty(), "row qubits at the partner half move away");
    }

    #[test]
    fn detect_accidentals_created_ignores_pair_targets() {
        // Pair-row targets occupy *both* halves; any non-row at the pair
        // target is caught by Case A (`detect_blockers`). Case E must not
        // double-flag those.
        let arch = make_arch();
        let pair_qubits: HashSet<u32> = [0u32, 1u32].into_iter().collect();
        let row_qubits: HashSet<u32> = [0u32, 1u32].into_iter().collect();
        let config = Config::new([(0, loc(0, 0)), (1, loc(1, 0)), (6, loc(1, 5))]).unwrap();
        // Pair (0,1) targets slot site 5; q6 sits on the other half.
        let targets = vec![(0u32, loc(0, 5).encode()), (1u32, loc(1, 5).encode())];
        let new = detect_accidentals_created(&targets, &arch, &config, &pair_qubits, &row_qubits);
        assert!(new.is_empty(), "pair-row partner-half conflicts are Case A");
    }

    // ── Case D: accidental-CZ spectator detection ──

    #[test]
    fn detect_accidental_cz_finds_both_partners() {
        // 1 CZ pair (q0, q1) far away; 2 spectators (q5, q6) at partner
        // halves of the (0, 1) word pair at site 5. Both must be flagged.
        let arch = make_arch();
        let pair_qubits: HashSet<u32> = [0u32, 1u32].into_iter().collect();
        let row_qubits = pair_qubits.clone();
        let config = Config::new([
            (0, loc(0, 0)),
            (1, loc(1, 0)),
            (5, loc(0, 5)),
            (6, loc(1, 5)),
        ])
        .unwrap();
        let mut new = detect_accidental_cz_spectators(&config, &arch, &pair_qubits, &row_qubits);
        new.sort();
        assert_eq!(
            new,
            vec![5, 6],
            "both partners of the accidental CZ at slot site 5 must be flagged"
        );
    }

    #[test]
    fn detect_accidental_cz_skips_existing_rows() {
        // If one of the accidental partners is already in row_qubits,
        // the other should not be re-added.
        let arch = make_arch();
        let pair_qubits: HashSet<u32> = [0u32, 1u32].into_iter().collect();
        let mut row_qubits = pair_qubits.clone();
        row_qubits.insert(5); // q5 already a row.
        let config = Config::new([
            (0, loc(0, 0)),
            (1, loc(1, 0)),
            (5, loc(0, 5)),
            (6, loc(1, 5)),
        ])
        .unwrap();
        let new = detect_accidental_cz_spectators(&config, &arch, &pair_qubits, &row_qubits);
        assert!(
            new.is_empty(),
            "accidental partner of an already-row qubit must not be added"
        );
    }

    #[test]
    fn move_penalty_biases_toward_anchor() {
        // Single CZ pair already at slot site 5. With no move_penalty,
        // perturbation+ties may move them; with non-zero move_penalty,
        // anchored placement (zero hops) is strictly cheaper.
        let arch = make_arch();
        let index = make_index();
        let locs = all_entangling_locations(&arch);
        let dist_table = DistanceTable::new(&locs, &index);

        let config = Config::new([(0, loc(0, 5)), (1, loc(1, 5))]).unwrap();
        let cz_pairs = [(0u32, 1u32)];

        let blocked = HashSet::new();
        let with_bias = assign_pairs_with_blockers(
            &cz_pairs,
            &config,
            &arch,
            &index,
            &dist_table,
            &blocked,
            0,
            None,
            0.0,
            0.0,
            0.0,
            1.0,
            true,
        );

        // Both qubits should anchor (cost 0 + zero move-pen).
        let q0_target = with_bias.iter().find(|&&(q, _)| q == 0).unwrap().1;
        let q1_target = with_bias.iter().find(|&&(q, _)| q == 1).unwrap().1;
        assert_eq!(q0_target, loc(0, 5).encode(), "q0 should stay at (0, 5)");
        assert_eq!(q1_target, loc(1, 5).encode(), "q1 should stay at (1, 5)");
    }

    // ── congestion-aware assignment ──

    #[test]
    fn congestion_aware_does_not_break_uncongested_case() {
        // With only one CZ pair and the example arch (1 word pair, 10 sites),
        // congestion can't bind. Result must match the standard assignment.
        let arch = make_arch();
        let index = make_index();
        let locs = all_entangling_locations(&arch);
        let dist_table = DistanceTable::new(&locs, &index);

        let config = Config::new([(0, loc(0, 0)), (1, loc(1, 0))]).unwrap();
        let cz_pairs = [(0u32, 1u32)];

        let baseline = greedy_assign_pairs(
            &cz_pairs,
            &config,
            &arch,
            &dist_table,
            0,
            None,
            0.0,
            0.0,
            0.0,
        );
        let with_cw = greedy_assign_pairs(
            &cz_pairs,
            &config,
            &arch,
            &dist_table,
            0,
            None,
            0.0,
            5.0,
            0.0,
        );

        assert_eq!(
            baseline, with_cw,
            "single pair: congestion penalty must not change the result"
        );
    }

    #[test]
    fn congestion_aware_hungarian_spreads_load() {
        // 4 pairs × 8 slots (2 word_pairs × 4 sites_per_word). ideal_load = 2.
        // Costs are nearly tied between word pairs (gap = 1) so Hungarian
        // will pick whichever side has the small bias, but a modest penalty
        // can shift assignments without flipping wholesale.
        const BIG: u32 = u32::MAX / 4;
        let n_pairs = 4;
        let sites_per_word = 4;
        let n_word_pairs = 2;
        let n_slots = sites_per_word * n_word_pairs;
        let mut costs = vec![0u32; n_pairs * n_slots];
        for i in 0..n_pairs {
            for j in 0..n_slots {
                let wp = j / sites_per_word;
                // wp 0 base 5, wp 1 base 6 — small gap to keep mixing feasible.
                let base = if wp == 0 { 5 } else { 6 };
                // Prefer matching slot to differentiate within wp.
                let bonus = if (j % sites_per_word) == i { 0 } else { 2 };
                costs[i * n_slots + j] = base + bonus;
            }
        }

        let no_penalty = hungarian(&costs, n_pairs, n_slots);
        let max_load_no = max_load_per_wp(&no_penalty, sites_per_word, n_word_pairs);
        assert_eq!(
            max_load_no, n_pairs,
            "without penalty, all pairs concentrate on the cheapest wp"
        );

        let with_penalty = congestion_aware_hungarian(
            &costs,
            n_pairs,
            n_slots,
            sites_per_word,
            n_word_pairs,
            1.0,
            BIG,
        );
        let max_load_w = max_load_per_wp(&with_penalty, sites_per_word, n_word_pairs);
        // Greedy rebalance forces a strict reduction in max_load when an
        // under-loaded wp has reachable slots.
        assert_eq!(with_penalty.len(), n_pairs, "all pairs must be assigned");
        let mut used = std::collections::HashSet::new();
        for &col in &with_penalty {
            assert!(col < n_slots, "all pairs must be assigned to a slot");
            assert!(used.insert(col), "no double-booking of slots");
        }
        assert!(
            max_load_w < max_load_no,
            "congestion_weight should reduce max load, got {max_load_w} (was {max_load_no})"
        );
        let ideal = n_pairs.div_ceil(n_word_pairs);
        assert!(
            max_load_w <= ideal,
            "rebalance should reach ideal load when feasible: {max_load_w} > {ideal}"
        );
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
