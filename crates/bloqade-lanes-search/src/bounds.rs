//! Admissible completion bounds for branch-and-bound pruning.
//!
//! A search driver holding an incumbent solution of cost `C` may abandon a
//! branch as soon as `g + h(config) >= C`, where `h` is a **lower bound** on
//! the cost of any completion from `config`. With `h ≡ 0` that degenerates to
//! `g >= C`, which only cuts after the whole doomed prefix has been paid for.
//!
//! # Admissibility is a relationship, not a property
//!
//! `h` is admissible *relative to a specific [`Objective`]* — the same bound
//! paired with a different cost model can overestimate and cut away the
//! optimum. [`CompletionBound`] therefore names its objective as an associated
//! type, so combining or consuming mismatched bounds is a compile error, and
//! carries an [`ObjectiveId`] so a mismatch between two *instances* of the same
//! objective type (`tau = 1` vs `tau = 5`) is caught at construction rather
//! than becoming a silently wrong prune.
//!
//! # Ordering versus bounding
//!
//! [`CompletionBound::estimate`] is always the unweighted admissible value.
//! Search *ordering* may legitimately scale or perturb a heuristic — weighted
//! A\* (`f = g + weight * h`), the IDS reversal penalty, the entropy
//! generator's score perturbation — but a *pruning* decision may not use any of
//! those. There is deliberately no weight parameter anywhere in this trait.

use std::collections::{HashMap, HashSet, VecDeque};
use std::marker::PhantomData;

use crate::cost::UniformCost;
use crate::primitives::config::Config;
use crate::primitives::lane_index::LaneIndex;
use crate::primitives::weighted_distance::WeightedDistanceTable;
use crate::traits::{Objective, ObjectiveId};

/// Pruning statistics for one search episode.
///
/// The point of collecting these is to answer "is a stronger bound worth
/// building?" — a bound that never converts a cut, or converts them only one
/// level earlier than `g` alone would have, is not paying for itself.
///
/// Counters cover **both pruning gates** — the resume gate and the
/// child-expansion gate — deduplicated by node, so each distinct pruned node
/// contributes exactly once however many times it is re-tested. Counting only
/// the child gate would miss nearly all post-incumbent pruning, since the
/// driver jumps to a resume node once a goal is found.
///
/// All fields stay zero when the bound is
/// [`TRIVIAL`](CompletionBound::TRIVIAL): the `g >= C` arm of the prune test
/// still fires without a bound, but those cuts are not the bound's work and are
/// deliberately not recorded.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct BoundStats {
    /// Cuts where `g` alone already reached the incumbent — these would have
    /// fired without any bound.
    pub cuts_by_g: u64,
    /// Cuts that **only** the bound could make: `g < C <= g + h`. This is the
    /// bound's actual contribution.
    pub cuts_by_h: u64,
    /// Cuts from `h = +∞`: the bound proved no completion exists at all,
    /// independent of any incumbent.
    pub cuts_infeasible: u64,
    /// Summed depth at which `cuts_by_h` fired.
    pub cut_depth_sum: u64,
    /// Summed depth at which `g` alone *would* have reached the incumbent for
    /// those same cuts, i.e. `depth + ceil((C - g) / min_shot_cost)`.
    ///
    /// Against [`Self::cut_depth_sum`] this is the depth ratio: how much
    /// earlier the bound fired, and so roughly how much subtree it saved.
    pub cut_depth_g_only_sum: u64,
    /// `h(root)`: a lower bound on the optimal cost of the whole instance.
    ///
    /// Certified even though branch generation is sampled — `h` depends only
    /// on the configuration, never on which candidates the generator happened
    /// to produce. Against the final incumbent it gives an optimality gap.
    pub root_lower_bound: f64,
    /// Cost of the best solution found, or `None` if none was.
    ///
    /// `Option` rather than a `NaN` sentinel: `None` is unambiguously "no
    /// solution", so no cost value can be mistaken for it, and — since
    /// `BoundStats` derives `PartialEq` — a stats value stays equal to itself,
    /// which a `NaN` field would break (`NaN != NaN`).
    pub incumbent_cost: Option<f64>,
    /// Whether a non-trivial bound was in use.
    ///
    /// Without it, an unbounded run is indistinguishable from a bounded run
    /// that pruned nothing: both report zero cuts, and `root_lower_bound` of
    /// `0.0` would yield a spurious "gap" of 1.0 that reads like a measurement
    /// rather than an absence of one.
    pub bound_enabled: bool,
}

/// Tolerance for the optimality-gap sign test, in units of the gap itself
/// (which is a ratio, so an absolute tolerance is meaningful here).
///
/// Sized to absorb accumulated `f64` rounding in `g` against a `h(root)` that
/// matches it exactly, without hiding a real admissibility violation — those
/// are whole shots wide, not ulps.
const GAP_SLACK: f64 = 1e-9;

impl BoundStats {
    /// Total cuts made, however classified.
    pub fn total_cuts(&self) -> u64 {
        self.cuts_by_g + self.cuts_by_h + self.cuts_infeasible
    }

    /// Certified optimality gap `(incumbent - h(root)) / incumbent`, or `None`
    /// when no bound was in use, no solution was found, the incumbent is zero,
    /// or `h(root)` is not a finite measurement.
    ///
    /// `0.0` means the incumbent is provably optimal. A **negative** result is
    /// possible and is deliberately not clamped: `h(root) > incumbent` means
    /// the bound overestimated the true remaining cost, i.e. it is
    /// inadmissible, and pruning derived from it may have discarded the
    /// optimum. That is the one condition here that must never be silent —
    /// clamping it to `0.0` would report the strongest possible claim
    /// ("provably optimal") for the run least entitled to it.
    pub fn optimality_gap(&self) -> Option<f64> {
        if !self.bound_enabled {
            return None;
        }
        // `None` incumbent: no solution to measure against. A `Some` is always a
        // finite `g_score`, but the guard stays defensive.
        let incumbent = self.incumbent_cost?;
        if !incumbent.is_finite() || incumbent <= 0.0 {
            return None;
        }
        // An infinite `h(root)` is an infeasibility *claim*, not a measured
        // lower bound; there is no gap to report against a solution the bound
        // says cannot exist.
        if !self.root_lower_bound.is_finite() {
            return None;
        }
        let gap = (incumbent - self.root_lower_bound) / incumbent;
        debug_assert!(
            gap > -GAP_SLACK,
            "inadmissible bound: h(root) {} exceeds the incumbent {incumbent}",
            self.root_lower_bound,
        );
        // Absorb float noise around an exactly-optimal incumbent only.
        Some(if gap.abs() <= GAP_SLACK { 0.0 } else { gap })
    }
}

/// A lower bound on the cost to complete a partial plan, admissible with
/// respect to [`CompletionBound::Obj`].
///
/// Implementors must guarantee: for every reachable `config`, `estimate`
/// returns a value no greater than the true cost of the cheapest completion
/// under `Obj`. Returning `0.0` is always sound (and useless);
/// [`f64::INFINITY`] asserts that no completion exists at all.
pub trait CompletionBound: Sync {
    /// The objective this bound is admissible for.
    type Obj: Objective;

    /// `true` only for [`NoBound`]. Lets a driver monomorphize the bound test
    /// away entirely when bounding is disabled, so the "off" configuration is
    /// not merely equivalent but literally the same code.
    const TRIVIAL: bool = false;

    /// Identity of the objective *instance* this bound was built against.
    fn objective_id(&self) -> ObjectiveId;

    /// Lower bound on remaining cost. [`f64::INFINITY`] means infeasible.
    fn estimate(&self, config: &Config) -> f64;
}

/// The trivial bound: `h ≡ 0`, equivalent to no bounding at all.
///
/// Carries its objective's id purely so the API stays uniform; a bound that
/// never prunes cannot be mispaired in any observable way.
pub struct NoBound<O> {
    objective_id: ObjectiveId,
    _obj: PhantomData<fn() -> O>,
}

impl<O: Objective> NoBound<O> {
    pub fn for_objective(objective: &O) -> Self {
        Self {
            objective_id: objective.id(),
            _obj: PhantomData,
        }
    }
}

impl<O: Objective> CompletionBound for NoBound<O> {
    type Obj = O;
    const TRIVIAL: bool = true;

    fn objective_id(&self) -> ObjectiveId {
        self.objective_id
    }

    fn estimate(&self, _config: &Config) -> f64 {
        0.0
    }
}

/// `h0`: the max over unresolved atoms of the weighted distance from an atom's
/// location to its target.
///
/// # Why the max, and why it is admissible
///
/// Fix any completion and any unresolved atom `i`, and let `l_1..l_k` be the
/// lanes `i` traverses, in shots `s_1..s_k`. Those shots are **distinct** — a
/// `MoveSet` assigns each qubit at most one destination, so an atom advances at
/// most one lane hop per shot. Then, writing `w` for `Objective::lane_weight`:
///
/// ```text
/// remaining cost = Σ_{shots s} edge_cost(s)     (C1: per-shot additive)
///               >= Σ_{j=1..k}  edge_cost(s_j)   (C2: costs are non-negative)
///               >= Σ_{j=1..k}  w(l_j)           (C3: shot cost dominates each of its lanes)
///               >= wdist(loc_i, target_i)       (l_1..l_k is a path; wdist is the cheapest)
/// ```
///
/// So each unresolved atom's weighted distance is individually a floor on the
/// *total* remaining cost, and the max is the tightest of those floors.
///
/// **Never sum them.** A shot moves many atoms in parallel, so summing
/// double-counts shared shots, overestimates, and would cut away optimal
/// solutions.
pub struct WeightedDistanceBound<O> {
    table: WeightedDistanceTable,
    /// `(qubit, encoded target)` — the same target list the goal and the move
    /// generators use, so "unresolved" means the same thing everywhere.
    targets: Vec<(u32, u64)>,
    /// The carved-out locations, retained to tell "this atom is parked *on* a
    /// blocked site" (legal at the root, and no floor can be certified for it)
    /// apart from "this atom's routes were all severed by the carve" (a genuine
    /// infeasibility proof). Both leave the source absent from the table.
    blocked: HashSet<u64>,
    _obj: PhantomData<fn() -> O>,
}

impl<O: Objective> WeightedDistanceBound<O> {
    /// Build the per-solve weighted distance table and bind it to `targets`.
    ///
    /// `blocked` is carved out of the graph — see [`WeightedDistanceTable`] for
    /// why that tightens the bound while keeping it sound.
    pub fn new(
        objective: &O,
        targets: &[(u32, u64)],
        index: &LaneIndex,
        blocked: &HashSet<u64>,
    ) -> Self {
        let target_locs: Vec<u64> = targets.iter().map(|&(_, enc)| enc).collect();
        Self {
            table: WeightedDistanceTable::new(&target_locs, index, blocked, objective),
            targets: targets.to_vec(),
            blocked: blocked.clone(),
            _obj: PhantomData,
        }
    }

    /// Borrow the underlying table (diagnostics, instrumentation).
    pub fn table(&self) -> &WeightedDistanceTable {
        &self.table
    }
}

impl<O: Objective> CompletionBound for WeightedDistanceBound<O> {
    type Obj = O;

    fn objective_id(&self) -> ObjectiveId {
        self.table.objective_id()
    }

    fn estimate(&self, config: &Config) -> f64 {
        let mut worst = 0.0_f64;
        for &(qid, target_enc) in &self.targets {
            // A qubit the configuration does not know about cannot be routed
            // to its target at any price.
            let Some(loc) = config.location_of(qid) else {
                return f64::INFINITY;
            };
            let loc_enc = loc.encode();
            if loc_enc == target_enc {
                continue; // resolved — contributes nothing
            }
            let Some(d) = self.table.distance(loc_enc, target_enc) else {
                // An atom parked *on* a blocked site is absent from the carved
                // graph, so the table cannot answer for it — but that is
                // ignorance, not proof. `target_solver` notes that "nothing
                // forbids the root placement from sitting on one, and the
                // generators never check it", so this is a legal input shape,
                // and the atom's first move takes it back onto the graph.
                // Certifying no floor (`0.0`) is sound; claiming infeasibility
                // would cut a solvable instance at its root.
                //
                // Any *other* absence — an unblocked source whose routes the
                // carve severed — is a genuine proof that no completion exists,
                // and must stay `+inf`.
                if self.blocked.contains(&loc_enc) {
                    continue;
                }
                return f64::INFINITY; // no unblocked route to the target
            };
            worst = worst.max(d);
        }
        worst
    }
}

/// Words of group-mask storage `BusCutBound::estimate` keeps on the stack.
/// Architectures with more than `64 * INLINE_WORDS` bus groups spill to the heap.
const INLINE_WORDS: usize = 4;

/// How one targeted atom enters a [`BusCutBound`] estimate.
enum Atom {
    /// A targeted qubit is absent, or its target is unreachable from an
    /// unblocked site: no completion exists.
    Infeasible,
    /// Already at its target, or parked on a blocked site (see
    /// [`WeightedDistanceBound`]): contributes nothing.
    Idle,
    /// Must still move: `d` shots at least, from table row `row` to column `col`.
    Moving { d: f64, row: usize, col: usize },
}

/// The bus-cut bound, for [`UniformCost`] only.
///
/// A *group* is `(zone, bus, move type, direction)` — the key
/// `check_lane_group_consistency` enforces — and every shot uses exactly one.
/// Under `UniformCost` a shot costs `1`, so counting shots is counting cost.
///
/// ```text
/// h = max( h0, |R| )
/// ```
///
/// where, on the lane graph `H` with the solve's `blocked` sites carved out (the
/// graph [`WeightedDistanceBound`] uses):
///
/// - `h0` is the largest hop distance from an unresolved atom to its target;
/// - group `G` is **required** for an atom when its target is unreachable from
///   its location once `G`'s lanes are removed from `H`, and `R` is the union of
///   the required groups over all atoms.
///
/// # Why it is admissible
///
/// Any completion moves atom `i` along a walk in `H` (occupancy and collisions
/// only remove moves). If `G` is required for `i`, every such walk crosses a
/// lane of `G`, so the completion contains a shot of group `G`. A shot has one
/// group, so the shots for distinct groups in `R` are distinct, and `|R|` is a
/// floor. `h0` is admissible by the argument on [`WeightedDistanceBound`], and
/// the max of two floors is a floor.
///
/// Only the static `blocked` set is carved — never the sites of atoms that are
/// part of the solve — so `H` is no smaller than what the move generators allow.
///
/// `h` is also consistent (`h(n) <= 1 + h(n')` across a shot), which the
/// closed-set frontier driver relies on to stay optimal: both `h0` and `|R|`
/// drop by at most one per shot.
///
/// # Cost
///
/// All the graph work happens in [`BusCutBound::new`], per target and per group:
/// one reverse reachability pass over the lane graph with that group's lanes
/// removed. `estimate` is table lookups and popcounts.
pub struct BusCutBound {
    /// `d_i`: the very table `h0` reads, so the two bounds agree on distances.
    table: WeightedDistanceTable,
    /// `(qubit, encoded target)`, as in [`WeightedDistanceBound`].
    targets: Vec<(u32, u64)>,
    blocked: HashSet<u64>,
    /// `encoded location → row`, over the carved graph.
    loc_index: HashMap<u64, usize>,
    /// `encoded target → column`, unblocked targets only.
    target_col: HashMap<u64, usize>,
    n_loc: usize,
    /// `u64` words per group mask.
    words: usize,
    /// `required[(col * n_loc + row) * words ..][..words]`: the groups required
    /// for an atom at `row` heading for target `col`.
    required: Vec<u64>,
}

impl BusCutBound {
    /// Build the per-solve tables. `blocked` is carved out exactly as
    /// [`WeightedDistanceBound::new`] carves it.
    pub fn new(
        objective: &UniformCost,
        targets: &[(u32, u64)],
        index: &LaneIndex,
        blocked: &HashSet<u64>,
    ) -> Self {
        let target_locs: Vec<u64> = targets.iter().map(|&(_, enc)| enc).collect();
        let table = WeightedDistanceTable::new(&target_locs, index, blocked, objective);

        // The carved lane graph again, this time remembering each lane's group.
        let mut loc_index: HashMap<u64, usize> = HashMap::new();
        let mut preds: Vec<Vec<(usize, usize)>> = Vec::new(); // dst → [(src, group)]
        let mut intern = |enc: u64, preds: &mut Vec<Vec<(usize, usize)>>| {
            *loc_index.entry(enc).or_insert_with(|| {
                preds.push(Vec::new());
                preds.len() - 1
            })
        };
        let mut n_groups = 0_usize;
        for (mt, bus_id, zone_id, dir) in index.bus_groups() {
            let group = n_groups;
            n_groups += 1;
            for &lane in index.lanes_for(mt, bus_id, zone_id, dir) {
                let Some((src, dst)) = index.endpoints(&lane) else {
                    continue;
                };
                let (src_enc, dst_enc) = (src.encode(), dst.encode());
                if blocked.contains(&src_enc) || blocked.contains(&dst_enc) {
                    continue;
                }
                let (s, d) = (intern(src_enc, &mut preds), intern(dst_enc, &mut preds));
                preds[d].push((s, group));
            }
        }

        let mut unique: Vec<u64> = target_locs
            .iter()
            .copied()
            .filter(|t| !blocked.contains(t))
            .collect();
        unique.sort_unstable();
        unique.dedup();
        // An isolated target still needs a row, so that the atom already on it
        // and the table's `distance(t, t)` agree on what a known location is.
        for &t in &unique {
            intern(t, &mut preds);
        }

        let n_loc = preds.len();
        let words = n_groups.div_ceil(64).max(1);
        let target_col: HashMap<u64, usize> =
            unique.iter().enumerate().map(|(c, &t)| (t, c)).collect();
        let mut required = vec![0_u64; unique.len() * n_loc * words];

        for (col, &target) in unique.iter().enumerate() {
            let t_idx = loc_index[&target];
            let reach = reverse_reach(&preds, t_idx, None);
            for group in 0..n_groups {
                let without = reverse_reach(&preds, t_idx, Some(group));
                for row in 0..n_loc {
                    if reach[row] && !without[row] {
                        required[(col * n_loc + row) * words + group / 64] |= 1 << (group % 64);
                    }
                }
            }
        }

        Self {
            table,
            targets: targets.to_vec(),
            blocked: blocked.clone(),
            loc_index,
            target_col,
            n_loc,
            words,
            required,
        }
    }

    /// Classify one targeted atom, with the same infeasibility and
    /// blocked-source rules as [`WeightedDistanceBound::estimate`].
    fn atom(&self, config: &Config, qid: u32, target_enc: u64) -> Atom {
        let Some(loc) = config.location_of(qid) else {
            return Atom::Infeasible;
        };
        let loc_enc = loc.encode();
        if loc_enc == target_enc {
            return Atom::Idle;
        }
        let Some(d) = self.table.distance(loc_enc, target_enc) else {
            return if self.blocked.contains(&loc_enc) {
                Atom::Idle
            } else {
                Atom::Infeasible
            };
        };
        // The table answered, so both ends are in the carved graph we interned.
        let row = self.loc_index[&loc_enc];
        let col = self.target_col[&target_enc];
        Atom::Moving { d, row, col }
    }
}

/// Which locations reach `target_idx` on the reversed graph, optionally with one
/// group's lanes removed.
fn reverse_reach(
    preds: &[Vec<(usize, usize)>],
    target_idx: usize,
    skip: Option<usize>,
) -> Vec<bool> {
    let mut seen = vec![false; preds.len()];
    let mut queue = VecDeque::from([target_idx]);
    seen[target_idx] = true;
    while let Some(node) = queue.pop_front() {
        for &(pred, group) in &preds[node] {
            if !seen[pred] && skip != Some(group) {
                seen[pred] = true;
                queue.push_back(pred);
            }
        }
    }
    seen
}

impl CompletionBound for BusCutBound {
    type Obj = UniformCost;

    fn objective_id(&self) -> ObjectiveId {
        self.table.objective_id()
    }

    fn estimate(&self, config: &Config) -> f64 {
        let mut inline = [0_u64; INLINE_WORDS];
        let mut spill: Vec<u64>;
        let required: &mut [u64] = if self.words <= INLINE_WORDS {
            &mut inline[..self.words]
        } else {
            spill = vec![0; self.words];
            &mut spill
        };

        // h0, and R: every group some atom cannot do without.
        let mut farthest = 0.0_f64;
        for &(qid, target_enc) in &self.targets {
            match self.atom(config, qid, target_enc) {
                Atom::Infeasible => return f64::INFINITY,
                Atom::Idle => {}
                Atom::Moving { d, row, col } => {
                    farthest = farthest.max(d);
                    let base = (col * self.n_loc + row) * self.words;
                    for (w, slot) in required.iter_mut().enumerate() {
                        *slot |= self.required[base + w];
                    }
                }
            }
        }
        farthest.max(required.iter().map(|w| w.count_ones()).sum::<u32>() as f64)
    }
}

/// Comparison slack for the contract assertions, relative to the magnitude
/// being compared.
///
/// `f64::EPSILON` is the ulp at 1.0 and is meaningless as an absolute tolerance
/// for costs in the hundreds, where a single ulp is already far larger — an
/// objective that scales durations can differ from its own lane weight by
/// rounding alone and would fail spuriously.
#[cfg(any(test, feature = "test-util"))]
#[inline]
fn slack(magnitude: f64) -> f64 {
    magnitude.abs().max(1.0) * 1e-12
}

/// Assert the [`Objective`] contract (C2, C3, C4, C5) for every lane and a
/// selection of multi-lane shots on `index`.
///
/// These are the properties a weighted-distance bound's admissibility rests on,
/// so every `Objective` implementation should be run through this in a test
/// rather than have the argument re-made by review.
///
/// Test-only: the body is entirely assertions, so it is gated out of normal
/// builds rather than sitting in the public API where a caller could take a
/// panic in release. Enable the `test-util` feature to use it from another
/// crate's tests.
///
/// # Panics
///
/// On the first violation, naming the lane and the two costs involved.
#[cfg(any(test, feature = "test-util"))]
pub fn assert_objective_contract(objective: &impl Objective, index: &LaneIndex) {
    use crate::primitives::graph::MoveSet;

    // `edge_cost` for both shipped objectives is configuration-independent;
    // an empty configuration keeps this helper usable for any architecture.
    let config = Config::new([]).expect("empty configuration is valid");

    let min_shot = objective.min_shot_cost();
    assert!(
        min_shot > 0.0 && min_shot.is_finite(),
        "C4: min_shot_cost must be positive and finite, got {min_shot}"
    );

    let mut checked = 0_usize;
    for (mt, bus_id, zone_id, dir) in index.bus_groups() {
        let lanes = index.lanes_for(mt, bus_id, zone_id, dir);

        // Single-lane shots, plus one shot per group combining several lanes:
        // C3 must hold for every lane of a multi-lane shot, which is where a
        // "discount for co-moved atoms" style objective would break.
        let mut shots: Vec<MoveSet> = lanes.iter().map(|&l| MoveSet::new([l])).collect();
        if lanes.len() > 1 {
            shots.push(MoveSet::new(lanes.iter().copied().take(8)));
        }

        for shot in &shots {
            let cost = objective.edge_cost(shot, &config, &config);
            assert!(cost >= 0.0, "C2: negative shot cost {cost} for {shot:?}");
            assert!(
                cost >= min_shot - slack(min_shot),
                "C4: shot cost {cost} is below min_shot_cost {min_shot} for {shot:?}"
            );
            let decoded = shot.decode();
            for &lane in &decoded {
                let w = objective.lane_weight(lane);
                assert!(
                    cost >= w - slack(w),
                    "C3: shot cost {cost} is below lane weight {w} for {lane:?}"
                );
            }
            // C5: dropping any one lane from a multi-lane shot must not raise
            // its cost, at equal movers.
            //
            // Checked twice, because `edge_cost` is handed both configurations
            // and may inspect them. With the empty configuration every lane is
            // a filler and "same movers" holds trivially — but an objective
            // that discounts added lanes *only when a real atom moves* would
            // satisfy that and still violate C5 on every shot the generators
            // actually produce, which would silently make the exhaustive
            // generator's mover-tight deduplication lossy. So the second pass
            // puts an atom on the first lane's source and drops only the
            // others, whose sources stay empty: the mover set is identical
            // across both shots, which is exactly the condition C5 is stated
            // under.
            if decoded.len() > 1 {
                for (i, dropped) in decoded.iter().enumerate() {
                    let without = MoveSet::new(
                        decoded
                            .iter()
                            .enumerate()
                            .filter(|&(j, _)| j != i)
                            .map(|(_, &l)| l),
                    );
                    let cost_without = objective.edge_cost(&without, &config, &config);
                    assert!(
                        cost_without <= cost + slack(cost),
                        "C5: dropping filler lane {dropped:?} from {shot:?} raised the shot cost \
                         from {cost} to {cost_without}; adding a lane must never make a shot cheaper"
                    );
                }

                // Sources within a bus group are unique, so only lane 0's
                // source is occupied and every other lane is still a filler.
                if let Some((src, dst)) = index.endpoints(&decoded[0]) {
                    let from = Config::new([(0u32, src)]).expect("one atom is a valid config");
                    let to = Config::new([(0u32, dst)]).expect("one atom is a valid config");
                    let moved = objective.edge_cost(shot, &from, &to);
                    for (i, dropped) in decoded.iter().enumerate().skip(1) {
                        let without = MoveSet::new(
                            decoded
                                .iter()
                                .enumerate()
                                .filter(|&(j, _)| j != i)
                                .map(|(_, &l)| l),
                        );
                        let cost_without = objective.edge_cost(&without, &from, &to);
                        assert!(
                            cost_without <= moved + slack(moved),
                            "C5: with an atom moving on {:?}, dropping filler lane {dropped:?} \
                             from {shot:?} raised the shot cost from {moved} to {cost_without}; \
                             adding a lane must never make a shot cheaper",
                            decoded[0]
                        );
                    }
                }
            }
            checked += 1;
        }
    }
    assert!(
        checked > 0,
        "architecture exposed no lanes — the contract check verified nothing"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost::{UniformCost, WeightedDuration};
    use crate::primitives::graph::MoveSet;
    use crate::test_utils::{example_arch_json, loc};
    use crate::traits::CostFn;
    use bloqade_lanes_bytecode_core::arch::addr::{LaneAddr, LocationAddr};
    use bloqade_lanes_bytecode_core::arch::types::ArchSpec;

    /// A no-solution `BoundStats` must equal itself.
    ///
    /// `BoundStats` derives `PartialEq`, and the "no solution" incumbent used to
    /// be `f64::NAN`, which is unequal to itself — so a determinism or
    /// regression comparison of two identical unsolved results would report them
    /// as different. `None` fixed that; this pins it.
    #[test]
    fn a_no_solution_stats_value_equals_itself() {
        // Two independently built but identical no-solution results. Under the
        // old `f64::NAN` incumbent these compared *unequal* (`NaN != NaN`), so a
        // determinism check diffing two identical unsolved runs would flag a
        // spurious difference.
        let a = BoundStats {
            bound_enabled: true,
            incumbent_cost: None,
            ..BoundStats::default()
        };
        let b = BoundStats {
            bound_enabled: true,
            incumbent_cost: None,
            ..BoundStats::default()
        };
        assert_eq!(a, b);
        assert_eq!(BoundStats::default(), BoundStats::default());
        // And with no incumbent there is no gap to report.
        assert_eq!(a.optimality_gap(), None);
    }

    fn make_index() -> LaneIndex {
        let spec: ArchSpec = serde_json::from_str(example_arch_json()).unwrap();
        LaneIndex::new(spec)
    }

    fn bound_for(
        targets: &[(u32, u64)],
        index: &LaneIndex,
        blocked: &HashSet<u64>,
    ) -> WeightedDistanceBound<UniformCost> {
        WeightedDistanceBound::new(&UniformCost, targets, index, blocked)
    }

    // ── Objective contract ──

    #[test]
    fn uniform_cost_satisfies_the_objective_contract() {
        assert_objective_contract(&UniformCost, &make_index());
    }

    #[test]
    fn weighted_duration_satisfies_the_objective_contract() {
        let index = make_index();
        for tau in [0.5, 1.0, 10.0, 1000.0] {
            assert_objective_contract(&WeightedDuration::new(&index, tau), &index);
        }
    }

    /// An objective that rebates a shot for the lanes it carries: a shot costs
    /// `1 + 0.5 / len`, so *adding* a filler lane makes it cheaper. C2, C3 and
    /// C4 all hold (every shot costs at least `1.0`, which is both the lane
    /// weight and the shot floor), so the only constraint it breaks is C5.
    struct FillerDiscount;

    impl CostFn for FillerDiscount {
        fn edge_cost(&self, move_set: &MoveSet, _from: &Config, _to: &Config) -> f64 {
            1.0 + 0.5 / move_set.len().max(1) as f64
        }
    }

    impl Objective for FillerDiscount {
        fn lane_weight(&self, _lane: LaneAddr) -> f64 {
            1.0
        }
        fn min_shot_cost(&self) -> f64 {
            1.0
        }
        fn id(&self) -> ObjectiveId {
            ObjectiveId {
                kind: "test-filler-discount",
                params: 0,
            }
        }
    }

    /// The contract check must catch a lane-set *non*-monotone objective and
    /// name C5, so that a reviewer reading the panic knows which of the five
    /// constraints failed rather than which lane happened to trip it.
    #[test]
    fn a_filler_discounting_objective_fails_c5_by_name() {
        let index = make_index();
        let outcome =
            std::panic::catch_unwind(|| assert_objective_contract(&FillerDiscount, &index));
        let payload = outcome.expect_err("FillerDiscount must violate the contract");
        let message = payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
            .expect("panic payload is a message");
        assert!(
            message.starts_with("C5:"),
            "expected the C5 assertion to fire first, got: {message}"
        );
    }

    /// An objective that is lane-set monotone on an *empty* configuration and
    /// non-monotone as soon as an atom really moves.
    ///
    /// This is the hole the review found: `edge_cost` is handed both
    /// configurations and may inspect them, so a check that only ever passes
    /// the empty one certifies nothing about the shots the generators actually
    /// produce. C5 is what makes the exhaustive generator's mover-tight
    /// deduplication sound, so a violation it cannot see would quietly make
    /// that deduplication lossy.
    struct MoverOnlyDiscount;

    impl CostFn for MoverOnlyDiscount {
        fn edge_cost(&self, move_set: &MoveSet, from: &Config, to: &Config) -> f64 {
            // Indistinguishable from `UniformCost` while nothing moves.
            if from == to {
                return 1.0;
            }
            1.0 + 0.5 / move_set.len().max(1) as f64
        }
    }

    impl Objective for MoverOnlyDiscount {
        fn lane_weight(&self, _lane: LaneAddr) -> f64 {
            1.0
        }
        fn min_shot_cost(&self) -> f64 {
            1.0
        }
        fn id(&self) -> ObjectiveId {
            ObjectiveId {
                kind: "test-mover-only-discount",
                params: 0,
            }
        }
    }

    /// The contract check must exercise a real mover, not just the empty
    /// configuration, or an objective like this one passes it.
    #[test]
    fn a_discount_that_needs_a_mover_still_fails_c5() {
        let index = make_index();
        let outcome =
            std::panic::catch_unwind(|| assert_objective_contract(&MoverOnlyDiscount, &index));
        let payload = outcome.expect_err("MoverOnlyDiscount must violate the contract");
        let message = payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
            .expect("panic payload is a message");
        assert!(
            message.starts_with("C5:") && message.contains("with an atom moving on"),
            "expected the mover-configuration C5 assertion, got: {message}"
        );
    }

    // ── h0 ──

    #[test]
    fn zero_when_every_atom_is_resolved() {
        let index = make_index();
        let targets = [(0u32, loc(0, 5).encode()), (1u32, loc(1, 5).encode())];
        let bound = bound_for(&targets, &index, &HashSet::new());
        let config = Config::new([(0, loc(0, 5)), (1, loc(1, 5))]).unwrap();
        assert_eq!(bound.estimate(&config), 0.0);
    }

    #[test]
    fn single_atom_estimate_is_the_exact_weighted_distance() {
        let index = make_index();
        let target = loc(1, 0).encode();
        let targets = [(0u32, target)];
        let blocked = HashSet::new();
        let bound = bound_for(&targets, &index, &blocked);
        let config = Config::new([(0, loc(0, 0))]).unwrap();

        let expected = bound
            .table()
            .distance(loc(0, 0).encode(), target)
            .expect("reachable");
        assert_eq!(bound.estimate(&config), expected);
        // Under UniformCost the weighted distance is a hop count: site→word→site.
        assert_eq!(expected, 3.0);
    }

    #[test]
    fn takes_the_max_not_the_sum_over_atoms() {
        let index = make_index();
        // q0: site 0 → word 1 site 0 is 3 hops; q1: word 0 site 5 → word 1
        // site 5 is 1 hop. Distinct values, so max and sum cannot coincide.
        let targets = [(0u32, loc(1, 0).encode()), (1u32, loc(1, 5).encode())];
        let blocked = HashSet::new();
        let bound = bound_for(&targets, &index, &blocked);
        let config = Config::new([(0, loc(0, 0)), (1, loc(0, 5))]).unwrap();

        let d0 = bound
            .table()
            .distance(loc(0, 0).encode(), loc(1, 0).encode())
            .expect("q0 target reachable");
        let d1 = bound
            .table()
            .distance(loc(0, 5).encode(), loc(1, 5).encode())
            .expect("q1 target reachable");
        assert!(d0 != d1, "test needs distinct distances, got {d0} and {d1}");

        assert_eq!(bound.estimate(&config), d0.max(d1));
        assert!(
            bound.estimate(&config) < d0 + d1,
            "summing would overestimate: shots move many atoms in parallel"
        );
    }

    #[test]
    fn infinite_when_a_target_is_unreachable() {
        let index = make_index();
        let targets = [(0u32, loc(99, 99).encode())];
        let bound = bound_for(&targets, &index, &HashSet::new());
        let config = Config::new([(0, loc(0, 0))]).unwrap();
        assert_eq!(bound.estimate(&config), f64::INFINITY);
    }

    #[test]
    fn infinite_when_a_targeted_qubit_is_missing_from_the_configuration() {
        let index = make_index();
        let targets = [(7u32, loc(0, 5).encode())];
        let bound = bound_for(&targets, &index, &HashSet::new());
        let config = Config::new([(0, loc(0, 0))]).unwrap();
        assert_eq!(bound.estimate(&config), f64::INFINITY);
    }

    /// Carving blocked sites out can only raise the bound, never lower it —
    /// this is the tightening that makes h0 stronger than the arch-only hop
    /// heuristic, and it must never go the other way.
    #[test]
    fn blocking_never_loosens_the_bound() {
        let index = make_index();
        let target = loc(1, 0).encode();
        let targets = [(0u32, target)];
        let config = Config::new([(0, loc(0, 0))]).unwrap();

        let open = bound_for(&targets, &index, &HashSet::new()).estimate(&config);
        let blocked: HashSet<u64> = [loc(0, 5).encode()].into_iter().collect();
        let carved = bound_for(&targets, &index, &blocked).estimate(&config);

        assert!(
            carved >= open,
            "blocked exclusion must not lower the bound: {carved} < {open}"
        );
    }

    /// A duration-weighted objective yields a strictly larger bound than the
    /// unit-cost one, since each of its lane weights exceeds 1.
    #[test]
    fn bound_scales_with_the_objective() {
        let index = make_index();
        let target = loc(1, 0).encode();
        let targets = [(0u32, target)];
        let blocked = HashSet::new();
        let config = Config::new([(0, loc(0, 0))]).unwrap();

        let uniform = bound_for(&targets, &index, &blocked).estimate(&config);
        let weighted = WeightedDistanceBound::new(
            &WeightedDuration::new(&index, 10.0),
            &targets,
            &index,
            &blocked,
        )
        .estimate(&config);

        assert!(
            weighted > uniform,
            "duration-weighted bound {weighted} should exceed unit-cost {uniform}"
        );
    }

    // Compile-time, not runtime: a driver monomorphizes its bound test on
    // `TRIVIAL`, so "bounding disabled" must compile to the same code as no
    // bounding at all. If these ever became runtime-only facts, the flag-off
    // path would silently start branching per node.
    const _: () = assert!(<NoBound<UniformCost> as CompletionBound>::TRIVIAL);
    const _: () = assert!(!<WeightedDistanceBound<UniformCost> as CompletionBound>::TRIVIAL);

    #[test]
    fn no_bound_estimates_zero() {
        let config = Config::new([(0, loc(0, 0))]).unwrap();
        let none = NoBound::<UniformCost>::for_objective(&UniformCost);
        assert_eq!(none.estimate(&config), 0.0);
    }

    // ── Admissibility against brute-force optima ────────────────

    /// Minimum moveset count to reach `targets`, by breadth-first search over
    /// [`ExhaustiveGenerator`]'s successors under [`UniformCost`].
    ///
    /// `None` when the budget ran out before a goal was found — the caller must
    /// then skip the instance rather than treat "unsolved" as an optimum.
    ///
    /// # What this is and is not
    ///
    /// `ExhaustiveGenerator` enumerates AOD rectangles **one bus triplet at a
    /// time**, so a moveset combining lanes from two triplets is never
    /// produced. Whatever the architecture permits, this is a search over a
    /// *subgraph* of the true move space, so the value returned is an **upper
    /// bound** on the true optimum, not necessarily the optimum itself.
    ///
    /// That is the safe direction for asserting `h0 <= optimum`: a reference
    /// that is too large can never make an admissible bound look inadmissible,
    /// so this cannot fail spuriously. It does mean the check is a *necessary
    /// condition* rather than a proof — a violation hidden between the true
    /// optimum and this reference would slip through. It still catches the
    /// mistakes that matter most, notably summing per-atom distances instead of
    /// maxing them, which inflates `h0` by up to a factor of the atom count.
    ///
    /// [`h0_is_exact_for_a_single_atom`] covers the one case where the true
    /// optimum is known independently of any generator.
    fn brute_force_optimum(
        initial: &[(u32, LocationAddr)],
        targets: &[(u32, u64)],
        blocked: &HashSet<u64>,
        index: &LaneIndex,
        budget: u32,
    ) -> Option<f64> {
        use crate::drivers::frontier::{self, BfsFrontier};
        use crate::generators::exhaustive::{ExhaustiveGenerator, SeedPolicy};
        use crate::goals::AllAtTarget;
        use crate::observer::NoOpObserver;
        use crate::primitives::context::{SearchContext, SearchState};
        use crate::primitives::distance::DistanceTable;
        use crate::scorers::DistanceScorer;

        let target_locs: Vec<u64> = targets.iter().map(|&(_, l)| l).collect();
        let dist_table = DistanceTable::new(&target_locs, index);
        let ctx = SearchContext {
            index,
            dist_table: &dist_table,
            blocked,
            targets,
            cz_pairs: None,
        };
        let goal = AllAtTarget::new(targets);
        let mut frontier = BfsFrontier::new();
        let result = frontier::run_search(
            Config::new(initial.iter().copied()).unwrap(),
            &ExhaustiveGenerator::for_solve(&ctx, SeedPolicy::Any, None).unwrap(),
            &DistanceScorer,
            &UniformCost,
            &goal,
            &mut frontier,
            &ctx,
            &mut SearchState::default(),
            &mut NoOpObserver,
            Some(budget),
            None,
            None,
        );
        result.goal.map(|id| result.graph.g_score(id))
    }

    fn h0_at(
        initial: &[(u32, LocationAddr)],
        targets: &[(u32, u64)],
        blocked: &HashSet<u64>,
        index: &LaneIndex,
    ) -> f64 {
        WeightedDistanceBound::new(&UniformCost, targets, index, blocked)
            .estimate(&Config::new(initial.iter().copied()).unwrap())
    }

    /// Tier 1, and the only tier that does not depend on the generator: for a
    /// lone atom the optimum is **exactly** `h0`.
    ///
    /// Lower bound: `h0 = wdist` by the admissibility argument. Upper bound: a
    /// plan of that cost exists — walk the shortest path one lane per shot,
    /// each shot a single-lane moveset, each costing 1 under `UniformCost`. So
    /// equality must hold, which pins the weights and the graph rather than
    /// merely bounding them. A 1x1 rectangle is always enumerated, so the
    /// generator's triplet-at-a-time limitation cannot affect this case.
    #[test]
    fn h0_is_exact_for_a_single_atom() {
        let index = make_index();
        let blocked = HashSet::new();
        let cases = [
            (loc(0, 0), loc(0, 5)),
            (loc(0, 0), loc(1, 5)),
            (loc(0, 0), loc(1, 0)),
            (loc(0, 5), loc(1, 5)),
        ];
        for (start, target) in cases {
            let initial = [(0u32, start)];
            let targets = [(0u32, target.encode())];
            let h0 = h0_at(&initial, &targets, &blocked, &index);
            let optimum = brute_force_optimum(&initial, &targets, &blocked, &index, 20_000)
                .unwrap_or_else(|| panic!("{start:?} -> {target:?} should be solvable"));
            assert_eq!(
                h0, optimum,
                "h0 must equal the optimum for a lone atom: {start:?} -> {target:?}"
            );
        }
    }

    /// `(initial placement, target placement, blocked locations)`.
    type Instance = (
        Vec<(u32, LocationAddr)>,
        Vec<(u32, LocationAddr)>,
        Vec<LocationAddr>,
    );

    /// The hand-picked instances on the example arch, ordered by atom count.
    fn tiny_instances() -> Vec<Instance> {
        vec![
            // 1 atom
            (vec![(0, loc(0, 0))], vec![(0, loc(0, 5))], vec![]),
            (vec![(0, loc(0, 0))], vec![(0, loc(1, 0))], vec![]),
            // 1 atom, with an obstacle carved out of the graph
            (vec![(0, loc(0, 0))], vec![(0, loc(1, 0))], vec![loc(0, 6)]),
            // 2 atoms, parallel-friendly
            (
                vec![(0, loc(0, 0)), (1, loc(0, 1))],
                vec![(0, loc(0, 5)), (1, loc(0, 6))],
                vec![],
            ),
            // 2 atoms, crossing words
            (
                vec![(0, loc(0, 0)), (1, loc(0, 5))],
                vec![(0, loc(0, 5)), (1, loc(1, 5))],
                vec![],
            ),
            // 3 atoms
            (
                vec![(0, loc(0, 0)), (1, loc(0, 1)), (2, loc(0, 2))],
                vec![(0, loc(0, 5)), (1, loc(0, 6)), (2, loc(0, 7))],
                vec![],
            ),
        ]
    }

    /// Tier 2: hand-picked instances, growing from one atom to three, each
    /// verified against the brute-force reference.
    #[test]
    fn h0_never_exceeds_brute_force_optimum_on_tiny_instances() {
        let index = make_index();
        let instances = tiny_instances();

        let mut verified = 0_usize;
        for (initial, target_pairs, blocked_locs) in instances {
            let targets: Vec<(u32, u64)> =
                target_pairs.iter().map(|&(q, l)| (q, l.encode())).collect();
            let blocked: HashSet<u64> = blocked_locs.iter().map(|l| l.encode()).collect();
            let h0 = h0_at(&initial, &targets, &blocked, &index);
            let Some(optimum) = brute_force_optimum(&initial, &targets, &blocked, &index, 60_000)
            else {
                continue; // budget exhausted: no reference to compare against
            };
            assert!(
                h0 <= optimum,
                "h0 {h0} exceeded the brute-force optimum {optimum} for {initial:?} -> {target_pairs:?}"
            );
            verified += 1;
        }
        assert!(
            verified >= 4,
            "only {verified} instances produced a reference optimum; the ladder verified too little"
        );
    }

    /// Tier 3: randomized instances, still small, with a fixed seed so any
    /// failure reproduces.
    ///
    /// Targets are drawn from each atom's **reachable** set rather than
    /// uniformly: the fixture's lane graph is sparse, and sampling blindly
    /// produced instances that were almost all unreachable, so the bound
    /// short-circuited to `+inf` and the comparison never ran. The coverage
    /// assertion at the end is what caught that — without it this test would
    /// have passed while verifying one instance out of twenty-four.
    #[test]
    fn h0_never_exceeds_brute_force_optimum_on_randomized_instances() {
        let index = make_index();
        let mut verified = 0_usize;
        let mut skipped_budget = 0_usize;

        for (initial, targets, blocked) in random_instances(&index, 0xB01E_5EED, 8, false) {
            let h0 = h0_at(&initial, &targets, &blocked, &index);
            assert!(
                h0.is_finite(),
                "targets were sampled from reachable sets, so h0 must be finite: \
                 {initial:?} -> {targets:?}"
            );
            let Some(optimum) = brute_force_optimum(&initial, &targets, &blocked, &index, 60_000)
            else {
                skipped_budget += 1;
                continue;
            };
            assert!(
                h0 <= optimum,
                "h0 {h0} exceeded optimum {optimum} for {initial:?} -> {targets:?}"
            );
            verified += 1;
        }
        // 22 of 24 verify today (2 exceed the reference budget), and the run is
        // deterministic, so this threshold catches a real collapse in coverage
        // rather than only a total one.
        assert!(
            verified >= 18,
            "only {verified} randomized instances were verified \
             ({skipped_budget} skipped on budget); the test is not exercising the bound"
        );
    }

    // ── BusCutBound ──

    /// One word of sites on a line and one zone whose site buses are the given
    /// `(src, dst)` site lists.
    fn site_bus_arch_json(n_sites: u32, buses: &[(Vec<u32>, Vec<u32>)]) -> String {
        let join = |v: &[u32]| v.iter().map(u32::to_string).collect::<Vec<_>>().join(", ");
        let sites = (0..n_sites)
            .map(|i| format!("[{i}, 0]"))
            .collect::<Vec<_>>();
        let spacing = vec!["2.0"; n_sites as usize - 1].join(", ");
        let buses = buses
            .iter()
            .map(|(src, dst)| format!(r#"{{ "src": [{}], "dst": [{}] }}"#, join(src), join(dst)))
            .collect::<Vec<_>>();
        format!(
            r#"{{
                "version": "2.0",
                "words": [ {{ "sites": [{sites}] }} ],
                "zones": [
                    {{
                        "grid": {{ "x_start": 0.0, "y_start": 0.0, "x_spacing": [{spacing}], "y_spacing": [] }},
                        "site_buses": [{buses}],
                        "word_buses": [],
                        "words_with_site_buses": [0],
                        "sites_with_word_buses": [],
                        "entangling_pairs": []
                    }}
                ],
                "zone_buses": [],
                "modes": [ {{ "name": "default", "zones": [0], "bitstring_order": [] }} ]
            }}"#,
            sites = sites.join(", "),
            buses = buses.join(", "),
        )
    }

    /// Five sites with two independent site buses: a chain `0→1→2` (bus 0) and a
    /// lone hop `3→4` (bus 1). The components share no lane and no group, so
    /// atoms on them can never share a shot.
    fn two_component_arch_json() -> String {
        site_bus_arch_json(5, &[(vec![0, 1], vec![1, 2]), (vec![3], vec![4])])
    }

    fn index_for(json: &str) -> LaneIndex {
        let spec: ArchSpec = serde_json::from_str(json).expect("fixture json parses");
        let validation = spec.validate();
        assert!(
            validation.is_ok(),
            "fixture must be a legal spec: {validation:?}"
        );
        LaneIndex::new(spec)
    }

    fn bus_cut_at(
        initial: &[(u32, LocationAddr)],
        targets: &[(u32, u64)],
        blocked: &HashSet<u64>,
        index: &LaneIndex,
    ) -> f64 {
        BusCutBound::new(&UniformCost, targets, index, blocked)
            .estimate(&Config::new(initial.iter().copied()).unwrap())
    }

    /// `(initial, targets, blocked)` in the shapes the bounds and the
    /// brute-force reference take.
    type BoundInstance = (Vec<(u32, LocationAddr)>, Vec<(u32, u64)>, HashSet<u64>);

    fn encode_instance((initial, targets, blocked): Instance) -> BoundInstance {
        (
            initial,
            targets.iter().map(|&(q, l)| (q, l.encode())).collect(),
            blocked.iter().map(|l| l.encode()).collect(),
        )
    }

    /// Every location a lane of `index` touches, in encoded order.
    fn arch_locations(index: &LaneIndex) -> Vec<LocationAddr> {
        let mut by_enc = std::collections::BTreeMap::new();
        for (mt, bus_id, zone_id, dir) in index.bus_groups() {
            for &lane in index.lanes_for(mt, bus_id, zone_id, dir) {
                if let Some((src, dst)) = index.endpoints(&lane) {
                    by_enc.insert(src.encode(), src);
                    by_enc.insert(dst.encode(), dst);
                }
            }
        }
        by_enc.into_values().collect()
    }

    /// Small randomized instances on any architecture, seeded so a failure
    /// reproduces: one to three atoms, each with a distinct target drawn from
    /// its reachable set. With `block_one`, one further location is blocked.
    ///
    fn random_instances(
        index: &LaneIndex,
        seed: u64,
        per_size: usize,
        block_one: bool,
    ) -> Vec<BoundInstance> {
        use crate::primitives::distance::DistanceTable;
        use rand::SeedableRng;
        use rand::rngs::SmallRng;
        use rand::seq::{IndexedRandom, SliceRandom};

        let sites = arch_locations(index);
        let all_encs: Vec<u64> = sites.iter().map(|l| l.encode()).collect();
        let reach = DistanceTable::new(&all_encs, index);
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut out = Vec::new();

        for n_atoms in 1..=3usize.min(sites.len()) {
            for _ in 0..per_size {
                let mut picked = sites.clone();
                picked.shuffle(&mut rng);
                let starts: Vec<LocationAddr> = picked[..n_atoms].to_vec();

                let mut targets_used: Vec<LocationAddr> = Vec::new();
                let mut ok = true;
                for &start in &starts {
                    let options: Vec<LocationAddr> = sites
                        .iter()
                        .copied()
                        .filter(|&to| {
                            to != start && reach.distance(start.encode(), to.encode()).is_some()
                        })
                        .filter(|t| !targets_used.contains(t))
                        .collect();
                    match options.choose(&mut rng) {
                        Some(&t) => targets_used.push(t),
                        None => {
                            ok = false;
                            break;
                        }
                    }
                }
                if !ok {
                    continue;
                }

                let mut blocked = HashSet::new();
                if block_one {
                    let free: Vec<LocationAddr> = sites
                        .iter()
                        .copied()
                        .filter(|l| !starts.contains(l) && !targets_used.contains(l))
                        .collect();
                    if let Some(&b) = free.choose(&mut rng) {
                        blocked.insert(b.encode());
                    }
                }
                out.push((
                    starts
                        .iter()
                        .enumerate()
                        .map(|(i, &l)| (i as u32, l))
                        .collect(),
                    targets_used
                        .iter()
                        .enumerate()
                        .map(|(i, &l)| (i as u32, l.encode()))
                        .collect(),
                    blocked,
                ));
            }
        }
        out
    }

    /// Every architecture the exhaustive generator can enumerate, as
    /// `(name, index, instances)`. The example arch carries the same instances
    /// as the h0 tests; the others get seeded random ones, half of them with a
    /// blocked site.
    fn bus_cut_fixtures() -> Vec<(&'static str, LaneIndex, Vec<BoundInstance>)> {
        use crate::test_utils::{
            asymmetric_duration_arch_json, chain_arch_json, chain_with_siding_arch_json,
            storage_gate_arch_json, two_zone_aligned_site_bus_arch_json, two_zone_bus_arch_json,
        };

        let example = make_index();
        let mut example_instances: Vec<BoundInstance> =
            tiny_instances().into_iter().map(encode_instance).collect();
        example_instances.extend(random_instances(&example, 0xB01E_5EED, 8, false));
        example_instances.extend(random_instances(&example, 0xB10C_CED5, 4, true));

        let mut fixtures = vec![("example", example, example_instances)];
        let others: [(&'static str, LaneIndex); 7] = [
            ("two-component", index_for(&two_component_arch_json())),
            ("chain", index_for(&chain_arch_json())),
            (
                "chain with siding",
                index_for(chain_with_siding_arch_json()),
            ),
            (
                "two-zone aligned site buses",
                index_for(two_zone_aligned_site_bus_arch_json()),
            ),
            ("two-zone bus", index_for(two_zone_bus_arch_json())),
            ("storage and gate", index_for(storage_gate_arch_json())),
            (
                "asymmetric duration",
                index_for(&asymmetric_duration_arch_json()),
            ),
        ];
        for (i, (name, index)) in others.into_iter().enumerate() {
            let seed = 0xB05C_0000 + i as u64;
            let mut instances = random_instances(&index, seed, 6, false);
            instances.extend(random_instances(&index, seed ^ 0xFF, 6, true));
            fixtures.push((name, index, instances));
        }
        fixtures
    }

    /// Walk the exhaustive generator's graph breadth-first from `initial`, over
    /// at most `node_cap` expanded configurations, and assert
    /// `h(parent) <= 1 + h(child)` on every edge. Returns the edges checked.
    fn assert_consistent_on_edges(
        initial: &[(u32, LocationAddr)],
        targets: &[(u32, u64)],
        blocked: &HashSet<u64>,
        index: &LaneIndex,
        bound: &impl CompletionBound,
        node_cap: usize,
    ) -> usize {
        use crate::generators::exhaustive::{ExhaustiveGenerator, SeedPolicy};
        use crate::primitives::context::{SearchContext, SearchState};
        use crate::primitives::distance::DistanceTable;
        use crate::primitives::graph::NodeId;
        use crate::traits::MoveGenerator;

        let target_locs: Vec<u64> = targets.iter().map(|&(_, l)| l).collect();
        let dist_table = DistanceTable::new(&target_locs, index);
        let ctx = SearchContext {
            index,
            dist_table: &dist_table,
            blocked,
            targets,
            cz_pairs: None,
        };
        let generator = ExhaustiveGenerator::for_solve(&ctx, SeedPolicy::Any, None).unwrap();

        let root = Config::new(initial.iter().copied()).unwrap();
        let mut seen: HashSet<Config> = HashSet::from([root.clone()]);
        let mut queue = VecDeque::from([root]);
        let mut state = SearchState::default();
        let mut candidates = Vec::new();
        let (mut expanded, mut checked) = (0_usize, 0_usize);

        while let Some(config) = queue.pop_front() {
            if expanded >= node_cap {
                break;
            }
            expanded += 1;
            let h = bound.estimate(&config);
            candidates.clear();
            generator.generate(&config, NodeId(0), &ctx, &mut state, &mut candidates);
            for candidate in candidates.drain(..) {
                let h_child = bound.estimate(&candidate.new_config);
                assert!(
                    h <= 1.0 + h_child,
                    "h is inconsistent across one shot: h(parent) = {h}, h(child) = {h_child}; \
                     parent {config:?} -> child {:?}",
                    candidate.new_config
                );
                checked += 1;
                if seen.insert(candidate.new_config.clone()) {
                    queue.push_back(candidate.new_config);
                }
            }
        }
        checked
    }

    // Compile-time: the bound must stay a real (non-trivial) one, and is only
    // ever paired with the unit-cost objective.
    const _: () = assert!(!<BusCutBound as CompletionBound>::TRIVIAL);
    const _: fn() = || {
        fn same_objective<B: CompletionBound<Obj = UniformCost>>() {}
        same_objective::<BusCutBound>();
    };

    #[test]
    fn bus_cut_pairs_with_the_uniform_cost_objective() {
        let index = make_index();
        let bound = BusCutBound::new(&UniformCost, &[], &index, &HashSet::new());
        assert_eq!(bound.objective_id(), UniformCost.id());
    }

    /// The premises the bound counts shots on: a shot costs 1 whatever it
    /// carries and every lane weighs 1 (C1-C3), so `|R|` distinct shots cost
    /// `|R|`. `assert_objective_contract` covers C2-C5 on each architecture.
    #[test]
    fn bus_cut_premises_hold_under_uniform_cost() {
        for index in [make_index(), index_for(&two_component_arch_json())] {
            assert_objective_contract(&UniformCost, &index);
            assert_eq!(UniformCost.min_shot_cost(), 1.0);
            let config = Config::new([]).unwrap();
            for (mt, bus_id, zone_id, dir) in index.bus_groups() {
                let lanes = index.lanes_for(mt, bus_id, zone_id, dir);
                for &lane in lanes {
                    assert_eq!(UniformCost.lane_weight(lane), 1.0);
                }
                let shot = MoveSet::new(lanes.iter().copied());
                assert_eq!(UniformCost.edge_cost(&shot, &config, &config), 1.0);
            }
        }
    }

    #[test]
    fn bus_cut_is_zero_when_every_atom_is_resolved() {
        let initial = [(0u32, loc(0, 5)), (1, loc(1, 5))];
        let targets = [(0u32, loc(0, 5).encode()), (1, loc(1, 5).encode())];
        let h = bus_cut_at(&initial, &targets, &HashSet::new(), &make_index());
        assert_eq!(h, 0.0);
    }

    #[test]
    fn bus_cut_is_infinite_exactly_when_h0_is() {
        let index = make_index();
        let initial = [(0u32, loc(0, 0))];

        let unreachable = [(0u32, loc(99, 99).encode())];
        let missing = [(7u32, loc(0, 5).encode())];
        for targets in [&unreachable[..], &missing[..]] {
            let h = bus_cut_at(&initial, targets, &HashSet::new(), &index);
            assert_eq!(h, f64::INFINITY);
        }

        // A blocked target is unreachable for an unblocked atom, as for h0.
        let target = loc(1, 0).encode();
        let h = bus_cut_at(&initial, &[(0, target)], &HashSet::from([target]), &index);
        assert_eq!(h, f64::INFINITY);
    }

    /// An atom parked on a blocked site certifies no floor, exactly as in h0.
    #[test]
    fn bus_cut_ignores_an_atom_parked_on_a_blocked_site() {
        let index = make_index();
        let initial = [(0u32, loc(0, 0))];
        let targets = [(0u32, loc(1, 0).encode())];
        let blocked = HashSet::from([loc(0, 0).encode()]);
        let h = bus_cut_at(&initial, &targets, &blocked, &index);
        assert_eq!(h, 0.0);
        assert_eq!(
            h,
            h0_at(&initial, &targets, &blocked, &index),
            "must agree with h0"
        );
    }

    /// Sites of atoms that are part of the solve are never carved: an atom
    /// already on its target, sitting mid-chain, must not sever another
    /// atom's only route.
    #[test]
    fn bus_cut_does_not_carve_the_sites_of_atoms_in_the_solve() {
        let index = index_for(&two_component_arch_json());
        let initial = [(0u32, loc(0, 0)), (1, loc(0, 1))];
        let targets = [(0u32, loc(0, 2).encode()), (1, loc(0, 1).encode())];
        assert_eq!(bus_cut_at(&initial, &targets, &HashSet::new(), &index), 2.0);
    }

    /// `|R| > h0`: the atoms need different buses, so no single shot serves
    /// both, however short each atom's own trip is.
    #[test]
    fn bus_cut_exceeds_h0_when_atoms_depend_on_different_buses() {
        let index = index_for(&two_component_arch_json());
        let blocked = HashSet::new();
        let initial = [(0u32, loc(0, 0)), (1, loc(0, 3))];
        let targets = [(0u32, loc(0, 1).encode()), (1, loc(0, 4).encode())];

        let h0 = h0_at(&initial, &targets, &blocked, &index);
        let h = bus_cut_at(&initial, &targets, &blocked, &index);
        assert_eq!(h0, 1.0);
        assert_eq!(h, 2.0, "each hop is on its own bus, so two shots at least");
        assert_eq!(
            brute_force_optimum(&initial, &targets, &blocked, &index, 20_000),
            Some(2.0)
        );
    }

    /// The same effect on the example arch, across three groups: one atom per
    /// leg of site-bus-forward, word-bus-forward, site-bus-backward.
    #[test]
    fn bus_cut_counts_three_required_groups_on_the_example_arch() {
        let index = make_index();
        let blocked = HashSet::new();
        let initial = [(0u32, loc(0, 0)), (1, loc(0, 5)), (2, loc(1, 5))];
        let targets = [
            (0u32, loc(0, 5).encode()),
            (1, loc(1, 5).encode()),
            (2, loc(1, 0).encode()),
        ];

        let h0 = h0_at(&initial, &targets, &blocked, &index);
        let h = bus_cut_at(&initial, &targets, &blocked, &index);
        assert_eq!(h0, 1.0);
        assert_eq!(h, 3.0);
        assert_eq!(
            brute_force_optimum(&initial, &targets, &blocked, &index, 60_000),
            Some(3.0)
        );
    }

    /// For a lone atom `R` is contained in the groups of any shortest path, so
    /// `|R| <= d` and the bound must collapse to `h0`, which is exact.
    #[test]
    fn bus_cut_is_exact_for_a_single_atom() {
        let blocked = HashSet::new();
        let example = make_index();
        let two_component = index_for(&two_component_arch_json());
        let cases = [
            (&example, loc(0, 0), loc(0, 5)),
            (&example, loc(0, 0), loc(1, 5)),
            (&example, loc(0, 0), loc(1, 0)),
            (&example, loc(0, 5), loc(1, 5)),
            (&two_component, loc(0, 0), loc(0, 1)),
            (&two_component, loc(0, 0), loc(0, 2)),
            (&two_component, loc(0, 3), loc(0, 4)),
        ];
        for (index, start, target) in cases {
            let initial = [(0u32, start)];
            let targets = [(0u32, target.encode())];
            let h = bus_cut_at(&initial, &targets, &blocked, index);
            let optimum = brute_force_optimum(&initial, &targets, &blocked, index, 20_000)
                .unwrap_or_else(|| panic!("{start:?} -> {target:?} should be solvable"));
            assert_eq!(
                h, optimum,
                "bus-cut must equal the optimum for a lone atom: {start:?} -> {target:?}"
            );
            assert_eq!(h, h0_at(&initial, &targets, &blocked, index));
        }
    }

    /// Sandwiched between h0 and the brute-force optimum on every fixture.
    /// As in the h0 tests, the reference is an upper bound on the optimum, so
    /// this is a necessary check, not a proof.
    #[test]
    fn bus_cut_sits_between_h0_and_the_brute_force_optimum_on_every_fixture() {
        let (mut verified, mut strictly_above_h0) = (0_usize, 0_usize);
        for (name, index, instances) in bus_cut_fixtures() {
            let mut verified_here = 0_usize;
            for (initial, targets, blocked) in instances {
                let h0 = h0_at(&initial, &targets, &blocked, &index);
                let h = bus_cut_at(&initial, &targets, &blocked, &index);
                assert!(
                    h >= h0,
                    "[{name}] bus-cut {h} fell below h0 {h0} for {initial:?} -> {targets:?} \
                     (blocked {blocked:?})"
                );
                let Some(optimum) =
                    brute_force_optimum(&initial, &targets, &blocked, &index, 30_000)
                else {
                    continue; // budget exhausted: no reference to compare against
                };
                assert!(
                    h <= optimum,
                    "[{name}] bus-cut {h} exceeded the brute-force optimum {optimum} for \
                     {initial:?} -> {targets:?} (blocked {blocked:?})"
                );
                verified_here += 1;
                if h > h0 {
                    strictly_above_h0 += 1;
                }
            }
            assert!(
                verified_here >= 3,
                "[{name}] only {verified_here} instances were verified; the fixture is not \
                 exercising the bound"
            );
            verified += verified_here;
        }
        assert!(
            strictly_above_h0 >= 1,
            "bus-cut never beat h0 on {verified} verified instances; the strict-gain \
             cases are not being reached"
        );
    }

    /// `h(n) <= 1 + h(n')` on every edge of the exhaustive graph.
    #[test]
    fn bus_cut_is_consistent_on_every_exhaustive_edge() {
        let mut checked = 0_usize;
        for (_, index, instances) in bus_cut_fixtures() {
            for (initial, targets, blocked) in instances {
                let bound = BusCutBound::new(&UniformCost, &targets, &index, &blocked);
                checked +=
                    assert_consistent_on_edges(&initial, &targets, &blocked, &index, &bound, 120);
            }
        }
        assert!(
            checked >= 1_000,
            "only {checked} edges were checked; the consistency test is not exercising the bound"
        );
    }

    /// 130 independent one-hop buses are 260 groups: five `u64` mask words,
    /// one more than the inline buffer holds, so this takes the heap path and
    /// every word index beyond the first. One atom per bus needs one shot per
    /// bus, because no two buses share a group.
    #[test]
    fn bus_cut_counts_more_groups_than_one_mask_word_and_the_inline_buffer() {
        let n = 130_u32;
        let buses: Vec<(Vec<u32>, Vec<u32>)> =
            (0..n).map(|k| (vec![2 * k], vec![2 * k + 1])).collect();
        let json = site_bus_arch_json(2 * n, &buses);
        let index = index_for(&json);

        let initial: Vec<(u32, LocationAddr)> = (0..n).map(|k| (k, loc(0, 2 * k))).collect();
        let targets: Vec<(u32, u64)> = (0..n).map(|k| (k, loc(0, 2 * k + 1).encode())).collect();
        let blocked = HashSet::new();

        let bound = BusCutBound::new(&UniformCost, &targets, &index, &blocked);
        assert_eq!(bound.words, 5);
        assert!(bound.words > INLINE_WORDS);
        let config = Config::new(initial.iter().copied()).unwrap();
        assert_eq!(bound.estimate(&config), f64::from(n));
        assert_eq!(h0_at(&initial, &targets, &blocked, &index), 1.0);
    }
}
