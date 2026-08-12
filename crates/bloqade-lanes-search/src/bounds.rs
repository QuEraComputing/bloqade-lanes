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

use std::collections::HashSet;
use std::marker::PhantomData;

use crate::primitives::config::Config;
use crate::primitives::lane_index::LaneIndex;
use crate::primitives::weighted_distance::WeightedDistanceTable;
use crate::traits::{Heuristic, Objective, ObjectiveId};

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
    /// Cost of the best solution found, or [`f64::NAN`] if none was.
    pub incumbent_cost: f64,
    /// Whether a non-trivial bound was in use.
    ///
    /// Without it, an unbounded run is indistinguishable from a bounded run
    /// that pruned nothing: both report zero cuts, and `root_lower_bound` of
    /// `0.0` would yield a spurious "gap" of 1.0 that reads like a measurement
    /// rather than an absence of one.
    pub bound_enabled: bool,
}

impl BoundStats {
    /// Total cuts made, however classified.
    pub fn total_cuts(&self) -> u64 {
        self.cuts_by_g + self.cuts_by_h + self.cuts_infeasible
    }

    /// Certified optimality gap `(incumbent - h(root)) / incumbent`, or `None`
    /// when no bound was in use, no solution was found, or the incumbent is
    /// zero.
    ///
    /// `0.0` means the incumbent is provably optimal.
    pub fn optimality_gap(&self) -> Option<f64> {
        if !self.bound_enabled || !self.incumbent_cost.is_finite() || self.incumbent_cost <= 0.0 {
            return None;
        }
        Some(((self.incumbent_cost - self.root_lower_bound) / self.incumbent_cost).max(0.0))
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

    /// View this bound as a plain [`Heuristic`], for the frontier drivers.
    ///
    /// The returned closure is `Copy`, which the `Heuristic + Copy` call sites
    /// in strategy dispatch require. Note that passing a bound here *loses* the
    /// objective pairing — the frontier's `f = g + weight * h` may scale it for
    /// ordering. That is sound for weighted A\* (bounded suboptimal) but must
    /// never feed an incumbent prune.
    fn as_heuristic(&self) -> impl Heuristic + Copy + '_
    where
        Self: Sized,
    {
        move |config: &Config| self.estimate(config)
    }
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

/// Combine two bounds by taking the larger estimate.
///
/// The max of two admissible bounds is admissible — each is individually a
/// floor on the same quantity — and is at least as tight as either. This is how
/// later bounds (a zone-bus cut bound, a per-triplet phase decomposition) get
/// added without any driver change.
///
/// Fields are private and construction goes through [`MaxBound::new`]: public
/// tuple fields would let a caller skip the objective-instance check and
/// reintroduce, one level up, exactly the mismatch this module exists to
/// prevent. The type parameters guarantee the two bounds name the same
/// objective *type*; only the check in `new` covers the *instance*.
pub struct MaxBound<A, B> {
    a: A,
    b: B,
}

impl<A, B> MaxBound<A, B>
where
    A: CompletionBound,
    B: CompletionBound<Obj = A::Obj>,
{
    /// # Panics
    ///
    /// If the two bounds were built against different objective instances.
    pub fn new(a: A, b: B) -> Self {
        assert_eq!(
            a.objective_id(),
            b.objective_id(),
            "composed bounds must target the same objective instance"
        );
        Self { a, b }
    }
}

impl<A, B> CompletionBound for MaxBound<A, B>
where
    A: CompletionBound,
    B: CompletionBound<Obj = A::Obj>,
{
    type Obj = A::Obj;

    fn objective_id(&self) -> ObjectiveId {
        // Equal to `b`'s by the assert in `new`.
        self.a.objective_id()
    }

    fn estimate(&self, config: &Config) -> f64 {
        self.a.estimate(config).max(self.b.estimate(config))
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
                return f64::INFINITY; // no unblocked route exists
            };
            worst = worst.max(d);
        }
        worst
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

/// Assert the [`Objective`] contract (C2, C3, C4) for every lane and a
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
            for lane in shot.decode() {
                let w = objective.lane_weight(lane);
                assert!(
                    cost >= w - slack(w),
                    "C3: shot cost {cost} is below lane weight {w} for {lane:?}"
                );
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
    use crate::test_utils::{example_arch_json, loc};
    use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;
    use bloqade_lanes_bytecode_core::arch::types::ArchSpec;

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

    // ── Composition ──

    #[test]
    fn max_bound_takes_the_larger_estimate() {
        let index = make_index();
        let targets = [(0u32, loc(1, 0).encode())];
        let blocked = HashSet::new();
        let config = Config::new([(0, loc(0, 0))]).unwrap();

        let real = bound_for(&targets, &index, &blocked);
        let real_estimate = real.estimate(&config);
        let composed = MaxBound::new(real, NoBound::<UniformCost>::for_objective(&UniformCost));

        assert_eq!(composed.estimate(&config), real_estimate);
        assert_eq!(composed.objective_id(), UniformCost.id());
    }

    #[test]
    fn max_bound_composes_recursively() {
        let index = make_index();
        let targets = [(0u32, loc(1, 0).encode())];
        let blocked = HashSet::new();
        let config = Config::new([(0, loc(0, 0))]).unwrap();

        let expected = bound_for(&targets, &index, &blocked).estimate(&config);
        let composed = MaxBound::new(
            MaxBound::new(
                bound_for(&targets, &index, &blocked),
                NoBound::<UniformCost>::for_objective(&UniformCost),
            ),
            bound_for(&targets, &index, &blocked),
        );
        assert_eq!(composed.estimate(&config), expected);
    }

    /// Two bounds of the same type built against *different instances* of the
    /// same objective family must not compose. The type system cannot catch
    /// this — both are `WeightedDistanceBound<WeightedDuration>` — so the
    /// construction-time check is the only guard.
    #[test]
    #[should_panic(expected = "same objective instance")]
    fn max_bound_rejects_mismatched_objective_instances() {
        let index = make_index();
        let targets = [(0u32, loc(1, 0).encode())];
        let blocked = HashSet::new();

        let a = WeightedDistanceBound::new(
            &WeightedDuration::new(&index, 1.0),
            &targets,
            &index,
            &blocked,
        );
        let b = WeightedDistanceBound::new(
            &WeightedDuration::new(&index, 10.0),
            &targets,
            &index,
            &blocked,
        );
        let _ = MaxBound::new(a, b);
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
        use crate::generators::exhaustive::ExhaustiveGenerator;
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
            &ExhaustiveGenerator::new(None, None),
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

    /// Tier 2: hand-picked instances, growing from one atom to three, each
    /// verified against the brute-force reference.
    #[test]
    fn h0_never_exceeds_brute_force_optimum_on_tiny_instances() {
        let index = make_index();

        // (initial, targets, blocked) ordered by atom count.
        /// `(initial placement, target placement, blocked locations)`.
        type Instance = (
            Vec<(u32, LocationAddr)>,
            Vec<(u32, LocationAddr)>,
            Vec<LocationAddr>,
        );

        let instances: Vec<Instance> = vec![
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
        ];

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
        use crate::primitives::distance::DistanceTable;
        use rand::SeedableRng;
        use rand::rngs::SmallRng;
        use rand::seq::{IndexedRandom, SliceRandom};

        let index = make_index();
        let blocked = HashSet::new();

        let sites: Vec<LocationAddr> = (0..2u32)
            .flat_map(|word| (0..10u32).map(move |site| loc(word, site)))
            .collect();
        let all_encs: Vec<u64> = sites.iter().map(|l| l.encode()).collect();
        // Reachability is a property of the architecture, so one table answers
        // it for every instance below.
        let reach = DistanceTable::new(&all_encs, &index);
        let reachable_from = |from: LocationAddr| -> Vec<LocationAddr> {
            sites
                .iter()
                .copied()
                .filter(|&to| to != from && reach.distance(from.encode(), to.encode()).is_some())
                .collect()
        };

        let mut rng = SmallRng::seed_from_u64(0xB01E_5EED);
        let mut verified = 0_usize;
        let mut skipped_budget = 0_usize;

        for n_atoms in 1..=3usize {
            for _ in 0..8 {
                let mut picked = sites.clone();
                picked.shuffle(&mut rng);
                let starts: Vec<LocationAddr> = picked[..n_atoms].to_vec();

                // One reachable, distinct target per atom.
                let mut targets_used: Vec<LocationAddr> = Vec::new();
                let mut ok = true;
                for &start in &starts {
                    let options: Vec<LocationAddr> = reachable_from(start)
                        .into_iter()
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

                let initial: Vec<(u32, LocationAddr)> = starts
                    .iter()
                    .enumerate()
                    .map(|(i, &l)| (i as u32, l))
                    .collect();
                let targets: Vec<(u32, u64)> = targets_used
                    .iter()
                    .enumerate()
                    .map(|(i, &l)| (i as u32, l.encode()))
                    .collect();

                let h0 = h0_at(&initial, &targets, &blocked, &index);
                assert!(
                    h0.is_finite(),
                    "targets were sampled from reachable sets, so h0 must be finite: \
                     {initial:?} -> {targets:?}"
                );
                let Some(optimum) =
                    brute_force_optimum(&initial, &targets, &blocked, &index, 60_000)
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

    // ── Frontier interop ──

    /// The same bound object is consumable by the frontier drivers through the
    /// plain `Heuristic` trait, and the adapter is `Copy` as those call sites
    /// require.
    #[test]
    fn as_heuristic_matches_estimate_and_is_copy() {
        let index = make_index();
        let targets = [(0u32, loc(1, 0).encode())];
        let bound = bound_for(&targets, &index, &HashSet::new());
        let config = Config::new([(0, loc(0, 0))]).unwrap();

        let h = bound.as_heuristic();
        let h_copy = h; // requires Copy
        assert_eq!(h.estimate(&config), bound.estimate(&config));
        assert_eq!(h_copy.estimate(&config), bound.estimate(&config));
    }
}
