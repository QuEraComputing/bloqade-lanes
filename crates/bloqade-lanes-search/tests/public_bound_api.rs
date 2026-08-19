//! Every public entry point of the objective/completion-bound surface, driven
//! from **outside** the crate.
//!
//! `MaxBound`, `WeightedDuration`, `CompletionBound::as_heuristic`,
//! `entropy_search_with_objective` and `entropy_search_with_bound` are all
//! exported from `lib.rs` but have no non-test caller anywhere in the
//! workspace. A unit test inside the crate cannot tell whether they are
//! *usable* from outside it — whether the types a caller must name are
//! reachable, whether the generic bounds can be satisfied with public types,
//! whether `MaxBound::new`'s objective-pairing assert can be passed. An
//! integration test compiles as an external consumer, so it answers exactly
//! that.
//!
//! # What is asserted, and what deliberately is not
//!
//! Each entry point must return a **valid** plan: a goal is found, every qubit
//! ends on its target, and replaying the emitted move layers from the root
//! reproduces the goal configuration through legal, simultaneous moves. The
//! replay is the real check — a driver can report a goal while emitting a layer
//! that moves an atom off an empty site, or two atoms onto one site, and only
//! replaying catches that.
//!
//! Optimality is **not** asserted. Cost claims belong with the admissibility
//! ladder in `bounds.rs`, which checks `h0` against brute-force optima. Pinning
//! plan length here would add no coverage of the thing under test — reachability
//! of the public API — and would break on any ordering change.

use std::collections::HashSet;

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;
use bloqade_lanes_bytecode_core::arch::types::ArchSpec;
use bloqade_lanes_search::bounds::CompletionBound;
use bloqade_lanes_search::drivers::entropy::{
    EntropyParams, entropy_search_with_bound, entropy_search_with_objective,
};
use bloqade_lanes_search::drivers::frontier::{PriorityFrontier, run_search};
use bloqade_lanes_search::goals::AllAtTarget;
use bloqade_lanes_search::observer::NoOpObserver;
use bloqade_lanes_search::primitives::config::Config;
use bloqade_lanes_search::primitives::context::{SearchContext, SearchState};
use bloqade_lanes_search::primitives::distance::DistanceTable;
use bloqade_lanes_search::primitives::graph::MoveSet;
use bloqade_lanes_search::primitives::lane_index::LaneIndex;
use bloqade_lanes_search::{
    DistanceScorer, HeuristicGenerator, MaxBound, NoBound, SearchResult, UniformCost,
    WeightedDistanceBound, WeightedDuration,
};

/// The bundled Gemini **logical** spec: 20 single-site words, 20 locations,
/// 110 lanes. Small (14 KB) but a real shipped spec rather than a hand-rolled
/// fixture — and, unlike `examples/arch/*.json`, it carries transport paths, so
/// lane durations exist and `WeightedDuration` has something to weigh.
///
/// Durations come from waypoint geometry: each path's segment lengths are summed
/// as a linear distance, which on this spec yields **6 distinct** values from
/// ~325 µs to ~816 µs. That spread is what makes the weighted objective below a
/// genuine test rather than uniform cost under another name.
const LOGICAL_ARCH_JSON: &str =
    include_str!("../../../python/bloqade/lanes/arch/gemini/logical/_logical_spec.json");

fn loc(word: u32, site: u32) -> LocationAddr {
    LocationAddr {
        zone_id: 0,
        word_id: word,
        site_id: site,
    }
}

fn make_index() -> LaneIndex {
    let spec: ArchSpec = serde_json::from_str(LOGICAL_ARCH_JSON).expect("logical arch json parses");
    LaneIndex::new(spec)
}

/// `(initial placement, target placement)` for one routing instance.
type Instance = (Vec<(u32, LocationAddr)>, Vec<(u32, LocationAddr)>);

/// Two atoms that each have to cross the zone. Every word here holds one site,
/// and `word 0 -> word 3` is reachable in two hops, so this needs real routing
/// without being large enough to be slow.
fn instance() -> Instance {
    (
        vec![(0, loc(0, 0)), (1, loc(1, 0))],
        vec![(0, loc(3, 0)), (1, loc(4, 0))],
    )
}

/// Replay `layers` from `root` and return the configuration they produce.
///
/// A lane's `(src, dst)` endpoints name the site vacated and the site taken, so
/// the mover is whichever qubit currently sits at `src`. Every move in a layer
/// happens at once, which is why a destination held by an atom that is *also*
/// leaving in the same layer is legal while one held by a stationary atom is not.
///
/// Panics naming the offending layer if the plan is not executable. This is the
/// crate-private `search::verify` check restated over the public API, which an
/// external caller has no other way to reach.
fn replay(root: &Config, layers: &[MoveSet], index: &LaneIndex) -> Config {
    let mut config = root.clone();
    for (layer_idx, layer) in layers.iter().enumerate() {
        let mut moves: Vec<(u32, LocationAddr)> = Vec::new();
        let mut sources: HashSet<u64> = HashSet::new();
        let mut destinations: HashSet<u64> = HashSet::new();

        for lane in layer.decode() {
            let (src, dst) = index
                .endpoints(&lane)
                .unwrap_or_else(|| panic!("layer {layer_idx}: lane {lane:?} has no endpoints"));
            let qubit = config.qubit_at(src).unwrap_or_else(|| {
                panic!("layer {layer_idx}: lane {lane:?} moves an atom off empty site {src:?}")
            });
            assert!(
                sources.insert(src.encode()),
                "layer {layer_idx}: site {src:?} is vacated twice in one layer"
            );
            assert!(
                destinations.insert(dst.encode()),
                "layer {layer_idx}: site {dst:?} receives two atoms in one layer"
            );
            moves.push((qubit, dst));
        }
        assert!(!moves.is_empty(), "layer {layer_idx} moves nothing");

        for &(_, dst) in &moves {
            if let Some(sitting) = config.qubit_at(dst) {
                assert!(
                    sources.contains(&dst.encode()),
                    "layer {layer_idx}: atom {sitting} holds {dst:?} and does not move"
                );
            }
        }
        config = config.with_moves(&moves);
    }
    config
}

/// Bundle the per-solve inputs so each test body is one call plus one assert.
struct Fixture {
    index: LaneIndex,
    dist_table: DistanceTable,
    blocked: HashSet<u64>,
    targets: Vec<(u32, u64)>,
    goal: AllAtTarget,
    root: Config,
    target: Vec<(u32, LocationAddr)>,
}

impl Fixture {
    fn new(index: LaneIndex) -> Self {
        let (initial, target) = instance();
        let targets: Vec<(u32, u64)> = target.iter().map(|&(q, l)| (q, l.encode())).collect();
        let target_locs: Vec<u64> = targets.iter().map(|&(_, l)| l).collect();
        // `.with_time_distances` is a **precondition**, not a tuning option.
        // `w_t` is the entropy scorer's hops-versus-duration balance:
        // `blended_distance` returns `(1 - w_t) * hops + w_t * (time / fastest_lane)`,
        // so a `w_t > 0` solve is asking to be scored on duration as well as hop
        // count, and it needs a table that carries both. Given a hop-only table
        // the blend silently falls back to hop count — the solve still succeeds,
        // it just quietly ignores the balance it was configured with, which is
        // why the driver debug-asserts instead of leaving it to be noticed.
        //
        // `EntropyParams` defaults to `w_t > 0`, and the example specs carry no
        // durations, so the assert only has anything to check on a spec that
        // does — like this one.
        let dist_table = DistanceTable::new(&target_locs, &index).with_time_distances(&index);
        Self {
            goal: AllAtTarget::new(&targets),
            root: Config::new(initial).expect("valid root"),
            dist_table,
            blocked: HashSet::new(),
            targets,
            target,
            index,
        }
    }

    fn ctx(&self) -> SearchContext<'_> {
        SearchContext {
            index: &self.index,
            dist_table: &self.dist_table,
            blocked: &self.blocked,
            targets: &self.targets,
            cz_pairs: None,
        }
    }

    fn params() -> EntropyParams {
        EntropyParams::default()
    }

    /// Assert `result` is a valid plan for this instance; return its layer count.
    fn assert_valid_plan(&self, label: &str, result: &SearchResult) -> usize {
        let goal_id = result
            .goal
            .unwrap_or_else(|| panic!("{label}: no goal found"));
        let goal_config = result.graph.config(goal_id);

        for &(qubit, want) in &self.target {
            assert_eq!(
                goal_config.location_of(qubit),
                Some(want),
                "{label}: qubit {qubit} is not on its target"
            );
        }

        let layers = result
            .solution_path()
            .unwrap_or_else(|| panic!("{label}: reported a goal but produced no path"));
        assert!(
            !layers.is_empty(),
            "{label}: empty plan for a moved instance"
        );

        let replayed = replay(&self.root, &layers, &self.index);
        assert_eq!(
            &replayed, goal_config,
            "{label}: replaying the plan does not reproduce the reported goal"
        );
        layers.len()
    }
}

/// `entropy_search_with_objective` is reachable externally and solves with the
/// default `UniformCost`.
#[test]
fn entropy_search_with_objective_solves_under_uniform_cost() {
    let fx = Fixture::new(make_index());
    let result = entropy_search_with_objective(
        fx.root.clone(),
        &fx.goal,
        &Fixture::params(),
        &fx.ctx(),
        Some(2000),
        None,
        0,
        &mut NoOpObserver,
        &UniformCost,
    );
    fx.assert_valid_plan("uniform", &result);
}

/// The same entry point with `WeightedDuration`, the only other public
/// `Objective`, over a spec whose lane durations differ.
///
/// This is the objective-swappability the pluggable-`g` work exists for, run end
/// to end by an external caller: a different `g` must still yield an executable
/// plan, and `g` must reflect the weights rather than the layer count.
#[test]
fn entropy_search_with_objective_solves_under_weighted_duration() {
    let fx = Fixture::new(make_index());
    let objective = WeightedDuration::new(&fx.index, 100.0);
    let result = entropy_search_with_objective(
        fx.root.clone(),
        &fx.goal,
        &Fixture::params(),
        &fx.ctx(),
        Some(2000),
        None,
        0,
        &mut NoOpObserver,
        &objective,
    );
    let layers = fx.assert_valid_plan("weighted-duration", &result);

    // Every shot costs `1 + duration/tau` with a positive duration term, so `g`
    // must exceed the layer count — i.e. this really is a non-uniform objective
    // and not `UniformCost` reached under another name.
    let cost = result.graph.g_score(result.goal.expect("solved"));
    assert!(
        cost > layers as f64,
        "weighted g {cost} should exceed the {layers} layers it paid for"
    );
}

/// `entropy_search_with_bound` with each publicly constructible bound: the
/// trivial one, the real one, and the two composed.
#[test]
fn entropy_search_with_bound_solves_for_every_public_bound() {
    let fx = Fixture::new(make_index());
    let h0 = || WeightedDistanceBound::new(&UniformCost, &fx.targets, &fx.index, &fx.blocked);

    let unbounded = entropy_search_with_bound(
        fx.root.clone(),
        &fx.goal,
        &Fixture::params(),
        &fx.ctx(),
        Some(2000),
        None,
        0,
        &mut NoOpObserver,
        &UniformCost,
        &NoBound::for_objective(&UniformCost),
    );
    fx.assert_valid_plan("no-bound", &unbounded);
    assert!(
        !unbounded.bound_stats.bound_enabled,
        "NoBound must report itself as no measurement"
    );

    let bounded = entropy_search_with_bound(
        fx.root.clone(),
        &fx.goal,
        &Fixture::params(),
        &fx.ctx(),
        Some(2000),
        None,
        0,
        &mut NoOpObserver,
        &UniformCost,
        &h0(),
    );
    fx.assert_valid_plan("weighted-distance", &bounded);
    assert!(bounded.bound_stats.bound_enabled);

    // `MaxBound::new` asserts both children were built against the same
    // objective *instance*; passing that assert is part of what this covers.
    let composed = entropy_search_with_bound(
        fx.root.clone(),
        &fx.goal,
        &Fixture::params(),
        &fx.ctx(),
        Some(2000),
        None,
        0,
        &mut NoOpObserver,
        &UniformCost,
        &MaxBound::new(h0(), NoBound::for_objective(&UniformCost)),
    );
    fx.assert_valid_plan("max(h0, no-bound)", &composed);
    assert!(
        composed.bound_stats.bound_enabled,
        "composing a real bound with a trivial one is still a real bound"
    );
}

/// `MaxBound` must report the larger of its two children, from either position.
///
/// Composition is the documented way a second bound gets added "without any
/// driver change", so an external caller has to be able to build one and get
/// `max` — not "whichever child happens to be first".
#[test]
fn max_bound_reports_the_larger_child_from_either_side() {
    let fx = Fixture::new(make_index());
    let h0 = || WeightedDistanceBound::new(&UniformCost, &fx.targets, &fx.index, &fx.blocked);
    let real = h0().estimate(&fx.root);
    assert!(
        real > 0.0,
        "the instance must give the root a non-zero bound for this to discriminate"
    );

    let a_larger = MaxBound::new(h0(), NoBound::for_objective(&UniformCost));
    let b_larger = MaxBound::new(NoBound::for_objective(&UniformCost), h0());
    assert_eq!(a_larger.estimate(&fx.root), real, "larger child first");
    assert_eq!(b_larger.estimate(&fx.root), real, "larger child second");
}

/// `CompletionBound::as_heuristic` must yield something the frontier drivers
/// accept, and searching with it must still find a valid plan.
///
/// The adapter exists so a bound can be reused for *ordering*; its only
/// in-crate use is a unit test, so this is the first place the returned value
/// has to satisfy `run_search`'s `Heuristic + Copy` bound at a real call site.
#[test]
fn a_bound_used_as_a_frontier_heuristic_finds_a_valid_plan() {
    let fx = Fixture::new(make_index());
    let h0 = WeightedDistanceBound::new(&UniformCost, &fx.targets, &fx.index, &fx.blocked);

    let generator = HeuristicGenerator::new();
    let mut frontier = PriorityFrontier::astar(h0.as_heuristic(), 1.0);
    let mut state = SearchState::default();
    let result = run_search(
        fx.root.clone(),
        &generator,
        &DistanceScorer,
        &UniformCost,
        &fx.goal,
        &mut frontier,
        &fx.ctx(),
        &mut state,
        &mut NoOpObserver,
        Some(2000),
        None,
        None,
    );
    fx.assert_valid_plan("as_heuristic", &result);
}

/// A bound written by an *external* crate, returning a caller-chosen constant.
///
/// `CompletionBound` is public, so a downstream crate may implement it. That
/// makes the trait's documented contract — what each return value means — part
/// of the public API, and this is the only place it is exercised by an impl the
/// crate does not own.
struct ConstantBound {
    value: f64,
    objective_id: bloqade_lanes_search::traits::ObjectiveId,
}

impl ConstantBound {
    fn new(value: f64) -> Self {
        Self {
            value,
            objective_id: bloqade_lanes_search::traits::Objective::id(&UniformCost),
        }
    }
}

impl CompletionBound for ConstantBound {
    type Obj = UniformCost;

    fn objective_id(&self) -> bloqade_lanes_search::traits::ObjectiveId {
        self.objective_id
    }

    fn estimate(&self, _config: &Config) -> f64 {
        self.value
    }
}

/// Only `+∞` means "no completion exists". `-∞` is a sound-but-vacuous lower
/// bound — remaining cost is non-negative, so `-∞` never overestimates — and
/// must not be read as a proof of infeasibility.
///
/// The driver used to test `h.is_infinite()`, which is sign-agnostic, so a
/// downstream bound returning `-∞` had every branch pruned as infeasible and the
/// instance came back unsolved.
#[test]
fn only_positive_infinity_is_the_infeasibility_sentinel() {
    let fx = Fixture::new(make_index());

    let solve = |bound: &ConstantBound| {
        entropy_search_with_bound(
            fx.root.clone(),
            &fx.goal,
            &Fixture::params(),
            &fx.ctx(),
            Some(2000),
            None,
            0,
            &mut NoOpObserver,
            &UniformCost,
            bound,
        )
    };

    // `-inf`: no information, so the search proceeds and still solves.
    let vacuous = solve(&ConstantBound::new(f64::NEG_INFINITY));
    fx.assert_valid_plan("h = -inf", &vacuous);
    assert_eq!(
        vacuous.bound_stats.cuts_infeasible, 0,
        "-inf is not an infeasibility proof and must not be counted as one"
    );

    // `0.0`: the other sound-but-useless value, for contrast.
    let zero = solve(&ConstantBound::new(0.0));
    fx.assert_valid_plan("h = 0", &zero);
    assert_eq!(zero.bound_stats.cuts_infeasible, 0);

    // `+inf`: the documented sentinel. Every branch is refused, so the search
    // proper expands nothing and records the proof.
    //
    // Note what is *not* asserted: that no goal comes back. A goal may still be
    // reported, because `budget_exhaustion_fallback` is deliberately not
    // bound-aware — the driver runs to its iteration cap and then hands off to
    // the fallback exactly as an unbounded run would, and the fallback can graft
    // a plan the bound had declared impossible. That combination (an infinite
    // root bound beside a finite incumbent) is why `optimality_gap` reports
    // `None` for a non-finite lower bound rather than clamping it to "provably
    // optimal".
    // `nodes_expanded` is deliberately not asserted either: the fallback's own
    // expansions are counted in it, so it reflects the fallback's work rather
    // than the bound's refusals.
    let infeasible = solve(&ConstantBound::new(f64::INFINITY));
    assert!(
        infeasible.bound_stats.cuts_infeasible > 0,
        "the infeasibility proof must be recorded"
    );
    assert_eq!(
        infeasible.bound_stats.optimality_gap(),
        None,
        "an infinite lower bound is a claim, not a measurement"
    );
}

/// A bound that counts how many times the driver asked it for an estimate.
struct CountingBound {
    calls: std::sync::atomic::AtomicUsize,
    objective_id: bloqade_lanes_search::traits::ObjectiveId,
}

impl CountingBound {
    fn new() -> Self {
        Self {
            calls: std::sync::atomic::AtomicUsize::new(0),
            objective_id: bloqade_lanes_search::traits::Objective::id(&UniformCost),
        }
    }

    fn calls(&self) -> usize {
        self.calls.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl CompletionBound for CountingBound {
    type Obj = UniformCost;

    fn objective_id(&self) -> bloqade_lanes_search::traits::ObjectiveId {
        self.objective_id
    }

    fn estimate(&self, _config: &Config) -> f64 {
        self.calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        0.0
    }
}

/// `h` is evaluated at most once per node.
///
/// The driver probes the bound up to three times per iteration — the resume
/// gate, the resume-buffer insert, and the child gate the next iteration's
/// resume gate then repeats — so without memoization the call count runs well
/// past the number of nodes. Bounding this by `graph.len()` states the cache's
/// contract without depending on how many iterations the search happens to take.
///
/// It holds because a node's configuration never changes after insertion, which
/// is what makes caching by `NodeId` exact in the first place.
#[test]
fn the_bound_is_evaluated_at_most_once_per_node() {
    let fx = Fixture::new(make_index());
    let bound = CountingBound::new();

    let result = entropy_search_with_bound(
        fx.root.clone(),
        &fx.goal,
        &Fixture::params(),
        &fx.ctx(),
        Some(2000),
        None,
        0,
        &mut NoOpObserver,
        &UniformCost,
        &bound,
    );

    let nodes = result.graph.len();
    let calls = bound.calls();
    assert!(calls > 0, "the driver should have consulted the bound");
    assert!(
        calls <= nodes,
        "h was evaluated {calls} times over {nodes} nodes; the per-node cache is not holding"
    );
}
