//! End-to-end validation of the Push and Rotate router.
//!
//! Two properties, checked with [`AtomStateData`] — the same simulator the IR
//! analysis pipeline uses — rather than a reimplementation of it, since a bug
//! in a hand-rolled replay could mask exactly the bug it exists to catch:
//!
//! 1. **The plan rearranges the atoms correctly.** Feed each scheduled AOD
//!    operation to [`AtomStateData::apply_moves`] and compare the resulting
//!    `qubit_to_locations` against the requested target.
//! 2. **Every operation is a valid AOD move.** Each batch's lane group is
//!    checked with `ArchSpec::check_lanes`, the same authority the open
//!    solver's generators are held to.
//!
//! `AtomStateData` earns its place here by failing in ways a naive replay
//! would not notice:
//!
//! * A move onto an occupied site is recorded in `collision`, and **both**
//!   qubits are dropped from the location maps. Asserting `collision` is
//!   empty catches any operation that would crash two atoms together.
//! * `apply_moves` *silently skips* a lane whose source holds no qubit. A
//!   scheduler that emitted a lane for an atom that is not there would look
//!   fine on the final placement in some cases, so the total `move_count` is
//!   asserted against the number of lanes issued.

use std::collections::HashMap;
use std::collections::HashSet;

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;
use bloqade_lanes_bytecode_core::arch::types::ArchSpec;
use bloqade_lanes_bytecode_core::atom_state::AtomStateData;
use bloqade_lanes_search::feasibility::graph::{LaneGraph, VertexId};
use bloqade_lanes_search::primitives::lane_index::LaneIndex;
use bloqade_lanes_search::push_rotate::context::PlanCtx;
use bloqade_lanes_search::push_rotate::heuristics::{
    AlignmentHeuristics, DefaultHeuristics, PlanHeuristics,
};
use bloqade_lanes_search::push_rotate::instances::generate;
use bloqade_lanes_search::push_rotate::schedule::schedule;
use bloqade_lanes_search::push_rotate::state::PlanState;
use bloqade_lanes_search::push_rotate::{PlanError, plan, plan_with, solve_push_rotate};
use bloqade_lanes_search::search::result::SolveStatus;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

/// An `(initial, target)` pair of `(qubit, encoded location)` placements.
type Instance = (Vec<(u32, u64)>, Vec<(u32, u64)>);

const PHYSICAL: &str =
    include_str!("../../../python/bloqade/lanes/arch/gemini/physical/_physical_spec.json");
const LOGICAL: &str =
    include_str!("../../../python/bloqade/lanes/arch/gemini/logical/_logical_spec.json");

struct Fixture {
    spec: ArchSpec,
    index: LaneIndex,
    graph: LaneGraph,
}

fn fixture(arch_json: &str) -> Fixture {
    let spec: ArchSpec = serde_json::from_str(arch_json).expect("fixture parses");
    let index = LaneIndex::new(spec.clone());
    let graph = LaneGraph::build(&index, &Default::default());
    Fixture { spec, index, graph }
}

/// Plan, schedule, then validate both properties through `AtomStateData`.
///
/// `initial` and `target` are `(qubit, encoded location)`. Returns
/// `(operations, moves)`.
fn check_end_to_end(
    fx: &Fixture,
    initial: &[(u32, u64)],
    target: &[(u32, u64)],
    label: &str,
) -> (usize, usize) {
    let to_v = |pairs: &[(u32, u64)]| -> Vec<(u32, VertexId)> {
        pairs
            .iter()
            .map(|&(q, loc)| {
                (
                    q,
                    fx.graph.vertex_of(loc).expect("location is on the graph"),
                )
            })
            .collect()
    };

    let p = plan(&fx.index, &fx.graph, &to_v(initial), &to_v(target), 500_000)
        .unwrap_or_else(|e| panic!("{label}: planning failed: {e}"));
    let batches = schedule(&fx.index, &fx.graph, &p.moves)
        .unwrap_or_else(|| panic!("{label}: scheduling failed"));

    // ── Property 2: every operation is a legal AOD move ────────────
    for (bi, b) in batches.iter().enumerate() {
        let errors = fx.spec.check_lanes(&b.lanes);
        assert!(
            errors.is_empty(),
            "{label}: operation {bi} is not a valid AOD move: {errors:?}"
        );
        // A bus maps disjoint source and destination sets, so simultaneous
        // execution is only meaningful if no destination is also a source.
        let srcs: HashSet<VertexId> = b.moves.iter().map(|m| m.from).collect();
        for m in &b.moves {
            assert!(
                !srcs.contains(&m.to),
                "{label}: operation {bi} moves into {} while another move leaves it",
                m.to
            );
        }
    }

    // ── Property 1: the atoms end up where they were asked to ──────
    let start: Vec<(u32, LocationAddr)> = initial
        .iter()
        .map(|&(q, loc)| (q, LocationAddr::decode(loc)))
        .collect();
    let mut state = AtomStateData::from_locations(&start);

    let mut lanes_issued = 0usize;
    for (bi, b) in batches.iter().enumerate() {
        lanes_issued += b.lanes.len();
        state = state
            .apply_moves(&b.lanes, &fx.spec)
            .unwrap_or_else(|| panic!("{label}: operation {bi} has an unresolvable lane"));
        assert!(
            state.collision.is_empty(),
            "{label}: operation {bi} collided atoms: {:?}",
            state.collision
        );
    }

    // Every lane must have actually moved an atom. `apply_moves` skips a lane
    // whose source is empty, which would otherwise pass unnoticed.
    let moved: u32 = state.move_count.values().sum();
    assert_eq!(
        moved as usize, lanes_issued,
        "{label}: {lanes_issued} lanes issued but only {moved} atom moves took effect"
    );

    let want: HashMap<u32, LocationAddr> = target
        .iter()
        .map(|&(q, loc)| (q, LocationAddr::decode(loc)))
        .collect();
    assert_eq!(
        state.qubit_to_locations, want,
        "{label}: final placement does not match the requested target"
    );

    (batches.len(), p.moves.len())
}

/// Instance from the reachable-by-construction generator: atoms are displaced
/// to wherever a random walk of legal single-atom slides leaves them.
fn walk_instance(fx: &Fixture, arch: &'static str, k: usize, seed: u64) -> Instance {
    let inst = generate("t".into(), arch, k, 4 * k, seed).expect("instance");
    for (_, loc) in inst.initial.iter().chain(inst.target.iter()) {
        assert!(fx.graph.vertex_of(*loc).is_some());
    }
    (inst.initial, inst.target)
}

/// A pure **permutation**: the occupied sites are unchanged, the atoms are
/// shuffled among them.
///
/// This is the harder and more representative case. A displacement can often
/// be routed by pushing atoms into free space, but a permutation forces the
/// planner to exchange atoms in place — which is exactly what `swap` and
/// `rotate` exist for, and what CZ placement asks for when two qubits trade
/// entangling slots.
fn permutation_instance(fx: &Fixture, arch: &'static str, k: usize, seed: u64) -> Instance {
    let inst = generate("t".into(), arch, k, 0, seed).expect("instance");
    let sites: Vec<u64> = inst.initial.iter().map(|&(_, l)| l).collect();
    let mut shuffled = sites.clone();
    let mut rng = SmallRng::seed_from_u64(seed ^ 0xA5A5);
    // Reshuffle until at least one atom actually has to move, so the test is
    // not silently trivial.
    for _ in 0..32 {
        shuffled.shuffle(&mut rng);
        if shuffled.iter().zip(&sites).any(|(a, b)| a != b) {
            break;
        }
    }
    let target: Vec<(u32, u64)> = shuffled
        .into_iter()
        .enumerate()
        .map(|(q, l)| (q as u32, l))
        .collect();
    let _ = (fx, &mut rng);
    (inst.initial, target)
}

// ── Displacement instances ─────────────────────────────────────────

#[test]
fn rearranges_correctly_on_physical() {
    let fx = fixture(PHYSICAL);
    for k in [1usize, 2, 4, 8, 16] {
        for seed in 0..5u64 {
            let (i, t) = walk_instance(&fx, PHYSICAL, k, seed);
            check_end_to_end(&fx, &i, &t, &format!("physical/walk/k{k}/seed{seed}"));
        }
    }
}

#[test]
fn rearranges_correctly_on_logical() {
    let fx = fixture(LOGICAL);
    for k in [1usize, 2, 4, 8, 16] {
        for seed in 0..5u64 {
            let (i, t) = walk_instance(&fx, LOGICAL, k, seed);
            check_end_to_end(&fx, &i, &t, &format!("logical/walk/k{k}/seed{seed}"));
        }
    }
}

// ── Permutation instances ──────────────────────────────────────────

#[test]
fn rearranges_permutations_on_physical() {
    let fx = fixture(PHYSICAL);
    for k in [2usize, 4, 8, 16] {
        for seed in 0..5u64 {
            let (i, t) = permutation_instance(&fx, PHYSICAL, k, seed);
            check_end_to_end(&fx, &i, &t, &format!("physical/perm/k{k}/seed{seed}"));
        }
    }
}

#[test]
fn rearranges_permutations_on_logical() {
    let fx = fixture(LOGICAL);
    for k in [2usize, 4, 8, 16] {
        for seed in 0..5u64 {
            let (i, t) = permutation_instance(&fx, LOGICAL, k, seed);
            check_end_to_end(&fx, &i, &t, &format!("logical/perm/k{k}/seed{seed}"));
        }
    }
}

/// A two-atom transposition — the smallest thing that cannot be solved by
/// pushing alone and must go through `swap`.
#[test]
fn swaps_two_adjacent_atoms() {
    let fx = fixture(PHYSICAL);
    let mut rng = SmallRng::seed_from_u64(7);
    let mut found = 0;
    for _ in 0..200 {
        let v = rng.random_range(0..fx.graph.len());
        let nbrs = fx.graph.neighbors(v);
        if nbrs.is_empty() {
            continue;
        }
        let w = nbrs[rng.random_range(0..nbrs.len())];
        let (a, b) = (fx.graph.location_of(v), fx.graph.location_of(w));
        check_end_to_end(&fx, &[(0, a), (1, b)], &[(0, b), (1, a)], "transposition");
        found += 1;
        if found == 10 {
            break;
        }
    }
    assert_eq!(found, 10, "expected 10 adjacent pairs to test");
}

/// A move request that leaves an atom exactly where it is must produce a plan
/// that touches nothing.
#[test]
fn identity_target_produces_no_operations() {
    let fx = fixture(PHYSICAL);
    let (initial, _) = walk_instance(&fx, PHYSICAL, 8, 0);
    let (ops, moves) = check_end_to_end(&fx, &initial, &initial, "identity");
    assert_eq!((ops, moves), (0, 0), "identity should require no moves");
}

/// Guard the failure path too: a fully packed graph where an atom must
/// move is outside Push and Rotate's regime and must be reported
/// distinctly, not as unsolvable — zero empties does not prove the
/// instance impossible.
#[test]
fn packed_graph_is_reported_as_out_of_regime() {
    let fx = fixture(LOGICAL);
    let all: Vec<(u32, VertexId)> = fx
        .graph
        .vertices()
        .enumerate()
        .map(|(i, v)| (i as u32, v))
        .collect();

    // Swap the targets of two adjacent atoms so movement is required.
    let v0 = all[0].1;
    let &neighbor = fx
        .graph
        .neighbors(v0)
        .first()
        .expect("vertex 0 has a lane neighbour");
    let mut target = all.clone();
    let j = target
        .iter()
        .position(|&(_, v)| v == neighbor)
        .expect("the neighbour is occupied on a packed graph");
    target[0].1 = neighbor;
    target[j].1 = v0;

    let err = plan(&fx.index, &fx.graph, &all, &target, 10_000).expect_err("should refuse");
    assert!(matches!(err, PlanError::TooFewEmpty { .. }), "got {err:?}");
}

/// The regime gate is per component *and* per moving atom: an identity
/// target on a fully packed graph needs nothing moved, so it must succeed
/// with an empty plan rather than being refused.
#[test]
fn packed_graph_with_identity_target_is_a_trivial_success() {
    let fx = fixture(LOGICAL);
    let all: Vec<(u32, VertexId)> = fx
        .graph
        .vertices()
        .enumerate()
        .map(|(i, v)| (i as u32, v))
        .collect();
    let plan = plan(&fx.index, &fx.graph, &all, &all, 10_000).expect("identity is solvable");
    assert!(plan.moves.is_empty(), "identity should require no moves");
}

// ── Verdict semantics ──────────────────────────────────────────────
//
// `SolveStatus::Unsolvable` from this router is documented as a *proof*, and
// the fallback path promotes it over the search's verdict on that basis.
// These tests pin the two directions: proofs stay proofs, and everything
// that is not a proof must not claim to be one.

/// A target for a qubit absent from `initial` is unsatisfiable — no move
/// sequence creates an atom that does not exist. It must be reported as
/// `Unsolvable` (a genuine proof), never as `Solved` with a fabricated
/// placement.
#[test]
fn phantom_target_is_unsolvable_not_a_fabricated_success() {
    let fx = fixture(LOGICAL);
    let loc_a = LocationAddr::decode(fx.graph.location_of(0));
    let loc_b = LocationAddr::decode(fx.graph.location_of(1));

    let result = solve_push_rotate(&fx.index, &[(0, loc_a)], &[(7, loc_b)], &[], 10_000)
        .expect("valid config");
    assert_eq!(result.status, SolveStatus::Unsolvable);
}

/// A nearly-packed register asking for a single legal slide into the hole is
/// trivially solvable — but with one empty vertex it is outside Push and
/// Rotate's completeness regime, so the honest verdict is `BudgetExceeded`
/// ("gave up"), never `Unsolvable` ("proof").
#[test]
fn out_of_regime_reports_budget_exceeded_not_unsolvable() {
    let fx = fixture(LOGICAL);
    let n = fx.graph.len();
    // Occupy every vertex except the last; ask the hole's neighbour to
    // slide in.
    let hole = n - 1;
    let initial: Vec<(u32, LocationAddr)> = (0..n - 1)
        .map(|v| (v as u32, LocationAddr::decode(fx.graph.location_of(v))))
        .collect();
    let &mover = fx
        .graph
        .neighbors(hole)
        .first()
        .expect("the hole has a lane neighbour");
    let target = [(
        mover as u32,
        LocationAddr::decode(fx.graph.location_of(hole)),
    )];

    let result =
        solve_push_rotate(&fx.index, &initial, &target, &[], 10_000).expect("valid config");
    assert_eq!(
        result.status,
        SolveStatus::BudgetExceeded,
        "a solvable out-of-regime instance must not be reported as proven unsolvable"
    );
}

/// Spectators encoded as blocked locations — the primary production wiring —
/// are routed around: the solve succeeds and no emitted lane touches a
/// blocked location.
#[test]
fn blocked_spectators_are_routed_around() {
    let fx = fixture(LOGICAL);
    // Block the three highest-numbered vertices, then plan a move between
    // two vertices that remain connected in the carved graph.
    let n = fx.graph.len();
    let blocked_locs: Vec<LocationAddr> = (n - 3..n)
        .map(|v| LocationAddr::decode(fx.graph.location_of(v)))
        .collect();
    let blocked_enc: HashSet<u64> = blocked_locs.iter().map(|l| l.encode()).collect();
    let carved = LaneGraph::build(&fx.index, &blocked_enc);
    assert!(!carved.is_empty(), "carving must leave a graph");

    // Pick a start with a neighbour and route to a vertex reachable from it.
    let start = carved
        .vertices()
        .find(|&v| carved.degree(v) > 0)
        .expect("carved graph has edges");
    let dist = carved.distances_from(start, |_| false);
    let dest = carved
        .vertices()
        .filter(|&v| dist[v] != u32::MAX && v != start)
        .max_by_key(|&v| dist[v])
        .expect("something is reachable");

    let initial = [(0u32, LocationAddr::decode(carved.location_of(start)))];
    let target = [(0u32, LocationAddr::decode(carved.location_of(dest)))];
    let result = solve_push_rotate(&fx.index, &initial, &target, &blocked_locs, 10_000)
        .expect("valid config");
    assert_eq!(result.status, SolveStatus::Solved);

    for layer in &result.move_layers {
        for lane in layer.decode() {
            let (src, dst) = fx.index.endpoints(&lane).expect("lane resolves");
            assert!(
                !blocked_enc.contains(&src.encode()) && !blocked_enc.contains(&dst.encode()),
                "emitted lane touches a blocked location"
            );
        }
    }
}

/// Verdicts agree with exhaustive search on small carved instances.
///
/// The graph is carved down to ≤ 8 vertices with a blocked set (often
/// disconnecting it), atoms and targets are random, and a configuration-space
/// BFS is the ground truth. Two directions:
///
/// * **Soundness — hard assertions.** `Unsolvable` must imply the oracle
///   finds no solution, and `Solved` must imply it does. A violation of
///   either is a bug, full stop.
/// * **Completeness (solving) — asserted.** In-regime and oracle-solvable
///   must mean `Solved`; currently clean across all seeds.
///
/// One Theorem 1 obligation is deliberately *not* asserted: proving
/// unsolvability of every in-regime unsolvable instance. The planner's
/// proof checks are containment-form (weaker than the paper's `f = f'`,
/// which is unsound under our conservative `assign_agents`), so some
/// unsolvable instances return an honest `BudgetExceeded` instead of a
/// proof. One-sidedness is preserved — the fallback then simply keeps the
/// search's own verdict.
#[test]
fn verdicts_match_brute_force_on_carved_instances() {
    let fx = fixture(LOGICAL);
    let n_full = fx.graph.len();
    let keep = 8.min(n_full);

    let mut proofs = 0usize;
    let mut completeness_misses = 0usize;
    for seed in 0..120u64 {
        let mut rng = SmallRng::seed_from_u64(seed);

        // Carve: keep a random set of `keep` vertices, block the rest.
        let mut verts: Vec<VertexId> = (0..n_full).collect();
        verts.shuffle(&mut rng);
        let blocked_locs: Vec<LocationAddr> = verts[keep..]
            .iter()
            .map(|&v| LocationAddr::decode(fx.graph.location_of(v)))
            .collect();
        let blocked_enc: HashSet<u64> = blocked_locs.iter().map(|l| l.encode()).collect();
        let carved = LaneGraph::build(&fx.index, &blocked_enc);
        let n = carved.len();
        if n < 4 {
            continue;
        }

        // Random occupancy (1..=n-2 atoms) and random distinct targets.
        let mut order: Vec<VertexId> = carved.vertices().collect();
        order.shuffle(&mut rng);
        let n_atoms = rng.random_range(1..=n - 2);
        let initial_v: Vec<(u32, VertexId)> = order[..n_atoms]
            .iter()
            .enumerate()
            .map(|(q, &v)| (q as u32, v))
            .collect();
        let mut goal_order: Vec<VertexId> = carved.vertices().collect();
        goal_order.shuffle(&mut rng);
        let target_v: Vec<(u32, VertexId)> = goal_order[..n_atoms]
            .iter()
            .enumerate()
            .map(|(q, &v)| (q as u32, v))
            .collect();

        let to_locs = |pairs: &[(u32, VertexId)]| -> Vec<(u32, LocationAddr)> {
            pairs
                .iter()
                .map(|&(q, v)| (q, LocationAddr::decode(carved.location_of(v))))
                .collect()
        };
        let result = solve_push_rotate(
            &fx.index,
            &to_locs(&initial_v),
            &to_locs(&target_v),
            &blocked_locs,
            50_000,
        )
        .expect("valid config");

        let solvable = oracle_solvable(&carved, &initial_v, &target_v);
        match result.status {
            SolveStatus::Unsolvable => {
                proofs += 1;
                assert!(
                    !solvable,
                    "seed {seed}: Unsolvable claimed for an oracle-solvable instance \
                     (n={n}, initial={initial_v:?}, target={target_v:?})"
                );
            }
            SolveStatus::Solved => {}
            SolveStatus::BudgetExceeded => {
                // In-regime give-ups are the known completeness gap —
                // counted and bounded, not asserted away (see the test doc).
                if in_regime(&carved, &initial_v, &target_v) && solvable {
                    completeness_misses += 1;
                }
            }
        }
        if result.status != SolveStatus::Solved {
            continue;
        }
        assert!(
            solvable,
            "seed {seed}: Solved returned for an oracle-unsolvable instance"
        );
    }
    assert!(proofs > 0, "no seed exercised the proof path");
    assert_eq!(
        completeness_misses, 0,
        "{completeness_misses} in-regime oracle-solvable instances returned \
         BudgetExceeded — the planner must solve these (Theorem 1)"
    );
}

/// Ground truth: BFS over the configuration space.
fn oracle_solvable(
    graph: &LaneGraph,
    initial: &[(u32, VertexId)],
    target: &[(u32, VertexId)],
) -> bool {
    use std::collections::VecDeque;
    let mut occupant: Vec<Option<u32>> = vec![None; graph.len()];
    for &(q, v) in initial {
        occupant[v] = Some(q);
    }
    let goals: HashMap<u32, VertexId> = target.iter().copied().collect();
    let satisfied = |state: &[Option<u32>]| goals.iter().all(|(&q, &goal)| state[goal] == Some(q));

    if satisfied(&occupant) {
        return true;
    }
    let mut seen: HashSet<Vec<Option<u32>>> = HashSet::new();
    let mut queue: VecDeque<Vec<Option<u32>>> = VecDeque::new();
    seen.insert(occupant.clone());
    queue.push_back(occupant);
    while let Some(state) = queue.pop_front() {
        for v in graph.vertices() {
            let Some(q) = state[v] else { continue };
            for &w in graph.neighbors(v) {
                if state[w].is_some() {
                    continue;
                }
                let mut next = state.clone();
                next[v] = None;
                next[w] = Some(q);
                if satisfied(&next) {
                    return true;
                }
                if seen.insert(next.clone()) {
                    queue.push_back(next);
                }
            }
        }
    }
    false
}

/// In Push and Rotate's completeness regime: every component where an atom
/// must move has at least two empties.
fn in_regime(graph: &LaneGraph, initial: &[(u32, VertexId)], target: &[(u32, VertexId)]) -> bool {
    let (component, n_components) = graph.connected_components();
    let occupied: HashSet<VertexId> = initial.iter().map(|&(_, v)| v).collect();
    let mut empties = vec![0usize; n_components];
    for v in graph.vertices() {
        if !occupied.contains(&v) {
            empties[component[v]] += 1;
        }
    }
    let start: HashMap<u32, VertexId> = initial.iter().copied().collect();
    target.iter().all(|&(q, goal)| match start.get(&q) {
        Some(&s) if s != goal => empties[component[s]] >= 2,
        _ => true,
    })
}

// ── The heuristic seam ─────────────────────────────────────────────

/// A heuristic that inverts every default preference: farthest-first for
/// clear targets and swap vertices, reversed agent order, and a step score
/// that prefers *staying in the same bus group* as the previous edge.
///
/// It exists to prove the seam is real — that overriding these methods
/// actually changes the plan — and that correctness does not depend on the
/// choices being sensible. A heuristic can only reorder equally-good options,
/// so even a deliberately perverse one must still produce a valid plan.
struct ContraryHeuristics;

impl PlanHeuristics for ContraryHeuristics {
    fn agent_order(&self, ctx: &PlanCtx, state: &PlanState) -> Vec<u32> {
        let mut order = DefaultHeuristics.agent_order(ctx, state);
        order.reverse();
        order
    }

    fn rank_clear_target(
        &self,
        _ctx: &PlanCtx,
        _state: &PlanState,
        _clearing: VertexId,
        candidate: VertexId,
        _hops: u32,
    ) -> i64 {
        // Prefer the highest vertex id among the equally-near candidates.
        -(candidate as i64)
    }

    fn rank_swap_vertex(
        &self,
        _ctx: &PlanCtx,
        _state: &PlanState,
        _r: u32,
        _s: u32,
        candidate: VertexId,
        hops: u32,
    ) -> i64 {
        // Still distance-ordered — a swap vertex must stay reachable — but
        // inverted within a distance tier.
        (hops as i64) * 1000 - (candidate as i64)
    }

    fn score_step(
        &self,
        ctx: &PlanCtx,
        _state: &PlanState,
        _agent: u32,
        from: VertexId,
        to: VertexId,
    ) -> f64 {
        // Prefer word-bus steps, the group carrying most of the parallelism
        // on this architecture.
        match ctx.edge(from, to) {
            Some(info) if info.group.0 == 1 => 1.0,
            _ => 0.0,
        }
    }
}

#[test]
fn a_custom_heuristic_changes_the_plan_but_not_its_validity() {
    let fx = fixture(PHYSICAL);
    let (initial, target) = walk_instance(&fx, PHYSICAL, 8, 3);
    let to_v = |pairs: &[(u32, u64)]| -> Vec<(u32, VertexId)> {
        pairs
            .iter()
            .map(|&(q, loc)| (q, fx.graph.vertex_of(loc).expect("on graph")))
            .collect()
    };
    let (iv, tv) = (to_v(&initial), to_v(&target));

    let base = plan(&fx.index, &fx.graph, &iv, &tv, 500_000).expect("default plans");
    let contrary = plan_with(&fx.index, &fx.graph, &iv, &tv, 500_000, &ContraryHeuristics)
        .expect("contrary heuristics still plan");

    assert_ne!(
        base.moves, contrary.moves,
        "overriding every decision point produced an identical plan — the seam is not wired"
    );

    // Both must still be correct. Validity is not the heuristic's job.
    for (label, p) in [("default", &base), ("contrary", &contrary)] {
        let batches = schedule(&fx.index, &fx.graph, &p.moves).expect("schedules");
        let start: Vec<(u32, LocationAddr)> = initial
            .iter()
            .map(|&(q, loc)| (q, LocationAddr::decode(loc)))
            .collect();
        let mut state = AtomStateData::from_locations(&start);
        for b in &batches {
            assert!(
                fx.spec.check_lanes(&b.lanes).is_empty(),
                "{label}: bad batch"
            );
            state = state.apply_moves(&b.lanes, &fx.spec).expect("applies");
            assert!(state.collision.is_empty(), "{label}: collision");
        }
        let want: HashMap<u32, LocationAddr> = target
            .iter()
            .map(|&(q, loc)| (q, LocationAddr::decode(loc)))
            .collect();
        assert_eq!(state.qubit_to_locations, want, "{label}: wrong placement");
    }
}

/// The default strategy must be exactly the trait defaults, since every
/// benchmark number is measured against it.
#[test]
fn default_heuristics_match_the_plain_entry_point() {
    let fx = fixture(PHYSICAL);
    let (initial, target) = walk_instance(&fx, PHYSICAL, 8, 1);
    let to_v = |pairs: &[(u32, u64)]| -> Vec<(u32, VertexId)> {
        pairs
            .iter()
            .map(|&(q, loc)| (q, fx.graph.vertex_of(loc).expect("on graph")))
            .collect()
    };
    let (iv, tv) = (to_v(&initial), to_v(&target));

    let a = plan(&fx.index, &fx.graph, &iv, &tv, 500_000).expect("plans");
    let b = plan_with(&fx.index, &fx.graph, &iv, &tv, 500_000, &DefaultHeuristics).expect("plans");
    assert_eq!(a.moves, b.moves);
}

/// The alignment heuristic ships, so it gets the same correctness treatment as
/// the default: valid AOD operations, no collisions, correct final placement.
#[test]
fn alignment_heuristics_produce_valid_plans() {
    let fx = fixture(PHYSICAL);
    for k in [2usize, 8, 16] {
        for seed in 0..3u64 {
            let (initial, target) = walk_instance(&fx, PHYSICAL, k, seed);
            let to_v = |pairs: &[(u32, u64)]| -> Vec<(u32, VertexId)> {
                pairs
                    .iter()
                    .map(|&(q, loc)| (q, fx.graph.vertex_of(loc).expect("on graph")))
                    .collect()
            };
            let p = plan_with(
                &fx.index,
                &fx.graph,
                &to_v(&initial),
                &to_v(&target),
                500_000,
                &AlignmentHeuristics::default(),
            )
            .unwrap_or_else(|e| panic!("k={k} seed={seed}: {e}"));

            let batches = schedule(&fx.index, &fx.graph, &p.moves).expect("schedules");
            let start: Vec<(u32, LocationAddr)> = initial
                .iter()
                .map(|&(q, loc)| (q, LocationAddr::decode(loc)))
                .collect();
            let mut state = AtomStateData::from_locations(&start);
            for b in &batches {
                assert!(fx.spec.check_lanes(&b.lanes).is_empty(), "k={k}: bad batch");
                state = state.apply_moves(&b.lanes, &fx.spec).expect("applies");
                assert!(state.collision.is_empty(), "k={k}: collision");
            }
            let want: HashMap<u32, LocationAddr> = target
                .iter()
                .map(|&(q, loc)| (q, LocationAddr::decode(loc)))
                .collect();
            assert_eq!(state.qubit_to_locations, want, "k={k} seed={seed}");
        }
    }
}

// ── Solver-level surface ───────────────────────────────────────────

mod solver_surface {
    use std::sync::Arc;

    use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;
    use bloqade_lanes_search::search::engine::SearchEngine;
    use bloqade_lanes_search::search::move_search::MoveSearch;
    use bloqade_lanes_search::search::options::{SolveOptions, Strategy};
    use bloqade_lanes_search::search::result::SolveStatus;
    use bloqade_lanes_search::search::target_solver::TargetSolver;

    use super::PHYSICAL;

    fn engine() -> Arc<SearchEngine> {
        Arc::new(SearchEngine::from_json(PHYSICAL).expect("spec parses"))
    }

    fn at(word_id: u32, site_id: u32) -> LocationAddr {
        LocationAddr {
            zone_id: 0,
            word_id,
            site_id,
        }
    }

    fn solver(opts: SolveOptions) -> TargetSolver {
        TargetSolver::new(engine(), MoveSearch::default().with_options(opts))
    }

    /// `Strategy::PushRotate` must be selectable and return a normal
    /// `SolveResult`, so it can be benchmarked next to the search strategies.
    #[test]
    fn push_rotate_is_selectable_as_a_strategy() {
        let s = solver(SolveOptions {
            strategy: Strategy::PushRotate,
            ..Default::default()
        });
        let r = s
            .solve([(0, at(0, 0))], [(0, at(0, 4))], [], None)
            .expect("valid placement");

        assert_eq!(r.status, SolveStatus::Solved);
        assert!(!r.move_layers.is_empty());
        assert_eq!(
            r.goal_config.location_of(0),
            Some(at(0, 4)),
            "goal config must report the requested target"
        );
        // Not a search: there is no frontier to expand.
        assert_eq!(r.nodes_expanded, 0);
    }

    /// The fallback is off unless asked for. Every existing caller must be
    /// unaffected, which is what lets this land without moving the committed
    /// benchmark baselines.
    #[test]
    fn fallback_is_off_by_default() {
        assert!(!SolveOptions::default().fallback_push_rotate);
    }

    /// With the fallback on, a search that cannot finish inside its budget
    /// still yields a schedule.
    #[test]
    fn fallback_recovers_a_budget_starved_search() {
        let initial: Vec<(u32, LocationAddr)> = (0..8u32).map(|w| (w, at(w, 0))).collect();
        let target: Vec<(u32, LocationAddr)> = (0..8u32).map(|w| (w, at((w + 1) % 8, 0))).collect();

        // One expansion is not enough for any search to finish this.
        let starved = SolveOptions {
            strategy: Strategy::AStar,
            ..Default::default()
        };
        let without = solver(starved.clone())
            .solve(initial.clone(), target.clone(), [], Some(1))
            .expect("valid placement");
        assert_ne!(
            without.status,
            SolveStatus::Solved,
            "fixture must actually starve the search"
        );

        let with = solver(SolveOptions {
            fallback_push_rotate: true,
            ..starved
        })
        .solve(initial, target, [], Some(1))
        .expect("valid placement");
        assert_eq!(
            with.status,
            SolveStatus::Solved,
            "the fallback should have recovered this"
        );
        assert!(!with.move_layers.is_empty());
    }

    /// On double failure, the planner's verdict is promoted only when it is
    /// a genuine proof. A target for a qubit that does not exist is one: no
    /// move sequence creates an atom, so the budget-starved search's
    /// non-answer is upgraded to `Unsolvable`.
    #[test]
    fn fallback_promotes_a_genuine_proof_on_double_failure() {
        let initial = [(0, at(0, 0))];
        let target = [(7, at(0, 4))]; // qubit 7 is not in `initial`
        let result = solver(SolveOptions {
            fallback_push_rotate: true,
            ..Default::default()
        })
        .solve(initial, target, [], Some(1))
        .expect("valid placement");
        assert_eq!(result.status, SolveStatus::Unsolvable);
    }

    /// The fallback must not disturb a search that already succeeded.
    #[test]
    fn fallback_leaves_a_successful_search_alone() {
        let initial = [(0, at(0, 0))];
        let target = [(0, at(0, 4))];
        let base = solver(SolveOptions::default())
            .solve(initial, target, [], None)
            .expect("valid");
        let with = solver(SolveOptions {
            fallback_push_rotate: true,
            ..Default::default()
        })
        .solve(initial, target, [], None)
        .expect("valid");

        assert_eq!(base.status, SolveStatus::Solved);
        assert_eq!(with.status, SolveStatus::Solved);
        assert_eq!(
            base.move_layers.len(),
            with.move_layers.len(),
            "the fallback changed a result it should not have touched"
        );
    }
}
