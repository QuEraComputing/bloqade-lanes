//! End-to-end router validation on a **one-dimensional conveyor**, where the
//! optimal plan length is known in closed form (issue #910).
//!
//! The fixture is a single row of sites joined by one site bus `0→1, 1→2, …`.
//! Its source and destination sets overlap, so it is the smallest architecture
//! on which conveyor chains are reachable at all — on the shipped Gemini specs
//! every bus keeps its endpoints disjoint and the chain paths are dead code.
//!
//! # Why the optimum is provable here
//!
//! Every lane on this bus moves an atom exactly one site, in one direction, and
//! `UniformCost` charges one unit per layer. So for any instance, the number of
//! AOD operations is **at least** the largest per-atom hop distance:
//!
//! ```text
//! layers >= max_i | target_i - start_i |
//! ```
//!
//! For a **rigid shift** of a packed block by `d` sites, every atom needs `d`
//! hops, so `d` is a lower bound — and it is achievable, because one AOD
//! operation can drive the whole block one site along (the block's cells form a
//! complete 1 × n rectangle on this bus, and each destination is vacated in the
//! same shot by the atom ahead). The optimum is therefore exactly `d`, and A*
//! with weight 1.0 over an admissible heuristic must return exactly `d` layers.
//!
//! Reaching that bound is a strictly stronger claim than `Solved`: it says the
//! chain assembled into **one** operation per shift rather than serializing into
//! several, which is the whole point of chain assembly.
//!
//! The two optimality tests exercise different layers, and it is worth knowing
//! which is which. When *every* atom in the block is targeted, scoring nominates
//! them all and the grid layer's repair closure (#887/#896) can assemble the shift
//! on its own — `rigid_shifts_are_solved_in_exactly_the_optimal_number_of_operations`
//! passes without any selection-time help. When some are not targeted,
//! `a_block_pushed_by_one_targeted_atom_is_optimal` is the case that needs
//! selection to close its own mover set over chains: without it the leader's
//! rectangle is unexecutable, the root has no successors at all, and the instance
//! comes back `Unsolvable` at any budget (#910).
//!
//! # The soundness half
//!
//! A 1D line also pins down what must be *unreachable*. Atoms on a path cannot
//! pass each other — no legal move sequence changes their left-to-right order —
//! so a cyclic permutation of atom identities across a fixed set of sites has no
//! solution. That is the exact shape a *rotation* would take, and a rotation is
//! the one thing the architecture cannot do (a bus is a set of edge transports,
//! never a permutation; `ArchSpec::validate` rejects cyclic buses for this
//! reason, #866/#874). So `router_never_fabricates_a_rotation` is the guard that
//! chain assembly did not quietly buy us a capability the hardware lacks.
//!
//! # The exhaustive half
//!
//! The hand-written families above pick particular hole layouts — a packed block
//! with all the free sites at one end, or a fixed site set being permuted — so on
//! their own they say nothing about instances with holes *inside* the atom set, or
//! about mixed compaction-and-shift targets.
//! [`router_matches_a_brute_force_optimum_on_every_1d_instance`] removes that
//! gap by enumerating **every** instance at several sizes and checking each
//! against a brute-force optimum computed by BFS over
//! [`ExhaustiveGenerator`](bloqade_lanes_search::generators::exhaustive::ExhaustiveGenerator)
//! — the generator that enumerates every rectangle the architecture admits, and
//! therefore the reference for what is reachable and in how few operations.
//! `fully_packed_line_admits_only_the_identity` covers the one regime the sweep
//! excludes, zero holes.
//!
//! Every `Solved` result here is also replayed through
//! `AtomStateData::validate_moves` + `apply_validated` inside
//! `solve_with_engine` before it is returned, so a plan the execution model
//! rejects panics rather than passing.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;
use bloqade_lanes_bytecode_core::arch::types::ArchSpec;
use bloqade_lanes_search::generators::exhaustive::ExhaustiveGenerator;
use bloqade_lanes_search::primitives::config::Config;
use bloqade_lanes_search::primitives::context::{SearchContext, SearchState};
use bloqade_lanes_search::primitives::distance::DistanceTable;
use bloqade_lanes_search::primitives::graph::SearchGraph;
use bloqade_lanes_search::primitives::lane_index::LaneIndex;
use bloqade_lanes_search::search::engine::SearchEngine;
use bloqade_lanes_search::search::move_search::MoveSearch;
use bloqade_lanes_search::search::result::SolveStatus;
use bloqade_lanes_search::search::target_solver::TargetSolver;
use bloqade_lanes_search::traits::MoveGenerator;

/// A one-dimensional architecture: one word of `n_sites` sites in a row, joined
/// by a single conveyor site bus `0→1, 1→2, …, (n-2)→(n-1)`.
fn line_arch_json(n_sites: u32) -> String {
    assert!(n_sites >= 2, "a conveyor needs at least one lane");
    let sites: Vec<String> = (0..n_sites).map(|i| format!("[{i}, 0]")).collect();
    let src: Vec<String> = (0..n_sites - 1).map(|i| i.to_string()).collect();
    let dst: Vec<String> = (1..n_sites).map(|i| i.to_string()).collect();
    let x_spacing: Vec<String> = (0..n_sites - 1).map(|_| "2.0".to_string()).collect();

    let json = format!(
        r#"{{
            "version": "2.0",
            "words": [ {{ "sites": [{sites}] }} ],
            "zones": [
                {{
                    "grid": {{
                        "x_start": 0.0,
                        "y_start": 0.0,
                        "x_spacing": [{x_spacing}],
                        "y_spacing": []
                    }},
                    "site_buses": [ {{ "src": [{src}], "dst": [{dst}] }} ],
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
        src = src.join(", "),
        dst = dst.join(", "),
        x_spacing = x_spacing.join(", "),
    );

    let spec: ArchSpec = serde_json::from_str(&json).expect("line arch parses");
    assert!(
        spec.validate().is_ok(),
        "line arch must be a legal spec: {:?}",
        spec.validate()
    );
    json
}

fn line_engine(n_sites: u32) -> Arc<SearchEngine> {
    Arc::new(SearchEngine::from_json(&line_arch_json(n_sites)).expect("engine builds"))
}

fn site(s: u32) -> LocationAddr {
    LocationAddr {
        zone_id: 0,
        word_id: 0,
        site_id: s,
    }
}

/// Expansion budget for the hand-written solvable instances.
///
/// Deliberately tight rather than generous. The optimal plan for every case
/// below is at most six expansions, so anything this side of a hundred is a
/// 100× margin — and the entropy driver keeps searching for further goal
/// candidates after its first (`EntropyOptions::max_goal_candidates`), so a
/// large budget costs seconds per solve to re-derive an answer it already had.
const SOLVABLE_BUDGET: u32 = 500;

/// Expansion budget for the exhaustive sweep, used for reachable and unreachable
/// targets alike.
///
/// Sized from measurement, not guesswork: across every instance the sweep visits,
/// the most any strategy needs to reach the optimum is **five** expansions
/// (entropy; A* and IDS peak at two), so this is a 20× margin. It doubles as the
/// cap on the unreachable instances, which do wander before draining — hence the
/// sweep asserts only `!= Solved` there, which `BudgetExceeded` satisfies just as
/// honestly as `Unsolvable`. Raising it buys nothing and costs real CI time: the
/// sweep runs thousands of solves, so its runtime scales with this number.
const SWEEP_BUDGET: u32 = 100;

/// Expansion budget for the unsolvable instances, which have to *exhaust* rather
/// than succeed. Every one of them actually drains its open list in fewer than
/// five expansions — the reachable space on a short line is a handful of
/// configurations — so this is three orders of magnitude of headroom, not a cap
/// the assertions lean on.
const UNSOLVABLE_BUDGET: u32 = 2_000;

/// Every search strategy that routes through the heuristic generator, plus the
/// entropy driver. Push and Rotate is excluded: it is a different router with
/// its own regime conditions (it needs two or more holes) and its own suite.
fn searches() -> Vec<(&'static str, MoveSearch)> {
    vec![
        ("astar", MoveSearch::astar(1.0)),
        ("ids", MoveSearch::ids()),
        ("entropy", MoveSearch::entropy()),
    ]
}

/// **Optimality.** A packed block of `n` atoms shifted `d` sites along the
/// conveyor must be solved in exactly `d` AOD operations — the provable optimum
/// — with every atom landing on its target.
///
/// The instances are packed on purpose: with no gap inside the block, no atom
/// can move without its neighbour moving in the same shot, so every layer here
/// *is* an assembled chain. Serializing a shift into per-atom moves would show
/// up immediately as more than `d` layers.
#[test]
fn rigid_shifts_are_solved_in_exactly_the_optimal_number_of_operations() {
    // (sites, atoms, shift): the block occupies 0..n and must land on d..n+d.
    let cases = [
        (4u32, 2u32, 1u32),
        (5, 3, 1),
        (5, 3, 2),
        (6, 4, 2),
        (8, 5, 3),
        (8, 2, 6),
        (10, 8, 2),
    ];

    for (n_sites, n_atoms, shift) in cases {
        assert!(n_atoms + shift <= n_sites, "case must fit on the line");
        let initial: Vec<(u32, LocationAddr)> = (0..n_atoms).map(|q| (q, site(q))).collect();
        let target: Vec<(u32, LocationAddr)> = (0..n_atoms).map(|q| (q, site(q + shift))).collect();
        let engine = line_engine(n_sites);

        for (name, search) in searches() {
            let solver = TargetSolver::new(Arc::clone(&engine), search);
            let result = solver
                .solve(
                    initial.iter().copied(),
                    target.iter().copied(),
                    std::iter::empty(),
                    Some(SOLVABLE_BUDGET),
                )
                .expect("instance is well formed");

            let label = format!("{name}: {n_atoms} atoms on {n_sites} sites, shift {shift}");
            assert_eq!(
                result.status,
                SolveStatus::Solved,
                "{label}: reported {:?} on a solvable rigid shift",
                result.status
            );
            for &(qid, want) in &target {
                assert_eq!(
                    result.goal_config.location_of(qid),
                    Some(want),
                    "{label}: q{qid} did not reach its target"
                );
            }
            assert_eq!(
                result.move_layers.len(),
                shift as usize,
                "{label}: expected the optimal {shift} operations, got {} \
                 (expanded {} nodes)",
                result.move_layers.len(),
                result.nodes_expanded
            );
            // The plan is not just optimal in length but *found* optimally: the
            // whole-row shift is the top-ranked candidate at every step, so the
            // walk is straight down with no backtracking.
            assert_eq!(
                result.nodes_expanded, shift,
                "{label}: expected a straight walk of {shift} expansions, got {}",
                result.nodes_expanded
            );
        }
    }
}

/// **Optimality where the chain is not nominated.** The same bound, on the
/// instance shape that used to have no plan at all: one targeted atom at the back
/// of a packed block, with the atoms ahead of it untargeted spectators.
///
/// The bound is unchanged — the goal constrains `q0` alone, `q0` needs `d` hops,
/// and a layer moves it one site, so `d` operations is optimal — but the route to
/// it is different. Scoring nominates only `q0`; the spectators have no target, so
/// nothing scores them and they never entered the bus group's mover set. That
/// left `q0`'s rectangle unexecutable and the whole instance without a single
/// successor, at any budget (#910). The fully-targeted shifts above were already
/// fine, because there the grid layer's repair closure had every follower to work
/// with; these are the cases that need selection to close its mover set first.
#[test]
fn a_block_pushed_by_one_targeted_atom_is_optimal() {
    // (sites, atoms in the block, hops q0 must travel).
    let cases = [(5u32, 4u32, 1u32), (8, 5, 3), (10, 4, 6), (10, 9, 1)];

    for (n_sites, n_atoms, shift) in cases {
        assert!(
            n_atoms + shift <= n_sites,
            "the block must have room to shift"
        );
        let initial: Vec<(u32, LocationAddr)> = (0..n_atoms).map(|q| (q, site(q))).collect();
        // Only q0 is targeted; q1.. are spectators that have to get out of the way.
        let target = [(0u32, site(shift))];
        let engine = line_engine(n_sites);

        for (name, search) in searches() {
            let solver = TargetSolver::new(Arc::clone(&engine), search);
            let result = solver
                .solve(
                    initial.iter().copied(),
                    target,
                    std::iter::empty(),
                    Some(SOLVABLE_BUDGET),
                )
                .expect("instance is well formed");

            let label = format!("{name}: q0 pushes {} spectators {shift} sites", n_atoms - 1);
            assert_eq!(
                result.status,
                SolveStatus::Solved,
                "{label}: reported {:?}",
                result.status
            );
            assert_eq!(
                result.goal_config.location_of(0),
                Some(site(shift)),
                "{label}: q0 did not reach its target"
            );
            assert_eq!(
                result.move_layers.len(),
                shift as usize,
                "{label}: expected the optimal {shift} operations, got {}",
                result.move_layers.len()
            );
            // Each operation carries the whole block, not just the targeted atom:
            // the spectators end up exactly `shift` sites along.
            for q in 1..n_atoms {
                assert_eq!(
                    result.goal_config.location_of(q),
                    Some(site(q + shift)),
                    "{label}: spectator q{q} did not ride the chain"
                );
            }
        }
    }
}

/// **Optimality, both directions.** The same bound holds running backwards along
/// the conveyor, which uses the reverse lane group. Asserted separately because
/// a chain is confined to one direction — a lane group shares one — so the
/// backward shift is a different set of lanes, not a mirror of the same ones.
#[test]
fn backward_shifts_are_also_optimal() {
    let n_sites = 8;
    let engine = line_engine(n_sites);
    for shift in 1u32..=3 {
        let n_atoms = 4u32;
        // Block starts at the far end and walks back toward site 0.
        let start = n_sites - n_atoms;
        let initial: Vec<(u32, LocationAddr)> =
            (0..n_atoms).map(|q| (q, site(start + q))).collect();
        let target: Vec<(u32, LocationAddr)> =
            (0..n_atoms).map(|q| (q, site(start + q - shift))).collect();

        for (name, search) in searches() {
            let solver = TargetSolver::new(Arc::clone(&engine), search);
            let result = solver
                .solve(
                    initial.iter().copied(),
                    target.iter().copied(),
                    std::iter::empty(),
                    Some(SOLVABLE_BUDGET),
                )
                .expect("instance is well formed");

            let label = format!("{name}: backward shift {shift}");
            assert_eq!(result.status, SolveStatus::Solved, "{label}: not solved");
            for &(qid, want) in &target {
                assert_eq!(
                    result.goal_config.location_of(qid),
                    Some(want),
                    "{label}: q{qid} did not reach its target"
                );
            }
            assert_eq!(
                result.move_layers.len(),
                shift as usize,
                "{label}: expected the optimal {shift} operations, got {}",
                result.move_layers.len()
            );
        }
    }
}

/// **Soundness.** A cyclic permutation of atom identities across a fixed set of
/// sites is unreachable on a line: atoms cannot pass each other, so their order
/// along the row is invariant, and a rotation of identities changes it.
///
/// Executing it would need the one operation the architecture does not have — a
/// rotation, as opposed to a set of edge transports (#866/#874). No strategy may
/// return `Solved` here. The verdict itself is not asserted: from a search
/// strategy `Unsolvable` is exhaustion, not proof (see `SolveStatus`), so
/// `Unsolvable` and `BudgetExceeded` are both honest answers and only `Solved`
/// is a bug.
#[test]
fn router_never_fabricates_a_rotation() {
    // Every non-identity rotation of 2..=4 atoms packed at the start of the
    // line, with and without a hole to work in.
    for n_sites in [4u32, 6, 8] {
        let engine = line_engine(n_sites);
        for n_atoms in 2..=4u32.min(n_sites - 1) {
            for shift in 1..n_atoms {
                let initial: Vec<(u32, LocationAddr)> =
                    (0..n_atoms).map(|q| (q, site(q))).collect();
                // q_i takes the site q_{i+shift mod n} held: a single cycle.
                let target: Vec<(u32, LocationAddr)> = (0..n_atoms)
                    .map(|q| (q, site((q + shift) % n_atoms)))
                    .collect();

                for (name, search) in searches() {
                    let solver = TargetSolver::new(Arc::clone(&engine), search);
                    let result = solver
                        .solve(
                            initial.iter().copied(),
                            target.iter().copied(),
                            std::iter::empty(),
                            Some(UNSOLVABLE_BUDGET),
                        )
                        .expect("instance is well formed");

                    assert_ne!(
                        result.status,
                        SolveStatus::Solved,
                        "{name}: claimed a solution for a rotation of {n_atoms} atoms by \
                         {shift} on {n_sites} sites, which no sequence of edge transports \
                         can realize; plan was {:?}",
                        result.move_layers
                    );
                }
            }
        }
    }
}

/// **Soundness, non-rotation reorderings.** The same order invariant rules out
/// *any* reordering, not just cyclic ones: a single transposition of two atoms
/// is unreachable on a line however much free space it is given.
#[test]
fn router_never_transposes_two_atoms_on_a_line() {
    for n_sites in [4u32, 8, 12] {
        let engine = line_engine(n_sites);
        // Two atoms with the whole line to themselves, asked to swap.
        let initial = [(0u32, site(0)), (1, site(1))];
        let target = [(0u32, site(1)), (1, site(0))];

        for (name, search) in searches() {
            let solver = TargetSolver::new(Arc::clone(&engine), search);
            let result = solver
                .solve(initial, target, std::iter::empty(), Some(UNSOLVABLE_BUDGET))
                .expect("instance is well formed");

            assert_ne!(
                result.status,
                SolveStatus::Solved,
                "{name}: claimed a swap on a {n_sites}-site line; plan was {:?}",
                result.move_layers
            );
        }
    }
}

/// **Zero holes.** A fully packed line cannot move at all: every lane's
/// destination is held by an atom, and the atom at the far end has no outgoing
/// lane, so no chain can complete and no rectangle is executable. Only the
/// identity target is reachable.
///
/// This is the regime the exhaustive sweep below deliberately excludes (it needs
/// at least one hole to have anything to enumerate), and it is the boundary at
/// which chain assembly must stop helping — a closure that "completed" a chain
/// here would be manufacturing a rotation.
#[test]
fn fully_packed_line_admits_only_the_identity() {
    for n_sites in [3u32, 5] {
        let engine = line_engine(n_sites);
        let initial: Vec<(u32, LocationAddr)> = (0..n_sites).map(|q| (q, site(q))).collect();

        for (name, search) in searches() {
            // The identity target is already met at the root: solved, no moves.
            let solver = TargetSolver::new(Arc::clone(&engine), search);
            let identity = solver
                .solve(
                    initial.iter().copied(),
                    initial.iter().copied(),
                    std::iter::empty(),
                    Some(SOLVABLE_BUDGET),
                )
                .expect("instance is well formed");
            assert_eq!(
                identity.status,
                SolveStatus::Solved,
                "{name}: the identity on a packed line is already solved"
            );
            assert!(
                identity.move_layers.is_empty(),
                "{name}: the identity needs no operations, got {:?}",
                identity.move_layers
            );

            // Anything else is unreachable: there is nowhere for any atom to go.
            let swapped: Vec<(u32, LocationAddr)> = (0..n_sites)
                .map(|q| {
                    (
                        q,
                        site(if q == 0 {
                            1
                        } else if q == 1 {
                            0
                        } else {
                            q
                        }),
                    )
                })
                .collect();
            let solver = TargetSolver::new(Arc::clone(&engine), MoveSearch::astar(1.0));
            let moved = solver
                .solve(
                    initial.iter().copied(),
                    swapped.iter().copied(),
                    std::iter::empty(),
                    Some(UNSOLVABLE_BUDGET),
                )
                .expect("instance is well formed");
            assert_ne!(
                moved.status,
                SolveStatus::Solved,
                "{name}: claimed a move on a fully packed line; plan was {:?}",
                moved.move_layers
            );
        }
    }
}

// ── Exhaustive sweep against a brute-force optimum ──

/// A configuration as a sorted, comparable key.
type ConfigKey = Vec<(u32, u64)>;

fn config_key(config: &Config) -> ConfigKey {
    let mut v: ConfigKey = config.iter().map(|(q, l)| (q, l.encode())).collect();
    v.sort_unstable();
    v
}

/// Optimal operation count from `start` to **every** reachable configuration, by
/// breadth-first search over
/// [`ExhaustiveGenerator`](bloqade_lanes_search::generators::exhaustive::ExhaustiveGenerator).
///
/// That generator enumerates every valid X × Y rectangle on every bus group, so
/// its successor relation *is* the architecture's move relation — which makes BFS
/// over it the ground truth for both questions this file asks: what is reachable,
/// and in how few AOD operations. It is only tractable because a short line has a
/// handful of configurations; it is a test oracle, not a router.
///
/// One BFS serves every target for this start: the generator reads only
/// `ctx.index` and `ctx.blocked`, never the targets or the distance table, so the
/// successor relation is target-independent.
fn optimal_distances_from(index: &LaneIndex, start: &Config) -> HashMap<ConfigKey, usize> {
    let no_targets: Vec<(u32, u64)> = Vec::new();
    let table = DistanceTable::new(&[], index);
    let blocked = HashSet::new();
    let ctx = SearchContext {
        index,
        dist_table: &table,
        blocked: &blocked,
        targets: &no_targets,
        cz_pairs: None,
    };
    let generator = ExhaustiveGenerator::new(None, None);
    // `NodeId` has no public constructor and the exhaustive generator ignores
    // the one it is handed, so borrow a root id from a throwaway graph.
    let scratch = SearchGraph::new(start.clone());
    let node = scratch.root();
    let mut state = SearchState::default();

    let mut dist: HashMap<ConfigKey, usize> = HashMap::new();
    dist.insert(config_key(start), 0);
    let mut queue: VecDeque<(Config, usize)> = VecDeque::new();
    queue.push_back((start.clone(), 0));

    while let Some((config, d)) = queue.pop_front() {
        let mut out = Vec::new();
        generator.generate(&config, node, &ctx, &mut state, &mut out);
        for candidate in out {
            let key = config_key(&candidate.new_config);
            if dist.contains_key(&key) {
                continue;
            }
            dist.insert(key, d + 1);
            queue.push_back((candidate.new_config, d + 1));
        }
    }

    dist
}

/// Every injective placement of `k` labelled atoms onto `n` sites, as site lists
/// indexed by qubit id.
fn placements(n: u32, k: u32) -> Vec<Vec<u32>> {
    fn rec(n: u32, k: u32, cur: &mut Vec<u32>, out: &mut Vec<Vec<u32>>) {
        if cur.len() == k as usize {
            out.push(cur.clone());
            return;
        }
        for s in 0..n {
            if !cur.contains(&s) {
                cur.push(s);
                rec(n, k, cur, out);
                cur.pop();
            }
        }
    }
    let mut out = Vec::new();
    rec(n, k, &mut Vec::new(), &mut out);
    out
}

/// Fewest operations from the BFS map to any reachable configuration satisfying
/// every `(qubit, encoded site)` constraint in `target`.
///
/// A *total* target names every atom, so at most one configuration can match and
/// this is a lookup. A *partial* target leaves the untargeted atoms free, so the
/// optimum is the minimum over every reachable configuration that places the named
/// ones correctly — which is exactly the goal predicate the solver uses
/// (`AllAtTarget` checks only the qubits it was given).
fn optimal_for_target(
    distances: &HashMap<ConfigKey, usize>,
    target: &[(u32, LocationAddr)],
) -> Option<usize> {
    distances
        .iter()
        .filter(|(key, _)| {
            target
                .iter()
                .all(|&(qid, loc)| key.contains(&(qid, loc.encode())))
        })
        .map(|(_, &d)| d)
        .min()
}

/// **Every 1D instance, against ground truth.** For each size, enumerate every
/// start placement and every target — total and single-atom partial — and check
/// the router against a brute-force optimum.
///
/// This is what closes the coverage gap the hand-written families leave: it
/// includes holes inside the atom set, holes at either end, compaction, spreading,
/// mixed compaction-and-shift, and every reordering — not just the packed blocks
/// with trailing holes that the shift tests use.
///
/// Start placements are restricted to *sorted* site lists, which loses nothing:
/// relabelling atoms is a symmetry of the instance, and sweeping all `k!`
/// relabellings of each start would only multiply the runtime.
///
/// **Both kinds of target are swept, and they are not interchangeable.** With a
/// total assignment on a line, any solvable instance is order-preserving, so every
/// atom that has to move wants to move the same way as the atom behind it and
/// scoring nominates them all — that half of the sweep passes with or without the
/// #910 closure. The single-atom targets are what reach the bug: the untargeted
/// atoms are spectators, nothing scores them, and before the closure a targeted
/// atom with a spectator in front of it had no successors at all.
///
/// Three properties, in decreasing order of how negotiable they are:
///
/// 1. **No impossible plans.** A `Solved` result must correspond to a reachable
///    target, and its length must be at least the optimum. Violating either means
///    the router used a move the architecture does not have — the failure mode a
///    chain-assembly bug would produce, since a chain that "completes" without a
///    free head is a rotation.
/// 2. **No missed solutions.** Every reachable target is found. This one *is*
///    negotiable in principle — the heuristic generators deliberately offer less
///    than the architecture, so a search failing to find a plan is a quality
///    result, not a correctness one — but at these sizes it currently holds
///    exactly.
/// 3. **Optimal length.** Every plan found is exactly as long as the optimum.
#[test]
fn router_matches_a_brute_force_optimum_on_every_1d_instance() {
    // (sites, atoms). Sizes are bounded by running three strategies per instance;
    // these cover 1 to 3 holes across three shapes.
    for (n_sites, n_atoms) in [(4u32, 2u32), (5, 3), (5, 4)] {
        let engine = line_engine(n_sites);
        let index = LaneIndex::new(
            serde_json::from_str::<ArchSpec>(&line_arch_json(n_sites)).expect("parses"),
        );
        let all = placements(n_sites, n_atoms);

        // Total targets: every atom placed. Partial targets: one atom placed, the
        // rest left as spectators.
        let mut targets: Vec<Vec<(u32, LocationAddr)>> = all
            .iter()
            .map(|sites| {
                sites
                    .iter()
                    .enumerate()
                    .map(|(q, &s)| (q as u32, site(s)))
                    .collect()
            })
            .collect();
        for qid in 0..n_atoms {
            for s in 0..n_sites {
                targets.push(vec![(qid, site(s))]);
            }
        }

        for start_sites in all.iter().filter(|v| v.windows(2).all(|w| w[0] < w[1])) {
            let initial: Vec<(u32, LocationAddr)> = start_sites
                .iter()
                .enumerate()
                .map(|(q, &s)| (q as u32, site(s)))
                .collect();
            let root = Config::new(initial.iter().copied()).expect("placement is injective");
            let distances = optimal_distances_from(&index, &root);

            for target in &targets {
                let optimal = optimal_for_target(&distances, target);
                let budget = SWEEP_BUDGET;

                for (name, search) in searches() {
                    let solver = TargetSolver::new(Arc::clone(&engine), search);
                    let result = solver
                        .solve(
                            initial.iter().copied(),
                            target.iter().copied(),
                            std::iter::empty(),
                            Some(budget),
                        )
                        .expect("instance is well formed");
                    let label = format!(
                        "{name}: {n_atoms} atoms on {n_sites} sites, {start_sites:?} -> {:?}",
                        target
                            .iter()
                            .map(|&(q, l)| (q, l.site_id))
                            .collect::<Vec<_>>()
                    );

                    match optimal {
                        // Unreachable: no plan may be claimed.
                        None => assert_ne!(
                            result.status,
                            SolveStatus::Solved,
                            "{label}: claimed a plan for an unreachable target; got {:?}",
                            result.move_layers
                        ),
                        Some(optimal) => {
                            assert_eq!(
                                result.status,
                                SolveStatus::Solved,
                                "{label}: reported {:?} on a target reachable in {optimal} \
                                 operations",
                                result.status
                            );
                            for &(qid, wanted) in target {
                                assert_eq!(
                                    result.goal_config.location_of(qid),
                                    Some(wanted),
                                    "{label}: q{qid} did not reach its target"
                                );
                            }
                            assert!(
                                result.move_layers.len() >= optimal,
                                "{label}: returned {} operations, fewer than the optimum \
                                 {optimal} — the plan uses a move the architecture does \
                                 not have",
                                result.move_layers.len()
                            );
                            assert_eq!(
                                result.move_layers.len(),
                                optimal,
                                "{label}: returned {} operations against an optimum of \
                                 {optimal}",
                                result.move_layers.len()
                            );
                        }
                    }
                }
            }
        }
    }
}
