//! Sound infeasibility detection for atom rearrangement.
//!
//! Answers the question the solvers cannot: *is this rearrangement possible at
//! all?* Today a solver that drains its frontier reports
//! [`SolveStatus::Unsolvable`](crate::search::result::SolveStatus::Unsolvable),
//! but for the entropy/DFS/IDS drivers — which run a pruning generator and are
//! not exhaustive — that is a guess, not a proof. This module supplies the
//! proof side, so callers can distinguish "the hardware cannot do this" from
//! "the heuristic gave up".
//!
//! ## Why pebble motion applies
//!
//! Atom rearrangement here is *exactly* classical pebble motion on an
//! undirected graph, for two reasons specific to this architecture:
//!
//! 1. A single-lane [`MoveSet`](crate::primitives::graph::MoveSet) is always a
//!    legal AOD operation — `ArchSpec::check_lanes` only applies the
//!    complete-rectangle geometry constraint to groups of more than one lane.
//!    So single-pebble moves are available.
//! 2. Every bus is a matching between *disjoint* source and destination sets,
//!    so a multi-lane move is a set of vertex-disjoint edges whose
//!    destinations must all already be empty. Such a move serializes into
//!    independent single-pebble moves in any order, and no move can rotate a
//!    cycle of atoms with no empty vertex.
//!
//! Together these mean the set of reachable configurations under AOD moves is
//! identical to the set reachable under one-atom-at-a-time pebble motion —
//! AOD parallelism changes the schedule, not what is reachable. Point 2 is a
//! property of the shipped architecture specs rather than of the format, so
//! [`validate_bus_disjointness`] checks it explicitly; if it ever fails, the
//! reduction in this module is invalid. In debug builds, [`check`] and
//! [`build_decomposition`] assert it on entry (together with
//! `ArchSpec::validate`); release builds skip the check and trust the caller
//! to pass a validated spec.
//!
//! Strictly, the reduction needs less than disjointness: a bus is by
//! definition a set of explicit transports along graph edges, never a
//! permutation, so the load-bearing property is that no bus's src→dst
//! relation contains a cycle (a chain `x→y, y→z` serializes in reverse
//! topological order and is pebble-equivalent; only a rotation is not).
//! Overlapping-but-acyclic buses, and relaxing this module's guard
//! accordingly, are tracked in issue #866.
//!
//! ## What this does and does not prove
//!
//! The verdict is **one-sided**. [`Feasibility::Infeasible`] is a proof.
//! [`Feasibility::NoObstructionFound`] is *not* a proof of feasibility — it
//! means none of the implemented obstructions fired.
//!
//! The obstructions come from the Kornhauser subgraph decomposition as
//! reconstructed by de Wilde, ter Mors & Witteveen (JAIR 51, 2014, §3.1):
//! goal containment (p. 457, "individual subgraphs are solvable in case the
//! goal positions of the agents assigned to a subgraph are inside the
//! subgraph or its planks") and Proposition 2 (p. 461, "if the priority
//! relation between subgraphs is cyclic, then the instance is not solvable").
//! Whether those two are jointly *sufficient* at `m ≥ 2` is not claimed by
//! the paper, and is not claimed here. A complete oracle requires running the
//! Push and Rotate planner itself, which returns `false` exactly when the
//! instance is unsolvable (Theorem 1) — that is follow-up work.
//!
//! Note also that Wilson's parity exceptions (cycle graphs and θ₀) only arise
//! with a *single* empty vertex. At `m ≥ 2` the group-theoretic obstruction
//! disappears, which is why nothing here computes permutation parity.

pub mod decomposition;
pub mod graph;

use std::collections::{HashMap, HashSet};

use bloqade_lanes_bytecode_core::arch::types::ArchSpec;

use crate::feasibility::decomposition::{
    Decomposition, find_precedence_cycle, subgraph_priorities,
};
use crate::feasibility::graph::{LaneGraph, VertexId};
use crate::primitives::config::Config;
use crate::primitives::lane_index::LaneIndex;

/// Minimum number of empty vertices — per connected component — for the
/// decomposition-based obstructions to apply to that component. Push and
/// Rotate is complete only at or above this threshold, and below it Wilson's
/// parity exceptions come into play. Empties are counted per component
/// because a disconnected instance is a product of independent pebble-motion
/// instances: an empty vertex in another component cannot help maneuvering.
pub const MIN_EMPTY_VERTICES: usize = 2;

/// A proven reason the instance cannot be solved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Obstruction {
    /// An atom sits on a location that is not a vertex of the lane graph —
    /// either blocked, or not an endpoint of any lane.
    AtomNotOnGraph { qubit: u32, location: u64 },
    /// A target location is not a vertex of the lane graph.
    TargetNotOnGraph { qubit: u32, location: u64 },
    /// Two atoms occupy the same location. The move semantics assume an
    /// injective placement; [`Config`] only rejects duplicate *qubit ids*.
    DuplicateOccupancy { location: u64, qubits: (u32, u32) },
    /// Two atoms are assigned the same target location. The solve entry
    /// points reject such requests as errors before any pass runs
    /// (`validate_target_assignment`); this obstruction remains as defence
    /// in depth for direct callers of [`check`].
    DuplicateTarget { location: u64, qubits: (u32, u32) },
    /// An atom's target lies in a different connected component of the lane
    /// graph — no sequence of moves can ever cross between them.
    TargetUnreachable { qubit: u32, from: u64, to: u64 },
    /// An agent confined to a subgraph (Proposition 1) has a goal outside
    /// that subgraph and its planks.
    GoalOutsideSubgraph {
        qubit: u32,
        subgraph: usize,
        goal: u64,
    },
    /// The subgraph precedence relation is cyclic (Proposition 2).
    CyclicPrecedence { subgraphs: Vec<usize> },
}

impl std::fmt::Display for Obstruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AtomNotOnGraph { qubit, location } => write!(
                f,
                "qubit {qubit} sits at location {location:#x}, which is blocked or not on any lane"
            ),
            Self::TargetNotOnGraph { qubit, location } => write!(
                f,
                "target {location:#x} for qubit {qubit} is blocked or not on any lane"
            ),
            Self::DuplicateOccupancy { location, qubits } => write!(
                f,
                "qubits {} and {} both occupy location {location:#x}",
                qubits.0, qubits.1
            ),
            Self::DuplicateTarget { location, qubits } => write!(
                f,
                "qubits {} and {} share target location {location:#x}",
                qubits.0, qubits.1
            ),
            Self::TargetUnreachable { qubit, from, to } => write!(
                f,
                "qubit {qubit} cannot reach {to:#x} from {from:#x}: different connected components"
            ),
            Self::GoalOutsideSubgraph {
                qubit,
                subgraph,
                goal,
            } => write!(
                f,
                "qubit {qubit} is confined to subgraph {subgraph}, but its goal {goal:#x} lies outside it and its planks"
            ),
            Self::CyclicPrecedence { subgraphs } => {
                write!(f, "cyclic subgraph precedence: {subgraphs:?}")
            }
        }
    }
}

/// One-sided feasibility verdict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Feasibility {
    /// The instance is provably unsolvable.
    Infeasible(Obstruction),
    /// No implemented obstruction fired. **Not** a proof of feasibility.
    NoObstructionFound,
}

impl Feasibility {
    /// Whether this verdict proves the instance unsolvable.
    pub fn is_infeasible(&self) -> bool {
        matches!(self, Self::Infeasible(_))
    }

    /// The obstruction, if the verdict is a proof.
    pub fn obstruction(&self) -> Option<&Obstruction> {
        match self {
            Self::Infeasible(o) => Some(o),
            Self::NoObstructionFound => None,
        }
    }
}

/// Verify that every bus maps a source set disjoint from its destination set.
///
/// The whole pebble-motion reduction rests on this: it is what rules out an
/// AOD move rotating a cycle of atoms with no empty vertex. Returns one
/// message per violating bus; an empty vector means the property holds.
pub fn validate_bus_disjointness(arch: &ArchSpec) -> Vec<String> {
    let mut errors = Vec::new();

    for (zone_id, zone) in arch.zones.iter().enumerate() {
        for (bus_id, bus) in zone.site_buses.iter().enumerate() {
            let src: HashSet<u16> = bus.src.iter().map(|s| s.0).collect();
            if bus.dst.iter().any(|d| src.contains(&d.0)) {
                errors.push(format!(
                    "zone {zone_id} site_bus {bus_id}: src and dst sets overlap"
                ));
            }
        }
        for (bus_id, bus) in zone.word_buses.iter().enumerate() {
            let src: HashSet<u16> = bus.src.iter().map(|s| s.0).collect();
            if bus.dst.iter().any(|d| src.contains(&d.0)) {
                errors.push(format!(
                    "zone {zone_id} word_bus {bus_id}: src and dst sets overlap"
                ));
            }
        }
    }
    for (bus_id, bus) in arch.zone_buses.iter().enumerate() {
        let src: HashSet<(u8, u16)> = bus.src.iter().map(|s| (s.zone_id, s.word_id)).collect();
        if bus
            .dst
            .iter()
            .any(|d| src.contains(&(d.zone_id, d.word_id)))
        {
            errors.push(format!("zone_bus {bus_id}: src and dst sets overlap"));
        }
    }

    errors
}

/// Debug-build guard for the assumptions the pebble-motion reduction rests
/// on: a structurally valid architecture whose buses keep their source and
/// destination sets disjoint (see the module docs).
///
/// Release builds skip this entirely — callers are expected to pass a
/// validated spec (e.g. loaded via `ArchSpec::from_json_validated`). The
/// disjointness requirement relaxes to per-bus acyclicity once that becomes
/// a validated format invariant (issue #866).
fn debug_assert_valid_arch(_index: &LaneIndex) {
    #[cfg(debug_assertions)]
    {
        if let Err(errors) = _index.arch_spec().validate() {
            panic!("feasibility requires a structurally valid ArchSpec: {errors:?}");
        }
        let overlaps = validate_bus_disjointness(_index.arch_spec());
        assert!(
            overlaps.is_empty(),
            "feasibility reduction requires src/dst-disjoint buses \
             (relaxation to acyclicity tracked in issue #866): {overlaps:?}"
        );
    }
}

/// Build the undirected graph and the Kornhauser decomposition for an
/// instance, without evaluating any obstruction.
///
/// Exposed because the Push and Rotate planner needs exactly this: the
/// subgraphs drive its `swap` feasibility, and the precedence order drives its
/// agent priorities. Each connected component is analysed with its own empty
/// count; components below [`MIN_EMPTY_VERTICES`] contribute no subgraphs.
/// Returns `None` when the whole graph has fewer than [`MIN_EMPTY_VERTICES`]
/// empty vertices (no component can qualify), and for malformed input — an
/// atom off the graph, or two atoms sharing a location — which [`check`]
/// reports as its own obstruction.
pub fn build_decomposition(
    index: &LaneIndex,
    initial: &Config,
    blocked: &HashSet<u64>,
) -> Option<(LaneGraph, Decomposition)> {
    debug_assert_valid_arch(index);
    let graph = LaneGraph::build(index, blocked);
    let occupant = occupancy(&graph, initial)?;
    let empty_count = graph.len().checked_sub(initial.len())?;
    if empty_count < MIN_EMPTY_VERTICES {
        return None;
    }

    let decomp = Decomposition::build(&graph, &occupant);
    Some((graph, decomp))
}

/// Map each graph vertex to the qubit occupying it, or `None`.
///
/// Returns `None` if any atom sits off the graph or two atoms share a
/// location — the caller reports those as its own obstructions.
fn occupancy(graph: &LaneGraph, initial: &Config) -> Option<Vec<Option<u32>>> {
    let mut occupant = vec![None; graph.len()];
    for (qubit, loc) in initial.iter() {
        let v = graph.vertex_of(loc.encode())?;
        if occupant[v].is_some() {
            return None;
        }
        occupant[v] = Some(qubit);
    }
    Some(occupant)
}

/// Check an instance for a proven obstruction.
///
/// `targets` is the desired `(qubit, encoded location)` assignment; qubits
/// absent from it are unconstrained. A target for a qubit that is not in
/// `initial` is ignored entirely — there is no atom to move, so it cannot
/// constrain anything, and it must not be able to produce an obstruction
/// (a duplicate target, an off-graph target, or a precedence edge) for an
/// otherwise solvable instance. `blocked` locations are treated as removed
/// from the graph entirely.
pub fn check(
    index: &LaneIndex,
    initial: &Config,
    targets: &[(u32, u64)],
    blocked: &HashSet<u64>,
) -> Feasibility {
    debug_assert_valid_arch(index);
    let graph = LaneGraph::build(index, blocked);

    // ── Well-formedness ────────────────────────────────────────────
    let mut occupant: Vec<Option<u32>> = vec![None; graph.len()];
    for (qubit, loc) in initial.iter() {
        let enc = loc.encode();
        let Some(v) = graph.vertex_of(enc) else {
            return Feasibility::Infeasible(Obstruction::AtomNotOnGraph {
                qubit,
                location: enc,
            });
        };
        if let Some(other) = occupant[v] {
            return Feasibility::Infeasible(Obstruction::DuplicateOccupancy {
                location: enc,
                qubits: (other.min(qubit), other.max(qubit)),
            });
        }
        occupant[v] = Some(qubit);
    }

    // Targets for qubits absent from `initial` are dropped before any check:
    // there is no atom to move, so such a target is vacuous, and letting it
    // reach the duplicate/off-graph checks below (or the precedence relation
    // later) could manufacture an obstruction for a solvable instance.
    let initial_qubits: HashSet<u32> = initial.iter().map(|(q, _)| q).collect();
    let mut target_vertex: HashMap<u32, VertexId> = HashMap::new();
    let mut target_seen: HashMap<u64, u32> = HashMap::new();
    for &(qubit, enc) in targets {
        if !initial_qubits.contains(&qubit) {
            continue;
        }
        let Some(v) = graph.vertex_of(enc) else {
            return Feasibility::Infeasible(Obstruction::TargetNotOnGraph {
                qubit,
                location: enc,
            });
        };
        if let Some(&other) = target_seen.get(&enc) {
            return Feasibility::Infeasible(Obstruction::DuplicateTarget {
                location: enc,
                qubits: (other.min(qubit), other.max(qubit)),
            });
        }
        target_seen.insert(enc, qubit);
        target_vertex.insert(qubit, v);
    }

    // ── Connectivity: a target in another component is unreachable ──
    let (component, _n_components) = graph.connected_components();
    for (qubit, loc) in initial.iter() {
        let Some(&goal_v) = target_vertex.get(&qubit) else {
            continue;
        };
        let from_v = graph
            .vertex_of(loc.encode())
            .expect("atom vertices were resolved above");
        if component[from_v] != component[goal_v] {
            return Feasibility::Infeasible(Obstruction::TargetUnreachable {
                qubit,
                from: loc.encode(),
                to: graph.location_of(goal_v),
            });
        }
    }

    // ── Decomposition-based obstructions ────────────────────────────
    match decomposition_obstruction(&graph, &occupant, &target_vertex) {
        Some(obstruction) => Feasibility::Infeasible(obstruction),
        None => Feasibility::NoObstructionFound,
    }
}

/// Decomposition-phase obstructions (goal containment and Proposition 2) on
/// a well-formed instance.
///
/// `occupant` must be injective over the graph's vertices, and
/// `target_vertex` must map only qubits that occupy a vertex — [`check`]
/// establishes both before calling. Kept separate from [`check`] so the
/// brute-force soundness tests can exercise the decomposition verdict on
/// synthetic [`LaneGraph`]s directly.
fn decomposition_obstruction(
    graph: &LaneGraph,
    occupant: &[Option<u32>],
    target_vertex: &HashMap<u32, VertexId>,
) -> Option<Obstruction> {
    let atom_count = occupant.iter().filter(|o| o.is_some()).count();
    let empty_count = graph.len().saturating_sub(atom_count);
    if empty_count < MIN_EMPTY_VERTICES {
        // No component can be in the Push and Rotate regime, so only the
        // cheap checks stand. Fast path — `build` would produce an empty
        // decomposition anyway, since components below the threshold get no
        // subgraphs.
        return None;
    }

    let decomp = Decomposition::build(graph, occupant);

    // Goal containment: Proposition 1 confines an assigned agent to its
    // subgraph and planks, so a goal outside that region is unreachable.
    for (&qubit, &sub) in &decomp.assignment {
        let Some(&goal_v) = target_vertex.get(&qubit) else {
            continue;
        };
        if !decomp.contains_in_subgraph_or_planks(sub, goal_v) {
            return Some(Obstruction::GoalOutsideSubgraph {
                qubit,
                subgraph: sub,
                goal: graph.location_of(goal_v),
            });
        }
    }

    // Proposition 2: a cyclic precedence relation proves unsolvability.
    let unassigned_goal_vertices: HashSet<VertexId> = target_vertex
        .iter()
        .filter(|(q, _)| !decomp.assignment.contains_key(q))
        .map(|(_, &v)| v)
        .collect();
    let edges = subgraph_priorities(&decomp, target_vertex, &unassigned_goal_vertices);
    if let Some(cycle) = find_precedence_cycle(decomp.subgraphs.len(), &edges) {
        return Some(Obstruction::CyclicPrecedence { subgraphs: cycle });
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{example_arch_json, loc};
    use bloqade_lanes_bytecode_core::arch::types::ArchSpec;

    /// Brute force over the configuration space: is there a reachable state
    /// where every targeted agent sits on its goal simultaneously? Ground
    /// truth for the decomposition verdict.
    fn instance_solvable(
        graph: &LaneGraph,
        occupant: &[Option<u32>],
        targets: &HashMap<u32, VertexId>,
    ) -> bool {
        use std::collections::VecDeque;
        let satisfied =
            |state: &[Option<u32>]| targets.iter().all(|(&q, &goal)| state[goal] == Some(q));
        let start: Vec<Option<u32>> = occupant.to_vec();
        if satisfied(&start) {
            return true;
        }
        let mut seen: HashSet<Vec<Option<u32>>> = HashSet::new();
        let mut queue: VecDeque<Vec<Option<u32>>> = VecDeque::new();
        seen.insert(start.clone());
        queue.push_back(start);
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

    /// End-to-end soundness property: whenever the decomposition phase
    /// reports an obstruction, exhaustive search must confirm the instance
    /// unsolvable. Random graphs are frequently disconnected, and half the
    /// seeds append an extra all-empty component — probing that empties
    /// stranded in one component cannot weaken or fabricate verdicts in
    /// another.
    #[test]
    fn decomposition_verdicts_match_brute_force_on_random_graphs() {
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};

        let mut fired = 0usize;
        for seed in 0..400u64 {
            let mut rng = SmallRng::seed_from_u64(seed);
            let n_core = rng.random_range(4..=8);
            let density = if seed % 2 == 0 { 0.25 } else { 0.4 };
            let mut edges: Vec<(VertexId, VertexId)> = Vec::new();
            for a in 0..n_core {
                for b in (a + 1)..n_core {
                    if rng.random_bool(density) {
                        edges.push((a, b));
                    }
                }
            }
            // Half the seeds: strand extra empties in a separate component.
            let stranded = if seed % 2 == 0 { 3 } else { 0 };
            let n = n_core + stranded;
            if stranded > 0 {
                edges.push((n_core, n_core + 1));
                edges.push((n_core + 1, n_core + 2));
            }
            let graph = LaneGraph::from_edges(n, &edges);

            // Atoms only on core vertices, ≥ 2 empties among them.
            let mut verts: Vec<VertexId> = (0..n_core).collect();
            for i in (1..n_core).rev() {
                let j = rng.random_range(0..=i);
                verts.swap(i, j);
            }
            let n_atoms = rng.random_range(1..=(n_core - 2));
            let mut occupant: Vec<Option<u32>> = vec![None; n];
            for (q, &v) in verts.iter().take(n_atoms).enumerate() {
                occupant[v] = Some(q as u32);
            }

            // Arbitrary targets (not necessarily solvable): a random subset
            // of agents onto random distinct core vertices.
            let n_targets = rng.random_range(1..=n_atoms);
            let mut goal_verts: Vec<VertexId> = (0..n_core).collect();
            for i in (1..n_core).rev() {
                let j = rng.random_range(0..=i);
                goal_verts.swap(i, j);
            }
            let target_vertex: HashMap<u32, VertexId> =
                (0..n_targets as u32).zip(goal_verts).collect();

            if let Some(obstruction) = decomposition_obstruction(&graph, &occupant, &target_vertex)
            {
                fired += 1;
                assert!(
                    !instance_solvable(&graph, &occupant, &target_vertex),
                    "seed {seed}: {obstruction:?} claimed for a brute-force-solvable \
                     instance (occupant {occupant:?}, targets {target_vertex:?})"
                );
            }
        }
        // The property is vacuous if obstructions rarely fire; keep the
        // fixture generator honest.
        eprintln!("obstructions fired on {fired}/400 seeds");
        assert!(
            fired > 0,
            "no seed produced an obstruction — weaken density"
        );
    }

    /// Adversarial per-component-m fixture: a dumbbell with 2 local empties
    /// plus 3 empties stranded in a separate component. Goals sit on plank
    /// positions that only exist if `m` were counted globally — under a
    /// global `m` this produced a precedence cycle; under per-component `m`
    /// the same instance is caught by goal containment instead. Either way
    /// the verdict must agree with exhaustive search (the instance is
    /// genuinely unsolvable: crossing the corridor needs more than the two
    /// local empties).
    #[test]
    fn stranded_empties_verdict_agrees_with_brute_force() {
        let graph = LaneGraph::from_edges(
            12,
            &[
                // dumbbell: triangles {0,1,2} and {6,7,8}, corridor 3-4-5
                (0, 1),
                (1, 2),
                (2, 0),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 6),
                // stranded all-empty component
                (9, 10),
                (10, 11),
            ],
        );
        // Atoms at 0(B),2,3,4(X),5,6,8(A); empties 1,7 locally + 9,10,11.
        let occupant: Vec<Option<u32>> = vec![
            Some(0), // B
            None,
            Some(1),
            Some(2),
            Some(3), // X
            Some(4),
            Some(5),
            None,
            Some(6), // A
            None,
            None,
            None,
        ];
        // B: 0→6 (case-1 precedence edge S1≺S0, m-independent).
        // A: 8→4 (on S0's plank only under inflated m → edge S0≺S1).
        // X: 4→3 (unassigned goal filling the 'between' slot for A's edge).
        let target_vertex: HashMap<u32, VertexId> =
            [(0u32, 6usize), (6u32, 4usize), (3u32, 3usize)]
                .into_iter()
                .collect();
        let verdict = decomposition_obstruction(&graph, &occupant, &target_vertex);
        let solvable = instance_solvable(&graph, &occupant, &target_vertex);
        assert!(!solvable, "fixture must be unsolvable — corridor too tight");
        assert!(
            verdict.is_some(),
            "the decomposition should catch this unsolvable instance"
        );
    }

    fn index_from(json: &str) -> LaneIndex {
        let spec: ArchSpec = serde_json::from_str(json).expect("arch json parses");
        LaneIndex::new(spec)
    }

    fn gemini_physical() -> LaneIndex {
        index_from(include_str!(
            "../../../../python/bloqade/lanes/arch/gemini/physical/_physical_spec.json"
        ))
    }

    #[test]
    fn shipped_gemini_specs_have_disjoint_bus_endpoints() {
        // The pebble-motion reduction is only valid while this holds.
        for (name, json) in [
            (
                "physical",
                include_str!(
                    "../../../../python/bloqade/lanes/arch/gemini/physical/_physical_spec.json"
                ),
            ),
            (
                "logical",
                include_str!(
                    "../../../../python/bloqade/lanes/arch/gemini/logical/_logical_spec.json"
                ),
            ),
        ] {
            let spec: ArchSpec = serde_json::from_str(json).expect("arch json parses");
            assert_eq!(
                validate_bus_disjointness(&spec),
                Vec::<String>::new(),
                "{name} spec must have src/dst-disjoint buses"
            );
        }
    }

    /// The debug-build guard must refuse a spec whose bus endpoints overlap:
    /// the reduction is not justified there, so silently returning verdicts
    /// would be unsound. Relaxing this to per-bus acyclicity is issue #866.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "src/dst-disjoint")]
    fn debug_guard_rejects_overlapping_bus() {
        let mut spec: ArchSpec =
            serde_json::from_str(example_arch_json()).expect("arch json parses");
        // Point the first destination at the second source, forming the
        // acyclic chain 0→1→6. Structural validation still passes (the
        // relation stays cycle-free and endpoint-unique, per issue #874);
        // disjointness does not.
        let bus = &mut spec.zones[0].site_buses[0];
        bus.dst[0] = bus.src[1];
        let index = LaneIndex::new(spec);
        let initial = Config::new([(0, loc(0, 0))]).expect("config");
        let _ = check(&index, &initial, &[], &HashSet::new());
    }

    #[test]
    fn trivial_instance_has_no_obstruction() {
        let index = index_from(example_arch_json());
        let initial = Config::new([(0, loc(0, 0))]).expect("config");
        let targets = [(0u32, loc(0, 5).encode())];
        assert_eq!(
            check(&index, &initial, &targets, &HashSet::new()),
            Feasibility::NoObstructionFound
        );
    }

    #[test]
    fn atom_on_blocked_site_is_infeasible() {
        let index = index_from(example_arch_json());
        let initial = Config::new([(0, loc(0, 0))]).expect("config");
        let blocked: HashSet<u64> = [loc(0, 0).encode()].into_iter().collect();
        let verdict = check(&index, &initial, &[], &blocked);
        assert!(matches!(
            verdict,
            Feasibility::Infeasible(Obstruction::AtomNotOnGraph { qubit: 0, .. })
        ));
    }

    #[test]
    fn target_off_the_graph_is_infeasible() {
        let index = index_from(example_arch_json());
        let initial = Config::new([(0, loc(0, 0))]).expect("config");
        // Word 99 does not exist in the example architecture.
        let targets = [(0u32, loc(99, 0).encode())];
        let verdict = check(&index, &initial, &targets, &HashSet::new());
        assert!(matches!(
            verdict,
            Feasibility::Infeasible(Obstruction::TargetNotOnGraph { qubit: 0, .. })
        ));
    }

    #[test]
    fn two_atoms_sharing_a_location_is_infeasible() {
        let index = index_from(example_arch_json());
        // `Config` only rejects duplicate qubit ids, not duplicate locations.
        let initial = Config::new([(0, loc(0, 0)), (1, loc(0, 0))]).expect("config");
        let verdict = check(&index, &initial, &[], &HashSet::new());
        assert!(matches!(
            verdict,
            Feasibility::Infeasible(Obstruction::DuplicateOccupancy { .. })
        ));
    }

    #[test]
    fn build_decomposition_rejects_duplicate_occupancy() {
        let index = index_from(example_arch_json());
        // `Config` only rejects duplicate qubit ids, not duplicate locations;
        // rather than silently dropping an atom, decline to decompose.
        let initial = Config::new([(0, loc(0, 0)), (1, loc(0, 0))]).expect("config");
        assert!(build_decomposition(&index, &initial, &HashSet::new()).is_none());
    }

    #[test]
    fn targets_for_absent_qubits_are_ignored_entirely() {
        let index = index_from(example_arch_json());
        let initial = Config::new([(0, loc(0, 0))]).expect("config");
        // Qubits 7 and 8 are not in `initial`: their targets are vacuous and
        // must not produce an obstruction of any kind — not a collision with
        // a real target (qubit 0's), and not an off-graph location. The
        // instance is solvable without moving anything for them.
        let targets = [
            (0u32, loc(0, 5).encode()),
            (7u32, loc(0, 5).encode()),  // collides with qubit 0's target
            (8u32, loc(99, 0).encode()), // word 99 is off the graph
        ];
        assert_eq!(
            check(&index, &initial, &targets, &HashSet::new()),
            Feasibility::NoObstructionFound
        );
    }

    #[test]
    fn two_atoms_sharing_a_target_is_infeasible() {
        let index = index_from(example_arch_json());
        let initial = Config::new([(0, loc(0, 0)), (1, loc(0, 1))]).expect("config");
        let targets = [(0u32, loc(0, 5).encode()), (1u32, loc(0, 5).encode())];
        let verdict = check(&index, &initial, &targets, &HashSet::new());
        assert!(matches!(
            verdict,
            Feasibility::Infeasible(Obstruction::DuplicateTarget { .. })
        ));
    }

    #[test]
    fn blocking_a_corridor_makes_the_target_unreachable() {
        // Cut every lane out of the atom's site by blocking its neighbours,
        // stranding it in a component that excludes the target.
        let index = index_from(example_arch_json());
        let start = loc(0, 0);
        let graph_all = LaneGraph::build(&index, &HashSet::new());
        let v = graph_all
            .vertex_of(start.encode())
            .expect("start is on the graph");
        let blocked: HashSet<u64> = graph_all
            .neighbors(v)
            .iter()
            .map(|&w| graph_all.location_of(w))
            .collect();
        assert!(!blocked.is_empty(), "fixture needs a non-isolated start");

        let initial = Config::new([(0, start)]).expect("config");
        // Any target that survives the blocking but is in another component.
        let graph_cut = LaneGraph::build(&index, &blocked);
        let (component, _) = graph_cut.connected_components();
        let start_v = graph_cut
            .vertex_of(start.encode())
            .expect("start survives blocking");
        let far = graph_cut
            .vertices()
            .find(|&w| component[w] != component[start_v]);
        let Some(far) = far else {
            // Architecture stayed connected — nothing to assert here.
            return;
        };
        let targets = [(0u32, graph_cut.location_of(far))];
        let verdict = check(&index, &initial, &targets, &blocked);
        assert!(
            matches!(
                verdict,
                Feasibility::Infeasible(Obstruction::TargetUnreachable { qubit: 0, .. })
            ),
            "expected unreachable target, got {verdict:?}"
        );
    }

    #[test]
    fn gemini_physical_decomposes_and_finds_no_obstruction() {
        // A realistic sparse instance on the shipped architecture must not
        // be reported infeasible — a false positive here would gate real
        // compilations.
        let index = gemini_physical();
        let graph = LaneGraph::build(&index, &HashSet::new());
        assert!(!graph.is_empty(), "gemini physical has lane endpoints");

        let verts: Vec<VertexId> = graph.vertices().take(8).collect();
        let initial = Config::new(
            verts
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    (
                        i as u32,
                        bloqade_lanes_bytecode_core::arch::addr::LocationAddr::decode(
                            graph.location_of(v),
                        ),
                    )
                })
                .collect::<Vec<_>>(),
        )
        .expect("config");

        let verdict = check(&index, &initial, &[], &HashSet::new());
        assert_eq!(verdict, Feasibility::NoObstructionFound);

        let built = build_decomposition(&index, &initial, &HashSet::new());
        let (_, decomp) = built.expect("gemini physical has ≥ 2 empty vertices");
        assert!(
            !decomp.subgraphs.is_empty(),
            "expected at least one subgraph"
        );
        assert!(decomp.empty_count >= MIN_EMPTY_VERTICES);
    }

    /// Soundness: an instance built by *replaying legal single-atom moves*
    /// is reachable by construction, so the checker must never call it
    /// infeasible. A false positive here would gate a real compilation.
    ///
    /// Single-atom moves are the right generator: a one-lane `MoveSet` is
    /// always a legal AOD operation, so every configuration produced by this
    /// walk is genuinely reachable from `initial` on hardware.
    fn assert_no_false_positive(index: &LaneIndex, n_atoms: usize, n_moves: usize, seed: u64) {
        use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};

        let graph = LaneGraph::build(index, &HashSet::new());
        assert!(
            graph.len() > n_atoms + MIN_EMPTY_VERTICES,
            "fixture needs room for {n_atoms} atoms plus empties"
        );
        let mut rng = SmallRng::seed_from_u64(seed);

        // Scatter atoms over distinct vertices.
        let mut placement: Vec<Option<u32>> = vec![None; graph.len()];
        let mut position: Vec<VertexId> = Vec::with_capacity(n_atoms);
        let mut placed = 0u32;
        while (placed as usize) < n_atoms {
            let v = rng.random_range(0..graph.len());
            if placement[v].is_some() || graph.degree(v) == 0 {
                continue;
            }
            placement[v] = Some(placed);
            position.push(v);
            placed += 1;
        }
        let initial = Config::new(
            position
                .iter()
                .enumerate()
                .map(|(q, &v)| (q as u32, LocationAddr::decode(graph.location_of(v))))
                .collect::<Vec<_>>(),
        )
        .expect("distinct vertices give a valid config");

        // Random walk: repeatedly slide one atom into an empty neighbour.
        for _ in 0..n_moves {
            let q = rng.random_range(0..n_atoms);
            let from = position[q];
            let empty: Vec<VertexId> = graph
                .neighbors(from)
                .iter()
                .copied()
                .filter(|&w| placement[w].is_none())
                .collect();
            if empty.is_empty() {
                continue;
            }
            let to = empty[rng.random_range(0..empty.len())];
            placement[from] = None;
            placement[to] = Some(q as u32);
            position[q] = to;
        }

        let targets: Vec<(u32, u64)> = position
            .iter()
            .enumerate()
            .map(|(q, &v)| (q as u32, graph.location_of(v)))
            .collect();

        let verdict = check(index, &initial, &targets, &HashSet::new());
        assert_eq!(
            verdict,
            Feasibility::NoObstructionFound,
            "reachable-by-construction instance reported infeasible (seed {seed})"
        );
    }

    #[test]
    fn never_reports_reachable_instances_infeasible_on_example_arch() {
        let index = index_from(example_arch_json());
        for seed in 0..25 {
            assert_no_false_positive(&index, 3, 40, seed);
        }
    }

    #[test]
    fn never_reports_reachable_instances_infeasible_on_gemini_physical() {
        let index = gemini_physical();
        for seed in 0..5 {
            assert_no_false_positive(&index, 24, 300, seed);
        }
    }

    #[test]
    fn no_empty_vertices_skips_the_decomposition() {
        // Fill the graph completely: `m = 0` is outside the Push and Rotate
        // regime, so we must fall back to the cheap checks only.
        let index = index_from(example_arch_json());
        let graph = LaneGraph::build(&index, &HashSet::new());
        let initial = Config::new(
            graph
                .vertices()
                .enumerate()
                .map(|(i, v)| {
                    (
                        i as u32,
                        bloqade_lanes_bytecode_core::arch::addr::LocationAddr::decode(
                            graph.location_of(v),
                        ),
                    )
                })
                .collect::<Vec<_>>(),
        )
        .expect("config");

        assert!(build_decomposition(&index, &initial, &HashSet::new()).is_none());
        assert_eq!(
            check(&index, &initial, &[], &HashSet::new()),
            Feasibility::NoObstructionFound
        );
    }
}
