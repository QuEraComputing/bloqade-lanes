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

/// Minimum number of empty vertices for the decomposition-based obstructions
/// to apply. Push and Rotate is complete only at or above this threshold, and
/// below it Wilson's parity exceptions come into play.
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
    /// Two atoms are assigned the same target location.
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
/// agent priorities. Returns `None` when there are fewer than
/// [`MIN_EMPTY_VERTICES`] empty vertices (the decomposition's guarantees do
/// not hold there), and for malformed input — an atom off the graph, or two
/// atoms sharing a location — which [`check`] reports as its own obstruction.
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

    let decomp = Decomposition::build(&graph, &occupant, empty_count);
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
/// `initial` is checked for well-formedness but otherwise ignored — there is
/// no atom to move. `blocked` locations are treated as removed from the
/// graph entirely.
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

    let mut target_vertex: HashMap<u32, VertexId> = HashMap::new();
    let mut target_seen: HashMap<u64, u32> = HashMap::new();
    for &(qubit, enc) in targets {
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
    let atom_count = initial.len();
    let Some(empty_count) = graph.len().checked_sub(atom_count) else {
        // More atoms than vertices is caught by DuplicateOccupancy above,
        // so this is unreachable in practice; stay conservative anyway.
        return Feasibility::NoObstructionFound;
    };
    if empty_count < MIN_EMPTY_VERTICES {
        // Below the Push and Rotate regime the decomposition's guarantees do
        // not hold, so we report only what the cheap checks established.
        return Feasibility::NoObstructionFound;
    }

    let decomp = Decomposition::build(&graph, &occupant, empty_count);

    // Goal containment: Proposition 1 confines an assigned agent to its
    // subgraph and planks, so a goal outside that region is unreachable.
    for (&qubit, &sub) in &decomp.assignment {
        let Some(&goal_v) = target_vertex.get(&qubit) else {
            continue;
        };
        if !decomp.contains_in_subgraph_or_planks(sub, goal_v) {
            return Feasibility::Infeasible(Obstruction::GoalOutsideSubgraph {
                qubit,
                subgraph: sub,
                goal: graph.location_of(goal_v),
            });
        }
    }

    // Proposition 2: a cyclic precedence relation proves unsolvability.
    // A target for a qubit absent from `initial` constrains nothing — there
    // is no atom to move. It is well-formedness-checked above, but excluded
    // here so a phantom goal cannot fabricate precedence edges (and, in the
    // worst case, a spurious cycle).
    let initial_qubits: HashSet<u32> = initial.iter().map(|(q, _)| q).collect();
    let unassigned_goal_vertices: HashSet<VertexId> = target_vertex
        .iter()
        .filter(|(q, _)| initial_qubits.contains(q) && !decomp.assignment.contains_key(q))
        .map(|(_, &v)| v)
        .collect();
    let edges = subgraph_priorities(&decomp, &target_vertex, &unassigned_goal_vertices);
    if let Some(cycle) = find_precedence_cycle(decomp.subgraphs.len(), &edges) {
        return Feasibility::Infeasible(Obstruction::CyclicPrecedence { subgraphs: cycle });
    }

    Feasibility::NoObstructionFound
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{example_arch_json, loc};
    use bloqade_lanes_bytecode_core::arch::types::ArchSpec;

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
        // Point the first destination back at the first source. Structural
        // validation still passes (lengths and index ranges are unchanged);
        // disjointness does not.
        let bus = &mut spec.zones[0].site_buses[0];
        bus.dst[0] = bus.src[0];
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
    fn target_for_absent_qubit_is_ignored() {
        let index = index_from(example_arch_json());
        let initial = Config::new([(0, loc(0, 0))]).expect("config");
        // Qubit 7 is not in `initial`: its target is well-formedness checked
        // but must not constrain feasibility (there is no atom to move).
        let targets = [(7u32, loc(0, 5).encode())];
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
