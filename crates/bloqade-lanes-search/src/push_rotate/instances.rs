//! Deterministic routing-instance generation for benchmarks.
//!
//! Instances are built by **replaying legal single-atom moves** from a random
//! start placement: scatter `k` atoms, then repeatedly slide one into an empty
//! neighbour. The end placement becomes the target.
//!
//! This matters for the benchmark's headline metric. If targets were sampled
//! at random, some would be genuinely unreachable, and a router "failing" them
//! would be correct behaviour indistinguishable from a defect — so a success
//! *rate* would mean nothing. Generating by replay guarantees every instance
//! has a solution, which makes any failure a real deficiency of the router.
//!
//! Single-atom moves are the right generator because a one-lane `MoveSet` is
//! always a legal AOD operation, so every configuration reached this way is
//! reachable on hardware.

use std::collections::HashMap;

use crate::primitives::lane_index::LaneIndex;
use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;
use bloqade_lanes_bytecode_core::arch::types::ArchSpec;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// One routing problem: where the atoms start, where they must end up.
#[derive(Debug, Clone)]
pub struct Instance {
    /// Human-readable id, e.g. `physical/k8/seed3`.
    pub id: String,
    /// Architecture this instance is posed on.
    pub arch_json: &'static str,
    /// `(qubit, encoded location)` start placement.
    pub initial: Vec<(u32, u64)>,
    /// `(qubit, encoded location)` target placement.
    pub target: Vec<(u32, u64)>,
    /// Number of single-atom slides used to build the target. An upper bound
    /// on the optimal solution length, and a rough difficulty dial.
    pub walk_len: usize,
}

/// Undirected adjacency over lane endpoints, keyed by encoded location.
fn adjacency(index: &LaneIndex) -> (Vec<u64>, HashMap<u64, Vec<u64>>) {
    let mut ids: Vec<u64> = Vec::new();
    let mut idx: HashMap<u64, usize> = HashMap::new();
    let mut adj: Vec<Vec<usize>> = Vec::new();

    let mut intern = |loc: u64, ids: &mut Vec<u64>, adj: &mut Vec<Vec<usize>>| -> usize {
        *idx.entry(loc).or_insert_with(|| {
            ids.push(loc);
            adj.push(Vec::new());
            ids.len() - 1
        })
    };

    for (mt, bus_id, zone_id, dir) in index.bus_groups() {
        for lane in index.lanes_for(mt, bus_id, zone_id, dir) {
            let Some((src, dst)) = index.endpoints(lane) else {
                continue;
            };
            let a = intern(src.encode(), &mut ids, &mut adj);
            let b = intern(dst.encode(), &mut ids, &mut adj);
            if a != b {
                adj[a].push(b);
                adj[b].push(a);
            }
        }
    }
    for list in &mut adj {
        list.sort_unstable();
        list.dedup();
    }

    // Sort by encoded location before returning. `LaneIndex::bus_groups`
    // iterates a `HashMap`, so the order locations are discovered in varies
    // between processes — and since the generator indexes into this vector
    // with a seeded RNG, unsorted order would make the same seed produce a
    // *different instance on every run*, which would silently invalidate any
    // comparison across runs.
    let mut order: Vec<usize> = (0..ids.len()).collect();
    order.sort_unstable_by_key(|&i| ids[i]);

    // Adjacency is keyed and valued by *encoded location*, not by index, so
    // it stays correct independently of the vector ordering above. Each
    // neighbour list is re-sorted by location for the same reason the vertex
    // list is: the walk picks a neighbour by index, so index order determined
    // by HashMap discovery order would leak nondeterminism back in.
    let by_loc: HashMap<u64, Vec<u64>> = ids
        .iter()
        .enumerate()
        .map(|(i, &l)| {
            let mut nbrs: Vec<u64> = adj[i].iter().map(|&j| ids[j]).collect();
            nbrs.sort_unstable();
            (l, nbrs)
        })
        .collect();
    let sorted_ids: Vec<u64> = order.into_iter().map(|i| ids[i]).collect();
    let _ = idx;
    (sorted_ids, by_loc)
}

/// Build one instance with `n_atoms` atoms and `walk_len` slides.
///
/// Returns `None` if the architecture has too few locations to place the
/// atoms and still leave room to move.
pub fn generate(
    id: String,
    arch_json: &'static str,
    n_atoms: usize,
    walk_len: usize,
    seed: u64,
) -> Option<Instance> {
    let spec: ArchSpec = serde_json::from_str(arch_json).ok()?;
    let index = LaneIndex::new(spec);
    let (locations, adj) = adjacency(&index);
    if locations.len() < n_atoms + 2 {
        return None;
    }

    let mut rng = SmallRng::seed_from_u64(seed);
    let mut occupant: HashMap<u64, u32> = HashMap::new();
    let mut position: Vec<u64> = Vec::with_capacity(n_atoms);

    while position.len() < n_atoms {
        let loc = locations[rng.random_range(0..locations.len())];
        if occupant.contains_key(&loc) || adj.get(&loc).is_none_or(|a| a.is_empty()) {
            continue;
        }
        occupant.insert(loc, position.len() as u32);
        position.push(loc);
    }

    let initial: Vec<(u32, u64)> = position
        .iter()
        .enumerate()
        .map(|(q, &l)| (q as u32, l))
        .collect();

    for _ in 0..walk_len {
        let q = rng.random_range(0..n_atoms);
        let from = position[q];
        let empty: Vec<u64> = adj[&from]
            .iter()
            .copied()
            .filter(|l| !occupant.contains_key(l))
            .collect();
        if empty.is_empty() {
            continue;
        }
        let to = empty[rng.random_range(0..empty.len())];
        occupant.remove(&from);
        occupant.insert(to, q as u32);
        position[q] = to;
    }

    let target: Vec<(u32, u64)> = position
        .iter()
        .enumerate()
        .map(|(q, &l)| (q as u32, l))
        .collect();

    Some(Instance {
        id,
        arch_json,
        initial,
        target,
        walk_len,
    })
}

/// Decode an encoded location. Convenience for callers building reports.
pub fn decode(loc: u64) -> LocationAddr {
    LocationAddr::decode(loc)
}
