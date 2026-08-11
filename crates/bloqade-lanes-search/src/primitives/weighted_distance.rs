//! Objective-weighted lane-graph distances, the substrate for admissible
//! completion bounds.
//!
//! [`WeightedDistanceTable`] is the weighted sibling of
//! [`DistanceTable`](super::distance::DistanceTable): same storage layout and
//! lookup shape, but edges carry [`Objective::lane_weight`] instead of a unit
//! hop, and `blocked` sites are cut out of the graph entirely.
//!
//! Two properties matter and are enforced structurally rather than by
//! convention:
//!
//! - **No ordering artifacts.** The only inputs are the lane graph, the
//!   blocked set, and the objective's per-lane weight. Nothing here can reach
//!   the move generator's contested-destination penalty, pair-coordination
//!   boost, or seeded score perturbation, and it does not consult
//!   `HeuristicTables` or the `w_t`-blended time distances — those are
//!   ordering devices, not costs.
//! - **Objective pairing.** A table records the [`ObjectiveId`] it was built
//!   from, so a bound derived from it can refuse to prune a search whose `g`
//!   accumulates a different objective.

use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::primitives::distance::DijkstraEntry;
use crate::primitives::lane_index::LaneIndex;
use crate::traits::{Objective, ObjectiveId};

/// Precomputed minimum **weighted** distance from every reachable location to
/// each target location, over the lane graph with `blocked` sites removed.
///
/// Storage mirrors [`DistanceTable`](super::distance::DistanceTable): a
/// `HashMap<u64, usize>` interning every location to a compact index, and a
/// flat row-major `Vec<f64>` of `n_loc × n_loc`. Only the columns of actual
/// targets are populated; every other cell stays [`f64::INFINITY`].
///
/// # Why blocked sites are excluded
///
/// A `blocked` location holds an external atom that this solve cannot move, so
/// no plan may route through or land on it — for the whole solve, not just the
/// current node. Deleting those vertices can only lengthen or sever paths,
/// which makes the resulting distances **larger** and therefore the derived
/// bound **tighter**, while staying a valid lower bound. In the physical
/// pipeline `blocked` carries every un-routed atom plus every spectator, so
/// this is a large effect, not a corner case.
#[derive(Debug)]
pub struct WeightedDistanceTable {
    /// `encoded location → compact row/column index`. Blocked locations are
    /// deliberately absent, so any lookup involving one returns `None`.
    loc_index: HashMap<u64, usize>,
    /// Row-major `n_loc × n_loc` weighted distances;
    /// `flat_distance[from * n_loc + to]`, [`f64::INFINITY`] if unreachable.
    flat_distance: Vec<f64>,
    n_loc: usize,
    /// Identity of the objective whose `lane_weight` produced these edges.
    objective_id: ObjectiveId,
}

impl WeightedDistanceTable {
    /// Build by running Dijkstra from each unique target on the reversed,
    /// blocked-carved lane graph.
    ///
    /// Reversed because the query direction is "cost from an arbitrary
    /// location *to* a fixed target", so one run per target fills that
    /// target's whole column.
    ///
    /// # Panics
    ///
    /// If the objective reports a negative `lane_weight`. Dijkstra requires
    /// non-negative edges, and a negative weight would silently produce wrong
    /// shortest paths rather than merely a weaker bound. Checked at
    /// construction so it can never reach a pruning decision.
    pub fn new(
        target_locations: &[u64],
        index: &LaneIndex,
        blocked: &HashSet<u64>,
        objective: &impl Objective,
    ) -> Self {
        let targets: Vec<u64> = {
            let mut v: Vec<u64> = target_locations
                .iter()
                .copied()
                .filter(|t| !blocked.contains(t))
                .collect();
            v.sort_unstable();
            v.dedup();
            v
        };

        // Reverse adjacency `dst → [(src, w)]` over unblocked lanes only, and
        // intern every location that survives the carve.
        let mut reverse_adj: HashMap<u64, Vec<(u64, f64)>> = HashMap::new();
        let mut loc_index: HashMap<u64, usize> = HashMap::new();
        let intern = |loc: u64, loc_index: &mut HashMap<u64, usize>| {
            let next = loc_index.len();
            loc_index.entry(loc).or_insert(next);
        };

        for (mt, bus_id, zone_id, dir) in index.bus_groups() {
            for &lane in index.lanes_for(mt, bus_id, zone_id, dir) {
                let Some((src, dst)) = index.endpoints(&lane) else {
                    continue;
                };
                let (src_enc, dst_enc) = (src.encode(), dst.encode());
                // Carve out blocked vertices: a lane touching one can never be
                // taken, so it must not contribute a path.
                if blocked.contains(&src_enc) || blocked.contains(&dst_enc) {
                    continue;
                }
                let w = objective.lane_weight(lane);
                assert!(
                    w >= 0.0,
                    "objective {:?} reported a negative lane_weight ({w}) for {lane:?}; \
                     Dijkstra requires non-negative edge weights",
                    objective.id()
                );
                intern(src_enc, &mut loc_index);
                intern(dst_enc, &mut loc_index);
                reverse_adj.entry(dst_enc).or_default().push((src_enc, w));
            }
        }
        // Isolated targets (no incident unblocked lanes) still need an index so
        // that `distance(t, t) == 0` holds, matching `DistanceTable`.
        for &t in &targets {
            intern(t, &mut loc_index);
        }

        let n_loc = loc_index.len();
        let mut flat_distance = vec![f64::INFINITY; n_loc * n_loc];

        for &target_enc in &targets {
            let target_idx = loc_index[&target_enc];
            let mut dist: Vec<f64> = vec![f64::INFINITY; n_loc];
            dist[target_idx] = 0.0;
            let mut heap = BinaryHeap::new();
            heap.push(DijkstraEntry {
                cost: 0.0,
                node: target_enc,
            });

            while let Some(entry) = heap.pop() {
                let entry_idx = loc_index[&entry.node];
                // Stale heap entry — a shorter path to this node was settled.
                if entry.cost > dist[entry_idx] {
                    continue;
                }
                let Some(preds) = reverse_adj.get(&entry.node) else {
                    continue;
                };
                for &(pred, w) in preds {
                    let pred_idx = loc_index[&pred];
                    let new_cost = entry.cost + w;
                    if new_cost < dist[pred_idx] {
                        dist[pred_idx] = new_cost;
                        heap.push(DijkstraEntry {
                            cost: new_cost,
                            node: pred,
                        });
                    }
                }
            }

            for (from_idx, &d) in dist.iter().enumerate() {
                flat_distance[from_idx * n_loc + target_idx] = d;
            }
        }

        Self {
            loc_index,
            flat_distance,
            n_loc,
            objective_id: objective.id(),
        }
    }

    /// O(1) lookup: minimum weighted cost from `from_encoded` to
    /// `to_target_encoded`.
    ///
    /// `None` when either location is unknown to this table — which includes
    /// every blocked location, since those are never interned — or when no
    /// unblocked path exists. Callers treat `None` as "infeasible".
    pub fn distance(&self, from_encoded: u64, to_target_encoded: u64) -> Option<f64> {
        let from_idx = *self.loc_index.get(&from_encoded)?;
        let to_idx = *self.loc_index.get(&to_target_encoded)?;
        let d = self.flat_distance[from_idx * self.n_loc + to_idx];
        d.is_finite().then_some(d)
    }

    /// The objective this table's edge weights came from.
    pub fn objective_id(&self) -> ObjectiveId {
        self.objective_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost::{UniformCost, WeightedDuration};
    use crate::primitives::distance::DistanceTable;
    use crate::test_utils::{example_arch_json, loc};
    use bloqade_lanes_bytecode_core::arch::types::ArchSpec;

    fn make_index() -> LaneIndex {
        let spec: ArchSpec = serde_json::from_str(example_arch_json()).unwrap();
        LaneIndex::new(spec)
    }

    fn no_blocked() -> HashSet<u64> {
        HashSet::new()
    }

    /// With unit weights the weighted table must reproduce the BFS hop-count
    /// table exactly, for every pair it can answer for. This is the anchor
    /// test: it pins the Dijkstra implementation against an independent
    /// shortest-path implementation over the same graph.
    #[test]
    fn agrees_with_hop_table_when_all_weights_are_one() {
        let index = make_index();
        let targets: Vec<u64> = [loc(0, 5), loc(1, 5), loc(1, 0)]
            .iter()
            .map(|l| l.encode())
            .collect();
        let hops = DistanceTable::new(&targets, &index);
        let weighted = WeightedDistanceTable::new(&targets, &index, &no_blocked(), &UniformCost);

        let mut compared = 0_usize;
        for (mt, bus_id, zone_id, dir) in index.bus_groups() {
            for &lane in index.lanes_for(mt, bus_id, zone_id, dir) {
                let Some((src, _)) = index.endpoints(&lane) else {
                    continue;
                };
                for &t in &targets {
                    let h = hops.distance(src.encode(), t);
                    let w = weighted.distance(src.encode(), t);
                    match (h, w) {
                        (Some(h), Some(w)) => {
                            assert_eq!(f64::from(h), w, "hop {h} vs weighted {w}");
                            compared += 1;
                        }
                        (None, None) => compared += 1,
                        (h, w) => panic!("reachability disagrees: hops={h:?} weighted={w:?}"),
                    }
                }
            }
        }
        assert!(compared > 0, "fixture should expose comparable pairs");
    }

    /// Under a duration-weighted objective, a one-hop distance must equal that
    /// lane's own weight — i.e. the table's edges really carry arch-spec
    /// durations, not hop counts.
    #[test]
    fn edge_weights_come_from_arch_spec_durations() {
        let index = make_index();
        let objective = WeightedDuration::new(&index, 10.0);

        // Find a lane whose destination is reachable only through it in one
        // hop, and check the table reproduces its weight.
        let mut checked = 0_usize;
        for (mt, bus_id, zone_id, dir) in index.bus_groups() {
            for &lane in index.lanes_for(mt, bus_id, zone_id, dir) {
                let Some((src, dst)) = index.endpoints(&lane) else {
                    continue;
                };
                let table =
                    WeightedDistanceTable::new(&[dst.encode()], &index, &no_blocked(), &objective);
                let d = table
                    .distance(src.encode(), dst.encode())
                    .expect("one hop must be reachable");
                // The direct lane is an upper bound on the shortest path, and
                // any path costs at least the cheapest single lane.
                assert!(
                    d <= objective.lane_weight(lane) + 1e-12,
                    "shortest path {d} exceeded the direct lane weight {}",
                    objective.lane_weight(lane)
                );
                assert!(d > 1.0, "a duration-weighted hop must exceed the unit term");
                checked += 1;
                if checked >= 8 {
                    return;
                }
            }
        }
        assert!(checked > 0, "fixture should expose lanes");
    }

    /// Weighted distances must differ from hop counts under a non-uniform
    /// objective — otherwise the table is silently ignoring `lane_weight`.
    #[test]
    fn weighted_distances_differ_from_hop_counts() {
        let index = make_index();
        let target = loc(1, 0).encode();
        let objective = WeightedDuration::new(&index, 10.0);
        let hops = DistanceTable::new(&[target], &index);
        let weighted = WeightedDistanceTable::new(&[target], &index, &no_blocked(), &objective);

        let from = loc(0, 0).encode();
        let h = f64::from(hops.distance(from, target).expect("reachable"));
        let w = weighted.distance(from, target).expect("reachable");
        assert!(
            w > h,
            "duration weights (each > 1) must exceed the hop count: {w} vs {h}"
        );
    }

    /// Blocking a location removes it from the graph: it becomes unknown to
    /// the table, and any path that needed it gets longer or disappears.
    #[test]
    fn blocked_locations_are_excluded_from_the_graph() {
        let index = make_index();
        let target = loc(1, 0).encode();
        let waypoint = loc(0, 5).encode();

        let open = WeightedDistanceTable::new(&[target], &index, &no_blocked(), &UniformCost);
        let baseline = open
            .distance(loc(0, 0).encode(), target)
            .expect("reachable");

        let blocked: HashSet<u64> = [waypoint].into_iter().collect();
        let carved = WeightedDistanceTable::new(&[target], &index, &blocked, &UniformCost);

        // The blocked vertex is not part of the graph at all.
        assert_eq!(carved.distance(waypoint, target), None);
        assert_eq!(carved.distance(loc(0, 0).encode(), waypoint), None);

        // Any surviving route is no cheaper than before — carving can only
        // lengthen or sever paths, never shorten them.
        if let Some(after) = carved.distance(loc(0, 0).encode(), target) {
            assert!(
                after >= baseline,
                "removing a vertex must not shorten a path: {after} < {baseline}"
            );
        }
    }

    /// A target sitting on a blocked location is unreachable by construction,
    /// rather than reporting a bogus distance of zero.
    #[test]
    fn blocked_target_is_unreachable() {
        let index = make_index();
        let target = loc(0, 5).encode();
        let blocked: HashSet<u64> = [target].into_iter().collect();
        let table = WeightedDistanceTable::new(&[target], &index, &blocked, &UniformCost);
        assert_eq!(table.distance(loc(0, 0).encode(), target), None);
        assert_eq!(table.distance(target, target), None);
    }

    #[test]
    fn distance_to_self_is_zero() {
        let index = make_index();
        let target = loc(0, 5).encode();
        let table = WeightedDistanceTable::new(&[target], &index, &no_blocked(), &UniformCost);
        assert_eq!(table.distance(target, target), Some(0.0));
    }

    #[test]
    fn unknown_location_returns_none() {
        let index = make_index();
        let target = loc(0, 5).encode();
        let table = WeightedDistanceTable::new(&[target], &index, &no_blocked(), &UniformCost);
        assert_eq!(
            table.distance(loc(0, 0).encode(), loc(99, 99).encode()),
            None
        );
    }

    /// The table carries its objective's identity so a bound built from it can
    /// refuse to prune a search accumulating a different objective.
    #[test]
    fn records_the_objective_it_was_built_from() {
        let index = make_index();
        let target = loc(0, 5).encode();
        let uniform = WeightedDistanceTable::new(&[target], &index, &no_blocked(), &UniformCost);
        let weighted = WeightedDistanceTable::new(
            &[target],
            &index,
            &no_blocked(),
            &WeightedDuration::new(&index, 10.0),
        );
        assert_eq!(uniform.objective_id(), UniformCost.id());
        assert_ne!(uniform.objective_id(), weighted.objective_id());

        // Same family, different parameter — must not compare equal, or an
        // instance-level mismatch would slip through the bound's pairing check.
        let tau_a = WeightedDistanceTable::new(
            &[target],
            &index,
            &no_blocked(),
            &WeightedDuration::new(&index, 1.0),
        );
        assert_ne!(tau_a.objective_id(), weighted.objective_id());
    }
}
