//! Exhaustive move generator for search.
//!
//! Port of Python's `ExhaustiveMoveGenerator`. Enumerates all valid AOD
//! rectangle move sets from a configuration: for each bus triplet, builds
//! position grids and yields every valid X×Y subset within capacity.

use std::collections::{BTreeSet, HashMap, HashSet};

use bloqade_lanes_bytecode_core::arch::addr::{LaneAddr, LocationAddr};

use crate::config::Config;
use crate::context::{MoveCandidate, SearchContext, SearchState};
use crate::graph::{MoveSet, NodeId};
use crate::lane_index::LaneIndex;
use crate::traits::MoveGenerator;

/// Exhaustive AOD-rectangle move generator.
///
/// For each `(move_type, bus_id, direction)` triplet, enumerates all valid
/// rectangular subsets of source positions within AOD capacity, and yields
/// them as [`MoveCandidate`] values.
#[derive(Debug)]
pub struct ExhaustiveGenerator {
    /// Maximum AOD X capacity (None = unlimited).
    max_x_capacity: Option<usize>,
    /// Maximum AOD Y capacity (None = unlimited).
    max_y_capacity: Option<usize>,
}

impl ExhaustiveGenerator {
    /// Create a new generator with the given AOD capacity constraints.
    pub fn new(max_x_capacity: Option<usize>, max_y_capacity: Option<usize>) -> Self {
        Self {
            max_x_capacity,
            max_y_capacity,
        }
    }

    /// Build the set of all occupied encoded locations (config qubits + blocked).
    fn occupied_set(config: &Config, blocked: &HashSet<u64>) -> HashSet<u64> {
        let mut occupied = blocked.clone();
        for (_, loc) in config.iter() {
            occupied.insert(loc.encode());
        }
        occupied
    }
}

impl MoveGenerator for ExhaustiveGenerator {
    fn generate(
        &self,
        config: &Config,
        _node_id: NodeId,
        ctx: &SearchContext,
        _state: &mut SearchState,
        out: &mut Vec<MoveCandidate>,
    ) {
        let expand_ctx = ExpandContext {
            occupied: Self::occupied_set(config, ctx.blocked),
            loc_to_qubit: config.location_to_qubit_map(),
            config,
            index: ctx.index,
            max_x_capacity: self.max_x_capacity,
            max_y_capacity: self.max_y_capacity,
        };

        for (mt, bus_id, dir) in ctx.index.bus_groups_no_zone() {
            let lanes: Vec<LaneAddr> = ctx
                .index
                .lanes_for_all_zones(mt, bus_id, dir)
                .copied()
                .collect();
            if lanes.is_empty() {
                continue;
            }

            rectangles_to_move_sets(&lanes, &expand_ctx, out);
        }
    }
}

/// Shared context for rectangle enumeration, built once per `generate()` call.
struct ExpandContext<'a> {
    occupied: HashSet<u64>,
    loc_to_qubit: HashMap<u64, u32>,
    config: &'a Config,
    index: &'a LaneIndex,
    max_x_capacity: Option<usize>,
    max_y_capacity: Option<usize>,
}

/// Per-triplet data built during rectangle enumeration.
struct TripletData {
    pos_to_info: HashMap<(u64, u64), (LocationAddr, LaneAddr)>,
    invalid_locs: HashSet<u64>,
}

/// Enumerate all valid AOD rectangles for a set of lanes and push results.
///
/// Direct port of Python's `_rectangles_to_move_sets` + `_enumerate_xy_combinations`.
fn rectangles_to_move_sets(
    lanes: &[LaneAddr],
    ctx: &ExpandContext<'_>,
    out: &mut Vec<MoveCandidate>,
) {
    let mut pos_to_info: HashMap<(u64, u64), (LocationAddr, LaneAddr)> = HashMap::new();
    let mut unique_x: BTreeSet<u64> = BTreeSet::new();
    let mut unique_y: BTreeSet<u64> = BTreeSet::new();
    let mut invalid_locs: HashSet<u64> = HashSet::new();

    for &lane in lanes {
        let (src, dst) = match ctx.index.endpoints(&lane) {
            Some(ep) => ep,
            None => continue,
        };
        let (x, y) = match ctx.index.position(src) {
            Some(p) => p,
            None => continue,
        };
        let xb = x.to_bits();
        let yb = y.to_bits();
        pos_to_info.insert((xb, yb), (src, lane));
        unique_x.insert(xb);
        unique_y.insert(yb);

        let src_enc = src.encode();
        if ctx.occupied.contains(&src_enc) && ctx.occupied.contains(&dst.encode()) {
            invalid_locs.insert(src_enc);
        }
    }

    let sorted_xs: Vec<u64> = unique_x.into_iter().collect();
    let sorted_ys: Vec<u64> = unique_y.into_iter().collect();

    let max_nx = ctx
        .max_x_capacity
        .unwrap_or(sorted_xs.len())
        .min(sorted_xs.len());
    let max_ny = ctx
        .max_y_capacity
        .unwrap_or(sorted_ys.len())
        .min(sorted_ys.len());

    let td = TripletData {
        pos_to_info,
        invalid_locs,
    };

    for nx in 1..=max_nx {
        let mut x_indices = vec![0usize; nx];
        loop {
            let x_subset: Vec<u64> = x_indices.iter().map(|&i| sorted_xs[i]).collect();

            for ny in 1..=max_ny {
                let mut y_indices = vec![0usize; ny];
                loop {
                    let y_subset: Vec<u64> = y_indices.iter().map(|&i| sorted_ys[i]).collect();

                    try_rectangle(&x_subset, &y_subset, &td, ctx, out);

                    if !next_combination(&mut y_indices, sorted_ys.len()) {
                        break;
                    }
                }
            }

            if !next_combination(&mut x_indices, sorted_xs.len()) {
                break;
            }
        }
    }
}

/// Try a single X×Y rectangle and push to `out` if valid.
fn try_rectangle(
    x_subset: &[u64],
    y_subset: &[u64],
    td: &TripletData,
    ctx: &ExpandContext<'_>,
    out: &mut Vec<MoveCandidate>,
) {
    let mut lane_addrs: Vec<LaneAddr> = Vec::new();
    let mut moves: Vec<(u32, LocationAddr)> = Vec::new();
    let mut has_atom = false;

    for &xb in x_subset {
        for &yb in y_subset {
            let Some(&(src, lane)) = td.pos_to_info.get(&(xb, yb)) else {
                return;
            };
            let src_enc = src.encode();
            if td.invalid_locs.contains(&src_enc) {
                return;
            }
            lane_addrs.push(lane);

            if let Some(&qid) = ctx.loc_to_qubit.get(&src_enc) {
                has_atom = true;
                if let Some((_, dst)) = ctx.index.endpoints(&lane) {
                    moves.push((qid, dst));
                }
            }
        }
    }

    if !has_atom {
        return;
    }

    let move_set = MoveSet::new(lane_addrs);
    let new_config = ctx.config.with_moves(&moves);
    out.push(MoveCandidate {
        move_set,
        new_config,
    });
}

/// Advance a combination of `k` indices chosen from `0..n` to the next
/// lexicographic combination. Returns `false` when exhausted.
fn next_combination(indices: &mut [usize], n: usize) -> bool {
    let k = indices.len();
    if k == 0 {
        return false;
    }
    // Find the rightmost index that can be incremented.
    let mut i = k;
    while i > 0 {
        i -= 1;
        if indices[i] < n - k + i {
            indices[i] += 1;
            // Reset all indices to the right.
            for j in (i + 1)..k {
                indices[j] = indices[j - 1] + 1;
            }
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use bloqade_lanes_bytecode_core::arch::types::ArchSpec;

    use super::*;
    use crate::heuristic::DistanceTable;
    use crate::observer::NoOpObserver;
    use crate::test_utils::{example_arch_json, loc};

    fn make_index() -> LaneIndex {
        let spec: ArchSpec = serde_json::from_str(example_arch_json()).unwrap();
        LaneIndex::new(spec)
    }

    fn make_ctx<'a>(
        index: &'a LaneIndex,
        dist_table: &'a DistanceTable,
        targets: &'a [(u32, u64)],
        blocked: &'a HashSet<u64>,
    ) -> SearchContext<'a> {
        SearchContext {
            index,
            dist_table,
            blocked,
            targets,
            cz_pairs: None,
        }
    }

    /// Helper to run the generator with default context (no blocked locations).
    fn run_generator(
        generator: &ExhaustiveGenerator,
        config: &Config,
        index: &LaneIndex,
    ) -> Vec<MoveCandidate> {
        let targets_raw: Vec<(u32, u64)> = vec![(0, loc(0, 5).encode())];
        let target_locs: Vec<u64> = vec![loc(0, 5).encode()];
        let dist_table = DistanceTable::new(&target_locs, index);
        let blocked = HashSet::new();
        let ctx = make_ctx(index, &dist_table, &targets_raw, &blocked);
        let mut state = SearchState::default();
        let mut out = Vec::new();
        generator.generate(config, NodeId(0), &ctx, &mut state, &mut out);
        out
    }

    /// Helper to run the generator with blocked locations.
    fn run_generator_blocked(
        generator: &ExhaustiveGenerator,
        config: &Config,
        index: &LaneIndex,
        blocked_locs: &[LocationAddr],
    ) -> Vec<MoveCandidate> {
        let targets_raw: Vec<(u32, u64)> = vec![(0, loc(0, 5).encode())];
        let target_locs: Vec<u64> = vec![loc(0, 5).encode()];
        let dist_table = DistanceTable::new(&target_locs, index);
        let blocked: HashSet<u64> = blocked_locs.iter().map(|l| l.encode()).collect();
        let ctx = make_ctx(index, &dist_table, &targets_raw, &blocked);
        let mut state = SearchState::default();
        let mut out = Vec::new();
        generator.generate(config, NodeId(0), &ctx, &mut state, &mut out);
        out
    }

    #[test]
    fn next_combination_basic() {
        let mut idx = vec![0, 1, 2];
        assert!(next_combination(&mut idx, 5));
        assert_eq!(idx, vec![0, 1, 3]);
        assert!(next_combination(&mut idx, 5));
        assert_eq!(idx, vec![0, 1, 4]);
        assert!(next_combination(&mut idx, 5));
        assert_eq!(idx, vec![0, 2, 3]);
    }

    #[test]
    fn next_combination_exhausted() {
        let mut idx = vec![2, 3, 4];
        assert!(!next_combination(&mut idx, 5));
    }

    #[test]
    fn next_combination_single() {
        let mut idx = vec![0];
        assert!(next_combination(&mut idx, 3));
        assert_eq!(idx, vec![1]);
        assert!(next_combination(&mut idx, 3));
        assert_eq!(idx, vec![2]);
        assert!(!next_combination(&mut idx, 3));
    }

    #[test]
    fn generate_produces_moves() {
        let index = make_index();
        let config = Config::new([(0, loc(0, 0))]).unwrap();
        let generator = ExhaustiveGenerator::new(None, None);

        let out = run_generator(&generator, &config, &index);

        // Should produce at least one move set (site bus forward moves qubit to site 5).
        assert!(!out.is_empty());

        // At least one move should place qubit 0 at site 5 (forward site bus).
        let has_site5 = out
            .iter()
            .any(|c| c.new_config.location_of(0) == Some(loc(0, 5)));
        assert!(
            has_site5,
            "should have a move to site 5 via site bus forward"
        );
    }

    #[test]
    fn generate_respects_blocked() {
        let index = make_index();
        // Qubit 0 at word 0, site 0. Block site 5 (the forward destination).
        let config = Config::new([(0, loc(0, 0))]).unwrap();
        let generator = ExhaustiveGenerator::new(None, None);

        let out = run_generator_blocked(&generator, &config, &index, &[loc(0, 5)]);

        // No move should place qubit 0 at blocked site 5.
        let has_site5 = out
            .iter()
            .any(|c| c.new_config.location_of(0) == Some(loc(0, 5)));
        assert!(!has_site5, "blocked destination should be excluded");
    }

    #[test]
    fn generate_no_moves_when_no_atoms() {
        let index = make_index();
        // Empty config -> no atoms -> no moves.
        let config = Config::new(std::iter::empty::<(u32, LocationAddr)>()).unwrap();
        let generator = ExhaustiveGenerator::new(None, None);

        let out = run_generator(&generator, &config, &index);
        assert!(out.is_empty());
    }

    #[test]
    fn generate_collision_prefilter() {
        let index = make_index();
        // Qubit 0 at site 0, qubit 1 at site 5 (destination of site 0 forward).
        // The pre-filter should mark site 0 as invalid for forward moves
        // (src occupied, dst occupied).
        let config = Config::new([(0, loc(0, 0)), (1, loc(0, 5))]).unwrap();
        let generator = ExhaustiveGenerator::new(None, None);

        let out = run_generator(&generator, &config, &index);

        // No forward site bus move should move qubit 0 to site 5 (collision).
        let collision = out
            .iter()
            .any(|c| c.new_config.location_of(0) == Some(loc(0, 5)));
        assert!(!collision, "collision should be pre-filtered");
    }

    #[test]
    fn generate_parallel_moves() {
        let index = make_index();
        // Two qubits at site bus source positions in same word.
        let config = Config::new([(0, loc(0, 0)), (1, loc(0, 1))]).unwrap();
        let generator = ExhaustiveGenerator::new(None, None);

        let out = run_generator(&generator, &config, &index);

        // Should have a move set that moves both qubits simultaneously.
        let has_parallel = out.iter().any(|c| {
            c.move_set.len() >= 2
                && c.new_config.location_of(0) == Some(loc(0, 5))
                && c.new_config.location_of(1) == Some(loc(0, 6))
        });
        assert!(
            has_parallel,
            "should generate parallel moves for multiple qubits"
        );
    }

    #[test]
    fn generate_with_search_finds_solution() {
        use crate::cost::UniformCost;
        use crate::frontier::{self, PriorityFrontier};
        use crate::goals::AllAtTarget;
        use crate::heuristic::HopDistanceHeuristic;
        use crate::scorers::DistanceScorer;

        let index = make_index();
        let config = Config::new([(0, loc(0, 0))]).unwrap();
        let target_loc = loc(0, 5);

        let targets = vec![(0u32, target_loc)];
        let target_encoded: Vec<(u32, u64)> =
            targets.iter().map(|&(q, l)| (q, l.encode())).collect();
        let target_locs: Vec<u64> = targets.iter().map(|&(_, l)| l.encode()).collect();
        let dist_table = DistanceTable::new(&target_locs, &index);
        let blocked = HashSet::new();
        let ctx = make_ctx(&index, &dist_table, &target_encoded, &blocked);

        let h = HopDistanceHeuristic::new(targets.clone(), &dist_table);
        let generator = ExhaustiveGenerator::new(None, None);
        let scorer = DistanceScorer;
        let cost = UniformCost;
        let goal = AllAtTarget::new(&target_encoded);
        let mut f = PriorityFrontier::astar(|cfg: &Config| h.estimate_max(cfg), 1.0);

        let result = frontier::run_search(
            config,
            &generator,
            &scorer,
            &cost,
            &goal,
            &mut f,
            &ctx,
            &mut SearchState::default(),
            &mut NoOpObserver,
            Some(100),
            None,
        );

        assert!(result.goal.is_some());
        let path = result.solution_path().unwrap();
        // Site 0 -> site 5 is one site bus forward move.
        assert_eq!(path.len(), 1);
    }
}
