//! Exhaustive move generator for A* search.
//!
//! Port of Python's `ExhaustiveMoveGenerator`. Enumerates all valid AOD
//! rectangle move sets from a configuration: for each bus triplet, builds
//! position grids and yields every valid X×Y subset within capacity.

use std::collections::{BTreeSet, HashMap, HashSet};

use bloqade_lanes_bytecode_core::arch::addr::{LaneAddr, LocationAddr};

use crate::astar::Expander;
use crate::config::Config;
use crate::graph::MoveSet;
use crate::lane_index::LaneIndex;

/// Exhaustive AOD-rectangle move generator.
///
/// For each `(move_type, bus_id, direction)` triplet, enumerates all valid
/// rectangular subsets of source positions within AOD capacity, and yields
/// them as move sets.
pub struct ExhaustiveExpander<'a> {
    index: &'a LaneIndex,
    /// Encoded locations that are blocked (external obstacles).
    blocked: HashSet<u32>,
    /// Maximum AOD X capacity (None = unlimited).
    max_x_capacity: Option<usize>,
    /// Maximum AOD Y capacity (None = unlimited).
    max_y_capacity: Option<usize>,
}

impl<'a> ExhaustiveExpander<'a> {
    /// Create a new expander.
    pub fn new(
        index: &'a LaneIndex,
        blocked: impl IntoIterator<Item = LocationAddr>,
        max_x_capacity: Option<usize>,
        max_y_capacity: Option<usize>,
    ) -> Self {
        Self {
            index,
            blocked: blocked.into_iter().map(|l| l.encode()).collect(),
            max_x_capacity,
            max_y_capacity,
        }
    }

    /// Build the set of all occupied encoded locations (config qubits + blocked).
    fn occupied_set(&self, config: &Config) -> HashSet<u32> {
        let mut occupied = self.blocked.clone();
        for (_, loc) in config.iter() {
            occupied.insert(loc.encode());
        }
        occupied
    }
}

impl Expander for ExhaustiveExpander<'_> {
    fn expand(&self, config: &Config, out: &mut Vec<(MoveSet, Config, f64)>) {
        let occupied = self.occupied_set(config);

        for (mt, bus_id, dir) in self.index.triplets() {
            let lanes = self.index.lanes_for(mt, bus_id, dir);
            if lanes.is_empty() {
                continue;
            }

            // Collect source info for this triplet.
            rectangles_to_move_sets(
                lanes,
                &occupied,
                config,
                self.index,
                self.max_x_capacity,
                self.max_y_capacity,
                out,
            );
        }
    }
}

/// Enumerate all valid AOD rectangles for a set of lanes and push results.
///
/// Direct port of Python's `_rectangles_to_move_sets` + `_enumerate_xy_combinations`.
fn rectangles_to_move_sets(
    lanes: &[LaneAddr],
    occupied: &HashSet<u32>,
    config: &Config,
    index: &LaneIndex,
    max_x_cap: Option<usize>,
    max_y_cap: Option<usize>,
    out: &mut Vec<(MoveSet, Config, f64)>,
) {
    // Build position → (location, lane) and track unique X/Y.
    // Use BTreeSet<u64> for sorted unique positions (f64 bits for exact comparison).
    let mut pos_to_info: HashMap<(u64, u64), (LocationAddr, LaneAddr)> = HashMap::new();
    let mut unique_x: BTreeSet<u64> = BTreeSet::new();
    let mut unique_y: BTreeSet<u64> = BTreeSet::new();
    let mut invalid_locs: HashSet<u32> = HashSet::new();

    for &lane in lanes {
        let (src, dst) = match index.endpoints(&lane) {
            Some(ep) => ep,
            None => continue,
        };
        let (x, y) = match index.position(src) {
            Some(p) => p,
            None => continue,
        };
        let xb = x.to_bits();
        let yb = y.to_bits();
        pos_to_info.insert((xb, yb), (src, lane));
        unique_x.insert(xb);
        unique_y.insert(yb);

        // Pre-filter: if src is occupied and dst is also occupied, mark invalid.
        let src_enc = src.encode();
        if occupied.contains(&src_enc) && occupied.contains(&dst.encode()) {
            invalid_locs.insert(src_enc);
        }
    }

    let sorted_xs: Vec<u64> = unique_x.into_iter().collect();
    let sorted_ys: Vec<u64> = unique_y.into_iter().collect();

    let max_nx = max_x_cap.unwrap_or(sorted_xs.len()).min(sorted_xs.len());
    let max_ny = max_y_cap.unwrap_or(sorted_ys.len()).min(sorted_ys.len());

    // Enumerate all nx × ny combinations.
    for nx in 1..=max_nx {
        let mut x_indices = vec![0usize; nx];
        loop {
            // Initialize x_subset from current x_indices.
            let x_subset: Vec<u64> = x_indices.iter().map(|&i| sorted_xs[i]).collect();

            for ny in 1..=max_ny {
                let mut y_indices = vec![0usize; ny];
                loop {
                    let y_subset: Vec<u64> = y_indices.iter().map(|&i| sorted_ys[i]).collect();

                    // Check this rectangle.
                    try_rectangle(
                        &x_subset,
                        &y_subset,
                        &pos_to_info,
                        &invalid_locs,
                        config,
                        index,
                        out,
                    );

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
    pos_to_info: &HashMap<(u64, u64), (LocationAddr, LaneAddr)>,
    invalid_locs: &HashSet<u32>,
    config: &Config,
    index: &LaneIndex,
    out: &mut Vec<(MoveSet, Config, f64)>,
) {
    let mut lane_addrs: Vec<LaneAddr> = Vec::new();
    let mut moves: Vec<(u32, LocationAddr)> = Vec::new();
    let mut has_atom = false;

    for &xb in x_subset {
        for &yb in y_subset {
            let Some(&(src, lane)) = pos_to_info.get(&(xb, yb)) else {
                return; // Position not in grid → invalid rectangle.
            };
            let src_enc = src.encode();
            if invalid_locs.contains(&src_enc) {
                return; // Source with occupied destination → invalid.
            }
            lane_addrs.push(lane);

            // If an atom is at this source, record the move.
            if let Some(qid) = config.qubit_at(src) {
                has_atom = true;
                if let Some((_, dst)) = index.endpoints(&lane) {
                    moves.push((qid, dst));
                }
            }
        }
    }

    if !has_atom {
        return; // No atoms in this rectangle.
    }

    let move_set = MoveSet::new(lane_addrs);
    let new_config = config.with_moves(&moves);
    out.push((move_set, new_config, 1.0));
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
    use super::*;
    use bloqade_lanes_bytecode_core::arch::types::ArchSpec;

    fn example_arch_json() -> &'static str {
        r#"{
            "version": "2.0",
            "geometry": {
                "sites_per_word": 10,
                "words": [
                    {
                        "positions": { "x_start": 1.0, "y_start": 2.5, "x_spacing": [2.0, 2.0, 2.0, 2.0], "y_spacing": [2.5] },
                        "site_indices": [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [0, 1], [1, 1], [2, 1], [3, 1], [4, 1]]
                    },
                    {
                        "positions": { "x_start": 1.0, "y_start": 12.5, "x_spacing": [2.0, 2.0, 2.0, 2.0], "y_spacing": [2.5] },
                        "site_indices": [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [0, 1], [1, 1], [2, 1], [3, 1], [4, 1]]
                    }
                ]
            },
            "buses": {
                "site_buses": [
                    { "src": [0, 1, 2, 3, 4], "dst": [5, 6, 7, 8, 9] }
                ],
                "word_buses": [
                    { "src": [0], "dst": [1] }
                ]
            },
            "words_with_site_buses": [0, 1],
            "sites_with_word_buses": [5, 6, 7, 8, 9],
            "zones": [
                { "words": [0, 1] }
            ],
            "entangling_zones": [[[0, 1]]],
            "blockade_radius": 2.0,
            "measurement_mode_zones": [0]
        }"#
    }

    fn make_index() -> LaneIndex {
        let spec: ArchSpec = serde_json::from_str(example_arch_json()).unwrap();
        LaneIndex::new(spec)
    }

    fn loc(word: u32, site: u32) -> LocationAddr {
        LocationAddr {
            word_id: word,
            site_id: site,
        }
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
    fn expand_produces_moves() {
        let index = make_index();
        // Qubit 0 at word 0, site 0 (a site bus source position).
        let config = Config::new([(0, loc(0, 0))]);
        let expander = ExhaustiveExpander::new(&index, std::iter::empty(), None, None);

        let mut out = Vec::new();
        expander.expand(&config, &mut out);

        // Should produce at least one move set (site bus forward moves qubit to site 5).
        assert!(!out.is_empty());

        // At least one move should place qubit 0 at site 5 (forward site bus).
        let has_site5 = out
            .iter()
            .any(|(_, cfg, _)| cfg.location_of(0) == Some(loc(0, 5)));
        assert!(
            has_site5,
            "should have a move to site 5 via site bus forward"
        );
    }

    #[test]
    fn expand_respects_blocked() {
        let index = make_index();
        // Qubit 0 at word 0, site 0. Block site 5 (the forward destination).
        let config = Config::new([(0, loc(0, 0))]);
        let expander = ExhaustiveExpander::new(&index, [loc(0, 5)].into_iter(), None, None);

        let mut out = Vec::new();
        expander.expand(&config, &mut out);

        // No move should place qubit 0 at blocked site 5.
        let has_site5 = out
            .iter()
            .any(|(_, cfg, _)| cfg.location_of(0) == Some(loc(0, 5)));
        assert!(!has_site5, "blocked destination should be excluded");
    }

    #[test]
    fn expand_no_moves_when_no_atoms() {
        let index = make_index();
        // Empty config → no atoms → no moves.
        let config = Config::new(std::iter::empty::<(u32, LocationAddr)>());
        let expander = ExhaustiveExpander::new(&index, std::iter::empty(), None, None);

        let mut out = Vec::new();
        expander.expand(&config, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn expand_collision_prefilter() {
        let index = make_index();
        // Qubit 0 at site 0, qubit 1 at site 5 (destination of site 0 forward).
        // The pre-filter should mark site 0 as invalid for forward moves
        // (src occupied, dst occupied).
        let config = Config::new([(0, loc(0, 0)), (1, loc(0, 5))]);
        let expander = ExhaustiveExpander::new(&index, std::iter::empty(), None, None);

        let mut out = Vec::new();
        expander.expand(&config, &mut out);

        // No forward site bus move should move qubit 0 to site 5 (collision).
        let collision = out
            .iter()
            .any(|(_, cfg, _)| cfg.location_of(0) == Some(loc(0, 5)));
        assert!(!collision, "collision should be pre-filtered");
    }

    #[test]
    fn expand_parallel_moves() {
        let index = make_index();
        // Two qubits at site bus source positions in same word.
        let config = Config::new([(0, loc(0, 0)), (1, loc(0, 1))]);
        let expander = ExhaustiveExpander::new(&index, std::iter::empty(), None, None);

        let mut out = Vec::new();
        expander.expand(&config, &mut out);

        // Should have a move set that moves both qubits simultaneously.
        let has_parallel = out.iter().any(|(ms, cfg, _)| {
            ms.len() >= 2
                && cfg.location_of(0) == Some(loc(0, 5))
                && cfg.location_of(1) == Some(loc(0, 6))
        });
        assert!(
            has_parallel,
            "should generate parallel moves for multiple qubits"
        );
    }

    #[test]
    fn expand_with_astar_finds_solution() {
        use crate::astar::astar;

        let index = make_index();
        // Qubit 0 at word 0 site 0, target: word 0 site 5.
        let config = Config::new([(0, loc(0, 0))]);
        let target_loc = loc(0, 5);

        let expander = ExhaustiveExpander::new(&index, std::iter::empty(), None, None);
        let result = astar(
            config,
            |cfg| cfg.location_of(0) == Some(target_loc),
            |cfg| {
                if cfg.location_of(0) == Some(target_loc) {
                    0.0
                } else {
                    1.0
                }
            },
            &expander,
            Some(100),
        );

        assert!(result.goal.is_some());
        let path = result.solution_path().unwrap();
        // Site 0 → site 5 is one site bus forward move.
        assert_eq!(path.len(), 1);
    }
}
