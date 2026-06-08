//! Two-phase AOD grid construction for the heuristic expander.
//!
//! Ports the Python `BusContext.build_aod_grids()` algorithm: a greedy
//! sequential pass forms initial rectangular clusters, then an iterative
//! merge pass combines compatible clusters into larger rectangles.

use std::collections::{BTreeSet, HashMap, HashSet};

use bloqade_lanes_bytecode_core::arch::addr::{Direction, LaneAddr, MoveType};

use crate::primitives::lane_index::LaneIndex;

/// A cluster represented by its X and Y coordinate sets.
/// The rectangle covers the Cartesian product X × Y.
/// Coordinates are stored as `f64::to_bits()` for cheap equality.
type Cluster = (BTreeSet<u64>, BTreeSet<u64>);

/// Context for building AOD-compatible rectangular grids on one bus group.
///
/// Built from ALL lanes on the bus (via [`LaneIndex::lanes_for`]), not just
/// the scored/selected triples. The `movers` set passed to grid construction
/// identifies which sources correspond to selected moving atoms; empty
/// non-mover sources may still fill out the complete AOD rectangle.
pub(crate) struct BusGridContext {
    /// `(x_bits, y_bits) → encoded source location` for ALL bus positions.
    pos_to_src: HashMap<(u64, u64), u64>,
    /// `encoded source → encoded lane address` for ALL bus lanes.
    src_to_lane: HashMap<u64, u64>,
    /// `encoded source → encoded destination location` for ALL bus lanes.
    src_to_dst: HashMap<u64, u64>,
    /// `encoded source → (x_bits, y_bits)` reverse lookup.
    src_to_pos: HashMap<u64, (u64, u64)>,
    /// Locations occupied by atoms or blocked locations in the current config.
    occupied_locs: HashSet<u64>,
}

impl BusGridContext {
    /// Build a grid context from all lanes on a bus group.
    ///
    /// `occupied` is the set of encoded locations currently occupied by atoms.
    /// When `zone_id` is `None`, lanes from all zones are included.
    pub(crate) fn new(
        index: &LaneIndex,
        mt: MoveType,
        bus_id: u32,
        zone_id: Option<u32>,
        dir: Direction,
        occupied: &HashSet<u64>,
    ) -> Self {
        let mut pos_to_src: HashMap<(u64, u64), u64> = HashMap::new();
        let mut src_to_lane: HashMap<u64, u64> = HashMap::new();
        let mut src_to_dst: HashMap<u64, u64> = HashMap::new();
        let mut src_to_pos: HashMap<u64, (u64, u64)> = HashMap::new();

        let lanes_vec: Vec<LaneAddr> = match zone_id {
            Some(z) => index.lanes_for(mt, bus_id, z, dir).to_vec(),
            None => index
                .lanes_for_all_zones(mt, bus_id, dir)
                .copied()
                .collect(),
        };
        for &lane in &lanes_vec {
            let Some((src, dst)) = index.endpoints(&lane) else {
                continue;
            };
            let Some((x, y)) = index.position(src) else {
                continue;
            };
            let src_enc = src.encode();
            let lane_enc = lane.encode_u64();
            let pos = (x.to_bits(), y.to_bits());

            pos_to_src.insert(pos, src_enc);
            src_to_lane.insert(src_enc, lane_enc);
            src_to_dst.insert(src_enc, dst.encode());
            src_to_pos.insert(src_enc, pos);
        }

        Self {
            pos_to_src,
            src_to_lane,
            src_to_dst,
            src_to_pos,
            occupied_locs: occupied.clone(),
        }
    }

    /// Check if every position in the X × Y rectangle is valid.
    ///
    /// A selected mover source is valid when its destination is unoccupied or
    /// occupied by another atom moving in the same rectangle. A non-mover source
    /// may only fill the rectangle when both its source and destination avoid
    /// stationary atoms.
    fn is_valid_rect(&self, xs: &BTreeSet<u64>, ys: &BTreeSet<u64>, movers: &HashSet<u64>) -> bool {
        let mut rect_sources = HashSet::new();
        for &x in xs {
            for &y in ys {
                let Some(&src_enc) = self.pos_to_src.get(&(x, y)) else {
                    return false;
                };
                rect_sources.insert(src_enc);
            }
        }

        for &x in xs {
            for &y in ys {
                let Some(&src_enc) = self.pos_to_src.get(&(x, y)) else {
                    return false;
                };
                let Some(&dst_enc) = self.src_to_dst.get(&src_enc) else {
                    return false;
                };
                let src_is_mover = movers.contains(&src_enc);
                let dst_is_rect_mover =
                    movers.contains(&dst_enc) && rect_sources.contains(&dst_enc);

                if !src_is_mover && self.occupied_locs.contains(&src_enc) {
                    return false;
                }
                if self.occupied_locs.contains(&dst_enc) && !dst_is_rect_mover {
                    return false;
                }
            }
        }
        true
    }

    /// Convert a cluster's X × Y rectangle to a vector of encoded lane addresses.
    fn rect_to_lanes(&self, xs: &BTreeSet<u64>, ys: &BTreeSet<u64>) -> Vec<u64> {
        let mut lanes = Vec::with_capacity(xs.len() * ys.len());
        for &x in xs {
            for &y in ys {
                if let Some(&src_enc) = self.pos_to_src.get(&(x, y))
                    && let Some(&lane_enc) = self.src_to_lane.get(&src_enc)
                {
                    lanes.push(lane_enc);
                }
            }
        }
        lanes
    }

    /// Build AOD-compatible rectangular grids from scored entry lanes.
    ///
    /// `entries` maps `encoded_src → encoded_lane` for the scored/selected
    /// moving atoms. Returned grids may also include empty filler lanes so each
    /// lane set remains a complete AOD rectangle.
    ///
    /// Returns a list of lane sets, each forming a valid AOD rectangle.
    pub(crate) fn build_aod_grids(&self, entries: &HashMap<u64, u64>) -> Vec<Vec<u64>> {
        if entries.is_empty() {
            return Vec::new();
        }

        // Movers = all source locations from the entries.
        let movers: HashSet<u64> = entries.keys().copied().collect();

        let clusters = self.greedy_init(entries, &movers);
        let solved = self.merge_clusters(clusters, &movers);

        solved
            .iter()
            .map(|(xs, ys)| self.rect_to_lanes(xs, ys))
            .filter(|lanes| !lanes.is_empty())
            .collect()
    }

    /// Form initial clusters via greedy sequential expansion.
    ///
    /// Processes entries in order and greedily expands a rectangle. Entries
    /// that don't fit are put aside for the next round. Repeats until all
    /// entries are assigned or no progress is made.
    fn greedy_init(&self, entries: &HashMap<u64, u64>, movers: &HashSet<u64>) -> Vec<Cluster> {
        let mut clusters: Vec<Cluster> = Vec::new();
        // Sort by src_encoded for deterministic iteration order.
        let mut remaining: Vec<(u64, u64)> = entries.iter().map(|(&s, &l)| (s, l)).collect();
        remaining.sort_by_key(|&(src, _)| src);

        while !remaining.is_empty() {
            let mut xs: BTreeSet<u64> = BTreeSet::new();
            let mut ys: BTreeSet<u64> = BTreeSet::new();
            let mut leftover: Vec<(u64, u64)> = Vec::new();

            for &(src_enc, lane_enc) in &remaining {
                let Some(&(x, y)) = self.src_to_pos.get(&src_enc) else {
                    leftover.push((src_enc, lane_enc));
                    continue;
                };

                // Skip if both coordinates already in rectangle (atom already covered).
                if xs.contains(&x) && ys.contains(&y) {
                    continue;
                }

                let mut new_xs = xs.clone();
                let mut new_ys = ys.clone();
                new_xs.insert(x);
                new_ys.insert(y);

                if self.is_valid_rect(&new_xs, &new_ys, movers) {
                    xs = new_xs;
                    ys = new_ys;
                } else {
                    leftover.push((src_enc, lane_enc));
                }
            }

            if xs.is_empty() || ys.is_empty() {
                break;
            }

            clusters.push((xs, ys));
            remaining = leftover;
        }

        clusters
    }

    /// Merge clusters until no more merges are possible.
    ///
    /// Each pass tries all pairs (i, j). If the union rectangle is valid,
    /// cluster i absorbs j. Clusters that don't participate in any merge
    /// are promoted to "solved" and removed — merged clusters only grow,
    /// so a non-merging cluster will never merge later.
    fn merge_clusters(&self, mut clusters: Vec<Cluster>, movers: &HashSet<u64>) -> Vec<Cluster> {
        let mut solved: Vec<Cluster> = Vec::new();

        while clusters.len() > 1 {
            let n = clusters.len();
            let mut consumed: HashSet<usize> = HashSet::new();
            let mut merged_flags = vec![false; n];

            for i in 0..n {
                if consumed.contains(&i) {
                    continue;
                }
                for j in (i + 1)..n {
                    if consumed.contains(&j) {
                        continue;
                    }
                    let merged_xs: BTreeSet<u64> =
                        clusters[i].0.union(&clusters[j].0).copied().collect();
                    let merged_ys: BTreeSet<u64> =
                        clusters[i].1.union(&clusters[j].1).copied().collect();

                    if self.is_valid_rect(&merged_xs, &merged_ys, movers) {
                        clusters[i] = (merged_xs, merged_ys);
                        consumed.insert(j);
                        merged_flags[i] = true;
                        merged_flags[j] = true;
                    }
                }
            }

            if !merged_flags.iter().any(|&f| f) {
                break;
            }

            let mut active: Vec<Cluster> = Vec::new();
            for i in 0..n {
                if consumed.contains(&i) {
                    continue;
                }
                if merged_flags[i] {
                    active.push(std::mem::take(&mut clusters[i]));
                } else {
                    solved.push(std::mem::take(&mut clusters[i]));
                }
            }
            clusters = active;
        }

        solved.extend(clusters);
        solved
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a BusGridContext from raw position/lane/collision data.
    fn make_context(
        positions: &[((u64, u64), u64)], // ((x, y), src_encoded)
        lanes: &[(u64, u64)],            // (src_encoded, lane_encoded)
        collisions: &[u64],              // src_encoded values with occupied destinations
    ) -> BusGridContext {
        make_context_with_occupied(positions, lanes, collisions, &[])
    }

    fn make_context_with_occupied(
        positions: &[((u64, u64), u64)], // ((x, y), src_encoded)
        lanes: &[(u64, u64)],            // (src_encoded, lane_encoded)
        collisions: &[u64],              // src_encoded values with stationary occupied destinations
        occupied: &[u64],                // encoded locations occupied by stationary atoms
    ) -> BusGridContext {
        const TEST_DST_OFFSET: u64 = 1_000_000;

        let lanes_with_dst: Vec<(u64, u64, u64)> = lanes
            .iter()
            .map(|&(src_enc, lane_enc)| (src_enc, lane_enc, src_enc + TEST_DST_OFFSET))
            .collect();
        let mut occupied_locs: Vec<u64> = occupied.to_vec();
        occupied_locs.extend(collisions.iter().map(|src_enc| src_enc + TEST_DST_OFFSET));
        make_context_with_endpoints(positions, &lanes_with_dst, &occupied_locs)
    }

    fn make_context_with_endpoints(
        positions: &[((u64, u64), u64)], // ((x, y), src_encoded)
        lanes: &[(u64, u64, u64)],       // (src_encoded, lane_encoded, dst_encoded)
        occupied_locs_input: &[u64],     // all encoded occupied locations
    ) -> BusGridContext {
        let mut pos_to_src = HashMap::new();
        let mut src_to_pos = HashMap::new();
        for &(pos, src_enc) in positions {
            pos_to_src.insert(pos, src_enc);
            src_to_pos.insert(src_enc, pos);
        }

        let mut src_to_lane = HashMap::new();
        let mut src_to_dst = HashMap::new();
        for &(src_enc, lane_enc, dst_enc) in lanes {
            src_to_lane.insert(src_enc, lane_enc);
            src_to_dst.insert(src_enc, dst_enc);
        }

        let occupied_locs: HashSet<u64> = occupied_locs_input.iter().copied().collect();

        BusGridContext {
            pos_to_src,
            src_to_lane,
            src_to_dst,
            src_to_pos,
            occupied_locs,
        }
    }

    #[test]
    fn is_valid_rect_all_movers() {
        // 2×2 grid, all positions are movers, no collisions.
        let ctx = make_context(
            &[((0, 0), 10), ((1, 0), 11), ((0, 1), 12), ((1, 1), 13)],
            &[(10, 100), (11, 101), (12, 102), (13, 103)],
            &[],
        );
        let movers: HashSet<u64> = [10, 11, 12, 13].into_iter().collect();
        let xs: BTreeSet<u64> = [0, 1].into_iter().collect();
        let ys: BTreeSet<u64> = [0, 1].into_iter().collect();

        assert!(ctx.is_valid_rect(&xs, &ys, &movers));
    }

    #[test]
    fn is_valid_rect_missing_mover() {
        // 2×2 grid but position (1,1) is not a mover.
        let ctx = make_context(
            &[((0, 0), 10), ((1, 0), 11), ((0, 1), 12), ((1, 1), 13)],
            &[(10, 100), (11, 101), (12, 102), (13, 103)],
            &[],
        );
        let movers: HashSet<u64> = [10, 11, 12].into_iter().collect(); // 13 missing
        let xs: BTreeSet<u64> = [0, 1].into_iter().collect();
        let ys: BTreeSet<u64> = [0, 1].into_iter().collect();

        assert!(ctx.is_valid_rect(&xs, &ys, &movers));
    }

    #[test]
    fn build_aod_grids_keeps_empty_filler_lane() {
        // 2×2 rectangle where (1,1) is empty. The AOD shot should still
        // contain all four lanes so lane-group geometry remains complete.
        let ctx = make_context(
            &[((0, 0), 10), ((1, 0), 11), ((0, 1), 12), ((1, 1), 13)],
            &[(10, 100), (11, 101), (12, 102), (13, 103)],
            &[],
        );
        let entries: HashMap<u64, u64> = [(10, 100), (11, 101), (12, 102)].into_iter().collect();

        let grids = ctx.build_aod_grids(&entries);
        assert_eq!(grids.len(), 1);

        let mut sorted = grids[0].clone();
        sorted.sort();
        assert_eq!(sorted, vec![100, 101, 102, 103]);
    }

    #[test]
    fn build_aod_grids_rejects_empty_source_with_filled_destination() {
        // The missing mover at source 13 is empty, but its destination is
        // occupied by a stationary atom. It must not be used as a filler lane.
        let ctx = make_context(
            &[((0, 0), 10), ((1, 0), 11), ((0, 1), 12), ((1, 1), 13)],
            &[(10, 100), (11, 101), (12, 102), (13, 103)],
            &[13],
        );
        let entries: HashMap<u64, u64> = [(10, 100), (11, 101), (12, 102)].into_iter().collect();

        let grids = ctx.build_aod_grids(&entries);
        assert!(!grids.iter().any(|grid| grid.len() == 4));
    }

    #[test]
    fn build_aod_grids_allows_empty_filler_destination_with_rect_mover() {
        // The missing mover at source 13 is empty. Its destination is occupied,
        // but by source 12, which is selected to move in the same rectangle.
        // This is a valid AOD filler lane because it does not interact with a
        // stationary atom.
        let ctx = make_context_with_endpoints(
            &[((0, 0), 10), ((1, 0), 11), ((0, 1), 12), ((1, 1), 13)],
            &[(10, 100, 20), (11, 101, 21), (12, 102, 22), (13, 103, 12)],
            &[10, 11, 12],
        );
        let entries: HashMap<u64, u64> = [(10, 100), (11, 101), (12, 102)].into_iter().collect();

        let grids = ctx.build_aod_grids(&entries);

        assert_eq!(grids.len(), 1);
        let mut sorted = grids[0].clone();
        sorted.sort();
        assert_eq!(sorted, vec![100, 101, 102, 103]);
    }

    #[test]
    fn build_aod_grids_rejects_occupied_non_mover_source() {
        // Source 13 has a spectator atom, so it cannot be used as a filler lane.
        let ctx = make_context_with_occupied(
            &[((0, 0), 10), ((1, 0), 11), ((0, 1), 12), ((1, 1), 13)],
            &[(10, 100), (11, 101), (12, 102), (13, 103)],
            &[],
            &[13],
        );
        let entries: HashMap<u64, u64> = [(10, 100), (11, 101), (12, 102)].into_iter().collect();

        let grids = ctx.build_aod_grids(&entries);
        assert!(!grids.iter().any(|grid| grid.len() == 4));
    }

    #[test]
    fn build_aod_grids_color_code_sparse_rectangle() {
        let mut positions = Vec::new();
        let mut lanes = Vec::new();
        let mut entries = HashMap::new();
        let mut src = 100u64;
        let mut lane = 1000u64;

        for x in 0..4 {
            for y in 0..5 {
                positions.push(((x, y), src));
                lanes.push((src, lane));
                if x < 3 || y < 2 {
                    entries.insert(src, lane);
                }
                src += 1;
                lane += 1;
            }
        }

        let ctx = make_context(&positions, &lanes, &[]);

        let grids = ctx.build_aod_grids(&entries);
        assert_eq!(grids.len(), 1);
        assert_eq!(grids[0].len(), 20);
    }

    #[test]
    fn is_valid_rect_collision() {
        // 2×2 grid, all movers, but (1,0) has a collision.
        let ctx = make_context(
            &[((0, 0), 10), ((1, 0), 11), ((0, 1), 12), ((1, 1), 13)],
            &[(10, 100), (11, 101), (12, 102), (13, 103)],
            &[11], // collision at src 11
        );
        let movers: HashSet<u64> = [10, 11, 12, 13].into_iter().collect();
        let xs: BTreeSet<u64> = [0, 1].into_iter().collect();
        let ys: BTreeSet<u64> = [0, 1].into_iter().collect();

        assert!(!ctx.is_valid_rect(&xs, &ys, &movers));
    }

    #[test]
    fn greedy_init_single_cluster() {
        // 2×2 grid, all valid — should form one cluster.
        let ctx = make_context(
            &[((0, 0), 10), ((1, 0), 11), ((0, 1), 12), ((1, 1), 13)],
            &[(10, 100), (11, 101), (12, 102), (13, 103)],
            &[],
        );
        let entries: HashMap<u64, u64> = [(10, 100), (11, 101), (12, 102), (13, 103)]
            .into_iter()
            .collect();
        let movers: HashSet<u64> = entries.keys().copied().collect();

        let clusters = ctx.greedy_init(&entries, &movers);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].0.len(), 2); // 2 unique X
        assert_eq!(clusters[0].1.len(), 2); // 2 unique Y
    }

    #[test]
    fn greedy_init_splits_incompatible() {
        // 3 positions are movers, while the fourth source is occupied by a
        // spectator. A 2×2 rectangle would move that spectator, so it splits.
        let ctx = make_context_with_occupied(
            &[
                ((0, 0), 10),
                ((1, 0), 11),
                ((0, 1), 12),
                ((1, 1), 13), // exists on bus but not a mover
            ],
            &[(10, 100), (11, 101), (12, 102), (13, 103)],
            &[],
            &[13],
        );
        let entries: HashMap<u64, u64> = [(10, 100), (11, 101), (12, 102)].into_iter().collect();
        let movers: HashSet<u64> = entries.keys().copied().collect();

        let clusters = ctx.greedy_init(&entries, &movers);
        // Cannot form a 2×2, so should have multiple smaller clusters.
        assert!(!clusters.is_empty());
        let total_positions: usize = clusters.iter().map(|(xs, ys)| xs.len() * ys.len()).sum();
        // All 3 movers should be covered across all clusters.
        assert!(total_positions >= 2); // At least the first cluster should have entries
    }

    #[test]
    fn merge_clusters_combines_compatible() {
        // Two 1×1 clusters at (0,0) and (1,0). Both are movers.
        let ctx = make_context(&[((0, 0), 10), ((1, 0), 11)], &[(10, 100), (11, 101)], &[]);
        let movers: HashSet<u64> = [10, 11].into_iter().collect();

        let clusters = vec![
            ([0u64].into_iter().collect(), [0u64].into_iter().collect()),
            ([1u64].into_iter().collect(), [0u64].into_iter().collect()),
        ];

        let solved = ctx.merge_clusters(clusters, &movers);
        // Should merge into one 2×1 rectangle.
        assert_eq!(solved.len(), 1);
        assert_eq!(solved[0].0.len(), 2);
        assert_eq!(solved[0].1.len(), 1);
    }

    #[test]
    fn build_aod_grids_empty_entries() {
        let ctx = make_context(&[], &[], &[]);
        let entries = HashMap::new();
        let grids = ctx.build_aod_grids(&entries);
        assert!(grids.is_empty());
    }

    #[test]
    fn build_aod_grids_end_to_end() {
        // 2×2 grid, all 4 positions are movers.
        let ctx = make_context(
            &[((0, 0), 10), ((1, 0), 11), ((0, 1), 12), ((1, 1), 13)],
            &[(10, 100), (11, 101), (12, 102), (13, 103)],
            &[],
        );
        let entries: HashMap<u64, u64> = [(10, 100), (11, 101), (12, 102), (13, 103)]
            .into_iter()
            .collect();

        let grids = ctx.build_aod_grids(&entries);
        assert_eq!(grids.len(), 1);
        assert_eq!(grids[0].len(), 4);
        // All 4 lane encodings should be present.
        let mut sorted = grids[0].clone();
        sorted.sort();
        assert_eq!(sorted, vec![100, 101, 102, 103]);
    }
}
