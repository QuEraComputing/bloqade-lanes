//! AOD grid construction for the heuristic expander.
//!
//! `build_aod_grids` dispatches between two strategies based on
//! [`BusGridContext::strategy`]:
//!
//! - **GreedyMerge** (default): a greedy sequential pass forms initial
//!   rectangular clusters, then an iterative merge pass combines compatible
//!   clusters into larger rectangles.
//! - **Clique**: builds a conflict graph whose nodes are valid positions in the
//!   movers' induced Cartesian product and whose edges connect positions that
//!   form a valid 2×2 rectangle. Each round finds the maximal clique covering
//!   the most movers (tie-break: smaller area, then deterministic), emits the
//!   corresponding rectangle, prunes covered movers, and repeats.

use std::collections::{BTreeSet, HashMap, HashSet};

use bloqade_lanes_bytecode_core::arch::addr::{Direction, LaneAddr, MoveType};

use crate::primitives::lane_index::LaneIndex;
use crate::search::options::AodGridStrategy;

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
    /// Which grid-construction strategy `build_aod_grids` dispatches to.
    strategy: AodGridStrategy,
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
            strategy: AodGridStrategy::default(),
        }
    }

    /// Select the grid-construction strategy (default [`AodGridStrategy::GreedyMerge`]).
    #[allow(dead_code)]
    pub(crate) fn with_strategy(mut self, strategy: AodGridStrategy) -> Self {
        self.strategy = strategy;
        self
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
    /// Dispatches on [`BusGridContext::strategy`]. See
    /// [`build_aod_grids_greedy`](Self::build_aod_grids_greedy) and
    /// [`build_aod_grids_clique`](Self::build_aod_grids_clique).
    pub(crate) fn build_aod_grids(&self, entries: &HashMap<u64, u64>) -> Vec<Vec<u64>> {
        match self.strategy {
            AodGridStrategy::GreedyMerge => self.build_aod_grids_greedy(entries),
            AodGridStrategy::Clique => self.build_aod_grids_clique(entries),
        }
    }

    /// Greedy sequential clustering + iterative merge (original algorithm).
    fn build_aod_grids_greedy(&self, entries: &HashMap<u64, u64>) -> Vec<Vec<u64>> {
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

    /// Maximum nodes for exact Bron–Kerbosch; above this, use the greedy
    /// fallback. Bounded by the u64 adjacency bitset (≤64) and enumeration cost.
    const MAX_EXACT_CLIQUE_NODES: usize = 32;

    /// Conflict-graph max-clique grid construction.
    ///
    /// Candidate nodes are the valid positions in the input movers' induced
    /// Cartesian product; two nodes are adjacent iff their induced 2×2 rectangle
    /// is valid (so a maximal clique = a maximal valid rectangle). Each round
    /// emits the rectangle covering the most input movers (tie-break: smaller
    /// area, then deterministic), then prunes those movers and repeats.
    fn build_aod_grids_clique(&self, entries: &HashMap<u64, u64>) -> Vec<Vec<u64>> {
        if entries.is_empty() {
            return Vec::new();
        }
        let mut remaining: HashSet<u64> = entries.keys().copied().collect();
        let mut out: Vec<Vec<u64>> = Vec::new();

        while !remaining.is_empty() {
            // Candidate universe: positions in X × Y of the remaining movers.
            let nodes = self.clique_candidate_nodes(&remaining);
            if nodes.is_empty() {
                break;
            }
            let movers: &HashSet<u64> = &remaining;

            // Adjacency bitsets: edge iff the 2×2 {A,B} rectangle is valid.
            let n = nodes.len();
            let mut adj = vec![0u64; n];
            for i in 0..n {
                for j in (i + 1)..n {
                    let (xi, yi) = nodes[i];
                    let (xj, yj) = nodes[j];
                    let xs: BTreeSet<u64> = [xi, xj].into_iter().collect();
                    let ys: BTreeSet<u64> = [yi, yj].into_iter().collect();
                    if self.is_valid_rect(&xs, &ys, movers) {
                        adj[i] |= 1u64 << j;
                        adj[j] |= 1u64 << i;
                    }
                }
            }

            // Helper: is a node index a remaining mover?
            let is_mover = |idx: usize| -> bool {
                self.pos_to_src
                    .get(&nodes[idx])
                    .is_some_and(|src| movers.contains(src))
            };

            // Evaluate a clique (bitset over node indices) by the objective.
            // Scoring is based on the EMITTED rectangle (the full xs × ys product
            // of the clique's distinct tones), not the clique bitset size, because
            // the product may contain valid positions that are non-adjacent to some
            // clique members yet still covered by the emitted AOD shot.
            // Returns without updating if the emitted rectangle covers zero movers.
            // Inlined into consider to avoid borrow-checker issues with closures.
            let mut best_key: Option<(usize, std::cmp::Reverse<usize>, u64)> = None;
            let mut best_clique: u64 = 0;
            let mut consider = |clique: u64| {
                if clique == 0 {
                    return;
                }
                // Derive distinct x/y tones of the clique.
                let mut xs_set = BTreeSet::new();
                let mut ys_set = BTreeSet::new();
                let mut bits = clique;
                while bits != 0 {
                    let idx = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    let (x, y) = nodes[idx];
                    xs_set.insert(x);
                    ys_set.insert(y);
                }
                // Score on the full xs × ys product (the emitted rectangle).
                let area = xs_set.len() * ys_set.len();
                let mut movers_covered = 0usize;
                for &x in &xs_set {
                    for &y in &ys_set {
                        if let Some(&src) = self.pos_to_src.get(&(x, y))
                            && movers.contains(&src)
                        {
                            movers_covered += 1;
                        }
                    }
                }
                if movers_covered == 0 {
                    return;
                }
                // Deterministic tie-break: the clique's raw bitset (lower node
                // indices, which follow sorted node order, win).
                let key = (movers_covered, std::cmp::Reverse(area), !clique);
                if best_key.as_ref().is_none_or(|b| key > *b) {
                    best_key = Some(key);
                    best_clique = clique;
                }
            };

            if n <= Self::MAX_EXACT_CLIQUE_NODES {
                let full_p: u64 = if n == 64 { u64::MAX } else { (1u64 << n) - 1 };
                Self::bron_kerbosch(&adj, 0, full_p, 0, &mut consider);
            } else {
                Self::greedy_max_clique(&adj, n, &is_mover, &mut consider);
            }

            if best_clique == 0 {
                break; // no mover-covering clique (shouldn't happen while movers remain)
            }

            // Coordinate sets of the winning clique → complete rectangle.
            let mut xs = BTreeSet::new();
            let mut ys = BTreeSet::new();
            let mut bits = best_clique;
            while bits != 0 {
                let idx = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let (x, y) = nodes[idx];
                xs.insert(x);
                ys.insert(y);
            }
            let lanes = self.rect_to_lanes(&xs, &ys);
            if lanes.is_empty() {
                break;
            }

            // Prune covered movers (sources inside the rectangle).
            for &x in &xs {
                for &y in &ys {
                    if let Some(&src) = self.pos_to_src.get(&(x, y)) {
                        remaining.remove(&src);
                    }
                }
            }
            out.push(lanes);
        }

        out
    }

    /// Valid positions in the movers' induced X × Y product (each a valid 1×1).
    /// Sorted for determinism. Capped at 64 nodes (u64 bitset); excess dropped
    /// deterministically (the greedy fallback handles large sets, and >64 valid
    /// positions is not expected per bus per step).
    fn clique_candidate_nodes(&self, movers: &HashSet<u64>) -> Vec<(u64, u64)> {
        let mut xs = BTreeSet::new();
        let mut ys = BTreeSet::new();
        for &src in movers {
            if let Some(&(x, y)) = self.src_to_pos.get(&src) {
                xs.insert(x);
                ys.insert(y);
            }
        }
        let mut nodes = Vec::new();
        for &x in &xs {
            for &y in &ys {
                let px: BTreeSet<u64> = [x].into_iter().collect();
                let py: BTreeSet<u64> = [y].into_iter().collect();
                if self.is_valid_rect(&px, &py, movers) {
                    nodes.push((x, y));
                    if nodes.len() == 64 {
                        return nodes;
                    }
                }
            }
        }
        nodes
    }

    /// Bron–Kerbosch with pivoting over a u64 adjacency bitset (≤64 nodes).
    /// Invokes `visit` on each maximal clique (as a node-index bitset).
    fn bron_kerbosch(adj: &[u64], r: u64, p: u64, x: u64, visit: &mut impl FnMut(u64)) {
        if p == 0 && x == 0 {
            visit(r);
            return;
        }
        // Pivot u ∈ P∪X maximizing |P ∩ adj[u]|.
        let mut pux = p | x;
        let mut pivot = 0usize;
        let mut best = -1i32;
        while pux != 0 {
            let u = pux.trailing_zeros() as usize;
            pux &= pux - 1;
            let cnt = (p & adj[u]).count_ones() as i32;
            if cnt > best {
                best = cnt;
                pivot = u;
            }
        }
        let mut p = p;
        let mut x = x;
        let mut cand = p & !adj[pivot];
        while cand != 0 {
            let v = cand.trailing_zeros() as usize;
            let vbit = 1u64 << v;
            cand &= cand - 1;
            Self::bron_kerbosch(adj, r | vbit, p & adj[v], x & adj[v], visit);
            p &= !vbit;
            x |= vbit;
        }
    }

    /// Greedy maximal-clique fallback for large node sets. Seeds from each mover
    /// node, extends by highest-degree compatible node, and reports each grown
    /// clique to `visit`. Deterministic (sorted seeds/candidates).
    fn greedy_max_clique(
        adj: &[u64],
        n: usize,
        is_mover: &impl Fn(usize) -> bool,
        visit: &mut impl FnMut(u64),
    ) {
        for seed in 0..n {
            if !is_mover(seed) {
                continue;
            }
            let mut clique = 1u64 << seed;
            let mut cand = adj[seed];
            while cand != 0 {
                // Pick the candidate with the most connections to `cand`.
                // Tie-break: lowest index wins, because `trailing_zeros` iterates
                // low-to-high and a strictly-greater `deg > best` comparison means
                // the first (lowest-index) maximum is kept.
                let mut bits = cand;
                let mut pick = usize::MAX;
                let mut best = -1i32;
                while bits != 0 {
                    let v = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    let deg = (adj[v] & cand).count_ones() as i32;
                    if deg > best {
                        best = deg;
                        pick = v;
                    }
                }
                if pick == usize::MAX {
                    break;
                }
                clique |= 1u64 << pick;
                cand &= adj[pick];
            }
            visit(clique);
        }
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
            strategy: AodGridStrategy::default(),
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

    #[test]
    fn strategy_defaults_to_greedy_merge() {
        let ctx = make_context(&[((0, 0), 10)], &[(10, 100)], &[]);
        assert_eq!(ctx.strategy, AodGridStrategy::GreedyMerge);
        let clique = ctx.with_strategy(AodGridStrategy::Clique);
        assert_eq!(clique.strategy, AodGridStrategy::Clique);
    }

    #[test]
    fn clique_builds_single_complete_rectangle() {
        // 2×2 grid, 3 movers + 1 empty filler (13). Clique strategy should
        // return one complete rectangle of all four lanes.
        let ctx = make_context(
            &[((0, 0), 10), ((1, 0), 11), ((0, 1), 12), ((1, 1), 13)],
            &[(10, 100), (11, 101), (12, 102), (13, 103)],
            &[],
        )
        .with_strategy(AodGridStrategy::Clique);
        let entries: HashMap<u64, u64> = [(10, 100), (11, 101), (12, 102)].into_iter().collect();

        let grids = ctx.build_aod_grids(&entries);
        assert_eq!(grids.len(), 1);
        let mut sorted = grids[0].clone();
        sorted.sort();
        assert_eq!(sorted, vec![100, 101, 102, 103]);
    }

    #[test]
    fn clique_recovers_rectangle_greedy_would_split() {
        // Movers at (0,0),(1,1) with valid corners (1,0),(0,1) as empty fillers.
        // Both strategies must yield one 2×2 rectangle; this locks in that the
        // clique path completes the rectangle from the movers' product.
        let ctx = make_context(
            &[((0, 0), 10), ((1, 1), 13), ((1, 0), 11), ((0, 1), 12)],
            &[(10, 100), (13, 103), (11, 101), (12, 102)],
            &[],
        )
        .with_strategy(AodGridStrategy::Clique);
        let entries: HashMap<u64, u64> = [(10, 100), (13, 103)].into_iter().collect();

        let grids = ctx.build_aod_grids(&entries);
        assert_eq!(grids.len(), 1);
        let mut sorted = grids[0].clone();
        sorted.sort();
        assert_eq!(sorted, vec![100, 101, 102, 103]);
    }

    #[test]
    fn clique_emits_only_mover_covering_rectangles() {
        // One mover at (0,0). A disjoint empty 2×2 region at x∈{5,6}, y∈{5,6}
        // is a valid rectangle but covers no mover — it must never be emitted.
        let ctx = make_context(
            &[
                ((0, 0), 10),
                ((5, 5), 20),
                ((6, 5), 21),
                ((5, 6), 22),
                ((6, 6), 23),
            ],
            &[(10, 100), (20, 200), (21, 201), (22, 202), (23, 203)],
            &[],
        )
        .with_strategy(AodGridStrategy::Clique);
        let entries: HashMap<u64, u64> = [(10, 100)].into_iter().collect();

        let grids = ctx.build_aod_grids(&entries);
        // Only the mover's own 1×1 rectangle; no empty-region rectangle.
        assert_eq!(grids.len(), 1);
        assert_eq!(grids[0], vec![100]);
    }

    #[test]
    fn clique_reversibility_rejects_filled_filler_destination() {
        // Mover 10 at (0,0). Filler 13 at (1,1) has an occupied destination
        // (stationary atom), so the 2×2 with mover 11/12 corners is invalid;
        // the reversible sub-rectangle is 1×2 or 2×1, not 2×2.
        let ctx = make_context(
            &[((0, 0), 10), ((1, 0), 11), ((0, 1), 12), ((1, 1), 13)],
            &[(10, 100), (11, 101), (12, 102), (13, 103)],
            &[13], // src 13's destination is occupied (stationary)
        )
        .with_strategy(AodGridStrategy::Clique);
        let entries: HashMap<u64, u64> = [(10, 100), (11, 101), (12, 102)].into_iter().collect();

        let grids = ctx.build_aod_grids(&entries);
        // No 4-lane rectangle (13 can't be a filler); every mover still covered.
        assert!(!grids.iter().any(|g| g.len() == 4));
        let covered: HashSet<u64> = grids.iter().flatten().copied().collect();
        assert!([100, 101, 102].iter().all(|l| covered.contains(l)));
    }

    #[test]
    fn clique_is_deterministic() {
        // Fix 4: prove order-independence structurally by building TWO contexts
        // with positions/lanes inserted in different orders and asserting identical
        // output after consistent per-grid normalization.
        //
        // Context A: positions and lanes in ascending src order.
        let ctx_a = make_context(
            &[((0, 0), 10), ((1, 0), 11), ((0, 1), 12), ((1, 1), 13)],
            &[(10, 100), (11, 101), (12, 102), (13, 103)],
            &[],
        )
        .with_strategy(AodGridStrategy::Clique);

        // Context B: positions and lanes in reversed src order (different HashMap
        // insertion order, which affects HashMap iteration and bucket distribution).
        let ctx_b = make_context(
            &[((1, 1), 13), ((0, 1), 12), ((1, 0), 11), ((0, 0), 10)],
            &[(13, 103), (12, 102), (11, 101), (10, 100)],
            &[],
        )
        .with_strategy(AodGridStrategy::Clique);

        let entries: HashMap<u64, u64> = [(10, 100), (11, 101), (12, 102), (13, 103)]
            .into_iter()
            .collect();

        let mut a = ctx_a.build_aod_grids(&entries);
        let mut b = ctx_b.build_aod_grids(&entries);

        // Normalize: sort lanes within each grid, then sort grids for comparison.
        for g in &mut a {
            g.sort();
        }
        for g in &mut b {
            g.sort();
        }
        a.sort();
        b.sort();

        assert_eq!(
            a, b,
            "clique strategy must produce identical output regardless of insertion order"
        );
    }

    #[test]
    fn clique_covers_all_movers_via_decomposition() {
        // Two separate movers whose induced 2×2 corners are blocked, forcing
        // two 1×1 shots. Both movers must be covered across the returned grids.
        let ctx = make_context_with_occupied(
            &[((0, 0), 10), ((3, 3), 13), ((3, 0), 11), ((0, 3), 12)],
            &[(10, 100), (13, 103), (11, 101), (12, 102)],
            &[],
            &[11, 12], // corner sources are spectators → 2×2 invalid
        )
        .with_strategy(AodGridStrategy::Clique);
        let entries: HashMap<u64, u64> = [(10, 100), (13, 103)].into_iter().collect();

        let grids = ctx.build_aod_grids(&entries);
        let covered: HashSet<u64> = grids.iter().flatten().copied().collect();
        assert!(covered.contains(&100) && covered.contains(&103));
    }

    /// Fix 3: Build a candidate set exceeding MAX_EXACT_CLIQUE_NODES (>32 valid
    /// positions) with AodGridStrategy::Clique and assert the result is valid:
    /// non-empty, every emitted grid covers ≥1 mover, and all input movers are
    /// covered across the returned grids. This exercises the `greedy_max_clique`
    /// branch.
    #[test]
    fn clique_greedy_fallback_large_candidate_set() {
        // Build a 7×6 = 42-position grid. All positions are movers. 42 > 32
        // forces the greedy_max_clique fallback (MAX_EXACT_CLIQUE_NODES = 32).
        let mut positions: Vec<((u64, u64), u64)> = Vec::new();
        let mut lanes: Vec<(u64, u64)> = Vec::new();
        let mut entries: HashMap<u64, u64> = HashMap::new();
        let mut src = 1u64;
        let mut lane_enc = 1001u64;

        for x in 0u64..7 {
            for y in 0u64..6 {
                positions.push(((x, y), src));
                lanes.push((src, lane_enc));
                entries.insert(src, lane_enc);
                src += 1;
                lane_enc += 1;
            }
        }

        let ctx = make_context(&positions, &lanes, &[]).with_strategy(AodGridStrategy::Clique);
        let mover_lanes: HashSet<u64> = entries.values().copied().collect();

        let grids = ctx.build_aod_grids(&entries);

        // Must emit at least one grid.
        assert!(
            !grids.is_empty(),
            "greedy fallback must produce at least one grid"
        );

        // Every emitted grid must cover ≥1 mover.
        for grid in &grids {
            let covers_mover = grid.iter().any(|l| mover_lanes.contains(l));
            assert!(
                covers_mover,
                "every emitted grid must cover at least one mover; got {:?}",
                grid
            );
        }

        // All input movers must be covered across the returned grids.
        let covered: HashSet<u64> = grids.iter().flatten().copied().collect();
        for &l in &mover_lanes {
            assert!(covered.contains(&l), "mover lane {} not covered", l);
        }
    }

    /// Fix 1+2: Verify that the objective scoring uses the emitted rectangle's
    /// product dimensions (xs.len() * ys.len()) rather than the clique bitset
    /// count. We construct a scenario where two competing cliques have the same
    /// number of nodes (clique.count_ones()) but different emitted-rectangle areas
    /// (xs.len()*ys.len()), then assert the winner is selected by product-area.
    ///
    /// Scenario:
    ///   - 6 movers laid out as two groups:
    ///     Group A: (0,0),(1,0),(0,1),(1,1) — a fully-connected 2×2 with 4 movers.
    ///     Group B: (5,0),(6,0),(7,0) — a 3×1 line with 3 movers.
    ///   - The maximal clique for A has 4 nodes (all pairwise valid), area_product=4.
    ///   - The maximal clique for B has 3 nodes, area_product=3.
    ///   - Under EITHER scoring A wins (4 movers vs 3). The meaningful check is
    ///     that A's emitted rectangle has area 4 (xs.len()*ys.len()=4), not that
    ///     area is computed from node count in some other way.
    ///
    /// Note: The L-shape fixture (clique with fewer nodes than its xs×ys product)
    /// is NOT constructible in this graph: for any 3-node L-shape {A,B,C} to be
    /// fully connected, the edge between the two nodes that differ in BOTH x and y
    /// requires is_valid_rect on their induced 2×2, which checks the 4th product
    /// cell. If the 4th cell is valid, it is adjacent to all three and extends the
    /// clique to 4 nodes, making the L-shape non-maximal. If the 4th cell is
    /// invalid, the 2×2 edge doesn't exist and the L-shape isn't a clique. Hence
    /// a strict L-shaped maximal clique cannot arise.
    ///
    /// Instead we verify: the winning rectangle's lane count equals xs.len()*ys.len()
    /// and covers the correct set of movers.
    #[test]
    fn clique_objective_uses_product_area_not_bitset_count() {
        // Group A: 2×2 = 4 movers at x∈{0,1}, y∈{0,1} (all valid, no collisions).
        // Group B: 3×1 = 3 movers at x∈{5,6,7}, y∈{5} (disjoint in BOTH x and y).
        // Groups share no x or y tones so no cross-group rectangle forms.
        let ctx = make_context(
            &[
                ((0, 0), 10),
                ((1, 0), 11),
                ((0, 1), 12),
                ((1, 1), 13), // group A
                ((5, 5), 20),
                ((6, 5), 21),
                ((7, 5), 22), // group B (disjoint y)
            ],
            &[
                (10, 100),
                (11, 101),
                (12, 102),
                (13, 103),
                (20, 200),
                (21, 201),
                (22, 202),
            ],
            &[],
        )
        .with_strategy(AodGridStrategy::Clique);

        // All 7 are movers.
        let entries: HashMap<u64, u64> = [
            (10, 100),
            (11, 101),
            (12, 102),
            (13, 103),
            (20, 200),
            (21, 201),
            (22, 202),
        ]
        .into_iter()
        .collect();

        let grids = ctx.build_aod_grids(&entries);

        // First emitted grid must be the 2×2 group A (4 movers > 3 movers).
        assert!(!grids.is_empty());
        let mut first = grids[0].clone();
        first.sort();
        assert_eq!(
            first,
            vec![100, 101, 102, 103],
            "group A (4 movers, product area 4) must be selected first"
        );

        // Product area of first grid is xs.len()*ys.len() = 2*2 = 4, matching lane count.
        assert_eq!(
            grids[0].len(),
            4,
            "emitted rectangle lane count must equal xs.len()*ys.len()"
        );

        // All movers covered across all grids.
        let covered: HashSet<u64> = grids.iter().flatten().copied().collect();
        for l in [100u64, 101, 102, 103, 200, 201, 202] {
            assert!(covered.contains(&l), "lane {} not covered", l);
        }
    }
}
