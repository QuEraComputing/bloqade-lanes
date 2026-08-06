//! Two-phase AOD grid construction for the heuristic expander.
//!
//! Ports the Python `BusContext.build_aod_grids()` algorithm: a greedy
//! sequential pass forms initial rectangular clusters, then an iterative
//! merge pass combines compatible clusters into larger rectangles.
//!
//! The rectangles this module produces are **alternative candidates** — the
//! search takes one per step — rather than a schedule of operations to run
//! together. See [`BusGridContext::build_aod_grids`] for the full contract,
//! including why they neither cover every mover nor stay disjoint.

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap, HashSet};

use bloqade_lanes_bytecode_core::arch::addr::{Direction, MoveType};

use crate::primitives::bus_grid_maps::BusGridMaps;
use crate::primitives::lane_index::LaneIndex;

/// A cluster represented by its X and Y coordinate sets.
/// The rectangle covers the Cartesian product X × Y.
/// Coordinates are stored as `f64::to_bits()` for cheap equality.
type Cluster = (BTreeSet<u64>, BTreeSet<u64>);

/// Whether a rectangle is executable, and if not, whether *growing* it could
/// make it so. See [`BusGridContext::rect_outcome`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RectOutcome {
    /// Every cell satisfies both occupancy rules.
    Valid,
    /// Fails only because some destination's occupant moves in this group but
    /// its own cell lies outside the rectangle. Pulling those cells in may
    /// repair it — this is the conveyor-chain case.
    Repairable,
    /// Fails for a reason growth cannot fix: an unresolvable cell, a
    /// stationary atom on a filler source, or a destination held by an atom
    /// that does not move at all.
    Invalid,
}

/// The execution model's **uniform destination rule** (issue #866): a lane's
/// destination is available when it is unoccupied, or when its occupant is
/// itself one of the group's moving sources — it vacates in the same
/// simultaneous shot, so a conveyor chain `x→y, y→z` is legal.
///
/// The rule applies to empty-source *filler* lanes exactly as it does to
/// movers: the AOD trap site arrives at every lane's destination whether or
/// not that lane carried an atom, so landing on a stationary atom is a fault
/// either way. This mirrors
/// `AtomStateData::validate_moves`'s `DestinationOccupiedByStationaryAtom`
/// check, which is the normative statement of the same rule.
///
/// [`BusGridContext::is_valid_rect`] applies this rule inline against a
/// lazily-built sorted source index (it is the hottest loop in candidate
/// generation); every other caller should use this helper so the two cannot
/// drift apart.
pub(crate) fn destination_is_available(
    dst_enc: u64,
    occupied: &HashSet<u64>,
    group_mover_srcs: &HashSet<u64>,
) -> bool {
    !occupied.contains(&dst_enc) || group_mover_srcs.contains(&dst_enc)
}

/// Context for building AOD-compatible rectangular grids on one bus group.
///
/// Built from ALL lanes on the bus (via [`LaneIndex::lanes_for`]), not just
/// the scored/selected triples. The `movers` set passed to grid construction
/// identifies which sources correspond to selected moving atoms; empty
/// non-mover sources may still fill out the complete AOD rectangle.
///
/// The arch-derived lookup maps are borrowed from [`LaneIndex`]'s precomputed
/// cache when possible (the common all-zones case); only occupancy is
/// per-call state.
pub(crate) struct BusGridContext<'a> {
    /// Occupancy-independent bus lookups, borrowed from the `LaneIndex`
    /// cache in the all-zones case, or freshly built for a single zone.
    maps: Cow<'a, BusGridMaps>,
    /// Locations occupied by atoms or blocked locations in the current config.
    /// Borrowed in production (one context per bus group per node — cloning
    /// here was pure overhead); owned only by tests that fabricate contexts.
    occupied_locs: Cow<'a, HashSet<u64>>,
    /// Scratch buffers reused across `is_valid_rect` calls (the hottest loop
    /// in candidate generation) to avoid a fresh allocation per rectangle:
    /// the rectangle's `(src, dst)` cells and its sorted source encodings.
    cells_scratch: RefCell<Vec<(u64, u64)>>,
    srcs_scratch: RefCell<Vec<u64>>,
}

impl<'a> BusGridContext<'a> {
    /// Build a grid context from all lanes on a bus group.
    ///
    /// `occupied` is the set of encoded locations currently occupied by atoms.
    /// When `zone_id` is `None`, lanes from all zones are included and the
    /// arch maps are borrowed from the `LaneIndex` cache (zero rebuild). When
    /// `zone_id` is `Some`, the maps are built for that zone only.
    pub(crate) fn new(
        index: &'a LaneIndex,
        mt: MoveType,
        bus_id: u32,
        zone_id: Option<u32>,
        dir: Direction,
        occupied: &'a HashSet<u64>,
    ) -> Self {
        let maps = match zone_id {
            None => match index.bus_grid_maps(mt, bus_id, dir) {
                Some(cached) => Cow::Borrowed(cached),
                None => Cow::Owned(BusGridMaps::default()),
            },
            Some(z) => Cow::Owned(BusGridMaps::from_lanes(
                index,
                index.lanes_for(mt, bus_id, z, dir).iter().copied(),
            )),
        };

        Self {
            maps,
            occupied_locs: Cow::Borrowed(occupied),
            cells_scratch: RefCell::new(Vec::new()),
            srcs_scratch: RefCell::new(Vec::new()),
        }
    }

    /// Check if every position in the X × Y rectangle is valid.
    ///
    /// A selected mover source is valid when its destination is unoccupied or
    /// occupied by another atom moving in the same rectangle. A non-mover source
    /// may only fill the rectangle when both its source and destination avoid
    /// stationary atoms.
    ///
    /// The destination half of that is [`destination_is_available`]'s rule,
    /// specialized here to a lazily-built sorted source index for speed.
    fn is_valid_rect(&self, xs: &BTreeSet<u64>, ys: &BTreeSet<u64>, movers: &HashSet<u64>) -> bool {
        matches!(self.rect_outcome(xs, ys, movers, None), RectOutcome::Valid)
    }

    /// Validate the X × Y rectangle, distinguishing failures that *growing* the
    /// rectangle could repair from those it never could.
    ///
    /// The distinction exists because this predicate is **non-monotone**: the
    /// destination rule exempts an occupied destination whose occupant is one
    /// of *this rectangle's own* moving sources, so adding cells can turn an
    /// invalid rectangle valid. A conveyor chain is exactly that case — the
    /// leader's cell alone is invalid because the follower is not yet a source
    /// of the rectangle. (Growth can never invalidate an already-passing cell:
    /// the source rule and cell resolution are per-cell, and the only
    /// rectangle-dependent term, the source set, only grows.)
    ///
    /// When `repairs` is `Some`, the encoded source locations whose cells would
    /// supply the missing exemptions are appended to it — those are the
    /// positions [`Self::try_add_point`] pulls in. Passing `None` keeps the hot
    /// path allocation-free and early-exiting, which is what
    /// [`Self::is_valid_rect`] does.
    fn rect_outcome(
        &self,
        xs: &BTreeSet<u64>,
        ys: &BTreeSet<u64>,
        movers: &HashSet<u64>,
        mut repairs: Option<&mut Vec<u64>>,
    ) -> RectOutcome {
        // Resolve every cell's (src, dst) once into a reused scratch buffer.
        let mut cells = self.cells_scratch.borrow_mut();
        cells.clear();
        cells.reserve(xs.len() * ys.len());
        for &x in xs {
            for &y in ys {
                let Some(&src_enc) = self.maps.pos_to_src.get(&(x, y)) else {
                    return RectOutcome::Invalid;
                };
                let Some(&dst_enc) = self.maps.src_to_dst.get(&src_enc) else {
                    return RectOutcome::Invalid;
                };
                cells.push((src_enc, dst_enc));
            }
        }

        // "Is this destination one of the rectangle's own sources?" is only
        // asked for cells whose destination is occupied, so build the sorted
        // source index lazily on first ask: rectangles whose destinations are
        // free (the common case) never pay for it, while a dense block move
        // over a full-width rectangle — 160 cells on the physical spec — gets
        // binary search instead of a nested scan.
        let mut srcs = self.srcs_scratch.borrow_mut();
        let mut srcs_ready = false;
        let mut repairable = false;

        for &(src_enc, dst_enc) in cells.iter() {
            if !movers.contains(&src_enc) && self.occupied_locs.contains(&src_enc) {
                return RectOutcome::Invalid;
            }
            if self.occupied_locs.contains(&dst_enc) {
                let occupant_moves = movers.contains(&dst_enc);
                let in_rect = occupant_moves && {
                    if !srcs_ready {
                        srcs.clear();
                        srcs.extend(cells.iter().map(|&(s, _)| s));
                        srcs.sort_unstable();
                        srcs_ready = true;
                    }
                    srcs.binary_search(&dst_enc).is_ok()
                };
                if in_rect {
                    continue;
                }
                // The occupant moves, but its own cell is outside this
                // rectangle: pulling that cell in would satisfy the rule.
                // Anything else is a stationary atom in the way, which no
                // amount of growth can fix.
                match (occupant_moves, repairs.as_deref_mut()) {
                    (true, Some(out)) => {
                        repairable = true;
                        out.push(dst_enc);
                    }
                    (true, None) => return RectOutcome::Repairable,
                    (false, _) => return RectOutcome::Invalid,
                }
            }
        }

        if repairable {
            RectOutcome::Repairable
        } else {
            RectOutcome::Valid
        }
    }

    /// Convert a cluster's X × Y rectangle to a vector of encoded lane addresses.
    fn rect_to_lanes(&self, xs: &BTreeSet<u64>, ys: &BTreeSet<u64>) -> Vec<u64> {
        let mut lanes = Vec::with_capacity(xs.len() * ys.len());
        for &x in xs {
            for &y in ys {
                if let Some(&src_enc) = self.maps.pos_to_src.get(&(x, y))
                    && let Some(&lane_enc) = self.maps.src_to_lane.get(&src_enc)
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
    /// moving atoms. Each returned lane set is a complete X × Y rectangle on
    /// this one bus group: an AOD addresses rows and columns independently, so
    /// selecting a column drives it at *every* selected row. Empty **filler
    /// lanes** are therefore included where needed to complete the product —
    /// they carry no atom but let a rectangle grow to cover more movers.
    ///
    /// # What the result is, precisely
    ///
    /// The returned rectangles are **alternatives**, not a schedule. Every
    /// caller turns each rectangle into its own search candidate — one edge out
    /// of the current node — and the search picks *one*. They are not steps to
    /// execute together.
    ///
    /// Two properties are worth stating because neither is guaranteed, and the
    /// original design intent was that both would be:
    ///
    /// - **Not a complete covering.** A mover that cannot sit in any valid
    ///   rectangle — its destination is held by a stationary atom, say — is
    ///   silently omitted. `greedy_init` stops once a pass places nothing new,
    ///   dropping whatever is left.
    /// - **Not pairwise disjoint** (since issue #887). Repairing a chain pulls
    ///   the blocking mover's cell into the rectangle being grown, even when an
    ///   earlier rectangle already covered that mover, so the same atom may
    ///   appear in several alternatives. Under alternatives semantics that is
    ///   harmless — it simply means the atom has several possible moves — but
    ///   it does mean the rectangles no longer partition the movers.
    ///
    /// # If you want to execute several rectangles together
    ///
    /// Parallelising across bus groups needs the chosen rectangles to be
    /// **atom-disjoint**, or the same atom is picked up by two AOD operations
    /// at once. This output does not provide that, and it cannot be obtained by
    /// filtering: removing an atom from a rectangle may break the Cartesian
    /// product *and* strip an exemption another cell depended on (dropping a
    /// chain's follower invalidates its leader). Such a caller must therefore
    /// *regenerate* with the reserved atoms treated as immovable from the
    /// start, rather than subtracting them afterwards.
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

    /// Grow the rectangle to cover `(x, y)`, pulling in whatever else that
    /// requires, and report whether it stayed executable.
    ///
    /// Adding a point is never adding one cell: the rectangle is the complete
    /// Cartesian product, so a new column is driven at every selected row and
    /// vice versa. Every forced cell is validated, and when the only failures
    /// are destinations whose occupants move in this group, those occupants'
    /// own cells are pulled in too — repeating to a fixpoint, since each
    /// repair can force further cells of its own.
    ///
    /// Terminates because every iteration adds at least one new coordinate to
    /// a finite set, and rolls the rectangle back untouched on failure.
    fn try_add_point(
        &self,
        xs: &mut BTreeSet<u64>,
        ys: &mut BTreeSet<u64>,
        movers: &HashSet<u64>,
        x: u64,
        y: u64,
    ) -> bool {
        // Grow in place and undo exactly what we inserted, rather than
        // snapshotting the sets: this is the hottest loop in candidate
        // generation, and the three buffers below stay unallocated unless a
        // chain actually needs repairing — which cannot happen on an
        // endpoint-disjoint bus, so the shipped specs pay nothing for this.
        let mut added_x: Vec<u64> = Vec::new();
        let mut added_y: Vec<u64> = Vec::new();
        let mut repairs: Vec<u64> = Vec::new();
        if xs.insert(x) {
            added_x.push(x);
        }
        if ys.insert(y) {
            added_y.push(y);
        }

        loop {
            repairs.clear();
            match self.rect_outcome(xs, ys, movers, Some(&mut repairs)) {
                RectOutcome::Valid => return true,
                RectOutcome::Invalid => break,
                RectOutcome::Repairable => {
                    // Pull in each blocking occupant's own cell. If no repair
                    // adds a coordinate we lack, the rectangle cannot
                    // converge and we stop.
                    let mut grew = false;
                    for src_enc in repairs.iter() {
                        let Some(&(rx, ry)) = self.maps.src_to_pos.get(src_enc) else {
                            // The occupant moves on this bus but has no
                            // position in this grid (another zone or bus
                            // group), so its cell can never join.
                            grew = false;
                            break;
                        };
                        if xs.insert(rx) {
                            added_x.push(rx);
                            grew = true;
                        }
                        if ys.insert(ry) {
                            added_y.push(ry);
                            grew = true;
                        }
                    }
                    if !grew {
                        break;
                    }
                }
            }
        }

        for x in added_x {
            xs.remove(&x);
        }
        for y in added_y {
            ys.remove(&y);
        }
        false
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
                let Some(&(x, y)) = self.maps.src_to_pos.get(&src_enc) else {
                    leftover.push((src_enc, lane_enc));
                    continue;
                };

                // Skip if both coordinates already in rectangle (atom already covered).
                if xs.contains(&x) && ys.contains(&y) {
                    continue;
                }

                if !self.try_add_point(&mut xs, &mut ys, movers, x, y) {
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

    /// A conveyor chain `a→b, b→c` where the atoms sit on `a` and `b`.
    ///
    /// `is_valid_rect` accepts the **complete** 2-cell rectangle: `a`'s
    /// destination `b` is occupied, but `b` is one of the rectangle's own
    /// moving sources, so it vacates in the same shot. Yet `build_aod_grids`
    /// still fails to emit it — see
    /// [`greedy_init_cannot_assemble_a_chain_rectangle`] — which is why no
    /// generator assembles chains today (issue #887).
    fn chain_context() -> BusGridContext<'static> {
        // Positions: a=(0,0), b=(1,0) — same row, adjacent columns.
        // Lanes: a→b, b→c. Atoms occupy a and b; c is free.
        const A: u64 = 10;
        const B: u64 = 20;
        const C: u64 = 30;
        make_context_with_endpoints(
            &[((0, 0), A), ((1, 0), B)],
            &[(A, 101, B), (B, 102, C)],
            &[A, B],
        )
    }

    /// The rule itself admits a chain: the full rectangle is valid.
    #[test]
    fn is_valid_rect_accepts_a_complete_chain_rectangle() {
        let ctx = chain_context();
        let movers: HashSet<u64> = [10u64, 20].into_iter().collect();
        let xs: BTreeSet<u64> = [0u64, 1].into_iter().collect();
        let ys: BTreeSet<u64> = [0u64].into_iter().collect();
        assert!(
            ctx.is_valid_rect(&xs, &ys, &movers),
            "a→b, b→c is legal: b vacates in the same shot"
        );
        // ...but the leader's cell *alone* is not, because b is not yet one of
        // the rectangle's sources. This non-monotonicity is the trap below.
        let xs_leader: BTreeSet<u64> = [0u64].into_iter().collect();
        assert!(
            !ctx.is_valid_rect(&xs_leader, &ys, &movers),
            "the chain's prefix is invalid in isolation"
        );
    }

    /// Growth repairs the chain (issue #887): the leader's cell is invalid on
    /// its own, so `try_add_point` pulls in the follower's cell — the exact
    /// thing that supplies the missing exemption — and the complete rectangle
    /// is emitted as one AOD shot.
    ///
    /// Before the repair closure, `greedy_init` set the leader aside the
    /// moment its *intermediate* rectangle failed, and only the follower's
    /// single-cell rectangle survived.
    #[test]
    fn greedy_init_assembles_a_chain_rectangle() {
        let ctx = chain_context();
        let entries: HashMap<u64, u64> = [(10u64, 101u64), (20, 102)].into_iter().collect();
        let grids = ctx.build_aod_grids(&entries);
        assert_eq!(grids.len(), 1, "expected one rectangle: {grids:?}");
        let mut lanes = grids[0].clone();
        lanes.sort_unstable();
        assert_eq!(
            lanes,
            vec![101, 102],
            "both chain hops must ride in one rectangle"
        );
    }

    /// The repair is not a blanket pass for occupied destinations: it only
    /// fires when the occupant actually moves in this group. Here the atom
    /// ahead has no lane at all, so no amount of growth helps and the
    /// rectangle is rejected outright.
    #[test]
    fn growth_does_not_repair_a_stationary_blocker() {
        const A: u64 = 10;
        const B: u64 = 20;
        const C: u64 = 30;
        // Same geometry as `chain_context`, but B is occupied by an atom that
        // is *not* offered as a mover.
        let ctx = make_context_with_endpoints(
            &[((0, 0), A), ((1, 0), B)],
            &[(A, 101, B), (B, 102, C)],
            &[A, B],
        );
        // Only A is a mover; B stays put.
        let entries: HashMap<u64, u64> = [(A, 101u64)].into_iter().collect();
        let grids = ctx.build_aod_grids(&entries);
        assert!(
            grids.is_empty(),
            "landing on a stationary atom must stay rejected: {grids:?}"
        );
    }

    /// Sorted lanes of every emitted grid, for order-insensitive assertions.
    fn sorted_grids(grids: &[Vec<u64>]) -> Vec<Vec<u64>> {
        let mut out: Vec<Vec<u64>> = grids
            .iter()
            .map(|g| {
                let mut g = g.clone();
                g.sort_unstable();
                g
            })
            .collect();
        out.sort();
        out
    }

    /// A three-hop chain `a→b→c→d` with atoms on a, b and c. Repair has to
    /// cascade twice: pulling in b exposes b's own blocked destination, which
    /// pulls in c.
    #[test]
    fn growth_repairs_a_three_hop_chain() {
        const A: u64 = 10;
        const B: u64 = 20;
        const C: u64 = 30;
        const D: u64 = 40;
        let ctx = make_context_with_endpoints(
            &[((0, 0), A), ((1, 0), B), ((2, 0), C)],
            &[(A, 101, B), (B, 102, C), (C, 103, D)],
            &[A, B, C],
        );
        let entries: HashMap<u64, u64> = [(A, 101u64), (B, 102), (C, 103)].into_iter().collect();
        assert_eq!(
            sorted_grids(&ctx.build_aod_grids(&entries)),
            vec![vec![101, 102, 103]],
            "all three hops must ride in one rectangle"
        );
    }

    /// The outcome must not depend on the order entries happen to be visited
    /// in: a chain assembles whether the leader or the follower is seen first.
    #[test]
    fn chain_assembly_is_entry_order_independent() {
        const A: u64 = 10;
        const B: u64 = 20;
        const C: u64 = 30;
        let ctx = make_context_with_endpoints(
            &[((0, 0), A), ((1, 0), B)],
            &[(A, 101, B), (B, 102, C)],
            &[A, B],
        );
        // `entries` is a HashMap, so build both insertion orders explicitly.
        let leader_first: HashMap<u64, u64> = [(A, 101u64), (B, 102)].into_iter().collect();
        let follower_first: HashMap<u64, u64> = [(B, 102u64), (A, 101)].into_iter().collect();
        assert_eq!(
            sorted_grids(&ctx.build_aod_grids(&leader_first)),
            sorted_grids(&ctx.build_aod_grids(&follower_first))
        );
        assert_eq!(
            sorted_grids(&ctx.build_aod_grids(&leader_first)),
            vec![vec![101, 102]]
        );
    }

    /// A chain whose *head* is blocked by a stationary atom cannot move at
    /// all: the cascade reaches the head, finds a non-mover in the way, and
    /// unwinds. No partial rectangle may be emitted — a partial one would
    /// drive the trailing atoms into their neighbours.
    #[test]
    fn a_blocked_head_rejects_the_whole_chain() {
        const A: u64 = 10;
        const B: u64 = 20;
        const C: u64 = 30;
        const D: u64 = 40;
        let ctx = make_context_with_endpoints(
            &[((0, 0), A), ((1, 0), B), ((2, 0), C)],
            &[(A, 101, B), (B, 102, C), (C, 103, D)],
            // D is occupied by an atom that is never offered as a mover.
            &[A, B, C, D],
        );
        let entries: HashMap<u64, u64> = [(A, 101u64), (B, 102), (C, 103)].into_iter().collect();
        assert!(
            ctx.build_aod_grids(&entries).is_empty(),
            "a chain with nowhere to go must emit nothing"
        );
    }

    /// An independent mover on the same row simply extends the rectangle:
    /// adding its column forces only its own cell, which is free.
    #[test]
    fn a_compatible_extra_point_joins_the_chain_rectangle() {
        const A: u64 = 10;
        const B: u64 = 20;
        const C: u64 = 30;
        const E: u64 = 50;
        const F: u64 = 60;
        let ctx = make_context_with_endpoints(
            &[((0, 0), A), ((1, 0), B), ((2, 0), E)],
            &[(A, 101, B), (B, 102, C), (E, 104, F)],
            &[A, B, E],
        );
        let entries: HashMap<u64, u64> = [(A, 101u64), (B, 102), (E, 104)].into_iter().collect();
        assert_eq!(
            sorted_grids(&ctx.build_aod_grids(&entries)),
            vec![vec![101, 102, 104]],
            "an unobstructed neighbour rides along in the same shot"
        );
    }

    /// **Cascade** — the case a reverse-topological *sort* would get wrong.
    ///
    /// Two rows. Repairing the leader in row 0 pulls in column 1, which forces
    /// the cell at (1, row 1) as well; that cell's own destination is occupied
    /// by a third mover, forcing column 2, and so on. The rectangle only
    /// becomes valid once the whole cascade is resolved, which is why the fix
    /// has to iterate to a fixpoint rather than just order the entries.
    #[test]
    fn growth_repairs_a_multi_row_cascade() {
        // Row 0: a0 → b0 → c0.  Row 1: a1 → b1 → c1.
        // Atoms sit on a0, b0, a1, b1; the c column is free.
        const A0: u64 = 10;
        const B0: u64 = 20;
        const C0: u64 = 30;
        const A1: u64 = 11;
        const B1: u64 = 21;
        const C1: u64 = 31;
        let ctx = make_context_with_endpoints(
            &[((0, 0), A0), ((1, 0), B0), ((0, 1), A1), ((1, 1), B1)],
            &[(A0, 101, B0), (B0, 102, C0), (A1, 111, B1), (B1, 112, C1)],
            &[A0, B0, A1, B1],
        );

        // Offer only the two leaders first in iteration order; the followers
        // must be pulled in by the repair loop, in both rows.
        let entries: HashMap<u64, u64> = [(A0, 101u64), (B0, 102), (A1, 111), (B1, 112)]
            .into_iter()
            .collect();
        let grids = ctx.build_aod_grids(&entries);
        assert_eq!(grids.len(), 1, "expected one 2×2 rectangle: {grids:?}");
        let mut lanes = grids[0].clone();
        lanes.sort_unstable();
        assert_eq!(
            lanes,
            vec![101, 102, 111, 112],
            "every cell of the 2×2 rectangle must ride together"
        );
    }

    /// A wider cascade: three columns × two rows, every row a chain. Repairing
    /// row 0 drags in columns that force row 1's cells, which need repairs of
    /// their own — the closure has to settle all six cells at once.
    #[test]
    fn growth_repairs_a_three_by_two_cascade() {
        // Row 0: a0→b0→c0→(free).  Row 1: a1→b1→c1→(free).
        let (a0, b0, c0, free0) = (10u64, 20, 30, 40);
        let (a1, b1, c1, free1) = (11u64, 21, 31, 41);
        let ctx = make_context_with_endpoints(
            &[
                ((0, 0), a0),
                ((1, 0), b0),
                ((2, 0), c0),
                ((0, 1), a1),
                ((1, 1), b1),
                ((2, 1), c1),
            ],
            &[
                (a0, 101, b0),
                (b0, 102, c0),
                (c0, 103, free0),
                (a1, 111, b1),
                (b1, 112, c1),
                (c1, 113, free1),
            ],
            &[a0, b0, c0, a1, b1, c1],
        );
        let entries: HashMap<u64, u64> = [
            (a0, 101u64),
            (b0, 102),
            (c0, 103),
            (a1, 111),
            (b1, 112),
            (c1, 113),
        ]
        .into_iter()
        .collect();
        assert_eq!(
            sorted_grids(&ctx.build_aod_grids(&entries)),
            vec![vec![101, 102, 103, 111, 112, 113]],
            "all six cells of the 3×2 rectangle must ride together"
        );
    }

    /// An extra mover whose column has **no site in the second row** cannot
    /// join a two-row rectangle: the forced cell does not resolve. It must be
    /// rolled back cleanly and end up in its own cluster, leaving the chain
    /// rectangle intact — a non-fitting point must never corrupt or block the
    /// chain that already assembled.
    #[test]
    fn a_point_that_cannot_fit_gets_its_own_cluster() {
        let (a0, b0, c0) = (10u64, 20, 30);
        let (a1, b1, c1) = (11u64, 21, 31);
        // `e` sits at column 2 of row 0 only — there is no position at (2, 1),
        // so widening the 2-row rectangle to include it leaves a hole.
        let (e, f) = (50u64, 60);
        let ctx = make_context_with_endpoints(
            &[
                ((0, 0), a0),
                ((1, 0), b0),
                ((0, 1), a1),
                ((1, 1), b1),
                ((2, 0), e),
            ],
            &[
                (a0, 101, b0),
                (b0, 102, c0),
                (a1, 111, b1),
                (b1, 112, c1),
                (e, 104, f),
            ],
            &[a0, b0, a1, b1, e],
        );
        let entries: HashMap<u64, u64> = [(a0, 101u64), (b0, 102), (a1, 111), (b1, 112), (e, 104)]
            .into_iter()
            .collect();

        let grids = sorted_grids(&ctx.build_aod_grids(&entries));
        assert!(
            grids.contains(&vec![101, 102, 111, 112]),
            "the 2×2 chain must still assemble: {grids:?}"
        );
        assert!(
            grids.contains(&vec![104]),
            "the non-fitting mover must still get its own rectangle: {grids:?}"
        );
        assert_eq!(grids.len(), 2, "and nothing else: {grids:?}");
    }

    /// Two chains on different rows *and* different columns cannot share a
    /// rectangle — the Cartesian product would demand cells that do not
    /// exist — so each becomes its own AOD operation. This is the limit of
    /// what one shot can batch: chains combine only when their rows and
    /// columns line up into a full grid.
    #[test]
    fn two_misaligned_chains_form_separate_rectangles() {
        // Row 0 chain occupies columns 0–1; row 1 chain occupies columns 2–3.
        let (a0, b0, free0) = (10u64, 20, 30);
        let (a1, b1, free1) = (11u64, 21, 31);
        let ctx = make_context_with_endpoints(
            &[((0, 0), a0), ((1, 0), b0), ((2, 1), a1), ((3, 1), b1)],
            &[
                (a0, 101, b0),
                (b0, 102, free0),
                (a1, 111, b1),
                (b1, 112, free1),
            ],
            &[a0, b0, a1, b1],
        );
        let entries: HashMap<u64, u64> = [(a0, 101u64), (b0, 102), (a1, 111), (b1, 112)]
            .into_iter()
            .collect();
        assert_eq!(
            sorted_grids(&ctx.build_aod_grids(&entries)),
            vec![vec![101, 102], vec![111, 112]],
            "misaligned chains cannot share one rectangle"
        );
    }

    /// A repair may pull in a mover that an **earlier cluster already claimed**.
    /// That is legal and intended: each rectangle is an *alternative* candidate
    /// the search chooses between, not a step executed alongside the others, so
    /// the same atom may appear in several of them.
    ///
    /// Here the follower `b` first joins a rectangle with `p` (a neighbour in
    /// its own column), and is then pulled into a second rectangle when the
    /// leader `a` — set aside in the first pass — repairs itself.
    #[test]
    fn a_repair_may_reuse_a_mover_from_an_earlier_cluster() {
        // Visit order is by encoded source, so p (5) is seen before a (10).
        let (p, p_dst) = (5u64, 55);
        let (a, b, c) = (10u64, 20, 30);
        let ctx = make_context_with_endpoints(
            // (0,1) is a hole: no site there, so no rectangle may span both
            // column 0 and row 1.
            &[((1, 1), p), ((0, 0), a), ((1, 0), b)],
            &[(p, 105, p_dst), (a, 101, b), (b, 102, c)],
            &[p, a, b],
        );
        let entries: HashMap<u64, u64> = [(p, 105u64), (a, 101), (b, 102)].into_iter().collect();

        let grids = sorted_grids(&ctx.build_aod_grids(&entries));
        assert!(
            grids.contains(&vec![102, 105]),
            "b should first ride with p: {grids:?}"
        );
        assert!(
            grids.contains(&vec![101, 102]),
            "and b's cell is reused when a repairs itself: {grids:?}"
        );
        assert_eq!(grids.len(), 2, "exactly two alternatives: {grids:?}");
    }

    /// The result is **not a covering**: a mover that fits in no valid
    /// rectangle is silently dropped rather than emitted alone or reported.
    /// Callers that assume every offered mover comes back in some rectangle
    /// would be wrong — the search relies on this only as a source of
    /// candidates, so an unplaceable atom simply yields no candidate for it.
    #[test]
    fn unplaceable_movers_are_omitted_not_reported() {
        let (a, b) = (10u64, 20);
        let (e, f) = (50u64, 60);
        let ctx = make_context_with_endpoints(
            &[((0, 0), a), ((1, 0), e)],
            &[(a, 101, b), (e, 104, f)],
            // `b` is free so `a` can move; `f` holds a stationary atom, so `e`
            // is stuck no matter how the rectangle grows.
            &[a, e, f],
        );
        let entries: HashMap<u64, u64> = [(a, 101u64), (e, 104)].into_iter().collect();

        let grids = sorted_grids(&ctx.build_aod_grids(&entries));
        assert_eq!(
            grids,
            vec![vec![101]],
            "the movable atom is offered; the stuck one vanishes silently"
        );
    }

    /// **Limit of the algorithm, documented deliberately.** The grid layer has
    /// no acyclicity check: handed a *rotation* (`a→b, b→a`, both occupied) it
    /// assembles it happily, because each atom's destination is vacated by the
    /// other in the same shot — locally the rule is satisfied.
    ///
    /// Nothing here prevents that; what prevents it is `ArchSpec::validate`
    /// rejecting cyclic buses outright (#874), so a rotation can never reach
    /// this code from a legal spec. This test exists to make that dependency
    /// visible: if the arch-level check were ever weakened, the search would
    /// silently emit physically impossible rotations rather than failing.
    #[test]
    fn the_grid_layer_relies_on_arch_level_acyclicity() {
        const A: u64 = 10;
        const B: u64 = 20;
        let ctx = make_context_with_endpoints(
            &[((0, 0), A), ((1, 0), B)],
            // A rotation: a→b and b→a.
            &[(A, 101, B), (B, 102, A)],
            &[A, B],
        );
        let entries: HashMap<u64, u64> = [(A, 101u64), (B, 102)].into_iter().collect();
        assert_eq!(
            sorted_grids(&ctx.build_aod_grids(&entries)),
            vec![vec![101, 102]],
            "the grid layer cannot tell a rotation from a chain — #874 must"
        );
    }

    /// The same question from the other direction: an extra mover that *is*
    /// geometrically compatible but whose destination is held by a stationary
    /// atom. Growth cannot repair it, so it must be rolled back without
    /// disturbing the chain — and must not be emitted on its own either.
    #[test]
    fn a_blocked_extra_point_does_not_disturb_the_chain() {
        let (a, b, c) = (10u64, 20, 30);
        let (e, f) = (50u64, 60);
        let ctx = make_context_with_endpoints(
            &[((0, 0), a), ((1, 0), b), ((2, 0), e)],
            &[(a, 101, b), (b, 102, c), (e, 104, f)],
            // `f` holds a stationary atom, so `e` has nowhere to go.
            &[a, b, e, f],
        );
        let entries: HashMap<u64, u64> = [(a, 101u64), (b, 102), (e, 104)].into_iter().collect();

        let grids = sorted_grids(&ctx.build_aod_grids(&entries));
        assert_eq!(
            grids,
            vec![vec![101, 102]],
            "the chain assembles; the blocked mover is dropped, not batched"
        );
    }

    /// Helper: build a BusGridContext from raw position/lane/collision data.
    fn make_context(
        positions: &[((u64, u64), u64)], // ((x, y), src_encoded)
        lanes: &[(u64, u64)],            // (src_encoded, lane_encoded)
        collisions: &[u64],              // src_encoded values with occupied destinations
    ) -> BusGridContext<'static> {
        make_context_with_occupied(positions, lanes, collisions, &[])
    }

    fn make_context_with_occupied(
        positions: &[((u64, u64), u64)], // ((x, y), src_encoded)
        lanes: &[(u64, u64)],            // (src_encoded, lane_encoded)
        collisions: &[u64],              // src_encoded values with stationary occupied destinations
        occupied: &[u64],                // encoded locations occupied by stationary atoms
    ) -> BusGridContext<'static> {
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
    ) -> BusGridContext<'static> {
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
            maps: Cow::Owned(BusGridMaps {
                pos_to_src,
                src_to_lane,
                src_to_dst,
                src_to_pos,
            }),
            occupied_locs: Cow::Owned(occupied_locs),
            cells_scratch: RefCell::new(Vec::new()),
            srcs_scratch: RefCell::new(Vec::new()),
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
