//! Exhaustive move generator: the search space made executable.
//!
//! [`ExhaustiveGenerator`] emits, for a configuration, every valid shot the
//! hardware model accepts — the set `E_ex(c)` of the branch-and-bound spec —
//! one representative per bus group and child configuration. A shot is valid
//! when it passes the static lane-group rules (S1–S5, `ArchSpec::check_lanes`),
//! the execution model's occupancy rules (D1–D3, `validate_moves`) and the
//! three search-model rules: blocked sites are immovable and never a source
//! (B1), every shot moves an atom (B2), and a shot spans at most the AOD's
//! tone capacity per axis (B3). Among the valid shots that move the same
//! atoms to the same places only the *mover-tight* rectangle — the one with
//! the fewest filler lanes — is emitted; under the objective contract's C5 it
//! is never costlier than the others.
//!
//! The generator's model lays each bus group's lanes out as cells on the
//! group's source grid. That model is well defined only under two properties
//! of the architecture that `ArchSpec::validate` does not enforce:
//! [`ExhaustivePrecondition::PositionCollision`] (P1) and
//! [`ExhaustivePrecondition::NonSeparable`] (P2). [`ExhaustiveGenerator::for_solve`]
//! checks them and refuses a spec that violates either, rather than
//! enumerate a model that does not fit it.
//!
//! Two knobs give nested families of shot sets and are the levels a widening
//! schedule is built from: the [`SeedPolicy`] (which atoms may anchor a
//! rectangle) and the capacity (how large a rectangle may be).

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;

use bloqade_lanes_bytecode_core::arch::addr::{Direction, LaneAddr, LocationAddr, MoveType};

use crate::primitives::config::Config;
use crate::primitives::context::{AodCapacity, MoveCandidate, SearchContext, SearchState};
use crate::primitives::graph::{MoveSet, NodeId};
use crate::primitives::lane_index::LaneIndex;
use crate::traits::MoveGenerator;

#[cfg(test)]
mod oracle;

/// Which atoms may anchor a rectangle.
///
/// Under `Any` every atom at a lane source may; the output is all of `E_ex`.
/// Under `Unresolved` only atoms not yet at their target may seed a shot:
/// resolved atoms join a rectangle only when the tone geometry or a conveyor
/// dependency forces them in. `Unresolved ⊆ Any` by definition, and the two
/// coincide when no atom is resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SeedPolicy {
    /// Only atoms away from their target anchor a shot.
    Unresolved,
    /// Any atom at a lane source anchors a shot.
    Any,
}

/// A bus group: the four fields every lane of one shot must share (S3).
///
/// Ordered by field in this order, which is the emission order of the
/// generator's groups.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GroupKey {
    pub move_type: MoveType,
    pub bus_id: u32,
    pub zone_id: u32,
    pub direction: Direction,
}

impl GroupKey {
    /// The group a lane belongs to.
    pub fn of(lane: &LaneAddr) -> Self {
        Self {
            move_type: lane.move_type,
            bus_id: lane.bus_id,
            zone_id: lane.zone_id,
            direction: lane.direction,
        }
    }
}

impl fmt::Display for GroupKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:?} bus {} zone {} {:?}",
            self.move_type, self.bus_id, self.zone_id, self.direction
        )
    }
}

/// Why the exhaustive model does not fit an architecture.
///
/// Both are properties of the spec alone. They hold on every bundled Gemini
/// spec, and the format does not require either, so they are checked rather
/// than assumed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExhaustivePrecondition {
    /// P1: two lane sources of one bus group share a physical position.
    ///
    /// "Cells on the source grid" is then not well defined: two lanes occupy
    /// one cell, a position-keyed map keeps one of them, and the validator's
    /// geometry check — a set comparison over positions — cannot see two tones
    /// at one point either.
    PositionCollision {
        group: GroupKey,
        a: LocationAddr,
        b: LocationAddr,
    },
    /// P2: the group's source→destination position map does not carry
    /// rectangles to rectangles.
    ///
    /// Two cells in one column with different destination columns, or two
    /// cells in one row with different destination rows. The validator checks
    /// a shot's geometry on the lane-address locations, which for a backward
    /// lane are the shot's destinations; the generators build rectangles on
    /// sources. The two agree only when the map is separable. `witness` is a
    /// pair of lane sources that breaks it.
    NonSeparable {
        group: GroupKey,
        witness: (LocationAddr, LocationAddr),
    },
}

impl fmt::Display for ExhaustivePrecondition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PositionCollision { group, a, b } => write!(
                f,
                "P1 violated on {group}: lane sources {a:?} and {b:?} share a physical position"
            ),
            Self::NonSeparable { group, witness } => write!(
                f,
                "P2 violated on {group}: the source→destination map is not separable \
                 (sources {:?} and {:?} share a column or row but their destinations do not)",
                witness.0, witness.1
            ),
        }
    }
}

impl std::error::Error for ExhaustivePrecondition {}

/// Index of a cell within its group's `cells`.
type CellIdx = u32;

/// A lane of a bus group laid out on the group's source grid.
#[derive(Debug, Clone)]
#[allow(dead_code)] // read by the closure enumeration, which lands in the next commit
struct Cell {
    /// Index into `GroupTables::cols` of the source column.
    col: u32,
    /// Index into `GroupTables::rows` of the source row.
    row: u32,
    src_enc: u64,
    dst_enc: u64,
    lane_enc: u64,
    /// The cell whose *source* is this cell's destination, if any: firing
    /// this cell while that source holds an atom forces that cell too
    /// (the conveyor dependency). Static geometry; whether it is active at a
    /// node depends on occupancy.
    dep: Option<CellIdx>,
}

/// One bus group's geometry and the cells `blocked` kills, built once per
/// solve. Sizes come from the spec: `cols × rows` cells, however many that is.
#[allow(dead_code)] // read by the closure enumeration, which lands in the next commit
struct GroupTables {
    key: GroupKey,
    /// Distinct source x positions (bit patterns), sorted.
    cols: Vec<u64>,
    /// Distinct source y positions (bit patterns), sorted.
    rows: Vec<u64>,
    cells: Vec<Cell>,
    /// `cell_at[col * rows.len() + row]`.
    cell_at: Vec<Option<CellIdx>>,
    src_to_cell: HashMap<u64, CellIdx>,
    dst_to_cell: HashMap<u64, CellIdx>,
    /// Cells that can never fire in this solve: source or destination blocked
    /// (B1, and D2 against an immovable occupant).
    dead_static: Vec<bool>,
}

impl GroupTables {
    fn build(key: GroupKey, lanes: &[LaneAddr], index: &LaneIndex, blocked: &HashSet<u64>) -> Self {
        // Sorted by lane encoding so cell indices are deterministic.
        let mut resolved: Vec<(u64, LaneAddr, LocationAddr, LocationAddr, u64, u64)> = lanes
            .iter()
            .map(|lane| {
                let (src, dst) = index
                    .endpoints(lane)
                    .expect("a registered lane resolves to endpoints");
                let (x, y) = index.position(src).expect("a lane source has a position");
                (lane.encode_u64(), *lane, src, dst, x.to_bits(), y.to_bits())
            })
            .collect();
        resolved.sort_unstable_by_key(|r| r.0);

        let cols: Vec<u64> = resolved
            .iter()
            .map(|r| r.4)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let rows: Vec<u64> = resolved
            .iter()
            .map(|r| r.5)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();

        let mut cells: Vec<Cell> = Vec::with_capacity(resolved.len());
        let mut cell_at: Vec<Option<CellIdx>> = vec![None; cols.len() * rows.len()];
        let mut src_to_cell: HashMap<u64, CellIdx> = HashMap::with_capacity(resolved.len());
        let mut dst_to_cell: HashMap<u64, CellIdx> = HashMap::with_capacity(resolved.len());
        for (lane_enc, _lane, src, dst, xb, yb) in &resolved {
            let col = cols.binary_search(xb).expect("column is in cols") as u32;
            let row = rows.binary_search(yb).expect("row is in rows") as u32;
            let idx = cells.len() as CellIdx;
            let slot = &mut cell_at[col as usize * rows.len() + row as usize];
            debug_assert!(slot.is_none(), "P1 was checked: one lane per cell");
            *slot = Some(idx);
            src_to_cell.insert(src.encode(), idx);
            dst_to_cell.insert(dst.encode(), idx);
            cells.push(Cell {
                col,
                row,
                src_enc: src.encode(),
                dst_enc: dst.encode(),
                lane_enc: *lane_enc,
                dep: None,
            });
        }
        for cell in &mut cells {
            cell.dep = src_to_cell.get(&cell.dst_enc).copied();
        }
        let dead_static = cells
            .iter()
            .map(|c| blocked.contains(&c.src_enc) || blocked.contains(&c.dst_enc))
            .collect();

        Self {
            key,
            cols,
            rows,
            cells,
            cell_at,
            src_to_cell,
            dst_to_cell,
            dead_static,
        }
    }
}

/// Exhaustive AOD-rectangle move generator: emits `E_ex(c)` at a
/// configuration, one mover-tight representative per bus group and child
/// configuration, at the generator's seed policy and capacity.
///
/// Built per solve by [`Self::for_solve`], which checks the architecture
/// preconditions and precomputes each bus group's geometry and the cells the
/// solve's blocked sites kill. Plain data afterwards, `Sync`, and shared by
/// reference across parallel restarts.
#[derive(Debug)]
pub struct ExhaustiveGenerator {
    seed: SeedPolicy,
    /// The generator's own capacity; the effective cap at a node is
    /// `AodCapacity::tighten(self.cap, ctx.capacity)`.
    cap: Option<AodCapacity>,
    /// One entry per bus group with lanes, sorted by [`GroupKey`].
    #[allow(dead_code)] // read by the closure enumeration, which lands in the next commit
    groups: Vec<GroupTables>,
}

impl fmt::Debug for GroupTables {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GroupTables")
            .field("key", &self.key)
            .field("cols", &self.cols.len())
            .field("rows", &self.rows.len())
            .field("cells", &self.cells.len())
            .finish_non_exhaustive()
    }
}

impl ExhaustiveGenerator {
    /// Check the architecture preconditions the enumeration model rests on.
    ///
    /// Per bus group: **P1**, distinct lane sources have distinct positions;
    /// **P2**, cells in one source column share a destination column and cells
    /// in one source row share a destination row (so rectangles map to
    /// rectangles). Every group is checked in both directions, since the index
    /// registers a backward twin for every lane, which is what makes P1 on
    /// sources also cover destinations.
    ///
    /// A property of the spec alone; `SearchEngine` caches the verdict.
    pub fn check_preconditions(index: &LaneIndex) -> Result<(), ExhaustivePrecondition> {
        for (group, lanes) in sorted_groups(index) {
            let mut pos_to_src: HashMap<(u64, u64), LocationAddr> = HashMap::new();
            let mut col_to_dst_x: HashMap<u64, (u64, LocationAddr)> = HashMap::new();
            let mut row_to_dst_y: HashMap<u64, (u64, LocationAddr)> = HashMap::new();
            for lane in lanes {
                let (src, dst) = index
                    .endpoints(&lane)
                    .expect("a registered lane resolves to endpoints");
                let (sx, sy) = index.position(src).expect("a lane source has a position");
                let (dx, dy) = index
                    .position(dst)
                    .expect("a lane destination has a position");
                let (sx, sy, dx, dy) = (sx.to_bits(), sy.to_bits(), dx.to_bits(), dy.to_bits());
                if let Some(&other) = pos_to_src.get(&(sx, sy)) {
                    return Err(ExhaustivePrecondition::PositionCollision {
                        group,
                        a: other,
                        b: src,
                    });
                }
                pos_to_src.insert((sx, sy), src);
                match col_to_dst_x.get(&sx) {
                    Some(&(other_dx, other_src)) if other_dx != dx => {
                        return Err(ExhaustivePrecondition::NonSeparable {
                            group,
                            witness: (other_src, src),
                        });
                    }
                    Some(_) => {}
                    None => {
                        col_to_dst_x.insert(sx, (dx, src));
                    }
                }
                match row_to_dst_y.get(&sy) {
                    Some(&(other_dy, other_src)) if other_dy != dy => {
                        return Err(ExhaustivePrecondition::NonSeparable {
                            group,
                            witness: (other_src, src),
                        });
                    }
                    Some(_) => {}
                    None => {
                        row_to_dst_y.insert(sy, (dy, src));
                    }
                }
            }
        }
        Ok(())
    }

    /// Build the generator for one solve.
    ///
    /// Runs [`Self::check_preconditions`] on `ctx.index`, then precomputes
    /// every bus group's cells and the cells `ctx.blocked` kills. `seed` and
    /// `cap` are the generator's level; a `cap` of `None` is unlimited, and
    /// the solve's own `ctx.capacity` still applies on top.
    pub fn for_solve(
        ctx: &SearchContext<'_>,
        seed: SeedPolicy,
        cap: Option<AodCapacity>,
    ) -> Result<Self, ExhaustivePrecondition> {
        Self::check_preconditions(ctx.index)?;
        let groups = sorted_groups(ctx.index)
            .into_iter()
            .map(|(key, lanes)| GroupTables::build(key, &lanes, ctx.index, ctx.blocked))
            .collect();
        Ok(Self { seed, cap, groups })
    }

    pub fn seed(&self) -> SeedPolicy {
        self.seed
    }

    pub fn capacity(&self) -> Option<AodCapacity> {
        self.cap
    }

    /// The bus groups with lanes, in emission order.
    pub fn group_keys(&self) -> impl Iterator<Item = GroupKey> + '_ {
        self.groups.iter().map(|g| g.key)
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

/// Every bus group of the index with its lanes, sorted by key. `LaneIndex`
/// walks a `HashMap`, so the order has to be imposed here.
fn sorted_groups(index: &LaneIndex) -> Vec<(GroupKey, Vec<LaneAddr>)> {
    let mut groups: BTreeMap<GroupKey, Vec<LaneAddr>> = BTreeMap::new();
    for (mt, bus_id, zone_id, dir) in index.bus_groups() {
        let key = GroupKey {
            move_type: mt,
            bus_id,
            zone_id,
            direction: dir,
        };
        let lanes = index.lanes_for(mt, bus_id, zone_id, dir).to_vec();
        if !lanes.is_empty() {
            groups.insert(key, lanes);
        }
    }
    groups.into_iter().collect()
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
        let cap = AodCapacity::tighten(self.cap, ctx.capacity);
        let expand_ctx = ExpandContext {
            occupied: Self::occupied_set(config, ctx.blocked),
            loc_to_qubit: config.location_to_qubit_map(),
            config,
            index: ctx.index,
            max_x_capacity: cap.map(|c| c.x),
            max_y_capacity: cap.map(|c| c.y),
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

    for &lane in lanes {
        let Some((src, _dst)) = ctx.index.endpoints(&lane) else {
            continue;
        };
        let Some((x, y)) = ctx.index.position(src) else {
            continue;
        };
        let xb = x.to_bits();
        let yb = y.to_bits();
        pos_to_info.insert((xb, yb), (src, lane));
        unique_x.insert(xb);
        unique_y.insert(yb);
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

    let td = TripletData { pos_to_info };

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
    // Every cell's (src, dst) plus the sources this rectangle actually moves
    // an atom out of — the group's mover set, needed to judge destinations.
    let mut cells: Vec<(u64, u64)> = Vec::with_capacity(x_subset.len() * y_subset.len());
    let mut mover_srcs: HashSet<u64> = HashSet::new();

    for &xb in x_subset {
        for &yb in y_subset {
            let Some(&(src, lane)) = td.pos_to_info.get(&(xb, yb)) else {
                return;
            };
            let Some((_, dst)) = ctx.index.endpoints(&lane) else {
                return;
            };
            let src_enc = src.encode();
            lane_addrs.push(lane);
            cells.push((src_enc, dst.encode()));

            if let Some(&qid) = ctx.loc_to_qubit.get(&src_enc) {
                mover_srcs.insert(src_enc);
                moves.push((qid, dst));
            }
        }
    }

    if moves.is_empty() {
        return;
    }

    // Uniform destination rule (#866): every cell's destination — mover and
    // empty-source filler alike — must be free or vacated by this same
    // rectangle. Judged against the pre-move occupancy once the whole mover
    // set is known, which is why it cannot be a per-lane prefilter: a
    // conveyor chain is only legal *because* the atom ahead moves too.
    for &(_, dst_enc) in &cells {
        if !crate::ops::aod_grid::destination_is_available(dst_enc, &ctx.occupied, &mover_srcs) {
            return;
        }
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
    use crate::observer::NoOpObserver;
    use crate::primitives::distance::DistanceTable;
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
            capacity: None,
        }
    }

    /// Run an `Any`-seeded, uncapped generator with the default context (target
    /// `site 5` for qubit 0, no blocked locations).
    fn run_generator(config: &Config, index: &LaneIndex) -> Vec<MoveCandidate> {
        run_generator_blocked(config, index, &[])
    }

    /// As [`run_generator`], with blocked locations.
    fn run_generator_blocked(
        config: &Config,
        index: &LaneIndex,
        blocked_locs: &[LocationAddr],
    ) -> Vec<MoveCandidate> {
        let targets_raw: Vec<(u32, u64)> = vec![(0, loc(0, 5).encode())];
        let target_locs: Vec<u64> = vec![loc(0, 5).encode()];
        let dist_table = DistanceTable::new(&target_locs, index);
        let blocked: HashSet<u64> = blocked_locs.iter().map(|l| l.encode()).collect();
        let ctx = make_ctx(index, &dist_table, &targets_raw, &blocked);
        let generator = ExhaustiveGenerator::for_solve(&ctx, SeedPolicy::Any, None).unwrap();
        let mut state = SearchState::default();
        let mut out = Vec::new();
        generator.generate(config, NodeId(0), &ctx, &mut state, &mut out);
        out
    }

    // ── Preconditions P1 / P2 and per-solve tables ──

    fn ctx_for<'a>(
        index: &'a LaneIndex,
        dist_table: &'a DistanceTable,
        blocked: &'a HashSet<u64>,
    ) -> SearchContext<'a> {
        make_ctx(index, dist_table, &[], blocked)
    }

    fn preconditions_of(json: &str) -> Result<(), ExhaustivePrecondition> {
        let spec: ArchSpec = serde_json::from_str(json).unwrap();
        ExhaustiveGenerator::check_preconditions(&LaneIndex::new(spec))
    }

    /// `full.json` is a legal spec whose words 0–2 coincide: P1 fails.
    #[test]
    fn full_json_violates_p1() {
        let err = preconditions_of(crate::test_utils::full_arch_json()).unwrap_err();
        match &err {
            ExhaustivePrecondition::PositionCollision { a, b, .. } => {
                assert_ne!(a, b);
                assert_ne!(
                    a.word_id, b.word_id,
                    "coincident words are the P1 witness: {err}"
                );
            }
            other => panic!("expected a position collision, got {other}"),
        }
    }

    /// The non-separable bus maps a source row to a diagonal pair: P2 fails,
    /// and P1 holds (so the report is the right one).
    #[test]
    fn non_separable_bus_violates_p2() {
        let err = preconditions_of(crate::test_utils::non_separable_bus_arch_json()).unwrap_err();
        assert!(
            matches!(err, ExhaustivePrecondition::NonSeparable { .. }),
            "expected non-separability, got {err}"
        );
        let text = err.to_string();
        assert!(text.contains("P2"), "{text}");
    }

    /// Every other fixture and both shipped specs satisfy P1 and P2.
    #[test]
    fn fixtures_and_shipped_specs_satisfy_the_preconditions() {
        use crate::test_utils::*;
        let specs: Vec<(&str, String)> = vec![
            ("example", example_arch_json().to_string()),
            ("chain", chain_arch_json()),
            (
                "chain with siding",
                chain_with_siding_arch_json().to_string(),
            ),
            ("two-zone bus", two_zone_bus_arch_json().to_string()),
            (
                "two-zone aligned",
                two_zone_aligned_site_bus_arch_json().to_string(),
            ),
            ("asymmetric durations", asymmetric_duration_arch_json()),
            ("physical", oracle::physical_spec_json().to_string()),
            ("logical", oracle::logical_spec_json().to_string()),
        ];
        for (name, json) in specs {
            assert_eq!(preconditions_of(&json), Ok(()), "{name}");
        }
    }

    /// `for_solve` reports the failing precondition instead of building.
    #[test]
    fn for_solve_refuses_a_spec_that_fails_a_precondition() {
        let spec: ArchSpec =
            serde_json::from_str(crate::test_utils::non_separable_bus_arch_json()).unwrap();
        let index = LaneIndex::new(spec);
        let dist_table = DistanceTable::new(&[], &index);
        let blocked = HashSet::new();
        let ctx = ctx_for(&index, &dist_table, &blocked);
        assert!(matches!(
            ExhaustiveGenerator::for_solve(&ctx, SeedPolicy::Any, None),
            Err(ExhaustivePrecondition::NonSeparable { .. })
        ));
    }

    /// Groups come out sorted by key, so emission order is deterministic
    /// whatever order `LaneIndex` walks its map in.
    #[test]
    fn groups_are_sorted_by_key() {
        for json in [
            crate::test_utils::example_arch_json().to_string(),
            crate::test_utils::two_zone_aligned_site_bus_arch_json().to_string(),
            oracle::physical_spec_json().to_string(),
        ] {
            let spec: ArchSpec = serde_json::from_str(&json).unwrap();
            let index = LaneIndex::new(spec);
            let dist_table = DistanceTable::new(&[], &index);
            let blocked = HashSet::new();
            let ctx = ctx_for(&index, &dist_table, &blocked);
            let generator = ExhaustiveGenerator::for_solve(&ctx, SeedPolicy::Any, None).unwrap();
            let keys: Vec<GroupKey> = generator.group_keys().collect();
            assert!(!keys.is_empty());
            assert!(keys.windows(2).all(|w| w[0] < w[1]), "{keys:?}");
            assert_eq!(keys.len(), index.bus_groups().count());
        }
    }

    /// The per-solve tables: one cell per lane, cells addressable by source,
    /// destination and grid position, conveyor dependencies where a
    /// destination is another cell's source, and blocked sites marking cells
    /// dead.
    #[test]
    fn tables_describe_the_group_geometry() {
        // The chain fixture: site bus 0 is `0→1, 1→2, 2→3, 3→4` on two words.
        let spec: ArchSpec = serde_json::from_str(&crate::test_utils::chain_arch_json()).unwrap();
        let index = LaneIndex::new(spec);
        let dist_table = DistanceTable::new(&[], &index);
        let blocked: HashSet<u64> = [loc(0, 2).encode()].into_iter().collect();
        let ctx = ctx_for(&index, &dist_table, &blocked);
        let generator = ExhaustiveGenerator::for_solve(&ctx, SeedPolicy::Any, None).unwrap();

        let key = GroupKey {
            move_type: MoveType::SiteBus,
            bus_id: 0,
            zone_id: 0,
            direction: Direction::Forward,
        };
        let group = generator
            .groups
            .iter()
            .find(|g| g.key == key)
            .expect("forward chain group");
        // Four chain sources per word, two words: 4 columns × 2 rows.
        assert_eq!(group.cols.len(), 4);
        assert_eq!(group.rows.len(), 2);
        assert_eq!(group.cells.len(), 8);
        assert_eq!(group.cell_at.iter().filter(|c| c.is_some()).count(), 8);
        for (i, cell) in group.cells.iter().enumerate() {
            assert_eq!(group.src_to_cell[&cell.src_enc], i as CellIdx);
            assert_eq!(group.dst_to_cell[&cell.dst_enc], i as CellIdx);
            assert_eq!(
                group.cell_at[cell.col as usize * group.rows.len() + cell.row as usize],
                Some(i as CellIdx)
            );
            // `0→1` depends on `1→2`, …, `3→4` on nothing (site 4 is no source).
            let src = LocationAddr::decode(cell.src_enc);
            let expected_dep = group
                .src_to_cell
                .get(&loc(src.word_id, src.site_id + 1).encode())
                .copied();
            assert_eq!(cell.dep, expected_dep, "dep of {src:?}");
            // Site 2 of word 0 is blocked: the cell sourced there and the cell
            // landing there (`1→2`) are statically dead, nothing else is.
            let dead = src.word_id == 0 && (src.site_id == 2 || src.site_id == 1);
            assert_eq!(group.dead_static[i], dead, "dead_static of {src:?}");
        }
    }

    /// The engine computes the verdict once and hands out the same result.
    #[test]
    fn engine_caches_the_precondition_verdict() {
        use crate::search::engine::SearchEngine;
        let bad = SearchEngine::from_json(crate::test_utils::full_arch_json()).unwrap();
        assert!(bad.exhaustive_preconditions().is_err());
        assert!(std::ptr::eq(
            bad.exhaustive_preconditions(),
            bad.exhaustive_preconditions()
        ));
        let good = SearchEngine::from_json(example_arch_json()).unwrap();
        assert_eq!(good.exhaustive_preconditions(), &Ok(()));
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
        let out = run_generator(&config, &index);

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
        let out = run_generator_blocked(&config, &index, &[loc(0, 5)]);

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
        let out = run_generator(&config, &index);
        assert!(out.is_empty());
    }

    #[test]
    fn generate_rejects_move_onto_stationary_atom() {
        let index = make_index();
        // Qubit 0 at site 0, qubit 1 at site 5 (destination of site 0 forward).
        // Qubit 1 has no lane of its own on this bus, so it cannot vacate in
        // the same shot: the uniform destination rule rejects the rectangle.
        let config = Config::new([(0, loc(0, 0)), (1, loc(0, 5))]).unwrap();
        let out = run_generator(&config, &index);

        // No forward site bus move should move qubit 0 to site 5 (collision).
        let collision = out
            .iter()
            .any(|c| c.new_config.location_of(0) == Some(loc(0, 5)));
        assert!(!collision, "landing on a stationary atom must be rejected");
    }

    fn make_chain_index() -> LaneIndex {
        let spec: ArchSpec = serde_json::from_str(&crate::test_utils::chain_arch_json()).unwrap();
        LaneIndex::new(spec)
    }

    /// On a conveyor-chain bus (0→1, 1→2, …) two adjacent atoms can shift
    /// together in one shot: the leading atom's destination is occupied, but
    /// its occupant moves in the same rectangle. Before #876 the whole
    /// rectangle was discarded because both endpoints were occupied.
    #[test]
    fn generate_admits_conveyor_chain() {
        let index = make_chain_index();
        let config = Config::new([(0, loc(0, 0)), (1, loc(0, 1))]).unwrap();
        let targets_raw: Vec<(u32, u64)> = vec![(0, loc(0, 1).encode())];
        let target_locs: Vec<u64> = vec![loc(0, 1).encode()];
        let dist_table = DistanceTable::new(&target_locs, &index);
        let blocked = HashSet::new();
        let ctx = make_ctx(&index, &dist_table, &targets_raw, &blocked);
        let generator = ExhaustiveGenerator::for_solve(&ctx, SeedPolicy::Any, None).unwrap();
        let mut state = SearchState::default();
        let mut out = Vec::new();
        generator.generate(&config, NodeId(0), &ctx, &mut state, &mut out);

        let has_chain = out.iter().any(|c| {
            c.move_set.len() >= 2
                && c.new_config.location_of(0) == Some(loc(0, 1))
                && c.new_config.location_of(1) == Some(loc(0, 2))
        });
        assert!(
            has_chain,
            "conveyor chain 0→1, 1→2 must be generated as one simultaneous move set"
        );
    }

    /// The chain exemption is not a blanket pass for occupied destinations:
    /// the atom ahead must actually move. Here site 1 holds an atom with no
    /// lane in the group (site 1's own lane is not part of a 1-cell rectangle
    /// at site 0), so the single-cell rectangle at site 0 stays invalid.
    #[test]
    fn generate_chain_still_rejects_stationary_leader() {
        let index = make_chain_index();
        // Qubit 1 sits on site 4 — the end of the chain, which has no
        // outgoing lane on this bus — so qubit 0's 3→4 lane can never fire.
        let config = Config::new([(0, loc(0, 3)), (1, loc(0, 4))]).unwrap();
        let targets_raw: Vec<(u32, u64)> = vec![(0, loc(0, 4).encode())];
        let target_locs: Vec<u64> = vec![loc(0, 4).encode()];
        let dist_table = DistanceTable::new(&target_locs, &index);
        let blocked = HashSet::new();
        let ctx = make_ctx(&index, &dist_table, &targets_raw, &blocked);
        let generator = ExhaustiveGenerator::for_solve(&ctx, SeedPolicy::Any, None).unwrap();
        let mut state = SearchState::default();
        let mut out = Vec::new();
        generator.generate(&config, NodeId(0), &ctx, &mut state, &mut out);

        let lands_on_occupant = out
            .iter()
            .any(|c| c.new_config.location_of(0) == Some(loc(0, 4)));
        assert!(
            !lands_on_occupant,
            "the atom ahead has no lane on this bus, so the move must be rejected"
        );
    }

    #[test]
    fn generate_parallel_moves() {
        let index = make_index();
        // Two qubits at site bus source positions in same word.
        let config = Config::new([(0, loc(0, 0)), (1, loc(0, 1))]).unwrap();
        let out = run_generator(&config, &index);

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
        use crate::drivers::frontier::{self, PriorityFrontier};
        use crate::goals::AllAtTarget;
        use crate::primitives::distance::HopDistanceHeuristic;
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
        let generator = ExhaustiveGenerator::for_solve(&ctx, SeedPolicy::Any, None).unwrap();
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
            None,
        );

        assert!(result.goal.is_some());
        let path = result.solution_path().unwrap();
        // Site 0 -> site 5 is one site bus forward move.
        assert_eq!(path.len(), 1);
    }
}
