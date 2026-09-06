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

use bloqade_lanes_bytecode_core::arch::addr::{LaneAddr, LocationAddr};

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

pub use crate::primitives::ordering::GroupKey;

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
        let cap = (
            cap.map_or(usize::MAX, |c| c.x),
            cap.map_or(usize::MAX, |c| c.y),
        );
        if cap.0 == 0 || cap.1 == 0 {
            return;
        }
        // Resolved atoms sit where `ctx.targets` puts them; everything else,
        // including an atom the targets do not mention, is unresolved.
        let resolved: HashSet<u32> = ctx
            .targets
            .iter()
            .filter(|&&(q, target)| config.location_of(q).is_some_and(|l| l.encode() == target))
            .map(|&(q, _)| q)
            .collect();
        let atoms: Vec<(u32, u64)> = config.iter().map(|(q, l)| (q, l.encode())).collect();

        for tables in &self.groups {
            let Some(node) = NodeGroup::classify(tables, &atoms, &resolved) else {
                continue;
            };
            node.enumerate(self.seed, cap, config, out);
        }
    }
}

/// A fixed-width bitset over one group's rows, sized at classification time.
/// The width is a property of the spec, not a constant of the generator.
#[derive(Clone, PartialEq, Eq, Debug)]
struct Bits {
    words: Vec<u64>,
}

impl Bits {
    fn new(width: usize) -> Self {
        Self {
            words: vec![0; width.div_ceil(64).max(1)],
        }
    }

    fn set(&mut self, i: usize) {
        self.words[i / 64] |= 1 << (i % 64);
    }

    fn contains(&self, i: usize) -> bool {
        (self.words[i / 64] >> (i % 64)) & 1 == 1
    }

    fn intersects(&self, other: &Self) -> bool {
        self.words.iter().zip(&other.words).any(|(a, b)| a & b != 0)
    }

    fn and(&self, other: &Self) -> Self {
        Self {
            words: self
                .words
                .iter()
                .zip(&other.words)
                .map(|(a, b)| a & b)
                .collect(),
        }
    }

    fn or_assign(&mut self, other: &Self) {
        for (a, b) in self.words.iter_mut().zip(&other.words) {
            *a |= b;
        }
    }

    /// Set bits in ascending order.
    fn ones(&self) -> impl Iterator<Item = usize> + '_ {
        self.words.iter().enumerate().flat_map(|(wi, &w)| {
            let mut w = w;
            std::iter::from_fn(move || {
                if w == 0 {
                    None
                } else {
                    let b = w.trailing_zeros() as usize;
                    w &= w - 1;
                    Some(wi * 64 + b)
                }
            })
        })
    }
}

/// One bus group classified at one configuration: which cells hold a mover,
/// which are dead, which conveyor dependencies are active — and the per-column
/// row bitsets the enumeration runs on.
struct NodeGroup<'g> {
    tables: &'g GroupTables,
    /// The qubit at each cell's source, if any.
    mover: Vec<Option<u32>>,
    /// A cell's `dep` is active when the atom at its destination has a lane in
    /// this group; the cell then forces its `dep`.
    active_dep: Vec<bool>,
    /// Rows of each column whose cell exists and can fire.
    live_rows: Vec<Bits>,
    /// Rows of each column whose cell is a live mover cell.
    mover_rows: Vec<Bits>,
    /// Rows of each column whose cell is a live *unresolved* mover cell.
    unresolved_rows: Vec<Bits>,
    /// Columns with at least one live mover cell, ascending.
    mover_cols: Vec<usize>,
}

impl<'g> NodeGroup<'g> {
    /// Classify the cells the atoms touch, `O(atoms)` lookups per group, and
    /// propagate deadness through active dependencies. `None` when no atom
    /// sits at a source of this group — the group cannot move anything.
    fn classify(
        tables: &'g GroupTables,
        atoms: &[(u32, u64)],
        resolved: &HashSet<u32>,
    ) -> Option<Self> {
        let n_cells = tables.cells.len();
        let mut mover: Vec<Option<u32>> = vec![None; n_cells];
        let mut unresolved = vec![false; n_cells];
        let mut dead = tables.dead_static.clone();
        let mut active_dep = vec![false; n_cells];
        let mut any_mover = false;

        for &(qubit, enc) in atoms {
            if let Some(&c) = tables.src_to_cell.get(&enc) {
                mover[c as usize] = Some(qubit);
                unresolved[c as usize] = !resolved.contains(&qubit);
                any_mover = true;
            }
            if let Some(&d) = tables.dst_to_cell.get(&enc) {
                // The atom holds cell `d`'s destination. If it has a lane in
                // this group, `d` may fire only together with that lane (the
                // conveyor rule); otherwise `d` can never fire here.
                if tables.src_to_cell.contains_key(&enc) {
                    debug_assert!(tables.cells[d as usize].dep.is_some());
                    active_dep[d as usize] = true;
                } else {
                    dead[d as usize] = true;
                }
            }
        }
        if !any_mover {
            return None;
        }

        // A cell whose active dependency is dead is dead. Chains are acyclic
        // (bus validation), so this converges in at most chain-length passes.
        loop {
            let mut changed = false;
            for (i, cell) in tables.cells.iter().enumerate() {
                if !dead[i] && active_dep[i] {
                    let dep = cell.dep.expect("an active dep names a cell") as usize;
                    if dead[dep] {
                        dead[i] = true;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }

        let n_rows = tables.rows.len();
        let n_cols = tables.cols.len();
        let mut live_rows = vec![Bits::new(n_rows); n_cols];
        let mut mover_rows = vec![Bits::new(n_rows); n_cols];
        let mut unresolved_rows = vec![Bits::new(n_rows); n_cols];
        for (i, cell) in tables.cells.iter().enumerate() {
            if dead[i] {
                continue;
            }
            let (col, row) = (cell.col as usize, cell.row as usize);
            live_rows[col].set(row);
            if mover[i].is_some() {
                mover_rows[col].set(row);
                if unresolved[i] {
                    unresolved_rows[col].set(row);
                }
            }
        }
        let mover_cols: Vec<usize> = (0..n_cols)
            .filter(|&c| mover_rows[c].ones().next().is_some())
            .collect();
        if mover_cols.is_empty() {
            return None;
        }

        Some(Self {
            tables,
            mover,
            active_dep,
            live_rows,
            mover_rows,
            unresolved_rows,
            mover_cols,
        })
    }

    fn cell(&self, col: usize, row: usize) -> Option<CellIdx> {
        self.tables.cell_at[col * self.tables.rows.len() + row]
    }

    /// Emit every tight, closed rectangle within `cap` that the seed policy
    /// admits, in the order (columns in DFS order, rows in combination order).
    fn enumerate(
        &self,
        seed: SeedPolicy,
        cap: (usize, usize),
        config: &Config,
        out: &mut Vec<MoveCandidate>,
    ) {
        let all_rows = {
            let mut b = Bits::new(self.tables.rows.len());
            for r in 0..self.tables.rows.len() {
                b.set(r);
            }
            b
        };
        let mut x: Vec<usize> = Vec::new();
        self.dfs_columns(&mut x, 0, &all_rows, seed, cap, config, out);
    }

    /// DFS over `X ⊆ mover_cols` in column order. `y_ok` is the intersection
    /// of the live rows of the columns chosen so far.
    #[allow(clippy::too_many_arguments)]
    fn dfs_columns(
        &self,
        x: &mut Vec<usize>,
        start: usize,
        y_ok: &Bits,
        seed: SeedPolicy,
        cap: (usize, usize),
        config: &Config,
        out: &mut Vec<MoveCandidate>,
    ) {
        if !x.is_empty() {
            // Tightness needs every chosen column to keep a mover row inside
            // `y_ok`. Adding columns only shrinks `y_ok`, so if some column
            // has lost all of its mover rows no superset of `X` is tight
            // either: prune the whole subtree.
            if x.iter().any(|&c| !self.mover_rows[c].intersects(y_ok)) {
                return;
            }
            self.emit_rows(x, y_ok, seed, cap, config, out);
        }
        if x.len() == cap.0 {
            return;
        }
        for (i, &col) in self.mover_cols.iter().enumerate().skip(start) {
            let narrowed = y_ok.and(&self.live_rows[col]);
            x.push(col);
            self.dfs_columns(x, i + 1, &narrowed, seed, cap, config, out);
            x.pop();
        }
    }

    /// For a fixed `X`, every `Y ⊆ y_ok` that is tight (`Y` is covered by the
    /// mover rows of `X`, and every column of `X` has a mover row in `Y`),
    /// within `cap.1`, and closed under the active dependencies.
    fn emit_rows(
        &self,
        x: &[usize],
        y_ok: &Bits,
        seed: SeedPolicy,
        cap: (usize, usize),
        config: &Config,
        out: &mut Vec<MoveCandidate>,
    ) {
        // Rows that hold a mover in some chosen column, restricted to y_ok:
        // the only rows a tight Y may use.
        let mut covered = Bits::new(self.tables.rows.len());
        for &c in x {
            covered.or_assign(&self.mover_rows[c]);
        }
        let candidates: Vec<usize> = covered.and(y_ok).ones().collect();
        let k_max = cap.1.min(candidates.len());

        // Combinations of `candidates` of every size 1..=k_max, in
        // lexicographic order.
        let mut idx: Vec<usize> = Vec::new();
        for k in 1..=k_max {
            idx.clear();
            idx.extend(0..k);
            loop {
                let y: Vec<usize> = idx.iter().map(|&i| candidates[i]).collect();
                self.try_emit(x, &y, seed, config, out);
                if !next_combination(&mut idx, candidates.len()) {
                    break;
                }
            }
        }
    }

    /// Emit `X×Y` if it is tight in `Y`, closed under active dependencies,
    /// and admitted by the seed policy.
    fn try_emit(
        &self,
        x: &[usize],
        y: &[usize],
        seed: SeedPolicy,
        config: &Config,
        out: &mut Vec<MoveCandidate>,
    ) {
        let n_rows = self.tables.rows.len();
        let mut y_bits = Bits::new(n_rows);
        for &r in y {
            y_bits.set(r);
        }
        // Every column of X must hold a mover in Y (rows of Y are covered by
        // construction of the candidate list).
        if x.iter().any(|&c| !self.mover_rows[c].intersects(&y_bits)) {
            return;
        }

        let in_x = |c: usize| x.binary_search(&c).is_ok();
        let mut lanes: Vec<u64> = Vec::with_capacity(x.len() * y.len());
        let mut moves: Vec<(u32, LocationAddr)> = Vec::new();
        for &c in x {
            for &r in y {
                let idx = self
                    .cell(c, r)
                    .expect("Y ⊆ live rows of every column of X, so the cell exists")
                    as usize;
                let cell = &self.tables.cells[idx];
                if self.active_dep[idx] {
                    let dep = &self.tables.cells[cell.dep.expect("active dep") as usize];
                    if !(in_x(dep.col as usize) && y_bits.contains(dep.row as usize)) {
                        // Not closed: this rectangle's closure is a different,
                        // larger rectangle, emitted under its own (X, Y).
                        return;
                    }
                }
                lanes.push(cell.lane_enc);
                if let Some(q) = self.mover[idx] {
                    moves.push((q, LocationAddr::decode(cell.dst_enc)));
                }
            }
        }
        debug_assert_eq!(lanes.len(), x.len() * y.len(), "a complete product");
        debug_assert!(!moves.is_empty(), "tight rectangles hold a mover (B2)");

        if seed == SeedPolicy::Unresolved && !self.seeded_by_unresolved(x, &y_bits) {
            return;
        }

        out.push(MoveCandidate {
            move_set: MoveSet::from_encoded(lanes),
            new_config: config.with_moves(&moves),
        });
    }

    /// Whether the closure of `X×Y`'s unresolved movers — under the product
    /// rule and the active dependencies — is all of `X×Y`. Resolved atoms may
    /// ride along only when the geometry or a chain forces them.
    fn seeded_by_unresolved(&self, x: &[usize], y: &Bits) -> bool {
        let n_rows = self.tables.rows.len();
        let mut cols = Bits::new(self.tables.cols.len());
        let mut rows = Bits::new(n_rows);
        for &c in x {
            let seeds = self.unresolved_rows[c].and(y);
            if seeds.ones().next().is_some() {
                cols.set(c);
                rows.or_assign(&seeds);
            }
        }
        if rows.ones().next().is_none() {
            return false;
        }
        // Grow: the product of the current cols × rows, plus every active
        // dependency it forces, until nothing new appears.
        loop {
            let mut grew = false;
            for c in cols.ones().collect::<Vec<_>>() {
                for r in rows.ones().collect::<Vec<_>>() {
                    let Some(idx) = self.cell(c, r) else { continue };
                    let idx = idx as usize;
                    if self.active_dep[idx] {
                        let dep = &self.tables.cells
                            [self.tables.cells[idx].dep.expect("active dep") as usize];
                        let (dc, dr) = (dep.col as usize, dep.row as usize);
                        if !cols.contains(dc) {
                            cols.set(dc);
                            grew = true;
                        }
                        if !rows.contains(dr) {
                            rows.set(dr);
                            grew = true;
                        }
                    }
                }
            }
            if !grew {
                break;
            }
        }
        x.iter().all(|&c| cols.contains(c)) && cols.ones().count() == x.len() && rows == *y
    }
}

/// Advance a combination of `k` indices chosen from `0..n` to the next
/// lexicographic combination. Returns `false` when exhausted.
fn next_combination(indices: &mut [usize], n: usize) -> bool {
    let k = indices.len();
    if k == 0 {
        return false;
    }
    let mut i = k;
    while i > 0 {
        i -= 1;
        if indices[i] < n - k + i {
            indices[i] += 1;
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

    use bloqade_lanes_bytecode_core::arch::addr::{Direction, MoveType};
    use bloqade_lanes_bytecode_core::arch::types::ArchSpec;

    use super::*;
    use crate::observer::NoOpObserver;
    use crate::primitives::distance::DistanceTable;
    use crate::test_utils::{example_arch_json, lane, loc};

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

    /// B1 on the source side: a blocked site may not be a filler. With qubit
    /// 0 at site 0 and site 1 blocked, no emitted shot may contain site 1's
    /// lane — the 2×1 rectangle over sites 0 and 1 would drag the blocked
    /// atom along.
    #[test]
    fn generate_rejects_blocked_source_filler() {
        let index = make_index();
        let config = Config::new([(0, loc(0, 0))]).unwrap();
        let out = run_generator_blocked(&config, &index, &[loc(0, 1)]);
        assert!(!out.is_empty());
        let site1 = loc(0, 1).encode();
        for cand in &out {
            for lane in cand.move_set.decode() {
                let (src, _) = index.endpoints(&lane).unwrap();
                assert_ne!(
                    src.encode(),
                    site1,
                    "blocked site used as a source: {lane:?}"
                );
            }
        }
    }

    /// Only mover-tight rectangles are emitted: with one atom every shot is a
    /// single lane, and each child configuration appears once per group.
    #[test]
    fn generate_emits_one_tight_shot_per_child() {
        let index = make_index();
        let config = Config::new([(0, loc(0, 0))]).unwrap();
        let out = run_generator(&config, &index);
        assert!(!out.is_empty());
        assert!(
            out.iter().all(|c| c.move_set.len() == 1),
            "one atom, one lane"
        );
        let mut seen: HashSet<(GroupKey, Vec<(u32, u64)>)> = HashSet::new();
        for cand in &out {
            let key = GroupKey::of(&cand.move_set.decode()[0]);
            assert!(
                seen.insert((key, cand.new_config.as_entries().to_vec())),
                "child emitted twice in one group"
            );
        }
    }

    /// Two fresh generators over the same inputs emit identical sequences:
    /// groups are sorted, columns run in DFS order, rows in combination order.
    #[test]
    fn emission_order_is_deterministic() {
        let spec: ArchSpec =
            serde_json::from_str(oracle::physical_spec_json()).expect("physical spec parses");
        let index = LaneIndex::new(spec);
        let dist_table = DistanceTable::new(&[], &index);
        let blocked = HashSet::new();
        let ctx = ctx_for(&index, &dist_table, &blocked);
        // A handful of atoms on lane sources of the physical spec.
        let sources: Vec<LocationAddr> = {
            let mut v: Vec<LocationAddr> = index
                .bus_groups()
                .flat_map(|(mt, b, z, d)| index.lanes_for(mt, b, z, d).iter().copied())
                .filter_map(|l| index.endpoints(&l).map(|(s, _)| s))
                .collect();
            v.sort_by_key(|l| l.encode());
            v.dedup();
            v.into_iter().step_by(7).take(6).collect()
        };
        let config = Config::new(sources.iter().enumerate().map(|(q, &l)| (q as u32, l))).unwrap();
        let run = || {
            let generator = ExhaustiveGenerator::for_solve(&ctx, SeedPolicy::Any, None).unwrap();
            let mut out = Vec::new();
            generator.generate(
                &config,
                NodeId(0),
                &ctx,
                &mut SearchState::default(),
                &mut out,
            );
            out.into_iter()
                .map(|c| {
                    (
                        c.move_set.encoded_lanes().to_vec(),
                        c.new_config.as_entries().to_vec(),
                    )
                })
                .collect::<Vec<_>>()
        };
        let first = run();
        assert!(first.len() > 6, "several atoms yield several shots");
        assert_eq!(first, run());
    }

    /// Under `SeedPolicy::Unresolved` a rectangle needs an unresolved mover in
    /// every column and row it spans, unless a chain forces the rest. Qubit 1
    /// sits at its target, so the 2×1 shot moving both atoms is admitted only
    /// under `Any`; qubit 0's own 1×1 shot is admitted under both.
    #[test]
    fn unresolved_seed_excludes_rectangles_anchored_on_resolved_atoms() {
        let index = make_index();
        let config = Config::new([(0, loc(0, 0)), (1, loc(0, 1))]).unwrap();
        // Qubit 0 is unresolved (target site 5); qubit 1 is resolved (at site 1).
        let targets: Vec<(u32, u64)> = vec![(0, loc(0, 5).encode()), (1, loc(0, 1).encode())];
        let dist_table = DistanceTable::new(&[], &index);
        let blocked = HashSet::new();
        let ctx = make_ctx(&index, &dist_table, &targets, &blocked);

        let shots = |seed: SeedPolicy| -> Vec<Vec<u64>> {
            let generator = ExhaustiveGenerator::for_solve(&ctx, seed, None).unwrap();
            let mut out = Vec::new();
            generator.generate(
                &config,
                NodeId(0),
                &ctx,
                &mut SearchState::default(),
                &mut out,
            );
            out.into_iter()
                .map(|c| c.move_set.encoded_lanes().to_vec())
                .collect()
        };
        let any = shots(SeedPolicy::Any);
        let unresolved = shots(SeedPolicy::Unresolved);

        let both = vec![lane(0, 0, 0).encode_u64(), lane(0, 1, 0).encode_u64()];
        let alone = vec![lane(0, 0, 0).encode_u64()];
        assert!(
            any.contains(&both),
            "Any admits the 2×1 rectangle over both atoms"
        );
        assert!(
            !unresolved.contains(&both),
            "Unresolved refuses a rectangle anchored on qubit 1"
        );
        assert!(unresolved.contains(&alone));
        // Nesting: every Unresolved shot is an Any shot, and only qubit 0 moves.
        for shot in &unresolved {
            assert!(any.contains(shot), "{shot:#x?} not in the Any output");
        }
        assert!(unresolved.len() < any.len());
    }

    // ── Inter-zone moves survive per-zone grouping ──

    /// On the two-zone fixture the only lanes are one zone bus, whose
    /// forward lane moves `(zone 1, word 1)` to `(zone 0, word 0)`. Grouping
    /// on the full key keeps that shot in one group (a zone-bus lane carries
    /// its forward source zone), so cross-zone routing stays expressible under
    /// every seed policy and at unit capacity.
    #[test]
    fn exhaustive_emits_the_zone_bus_lane_across_zones() {
        let spec: ArchSpec =
            serde_json::from_str(crate::test_utils::two_zone_bus_arch_json()).unwrap();
        let index = LaneIndex::new(spec);
        let memory = LocationAddr {
            zone_id: 1,
            word_id: 1,
            site_id: 0,
        };
        let gate = LocationAddr {
            zone_id: 0,
            word_id: 0,
            site_id: 0,
        };
        let config = Config::new([(0, memory)]).unwrap();
        let targets = vec![(0u32, gate.encode())];
        let dist_table = DistanceTable::new(&[gate.encode()], &index);
        let blocked = HashSet::new();
        let ctx = make_ctx(&index, &dist_table, &targets, &blocked);
        for seed in [SeedPolicy::Unresolved, SeedPolicy::Any] {
            let generator =
                ExhaustiveGenerator::for_solve(&ctx, seed, Some(AodCapacity { x: 1, y: 1 }))
                    .unwrap();
            let mut out = Vec::new();
            generator.generate(
                &config,
                NodeId(0),
                &ctx,
                &mut SearchState::default(),
                &mut out,
            );
            let crosses = out.iter().any(|c| {
                c.new_config.location_of(0) == Some(gate)
                    && c.move_set.decode()[0].move_type == MoveType::ZoneBus
            });
            assert!(
                crosses,
                "{seed:?}: the zone-bus shot into zone 0 must be emitted"
            );
        }
    }

    /// The same inter-zone shot through the heuristic generator, which now
    /// groups per zone too: a zone bus is one group, so nothing about the
    /// per-zone change can split the move it needs.
    #[test]
    fn heuristic_emits_the_zone_bus_lane_across_zones() {
        use crate::generators::{DeadlockPolicy, HeuristicGenerator};
        let spec: ArchSpec =
            serde_json::from_str(crate::test_utils::two_zone_bus_arch_json()).unwrap();
        let index = LaneIndex::new(spec);
        let memory = LocationAddr {
            zone_id: 1,
            word_id: 1,
            site_id: 0,
        };
        let gate = LocationAddr {
            zone_id: 0,
            word_id: 0,
            site_id: 0,
        };
        let config = Config::new([(0, memory)]).unwrap();
        let targets = vec![(0u32, gate.encode())];
        let dist_table = DistanceTable::new(&[gate.encode()], &index);
        let blocked = HashSet::new();
        let ctx = make_ctx(&index, &dist_table, &targets, &blocked);
        let generator = HeuristicGenerator::configured(0, DeadlockPolicy::Skip, false, None);
        let mut out = Vec::new();
        generator.generate(
            &config,
            NodeId(0),
            &ctx,
            &mut SearchState::default(),
            &mut out,
        );
        assert!(
            out.iter()
                .any(|c| c.new_config.location_of(0) == Some(gate)),
            "the heuristic generator must still route across the zone bus"
        );
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
