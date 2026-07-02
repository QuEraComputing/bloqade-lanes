# AOD Grid Clique Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a selectable clique-based strategy to `BusGridContext` that turns a set of input mover lanes into valid AOD rectangles by modeling lanes as a conflict graph and extracting the clique that covers the most input movers.

**Architecture:** A new `AodGridStrategy` enum selects between the existing greedy+merge builder and a new clique builder. `BusGridContext` gains a `strategy` field (default `GreedyMerge`, opt-in via a `with_strategy` builder so no existing call site breaks). The clique builder reuses the existing `is_valid_rect` as both the node predicate (valid 1×1) and edge predicate (valid 2×2), so a maximal clique is exactly a maximal valid rectangle. Bron–Kerbosch with pivoting enumerates maximal cliques (≤64 nodes via u64 bitsets; greedy fallback above a cap); the best is chosen by `(movers_covered desc, area asc, deterministic)`. The result is decomposed by pruning covered movers and rerunning.

**Tech Stack:** Rust, `bloqade-lanes-search` crate, `ops/aod_grid.rs`, `search/options.rs`. No new dependencies (hand-rolled Bron–Kerbosch).

## Global Constraints

- Existing `build_aod_grids` return type stays `Vec<Vec<u64>>`; each element is one AOD-rectangle lane set.
- Default behavior is unchanged: `BusGridContext::new` yields `AodGridStrategy::GreedyMerge`.
- All output must be deterministic (sorted iteration, deterministic tie-breaks) — the search depends on reproducibility.
- Coordinates are `f64::to_bits()` (`u64`); positions are `(x_bits, y_bits)`.
- `movers` are identified by `entries.keys()` (encoded source locations).
- Run Rust tests with `cargo test -p bloqade-lanes-search`. Lint with `cargo clippy -p bloqade-lanes-search --all-targets -- -D warnings` and `cargo fmt`.

---

## Task 1: `AodGridStrategy` enum + `BusGridContext` dispatch

**Files:**
- Modify: `crates/bloqade-lanes-search/src/search/options.rs` (add enum, after the `Strategy` enum ~line 41)
- Modify: `crates/bloqade-lanes-search/src/ops/aod_grid.rs` (add field, builder, dispatch; rename current body)

**Interfaces:**
- Produces: `pub enum AodGridStrategy { GreedyMerge, Clique }` (`Default = GreedyMerge`); `BusGridContext::with_strategy(self, AodGridStrategy) -> BusGridContext`; `BusGridContext::build_aod_grids(&self, &HashMap<u64,u64>) -> Vec<Vec<u64>>` (unchanged signature, now dispatches); private `build_aod_grids_greedy` (renamed existing body).

- [ ] **Step 1: Add the enum**

In `crates/bloqade-lanes-search/src/search/options.rs`, after the `Strategy` enum (after line 41), add:

```rust
/// Strategy for turning selected mover lanes into valid AOD rectangles in
/// [`BusGridContext`](crate::ops::aod_grid::BusGridContext).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AodGridStrategy {
    /// Greedy sequential clustering + iterative rectangle merge (original).
    #[default]
    GreedyMerge,
    /// Conflict-graph max-clique: pick the rectangle covering the most input
    /// movers, completing it with filler/no-op lanes.
    Clique,
}
```

- [ ] **Step 2: Add the field, builder, and dispatch in `aod_grid.rs`**

Add the import at the top of `crates/bloqade-lanes-search/src/ops/aod_grid.rs` (after the existing `use` block, line 11):

```rust
use crate::search::options::AodGridStrategy;
```

Add a field to `BusGridContext` (inside the struct, after `occupied_locs`, line 34):

```rust
    /// Which grid-construction strategy `build_aod_grids` dispatches to.
    strategy: AodGridStrategy,
```

In `BusGridContext::new`, set the default in the returned struct literal (in the `Self { ... }` block, add after `occupied_locs: occupied.clone(),`):

```rust
            strategy: AodGridStrategy::default(),
```

Add the builder immediately after `new` (before `is_valid_rect`):

```rust
    /// Select the grid-construction strategy (default [`AodGridStrategy::GreedyMerge`]).
    pub(crate) fn with_strategy(mut self, strategy: AodGridStrategy) -> Self {
        self.strategy = strategy;
        self
    }
```

Rename the current `build_aod_grids` method body to `build_aod_grids_greedy` and add a dispatching `build_aod_grids`. Replace the existing method signature line
`pub(crate) fn build_aod_grids(&self, entries: &HashMap<u64, u64>) -> Vec<Vec<u64>> {`
with:

```rust
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
```

(The rest of the original method body — the `if entries.is_empty()` block through the `.collect()` — stays as the body of `build_aod_grids_greedy`.)

- [ ] **Step 3: Add a temporary stub for the clique method so it compiles**

Add (will be implemented in Task 2):

```rust
    /// Conflict-graph max-clique grid construction (implemented in Task 2).
    fn build_aod_grids_clique(&self, entries: &HashMap<u64, u64>) -> Vec<Vec<u64>> {
        // Placeholder until Task 2. Falls back to greedy so callers still work.
        self.build_aod_grids_greedy(entries)
    }
```

- [ ] **Step 4: Add a test that the strategy field is honored**

Add to the `tests` module in `aod_grid.rs`:

```rust
    #[test]
    fn strategy_defaults_to_greedy_merge() {
        let ctx = make_context(&[((0, 0), 10)], &[(10, 100)], &[]);
        assert_eq!(ctx.strategy, AodGridStrategy::GreedyMerge);
        let clique = ctx.with_strategy(AodGridStrategy::Clique);
        assert_eq!(clique.strategy, AodGridStrategy::Clique);
    }
```

- [ ] **Step 5: Build and run tests**

Run: `cargo test -p bloqade-lanes-search aod_grid`
Expected: PASS — all existing 8 grid tests still pass (greedy path unchanged), plus `strategy_defaults_to_greedy_merge`.

- [ ] **Step 6: Commit**

```bash
git add crates/bloqade-lanes-search/src/search/options.rs crates/bloqade-lanes-search/src/ops/aod_grid.rs
git commit -m "feat(search): add AodGridStrategy enum + BusGridContext dispatch"
```

---

## Task 2: Clique grid construction

**Files:**
- Modify: `crates/bloqade-lanes-search/src/ops/aod_grid.rs` (replace the Task-1 stub `build_aod_grids_clique`; add helpers + tests)

**Interfaces:**
- Consumes: `is_valid_rect(&self, &BTreeSet<u64>, &BTreeSet<u64>, &HashSet<u64>) -> bool`; `rect_to_lanes(&self, &BTreeSet<u64>, &BTreeSet<u64>) -> Vec<u64>`; `self.src_to_pos: HashMap<u64,(u64,u64)>`; `self.pos_to_src: HashMap<(u64,u64),u64>`.
- Produces: `build_aod_grids_clique(&self, &HashMap<u64,u64>) -> Vec<Vec<u64>>`; module-private helpers `clique_candidate_nodes`, `bron_kerbosch`, `greedy_max_clique`.

- [ ] **Step 1: Write the failing test (clique finds one complete rectangle)**

Add to the `tests` module:

```rust
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
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `cargo test -p bloqade-lanes-search clique_builds_single_complete_rectangle`
Expected: FAIL (stub falls back to greedy which would also pass here — so ALSO add the discriminating test in Step 3 before implementing; run both after implementing).

- [ ] **Step 3: Write the discriminating test (clique beats greedy split)**

Add to the `tests` module:

```rust
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
```

- [ ] **Step 4: Implement `build_aod_grids_clique` and helpers**

Replace the Task-1 stub `build_aod_grids_clique` with:

```rust
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

            // Evaluate a clique (bitset over node indices) by the objective.
            // Returns None if it covers zero input movers.
            let is_mover = |idx: usize| -> bool {
                self.pos_to_src
                    .get(&nodes[idx])
                    .is_some_and(|src| movers.contains(src))
            };
            let eval = |clique: u64| -> Option<(usize, std::cmp::Reverse<usize>, u64)> {
                if clique == 0 {
                    return None;
                }
                let mut movers_covered = 0usize;
                let mut bits = clique;
                while bits != 0 {
                    let idx = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    if is_mover(idx) {
                        movers_covered += 1;
                    }
                }
                if movers_covered == 0 {
                    return None;
                }
                let area = clique.count_ones() as usize; // clique == full rectangle
                // Deterministic tie-break: the clique's raw bitset (lower node
                // indices, which follow sorted node order, win).
                Some((movers_covered, std::cmp::Reverse(area), !clique))
            };

            // Find the best maximal clique.
            let mut best_key: Option<(usize, std::cmp::Reverse<usize>, u64)> = None;
            let mut best_clique: u64 = 0;
            let mut consider = |clique: u64| {
                if let Some(key) = eval(clique)
                    && best_key.as_ref().is_none_or(|b| key > *b)
                {
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

            // Tone sets of the winning clique → complete rectangle.
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
    fn bron_kerbosch(
        adj: &[u64],
        r: u64,
        p: u64,
        x: u64,
        visit: &mut impl FnMut(u64),
    ) {
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
                // Pick the candidate with the most connections to `cand`
                // (deterministic: lowest index breaks ties).
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
```

- [ ] **Step 5: Run the clique tests**

Run: `cargo test -p bloqade-lanes-search clique`
Expected: PASS — `clique_builds_single_complete_rectangle`, `clique_recovers_rectangle_greedy_would_split`, `clique_emits_only_mover_covering_rectangles`, `clique_reversibility_rejects_filled_filler_destination`.

- [ ] **Step 6: Add determinism + fallback tests**

```rust
    #[test]
    fn clique_is_deterministic() {
        let ctx = make_context(
            &[((0, 0), 10), ((1, 0), 11), ((0, 1), 12), ((1, 1), 13)],
            &[(10, 100), (11, 101), (12, 102), (13, 103)],
            &[],
        )
        .with_strategy(AodGridStrategy::Clique);
        let entries: HashMap<u64, u64> =
            [(10, 100), (11, 101), (12, 102), (13, 103)].into_iter().collect();
        let a = ctx.build_aod_grids(&entries);
        let b = ctx.build_aod_grids(&entries);
        assert_eq!(a, b);
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
```

- [ ] **Step 7: Run all aod_grid tests + clippy + fmt**

Run: `cargo test -p bloqade-lanes-search aod_grid`
Expected: PASS (all greedy + clique tests).
Run: `cargo clippy -p bloqade-lanes-search --all-targets -- -D warnings`
Expected: clean.
Run: `cargo fmt -p bloqade-lanes-search`
Expected: no diff (or apply).

- [ ] **Step 8: Commit**

```bash
git add crates/bloqade-lanes-search/src/ops/aod_grid.rs
git commit -m "feat(search): clique-based AOD grid construction strategy"
```

---

## Task 3: Wire the strategy through config to call sites

**Files:**
- Modify: `crates/bloqade-lanes-search/src/search/options.rs` (add `aod_grid_strategy` to `EntropyOptions`)
- Modify: `crates/bloqade-lanes-search/src/drivers/entropy.rs` (3 sites: lines ~417, ~962, ~2451)
- Modify: `crates/bloqade-lanes-search/src/generators/greedy.rs` (line ~93)
- Modify: `crates/bloqade-lanes-search/src/generators/heuristic.rs` (line ~554)
- Modify: `crates/bloqade-lanes-search/src/dsl/pipeline.rs` (line ~112)

**Interfaces:**
- Consumes: `AodGridStrategy` (Task 1), `BusGridContext::with_strategy` (Task 1).
- Produces: `EntropyOptions.aod_grid_strategy: AodGridStrategy`.

- [ ] **Step 1: Add the config field**

In `crates/bloqade-lanes-search/src/search/options.rs`, add to `EntropyOptions` (in the struct, and set its default in the `Default`/constructor for `EntropyOptions`):

```rust
    /// Strategy for AOD grid construction from selected movers.
    pub aod_grid_strategy: AodGridStrategy,
```

Add `aod_grid_strategy: AodGridStrategy::default(),` to the `EntropyOptions` default initializer. (`AodGridStrategy` is in the same module.)

- [ ] **Step 2: Thread into the 3 entropy call sites**

At each `let grid_ctx = BusGridContext::new(...);` in `drivers/entropy.rs` (lines ~417, ~962, ~2451), append `.with_strategy(<opts>.aod_grid_strategy)`, where `<opts>` is the `EntropyOptions` in scope at that site (the entropy driver already carries its options — locate the `EntropyOptions` binding in each function and read `.aod_grid_strategy`). Example for line ~417:

```rust
let grid_ctx = BusGridContext::new(ctx.index, mt, bus_id, None, dir, occupied)
    .with_strategy(opts.aod_grid_strategy);
```

Verify the exact `EntropyOptions` variable name in each of the three functions before editing; if a function does not already have the options in scope, thread it in from its caller (the entropy driver constructs from `EntropyOptions`).

- [ ] **Step 3: Thread into greedy / heuristic / dsl generators**

These generators are not entropy-specific. For each, plumb the strategy from the generator's own configuration:

- `generators/greedy.rs` (~line 93): append `.with_strategy(self.aod_grid_strategy)` and add an `aod_grid_strategy: AodGridStrategy` field to the greedy generator struct (default via its constructor).
- `generators/heuristic.rs` (~line 554): same pattern — add the field to `HeuristicGenerator` and append `.with_strategy(self.aod_grid_strategy)`.
- `dsl/pipeline.rs` (~line 112): the DSL pipeline builds the context directly; read the strategy from the pipeline's config struct (add an `aod_grid_strategy` field there) and append `.with_strategy(...)`.

For each struct that gains the field, default it to `AodGridStrategy::default()` so existing constructors and tests are unaffected.

- [ ] **Step 4: Build + full crate tests + lint**

Run: `cargo test -p bloqade-lanes-search`
Expected: PASS (no behavior change — everything defaults to `GreedyMerge`).
Run: `cargo clippy -p bloqade-lanes-search --all-targets -- -D warnings` and `cargo fmt -p bloqade-lanes-search`
Expected: clean.

- [ ] **Step 5: Add an end-to-end wiring test**

Add a test (in `drivers/entropy.rs` tests or a suitable integration test) that constructs `EntropyOptions { aod_grid_strategy: AodGridStrategy::Clique, ..Default::default() }`, runs the entropy path on a small config with a known multi-mover bus, and asserts a valid moveset is produced (grids non-empty, movers covered). Mirror an existing entropy test's setup for the fixture.

- [ ] **Step 6: Commit**

```bash
git add crates/bloqade-lanes-search/src/search/options.rs \
        crates/bloqade-lanes-search/src/drivers/entropy.rs \
        crates/bloqade-lanes-search/src/generators/greedy.rs \
        crates/bloqade-lanes-search/src/generators/heuristic.rs \
        crates/bloqade-lanes-search/src/dsl/pipeline.rs
git commit -m "feat(search): thread AodGridStrategy through config to grid call sites"
```

---

## Notes for the Implementer

- **Why reuse `is_valid_rect` for edges:** a node is a valid 1×1 rectangle; an edge A–B is a valid 2×2 rectangle `{xₐ,x_b}×{yₐ,y_b}`. Because every cell of a clique's full `X×Y` product is a corner of some node pair, all-edges-valid ⟹ the full rectangle is valid — so a maximal clique is exactly a maximal valid rectangle. This also inherits the mover / rect-mover / reversibility handling already in `is_valid_rect` with no duplication.
- **Objective:** `(movers_covered desc, area asc, deterministic)` with `movers_covered ≥ 1`. `area == clique.count_ones()` because a maximal clique equals its full rectangle.
- **`is_none_or` / `is_some_and`:** stable since Rust 1.82 / 1.70; this crate is edition 2024, so they are available.
- **Bitset limit:** ≤64 nodes (u64). `clique_candidate_nodes` caps at 64; the exact/greedy split is at `MAX_EXACT_CLIQUE_NODES = 32`. Both are deterministic. If profiling shows large per-bus candidate sets, raising the cap or adding a maximal-clique-count guard is a follow-up.
- **Determinism:** `BTreeSet` iteration + sorted node vector + `!clique` tie-break keep output stable regardless of `HashMap` iteration order.
