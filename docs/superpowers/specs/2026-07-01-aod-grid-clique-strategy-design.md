# AOD Grid Clique Strategy Design

**Goal:** Add a second, selectable strategy for turning an arbitrary set of
input mover lanes into valid AOD rectangles in
`crates/bloqade-lanes-search/src/ops/aod_grid.rs`. The new strategy models the
problem as a **conflict graph over lanes** and extracts **cliques** (each a valid
AOD rectangle), maximizing the number of input movers covered per shot while
completing each rectangle with filler/no-op lanes for full AOD utilization. The
existing greedy-init + merge strategy stays as the default.

**Non-goals:** Changing the rectangle/AOD validity model itself; changing the
`Vec<Vec<u64>>` return contract; multi-bus solving (still per bus group).

**Tech stack:** Rust (`bloqade-lanes-search` crate), `BusGridContext` in
`ops/aod_grid.rs`, `search/options.rs` (existing `Strategy` enum + option
bundles), hand-rolled Bron–Kerbosch (no `petgraph`/graph dep in this crate).

---

## Background

`BusGridContext` is built per bus group `(MoveType, bus_id, Direction)` from ALL
lanes on that bus, plus the set of `occupied` locations (atoms / blocked). Its
`build_aod_grids(entries)` takes `entries: HashMap<encoded_src, encoded_lane>`
(the selected movers) and returns `Vec<Vec<u64>>` — a list of valid AOD
rectangles (lane sets). Crossed AODs illuminate the Cartesian product of the
selected X-tones × Y-tones, so a valid shot is a **rectangle**: every position
in `X × Y` must be present and safe. "Rectangle" means the product of two
coordinate **sets** — the coordinates need not be contiguous.

The current strategy builds rectangles greedily then iteratively merges
compatible ones. Consumers (`drivers/entropy.rs` ×3, `generators/greedy.rs`,
`generators/heuristic.rs`, `dsl/pipeline.rs`) iterate the returned grids, score
each, and select the best.

### Validity & reversibility

A rectangle cell (a lane `src → dst`) is valid iff moving it is **reversible**:

- A **mover** cell: `src` holds an atom we intend to move; valid iff `dst` is
  unoccupied *or* occupied by another mover inside this rectangle.
- A **filler** (no-op) cell: `src` is empty; valid iff `dst` is **also** empty.
  An empty-source / filled-destination cell is invalid even though the forward
  move collides with nothing — because the *reverse* move would drag the
  destination atom, which we never intended to move.

This is the logic in today's `is_valid_rect`; it is preserved.

### Key insight (why cliques model rectangles)

Model each candidate lane as a graph node and connect two lanes
`A = (xₐ,yₐ)` and `B = (x_b,y_b)` with a **non-conflict edge** iff both induced
corners `(xₐ, y_b)` and `(x_b, yₐ)` are present as valid cells. Then **any clique
induces a fully valid rectangle**: every cell `(xᵢ, yⱼ)` of the clique's
`X × Y` product is a corner of the pair `(i, j)`, and diagonal cells `(xᵢ, yᵢ)`
are the nodes themselves — so all-pairs-valid ⟹ all-cells-valid. This makes the
pairwise conflict graph a sound encoding of the (global) rectangle constraint.

---

## Section 1 — Strategy enum + wiring

- Add `AodGridStrategy { GreedyMerge, Clique }` to `search/options.rs`, next to
  the existing `Strategy` enum. `#[derive(..., Default)]` with
  `#[default] GreedyMerge` so current behavior is unchanged.
- Store the selected strategy as a field on `BusGridContext`, set in
  `BusGridContext::new(...)` (all six construction sites already call it). The
  public `build_aod_grids(&entries)` signature is **unchanged**; internally it
  dispatches to the greedy path or the new clique path based on the field.
- Thread the enum from the option bundles (e.g. `EntropyOptions`, and the
  greedy/heuristic/dsl generator configs) down to each `BusGridContext::new`
  call. This is the configurable wiring: flip one config value to select the
  strategy everywhere it is plumbed.

**Touch points:** `search/options.rs` (new enum + option-bundle field),
`ops/aod_grid.rs` (field + dispatch + new method), and the 6 `BusGridContext::new`
call sites (pass the strategy through from their config). Call sites that don't
yet plumb a config keep the `Default` (`GreedyMerge`).

---

## Section 2 — Clique algorithm (`build_aod_grids_clique`)

Operates on the same `BusGridContext` data. Given `entries` (input movers):

1. **Candidate universe = the movers' induced Cartesian product.**
   `X = { distinct x-tone of each input mover }`,
   `Y = { distinct y-tone of each input mover }`. Candidate positions are the
   full product `X × Y`. This bounds the problem to the movers' tone span (no
   growth beyond it).
2. **Nodes.** For each position in `X × Y`, add a node for its lane iff the cell
   is **reversibility-valid** (mover cell with free/rect-mover destination, or
   filler cell with empty source *and* empty destination). Positions with a
   spectator source, a blocked destination, or no lane produce no node.
3. **Edges (non-conflict).** Connect nodes `A`, `B` iff their induced corners
   `(xₐ,y_b)` and `(x_b,yₐ)` are both present as valid nodes.
4. **Select the winning clique.** Objective, in order:
   1. maximize the number of **input movers** covered by the clique's rectangle;
   2. tie-break: **smaller** rectangle area `|X_clique| × |Y_clique|` (don't
      inflate with unnecessary no-op tones);
   3. tie-break: deterministic (sorted lane/src encoding).
   Every emitted shot must cover **≥ 1 input mover** — a clique covering zero
   input movers can never win, so mover-less rectangles never dominate.
   Solver: exact **Bron–Kerbosch with pivoting** (hand-rolled), evaluating
   maximal cliques by the objective; above a node-count cap
   (const, e.g. 32) fall back to a greedy degeneracy-order heuristic to bound
   worst-case runtime.
5. **Emit the complete rectangle.** `rect_to_lanes` over the winning clique's
   `X × Y` → all lanes at those positions, **including filler/no-op lanes** (full
   AOD utilization: the shot is a complete rectangle, not a sparse subset).
6. **Decompose.** Remove the covered input movers, recompute the candidate
   product from the **remaining** movers, and rerun steps 1–5 until no input
   movers remain. Returns `Vec<Vec<u64>>` — same contract as the greedy path;
   consumers still score each grid and pick the best.

---

## Section 3 — Determinism, fallback, refactor, edge cases

- **Determinism:** sorted node ordering, deterministic pivot selection, and the
  encoding-order final tie-break keep output reproducible (the search relies on
  this).
- **Fallback:** node-count cap → greedy degeneracy-order clique heuristic;
  `log`/comment notes the cap so silent truncation is visible.
- **Refactor:** extract the per-cell validity check out of `is_valid_rect` into
  a shared helper (e.g. `is_valid_cell(src, dst, movers, rect_movers)`) used by
  both the greedy path (unchanged behavior) and the clique edge predicate.
- **Edge cases:** empty `entries` → empty `Vec` (as today); a mover that
  conflicts with every other candidate → its own 1×1 rectangle; a mover whose
  own cell is invalid (destination blocked) → excluded (cannot be placed this
  shot), consistent with the greedy path.

---

## Section 4 — Testing

Rust unit tests in `ops/aod_grid.rs`, mirroring the existing helpers
(`make_context*`):

- **Behavioral parity where expected:** all-movers 2×2, sparse color-code
  rectangle, spectator/reversibility rejection, empty-filler inclusion — assert
  the clique strategy produces valid complete rectangles.
- **Clique beats greedy:** a configuration where the clique strategy finds a
  larger single valid rectangle than greedy-init+merge would.
- **Mover-less domination guard:** a large valid empty region adjacent to a few
  movers — assert emitted shots always contain ≥1 mover and are not the empty
  rectangle.
- **Tie-break:** equal-mover cliques of different areas → smaller area chosen.
- **Fallback path:** synthetic node set above the cap → still returns valid
  rectangles (heuristic).
- **Determinism:** identical output across repeated runs / shuffled input order.
- **Existing 8 greedy tests stay green** under `GreedyMerge` (unchanged).

---

## File Layout

| File | Status | Responsibility |
|------|--------|----------------|
| `crates/bloqade-lanes-search/src/search/options.rs` | EDIT | Add `AodGridStrategy` enum; add field to the option bundle(s) that reach grid construction |
| `crates/bloqade-lanes-search/src/ops/aod_grid.rs` | EDIT | Add strategy field to `BusGridContext`; dispatch in `build_aod_grids`; add `build_aod_grids_clique`; extract `is_valid_cell`; hand-rolled Bron–Kerbosch; new tests |
| `crates/bloqade-lanes-search/src/drivers/entropy.rs` | EDIT | Pass strategy into the 3 `BusGridContext::new` calls from config |
| `crates/bloqade-lanes-search/src/generators/greedy.rs` | EDIT | Pass strategy into `BusGridContext::new` |
| `crates/bloqade-lanes-search/src/generators/heuristic.rs` | EDIT | Pass strategy into `BusGridContext::new` |
| `crates/bloqade-lanes-search/src/dsl/pipeline.rs` | EDIT | Pass strategy into `BusGridContext::new` |

---

## Objective, stated precisely

For a candidate clique `C` with induced tone sets `X_C`, `Y_C`:

```
key(C) = ( movers_covered(C),        // maximize
          -(|X_C| * |Y_C|),          // then maximize (i.e. prefer smaller area)
           deterministic_tiebreak )   // then maximize (stable)
```

pick `argmax key(C)` subject to `movers_covered(C) >= 1`.

**"Utilization" vs. "smaller area" — not a contradiction.** Utilization means the
emitted shot is a *complete* rectangle: every valid cell of the winning clique's
`X × Y` is driven, including no-op filler lanes (step 5). The smaller-area
tie-break only chooses *between cliques that cover the same movers* — it avoids
padding the rectangle with extra tones the movers don't already span. We complete
the rectangle we pick; we don't grow it just to add no-ops.

---

## Follow-Ups (Out of Scope)

- Wiring the enum through the Python/PyO3 surface or the DSL config, if callers
  want to select it from Python.
- Multi-bus / cross-group AOD scheduling.
- Weighting the clique objective by mover *score* (fidelity/priority) rather than
  raw count.
