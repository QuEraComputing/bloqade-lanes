# Crate Review: bloqade-lanes-search (2026-06-16)

> Triggered after the `refactor/706-remove-legacy-movesolver` branch landed two
> refactor commits removing the legacy `MoveSolver`/`PyMoveSolver` facade and the
> `temp_regression` test infrastructure. Prior reviews: `2026-05-31-bloqade-lanes-search-review.md`
> and `2026-05-31-bloqade-lanes-search-post-pr600-review.md`.

## 1. Context

`bloqade-lanes-search` is the Rust search/placement engine of the compilation
pipeline — it turns a CZ-stage placement problem (initial config, controls/targets,
blocked sites) into physical atom-move layers (`SolveResult`/`MoveSet`). It sits
mid-stack: it depends on `bloqade-lanes-bytecode-core` (arch spec) and
`bloqade-lanes-dsl-core` (the `MoveSet` primitive + Starlark substrate), and is
consumed by `bloqade-lanes-bytecode-python` (PyO3 bindings) and
`bloqade-lanes-bytecode-cli` (policy eval/trace). Change activity is moderate —
**8 commits in 30 days** — but concentrated: the active branch refactor touched 14
source files across `search/`, `placement/`, `generators/`, and `ops/`.

The headline finding: **the `MoveSolver` removal is clean** — no `MoveSolver`/
`PyMoveSolver`/`temp_regression` symbols survive anywhere in `src/`, and no
`pub(crate)` type lingers solely to support the removed facade. What the refactor
*exposed* (rather than introduced) is a set of pre-existing transitional seams now
left without their original justification: a decorative `lib.rs` re-export facade,
a re-export shim at `search/solve.rs`, a vestigial `drivers/astar` legacy A* path,
and an upward layering dependency from `search::engine` into `placement::nohome`.

## 2. External API Surface

### Public Type Inventory

| Name | Kind | Purpose |
|------|------|---------|
| `SearchEngine` | struct | Arch-bound data layer: owns `LaneIndex` + lazy entangling/no-home caches; built once per arch, shared via `Arc`. |
| `MoveSearch` | struct | Pure value bundle pairing strategy + tuning knobs; clone-friendly, no per-arch state. |
| `TargetSolver` | struct | Single fixed-target router: `Arc<SearchEngine>` + `MoveSearch`, returns `SolveResult`. |
| `SolveResult` / `SolveStatus` | struct/enum | Always-returned solve outcome (`status`, `move_layers`, `goal_config`, `cost`, `nodes_expanded`, `deadlocks`, `entropy_trace`); tristate `Solved`/`Unsolvable`/`BudgetExceeded` with `as_label()`. |
| `SolveOptions` / `EntropyOptions` / `EntanglingOptions` | structs | Core / entropy-specific / loose-goal Hungarian tuning knobs. |
| `Strategy` / `InnerStrategy` | enums | Algorithm selector (`AStar`, `HeuristicDfs`, `Bfs`, `GreedyBestFirst`, `Ids`, `Cascade`, `Entropy`) + cascade Phase-1 selector. |
| `MultiSolveResult` / `CandidateAttempt` | structs | Multi-candidate solve result + per-candidate debug record. |
| `CzPlacement` | trait | Uniform CZ-stage placement interface implemented by all four peers. |
| `SingleHeuristicCzPlacement`, `LooseGoalCzPlacement`, `NoHomeCzPlacement`, `RecedingHorizonCzPlacement` | structs | The four placement strategy peers. |
| `NoHomeOptions`, `RecedingHorizonOptions`, `default_weight_grid` | options/fn | Strategy-specific knobs. |
| `TargetGenerator`, `TargetContext`, `DefaultTargetGenerator`, `CandidateError` | trait/struct/enum | Candidate target-config plugin for single-heuristic placement. |
| `LaneIndex` | struct | Precomputed lane lookups from `ArchSpec`. **Used by every consumer.** |
| `DistanceTable`, `Config`, `ConfigError`, `SearchContext`, `SearchState`, `MoveCandidate` | structs | Search primitives, exposed so the Python `PyEntropyScorer` can replicate the entropy formula outside the solver loop. |
| `MoveSet`, `NodeId`, `SearchGraph`, `SearchResult` | re-export/structs | Search-graph types. |
| `DeadlockPolicy` | enum | `Skip` / `MoveBlockers` / `AllMoves`. **Used by `search_python`.** |
| Trait set: `MoveGenerator`, `CandidateScorer`, `CostFn`, `Goal`, `Heuristic` + impls (`goals`, `scorers`, `generators`, `heuristics`, `cost`) | various | Composable search extension points. |
| `dsl::move_policy_dsl` / `dsl::target_generator_dsl` / `dsl::fixture` | mods | Starlark sidecar surface (kernels, observers, fixture loader). **Fully exercised by python + cli.** |
| `drivers::entropy` (`EntropyParams`, `EntropyTrace`, `compute_moveset_metrics`, …) | mod | Entropy driver types. **Used by `search_python`.** |

### Responsibility Portraits (from usage evidence)

- **`SearchEngine`** — consumers build one per architecture, wrap it in `Arc`, and
  hand it to `TargetSolver` and every `*CzPlacement` constructor; they never call
  its methods beyond construction. Its whole contract is "be the shareable arch-bound
  cache." Lazy caches are `pub(crate)`, so callers correctly treat it as opaque.
- **`TargetSolver` / `*CzPlacement` peers** — the primary entry points the Python
  bindings wrap one-to-one. Callers pass `(initial, controls/target, blocked,
  max_expansions)` and expect a `SolveResult` (or `MultiSolveResult`). The
  `engine()`/`search()` accessors exist mainly so `PySingleHeuristicCzPlacement::new`
  can re-derive a fresh `TargetSolver` — a friction point.
- **`SolveResult` / `SolveStatus`** — the single result contract every solver
  returns; the Python wrapper reads every public field, and `as_label()` is the
  load-bearing string bridge to Python.
- **`LaneIndex`** — the most widely consumed type; built directly in both Python DSL
  bindings and both CLI files, then `Arc`-wrapped into the DSL kernels. The kernels
  treat it as *the* canonical architecture handle.
- **`Config` / `DistanceTable` / `SearchContext` / `MoveCandidate`** — exposed
  specifically so `search_python::PyEntropyScorer` can reconstruct a `Config` from a
  Python dict, build a `DistanceTable`, assemble a `SearchContext` literal, and call
  `compute_moveset_metrics` to replicate the entropy formula.
- **DSL kernels** — a genuinely large, fully-exercised public surface; `PolicyStatus`
  is exhaustively matched to map terminal states to strings/exit codes, and the
  observer traits are implemented by the CLI for tracing.

### API Friction Points

- `search_python.rs:1559` — `PySingleHeuristicCzPlacement::new` rebuilds a
  `TargetSolver` via `TargetSolver::new(solver.inner.engine().clone(),
  solver.inner.search().clone())`. A `TargetSolver: Clone` impl (or a
  `SingleHeuristicCzPlacement::from_solver(&TargetSolver)`) would remove the manual
  reconstruction; the accessors exist only for this.
- `search_python.rs:587` / `policy_runner_python.rs:285` / `eval.rs:71` —
  `EntropyParams` and `PolicyOptions` are assembled field-by-field across consumers,
  with default fallbacks (`max_expansions` of `100_000` vs `5_000`) living in the
  *consumers*, not the crate.
- `search_python.rs:209` / `policy_runner_python.rs:177` — the
  `MoveSet::decode()` → lane-tuple `(u8,u8,u32,u32,u32,u32)` conversion is
  re-implemented by hand in `PySolveResult`, `PyMultiSolveResult`, and
  `PyPolicySolveResult`. A `MoveSet::to_wire_tuples()` helper would dedupe it.
- `policy_runner_python.rs:241` — `PyPolicyRunner::from_arch_spec` serializes the
  `ArchSpec` to JSON and re-parses it, even though `LaneIndex::from_arch_spec(&ArchSpec)`
  exists — reintroducing the JSON round-trip `from_arch_spec` was added to eliminate.

### Dead Public Surface

- **MoveSolver removal is clean** — no `MoveSolver`/`PyMoveSolver` symbols remain;
  no public item references the removed concept.
- **The `lib.rs` top-level `pub use` re-export block (lines 40–70) is now almost
  entirely orphaned.** Of ~50 re-exported names, only `LaneIndex` and `DeadlockPolicy`
  are consumed via the short path; every other consumer imports through deep `pub mod`
  paths (`search::`, `placement::`, `dsl::`, `primitives::`). The facade is decorative
  while the internals are fully `pub mod` and consumers bypass it.
- Smaller dead items inside live modules: `HopDistanceHeuristic::estimate`
  (`primitives/distance.rs:380`, `#[deprecated]`, no caller); `LaneIndex::triplets`
  (`primitives/lane_index.rs:249`, `#[deprecated]` alias of `bus_groups`, no caller);
  `MisplacedHeuristic` (`primitives/distance.rs:285`, fully public, used only in its
  own module tests).

## 3. Internal Architecture

### Module Map (abridged)

`lib` (root + re-exports) · `traits` (the 5 search traits) · `cost` · `goals` ·
`heuristics` · `observer` · `primitives/{config, graph, context, distance,
lane_index, ordering, path}` · `ops/{aod_grid, entangling}` · `drivers/{frontier,
astar, entropy}` · `scorers/{distance, entropy}` · `generators/{heuristic,
exhaustive, greedy, entropy, loose_target}` · `search/{engine, move_search, options,
result, restarts, solve, target_solver}` · `placement/{cz_placement,
target_generator, single_heuristic, loose_goal, nohome, receding_horizon}` ·
`dsl/{pipeline, fixture, move_policy_dsl/*, target_generator_dsl/*}` · `test_utils` ·
`bin/policies-primer`.

### Internal Interaction Graph (high-level)

- **Foundation:** everything depends on `primitives::config` (`Config`) and
  `primitives::graph` (`MoveSet`, `NodeId`, `SearchGraph`); `traits` defines the
  shared vocabulary over `primitives::{config, context, graph}`.
- **Drivers hub:** `drivers::frontier` is the production search loop (`run_search` v2);
  `drivers::astar` and `drivers::frontier` form a **circular, test-only** pair via
  `run_search_legacy`/`Expander`. `drivers::entropy` is the largest, most-coupled
  module (imports 8 primitives modules + `ops::aod_grid` + `observer` + `traits`).
- **Dispatch hub:** `search::restarts::run_with_components` is the single strategy-
  dispatch + restart point that all four placement peers and `TargetSolver` route
  through — parameterized by a `MkGen: Fn(u64, DeadlockPolicy) -> Gen` closure.
- **Notable upward edge:** `search::engine` → `placement::nohome::home_sites` — the
  data layer reaches *up* into a placement strategy for an arch-derived precompute.

### pub(crate) Type Inventory (invariant-bearing)

| Name | Defined In | Invariant callers trust |
|------|------------|-------------------------|
| `Config` | `primitives::config` | Immutable, order-independent, stable FNV hash — transposition-table dedup depends on it. |
| `MoveSet` | `primitives::graph` (re-export) | Canonicalized (sorted + deduped) lanes — equality is order-independent. |
| `SearchGraph` | `primitives::graph` | Lazy-deletion lower-g-score invariant; `NodeId(0)` is always root; dense monotonic IDs for the closed-set. |
| `SearchContext` | `primitives::context` | `cz_pairs.is_some()` gates loose-goal-specific scoring branches in `HeuristicGenerator`. |
| `DistanceTable` | `primitives::distance` | Admissible min-hop; `None` → `INFINITY`/`u32::MAX` to prune. |
| `BusGridContext` | `ops::aod_grid` | Returns only AOD-legal X×Y rectangles; only a `debug_assert` validates. |
| `EntropyParams` | `drivers::entropy` | Field defaults shared by scorer/generator/restarts; must match Python parity. |
| `EntanglingCache` / `NoHomeCache` | `search::engine` | Lazy `OnceLock` arch-derived precomputes. |

### Internal Coupling Hotspots (modules importing from 3+ siblings)

- `search::restarts` — `cost`, `drivers::{astar, entropy, frontier}`,
  `generators::heuristic`, `observer`, `primitives::{config, context}`, `scorers`,
  `search::{options, result}`, `traits` (central dispatch hub).
- `placement::receding_horizon` and `placement::loose_goal` — the two biggest
  hotspots, each pulling from `generators`, `goals`, `ops::entangling`, `scorers`,
  multiple `primitives::*`, and the full `search::*` set.
- `drivers::entropy` — `drivers::astar`, `observer`, `ops::aod_grid`, 8×
  `primitives::*`, `traits`.
- `search::target_solver` — `generators` ×2, `goals`, `primitives::*`,
  `search::{engine, move_search, options, restarts, result}`.

## 4. Critical Evaluation

### Contract vs Implementation Divergence

| Public Type | Classification | Explanation |
|-------------|----------------|-------------|
| `SearchEngine` | MATCHES | `OnceLock` lazy caches, `Arc`-shareable, three constructors deliver "build once per arch." |
| `MoveSearch` / `TargetSolver` | MATCHES (with friction) | Value-bundle vs Arc-bound-router split holds; friction is the `search_python.rs:1559` rebuild, not divergence. |
| `SolveResult` / `SolveStatus` | MATCHES | `extract`/`pick_best` in `restarts.rs` are the sole translation point; `BudgetExceeded` vs `Unsolvable` discriminated on `nodes_expanded >= max`. |
| `SolveOptions` / `EntropyOptions` / `EntanglingOptions` | MATCHES | Documented-default knob bundles; `upgraded_for_entangling` / `clipped_future_layers` coherent. |
| `Strategy` / `InnerStrategy` | MATCHES | Dispatched exhaustively in `restarts.rs` with a guarded `unreachable!` for pre-filtered variants. |
| `CzPlacement` + 4 peers | MATCHES | Uniform trait; each peer routes through `run_with_components`. |
| `HeuristicGenerator` | GAP (communication) | `configured()`'s "three call sites" claim is verified true, but `cz_pairs.is_some()` gating creates an implicit two-mode contract (legacy fixed-target vs loose-goal/no-home/RH) documented only in scattered inline comments, not in the type's doc. |
| `DeadlockPolicy` | MATCHES | Clean enum, consumed by `search_python`. |
| `lib.rs` `pub use` facade | GAP (design drift) | Decorative re-export block; lines 64–69 still re-export `SolveOptions`/`Strategy`/`InnerStrategy`/`CandidateAttempt`/`MultiSolveResult` via `search::solve::*`, but `solve.rs` is *itself* a re-export shim of `options.rs`/`result.rs`. The facade points at a shim that points at the real module. |

### Rust Health Findings
*(hotspot files: `generators/heuristic.rs`, `search/options.rs`, `search/restarts.rs`)*

- `generators/heuristic.rs:571,576` — **note**: two `unreachable!("invalid
  MoveType/Direction discriminant")` reconstructing typed enums from `u8`. The
  triplet key was *built* from `as u8` casts of the same enums (396/424/504), so the
  round-trip is sound — but the `unreachable!` is load-bearing on a hand-maintained
  encode/decode invariant a newtype would make unrepresentable.
- `generators/heuristic.rs:623` — **warn**: `candidates.iter().any(|(_, ms, _)| *ms
  == move_set)` is an O(n²) dedup inside the per-group grid loop. Fine for the
  advertised "5–15 candidates", but quadratic in moveset count with no guard on dense
  bus groups. No correctness issue; a `HashSet<MoveSet>` would make it O(n).
- `generators/heuristic.rs:97,653,667` — **note**: `Cell<u32>` deadlock counter
  with interior mutability behind `&self`. Sound for the documented "each restart gets
  its own generator" usage, but a subtle contract resting entirely on a comment.
- `restarts.rs:64` — **note**: `.expect("non-empty results")` in `pick_best`.
  Invariant holds (callers always pass non-empty vecs) but is an unchecked precondition
  on a `pub(crate)` fn.
- `restarts.rs:347` — **note**: `unreachable!("IDS/DFS/Cascade/Entropy handled
  before run_strategy_v2")`. Correct — `run_once` filters those variants first —
  but load-bearing on caller match-arm ordering.
- No `unsafe`, no `todo!`/`unimplemented!`, no production `panic!` in the three
  hotspot files. No problematic lifetime complexity.

### Architectural Health Findings

- **Decorative `lib.rs` facade (lib.rs:40–70)** — every module is `pub mod` and
  consumers import via deep paths; the re-export block is hand-maintained dead weight
  that has *already drifted* (still references `search::solve::{SolveOptions, Strategy,
  InnerStrategy}` after those moved to `search::options`). A staleness hazard with no
  compile-time guard. Decision needed: commit to the facade (make modules `pub(crate)`,
  force `lanes_search::Foo`) or delete the block.
- **`search::engine` → `placement::nohome::home_sites` upward dependency
  (engine.rs:17,123)** — the data layer (documented as "below the `CzPlacement`
  peers") reaches *up* into a placement strategy; `home_sites` is `pub fn` purely to
  enable this. `home_sites` is stateless arch math and belongs in `ops/` or
  `primitives/`.
- **Vestigial `drivers/astar` seam** — `Expander`, `astar()`, and
  `run_search_legacy` are `#[allow(dead_code)]`, exercised only by in-file
  `#[cfg(test)]` tests. The production `SearchResult` type *also* lives here and is
  the crate's public `SearchResult`, so the file can't be deleted outright. Extract
  `SearchResult`/`solution_path` to its own module so the legacy A* shim can be
  removed cleanly.
- **`solve.rs` re-export shim** — only `CandidateAttempt`/`MultiSolveResult` are
  defined here; its 460-line `#[cfg(test)]` block is the de-facto integration suite
  for `solve_with_engine`/`solve_loose_goal`/`solve_single_heuristic`. A documented
  "will dissolve" transitional state that has persisted.
- **Internal structure supports the external contract** — yes. `run_with_components`
  is the single dispatch point all peers + `TargetSolver` share; the generic `MkGen`
  closure cleanly parameterizes per-strategy generator construction.

### AI-Drift Findings

- Commit `c4150d9` (`remove dead internal MoveSolver facade`) — **clean**. No
  `MoveSolver`/`PyMoveSolver` references survive across `src/`; none of the 14 touched
  files left an orphaned helper. The `configured()` factory in `heuristic.rs:151` is
  wired at all three documented call sites (`target_solver:138`, `loose_goal:246`,
  `receding_horizon:508`). No declared-but-unwired structs, empty modules, or `todo!`
  bodies. Stylistically coherent.
- Commit `cf2c037` (`remove legacy MoveSolver/PyMoveSolver and temp_regression
  infra`) — removal-only; no new surface to verify beyond the clean grep above.
- Commit `68ad857` (`cross-platform deterministic entropy`) — `EntropyOptions::seed`
  doc (`options.rs:119–127`) and the `base_seed.max(1)` semantics at `restarts.rs:195`
  agree. Coherent.
- Commit `96b4384` (`CzPlacement typed surface`) — the four-peer trait surface +
  `run_with_components` hub are internally consistent; the `lib.rs` facade staleness
  (still re-exporting from `search::solve` rather than `search::options`) is the
  residue of this large migration not finishing its re-export cleanup.
- **AI-comment-density signal**: `heuristic.rs` carries unusually heavy inline
  rationale (e.g. the `cz_pairs.is_some()` gating comments at 370–383, 459–470). The
  comments are accurate but *compensate* for an implicit two-mode contract the type
  signature doesn't express — documenting complexity rather than removing it.

### ⚠ Emerging Patterns

```
⚠ Emerging Pattern: "frontier::run_search call-site boilerplate"
  Appears in: restarts.rs:113–125, restarts.rs:131–143, restarts.rs:215–227,
              restarts.rs:300–312/314–328/331–344, heuristic.rs:894/938/1004
  Similarity: identical 11-argument run_search(...) call shape repeated 6× in
              production restarts.rs alone, differing only in the frontier `f`
              and occasionally `max_depth`.
  Signal: 6+ production instances, last touched this week (c4150d9)
  Suggested abstraction: a run_with_frontier(root, gen, goal, ctx, &mut f, budget,
              max_depth) helper fixing scorer/cost_fn/observer/state to defaults.
  Readiness: still evolving (restarts.rs touched within 7 days) — monitor; do not
              abstract mid-refactor.
```

```
⚠ Emerging Pattern: "single-lane escape move generation"
  Appears in: heuristic.rs:209–237 (generate_blocker_escape),
              heuristic.rs:240–263 (generate_all_escape),
              heuristic.rs:496–516 (inline spectator escape fallback)
  Similarity: three copies of the same inner loop — outgoing_lanes(loc) → skip if
              occupied → MoveSet::from_encoded(vec![lane]) + config.with_moves. The
              third is an inline open-coded copy of the first two.
  Signal: 3 instances, last touched this week
  Suggested abstraction: fn escape_moves_from(loc, occupied, index, qid) ->
              impl Iterator<MoveCandidate> consumed by all three.
  Readiness: still evolving (heuristic.rs is the top hotspot, 4 commits) — monitor.
```

```
⚠ Emerging Pattern: "u8-discriminant triplet round-trip"
  Appears in: heuristic.rs:396/424/504 (encode move_type/direction as u8),
              heuristic.rs:567–577 (decode via match-with-unreachable!)
  Similarity: enum → u8 → enum round-trip with hand-written reconstruction guarded
              by unreachable!.
  Signal: 4 encode sites + 1 decode site, all in one file
  Suggested abstraction: keep a typed (MoveType, u32, Direction) TripletKey (or a
              newtype) so no u8 cast/reconstruct is needed; the unreachable! arms
              disappear.
  Readiness: ready to abstract (contained to one file, encode/decode co-located) —
              but verify primitives::ordering ordering/hashing isn't relying on u8.
```

## 5. Open Questions

### Contract Divergence
1. Should `HeuristicGenerator` expose its two behavioral modes (legacy fixed-target
   vs entangling, gated on `ctx.cz_pairs.is_some()`) as an explicit type/enum rather
   than an undocumented runtime branch? The flag silently changes the
   contested-destination penalty (line 376), the no-positive-scores fallback width
   (3 vs 1, line 469), and pair-boosting (line 523).
2. Is the `search::solve` → `search::options` re-export chain a permanent
   compatibility layer or transitional? If permanent, why does `lib.rs` route through
   `solve::*` instead of the canonical `options::*`?

### Rust Health
1. `heuristic.rs:623` O(n²) candidate dedup: is there a known upper bound on
   candidates per bus group, or could a pathological dense arch make this hot? A
   `HashSet<MoveSet>` membership check would make it O(n).
2. `pick_best`'s `.expect("non-empty results")` (`restarts.rs:64`): worth converting
   to a `NonEmpty` input or returning `Option` to make the precondition type-enforced
   rather than runtime-panicking?

### Architectural Health
1. Where should `nohome::home_sites` live so `SearchEngine` stops reaching upward
   into a placement strategy? `ops/` (it's stateless arch math) seems the natural
   home — is there a reason it was placed in `nohome`?
2. Can `SearchResult` be extracted from `drivers/astar.rs` into `drivers/result.rs`
   (or `primitives`) so the `#[allow(dead_code)]` `Expander`/`astar`/`run_search_legacy`
   trio can be deleted? What is the plan to port the in-`astar.rs` tests to the v2
   `run_search` path that `drivers/mod.rs` flags as the prerequisite?

### AI-Drift
1. The `lib.rs` facade drifted (still references the pre-`96b4384` `solve::*` path).
   Should there be a lint/test asserting public re-exports match canonical module
   paths, or should the facade be deleted to remove the drift surface entirely?

### Emerging Patterns
1. The 11-arg `frontier::run_search` signature drives the call-site boilerplate. Is
   a builder/parameter-struct on the roadmap, or is the explicit arg list intentional
   for the in-progress refactor? (Abstracting now risks churn against the active branch.)
2. For the three escape-move generators: is `generate_all_escape` vs
   `generate_blocker_escape` vs the inline spectator fallback a deliberate performance
   distinction, or did the spectator path (line 496) get open-coded because the helpers
   took shapes that didn't fit?
