# Crate Review: bloqade-lanes-search (2026-05-31)

## 1. Context

`bloqade-lanes-search` is the Rust move-synthesis search engine that powers
neutral-atom placement and routing. It sits one hop above
`bloqade-lanes-bytecode-core` (its only Rust dependency besides `rand`,
`rayon`, `serde`, `serde_json`) and is consumed exclusively by
`bloqade-lanes-bytecode-python` — a thin PyO3 wrapper at
`crates/bloqade-lanes-bytecode-python/src/search_python.rs`. The crate is
a single library target with no binary or example targets.

Change activity is bursty: only 3 commits hit the crate in the last 30 days
but 11 in the last 90, dominated by AI-assisted feature work
(`Co-Authored-By: Claude` on 6 of 11). The activity skew is concentrated
in `entropy.rs` (7×), `solve.rs` (6×), `generators/heuristic.rs` (5×),
`frontier.rs` (5×), and `lib.rs` (5×). A `tests/temp_regression.rs`
integration suite was just landed to lock in current behavior ahead of an
upcoming refactor — this review's goal is to identify the highest-value
targets for that refactor.

## 2. External API Surface

### Public Type Inventory

| Name | Kind | Fields / Signature | Purpose |
|------|------|--------------------|---------|
| `MoveSolver` | struct (in `solve.rs`, **not** re-exported at root) | opaque: `index: LaneIndex`, `entangling_cache: OnceLock<…>`, `nohome_cache: OnceLock<…>` | Reusable solver: parse arch once, run many `solve*` calls. |
| `SolveResult` | struct (in `solve.rs`, **not** re-exported) | `status: SolveStatus`, `move_layers: Vec<MoveSet>`, `goal_config: Config`, `nodes_expanded: u32`, `cost: f64`, `deadlocks: u32`, `entropy_trace: Option<EntropyTrace>` | Outcome of any solve entry point. |
| `SolveStatus` | enum (in `solve.rs`, **not** re-exported) | `Solved`, `Unsolvable`, `BudgetExceeded` | Tristate solve outcome. |
| `SolveOptions` | struct (re-exported) | `strategy: Strategy`, `weight: f64`, `restarts: u32`, `deadlock_policy: DeadlockPolicy`, `lookahead: bool`, `top_c: Option<usize>` | Shared search-tuning knobs. |
| `Strategy` | enum (re-exported) | `AStar`, `HeuristicDfs`, `Bfs`, `GreedyBestFirst`, `Ids`, `Cascade { inner: InnerStrategy }`, `Entropy` | Selects search algorithm. |
| `InnerStrategy` | enum (re-exported) | `Ids`, `Dfs`, `Entropy` | Inner phase of `Strategy::Cascade`. |
| `EntropyOptions` | struct (**not** re-exported) | `max_movesets_per_group: usize`, `max_goal_candidates: usize`, `w_t: f64`, `collect_entropy_trace: bool` | Entropy-strategy tuning. |
| `EntanglingOptions` | struct (**not** re-exported) | `congestion_weight: f64`, `occupancy_penalty: f64`, `hungarian_horizon: Option<usize>` | Loose-goal Hungarian tuning. |
| `CandidateAttempt` | struct (re-exported) | `candidate_index: usize`, `status: SolveStatus`, `nodes_expanded: u32` | Per-candidate row in `MultiSolveResult`. |
| `MultiSolveResult` | struct (re-exported) | `result: SolveResult`, `candidate_index: Option<usize>`, `total_expansions: u32`, `candidates_tried: usize`, `attempts: Vec<CandidateAttempt>` | Result of `solve_with_generator`. |
| `Config` | struct (re-exported) | opaque; `new`, `len`, `is_empty`, `location_of`, `qubit_at`, `is_occupied`, `with_moves`, `iter`, … | Canonical hash-cached qubit-placement value. |
| `ConfigError` | struct (re-exported) | `duplicate_qubit_id: u32` | `Config::new` construction error. |
| `MoveCandidate` | struct (re-exported) | `move_set: MoveSet`, `new_config: Config` | Generator → search-loop transfer object. |
| `SearchContext<'a>` | struct (re-exported) | `index: &'a LaneIndex`, `dist_table: &'a DistanceTable`, `blocked: &'a HashSet<u64>`, `targets: &'a [(u32, u64)]`, `cz_pairs: Option<&'a [(u32, u32)]>` | Read-only architecture/problem context. |
| `SearchState` | struct (re-exported) | `entropy_map: HashMap<NodeId, EntropyNodeState>` | Mutable per-search state. |
| `MoveSet` | struct (re-exported) | sorted-encoded lanes; `new`, `from_encoded`, `decode`, `len`, `is_empty`, `encoded_lanes` | Compact sorted set of `LaneAddr`s. |
| `NodeId` | newtype `u32` (re-exported) | opaque | Arena index into `SearchGraph`. |
| `SearchGraph` | struct (re-exported) | opaque arena + transposition table | A* / search expansion graph. |
| `SearchResult` | struct (re-exported) | `goal: Option<NodeId>`, `nodes_expanded: u32`, `max_depth_reached: u32`, `graph: SearchGraph` | Low-level frontier-search output. |
| `UniformCost` | struct (re-exported) | unit | `CostFn` returning constant 1.0 per move-layer. |
| `DeadlockPolicy` | enum (re-exported) | `Skip`, `MoveBlockers`, `AllMoves` | How `HeuristicGenerator` handles deadlocks. |
| `ExhaustiveGenerator` / `GreedyGenerator` / `HeuristicGenerator` / `LooseTargetGenerator` | structs (re-exported) | builder-style | `MoveGenerator` implementors. |
| `AllAtTarget` / `EntanglingConstraintGoal` / `PartialPlacementGoal` | structs (re-exported) | constructors | `Goal` implementors. |
| `PairDistanceHeuristic` / `MaxHopHeuristic` / `SumHopHeuristic` | structs (re-exported) | `estimate_max`, `estimate_sum` | `Heuristic` implementors. |
| `LaneIndex` | struct (re-exported) | `arch_spec`, `outgoing_lanes`, `endpoints`, `position`, `lane_duration_us`, … | Precomputed lane/endpoint/position lookup. |
| `SearchEvent` / `SearchObserver` / `NoOpObserver` | enum + trait + struct (re-exported) | observer events | Search-tree visualization. |
| `RecedingHorizonOptions` | struct (re-exported) | 10 numeric/bool fields + `weight_grid: Vec<(f64, f64)>` | Receding-horizon knobs. |
| `default_weight_grid` | fn (re-exported) | `() -> Vec<(f64, f64)>` | Default `(cw, op)` sweep grid. |
| `DistanceScorer` / `EntropyScorer` | structs (re-exported) | `CandidateScorer` impls | Per-candidate scoring. |
| `DefaultTargetGenerator` / `TargetGenerator` / `TargetContext` / `CandidateError` | trait + impls (re-exported) | plugin system | Pluggable target-placement generator. |
| `CostFn`, `Goal`, `Heuristic`, `CandidateScorer`, `MoveGenerator` | traits (re-exported) | as in `traits.rs` | Core composable search-trait set. |
| Module-public-only: `entropy::{EntropyParams, EntropyTrace, EntropyTraceStep, MovesetMetrics, compute_moveset_metrics}` | mixed | — | Entropy bookkeeping + scoring formula. |
| Module-public-only: `heuristic::DistanceTable` | struct | precompute | Per-target BFS distance table. |
| Module-public-only: `nohome::NoHomeOptions` | struct | tuning | No-home placement tuning. |

### Responsibility Portraits

**MoveSolver** — The only solver type the consumer constructs. The PyO3
wrapper builds it once from arch JSON (or via a JSON round-trip from
`PyArchSpec`) and then calls `solve`, `solve_with_generator`,
`solve_entangling`, `solve_nohome`, `solve_entangling_rh`, and
`generate_candidates` repeatedly under `Py::allow_threads`. The contract is:
*given an arch, take (initial, targets-or-CZ-pairs, blocked, budget, option
bundle) and return a* `Result<SolveResult, ConfigError>` *(or
`MultiSolveResult`) without retaining GIL or holding Python refs*.

**SolveResult / SolveStatus** — Callers read every public field directly:
`status` (mapped to one of three strings), `move_layers` (decoded into
per-step `Vec<PyLaneAddr>`), `goal_config` (iterated for `(qid, LocationAddr)`),
`nodes_expanded`, `cost`, `deadlocks`, and the optional `entropy_trace`.
There is no method use besides field reads — the type is treated as a data
record. `SolveStatus`'s only use is the three-way match to a string.

**SolveOptions / EntropyOptions / EntanglingOptions / RecedingHorizonOptions / NoHomeOptions**
— All five are pure builder-target structs. The PyO3 layer 1:1 mirrors every
field as a Python keyword arg, validates ranges, and stores them. They are
never inspected after construction.

**Strategy / InnerStrategy / DeadlockPolicy** — Used only via exhaustive
`match` arms in both directions (Py enum ↔ Rust enum). Nine `Strategy`
variants and three `DeadlockPolicy` variants are individually surfaced as
Python enum members; any new variant requires a coordinated Python-side
change.

**TargetGenerator / DefaultTargetGenerator** — The consumer builds a
`Box<dyn TargetGenerator>` containing only `DefaultTargetGenerator` and the
wrapper *explicitly refuses* any other generator value. The trait's contract
reduces in practice to "be the default generator."

**LaneIndex** — Used inside `PyEntropyScorer`: built via `LaneIndex::new(parsed_arch)`,
then `index.endpoints(&lane)` is called to translate a lane into `(src, dst)`
`LocationAddr`s. Caller expects an O(1) lane→endpoint lookup keyed by `LaneAddr`.

**DistanceTable** — Built once in the entropy scorer via
`DistanceTable::new(target_locs, &index).with_time_distances(&index)`, then
passed by reference inside a `SearchContext` to `compute_moveset_metrics`.
Caller treats it as an opaque precompute.

**Config** — Reconstructed every `score_moveset` call from a Python
`BTreeMap<u32, LocationAddr>` via `Config::new(pairs)`, then mutated through
`with_moves(&[(qid, LocationAddr)])` and iterated via `iter()` / `qubit_at(loc)`.
Cheap to build, cheap to clone-and-update, queryable by qubit and by location.

**SearchContext** — Constructed manually (struct-literal) by the consumer in
`PyEntropyScorer::metrics`, passing `&index`, `&dist_table`, `&blocked`,
`&targets`, `cz_pairs: None`. Caller expects it to be a passive borrowed
bundle struct — five public fields, no methods needed.

**MoveSet** — Caller calls only `.decode() -> Vec<LaneAddr>`. Never builds
one, never inspects encoded form. Effectively a write-only solver output.

**EntropyParams / MovesetMetrics / compute_moveset_metrics /
EntropyTrace / EntropyTraceStep** — Wrapper builds `EntropyParams` from four
Python kwargs (`alpha, beta, gamma, w_t`) plus `..EntropyParams::default()`.
Calls `compute_moveset_metrics(old, new, occupied, ctx, params)` and exposes
every `MovesetMetrics` field plus a `mobility_gain()` method.
`EntropyTrace`/`EntropyTraceStep` are read field-by-field on every getter
(~17 step fields, each surfaced verbatim).

**Unused-by-consumer public surface** —
`LooseTargetGenerator`, `HeuristicGenerator`, `ExhaustiveGenerator`,
`GreedyGenerator`, `EntropyGenerator`, `Goal` impls, `Heuristic` impls,
`SearchResult`, `SearchGraph`, `NodeId`, `Frontier` family, `run_search`,
`SearchObserver` / `SearchEvent` (`NoOpObserver` only used implicitly),
`UniformCost`, `DistanceScorer`, `scorers::EntropyScorer`, all five core
traits, `CandidateError` / `validate_candidate`, all `nohome::*` helpers,
and `solve_entangling_rh_single`. These exist as scaffolding for strategies
that currently have only `MoveSolver` as their reachable entry point.

### API Friction Points

- `crates/bloqade-lanes-bytecode-python/src/search_python.rs:573-577` —
  `PyEntropyScorer::new` JSON-round-trips the arch (`serde_json::to_string`
  then `serde_json::from_str`) just to obtain an `ArchSpec` for
  `LaneIndex::new`. No `LaneIndex::from_arch_spec(&ArchSpec)` exists.
- `crates/bloqade-lanes-bytecode-python/src/search_python.rs:702-707` —
  `PyMoveSolver::from_arch_spec` repeats the same JSON round-trip.
  `MoveSolver::from_arch_spec(&ArchSpec)` would eliminate it.
- `crates/bloqade-lanes-bytecode-python/src/search_python.rs:485-490` —
  `PyMovesetMetrics::score` recomputes `alpha * D + beta * A + gamma * M`
  locally even though `MovesetMetrics::score(&EntropyParams)` exists in
  `entropy.rs:736`. The wrapper carries `alpha,beta,gamma` separately on
  `PyMovesetMetrics` to support this.
- `crates/bloqade-lanes-bytecode-python/src/search_python.rs:614-644` —
  `PyEntropyScorer::metrics` reconstructs a fresh `Config`, the `HashSet<u64>`
  of occupied locations, and a brand-new `SearchContext` literal on *every*
  call. No `SearchContext::builder()` or higher-level scorer-side helper.
- `crates/bloqade-lanes-bytecode-python/src/search_python.rs:205-216, 1596-1608` —
  Two places independently iterate `inner.move_layers`, call `MoveSet::decode()`,
  and rewrap each `LaneAddr` in `PyLaneAddr`.
- `crates/bloqade-lanes-bytecode-python/src/search_python.rs:194-198, 1542-1547` —
  `SolveStatus` is matched-to-string twice with identical arms. `SolveStatus`
  could provide `as_str(&self) -> &'static str`.
- `lib.rs:33-52` — Re-export surface is inconsistent: `SolveOptions`,
  `Strategy`, `InnerStrategy`, `CandidateAttempt`, `MultiSolveResult` are at
  the root, but their close siblings `MoveSolver`, `SolveResult`, `SolveStatus`,
  `EntropyOptions`, `EntanglingOptions` require `solve::` prefix. The Python
  wrapper imports both styles in the same file.

### Dead Public Surface

`pub` items not used by the only external consumer (many *are* exercised by
the new `tests/temp_regression.rs` and unit tests, so "dead" means
"unreachable from production code paths"):

- `SearchResult`, `SearchGraph`, `NodeId` (`graph.rs`, `astar.rs`)
- `MoveSet::new`, `MoveSet::from_encoded`, `MoveSet::len`, `MoveSet::is_empty`,
  `MoveSet::encoded_lanes` — only `decode` is consumer-reachable
- `MoveCandidate`, `SearchState` (`context.rs`)
- `UniformCost` (`cost.rs`)
- `ExhaustiveGenerator`, `GreedyGenerator`, `HeuristicGenerator`,
  `LooseTargetGenerator` (`generators/*.rs`)
- `AllAtTarget`, `EntanglingConstraintGoal`, `PartialPlacementGoal` (`goals.rs`)
- `PairDistanceHeuristic`, `MaxHopHeuristic`, `SumHopHeuristic`, `MisplacedHeuristic`,
  `HopDistanceHeuristic`, several `DistanceTable` methods (`heuristic.rs`,
  `heuristics.rs`)
- `NoOpObserver`, `SearchEvent`, `SearchObserver` (`observer.rs`)
- `DistanceScorer`, `scorers::EntropyScorer` (`scorers/*.rs`)
- `CandidateError`, `validate_candidate` (`target_generator.rs`)
- Core traits `CostFn`, `Goal`, `Heuristic`, `CandidateScorer`, `MoveGenerator`
  (`traits.rs`) — never implemented or named outside the crate
- `Frontier`, `PriorityFrontier`, `BfsFrontier`, `DfsFrontier`, `IdsFrontier`,
  `run_search` (`frontier.rs`)
- `EntropyGenerator` (`generators/entropy.rs`)
- `home_sites`, `partner_weights`, `nearest_home_layout`,
  `candidate_return_layouts`, `solve_entangling_rh_single` (`nohome.rs`,
  `receding_horizon.rs`)
- Multiple `LaneIndex` methods: `num_locations`, `lanes_for`, `lane_for_source`,
  `position`, `bus_groups`, `triplets`, `lanes_for_all_zones`, `bus_groups_no_zone`,
  `lane_duration_us`, `fastest_lane_duration_us`
- `MovesetMetrics::score(&EntropyParams)` and `EntropyParams.{w_d, w_m,
  reversion_steps, e_max, max_candidates}`

Concentration is highest in `traits.rs`, `frontier.rs`, `goals.rs`,
`observer.rs`, and the alternate generators (`Greedy`, `Exhaustive`,
`Heuristic`, `LooseTarget`) — scaffolding ahead of demand.

## 3. Internal Architecture

### Module Map

| Module | Responsibility |
|--------|----------------|
| `lib.rs` | Crate root: declares modules and re-exports the public surface (inconsistently). |
| `config` | Compact hash-cached canonical `(qubit_id, encoded_location)` state. |
| `graph` | Arena `SearchGraph` + `NodeId` + `MoveSet`; transposition table keyed by `Config`. |
| `lane_index` | Lane lookups off `ArchSpec` (lanes by triplet, outgoing lanes, endpoints, positions, durations). |
| `aod_grid` | `BusGridContext`: greedy-init + iterative-merge AOD rectangular lane grids per bus group. |
| `heuristic` | `DistanceTable` (precomputed BFS hop distances + optional Dijkstra time), `HopDistanceHeuristic`, `PairDistanceHeuristic`. |
| `heuristics` | `MaxHopHeuristic` / `SumHopHeuristic` trait wrappers around `HopDistanceHeuristic`. |
| `traits` | `MoveGenerator`, `CandidateScorer`, `CostFn`, `Goal`, `Heuristic`. |
| `context` | `SearchContext<'a>`, `SearchState`, `MoveCandidate`. |
| `cost` | `UniformCost` (`CostFn` returning 1.0 per move-layer). |
| `goals` | `AllAtTarget`, `PartialPlacementGoal`, `EntanglingConstraintGoal`. |
| `scorers/{mod,distance,entropy}` | `DistanceScorer` + `EntropyScorer` (entropy delegates to `entropy::score_moveset`). |
| `observer` | `SearchObserver` trait + `SearchEvent` enum + `NoOpObserver`. |
| `ordering` | Shared deterministic tie-break comparators. |
| `astar` | Legacy `Expander` trait, `SearchResult`, and a thin `astar()` convenience wrapper. |
| `frontier` | `Frontier` trait + four impls + generic `run_search` + legacy `run_search_legacy`. |
| `entropy` | Self-contained entropy-guided search engine PLUS shared utilities (`find_path_occupied`, `score_moveset`, `compute_moveset_metrics`, `generate_candidates`). |
| `entangling` | Entangling-pair queries + Hungarian-assignment family + calibrated constants. |
| `nohome` | Hungarian-based post-CZ "any home site" return assignment. |
| `generators/{mod,heuristic,exhaustive,greedy,entropy,loose_target}` | Five `MoveGenerator` impls. |
| `target_generator` | `TargetGenerator` trait + `DefaultTargetGenerator` + `validate_candidate`. |
| `solve` | `MoveSolver`, `Strategy`, `SolveOptions`, `SolveResult`: orchestration over all generators / heuristics / scorers / frontiers. |
| `receding_horizon` | MPC-style outer loop on top of IDS + Hungarian compass. |
| `test_utils` | `cfg(test)` helpers. |

### Internal Interaction Graph

```
Tier 0  config   lane_index
                  │
Tier 1  graph ──→ │     aod_grid ──→ lane_index
        heuristic → config, lane_index
        traits   → config, context, graph
        context  → config, graph, heuristic, lane_index
        ordering → config, graph

Tier 2  cost, goals, heuristics, scorers/*, observer, target_generator
        entangling → config, heuristic, lane_index
        entropy   → aod_grid, astar, config, context, graph, heuristic,
                    lane_index, ordering, traits
        astar    ⇄ frontier  (bidirectional cycle, see Architectural Health)

Tier 3  generators/heuristic → aod_grid, config, context, graph, heuristic,
                                lane_index, ordering, traits
        generators/exhaustive → config, context, graph, lane_index, traits
        generators/greedy    → aod_grid, config, context, entropy,
                                graph, traits  (reaches into entropy)
        generators/entropy   → config, context, entropy, graph, traits
        generators/loose_target → config, context, entangling,
                                   generators::heuristic, graph, heuristic,
                                   lane_index, traits

Tier 4  nohome   → config, entangling, entropy::find_path_occupied,
                    heuristic, lane_index
        solve    → 16 siblings (every module above except test_utils)
        receding_horizon → 14 siblings INCLUDING solve  (layering inversion)
```

### `pub(crate)` Type Inventory

| Name | Kind | Defined In | Purpose |
|------|------|------------|---------|
| `BusGridContext` | struct | `aod_grid` | Per-bus AOD-rectangle builder used by every grid-producing generator. |
| `TripletKey` | type alias `(u8, u32, u8)` | `ordering` | `(move_type, bus_id, direction)` discriminant tuple. |
| `cmp_triplet_entry_tiebreak`, `cmp_qubit_lane_dst_tiebreak`, `cmp_moveset_config_tiebreak` | fns | `ordering` | Total-order tiebreakers. |
| `find_path_occupied` | fn | `entropy` | BFS shortest-lane-path; used by `entropy`, `nohome`, `generators::greedy`. |
| `score_moveset` | fn | `entropy` | Moveset scoring formula; used by `entropy_search` and `EntropyScorer`. |
| `generate_candidates` | fn | `entropy` | Entropy candidate generation; used by `EntropyGenerator` and `entropy_search`. |
| `OCCUPANCY_PENALTY_DEFAULT`, `MOVE_PENALTY`, `LOOKAHEAD_BETA` | constants | `entangling` | Calibrated cost-matrix constants. |
| `hungarian` | fn | `entangling` | Rectangular Hungarian assignment solver. |
| `run_search_legacy` | fn | `frontier` | Legacy loop driven by `Expander`; called only from `astar::astar` and tests. |
| `Expander` | trait | `astar` | Legacy successor-generation trait — `pub(crate)` + `#[allow(dead_code)]`. |
| `NodeId(pub(crate) u32)` | newtype | `graph` | Opaque arena handle. |

### Responsibility Portraits (Internal Types)

**Config** — Equal-by-set, stable hash across construction order; `with_moves`
preserves sort invariant without invalidating the cached hash.
`SearchGraph`'s transposition-table correctness depends on this; divergent
hashes for equal configs would silently double-insert and break A*'s
optimality.

**MoveSet** — Equal-set ⇒ equal-MoveSet, regardless of construction order.
Every generator's anti-duplicate guard
(`candidates.iter().any(|(_, ms, _)| *ms == move_set)`) depends on this.

**SearchGraph / NodeId** — Frontier loops trust `graph.insert` returns
`(NodeId, is_new)` where `is_new == false` indicates a transposition at
equal-or-better g-cost. Closed-set tracking is keyed by the integer index
inside `NodeId(u32)`; replacing the keying would break A* optimality.

**LaneIndex** — Callers trust: (a) `outgoing_lanes(loc)` returns *every*
legal lane from that location, (b) `endpoints(&lane)` is `Some` for every
lane returned by any other index method, (c) the `arch_spec()` getter hands
out the same arch used to build the index, (d) `Clone` is cheap enough to
wrap in `Arc` (used by `LooseTargetGenerator`).

**DistanceTable** — Returns BFS-exact lane-hop counts (admissibility property
required by A*). Built *once* per solve; consumers trust it lives for the
entire run.

**SearchContext / MoveCandidate / SearchState** — Borrowed-bundle struct;
the `cz_pairs: Option<…>` field is used as a soft mode switch
("loose-goal vs. fixed-target") gating behavior in
`generators/heuristic.rs:350` and `heuristic.rs:443`. `SearchState::entropy_map`
is owned through the search and uniquely written by `EntropyGenerator`.

**HeuristicGenerator + DeadlockPolicy** — `solve`, `receding_horizon`, and
`LooseTargetGenerator` all *embed* one and depend on (a) builder-style
`with_*` composability, (b) `deadlock_count` accuracy for restart-quality
reporting, (c) the `cz_pairs.is_some()` branches toggling between fixed-target
and loose-goal behavior. The `Cell<u32>` deadlock counter is interior state;
each parallel restart constructs its own generator.

**Hungarian family in `entangling`** — Returned `Vec<(qubit_id, encoded_target)>`
must be a valid `ctx.targets` payload (one entry per active CZ qubit, no
duplicates, every target reachable). `seed != 0` perturbs the cost matrix
(for diverse parallel restarts); `congestion_weight > 0` triggers the
rebalance pass.

**BusGridContext** — Returned lane lists each form a complete, AOD-compatible
X×Y rectangle. `frontier::debug_assert_candidates_valid` re-validates against
`ArchSpec::check_lanes`, so any non-rectangular result trips a debug
assertion immediately.

**find_path_occupied (entropy)** — Returns `None` only when no
occupancy-respecting path exists; `Some(vec![])` only when source ==
destination. Read-only over `HashSet<u64>` (`generators::greedy` temporarily
mutates the set and re-inserts the moved qubit's home location).

**EntropyParams / score_moveset / generate_candidates** — `scorers::entropy::EntropyScorer`
and `generators::entropy::EntropyGenerator` are thin wrappers that delegate;
the actual scoring formula lives in `entropy::score_moveset`. Swapping these
wrappers in/out doesn't change behavior compared to running
`entropy::entropy_search` directly with the same `EntropyParams`.

### Internal Coupling Hotspots

- `solve` → 16 siblings — integration layer
- `receding_horizon` → 14 siblings including `solve` — second integration
  layer, depends on the first (layering inversion)
- `entropy` → 9 siblings — entropy module is essentially a parallel mini
  search engine
- `frontier` → 6 siblings, cycles with `astar`
- `generators/heuristic` → 8 siblings — default generator is the busiest leaf
- `generators/loose_target` → 8 siblings (including `generators::heuristic`)
- `nohome` → 5 siblings (reuses both Hungarian and entropy's BFS)
- `generators/greedy` → 6 siblings (pulls `find_path_occupied` from `entropy`)
- `goals` → 3 siblings (reaches into `entangling::build_partner_map`)
- `context` → 4 siblings (shared-state hub)
- `astar` → 3 siblings, **bidirectional cycle with `frontier`**

Key cycles and re-entrancy flagged: `astar` ⇄ `frontier` (one logical
module split across two files); `generators/loose_target` depends on
`generators/heuristic`; `nohome` and `generators/greedy` both reach into
`entropy::find_path_occupied`.

## 4. Critical Evaluation

### Contract vs Implementation Divergence

| Public Type | Classification | Explanation |
|-------------|----------------|-------------|
| `MoveSolver` | GAP (communication) | Works fine, but not re-exported from `lib.rs`; consumers reach `bloqade_lanes_search::solve::MoveSolver` while siblings (`SolveOptions`, `Strategy`) sit at the crate root. |
| `SolveResult` / `SolveStatus` / `EntropyOptions` / `EntanglingOptions` | GAP (communication) | Same as above; the Python wrapper string-matches `SolveStatus` ("solved"/"unsolvable"/"budget_exceeded") in two places because no `as_str()` / `Display` is provided. |
| `SearchContext` | GAP (drift) | Documented as the shared search-time context, but the only way to build one is a struct literal; `PyEntropyScorer` reconstructs the same five-field literal on every `metrics()` call. No constructor or builder. |
| `LaneIndex` | GAP (communication) | Constructed only via `LaneIndex::new(ArchSpec)`. Python consumers have a `PyArchSpec` and pay a JSON round-trip. No `from_arch_spec` shortcut. |
| `EntropyScorer` | GAP (drift) | Implements `CandidateScorer`, but the only external consumer (`PyEntropyScorer`) bypasses it and calls `entropy::compute_moveset_metrics` directly. Trait wrapper is dead from the consumer's view. |
| `MovesetMetrics::score` | GAP (communication) | Exists on the metrics struct, but the Python wrapper recomputes the formula because it carries `alpha/beta/gamma` separately and there is no `score_with_weights(alpha, beta, gamma)`. |
| `TargetGenerator` trait | GAP (drift) | Only `DefaultTargetGenerator` implements it; the Python wrapper "explicitly refuses custom generators." Plugin abstraction has no second implementor. |
| `Expander` trait (`astar.rs:15`) | GAP (drift) | `pub(crate)` + `#[allow(dead_code)]`. Survives only to keep `frontier::run_search_legacy` + `astar::astar` compiling for in-crate tests. `MoveGenerator` has fully superseded it. |
| `MoveCandidate`, `MoveSet`, `Config`, `SearchGraph`, `NodeId` | MATCHES | Used as documented; `MoveSet::decode()` is the only `MoveSet` method consumed and works. |
| `Frontier` trait + `PriorityFrontier`/`BfsFrontier`/`DfsFrontier`/`IdsFrontier` | MATCHES internally / UNREACHABLE externally | Traversal-strategy abstraction is implemented correctly but only reachable via `MoveSolver::solve`'s `Strategy` enum. |
| `MoveSolver::solve_with_generator` | MATCHES | Does what its docstring promises; `validate_candidate` + budget threading work. |
| `RecedingHorizonOptions`, `default_weight_grid` | MATCHES | Re-exported and used; signatures align with `solve_entangling_rh`. |
| `SolveOptions` / `Strategy` / `InnerStrategy` | MATCHES | Plain data carriers; behave as documented. |

### Rust Health Findings

*(hotspot files: `entropy.rs`, `solve.rs`, `generators/heuristic.rs`,
`frontier.rs`, `lib.rs`, `heuristic.rs`, `lane_index.rs`, `aod_grid.rs`)*

- `solve.rs:269` — **note** `pick_best` calls `.expect("non-empty results")`.
  Every caller passes a non-empty `Vec`, but the invariant lives at the call
  site, not at the type.
- `solve.rs:1489` — **warn** `Config::new(...).expect("initial was valid on entry")`.
  Correct as re-validation, but the duplicated "build empty unsolvable result"
  block appears 6 times in this file (lines 247, 1051, 1307, 1346, 1419, 1487).
  See Emerging Patterns.
- `solve.rs:533` — **note** `unreachable!("IDS/DFS/Cascade/Entropy handled before run_strategy_v2")`.
  Reachable only via misuse from within the file.
- `entropy.rs:636, 641` and `generators/heuristic.rs:545, 549` — **warn**
  `unreachable!()` on `MoveType` / `Direction` discriminants after `as u8`
  round-trip. Discriminant laundering is unnecessary if `TripletKey` stored
  the enums directly (`(MoveType, u32, Direction)`).
- `entropy.rs:440` — **note** `assert!(params.max_movesets_per_group > 0, …)`
  runs in a hot generation path on every node expansion.
- `entropy.rs:1164` — **warn** `let hard_limit = max_expansions.unwrap_or(ctx.index.num_locations() as u32 * 10);`
  and `iterations >= hard_limit * 2` can overflow `u32` on large architectures
  (~21k locations × 10 × 2 = 420k is safe today, but no guard).
- `entropy.rs:849-952` — **warn** `compute_moveset_metrics` (754–851) and
  `score_moveset` (854–952) implement essentially the same loop. An explicit
  test at line 2042 asserts no shared code path. Duplication is intentional
  but unstructured.
- `solve.rs`, `frontier.rs`, `generators/heuristic.rs`, `lane_index.rs`,
  `heuristic.rs`, `aod_grid.rs` — **clean** no `unsafe`, no `todo!()` /
  `unimplemented!()` / `panic!()` in production paths, no `unwrap()` /
  `expect()` outside tests.
- **Lifetimes:** `SearchContext<'a>` carries one lifetime that propagates
  correctly into `HopDistanceHeuristic<'a>`, `PairDistanceHeuristic<'a>`,
  `DistanceTable` (owned), `LaneIndex` (owned). No compounding lifetime
  parameters detected.

### Architectural Health Findings

- **Inconsistent re-export policy** (`lib.rs:33-52`): `pub use solve::{CandidateAttempt, InnerStrategy, MultiSolveResult, SolveOptions, Strategy}`
  but **not** `MoveSolver`, `SolveResult`, `SolveStatus`, `EntropyOptions`,
  `EntanglingOptions`. The main entry point is reached through `solve::*`
  while its companions sit at the root.
- **`entropy.rs` is two modules in one** (2195 lines): a self-contained
  entropy-guided search engine (`entropy_search`, `EntropyTrace`,
  `EntropyState`, resume buffer) *and* a utility module providing
  `find_path_occupied`, `score_moveset`, `compute_moveset_metrics`,
  `generate_candidates`, `blended_distance` to `nohome.rs`,
  `generators/greedy.rs`, `generators/entropy.rs`, `scorers/entropy.rs`. The
  cross-module reach makes "the module" hard to delete or refactor.
- **`astar.rs` ⇄ `frontier.rs` bidirectional cycle**: `astar::Expander` is
  `pub(crate)` + `#[allow(dead_code)]`; `astar::astar()` exists only for tests
  and calls `frontier::run_search_legacy`. They are one logical module split
  across two files. The only public artifact `astar.rs` exposes that callers
  use is `SearchResult`, which has no algorithmic content. Consolidating
  would reduce surface without losing functionality.
- **`solve.rs` is a 2005-line god module** importing 16 siblings, inlining
  per-strategy `match` dispatch, the cleanup pass, restart bookkeeping, and
  three caches (`EntanglingCache`, `NoHomeCache`, `OnceLock` lifecycle).
  `receding_horizon.rs` then imports `solve.rs` to reuse
  `SolveResult`/`SolveStatus`/`SolveOptions`, producing a **layering
  inversion** (the "high-level" orchestrator depends on `solve.rs` which
  itself orchestrates).
- **`SearchContext` is plumbing**: 5-field struct literal that callers —
  including the Python wrapper — assemble by hand. Either it deserves a
  builder, or it should split into `ArchContext` (per-arch: `index`,
  `dist_table`) and `ProblemContext` (per-call: `blocked`, `targets`,
  `cz_pairs`).
- **Dual entry points to `solve_entangling` via `solve_entangling_rh`**
  (`solve.rs:978-1107`): the fallback closure reconstructs
  `SolveOptions { restarts: 1, ..opts.clone() }` and calls
  `self.solve_entangling` recursively. The two paths share clipped-future
  logic, blocked encoding, opts-upgrade logic, and the same `pick_best`
  discriminator.
- **`Strategy::Cascade { inner: InnerStrategy }`** uses two enums to express
  "primary strategy + optional refinement". `InnerStrategy` is a strict
  subset of `Strategy` (Ids, Dfs, Entropy); the match in `run_inner`
  re-translates `InnerStrategy → Strategy` semantics. A single `Strategy`
  with an optional `refine_with_astar: bool` would express the same model.
- **Scorer + Generator double-sort** (`frontier.rs:713-718` +
  `generators/heuristic.rs:615`): the heuristic generator already sorts
  candidates by score before emitting them; `run_search` immediately re-sorts
  using `DistanceScorer`. Wasted work on every expansion.

### AI-Drift Findings

- Commit `17971e1` ("no-return placement family + loose-goal Rust solver"):
  introduced `nohome.rs`, `receding_horizon.rs`, `LooseTargetGenerator`,
  `generators/loose_target.rs`, `entangling.rs`, the generators/* tree and
  scorers/* tree. Stylistically coherent. **Wiring issues**: `EntropyScorer`
  (`scorers/entropy.rs:29`) was introduced as `impl CandidateScorer` but the
  only external consumer (`PyEntropyScorer`) bypasses it and calls the
  entropy primitives directly — the trait implementation is unreachable.
  `LooseTargetGenerator` wraps `HeuristicGenerator` and rewrites `ctx.targets`
  per call, introducing intra-generators coupling.
- Commit `65dc14c` ("entropy search tree visualization"): added
  `EntropyTrace` / `EntropyTraceStep` plumbing inside `entropy.rs`.
  **Wiring issue**: `EntropyTraceStep` carries 14 optional fields and is
  instantiated at ~9 call sites with each filling a different subset.
  The struct has the shape of a tagged union but is encoded as a struct.
- Commit `227851d` ("overhaul Rust entropy search and trace plumbing"):
  added `compute_moveset_metrics` alongside the existing `score_moveset`.
  Both walk identical loops over `ctx.targets`. The new code is documented
  as "extends `score_moveset`'s scalar output", but neither calls the other
  and an explicit test (`entropy.rs:2042`) asserts they don't share code.
- Commit `81466ee` ("Rust-native target generator plugin system"): added
  `target_generator.rs` with `TargetGenerator` / `TargetContext` /
  `CandidateError`. **Wiring issue**: only one implementor; the Python
  wrapper "explicitly refuses custom generators". The plugin abstraction
  was built ahead of demand.
- Commit `855b424` ("complete Rust search infrastructure with entropy,
  cascade, and IDS strategies"): laid down `frontier.rs`, `astar.rs`,
  `traits.rs`, `observer.rs`, `cost.rs`, `heuristics.rs`. **Wiring issues**:
  `MisplacedHeuristic`, `MaxHopHeuristic`, `SumHopHeuristic` are all `pub`
  and re-exported but only `HopDistanceHeuristic::estimate_max/estimate_sum`
  is invoked from `solve.rs`. `NoOpObserver`, `SearchEvent`, `SearchObserver`
  are public but the consumer only ever passes `&mut NoOpObserver`.
- Commit `74bcb1c` ("Rust A* search infrastructure"): introduced `astar.rs`
  + `expander.rs` (later deleted) + `heuristic_expander.rs` (later deleted).
  The leftover `Expander` trait in `astar.rs` is `pub(crate)` +
  `#[allow(dead_code)]` — a fossil from the deleted expander modules that
  survives only to keep test code compiling.

### ⚠ Emerging Patterns

⚠ Emerging Pattern: "Empty/Unsolvable SolveResult literal"
  Appears in: `solve.rs:247-255`, `solve.rs:1051-1059`, `solve.rs:1307-1315`,
              `solve.rs:1346-1354`, `solve.rs:1419-1427`, `solve.rs:1487-1499`
  Similarity: Each builds the same 7-field literal `{status, move_layers: Vec::new(),
              goal_config: <root>, nodes_expanded: <local>, cost: 0.0,
              deadlocks: <local>, entropy_trace: None}` with one of three status variants.
  Signal: 6 instances inside `solve.rs`, all touched in recent commits.
  Suggested abstraction: `SolveResult::unsolvable(root_config)`,
                         `SolveResult::budget_exceeded(root_config, nodes)`,
                         `SolveResult::from_search(SearchResult, deadlocks, max_exp)`.
  Readiness: still evolving — recent commits touch some of these blocks.

⚠ Emerging Pattern: "Scored-entry triplet sort with tiebreak"
  Appears in: `entropy.rs:137-150` (`cmp_scored_entries`),
              `entropy.rs:152-163` (`cmp_group_entries`),
              `entropy.rs:165-168` (`cmp_scored_candidates`),
              `generators/heuristic.rs:46-59` (`cmp_scored_triples`),
              `generators/heuristic.rs:61-64` (`cmp_candidates`).
  Similarity: All five compare a score descending, then delegate to
              `cmp_triplet_entry_tiebreak` / `cmp_moveset_config_tiebreak` /
              `cmp_qubit_lane_dst_tiebreak` from `ordering.rs`.
  Signal: 5 instances; `entropy.rs` carries 3.
  Suggested abstraction: a `ScoreOrd<S, T>` helper in `ordering.rs`, or move
                         the comparators themselves into `ordering.rs`.
  Readiness: ready to abstract — `ordering.rs` already exists for this purpose.

⚠ Emerging Pattern: "Per-qubit target loop with old/new occupancy + blended distance + mobility"
  Appears in: `entropy.rs:743-851` (`compute_moveset_metrics`),
              `entropy.rs:854-952` (`score_moveset`),
              `entropy.rs:490-569` (inside `generate_candidates`),
              `generators/heuristic.rs:319-381` (score-each-(qid,lane) loop),
              `entropy.rs:319-336` (`best_untried_moveset_score`).
  Similarity: Same pattern — iterate `ctx.targets`, decode src/dst, look up
              `dist_table.distance`, `blended_distance`, iterate
              `index.outgoing_lanes(loc)`, accumulate distance progress + mobility.
  Signal: 5 instances across 2 files; commit `227851d` added
          `compute_moveset_metrics` alongside the pre-existing `score_moveset`.
  Suggested abstraction: a `QubitMobilityWalk` iterator yielding
                         `(qid, d_before, d_after, mob_before_delta, mob_after_delta)`.
  Readiness: monitor — abstract once entropy params stop drifting.

⚠ Emerging Pattern: "make_generator closure for HeuristicGenerator with options"
  Appears in: `solve.rs:698-703` (in `solve`),
              `solve.rs:860-882` (in `solve_entangling`),
              implicitly in `receding_horizon::solve_entangling_rh_single`.
  Similarity: Each builds a closure `|seed, policy|
              HeuristicGenerator::new().with_deadlock_policy(policy)
              .with_lookahead(lookahead).with_seed(seed)[.with_top_c(c)]`,
              optionally wrapped by `LooseTargetGenerator`.
  Signal: 2–3 instances.
  Suggested abstraction: `GeneratorFactory` builder on `MoveSolver` or `SolveOptions`.
  Readiness: still evolving — `solve_entangling_rh` is recent.

⚠ Emerging Pattern: "Clipped-future-layers slice + opts-upgrade-from-Skip-to-MoveBlockers"
  Appears in: `solve.rs:779-783` + `solve.rs:836-845` (solve_entangling),
              `solve.rs:1005-1009` + `solve.rs:1013-1022` (solve_entangling_rh),
              `solve.rs:1166-1175` (solve_nohome — only the opts upgrade).
  Similarity: Identical match-on-`hungarian_horizon` slicing + identical
              "if Skip → MoveBlockers" struct-update.
  Signal: 3 instances of the slice, 3 of the upgrade.
  Suggested abstraction: `EntanglingOptions::clipped_future_layers(&self, all: &[…]) -> &[…]`
                         and `SolveOptions::upgraded_for_entangling(&self) -> Cow<SolveOptions>`.
  Readiness: ready to abstract — both helpers are pure and have stabilized.

⚠ Emerging Pattern: "JSON round-trip from PyArchSpec to ArchSpec to LaneIndex"
  Appears in: `crates/bloqade-lanes-bytecode-python/src/search_python.rs:573-577`,
              `crates/bloqade-lanes-bytecode-python/src/search_python.rs:702-707`.
  Similarity: Both reconstruct an `ArchSpec` by JSON-serializing then JSON-parsing
              a `PyArchSpec`.
  Signal: 2 instances in the consumer.
  Suggested abstraction: `LaneIndex::from_arch_spec(&ArchSpec)` and
                         `MoveSolver::from_arch_spec(&ArchSpec)`.
  Readiness: ready to abstract — both call sites are stable.

## 5. Open Questions

### Contract Divergence

1. Should `MoveSolver`, `SolveResult`, `SolveStatus`, `EntropyOptions`, and
   `EntanglingOptions` be re-exported at the crate root alongside
   `SolveOptions` and `Strategy`, or should the root-level re-exports of
   those companion types be removed so that everything is reached via
   `solve::*`? The current split surprises the Python consumer.
2. `EntropyScorer` implements `CandidateScorer` but the Python wrapper
   bypasses it and calls `entropy::compute_moveset_metrics` directly. Is
   `EntropyScorer` actually needed in production, or is it a vestige of the
   `CandidateScorer` abstraction that should be deleted along with the
   unused `MaxHopHeuristic` / `SumHopHeuristic` / `MisplacedHeuristic`
   siblings?
3. `SearchContext` is a 5-field struct literal with no constructor and is
   rebuilt every entropy-scorer call. Should `SearchContext` gain a builder,
   or split into `ArchContext` (per-arch) and `ProblemContext` (per-call)?

### Rust Health

1. `solve.rs:269`'s `pick_best` panics on empty input with
   `.expect("non-empty results")`. Would it be safer to take
   `(SolveResult, impl IntoIterator<Item=SolveResult>)` so the type system
   enforces non-emptiness?
2. `entropy.rs:1164`'s `hard_limit * 2` and `num_locations() as u32 * 10`
   can both silently overflow `u32` on large architectures. Should the
   iteration cap use `u64` arithmetic with a saturating cast?
3. The four `unreachable!()` calls in `entropy.rs:636-642` and
   `generators/heuristic.rs:545-551` exist solely because
   `TripletKey = (u8, u32, u8)` round-trips `MoveType` and `Direction`
   discriminants. Would changing `TripletKey` to
   `(MoveType, u32, Direction)` eliminate the discriminant laundering
   entirely?

### Architectural Health

1. `entropy.rs` (2195 lines) is both a self-contained search engine *and* a
   utility provider for `nohome.rs`, `generators/greedy.rs`,
   `generators/entropy.rs`, `scorers/entropy.rs`. Should the shared helpers
   (`find_path_occupied`, `score_moveset`, `compute_moveset_metrics`,
   `generate_candidates`, `blended_distance`) move to a separate
   `entropy_utils.rs` so the entropy *engine* can be deleted without
   churning four other modules?
2. `astar.rs` carries a `pub(crate)` `Expander` trait + `#[allow(dead_code)]`
   `astar()` function that only exist for `frontier::run_search_legacy`
   tests. Is the simpler answer to (a) delete `astar.rs` entirely, moving
   `SearchResult` into `frontier.rs` or a new `result.rs`, and (b) port the
   in-crate tests to the trait-based `run_search`?
3. `solve.rs` (2005 lines) imports 16 siblings and contains the result types
   that `receding_horizon.rs` then imports back. Should
   `SolveResult`/`SolveStatus`/`SolveOptions`/`EntropyOptions`/`EntanglingOptions`
   move into a `solve_result.rs` (or back into `lib.rs`) so the
   orchestration files are siblings of equal rank rather than parent/child?
4. `Strategy::Cascade { inner: InnerStrategy }` uses two enums where
   `InnerStrategy` is a strict subset of `Strategy` (Ids, Dfs, Entropy).
   Would folding cascade into a single field — e.g.
   `SolveOptions { strategy: Strategy, refine_with_astar: bool }` — better
   express the actual algorithm?
5. `frontier::run_search` re-sorts candidates with `DistanceScorer`
   (lines 713-718) even though `HeuristicGenerator::generate` already emits
   them in score order. Is the per-expansion re-sort intentional (for
   non-heuristic generators) or accidental cost?

### AI-Drift

1. Commit `81466ee` introduced `TargetGenerator` as a plugin trait, but
   `DefaultTargetGenerator` is still the only implementor and the Python
   wrapper "explicitly refuses custom generators." Should the trait be
   removed and `solve_with_generator` accept a closure
   (`impl Fn(&TargetContext) -> Vec<Vec<(u32, LocationAddr)>>`) instead?
2. Commit `855b424` added `MisplacedHeuristic`, `MaxHopHeuristic`,
   `SumHopHeuristic`, `BfsFrontier`, `DfsFrontier`, `IdsFrontier`,
   `NoOpObserver`, `SearchEvent`, `SearchObserver`, and the
   `EntropyGenerator` / `GreedyGenerator` / `ExhaustiveGenerator` scaffolding.
   The only one of these the Python consumer touches is `NoOpObserver`
   (implicitly via `MoveSolver::solve`). Should the unused scaffolding be
   `pub(crate)`-demoted until a second consumer materializes?
3. Commit `227851d` added `compute_moveset_metrics` next to the existing
   `score_moveset` (both in `entropy.rs`) with overlapping per-target loops
   and an explicit test asserting they don't share a code path. Was the
   no-sharing deliberate for hot-path perf, and if so, would a
   `#[inline]` shared inner helper preserve perf while removing the
   duplication?
4. Commit `65dc14c`'s `EntropyTraceStep` has 14 nullable fields and is
   instantiated as a verbose struct literal at ~9 call sites. Would
   `enum EntropyTraceEvent { Descend{…}, Revert{…}, EntropyBump{…},
   Goal{…}, FallbackStart{…} }` better match how the field subsets actually
   cluster?

### Emerging Patterns

1. The "Empty/Unsolvable SolveResult literal" pattern repeats 6 times across
   `solve.rs`. Should `SolveResult` grow constructor methods
   (`SolveResult::unsolvable(root)`,
   `SolveResult::budget_exceeded(root, nodes)`,
   `SolveResult::from_search(SearchResult, deadlocks, budget)`) before the
   refactor begins, so post-refactor diffs are smaller?
2. The "Scored-entry triplet sort with tiebreak" pattern lives in both
   `entropy.rs` and `generators/heuristic.rs` with five comparator functions
   all routing into `ordering.rs`. Should the comparators themselves move
   into `ordering.rs` next to their tiebreak helpers, or should `ordering.rs`
   expose a generic `score_then_tiebreak<S, F>(…)` adapter?
3. The "Per-qubit target loop with old/new occupancy + blended distance +
   mobility" pattern has 5 near-duplicate occurrences. Would a
   `QubitMobilityWalk` iterator (yielding
   `(qid, d_before, d_after, mob_before_delta, mob_after_delta)`)
   consolidate `score_moveset`, `compute_moveset_metrics`,
   `generate_candidates`' inner loop, and the heuristic generator's scoring
   loop into a single source of truth?
4. The "Clipped-future-layers slice + opts-upgrade-from-Skip-to-MoveBlockers"
   pattern appears verbatim in `solve_entangling`, `solve_entangling_rh`,
   and partially in `solve_nohome`. Would
   `EntanglingOptions::clipped_future_layers(…)` +
   `SolveOptions::upgraded_for_entangling()` be the right shape before the
   refactor proceeds?
5. The "JSON round-trip from `PyArchSpec` to `LaneIndex`" pattern occurs
   twice in the Python wrapper. Should this crate expose
   `LaneIndex::from_arch_spec(&ArchSpec)` and
   `MoveSolver::from_arch_spec(&ArchSpec)` so the wrapper can drop the JSON
   detour entirely?

## 6. Proposed Refactor Direction

Direction agreed in conversation following this review. Not yet implemented;
sequencing is in §6.4.

### 6.1 Two-layer object model

Replace the single `MoveSolver` god struct with a two-layer composition:

```
                MoveSearch (strategy: entropy | astar | ids | …)
                          ▲
                          │  picked once, orthogonal to CZ strategy
                          │
       ┌──────────────────┴───────────────────────┐
       │             TargetSolver                  │
       │   solve(initial, target, blocked) → moves │   ← today's MoveSolver::solve
       └───────────────────────────────────────────┘
                          ▲
             uses         │         uses
       ┌──────────────────┴────────────┐
       │                               │
┌──────┴──────────────┐         ┌──────┴───────────────┐
│ SingleHeuristicCz   │         │ LooseGoalCz          │
│                     │         │                      │
│  target_gen.gen()   │         │  EntanglingGoal +    │
│   → fixed target;   │         │  LooseTargetGenerator│
│  TargetSolver.solve │         │  drive MoveSearch    │
│   ↳ delegates       │         │  with ctx.targets    │
│                     │         │  rewritten per node  │
└─────────────────────┘         └──────────────────────┘
```

`RecedingHorizonCz` is a third peer of `SingleHeuristicCz` / `LooseGoalCz`
(MPC-style outer loop on top of loose-goal). It is *not* a wrapper around
the others — kills the `receding_horizon → solve` layering inversion called
out in §4.

**Search-algorithm × CZ-strategy are orthogonal.** `MoveSearch` carries the
strategy choice (entropy / astar / ids / …); each CZ strategy reuses that
algorithm choice without re-expressing it.

**The loose-goal wrinkle:** `LooseGoalCz` does *not* delegate to
`TargetSolver`. Its search has a set-membership goal (`EntanglingConstraintGoal`)
and a per-node Hungarian-rewriting generator (`LooseTargetGenerator`), so it
drives `MoveSearch` directly. Only `SingleHeuristicCz` delegates to
`TargetSolver`. This is a feature: the user's mental model is "pick a search
algorithm; pick a CZ strategy" with both choices independent.

### 6.2 Wins this nets

Each item is something the §4 review flagged that the split fixes naturally
rather than requiring a separate refactor:

- **`MoveSolver` caches relocate.** `entangling_cache` and `nohome_cache`
  follow the loose-goal logic into `LooseGoalCz` (and the RH variant).
  `TargetSolver` becomes genuinely stateless across problems — exactly the
  testable unit the `temp_regression` suite already drives.
- **The `Option<cz_pairs>` mode switch in `HeuristicGenerator` dies.**
  `SingleHeuristicCz` constructs a plain `HeuristicGenerator` with no
  `cz_pairs`; `LooseGoalCz` constructs a `LooseTargetGenerator` (its own
  type). The three `cz_pairs.is_some()` branches in `generators/heuristic.rs`
  disappear because the two cases now run different *types*, not different
  *modes of one type*. This is the highest-value direct consequence.
- **`TargetGenerator` plugin trait gets a real home.** It's a constructor
  dependency of `SingleHeuristicCz` and a natural extension point.
  `LooseGoalCz` never sees a `TargetGenerator`. The "trait with no
  implementor" smell from §4 goes away.
- **`SolveOptions::upgraded_for_entangling` and `clipped_future_layers`
  helpers** become natural methods on `EntanglingOptions` / `SolveOptions`
  inside the CZ-strategy implementations — collapsing the verbatim
  duplication flagged as an emerging pattern.
- **`receding_horizon → solve` layering inversion vanishes.** `SolveResult`
  / `SolveStatus` / `SolveOptions` move out of `solve.rs` (which goes away
  as a god module) into the result types alongside `MoveSearch`. RH becomes
  a sibling, not a child, of loose-goal.

### 6.3 Python composition

Constructor injection, with named factory helpers for the common cases so
users who don't care about the layering don't have to know it exists:

```python
# 1. Pick a search algorithm — one knob, orthogonal to CZ strategy
search = MoveSearch.entropy(max_movesets_per_group=3, max_goal_candidates=3)
# or MoveSearch.astar(weight=1.0)
# or MoveSearch.cascade(inner="entropy")

# 2a. CZ strategy A: target-generation heuristic + a TargetSolver
cz = SingleHeuristicCzPlacement(
    target_solver=TargetSolver(search=search),
    target_generator=DefaultTargetGenerator(),   # or user plugin
)

# 2b. CZ strategy B: integrated loose-goal (no separate target step)
cz = LooseGoalCzPlacement(
    search=search,
    entangling_options=EntanglingOptions(occupancy_penalty=1.0, …),
)

# 2c. CZ strategy C: receding-horizon MPC over loose-goal
cz = RecedingHorizonCzPlacement(
    search=search,
    options=RecedingHorizonOptions(weight_grid=default_weight_grid(), …),
)

# 3. Use it — shared interface across all three CZ strategies
result = cz.solve(initial, controls, targets, blocked=…, max_expansions=…)
```

`TargetSolver` is publicly constructible (the `temp_regression` suite
already proves it's the natural testable unit). It's also a clean unit for
users who already know their target and want "just run the moves."

`MoveSearch` carries the search-strategy knob exclusively. **Do not** put a
`Strategy` knob on `LooseGoalCz` or `RecedingHorizonCz` — if you do, you
re-introduce the today problem where the same algorithm choice is expressed
two ways.

### 6.4 Observer unification (Option A) — agreed sequencing first

Before the type split, unify observability under `SearchObserver`:

- `entropy_search` changes signature from `Option<&mut EntropyTrace>` to
  `&mut dyn SearchObserver`. `EntropyTrace` becomes a `SearchObserver` impl
  that translates `SearchEvent`s back into the legacy `EntropyTraceStep`
  shape so the PyO3 getter surface stays bit-identical.
- The 5 dead `SearchEvent` variants (`Descend`, `EntropyBump`, `Revert`,
  `FallbackStart`, `FallbackStep`) get producers. They were declared in
  commit `855b424` anticipating entropy would route through the observer,
  then commit `227851d` overhauled entropy with its own trace plumbing and
  never came back.
- `SolveResult.entropy_trace` stays as-is for this PR. Python sees nothing
  different; the `temp_regression` suite is the safety net.
- Variants live centralized in `observer.rs` (not split per-driver), so a
  future cross-driver observer (e.g., a CSV logger covering both A* and
  entropy goal-found events) is reachable.

### 6.5 Refactor migration strategy

For each new Rust-visible type below `MoveSolver`'s old surface, the
sequence is:

1. Leave the existing `MoveSolver` API intact in Rust.
2. Add the new layered types (`MoveSearch`, `TargetSolver`, the three
   `CzPlacement` variants) alongside.
3. Migrate the Python wrapper to construct via the new types. The Python
   placement-strategy layer (`PhysicalPlacementStrategy`) absorbs the
   composition behind a single factory so its callers don't see the
   restructuring.
4. Once the Python side is stable on the new types, remove the old
   `MoveSolver` interface from the Python wrapper, then from Rust.

This discipline only applies to externally-visible Rust APIs (the ones the
PyO3 wrapper imports). The observer rewiring in §6.4 is purely internal —
no parallel API is needed; in-place rewiring with the `temp_regression`
suite as the safety net is sufficient.

### 6.6 Open questions deferred

- **Sequencing within the type split.** Once observer unification lands,
  is the next slice "extract `MoveSearch` + `TargetSolver` first, then the
  CZ strategies," or "extract one CZ strategy at a time (start with
  `SingleHeuristicCz` because it's the simplest)"?
- **`Strategy::Cascade { inner }` reshape.** §4 flagged folding into
  `refine_with_astar: bool`. Worth doing inside `MoveSearch`'s constructor
  before the CZ-strategy split, or after?
- **Python-facing observers.** Should `SearchObserver` be exposed to
  Python so users can write their own observers, or kept internal with
  `EntropyTrace` as the only consumer?
- **Where do `SolveResult` / `SolveStatus` ultimately live?** Top-level
  `lib.rs` re-exports, or a dedicated `result.rs` module sibling to
  `MoveSearch`?
