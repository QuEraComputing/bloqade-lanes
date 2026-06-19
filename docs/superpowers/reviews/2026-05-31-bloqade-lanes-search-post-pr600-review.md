# Crate Review: bloqade-lanes-search — post PR #600 (2026-05-31)

Companion to `2026-05-31-bloqade-lanes-search-review.md` (the pre-PR-600
review, retained as historical context). PR #600 landed earlier today
(`c25c91a feat: Move Policy & Target Generator DSL framework (#597) (#600)`)
and added an entire Starlark-hosted DSL substrate to this crate. This
review re-runs the rust-crate-review workflow against the merged tree to
re-baseline the architecture before refactoring proceeds.

## 1. Context

`bloqade-lanes-search` post-PR-600 is materially larger: 5 dependencies →
16 (added `bloqade-lanes-dsl-core`, `starlark 0.13`, `allocative`, `schemars`,
`similar`, `thiserror`, and proc-macro deps `syn` / `quote` / `proc-macro2`),
1 lib target → 1 lib + 1 bin (`policies-primer`) + 7 integration tests, and
1 consumer crate → 2 (`bloqade-lanes-bytecode-cli` now links the search
crate to host the `eval-policy` / `trace-policy` commands).

Despite the size jump, the structural relationship is unchanged at its
core: this crate still sits one hop above `bloqade-lanes-bytecode-core`
and is still consumed only by workspace siblings. The DSL is a **parallel**
solve path, not a replacement — `MoveSolver::solve` and `solve_with_policy`
coexist as independent entry points, sharing only the primitive data
structures (`Config`, `MoveSet`, `SearchGraph`, `LaneIndex`,
`DistanceTable`).

Change activity remains AI-heavy: 8 commits with `Co-Authored-By: Claude`
in the 90-day window, of which PR #600 (`c25c91a`) is by far the largest
single drop. Hotspot files are unchanged from the pre-PR-600 picture
(`entropy.rs`, `solve.rs`, `lib.rs`, `generators/heuristic.rs`,
`frontier.rs`, `heuristic.rs`, `lane_index.rs`, `aod_grid.rs`,
`config.rs`, `graph.rs`) plus the new DSL hotspots that earned their
status by size and import-count rather than commit history.

The `tests/temp_regression/` fixture suite landed in parallel — 60
randomized scenarios replayed bit-for-bit through `MoveSolver::solve` —
and was verified to still pass after the PR #600 rebase, confirming that
the search engine's observable behavior was preserved end-to-end.

## 2. External API Surface

### Public Type Inventory — Search Core (pre-PR-600, unchanged)

The pre-PR-600 inventory carried in `2026-05-31-bloqade-lanes-search-review.md` §2
is unchanged. Notable items that PR #600 did **not** touch:

- `MoveSolver`, `SolveResult`, `SolveStatus`, `SolveOptions`, `EntropyOptions`,
  `EntanglingOptions` — all still in `solve.rs`, still with the same
  inconsistent re-export pattern at `lib.rs:33-57` (the prior review
  flagged this; PR #600 did not change it).
- `Strategy`, `InnerStrategy`, `CandidateAttempt`, `MultiSolveResult`,
  `Config`, `ConfigError`, `MoveCandidate`, `SearchContext`, `SearchState`,
  `NodeId`, `SearchGraph`, `SearchResult`, `UniformCost`, `DeadlockPolicy`,
  the five generator structs, the three goal structs, the three heuristic
  structs, `LaneIndex`, observer trio, RH options, the two scorers, the
  target-generator trio, the five core traits — all unchanged.

### Public Type Inventory — DSL Surface (added by PR #600)

Roughly 30 new public types under three umbrella namespaces.

| Name | Kind | Defined In | Purpose |
|------|------|------------|---------|
| `dsl::fixture::Problem` | enum | `dsl/fixture.rs` | `Move(MoveProblem) / Target(TargetProblem)` discriminated fixture for CLI commands. |
| `dsl::fixture::MoveProblem` | struct | `dsl/fixture.rs` | Move-DSL problem with `arch`, `initial`/`target`/`blocked` triples, optional `Budget`, `policy_params: Value`. |
| `dsl::fixture::TargetProblem` | struct | `dsl/fixture.rs` | Target-DSL problem with `current_placement`, `controls`, `targets`, `lookahead_cz_layers`, `cz_stage_index`. |
| `dsl::fixture::Budget` | struct | `dsl/fixture.rs` | `max_expansions: u64`, `timeout_s: f64`. |
| `dsl::fixture::FixtureError` | enum | `dsl/fixture.rs` | `Io / Parse / SchemaVersion / ArchResolve` (thiserror). |
| `dsl::fixture::load`, `dsl::fixture::json_schema_pretty` | fns | `dsl/fixture.rs` | Fixture loader + schemars schema dump (the latter used only by the in-crate `policies-primer` binary). |
| `dsl::move_policy_dsl::solve_with_policy` | fn | `move_policy_dsl/kernel.rs` | Move-DSL kernel entry: `(initial, target, blocked, Arc<LaneIndex>, PolicyOptions, &mut dyn MoveKernelObserver) -> Result<PolicyResult, DslError>`. |
| `dsl::move_policy_dsl::PolicyOptions` | struct | `move_policy_dsl/kernel.rs` | `policy_path`, `policy_params: Value`, `max_expansions`, `timeout_s`, `sandbox: SandboxConfig`. |
| `dsl::move_policy_dsl::PolicyResult` | struct | `move_policy_dsl/kernel.rs` | `status`, `move_layers: Vec<MoveSet>`, `goal_config: Config`, `nodes_expanded`, `policy_file`, `policy_params`. |
| `dsl::move_policy_dsl::PolicyStatus` | enum | `move_policy_dsl/kernel.rs` | 11 variants: `Solved`, `Unsolvable`, `BudgetExhausted`, `Timeout`, `Fallback(String)`, `SyntaxError(String)`, `RuntimeError(String)`, `SchemaError(String)`, `BadPolicy(String)`, `StarlarkBudget`, `StarlarkOOM`. |
| `dsl::move_policy_dsl::MoveKernelObserver` (trait) + `NoOpMoveObserver` + `JsonMoveTraceObserver<W>` | trait + structs | `move_policy_dsl/observer.rs` | `on_init` / `on_step` / `on_builtin` / `on_halt`. |
| `dsl::move_policy_dsl::PolicyGraphSnapshot`, `GraphDelta` | structs | `move_policy_dsl/observer.rs` | Event payloads. |
| `dsl::move_policy_dsl::actions::MoveAction` | enum | `move_policy_dsl/actions.rs` | `InsertChild / UpdateNodeState / UpdateGlobalState / EmitSolution / Halt / InvokeBuiltin`. |
| `dsl::move_policy_dsl::actions::PatchValue` | struct | `move_policy_dsl/actions.rs` | `pub Value` patch payload (read-only by external consumers). |
| `dsl::move_policy_dsl::graph_handle::PolicyGraph`, `PolicyGraphInner`, `InsertOutcome`, `BuiltinOutcome`, `NodeStateMap` | structs | `move_policy_dsl/graph_handle.rs` | Starlark-visible read handle (Arc-Mutex wrapper) + kernel-private inner state. |
| `dsl::move_policy_dsl::lib_move::LibMove`, `Ctx`, `StarlarkConfig`, `StarlarkScoredLane`, `StarlarkPackedCandidate` | structs | `move_policy_dsl/lib_move.rs` | Starlark globals (`lib`, `ctx`) + wrappers for Config / `ScoredLane` / `PackedCandidate`. |
| `dsl::target_generator_dsl::TargetPolicyRunner` | struct | `target_generator_dsl/kernel.rs` | Parse-once, generate-many-stages runner: `LoadedPolicy` + `SandboxConfig`. |
| `dsl::target_generator_dsl::TargetPolicyError` | enum | `target_generator_dsl/kernel.rs` | `Dsl(DslError) / BadPolicy { reason } / ShapeError(String) / InvalidCandidate { error: CandidateError }`. |
| `dsl::target_generator_dsl::run_target_policy` | fn | `target_generator_dsl/kernel.rs` | One-shot helper used by CLI. |
| `dsl::target_generator_dsl::TargetKernelObserver` (trait) + `NoOpTargetObserver` + `JsonTargetTraceObserver<W>` | trait + structs | `target_generator_dsl/observer.rs` | `on_invoke` / `on_result`. |
| `dsl::target_generator_dsl::StarlarkTargetContext`, `StarlarkLibTarget` | structs | `target_generator_dsl/{ctx_handle, lib_target}.rs` | Starlark `ctx` and `lib` globals. |
| `dsl::pipeline` (module) | module | `dsl/pipeline.rs` | Declared `pub mod` but every item inside is `pub(crate)`. |

### Responsibility Portraits — DSL surface

**`solve_with_policy / PolicyOptions / PolicyResult / PolicyStatus`** — The
PyO3 and CLI consumers both want a one-call "run this `.star` file against
(initial, target, blocked) on this arch and tell me what happened." They
expect `PolicyOptions` to be a plain bag of knobs (file path, params blob,
two budgets, sandbox), `PolicyResult` to be the entire post-mortem
(terminal state, move-layer path, final placement, expansion count, echo
of the policy file/params for tracing), and `PolicyStatus` to be
exhaustively matchable so they can map each variant onto a CLI exit code
or a Python status string. Callers never construct a `PolicyGraph` or
assemble `MoveAction`s themselves — those exist purely for the Starlark
side.

**`TargetPolicyRunner / run_target_policy / TargetPolicyError`** —
Consumers expect `TargetPolicyRunner::from_path` (or `from_source` in
tests) to parse the `.star` file once and then permit `generate(...)` to
be called many times across CZ stages, returning the validated
`Vec<Vec<(u32, LocationAddr)>>` candidate list exactly as the policy
ordered it. `run_target_policy` is a one-shot convenience the CLI uses.
`TargetPolicyError::InvalidCandidate { error: CandidateError }`
deliberately re-exposes the pre-existing search-core `CandidateError` so
consumers get the same validation reasons whether candidates come from
`DefaultTargetGenerator` or a Starlark policy.

**`MoveKernelObserver / TargetKernelObserver`** — The CLI's
`trace-policy` consumes the observer trait twice: once with the
crate-provided `JsonMoveTraceObserver` (NDJSON to a writer) and once with
its own hand-rolled `HumanMoveTraceObserver` that implements the trait
directly. The crate is expected to ship both the schema (versioned NDJSON
envelopes) and the default no-op + JSON impls, while leaving the
human-readable formatter to the CLI. Snapshot/Delta structs
(`PolicyGraphSnapshot`, `GraphDelta`, `TargetContextSnapshot`,
`CandidateSummary`) are read-only data carriers — callers only read their
public fields, never construct them.

**`dsl::fixture::Problem / load`** — The CLI is the only consumer. It
expects a single `load(path)` call to return both the parsed enum and the
resolved-relative absolute arch JSON path so it can pass the arch JSON to
`ArchSpec::from_json` itself. The `Move` and `Target` variants split into
two CLI code paths; consumers expect `MoveProblem.budget` to be `Option`
and to override CLI flags only when the flag is unset.
`json_schema_pretty()` exists solely so the `policies-primer` binary can
paste a schema into the generated primer markdown.

**`StarlarkTargetContext / StarlarkLibTarget / Ctx / LibMove`** —
External Rust consumers never construct or touch these directly —
they're Starlark-side façades the kernel builds internally per solve.
They appear in the public surface only because `starlark_simple_value!`
requires the type to be named, public, and `ProvidesStaticType +
StarlarkValue`.

### API Friction Points (post-PR-600)

- `crates/bloqade-lanes-bytecode-python/src/policy_runner_python.rs:63-77`
  and `crates/bloqade-lanes-bytecode-cli/src/policy/eval.rs:211-225` —
  `PolicyStatus` → string is hand-rolled twice with identical labels.
  `PolicyStatus` could provide `as_label(&self) -> &'static str`. This
  mirrors the pre-existing `SolveStatus` → string duplication flagged in
  the prior review (`search_python.rs:194-198, 1542-1547`). Now four
  sites across two status enums.
- `crates/bloqade-lanes-bytecode-cli/src/policy/eval.rs:68-92` and
  `trace.rs:60-66, 80-90` — both CLI commands hand-decode the
  fixture's `[i32; 3]` location triples into `LocationAddr { zone_id,
  word_id, site_id }`. `loc_from_triple` in `trace.rs` is a verbatim
  duplicate of the closure in `eval.rs`. `dsl::fixture` could provide
  `MoveProblem::initial_locations()` / `target_locations()` /
  `blocked_locations()` returning `impl Iterator<Item=(u32,
  LocationAddr)>` (or `Vec<LocationAddr>` for blocked).
- `crates/bloqade-lanes-bytecode-python/src/policy_runner_python.rs:175-205`
  re-decodes every `MoveSet` into 6-tuples manually, duplicating similar
  logic in `search_python.rs`. A `MoveSet::to_lane_tuples() ->
  Vec<(u8,u8,u32,u32,u32,u32)>` (or simply making `LaneAddr` Py-extractable)
  would let both consumers share one decode path.
- `crates/bloqade-lanes-bytecode-python/src/target_generator_dsl_python.rs:31-37`
  constructs `LaneIndex::new(arch_spec.inner.clone())` while the runner
  internally takes `Arc<LaneIndex>` as a per-call *argument* to
  `generate(...)`, forcing every consumer to wire the index through
  itself. A `TargetPolicyRunner::with_arch_spec(...) -> Self` (storing
  its own `Arc<LaneIndex>`) would shrink both `generate` and
  `run_target_policy` argument lists.
- `crates/bloqade-lanes-bytecode-cli/src/policy/eval.rs:32` and
  `trace.rs:35` — both commands call `ArchSpec::from_json` separately
  *after* `fixture::load`, even though `load` already canonicalised the
  arch path. `fixture::load` could return either the parsed `ArchSpec`
  directly (one fewer error-mapping site) or a typed handle the CLI can
  hand to `LaneIndex::new`.
- **Carried over from prior review** (unchanged, now compounded by DSL):
  `MoveSolver`, `SolveResult`, `SolveStatus`, `EntropyOptions`,
  `EntanglingOptions` remain not re-exported from `lib.rs` while
  `SolveOptions`, `Strategy`, `InnerStrategy` are. The DSL added another
  path-asymmetry: `dsl::move_policy_dsl::*` and
  `dsl::target_generator_dsl::*` are reachable only via their full
  submodule paths.

### Dead Public Surface (added by PR #600)

- `dsl::move_policy_dsl::adapter_impl` — declared `pub mod` in `mod.rs:4`
  but the `adapter_impl.rs` file is a two-line stub with zero items.
  Pure scaffolding for the `dsl-core::Policy` trait impl that the kernel
  does not currently wire (per `adapter_impl.rs:1`: "Glue between
  `dsl-core::Policy` trait and the Move kernel" — but no glue exists).
- `dsl::pipeline` — module is declared `pub mod` in `dsl.rs:10` but every
  item inside (`ScoredLane`, `TripletGroup`, `PackedCandidate`,
  `group_by_triplet`, `pack_aod_rectangles`) is `pub(crate)`. Module
  visibility could safely drop to `pub(crate) mod`.
- `dsl::fixture::json_schema_pretty()` — only consumer is the in-crate
  `bin/policies-primer.rs`, not any external consumer.
- `dsl::move_policy_dsl::actions::PatchValue`, `register_actions` — only
  consumed inside `actions.rs` and `kernel.rs`. External consumers never
  inspect patch contents.
- `dsl::move_policy_dsl::graph_handle::{PolicyGraph, PolicyGraphInner,
  InsertOutcome, BuiltinOutcome, NodeStateMap}` — all consumed only by
  the kernel. Lock methods (`inner_borrow`, `inner_borrow_mut`,
  `inner_arc`) are kernel-private in spirit even though declared `pub`.
- `dsl::move_policy_dsl::lib_move::{StarlarkConfig, Ctx, LibMove,
  StarlarkMoveSet (re-export)}` and `dsl::target_generator_dsl::{
  StarlarkTargetContext, StarlarkLibTarget}` — Starlark-side façades
  only.
- The prior review's dead surface (`Frontier` family, `MisplacedHeuristic`,
  `HopDistanceHeuristic`, `SearchObserver` family,
  unused `entangling::*` free fns, etc.) is unchanged.

### MoveSolver ↔ PolicyRunner relationship

They share *primitive* types but not *result* types or any common entry
trait. Both build an `Arc<LaneIndex>` from an `ArchSpec`, both produce
`Vec<MoveSet>` move layers, and both terminate with budget-or-success
semantics. But:

- **No shared result envelope.** `MoveSolver::solve` returns
  `SolveResult { status: SolveStatus, ... }` with three variants; `solve_with_policy`
  returns `PolicyResult { status: PolicyStatus, ... }` with eleven. The
  Python sidecar `PyPolicySolveResult` re-maps `PolicyStatus` into the
  three-state vocabulary (`policy_runner_python.rs:112-120`) precisely to
  *look like* `PySolveResult` on the surface.
- **No shared trait abstraction.** `solve_with_policy` is a free
  function on top of an internal `PolicyGraphInner`, *not* a method on
  `MoveSolver`. There is no `trait Solver { fn solve(...) -> ??? }` that
  both inhabit.
- **Overlap is real but partial.** Both can in principle solve the same
  move synthesis problem, but `MoveSolver` orchestrates the *search
  algorithm* (`Strategy::AStar` / `Entropy` / `Cascade` / …), while
  `PolicyRunner` delegates *all* search-strategy decisions to Starlark
  code that issues `MoveAction::InsertChild` actions against a
  `SearchGraph`. The Starlark policy could conceptually emulate any
  `Strategy` variant, but no built-in policy ships with the crate
  (`adapter_impl.rs` is an empty stub).
- **Target side mirrors this:** `DefaultTargetGenerator` (pre-PR-600
  `TargetGenerator` impl) and `TargetPolicyRunner` (PR-600 DSL) are
  alternate implementations of "produce CZ-stage candidates." They share
  `validate_candidate` and `CandidateError` cleanly — the DSL kernel
  wraps the same validator — but no `trait` unifies them.
  `TargetPolicyRunner` does not implement `TargetGenerator` (signature
  shape differs: trait takes `&TargetContext<'_>`, runner takes seven
  separate owned arguments).

Net: the two subsystems live side-by-side. There is no leak, but there
is an obvious unification opportunity (`trait Solver`, shared
`SolveStatus`, `TargetPolicyRunner: TargetGenerator`) that PR #600 did
not take.

## 3. Internal Architecture

### Module Map — New DSL modules

| Module | Responsibility |
|--------|----------------|
| `dsl.rs` | Umbrella; declares `fixture`, `pipeline`, `move_policy_dsl`, `target_generator_dsl`. |
| `dsl/fixture.rs` | JSON problem-fixture loader (`Problem::{Move,Target}`, `Budget`, schema-version check, `json_schema_pretty()`). |
| `dsl/pipeline.rs` | Pure-Rust `f64`-scored AOD candidate pipeline. Parallel substitute for stages 2–4 of `HeuristicGenerator`. |
| `dsl/move_policy_dsl/mod.rs` | Kernel + observer re-exports. |
| `dsl/move_policy_dsl/actions.rs` | Six-verb action vocabulary + Starlark `register_actions` builders + `MoveAction::try_from_json`. |
| `dsl/move_policy_dsl/builtins.rs` | `sequential_fallback` (per-qubit BFS routing) — v1's only builtin. |
| `dsl/move_policy_dsl/graph_handle.rs` | `PolicyGraph` (Arc-Mutex Starlark handle) + `PolicyGraphInner` (kernel-owned state) + outcome records. |
| `dsl/move_policy_dsl/kernel.rs` | 1073 lines: `solve_with_policy` loop; globals binding; init/step invocation; JSON ⇄ Starlark marshalling; action application; terminal-result construction. |
| `dsl/move_policy_dsl/lib_move.rs` | 761 lines: 5 Starlark wrappers + `LibMove` methods (`hop_distance`, `mobility`, `legal_lanes`, `pack_aod_rectangles`, etc.). |
| `dsl/move_policy_dsl/observer.rs` | `MoveKernelObserver` trait + `NoOpMoveObserver` + `JsonMoveTraceObserver` + payloads. |
| `dsl/move_policy_dsl/adapter_impl.rs` | Two-line stub for the `dsl-core::Policy` trait impl. |
| `dsl/target_generator_dsl/mod.rs` | Re-exports. |
| `dsl/target_generator_dsl/ctx_handle.rs` | `StarlarkTargetContext`. |
| `dsl/target_generator_dsl/kernel.rs` | `TargetPolicyRunner` + `run_target_policy` + `TargetPolicyError`. |
| `dsl/target_generator_dsl/lib_target.rs` | `StarlarkLibTarget` (v1: `arch_spec` + `cz_partner(loc)`). |
| `dsl/target_generator_dsl/observer.rs` | Target-DSL observer trio. |
| `bin/policies-primer.rs` | Reflects on Starlark `MethodsBuilder` registrations via `include_str!` + `syn` to auto-generate `policies/primer.md`. |

### Module Map — Pre-PR-600 modules (only those PR #600 changed or that are DSL entry points)

| Module | Responsibility | Changed by PR #600? |
|--------|----------------|---------------------|
| `lib.rs` | Crate root, re-exports. | Yes — added `pub mod dsl;`. No DSL re-exports at root. |
| `graph.rs` | `SearchGraph` arena, `NodeId`, `MoveSet`. | Yes — `MoveSet` body removed, replaced with `pub use bloqade_lanes_dsl_core::primitives::move_set::MoveSet;`. `NodeId` got `serde::Serialize`. |
| `config.rs` | `Config` with cached FNV-1a hash. | Yes — added `pub fn cached_hash(&self) -> u64` to expose the field to `StarlarkConfig.hash`. |
| `target_generator.rs` | `validate_candidate`, `TargetContext`, `DefaultTargetGenerator`. | Unchanged code; new caller from `dsl/target_generator_dsl/kernel.rs`. |
| `entropy.rs`, `entangling.rs`, `lane_index.rs`, `aod_grid.rs`, `heuristic.rs`, `ordering.rs`, `solve.rs`, `frontier.rs`, `astar.rs`, … | Search engine. | Either unchanged or cosmetic clippy-only changes. `solve.rs` has zero `crate::dsl::*` references. |

### Cross-Boundary Edges — DSL ↔ existing search modules

```
DSL module               existing module           types/fns crossing               coupling
--------------------------------------------------------------------------------------------
pipeline                 aod_grid                  BusGridContext::new,             tight (core algorithm)
                                                    build_aod_grids
pipeline                 config                    Config, with_moves, iter         tight
pipeline                 graph                     MoveSet, from_encoded            loose
pipeline                 lane_index                LaneIndex, endpoints             loose
pipeline                 ordering                  cmp_moveset_config_tiebreak      loose

move_policy_dsl/actions  graph                     MoveSet, NodeId                  loose

move_policy_dsl/builtins config                    Config, with_moves               loose
                         graph                     MoveSet                          loose
                         lane_index                LaneIndex, outgoing_lanes,       loose
                                                    endpoints

move_policy_dsl/graph_handle config                Config (in StarlarkConfig wrap)  loose
                         graph                     MoveSet, NodeId, SearchGraph     tight (most of SearchGraph
                                                                                      read API surfaced)

move_policy_dsl/kernel   config                    Config                           tight
                         graph                     MoveSet, NodeId, SearchGraph     tight
                                                    (.insert, .depth, .root, etc.)
                         heuristic                 DistanceTable                    loose
                         lane_index                LaneIndex                        loose

move_policy_dsl/lib_move config                    Config, cached_hash (NEW)        tight
                         heuristic                 DistanceTable                    tight
                         lane_index                LaneIndex                        tight

target_generator_dsl/kernel lane_index             LaneIndex                        loose
                         target_generator          validate_candidate,              tight (reuses safety
                                                    CandidateError                    enforcement)
```

**No DSL module imports from `solve`, `frontier`, `astar`, `entropy`,
`entangling`, `scorers`, `generators`, `goals`, `receding_horizon`,
`nohome`, `context`, `cost`, `observer`, `heuristics`, `traits`.** The
DSL is a parallel kernel; the search-engine layer is untouched.

### Internal Coupling Hotspots

- `dsl/move_policy_dsl/kernel.rs` → 5 sibling DSL modules + 4 existing
  search modules + 4 dsl-core paths. ~1073 lines mixing per-solve flow,
  globals binding, init/step invocation, value↔JSON marshalling, action
  application, terminal-result construction. **Worst-coupled file in the
  crate by import count.**
- `dsl/move_policy_dsl/lib_move.rs` → 3 existing-search modules + 1 DSL
  sibling + 3 dsl-core paths. ~761 lines holding 5 distinct Starlark
  wrapper types (`StarlarkConfig`, `StarlarkScoredLane`,
  `StarlarkPackedCandidate`, `LibMove`, `Ctx`).
- `dsl/pipeline.rs` → 5 sibling search modules. Tight but unambiguous:
  deliberate `f64`-scored fork of `HeuristicGenerator`'s pipeline.
- Pre-existing hotspots (`solve.rs`, `entropy.rs`, `frontier.rs`,
  `generators/heuristic.rs`, etc.) are unchanged — see the prior review.

### Sidecar verification

PR #600 is **genuinely sidecar** to the existing search engine:

1. `Config` — added one additive accessor (`cached_hash()`); no
   struct/invariant change.
2. `MoveSet` — definition relocated to dsl-core; `crate::graph::MoveSet`
   preserved via `pub use`. All callers see the same import path.
3. `NodeId` — gained `#[derive(serde::Serialize)]` (additive).
4. `SearchGraph`, `LaneIndex`, `Goal`, `MoveGenerator`, `traits.rs` —
   unchanged.
5. `MoveSolver::solve` / `solve.rs` — **zero** `crate::dsl::*` references.
6. `target_generator::validate_candidate` reused as-is by the DSL.
7. `lib.rs` added only `pub mod dsl;`. No `pub use dsl::*` at the crate
   root.
8. `adapter_impl.rs` is a one-line placeholder — confirming the design
   seam in `dsl-core::Policy` is currently inert; the kernel speaks
   dsl-core's `LoadedPolicy` adapter directly without trait dispatch.

The coupling is one-directional and concentrated inside `src/dsl/`. The
hotspot risk lives **inside** the DSL subtree (`kernel.rs` at 1073 lines
+ 5 sibling imports) rather than at the DSL↔search boundary.

## 4. Critical Evaluation

### Contract vs Implementation Divergence

| Public Type | Classification | Explanation |
|-------------|----------------|-------------|
| `MoveSolver` / `SolveResult` / `SolveStatus` / `EntropyOptions` / `EntanglingOptions` | GAP (communication) — **unchanged from prior review** | Still reached via `solve::*` while siblings sit at crate root. |
| `SearchContext` / `EntropyScorer` / `MovesetMetrics::score` / `TargetGenerator` / `Expander` | GAP (drift) — **unchanged** | The `TargetGenerator` smell intensified: `TargetPolicyRunner` plays the same conceptual role but doesn't implement the trait. |
| `MoveSet` | MATCHES (with caveat) | Re-export from `dsl-core` is load-bearing; removing it would break `solve.rs`, `frontier.rs`, `entropy.rs`, every generator. |
| `PolicyStatus` (new) | GAP (drift) — **over-specified** | 11 variants declared; 3 explicitly "Reserved; not produced by v1". Consumers collapse to 3-state at the PyO3 boundary. |
| `PolicyResult` (new) | GAP (drift) vs `SolveResult` | Overlaps with `SolveResult` (`status`, `move_layers`, `goal_config`, `nodes_expanded`) but adds two echo fields (`policy_file`, `policy_params`) and drops four (`cost`, `deadlocks`, `entropy_trace`, attempts). No common abstraction. |
| `TargetPolicyRunner` (new) | GAP (drift) | Parallel to `MoveSolver` in shape (parse-once, run-many) but signature diverges from `TargetGenerator` trait. Takes `Arc<LaneIndex>` per call instead of owning it. |
| `Config::cached_hash()` (new) | MATCHES — but precedent risk | First "internal field accessor a DSL needs". Sets a pattern. |
| `NodeId` Serialize (new) | MATCHES | Additive only. |
| `pub mod pipeline` (new) | GAP (communication) | Declared `pub mod` but every item inside is `pub(crate)`. |
| `dsl::move_policy_dsl::adapter_impl` (new) | GAP (drift) | Declared `pub mod`, file is a two-line stub. |
| `dsl::fixture::json_schema_pretty()` (new) | GAP (drift) | Only consumer is the in-crate binary; should be `pub(crate)`. |

### Rust Health Findings

*Pre-PR-600 hotspots — unchanged from prior review:*

- `src/solve.rs:247-1499` — **warn**. The 6-instance `SolveResult { … }`
  literal pattern still stands verbatim. PR #600 did not touch `solve.rs`.
- `src/solve.rs:779-783, 1005-1009, 1166-1175` — **warn**.
  Clipped-future-layers + Skip→MoveBlockers upgrade survives in 3 places.
- `src/solve.rs:698-703, 860-882` — **warn**. `make_generator` closure
  for `HeuristicGenerator` survives.
- `src/entropy.rs`, `src/frontier.rs`, `src/generators/heuristic.rs`,
  `src/heuristic.rs`, `src/lane_index.rs`, `src/aod_grid.rs` — unchanged.
- `src/lib.rs:33-57` — unchanged inconsistent re-export inventory plus
  one new line `pub mod dsl;`. No DSL re-exports added.
- `src/config.rs::cached_hash()` — clean, 1-line accessor.
- `src/graph.rs:23` — **note**. Load-bearing `pub use` re-export from
  dsl-core. Comment at lines 19-22 explains the dependency-direction
  reasoning.

*New DSL hotspots:*

- `src/dsl/move_policy_dsl/kernel.rs:311, 481, 734, 803, 853, 858, 869, 922` — **warn**.
  Eight non-test `.expect("PolicyGraphInner mutex poisoned")` calls. Each
  is defensible (no contention on single-threaded evaluator), but
  panic-on-poisoning is the wrong fallback for a Starlark-runtime-driven
  kernel where a panic terminates the policy run without `RuntimeError`
  translation. `PolicyGraph::inner_borrow` / `inner_borrow_mut` helpers
  exist at `graph_handle.rs:102-110` but `kernel.rs` bypasses them.
- `src/dsl/move_policy_dsl/kernel.rs` 1073 lines — **warn**. Natural
  extraction candidate: the value↔JSON marshalling layer (lines 579-692,
  5 functions, ~110 lines) is self-contained.
- `src/dsl/move_policy_dsl/kernel.rs:496, 528, 651, 825` — **note**.
  `format!("{e:?}")` debug-format error mappings. Starlark `Debug` is not
  stable user-facing output; `Display` would be safer.
- `src/dsl/move_policy_dsl/lib_move.rs` 761 lines — **warn**. 5
  Starlark wrapper types in one file; a `wrappers/` submodule layout
  matching `target_generator_dsl/` would be more consistent.
- `src/dsl/move_policy_dsl/graph_handle.rs:102-120` — **note**.
  `inner_borrow` and `inner_borrow_mut` return identical
  `MutexGuard` — there's no `_mut` distinction at the type level
  (the underlying mutex always gives mutable access). Misleading naming.
- `src/dsl/move_policy_dsl/graph_handle.rs:8-11` — **note**. The
  `Arc<Mutex>` apparatus is paying runtime cost for a safety property
  ("single-threaded evaluator never contends") the architecture already
  guarantees. A `SendSyncCell`-style wrapper around `RefCell` would
  express the intent more precisely.
- `src/dsl/target_generator_dsl/kernel.rs:117` — **note**.
  `Arc::new(index.arch_spec().clone())` clones the full `ArchSpec` per
  `generate()` call. Wasteful if `LaneIndex::arch_spec()` already returns
  Arc-friendly form.
- `src/dsl/pipeline.rs` — **clean**. `decode_move_type` /
  `decode_direction` return `Option` rather than `unreachable!()` —
  better discipline than `entropy.rs:636` / `generators/heuristic.rs:545`.
- **No `unsafe` anywhere in the DSL subtree.**
- **No `todo!()` / `unimplemented!()` anywhere.** Several `//Reserved`
  comments (kernel.rs:139-142) instead of `todo!()` panics.

### Architectural Health Findings

- **Sidecar discipline is verifiably clean** (Agent 2 confirmed). The
  DSL subtree shares only the primitive-data layer with the search
  engine.
- **`cached_hash()` precedent risk.** Adding one accessor to satisfy a
  Starlark wrapper's `get_attr` is harmless in isolation, but it sets up
  an expectation that any field a DSL wants to read becomes a `pub fn`.
- **`MoveSolver` ↔ `PolicyRunner` duality is currently justified** by
  different state-machine semantics (search loop vs init/step), but the
  integration cost is real. Both share `LaneIndex`, `MoveSet`, `Config`,
  `SearchGraph`. The duplicated "status → string" and
  "MoveSet → tuples" patterns at consumer sites are the symptom. A
  shared `trait Solver` is not the right answer yet (the `Result` types
  disagree on too many fields), but a shared `SolverStatus` label
  abstraction is a 10-line win the consumers want now.
- **`bytecode-cli → bytecode-search` dependency is scoped, not leaky.**
  Only `policy/eval.rs` and `policy/trace.rs` consume the search crate,
  both through the DSL public surface. But the CLI builds the full
  search crate including the Starlark machinery; a `dsl-only` feature
  flag would let `MoveSolver`-free CLI tools opt out.
- **`kernel.rs` and `lib_move.rs` want sub-division** along orthogonal
  axes: kernel.rs along the value-marshalling boundary; lib_move.rs
  along the wrapper-type boundary. Both are AI-authored, single-commit,
  unlikely to have seen a refactoring pass.
- **`dsl/pipeline.rs` is intentional duplication of `HeuristicGenerator`'s
  pipeline** (Starlark needs `f64` scores, the in-crate code uses `i32`).
  But the `BTreeMap` triplet-grouping, the `BusGridContext` rectangle
  build, the AOD lane→lift→candidate flow, and the
  sort-by-score-desc-then-tiebreak comparator are structurally
  identical. A generic `pack_candidates_from_groups<S: Ord>(...)` over
  score type would consolidate.
- **§6 refactor proposal viability post-PR-600.** The prior review's
  proposed `MoveSearch` × `TargetSolver` × {`SingleHeuristicCz`,
  `LooseGoalCz`, `RecedingHorizonCz`} split has been made *harder in
  surface area* (4th `CzPlacement` peer + 2nd `TargetGenerator`-shaped
  thing) but *easier in structural alignment* (the DSL is cleanly
  sidecar; the refactor of the old code path can proceed without
  dragging DSL state through it). The DSL's `solve_with_policy` and
  `TargetPolicyRunner::generate` are well-positioned to become "yet
  another `CzPlacement` impl" and "yet another `TargetGenerator` impl"
  once the underlying traits exist.
- **The DSL did not pick up the `astar.rs` ⇄ `frontier.rs`
  bidirectional cycle**, nor the `receding_horizon → solve` layering
  inversion. Those pre-PR-600 smells are unchanged.

### AI-Drift Findings

**Commit `c25c91a` (PR #600):** the dominant signal. Multiple
AI-drift patterns visible:

- **Declared-but-unwired surface.** `adapter_impl.rs` (one-line stub),
  `dsl::pipeline` (`pub mod` with all-`pub(crate)` items),
  `dsl::fixture::json_schema_pretty()` (only in-crate binary uses it).
- **Over-specified `PolicyStatus` enum.** 11 variants of which 3 are
  explicitly "Reserved; not produced by v1" (kernel.rs:139-142). The
  variant set is documentation-of-intent, not actual contract.
- **`adapter_impl.rs` as scaffolding.** A `pub mod` declaration plus a
  single-line doc comment is a classic AI pattern: the structural slot
  is created in anticipation of the implementation, then deferred.
- **Stylistic incoherence.** Pre-PR-600 search code uses `pub(crate)`
  aggressively (`aod_grid` declared `pub(crate) mod`, `ordering`
  declared `pub(crate) mod`). The DSL submodules under
  `move_policy_dsl/` and `target_generator_dsl/` are nearly all
  `pub mod` — even when the only items inside are `pub(crate)`.
- **Re-implementation of in-crate patterns.** `pipeline.rs` re-implements
  stages 2–4 of `generators/heuristic.rs`. `builtins.rs::bfs_path` is
  essentially a copy of `entropy.rs::find_path_occupied` with a slightly
  different occupancy contract.
- **Marshalling duplication.** Two near-identical `json_to_starlark`
  functions in the same submodule (`kernel.rs:658-692` and
  `graph_handle.rs:280-308`). The author considered consolidating
  ("Used by `graph.ns()`" comment) and didn't.
- **`Mutex` for never-contended shared state.** `Arc<Mutex<PolicyGraphInner>>`
  exists because `starlark_simple_value!` requires `Send + Sync`. The
  module doc admits it never contends.
- **`format!("{e:?}")` debug-formatted error tracebacks** at four sites
  in `kernel.rs`.

**Carryover from prior review** (still applies): `EntropyScorer` still
unreachable, `TargetGenerator` still has one implementor, `EntropyTraceStep`'s
14 nullable fields still wear a tagged-union shape, `MisplacedHeuristic` /
`MaxHopHeuristic` / `SumHopHeuristic` still unused.

### ⚠ Emerging Patterns

⚠ Emerging Pattern: "Empty/Unsolvable SolveResult literal"
  Appears in: `src/solve.rs:247-255, 1051-1059, 1307-1315, 1346-1354, 1419-1427, 1487-1499`
  Similarity: Same 7-field literal with one of three status variants.
  Signal: 6 instances; carried forward unchanged from prior review.
  Suggested abstraction: `SolveResult::unsolvable` / `budget_exceeded` / `from_search` constructors.
  Readiness: ready to abstract — solve.rs untouched by PR #600; all six sites stable 90+ days.

⚠ Emerging Pattern: "Status enum → string label"
  Appears in: `search_python.rs:194-198, 1542-1547` (`SolveStatus`, pre-existing),
              `policy_runner_python.rs:63-77` (`PolicyStatus`, new),
              `cli/policy/eval.rs:211-225` (`PolicyStatus`, new).
  Similarity: Each consumer hand-rolls `match status { Variant => "label", ... }`.
  Signal: 3 sites across 2 status enums; PR #600 added 2.
  Suggested abstraction: Inherent `as_label(&self) -> &'static str` on each enum, or a `StatusLabel` trait.
  Readiness: still evolving — PolicyStatus instances landed today.

⚠ Emerging Pattern: "JSON ⇄ Starlark value marshalling"
  Appears in: `src/dsl/move_policy_dsl/kernel.rs:586-692`,
              `src/dsl/move_policy_dsl/graph_handle.rs:280-308`.
  Similarity: Recursive `J::Null/Bool/Number/String/Array/Object` dispatch; identical apart from the
              i32/i64 numeric mapping cases.
  Signal: 2 instances within the same submodule.
  Suggested abstraction: Consolidate into `dsl/move_policy_dsl/marshal.rs` (or push to dsl-core).
  Readiness: still evolving — both instances landed today.

⚠ Emerging Pattern: "Arc<LaneIndex> construction from ArchSpec / JSON"
  Appears in: `search_python.rs:573-577, 702-707` (pre-existing JSON round-trip),
              `target_generator_dsl_python.rs:31-37` (new — `LaneIndex::new(arch_spec.inner.clone())`).
  Similarity: Each consumer manually builds `Arc<LaneIndex>` from a Python-side arch spec.
  Signal: 3 instances; PR #600 added a third without adding the abstraction.
  Suggested abstraction: `LaneIndex::from_arch_spec(&ArchSpec)` + Arc-friendly variant.
  Readiness: ready to abstract — old sites stable; new site is the same shape.

⚠ Emerging Pattern: "[i32; 3] location-triple decode"
  Appears in: `cli/policy/eval.rs:68-92` (initial/target/blocked decode),
              `cli/policy/trace.rs:60-66, 80-90` (initial/blocked decode).
  Similarity: Both CLI commands hand-decode `Vec<(u32, [i32; 3])>` to `LocationAddr` by index-into-array.
  Signal: 2 instances; both landed in PR #600.
  Suggested abstraction: `MoveProblem::initial_locations() -> Vec<(u32, LocationAddr)>` (and siblings).
  Readiness: still evolving — both sites landed today.

⚠ Emerging Pattern: "MoveSet decode → 6-tuple unpacking"
  Appears in: `search_python.rs` (pre-existing solve-result MoveSet decode),
              `policy_runner_python.rs:175-205` (new — same decode shape).
  Similarity: `MoveSet::decode()` → `Vec<LaneAddr>` → `Vec<(u8, u8, u32, u32, u32, u32)>`.
  Signal: 2 instances; PR #600 added the duplicate.
  Suggested abstraction: `LaneAddr::to_tuple_repr()` helper, or a search-crate-side helper producing the Python-shaped list.
  Readiness: still evolving — new site landed today.

⚠ Emerging Pattern: "Parallel candidate pipeline (i32 vs f64 scores)"
  Appears in: `src/generators/heuristic.rs` (i32 scoring + group_by_triplet + pack + sort),
              `src/dsl/pipeline.rs` (f64 scoring + group_by_triplet + pack + sort).
  Similarity: Identical structural pipeline; only the score type and final-step `f64`-readiness differ.
  Signal: 2 instances; the second is brand-new and explicitly labeled "parallel implementation".
  Suggested abstraction: `pack_candidates_from_groups<S: Score>` over a score-type trait — OR concede the duplication if the i32→f64 gap is too wide.
  Readiness: still evolving — pipeline.rs landed today.

⚠ Emerging Pattern: "Scored-entry triplet sort with tiebreak" (carryover)
  Appears in: `src/entropy.rs:137-150, 152-163, 165-168`,
              `src/generators/heuristic.rs:46-59, 61-64`,
              `src/dsl/pipeline.rs:164-171` (new — sort_by score_sum desc then cmp_moveset_config_tiebreak).
  Signal: 6 instances now (was 5).
  Suggested abstraction: `ordering.rs` grows a generic `ScoreOrd` or `score_then_tiebreak` adapter.
  Readiness: ready to abstract — 5 original sites stable; new site landed today.

⚠ Emerging Pattern: "BFS path with occupancy" (carryover, new instance)
  Appears in: `src/entropy.rs::find_path_occupied` (pre-existing),
              `src/dsl/move_policy_dsl/builtins.rs:64-113` (new — bfs_path).
  Similarity: Both implement BFS over `outgoing_lanes`, skipping occupied destinations except the goal, reconstructing via parent-lane visitation map.
  Signal: 2 instances; the new one was re-implemented rather than reused.
  Suggested abstraction: Promote a shared `bfs_lane_path(start, goal, occupied, index)`.
  Readiness: still evolving — new instance landed today.

⚠ Emerging Pattern: "Clipped-future-layers + Skip→MoveBlockers upgrade" (carryover, unchanged)
  Appears in: `src/solve.rs:779-783 + 836-845, 1005-1009 + 1013-1022, 1166-1175`.
  Suggested abstraction: `EntanglingOptions::clipped_future_layers(...)` + `SolveOptions::upgraded_for_entangling()`.
  Readiness: ready to abstract — solve.rs untouched by PR #600.

## 5. Open Questions

### Contract Divergence

1. `PolicyStatus` carries 11 variants but consumers collapse them to
   3-state at the PyO3 boundary, and 3 of the 11 are explicitly
   documented as "Reserved; not produced by v1." Should the enum be
   trimmed to the producing variants now and grown later, or should the
   consumer-side collapse become an inherent `PolicyStatus::category(&self)
   -> StatusCategory` method?
2. `dsl::pipeline` is declared `pub mod` in `dsl.rs:10` but every item
   inside is `pub(crate)`. Should the module declaration be
   `pub(crate) mod pipeline;` to match its inhabitants, or should select
   items (e.g., `ScoredLane`, `PackedCandidate`) become `pub` for
   downstream tooling?
3. `TargetPolicyRunner::generate` takes `Arc<LaneIndex>` as a per-call
   argument even though the runner was constructed from a policy that was
   frozen against a specific arch. Should the runner own the arch
   (`with_arch_spec(...)` constructor) and the public API drop the
   per-call `index` parameter?
4. `MoveSet`'s canonical home is now
   `bloqade_lanes_dsl_core::primitives::move_set::MoveSet`, re-exported
   via `pub use` in `graph.rs:23`. Is this re-export load-bearing in a
   way that should be documented at the crate root (e.g., a `lib.rs`
   re-export directly from `dsl-core`), or is the indirection through
   `graph.rs` the right encapsulation?

### Rust Health

1. `src/dsl/move_policy_dsl/kernel.rs` has 8 non-test
   `.expect("PolicyGraphInner mutex poisoned")` calls. Should the kernel
   translate `PoisonError` into a `DslError::Runtime { traceback: "mutex
   poisoned" }` so a partially-failed evaluator doesn't take down the
   host process?
2. `src/dsl/move_policy_dsl/graph_handle.rs:102-110` defines
   `inner_borrow` and `inner_borrow_mut` returning identical `MutexGuard`
   types, but `kernel.rs` bypasses both and calls `inner_arc.lock()`
   directly at 8 sites. Should the kernel be retrofitted to use the
   helpers, or should the helpers be removed?
3. `src/dsl/target_generator_dsl/kernel.rs:117` clones the full
   `ArchSpec` on every `generate(...)` call. If `LaneIndex::arch_spec()`
   already returns the arch in `Arc`-friendly form, can the clone be
   deferred so per-CZ-stage cost is just an `Arc` bump?
4. `src/dsl/move_policy_dsl/kernel.rs:496, 528, 651, 825` use
   `format!("{e:?}")` to populate the traceback string for Starlark
   errors. Would `format!("{e}")` (Display) produce more stable,
   user-facing output?

### Architectural Health

1. `src/dsl/move_policy_dsl/kernel.rs` is 1073 lines spanning per-solve
   flow, globals binding, init/step invocation, JSON↔Starlark
   marshalling, action application, and terminal-result construction. Is
   the marshalling layer (lines 579–692) the right first extraction, and
   should it move to a `marshal.rs` sibling or to
   `bloqade_lanes_dsl_core::marshal`?
2. `src/dsl/move_policy_dsl/lib_move.rs` defines five distinct Starlark
   wrapper types in 761 lines. Should each wrapper move to its own file
   under `dsl/move_policy_dsl/wrappers/`, leaving `lib_move.rs` to hold
   only `LibMove` + its method registration?
3. `bloqade-lanes-bytecode-cli` now depends on the full search crate to
   access the DSL `eval` + `trace` commands. Would a `dsl` cargo feature
   on `bloqade-lanes-search` (gating the DSL subtree and the `dsl-core`
   dep) let CLI tools that don't need policy evaluation opt out?
4. The prior review's §6 refactor proposes splitting `MoveSolver` into
   `TargetSolver` + `CzPlacement` peers. PR #600 has added a 4th
   candidate peer (`solve_with_policy`) and a 2nd target-generator
   (`TargetPolicyRunner`) that does not implement `TargetGenerator`.
   Does the §6 split now need to grow a `Policy` peer alongside
   `SingleHeuristicCz` / `LooseGoalCz` / `RecedingHorizonCz`, or is
   `solve_with_policy` orthogonal enough to live in `dsl` permanently?

### AI-Drift

1. `c25c91a` declared `pub mod adapter_impl` in
   `dsl/move_policy_dsl/mod.rs:4` but `adapter_impl.rs` is a two-line
   stub. Is the slot intentional scaffolding for a future
   `dsl-core::Policy` trait impl, or vestigial?
2. `c25c91a` added two near-identical `json_to_starlark` functions: one
   in `kernel.rs:658-692`, one in `graph_handle.rs:280-308`. Was the
   duplication intentional (different `Heap` lifetime contracts) or
   should they consolidate into a `dsl/move_policy_dsl/marshal.rs`?
3. `c25c91a` introduced `bfs_path` in `builtins.rs:64-113` even though
   `entropy.rs::find_path_occupied` already implements
   occupancy-respecting BFS. Was the re-implementation deliberate (the
   occupancy contracts differ subtly) or should `builtins.rs` delegate?
4. `c25c91a` introduced `PolicyStatus` with 11 variants, of which
   `SchemaError`, `StarlarkBudget`, `StarlarkOOM` are explicitly
   documented as "Reserved; not produced by v1." Should the reserved
   variants land when their producers do (smaller blast radius for v1
   `PolicyResult` exhaustiveness checks), or is the up-front declaration
   intentional?
5. `c25c91a` added `Arc<Mutex<PolicyGraphInner>>` for `Send + Sync`
   compatibility with `starlark_simple_value!`, but the module doc
   states the mutex never contends. Is a no-op `SendSyncCell`-style
   wrapper around `RefCell` sufficient to drop the runtime atomics?

### Emerging Patterns

1. The "Status enum → string label" pattern has 3 instances after PR #600
   (`SolveStatus` × 2 sites + `PolicyStatus` × 2 sites). Should the
   abstraction be a shared `StatusLabel` trait, or inherent `as_label(&self)
   -> &'static str` methods on each enum?
2. "Arc<LaneIndex> construction from ArchSpec" has now reached 3 sites.
   The prior review proposed `LaneIndex::from_arch_spec(&ArchSpec)` —
   should this land before the next consumer adds a 4th site?
3. The "Parallel candidate pipeline" pattern (`generators/heuristic.rs`
   vs `dsl/pipeline.rs`) duplicates triplet-grouping, AOD packing, and
   sort-by-score-then-tiebreak. Should a generic
   `pack_candidates_from_groups<Score: Ord>` substrate land in
   `aod_grid.rs` or `pipeline.rs`, or is the i32↔f64 gap big enough to
   live with the duplication?
4. The "JSON ⇄ Starlark value marshalling" pattern has 2 instances in
   the same submodule after PR #600. Should a
   `dsl/move_policy_dsl/marshal.rs` consolidate them, or should the
   marshalling move to `dsl-core` so the target-generator DSL can share?
5. The "[i32; 3] location-triple decode" pattern has 2 CLI sites after
   PR #600 with identical shape. Should `dsl::fixture::MoveProblem` grow
   `initial_locations()` / `target_locations()` / `blocked_locations()`
   accessors that hand back `Vec<(u32, LocationAddr)>` /
   `Vec<LocationAddr>` directly?

## 6. Refactor Direction Update

The pre-PR-600 review proposed (in §6) a two-layer composition:
`MoveSearch` × `TargetSolver` × {`SingleHeuristicCz`, `LooseGoalCz`,
`RecedingHorizonCz`}. That direction still holds. PR #600 affects it as
follows:

- **The DSL has not pre-empted the split.** `solve_with_policy` and
  `TargetPolicyRunner` are sidecar, sharing only primitive data
  structures with the existing search engine. The §6 split can proceed
  against `MoveSolver` / `solve_entangling` / `solve_nohome` /
  `solve_entangling_rh` without dragging the DSL state through it.
- **Two new peer slots have appeared.** Once `trait CzPlacement` and
  `trait TargetGenerator` exist (the latter does already, just unused
  by the DSL), `solve_with_policy` becomes a natural fourth
  `CzPlacement` impl (`PolicyCzPlacement` over a Starlark policy) and
  `TargetPolicyRunner` becomes a natural second `TargetGenerator` impl
  with a `with_arch_spec(...)` constructor and an `impl TargetGenerator
  for TargetPolicyRunner` block. Neither requires DSL-side code changes
  beyond signature alignment.
- **Observer unification (Option A from prior review) is still the
  right "first refactor."** The pre-PR-600 observer plan extends
  naturally to the DSL: `MoveKernelObserver` and `TargetKernelObserver`
  duplicate the `SearchObserver` shape with different event vocabularies.
  Either fold both into `SearchObserver` (broader event enum) or accept
  three observer traits and document the split.
- **The Empty/Unsolvable SolveResult literal**, **Clipped-future-layers
  + Skip→MoveBlockers**, and **make_generator closure** patterns inside
  `solve.rs` are all unchanged by PR #600 and remain "ready to abstract"
  before the type split begins. Knocking them out first keeps the
  type-split diff small.
- **New "ready to abstract" wins introduced by PR #600 itself:**
  - "Status enum → string label" — 4-site pattern across two enums.
  - "Arc<LaneIndex> from ArchSpec" — 3-site pattern; one extra call site
    landed today.
  - "[i32; 3] location-triple decode" — 2-site CLI-only pattern; lives
    inside the new code so it can be cleaned up without touching the
    pre-PR-600 surface.
- **New "watch but don't act yet" items:**
  - `Config::cached_hash()` precedent — keep monitoring whether further
    field accessors get added to satisfy DSL Starlark wrappers.
  - `dsl/pipeline.rs` vs `generators/heuristic.rs` duplication — wait
    until the type split clarifies where the shared substrate lives.
  - `PolicyStatus` over-specification — let the v1 producers stabilize
    before trimming the enum.

### Sequencing recommendation (revised post-PR-600)

1. **Observer unification** (Option A, internal-only Rust rewiring) —
   unchanged from prior review §6.4. Land before the type split.
2. **Drive-by cleanups inside `solve.rs`** — `SolveResult` constructors,
   clipped-future helpers, `make_generator` factory. All "ready to
   abstract" per the prior review; PR #600 did not change any of them.
3. **Drive-by cleanups introduced by PR #600** — `LaneIndex::from_arch_spec`,
   `MoveProblem::*_locations()`, `Status::as_label()`. Small,
   independent, unblock multiple consumer sites.
4. **Type split** — `MoveSearch` / `TargetSolver` / `CzPlacement` peers.
   Once the trait surface exists, retrofit `solve_with_policy` as
   `PolicyCzPlacement` and `TargetPolicyRunner: TargetGenerator`. This
   step closes the "4 parallel solvers, no shared abstraction" smell PR
   #600 left behind.
5. **DSL hotspot sub-divisions** — `kernel.rs` marshalling extraction,
   `lib_move.rs` wrapper split. Independent of the search-side refactor;
   can land anytime after PR #600 stabilizes.
