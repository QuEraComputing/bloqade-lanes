# Move Policy DSL — Plan C Design

**Status:** Draft for review (2026-05-01).
**Branch:** `jason/move-policy-dsl` (worktree at `~/.config/superpowers/worktrees/bloqade-lanes/move-policy-dsl/`).
**Predecessors:** Plan A (Move Policy DSL framework, commit `b9d9383`), Plan B (Target Generator DSL, commit `11c664f`).
**Companion plan file:** `docs/superpowers/plans/2026-04-29-move-policy-dsl.md` — front-matter line 15 and Plan B's "Out of scope" section (lines 3027–3033) define what Plan C inherits.

## 1. Goal

Land the Plan C deliverables that make the Move Policy / Target Generator DSL framework approachable and exercised:

1. **Reference Move policies** — `dfs.star`, `bfs.star`, `ids.star`, each a small educational demonstration of one frontier-management strategy.
2. **CLI harness** — `eval-policy` and `trace-policy` subcommands on `bloqade-bytecode`, supporting both Move and Target policies via a `"kind"` discriminator inside a single problem-fixture JSON file.
3. **Auto-generated `policies/primer.md`** — produced by a `policies-primer` binary that reflects on registered Starlark globals, preserves hand-written prose blocks via sentinel comments, and is CI-checked for staleness.
4. **Snapshot fixtures** — committed `policies/fixtures/{move,target}/{small,medium,large}/...` corpora driven by a Rust integration test that does structural-match comparison on key fields.

The four pieces share one substrate: they make the DSL usable without a notebook in the loop, and they protect against regressions in `Strategy::Entropy` parity and the reference policies once the work lands on `main`.

## 2. Non-goals

- Performance benchmarking. Wall-time is captured but not compared. Performance regression detection is out of scope; the existing `tests/` benchmarking harness covers that pipeline.
- Live-reload / watch-mode for the CLI. One-shot invocations only.
- Policy authoring helpers (Starlark linters, formatters, language servers). The primer is the only authoring aid.
- New `lib_target.*` primitives. Plan B's "deferred to Plan C or later" line 3032 explicitly punts these past Plan C.
- Cross-platform path normalisation tests for fixture files. The fixture loader resolves `arch` paths relative to the fixture file before any serialised expected output is produced; no host-specific path ever lands in committed expected files.

## 3. Architecture overview

Plan C touches three crates and adds two filesystem trees outside the source code:

| Surface | Crate / dir | Change |
|---|---|---|
| Observer trait + impls | `bloqade-lanes-search` | New sibling modules `move_policy_dsl/observer.rs` and `target_generator_dsl/observer.rs`. |
| Kernel hookup | `bloqade-lanes-search` | `solve_with_policy` (and the Target equivalent) gain `&mut dyn KernelObserver` parameters. Existing call sites pass `&mut NoOpMoveObserver` / `&mut NoOpTargetObserver`. |
| CLI subcommands | `bloqade-lanes-bytecode-cli` | New `eval-policy` and `trace-policy` subcommands; a new `src/policy/` module for fixture loading, observers, and output formatting. |
| Primer generator | `bloqade-lanes-search` | New `src/bin/policies-primer.rs` that reflects on registered Starlark globals and emits `policies/primer.md`. |
| Reference policies | `policies/reference/` | New `dfs.star`, `bfs.star`, `ids.star` (Move side; no new Target reference). |
| Fixture corpus | `policies/fixtures/` | New `move/{small,medium,large}/` and `target/small/`, each with `problem.json` and one or more `expected.<policy>.json`. |
| Tests | `bloqade-lanes-search` | New `tests/dsl_snapshot.rs` walking the corpus. |
| CLI smoke tests | `bloqade-lanes-bytecode-cli` | New `assert_cmd`-based integration tests for the new subcommands. |
| Build automation | `justfile` + `.github/workflows/ci.yml` | New `generate-primer`, `check-primer`, `regenerate-fixtures` recipes; CI runs `check-primer` alongside the existing `check-header`. |

### 3.1 Crate dependency graph

```
bloqade-lanes-bytecode-cli  →  bloqade-lanes-search          (NEW edge)
                            →  bloqade-lanes-dsl-core        (transitive)
                            →  bloqade-lanes-bytecode-core   (existing)
```

No reverse edges introduced. The CLI binary's release-mode size grows by what `starlark-rust` adds (~4–6 MB stripped). The C FFI library is unaffected — it does not link the DSL code paths.

### 3.2 Why a sibling module for the observer

`kernel.rs` already covers init/step/apply/budget/status mapping/evaluator setup. The observer adds ~150–300 lines (trait, `NoOpMoveObserver`, `JsonMoveTraceObserver`, per-record types and their `Serialize` impls). Sibling-module placement keeps `kernel.rs` focused on the run loop, co-locates the observer impls for unit testing, gives the public trait a discoverable home for re-export from `mod.rs`, and matches the existing one-purpose module style (`actions.rs`, `graph_handle.rs`, `lib_move.rs`, `builtins.rs`, `kernel.rs`, `adapter_impl.rs`).

The observer's *call sites* (`observer.on_step(...)`, etc.) remain in `kernel.rs` — only the trait definition and impls move to the sibling.

## 4. CLI surface

### 4.1 Subcommands

```
bloqade-bytecode eval-policy [OPTIONS] --policy <PATH> --problem <PATH>
bloqade-bytecode trace-policy [OPTIONS] --policy <PATH> --problem <PATH>
```

`eval-policy` runs the policy end-to-end and emits a single result summary. `trace-policy` runs the policy with a verbose observer installed and emits one record per kernel event.

### 4.2 Common flags

| Flag | Purpose |
|---|---|
| `--policy <PATH>` | `.star` file. Required. |
| `--problem <PATH>` | Problem fixture JSON. Required. The top-level `"kind"` field selects move vs. target dispatch. |
| `--params <PATH>` | Optional JSON file bound as `PARAMS_OVERRIDE` at policy load time. Mirrors the existing `policy_params` Python kwarg. |
| `--max-expansions <N>` | Override the fixture's budget. |
| `--timeout <SECS>` | Override the fixture's timeout. |
| `--json` | Switch from human format to structured output. |
| `--seed <N>` | Optional; passed through to the kernel for any future RNG-using primitives. Has no effect on v1 policies. |

### 4.3 `trace-policy`-only flags

| Flag | Purpose |
|---|---|
| `--out <PATH>` | Write trace records to file instead of stdout. NDJSON in `--json` mode, pretty in default. |

### 4.4 Exit codes

| Code | Meaning |
|---|---|
| `0` | Policy halted with `Halt`; the run is well-formed. |
| `1` | Runtime error: fixture parse failure, policy load/runtime error, IO error. |
| `2` | Policy completed but with a non-`Halt` status (`Fallback`, `BudgetExhausted`, `Timeout`). Distinguishes "the policy ran and chose to give up" from "the harness exploded." |

The 0/1/2 split lets snapshot tests use the exit code as a first-line filter before reading the full output.

### 4.5 `eval-policy` output

**Default (human):** A short table to stdout —

```
status         Halt
halt_reason    found_target
expansions     142
max_depth      7
wall_time      18.4 ms
```

**`--json`:** A single JSON object to stdout —

```json
{
  "v": 1,
  "kind": "move",
  "policy": "policies/reference/dfs.star",
  "problem": "policies/fixtures/move/small/problem.json",
  "status": "Solved",
  "halt_reason": "found_target",
  "expansions": 142,
  "max_depth": 7,
  "wall_time_ms": 18.4
}
```

`v` is the schema version. Snapshot fixture comparison ignores `wall_time_ms`, `policy`, and `problem` (they are verified separately by exists-checks).

### 4.6 `trace-policy` output

**Default (human):** One line per kernel event, formatted for readability.

**`--json`:** Newline-delimited JSON (NDJSON), one record per line, no trailing record:

```jsonl
{"v":1,"kind":"init","root":{...}}
{"v":1,"kind":"step","step":0,"depth":1,"action":{"type":"InsertChild","args":{...}},"delta":{...}}
{"v":1,"kind":"builtin","step":0,"name":"compute_score","ok":true}
{"v":1,"kind":"halt","status":{"type":"Solved"}}
```

The record schema is versioned by `v`, sharing the same version space as the `eval-policy --json` envelope.

## 5. Observer trait

### 5.1 Move side

```rust
// crates/bloqade-lanes-search/src/move_policy_dsl/observer.rs
pub trait MoveKernelObserver {
    fn on_init(&mut self, root: &PolicyGraphSnapshot) {}
    fn on_step(&mut self, step: u64, depth: u32, action: &MoveAction, delta: &GraphDelta) {}
    fn on_builtin(&mut self, step: u64, name: &str, ok: bool) {}
    fn on_halt(&mut self, status: &PolicyStatus) {}
}

pub struct NoOpMoveObserver;
impl MoveKernelObserver for NoOpMoveObserver {}

pub struct JsonMoveTraceObserver<W: io::Write> { /* writer + flush bookkeeping */ }
impl<W: io::Write> MoveKernelObserver for JsonMoveTraceObserver<W> { /* emits NDJSON */ }
```

`PolicyGraphSnapshot` and `GraphDelta` are small data-only types defined in `observer.rs`. They are *not* the live `PolicyGraph` Starlark value: the kernel materialises a snapshot once at init and a delta per step. `GraphDelta` carries the same `last_insert` / `last_builtin` / `node_state` writes the kernel already tracks for Starlark.

`depth` is the search-tree depth of the just-applied state, useful for IDS-style policies and general debugging.

### 5.2 Target side

```rust
// crates/bloqade-lanes-search/src/target_generator_dsl/observer.rs
pub trait TargetKernelObserver {
    fn on_invoke(&mut self, stage_idx: u64, ctx: &TargetContextSnapshot) {}
    fn on_result(&mut self, stage_idx: u64, summary: &CandidateSummary, ok: bool) {}
}

/// Snapshot of `TargetPolicyRunner::generate`'s `Vec<Vec<(u32, LocationAddr)>>`
/// result: the count of candidate placements and the size of the first candidate
/// (the latter is `0` when no candidates were produced).
pub struct CandidateSummary {
    pub num_candidates: usize,
    pub first_candidate_size: usize,
}

pub struct NoOpTargetObserver;
impl TargetKernelObserver for NoOpTargetObserver {}

pub struct JsonTargetTraceObserver<W: io::Write> { /* writer + flush bookkeeping */ }
impl<W: io::Write> TargetKernelObserver for JsonTargetTraceObserver<W> { /* emits NDJSON */ }
```

Target generation is a single-call-per-stage operation, so the observer just emits one `invoke` and one `result` record per CZ stage; there is no per-step loop.

The renamed types (`NoOpMoveObserver`, `NoOpTargetObserver`, `JsonMoveTraceObserver`, `JsonTargetTraceObserver`) avoid a collision with the existing `crate::observer::NoOpObserver` (used by the A*/entropy `SearchObserver` infrastructure) at the search-crate root. Neither set is re-exported from `lib.rs` at the unqualified name; consumers reach them via the submodule path.

### 5.3 Extensibility

The observer trait surface is intentionally minimal at v1. Adding `frontier_size`, `current_cost`, or any other field is a non-breaking change (default-implemented method + default-generated NDJSON field). The schema version (`v`) is bumped only when a field is *removed* or its semantics change.

## 6. Problem fixture schema

The schema is documented at `docs/superpowers/specs/move-policy-dsl-problem-schema.md` (one short page, hand-written, normative reference for the primer). Both kinds share a `"v"` and `"kind"` envelope.

### 6.1 `kind: "move"`

```json
{
  "v": 1,
  "kind": "move",
  "arch": "examples/arch/gemini.json",
  "initial": [[0, [1, 0, 0]], [1, [1, 0, 1]], [2, [1, 1, 0]]],
  "target":  [[0, [1, 1, 0]], [1, [1, 0, 0]], [2, [1, 0, 1]]],
  "blocked": [],
  "budget": { "max_expansions": 5000, "timeout_s": 10.0 },
  "policy_params": {}
}
```

- `arch` — path (relative to the fixture file) to an ArchSpec JSON. Allows fixture reuse across multiple sizes that share an arch. Inline-arch via `"arch_inline": {...}` is allowed but not the default.
- `initial` / `target` — `[[qubit_id, [layer, row, col]], ...]` matching `LocationAddr`'s three-tuple encoding.
- `blocked` — list of `[layer, row, col]` triples.
- `budget` — optional; CLI flags override. The loader supplies defaults if absent.
- `policy_params` — optional dict; the CLI's `--params` flag overrides per-key.

### 6.2 `kind: "target"`

```json
{
  "v": 1,
  "kind": "target",
  "arch": "examples/arch/gemini.json",
  "current_placement": [[0, [1, 0, 0]], [1, [1, 0, 1]]],
  "controls": [0],
  "targets":  [1],
  "lookahead_cz_layers": [],
  "cz_stage_index": 0,
  "policy_params": {}
}
```

The schema mirrors `TargetPolicyRunner::generate` exactly: `current_placement` (Move's `initial` shape), `controls` and `targets` as parallel qubit-id lists for the current CZ stage, `lookahead_cz_layers` as a list of `[controls, targets]` tuples for upcoming stages (default empty), and `cz_stage_index` (default `0`). `controls` and `targets` must have the same length.

### 6.3 Loader

The fixture loader lives at `crates/bloqade-lanes-search/src/fixture.rs` (a new public module on the `search` crate). It is shared by the CLI subcommands and by the snapshot integration test, avoiding a reverse dep edge from `search` → CLI. The CLI crate re-exports the loader for ergonomics but does not own it. The loader parses fixture files into a typed enum:

```rust
#[derive(serde::Deserialize)]
#[serde(tag = "kind")]
pub enum Problem {
    #[serde(rename = "move")]
    Move(MoveProblem),
    #[serde(rename = "target")]
    Target(TargetProblem),
}
```

Failure modes — unknown kind, missing fields, schema-version mismatch, malformed triples, unresolvable `arch` path — all map to a single `FixtureError` returned to the CLI as exit code `1`.

## 7. Reference policies

Three new files under `policies/reference/`. All Move side; all educational; all under ~120 lines including comments. Each starts with a sentinel-bracketed heredoc that the primer generator extracts for the "tour of reference policies" section.

| File | Demonstrates | Surface used | Approx body |
|---|---|---|---|
| `dfs.star` | Depth-first frontier management via `graph.last_insert`. Recurse on the most recently inserted child until none extend depth, then unwind. | `actions.insert_child`, `actions.halt`, `lib_move.expand_candidates`, `lib_move.is_target` | ~80 lines |
| `bfs.star` | Breadth-first via per-depth queue stored in `update_global_state`. Round-robins across siblings. | `actions.insert_child`, `actions.halt`, `update_global_state`, `lib_move.expand_candidates` | ~90 lines |
| `ids.star` | Iterative deepening: wraps DFS with a depth cap; on cap-reached, increments cap and restarts via `actions.reset_to_root`. | DFS surface + `actions.reset_to_root` + global state for the cap | ~110 lines |

The educational policies are paired with the *small* fixture only in the snapshot corpus; *medium* and *large* track only `entropy.star`. Rationale: the educational policies are not meant to be efficient enough to solve medium/large; budgeting them on small is enough to detect drift.

No new Target-side reference policy is shipped. `default_target.star` from Plan B remains the sole Target reference. If a future contributor wants `simple_target.star` we add it then; Plan C ships only the docs/CLI/fixtures for the Target side.

## 8. Primer generator

### 8.1 Binary

`crates/bloqade-lanes-search/src/bin/policies-primer.rs`. Lives in the `search` crate so it has direct module access to `move_policy_dsl::*` and `target_generator_dsl::*` registrations.

### 8.2 Invocation

```
cargo run -p bloqade-lanes-search --bin policies-primer            # writes policies/primer.md
cargo run -p bloqade-lanes-search --bin policies-primer -- --check # exits non-zero if file is stale
```

Wired up via `just generate-primer` and `just check-primer`. CI runs `check-primer` alongside the existing `check-header` step.

### 8.3 File structure

`policies/primer.md` is a single file with two kinds of blocks:

- **Sentinel-bracketed AUTOGEN sections** (`<!-- BEGIN AUTOGEN: name --> ... <!-- END AUTOGEN: name -->`) regenerated on every run.
- **Sentinel-bracketed PROSE sections** (`<!-- BEGIN PROSE: name --> ... <!-- END PROSE: name -->`) preserved across regeneration.

Layout:

```markdown
<!-- AUTOGEN: DO NOT EDIT BY HAND.
     Regenerate with `just generate-primer`. -->

# Move Policy & Target Generator DSL — Primer

<!-- BEGIN PROSE: intro -->
[hand-written intro paragraph]
<!-- END PROSE: intro -->

## Move Policy surface

<!-- BEGIN AUTOGEN: actions -->
### `actions.*` — kernel-driven verbs
[per-method: signature + doc string]
<!-- END AUTOGEN: actions -->

<!-- BEGIN AUTOGEN: lib_move -->
### `lib_move.*` — query primitives
<!-- END AUTOGEN: lib_move -->

<!-- BEGIN AUTOGEN: graph_handle -->
### `graph.*` — read-only graph accessors
<!-- END AUTOGEN: graph_handle -->

<!-- BEGIN PROSE: move-tour -->
[hand-written tour referencing dfs.star/bfs.star/ids.star/entropy.star]
<!-- END PROSE: move-tour -->

## Target Generator surface

<!-- BEGIN AUTOGEN: lib_target -->
<!-- END AUTOGEN: lib_target -->

<!-- BEGIN PROSE: target-tour -->
[hand-written tour referencing default_target.star]
<!-- END PROSE: target-tour -->

## Problem fixture schema

<!-- BEGIN AUTOGEN: schema -->
[generated from MoveProblem / TargetProblem via schemars]
<!-- END AUTOGEN: schema -->
```

### 8.4 Generation algorithm

1. **Read the existing `policies/primer.md`** if present and parse out the `BEGIN PROSE`/`END PROSE` blocks. If the file is missing, fall back to a stubbed prose block (`"TODO: write prose for <name>"`) so a fresh worktree generates a complete-but-incomplete file rather than failing.
2. **Re-emit AUTOGEN sections** by reflecting on the registered Starlark globals:
   - `actions::register_actions(&mut GlobalsBuilder)` — iterate the `MethodsBuilder` to extract method names, signatures, and `#[doc = "..."]` attributes attached at the registration site.
   - Same for `lib_move::register_lib_move`, `lib_target::register_lib_target`, `graph_handle::register_graph_methods`.
   - Each registration site is updated *as part of this plan* to attach a primer-friendly summary doc-comment.
   - **How source is read:** the generator binary uses `include_str!` at its own compile time to embed the registration files (e.g. `include_str!("../move_policy_dsl/actions.rs")`), then parses them via `syn`/`proc_macro2`. No proc-macro at runtime, no path resolution at runtime, no dependency on `cargo`'s manifest layout. The set of embedded files is small and stable (one per `register_*` site).
3. **Schema section** — generated via `schemars::schema_for!(MoveProblem)` / `schemars::schema_for!(TargetProblem)` rendered as markdown. `schemars` is added as a dev-dep on `bloqade-lanes-bytecode-cli`.
4. **Write or check.** Default mode renders the assembled markdown and writes it to `policies/primer.md`. `--check` mode renders to a buffer, compares byte-for-byte to the existing file, and exits `1` with a unified diff if they differ.

### 8.5 Non-goals for the generator

- It does not run the Starlark evaluator. Reflection is purely on registered builders + source-doc-comments.
- It does not catalog algorithmic complexity or worked examples. Those live in PROSE blocks.
- It does not document the Python-side API. That's a separate `mdBook` surface, already covered by `just doc`.

## 9. Snapshot fixtures

### 9.1 Layout

```
policies/fixtures/
├── README.md
├── move/
│   ├── small/
│   │   ├── problem.json
│   │   ├── expected.entropy.json
│   │   ├── expected.dfs.json
│   │   ├── expected.bfs.json
│   │   └── expected.ids.json
│   ├── medium/
│   │   ├── problem.json
│   │   └── expected.entropy.json
│   └── large/
│       ├── problem.json
│       └── expected.entropy.json
└── target/
    └── small/
        ├── problem.json
        └── expected.default_target.json
```

### 9.2 Sizes

- `small` — ≈ 4×4 grid, ~6 qubits.
- `medium` — ≈ 10×10 grid, ~16 qubits.
- `large` — ≈ 20×20 grid, ~40 qubits.

Pinned in this spec; final tuning during implementation against a 30 s CI budget for the snapshot suite.

### 9.3 Comparison fields

Move expected files compare on `{status, halt_reason, expansions, max_depth}`. Target expected files compare on `{ok, num_candidates, first_candidate_size}` (where `num_candidates` is the length of the outer `Vec<Vec<(u32, LocationAddr)>>` returned by `TargetPolicyRunner::generate`, and `first_candidate_size` is the length of the inner Vec for the first candidate, or `0` if the result list is empty). Excluded everywhere: `wall_time_ms`, `policy` and `problem` paths (verified separately as exists-checks), any future field starting with `_` (reserved for non-stable data).

### 9.4 Regeneration workflow

A `just regenerate-fixtures` recipe runs the CLI against every committed `problem.json` and rewrites the matching `expected.*.json`. **Not run in CI** — deliberate human action when a baseline shifts. The `policies/fixtures/README.md` documents the dance: regenerate, eyeball the diff, commit only if intentional.

### 9.5 Driver

`crates/bloqade-lanes-search/tests/dsl_snapshot.rs` walks the fixture tree at runtime via `std::fs::read_dir` on a `CARGO_MANIFEST_DIR`-relative path. For each `(problem.json, expected.<name>.json)` pair:

1. Load the problem via `bloqade_lanes_search::fixture` (the shared public loader from §6.3).
2. Resolve the policy path from the `expected.<name>.json` filename → `policies/reference/<name>.star`.
3. Run `solve_with_policy` (or the Target equivalent) with `NoOpMoveObserver` / `NoOpTargetObserver`.
4. Build the structural-match record from `PolicyResult` and compare to the parsed expected file.
5. On mismatch: emit a side-by-side diff and fail with a hint to run `just regenerate-fixtures`.

The driver lives in the search crate (not the CLI crate) because the DSL surface lives in `search`. The CLI gets its own smaller `assert_cmd`-based smoke test for the binary plumbing.

## 10. Testing strategy

| Test | Crate | Purpose |
|---|---|---|
| `fixture` unit tests | `bloqade-lanes-search` | Schema discrimination (kind=move/target/unknown), arch-path resolution, budget-default behaviour, malformed-input rejection. |
| `JsonTraceObserver` unit tests | `bloqade-lanes-search` | Record schema versioning, record ordering (init→step*→halt), NDJSON line discipline. |
| `policies-primer` golden-output test | `bloqade-lanes-search` | Run the generator against a committed fixture-input directory, compare to a committed fixture-output `primer.md`. Catches regressions in the AUTOGEN logic without making the live `policies/primer.md` the test oracle. |
| CLI integration tests (`assert_cmd`) | `bloqade-lanes-bytecode-cli` | Smoke tests for `eval-policy` and `trace-policy` against the small fixture; verify exit codes, JSON shape, NDJSON validity. |
| `dsl_snapshot.rs` | `bloqade-lanes-search` | The fixture-corpus regression suite (see §9.5). |

### 10.1 What's not tested in Plan C

- Live `cargo run` of the CLI from snapshot tests — the snapshot driver calls `solve_with_policy` directly. The CLI integration tests cover the CLI plumbing separately.
- Performance regression bounds. Wall-time is excluded from comparison on purpose.

## 11. Error handling

CLI-side error mapping:

| Error class | Source | Exit code | Human output |
|---|---|---|---|
| Fixture parse failure | `serde_json` | `1` | `error: <path>: <serde diagnostic>` to stderr |
| Policy load failure | `DslError::SyntaxError`, `BadPolicy` | `1` | Starlark parser diagnostic to stderr |
| Policy runtime error | `DslError::RuntimeError` | `1` | Starlark traceback to stderr |
| Budget / timeout | `PolicyStatus::BudgetExhausted`, `Timeout` | `2` | One-line summary to stderr; full record on stdout per output mode |
| Policy fallback | `PolicyStatus::Fallback(reason)` | `2` | Reason to stderr; full record on stdout |
| Halt (success) | `PolicyStatus::Solved` | `0` | Summary on stdout |
| IO failure (`--out` write) | `io::Error` | `1` | `error: writing <path>: <io diagnostic>` to stderr |

## 12. File-by-file additions

| File | Action | Owner crate |
|---|---|---|
| `crates/bloqade-lanes-search/src/move_policy_dsl/observer.rs` | new | search |
| `crates/bloqade-lanes-search/src/target_generator_dsl/observer.rs` | new | search |
| `crates/bloqade-lanes-search/src/move_policy_dsl/kernel.rs` | modify | search |
| `crates/bloqade-lanes-search/src/move_policy_dsl/mod.rs` | modify (re-export) | search |
| `crates/bloqade-lanes-search/src/target_generator_dsl/kernel.rs` | modify | search |
| `crates/bloqade-lanes-search/src/target_generator_dsl/mod.rs` | modify (re-export) | search |
| `crates/bloqade-lanes-search/src/fixture.rs` | new (shared fixture loader, public) | search |
| `crates/bloqade-lanes-search/src/lib.rs` | modify (re-export `fixture::*`) | search |
| `crates/bloqade-lanes-search/src/bin/policies-primer.rs` | new | search |
| `crates/bloqade-lanes-search/tests/dsl_snapshot.rs` | new | search |
| `crates/bloqade-lanes-search/tests/primer_golden.rs` | new (primer generator regression test) | search |
| `crates/bloqade-lanes-search/tests/fixtures/primer/input/` | new (curated stub registration files for the golden test) | search |
| `crates/bloqade-lanes-search/tests/fixtures/primer/expected.md` | new (committed golden output) | search |
| `crates/bloqade-lanes-search/Cargo.toml` | modify (add `schemars`, `syn`/`proc_macro2`, dev-deps for tests) | search |
| `crates/bloqade-lanes-bytecode-cli/Cargo.toml` | modify (add `bloqade-lanes-search`, `bloqade-lanes-dsl-core`, `assert_cmd` dev-dep) | cli |
| `crates/bloqade-lanes-bytecode-cli/src/main.rs` | modify (register subcommands) | cli |
| `crates/bloqade-lanes-bytecode-cli/src/policy/mod.rs` | new | cli |
| `crates/bloqade-lanes-bytecode-cli/src/policy/eval.rs` | new | cli |
| `crates/bloqade-lanes-bytecode-cli/src/policy/trace.rs` | new | cli |
| `crates/bloqade-lanes-bytecode-cli/src/policy/output.rs` | new | cli |
| `crates/bloqade-lanes-bytecode-cli/tests/cli_policy.rs` | new | cli |
| `policies/reference/dfs.star` | new | repo |
| `policies/reference/bfs.star` | new | repo |
| `policies/reference/ids.star` | new | repo |
| `policies/primer.md` | new (autogenerated, committed) | repo |
| `policies/fixtures/README.md` | new | repo |
| `policies/fixtures/move/{small,medium,large}/problem.json` | new | repo |
| `policies/fixtures/move/small/expected.{entropy,dfs,bfs,ids}.json` | new | repo |
| `policies/fixtures/move/{medium,large}/expected.entropy.json` | new | repo |
| `policies/fixtures/target/small/problem.json` | new | repo |
| `policies/fixtures/target/small/expected.default_target.json` | new | repo |
| `docs/superpowers/specs/move-policy-dsl-problem-schema.md` | new (hand-written normative reference) | repo |
| `justfile` | modify (`generate-primer`, `check-primer`, `regenerate-fixtures`) | repo |
| `.github/workflows/ci.yml` | modify (add `just check-primer` step) | repo |

## 13. Out of scope (deferred past Plan C)

- Performance benchmarking of policies; integration with the existing `tests/` benchmark harness.
- Live-reload / watch-mode for the CLI.
- Starlark linter / formatter / language server.
- New `lib_target.*` primitives (Plan B's deferral stands).
- `from_arch_spec_json` constructor on `TargetPolicyRunner` (Plan B's deferral stands).
- Python-side CLI parity. Python policy-running already works via the `policy_path` kwarg on `MoveSolver.solve` and `PhysicalPlacementStrategy(target_generator=...)`; no Python CLI is needed.

## 14. Open questions

None. All architectural decisions resolved during the brainstorming pass:

- Scope: one unified spec covering all four deliverables.
- `eval-policy` vs. `trace-policy`: run-once-summary vs. step-by-step verbose.
- Input shape: single self-contained `problem.json` with `kind` discriminator.
- Reference policies: educational character (small, well-commented), not competitive.
- Output format: human by default, `--json` for machine; `trace-policy --json` is NDJSON.
- Primer mechanism: build-time codegen via a CI-checked binary, sentinel-bracketed PROSE blocks preserved.
- Snapshot criterion: structural-match on key fields, in `policies/fixtures/`.
- Move/Target dispatch: `"kind"` field in `problem.json`; one pair of CLI subcommands.
- CLI dependency edge: CLI directly depends on `bloqade-lanes-search`.
- Trace plumbing: kernel observer trait, sibling-module placement.
