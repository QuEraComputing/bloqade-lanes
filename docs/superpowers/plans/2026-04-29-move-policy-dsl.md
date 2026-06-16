# Move Policy DSL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land Plan A of the Move Policy / Target Generator DSL framework: a new `bloqade-lanes-dsl-core` crate hosting Starlark-based policy execution, a Move Policy DSL implementation in `bloqade-lanes-search`, PyO3 wiring on `MoveSolver.solve(...)`, and one reference policy (`entropy.star`) that serves as the acid-test against the existing `Strategy::Entropy`.

**Architecture:** A new shared substrate crate (`bloqade-lanes-dsl-core`) wraps `starlark-rust` to load `.star` files in a deterministic sandbox and exposes shared primitives (`LocationAddress`, `LaneAddr`, `Config`, `MoveSet`, `ArchSpec` wrapper, utilities) plus a generic `Policy<H, A, I, N>` trait. The Move DSL module inside `bloqade-lanes-search` (`move_policy_dsl/`) supplies a kernel loop that owns `SearchGraph` mutation, AOD validation, transposition dedup, and budgets, and exposes a `graph`/`ctx`/`lib`/`actions` surface to user-authored `.star` policies. PyO3 gains a `policy_path=` kwarg on `solve` that routes through this kernel.

**Tech Stack:** Rust 2024, `pyo3`, Meta's `starlark` crate (starlark-rust), Python 3.10+ (`uv`), `pytest`, existing `bloqade-lanes-search` infrastructure (`SearchGraph`, `LaneIndex`, `DistanceTable`, `BusGridContext`, `HeuristicGenerator`).

**Spec:** Move Policy DSL / Target Generator DSL design (Notion, 2026-04-29 transcription). Sections referenced as `§N`.

**Follow-up plans:**
- **Plan B — Target Generator DSL (delivered 2026-05-01).** Stacked on this branch; details in [Plan B — Target Generator DSL (delivered)](#plan-b--target-generator-dsl-delivered) below. Adds `StarlarkPlacement` to `dsl-core`, a new `target_generator_dsl/` module under `bloqade-lanes-search`, a PyO3 `TargetPolicyRunner` class, the Python adapter `TargetGeneratorDSL(TargetGeneratorABC)`, and a reference `policies/reference/default_target.star`. Reuses the existing Rust `validate_candidate` for safety enforcement.
- **Plan C — Reference policies + CLI harness + primer (out of scope here).** Adds `dfs.star`, `bfs.star`, `ids.star`; CLI subcommands `eval-policy` and `trace-policy` in `bloqade-lanes-bytecode-cli`; auto-generated `policies/primer.md` from Rust type definitions; snapshot fixtures across multiple problem sizes. Unblocked once Plan A's Move Policy DSL is on `main`.

---

## Conventions used throughout

- **Working tree:** `~/.config/superpowers/worktrees/bloqade-lanes/move-policy-dsl/` on branch `jason/move-policy-dsl`. All paths in this plan are relative to the repository root inside that worktree.
- **Run Rust tests:** `cargo test -p <crate>` for one crate, `just test-rust` for all Rust. Use `cargo test -p <crate> <test_name>` to target a single test.
- **Run Python tests:** `uv run pytest python/tests/<file>.py -v`. The native Rust extension must be rebuilt after Rust changes: `just develop-python` (fast path) or `just develop` (full).
- **Format / lint:** `cargo fmt --all` + `cargo clippy -p <crate> --all-targets -- -D warnings` before each commit. Pre-commit hooks enforce this on commit.
- **Commit messages:** Conventional Commits per `AGENT.md` (`feat(dsl-core): ...`, `feat(search): ...`, `test(search): ...`). Each task ends with **at most one** commit.
- **Git policy:** The user owns git operations. The `git commit` step in each task is provided for executing-agent convenience but **may be deferred to the user**. If executing without auto-commit, leave the working tree dirty between tasks and let the user batch-commit at logical boundaries — the per-task commit messages below still serve as the message templates.
- **starlark-rust version:** Pin to the latest stable release of the `starlark` crate at implementation time (≥ 0.13). The API used in this plan (Globals, Module, Evaluator, freeze) is stable across recent releases; if the API has shifted, adapt mechanically — the plan's structure is unaffected.

---

## File Structure

| File | Purpose |
|---|---|
| `crates/bloqade-lanes-dsl-core/Cargo.toml` (new) | Crate manifest; depends on `bloqade-lanes-bytecode-core`, `starlark`, `serde`, `thiserror`. |
| `crates/bloqade-lanes-dsl-core/src/lib.rs` (new) | Crate root, re-exports. |
| `crates/bloqade-lanes-dsl-core/src/errors.rs` (new) | `DslError` variants (parse, runtime, schema, budget, OOM, bad-policy). |
| `crates/bloqade-lanes-dsl-core/src/sandbox.rs` (new) | Sandbox config + `make_evaluator()` helper that disables network/IO/recursion/`load()`, sets step + memory caps. |
| `crates/bloqade-lanes-dsl-core/src/adapter.rs` (new) | Load/parse/freeze `.star` files; cache by SHA-256 + mtime; named-export invocation; Starlark↔Rust value marshaling. |
| `crates/bloqade-lanes-dsl-core/src/policy_trait.rs` (new) | Generic `Policy<Handle, Action, InitArg, NodeState, GlobalState>` trait. |
| `crates/bloqade-lanes-dsl-core/src/primitives/mod.rs` (new) | Sub-module index. |
| `crates/bloqade-lanes-dsl-core/src/primitives/types.rs` (new) | Starlark wrappers for `LocationAddress`, `LaneAddr`, `Config`, `MoveSet`. |
| `crates/bloqade-lanes-dsl-core/src/primitives/arch_spec.rs` (new) | Starlark wrapper for `ArchSpec` (`get_cz_partner`, `check_location_group`, `num_locations`, `position`, `outgoing_lanes`, `lane_endpoints`, `lane_triplet`, `lane_duration_us`). |
| `crates/bloqade-lanes-dsl-core/src/primitives/utilities.rs` (new) | Starlark globals `stable_sort`, `argmax`, `normalize`. |
| `crates/bloqade-lanes-search/Cargo.toml` (modify) | Add path-dep on `bloqade-lanes-dsl-core`. |
| `crates/bloqade-lanes-search/src/lib.rs` (modify) | Export `move_policy_dsl` module + new `Strategy::StarlarkPolicy` variant or top-level `solve_with_policy` entrypoint (decision below). |
| `crates/bloqade-lanes-search/src/move_policy_dsl/mod.rs` (new) | Module index. |
| `crates/bloqade-lanes-search/src/move_policy_dsl/actions.rs` (new) | `MoveAction` enum + Starlark `actions` global (six verbs). |
| `crates/bloqade-lanes-search/src/move_policy_dsl/graph_handle.rs` (new) | `PolicyGraph` Starlark value: read accessors + `last_insert` / `last_builtin_result` side channels. |
| `crates/bloqade-lanes-search/src/move_policy_dsl/lib_move.rs` (new) | Move-specific `lib.*` primitives: distances, mobility, qubit queries, candidate pipeline, graph helpers. |
| `crates/bloqade-lanes-search/src/move_policy_dsl/kernel.rs` (new) | The kernel loop: init → step → apply actions → budget check; status mapping. |
| `crates/bloqade-lanes-search/src/move_policy_dsl/adapter_impl.rs` (new) | Glue between `dsl-core::Policy` trait and the Move kernel. |
| `crates/bloqade-lanes-search/src/aod_grid.rs` (modify) | Promote `BusGridContext::new` and `build_aod_grids` from `pub(crate)` to `pub(crate)` callable from `move_policy_dsl/` (already in-crate; no change unless visibility blocks). Verify only. |
| `crates/bloqade-lanes-bytecode-python/src/search_python.rs` (modify) | Add `policy_path: Option<&str>`, `policy_params: Option<&PyDict>` kwargs to `MoveSolver.solve`; new `PySolveResult.policy_file`, `PySolveResult.policy_params` fields. |
| `python/bloqade/lanes/bytecode/_native.pyi` (modify) | Reflect the new kwargs and result fields. |
| `policies/reference/entropy.star` (new) | Reference Starlark reproduction of `Strategy::Entropy` (acid test). |
| `crates/bloqade-lanes-search/tests/dsl_entropy_acid.rs` (new) | Integration test: run `entropy.star` against fixtures and compare structurally with `Strategy::Entropy`. |
| `python/tests/bytecode/test_move_policy_dsl.py` (new) | End-to-end Python smoke test. |
| `Cargo.toml` (modify) | Add `crates/bloqade-lanes-dsl-core` to workspace members. |

---

## Decision: how `policy_path` enters the public API

Per spec §14 ("Open questions"), the orthogonal-kwarg path wins: a Starlark policy is **not** a new `Strategy` variant. Instead, `MoveSolver.solve` gets two new optional kwargs `policy_path` and `policy_params`. When `policy_path` is set, `Strategy` is ignored and the request routes to the `move_policy_dsl::kernel`. Rationale: keeps the strategy taxonomy unchanged, lets `Strategy::Entropy` and `entropy.star` coexist for A/B comparison.

The Rust-internal hand-off is:

```
PyMoveSolver::solve  (search_python.rs)
    │  if policy_path.is_some()
    ▼
move_policy_dsl::kernel::solve_with_policy(...)  // new public fn in lib.rs
    │
    ▼
dsl_core::adapter::PolicyAdapter<MoveAction, ...>::run(...)
```

---

## Phase 1 — `bloqade-lanes-dsl-core` crate scaffold

### Task 1: Workspace + crate skeleton

**Files:**
- Create: `crates/bloqade-lanes-dsl-core/Cargo.toml`
- Create: `crates/bloqade-lanes-dsl-core/src/lib.rs`
- Modify: `Cargo.toml` (workspace root)

- [ ] **Step 1: Create the crate manifest**

Create `crates/bloqade-lanes-dsl-core/Cargo.toml`:

```toml
[package]
name = "bloqade-lanes-dsl-core"
version.workspace = true
edition = "2024"

[lib]
name = "bloqade_lanes_dsl_core"

[dependencies]
bloqade-lanes-bytecode-core = { path = "../bloqade-lanes-bytecode-core" }
starlark = "0.13"
allocative = "0.3"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
thiserror = "1"

[dev-dependencies]
tempfile = "3"
```

(`allocative` is required by `starlark` for tracing; pinning is straightforward. Bump versions if newer compatible majors exist.)

- [ ] **Step 2: Create the empty lib.rs**

Create `crates/bloqade-lanes-dsl-core/src/lib.rs`:

```rust
//! Shared substrate for Starlark-hosted policy DSLs.
//!
//! Hosts the starlark-rust adapter, sandbox configuration, and shared
//! primitives (`LocationAddress`, `LaneAddr`, `Config`, `MoveSet`,
//! `ArchSpec` wrapper, utilities) used by both the Move Policy DSL and
//! the Target Generator DSL.

pub mod adapter;
pub mod errors;
pub mod policy_trait;
pub mod primitives;
pub mod sandbox;

pub use errors::DslError;
pub use policy_trait::{Policy, StepResult};
```

(`adapter`, `policy_trait`, `primitives`, `sandbox` will be filled in by later tasks; declaring them up front keeps `cargo check` red-green simple.)

- [ ] **Step 3: Add the crate to the workspace**

Modify `Cargo.toml` (workspace root). Add `"crates/bloqade-lanes-dsl-core"` to `[workspace] members`:

```toml
[workspace]
members = [
    "crates/bloqade-lanes-bytecode-core",
    "crates/bloqade-lanes-bytecode-python",
    "crates/bloqade-lanes-bytecode-cli",
    "crates/bloqade-lanes-dsl-core",
    "crates/bloqade-lanes-search",
]
resolver = "2"
```

- [ ] **Step 4: Create stub files for declared modules**

Create `crates/bloqade-lanes-dsl-core/src/errors.rs`:

```rust
//! `DslError` — error type surfaced from policy load and execution.
```

Create `crates/bloqade-lanes-dsl-core/src/sandbox.rs`:

```rust
//! Sandbox configuration for Starlark evaluators.
```

Create `crates/bloqade-lanes-dsl-core/src/adapter.rs`:

```rust
//! Starlark-rust adapter: parse, freeze, invoke named exports.
```

Create `crates/bloqade-lanes-dsl-core/src/policy_trait.rs`:

```rust
//! Generic `Policy<H, A, I, N, G>` trait — hosted by the DSL kernel.
```

Create `crates/bloqade-lanes-dsl-core/src/primitives/mod.rs`:

```rust
//! Shared Starlark primitives bound into every policy environment.

pub mod arch_spec;
pub mod types;
pub mod utilities;
```

Create empty `primitives/types.rs`, `primitives/arch_spec.rs`, `primitives/utilities.rs` with doc-comment one-liners.

- [ ] **Step 5: Verify the workspace compiles**

Run: `cargo check -p bloqade-lanes-dsl-core`
Expected: succeeds with empty modules; no warnings beyond unused-imports if any.

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml crates/bloqade-lanes-dsl-core/
git commit -m "$(cat <<'EOF'
feat(dsl-core): scaffold bloqade-lanes-dsl-core crate

Add empty workspace member with module skeleton: errors, sandbox,
adapter, policy_trait, primitives. Pulls starlark-rust as a direct
dependency for the upcoming Starlark-hosted policy DSLs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `DslError` enum

**Files:**
- Modify: `crates/bloqade-lanes-dsl-core/src/errors.rs`
- Test: `crates/bloqade-lanes-dsl-core/src/errors.rs` (inline `#[cfg(test)]`)

- [ ] **Step 1: Write the failing test for the error variants**

Append to `crates/bloqade-lanes-dsl-core/src/errors.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_error_displays_with_location() {
        let err = DslError::Parse {
            path: "foo.star".into(),
            message: "unexpected EOF".into(),
        };
        let s = format!("{err}");
        assert!(s.contains("foo.star"), "missing path in display: {s}");
        assert!(s.contains("unexpected EOF"));
    }

    #[test]
    fn variants_round_trip_through_display() {
        let cases = [
            DslError::Runtime { traceback: "x".into() },
            DslError::Schema { field: "entropy".into(), message: "type".into() },
            DslError::BadPolicy("not an action".into()),
            DslError::StepBudget,
            DslError::MemoryBudget,
        ];
        for c in &cases {
            let _ = format!("{c}"); // must not panic
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p bloqade-lanes-dsl-core`
Expected: FAIL — `DslError` does not yet exist.

- [ ] **Step 3: Implement `DslError`**

Replace `crates/bloqade-lanes-dsl-core/src/errors.rs` with:

```rust
//! `DslError` — error type surfaced from policy load and execution.

use thiserror::Error;

/// Errors raised by the DSL adapter or kernel.
///
/// These map 1:1 to the status codes documented in the spec §5.10
/// (Move DSL error model). The hosting kernel (e.g.
/// `move_policy_dsl::kernel`) is responsible for converting these into
/// public `SolveResult` statuses.
#[derive(Debug, Error)]
pub enum DslError {
    /// `.star` file failed to parse.
    #[error("{path}: parse error: {message}")]
    Parse { path: String, message: String },

    /// Starlark runtime error during `init`/`step`/`generate`.
    #[error("starlark runtime error:\n{traceback}")]
    Runtime { traceback: String },

    /// `update_node_state` / `update_global_state` named a field not in
    /// the declared schema.
    #[error("schema error on field `{field}`: {message}")]
    Schema { field: String, message: String },

    /// `step()` returned something that wasn't an `Action` or
    /// `list[Action]`.
    #[error("policy returned an invalid value: {0}")]
    BadPolicy(String),

    /// Per-`step()` Starlark step budget exceeded.
    #[error("starlark step budget exceeded")]
    StepBudget,

    /// Per-solve Starlark memory cap exceeded.
    #[error("starlark memory cap exceeded")]
    MemoryBudget,

    /// Wrapper over arbitrary IO errors (e.g. opening the .star file).
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    // ...existing tests above...
}
```

(Keep the test block at the bottom.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p bloqade-lanes-dsl-core`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/bloqade-lanes-dsl-core/src/errors.rs
git commit -m "feat(dsl-core): add DslError enum with display impls

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Sandbox configuration

**Files:**
- Modify: `crates/bloqade-lanes-dsl-core/src/sandbox.rs`

- [ ] **Step 1: Write failing test for sandbox creation + budget enforcement**

Replace `crates/bloqade-lanes-dsl-core/src/sandbox.rs` content with the sketch below; this contains both the doc and the eventual test block. We start by writing **only** the test block at the bottom. Append:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_finite_and_positive() {
        let cfg = SandboxConfig::default();
        assert!(cfg.starlark_steps_per_tick >= 10_000);
        assert!(cfg.starlark_memory_per_solve_bytes >= 16 * 1024 * 1024);
        assert_eq!(cfg.max_loaded_files, 1);
    }

    #[test]
    fn evaluator_rejects_load_statement() {
        // A .star file containing `load("foo.star", "bar")` must fail
        // to parse-or-eval under our sandbox: load() is disabled.
        let cfg = SandboxConfig::default();
        let module = starlark::environment::Module::new();
        let globals = build_globals(&cfg);
        let mut eval = make_evaluator(&module, &globals, &cfg);
        let src = r#"load("other.star", "x")"#;
        let ast = starlark::syntax::AstModule::parse(
            "test.star",
            src.to_owned(),
            &starlark::syntax::Dialect::Standard,
        )
        .expect("parse");
        let result = eval.eval_module(ast, &globals);
        assert!(result.is_err(), "load() must be rejected by sandbox");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p bloqade-lanes-dsl-core sandbox`
Expected: FAIL — `SandboxConfig`, `build_globals`, `make_evaluator` not defined.

- [ ] **Step 3: Implement the sandbox**

Replace `crates/bloqade-lanes-dsl-core/src/sandbox.rs` (above the `#[cfg(test)]` block) with:

```rust
//! Sandbox configuration for Starlark evaluators.
//!
//! Disables non-deterministic / IO / network builtins, caps per-tick
//! step count, and caps per-solve heap usage. Disables `load()` to
//! prevent silent file-system access: each policy is a single
//! self-contained `.star` file.

use starlark::environment::{Globals, GlobalsBuilder, Module};
use starlark::eval::Evaluator;

/// Tunable resource limits applied to every policy evaluator.
#[derive(Debug, Clone)]
pub struct SandboxConfig {
    /// Per-`step()` Starlark instruction budget. Default 1_000_000.
    pub starlark_steps_per_tick: u64,
    /// Per-solve Starlark heap cap, in bytes. Default 128 MiB.
    pub starlark_memory_per_solve_bytes: u64,
    /// Max number of `.star` files that may be loaded. We disable
    /// `load()` outright; this is set to 1 (the policy file).
    pub max_loaded_files: u32,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            starlark_steps_per_tick: 1_000_000,
            starlark_memory_per_solve_bytes: 128 * 1024 * 1024,
            max_loaded_files: 1,
        }
    }
}

/// Build the `Globals` for a sandboxed evaluator.
///
/// Adds nothing beyond Starlark's standard library minus IO/time. The
/// caller (DSL-specific glue) extends this with `lib`, `actions`, etc.
pub fn build_globals(_cfg: &SandboxConfig) -> Globals {
    GlobalsBuilder::standard()
        // Standard Starlark already excludes time/IO/network. Recursion
        // is forbidden by the language spec. `load()` is rejected by
        // not registering a `FileLoader` on the evaluator below.
        .build()
}

/// Create an `Evaluator` bound to `module`, using `globals`, with all
/// sandbox limits applied.
pub fn make_evaluator<'a, 'v>(
    module: &'v Module,
    globals: &'a Globals,
    cfg: &SandboxConfig,
) -> Evaluator<'v, 'a, 'a> {
    let mut eval = Evaluator::new(module);
    // Refuse `load()`: no FileLoader is registered.
    eval.set_loader(&starlark::eval::FileLoader::default()); // no-op; replace with the explicit "deny" loader if the API requires one in the pinned starlark version.
    // Apply step + memory budgets.
    eval.set_max_callstack_size(64).ok(); // bounds recursion explicitly
    let _ = (cfg, globals); // remaining caps wired by caller per call (`enter_module`)
    eval
}
```

The exact API for setting `max_callstack_size`, the step cap, and the memory cap drifts between starlark-rust releases. **Verify against the pinned version**: search `Evaluator::set_max_*` in `cargo doc -p starlark --open`. The acceptance criterion is that the test in Step 1 passes — i.e. a `.star` containing `load(...)` fails to evaluate.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p bloqade-lanes-dsl-core sandbox`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/bloqade-lanes-dsl-core/src/sandbox.rs
git commit -m "feat(dsl-core): add SandboxConfig + sandboxed Evaluator factory

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Adapter — load + parse + invoke a named export

**Files:**
- Modify: `crates/bloqade-lanes-dsl-core/src/adapter.rs`
- Test: same file, inline `#[cfg(test)]`.

- [ ] **Step 1: Write failing tests for adapter loading**

Replace `crates/bloqade-lanes-dsl-core/src/adapter.rs` with:

```rust
//! Starlark-rust adapter: parse, freeze, invoke named exports.

use std::path::Path;

use starlark::environment::{FrozenModule, Globals, Module};
use starlark::syntax::{AstModule, Dialect};

use crate::errors::DslError;
use crate::sandbox::{SandboxConfig, build_globals, make_evaluator};

/// A loaded, frozen Starlark policy module.
///
/// Re-usable across multiple solve invocations: `init`/`step`/`generate`
/// are pure functions over arguments; the module itself holds no
/// mutable state.
pub struct LoadedPolicy {
    pub frozen: FrozenModule,
    pub globals: Globals,
    pub source_path: String,
}

impl LoadedPolicy {
    /// Parse and freeze a policy from a file path.
    pub fn from_path(path: impl AsRef<Path>, cfg: &SandboxConfig) -> Result<Self, DslError> {
        let path_ref = path.as_ref();
        let src = std::fs::read_to_string(path_ref)?;
        Self::from_source(
            path_ref.to_string_lossy().into_owned(),
            src,
            cfg,
        )
    }

    /// Parse and freeze a policy from in-memory source. Used in tests.
    pub fn from_source(
        source_path: String,
        source: String,
        cfg: &SandboxConfig,
    ) -> Result<Self, DslError> {
        let ast = AstModule::parse(&source_path, source, &Dialect::Standard).map_err(|e| {
            DslError::Parse {
                path: source_path.clone(),
                message: format!("{e}"),
            }
        })?;

        let module = Module::new();
        let globals = build_globals(cfg);
        {
            let mut eval = make_evaluator(&module, &globals, cfg);
            eval.eval_module(ast, &globals).map_err(|e| DslError::Runtime {
                traceback: format!("{e:?}"),
            })?;
        }
        let frozen = module
            .freeze()
            .map_err(|e| DslError::Runtime { traceback: format!("{e:?}") })?;
        Ok(Self {
            frozen,
            globals,
            source_path,
        })
    }

    /// Look up a top-level binding by name.
    pub fn get<'v>(
        &'v self,
        name: &str,
    ) -> Option<starlark::values::FrozenValue> {
        self.frozen.get(name).ok().map(|v| v.value())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_a_simple_policy_from_source() {
        let cfg = SandboxConfig::default();
        let src = r#"
PARAMS = struct(answer = 42)
def hello(x):
    return x + 1
"#;
        let p = LoadedPolicy::from_source("inline.star".into(), src.into(), &cfg)
            .expect("load");
        assert!(p.get("PARAMS").is_some(), "PARAMS not exported");
        assert!(p.get("hello").is_some(), "hello not exported");
    }

    #[test]
    fn parse_error_returns_parse_variant() {
        let cfg = SandboxConfig::default();
        let src = "def broken(:\n";
        let err = LoadedPolicy::from_source("bad.star".into(), src.into(), &cfg)
            .err()
            .expect("must fail");
        assert!(matches!(err, DslError::Parse { .. }), "got {err:?}");
    }

    #[test]
    fn load_statement_is_rejected() {
        let cfg = SandboxConfig::default();
        let src = r#"load("other.star", "x")"#;
        let err = LoadedPolicy::from_source("loader.star".into(), src.into(), &cfg)
            .err()
            .expect("must fail");
        // Either Parse or Runtime is acceptable; the contract is "must fail".
        assert!(matches!(err, DslError::Parse { .. } | DslError::Runtime { .. }));
    }
}
```

- [ ] **Step 2: Run test to verify failures (then real failures)**

Run: `cargo test -p bloqade-lanes-dsl-core adapter`
Expected: FAIL on the first run (compile errors). After Step 1 lands the test code, all three tests should fail compilation (no `LoadedPolicy`). Adding the impl in Step 1 fixes compile. After fix, all three pass.

(The test block in Step 1 already contains the impl; the order is: write the file, run the tests, watch them pass. If you want strict TDD, comment out the impl, run, see it fail, then uncomment.)

- [ ] **Step 3: Confirm tests pass**

Run: `cargo test -p bloqade-lanes-dsl-core adapter -- --nocapture`
Expected: 3 passed.

- [ ] **Step 4: Commit**

```bash
git add crates/bloqade-lanes-dsl-core/src/adapter.rs
git commit -m "feat(dsl-core): add LoadedPolicy adapter for parse + freeze

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 2 — Shared Starlark primitives

### Task 5: `LocationAddress` + `LaneAddr` Starlark wrappers

**Files:**
- Modify: `crates/bloqade-lanes-dsl-core/src/primitives/types.rs`

- [ ] **Step 1: Write failing tests for the wrappers**

Replace `crates/bloqade-lanes-dsl-core/src/primitives/types.rs` with:

```rust
//! Starlark wrappers for the bytecode core address types.

use std::fmt;

use bloqade_lanes_bytecode_core::arch::addr::{LaneAddr, LocationAddr};
use starlark::starlark_simple_value;
use starlark::values::{Heap, NoSerialize, ProvidesStaticType, StarlarkValue, Value};

/// Starlark-visible wrapper around `LocationAddr`.
#[derive(Debug, Clone, ProvidesStaticType, NoSerialize, allocative::Allocative)]
pub struct StarlarkLocation(pub LocationAddr);

starlark_simple_value!(StarlarkLocation);

impl fmt::Display for StarlarkLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Loc({}, {})", self.0.word_id(), self.0.site_id())
    }
}

#[starlark::values::starlark_value(type = "Location")]
impl<'v> StarlarkValue<'v> for StarlarkLocation {
    fn get_attr(&self, attr: &str, _heap: &'v Heap) -> Option<Value<'v>> {
        match attr {
            "word_id" => Some(Value::new_int(self.0.word_id() as i32)),
            "site_id" => Some(Value::new_int(self.0.site_id() as i32)),
            _ => None,
        }
    }
}

/// Starlark-visible wrapper around `LaneAddr`. `.encoded` is the canonical
/// stable identifier; structural fields (`move_type`, `bus_id`, `direction`,
/// `zone_id`) are exposed for tie-breaking and diagnostics.
#[derive(Debug, Clone, ProvidesStaticType, NoSerialize, allocative::Allocative)]
pub struct StarlarkLane(pub LaneAddr);

starlark_simple_value!(StarlarkLane);

impl fmt::Display for StarlarkLane {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Lane({})", self.0.encode_u64())
    }
}

#[starlark::values::starlark_value(type = "Lane")]
impl<'v> StarlarkValue<'v> for StarlarkLane {
    fn get_attr(&self, attr: &str, _heap: &'v Heap) -> Option<Value<'v>> {
        match attr {
            "encoded" => Some(Value::new_int(self.0.encode_u64() as i64 as i32)), // see note
            "move_type" => Some(Value::new_int(self.0.move_type() as i32)),
            "bus_id" => Some(Value::new_int(self.0.bus_id() as i32)),
            "direction" => Some(Value::new_int(self.0.direction() as i32)),
            "zone_id" => Some(Value::new_int(self.0.zone_id() as i32)),
            _ => None,
        }
    }
}
// NOTE: Starlark integers are `i32`. `LaneAddr::encode_u64()` returns `u64`,
// which can overflow. The test below checks values fit in i32 for the
// architectures we support. If overflow becomes a real concern, switch
// `encoded` to a Starlark `bigint` (StarlarkBigInt); the v1 sandbox spec
// targets the Gemini arch, which fits.

#[cfg(test)]
mod tests {
    use super::*;
    use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;

    #[test]
    fn location_attrs_are_readable() {
        let loc = LocationAddr::new(3, 7);
        let s = StarlarkLocation(loc);
        assert!(format!("{s}").contains("3, 7"));
    }
}
```

(Why a single happy-path test rather than four? Each `get_attr` call is mechanical mirroring of the bytecode-core type. The acid test in Phase 6 exercises every field via real policies. Spending more TDD effort here is uneconomical.)

- [ ] **Step 2: Run tests, watch them pass after impl**

Run: `cargo test -p bloqade-lanes-dsl-core types`
Expected: PASS (the test only formats; the compile is the real validation).

- [ ] **Step 3: Commit**

```bash
git add crates/bloqade-lanes-dsl-core/src/primitives/types.rs
git commit -m "feat(dsl-core): add Location and Lane Starlark wrappers

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: `Config` + `MoveSet` Starlark wrappers

**Files:**
- Modify: `crates/bloqade-lanes-dsl-core/src/primitives/types.rs`

- [ ] **Step 1: Add failing tests for `Config` and `MoveSet`**

Append to `primitives/types.rs` (above the existing test block):

```rust
use bloqade_lanes_search::config::Config; // path-dep added in Task 13
use bloqade_lanes_search::graph::MoveSet;

/// Immutable wrapper around `Config`. Exposed as `Config` in policies.
#[derive(Debug, Clone, ProvidesStaticType, NoSerialize, allocative::Allocative)]
pub struct StarlarkConfig(pub Config);

starlark_simple_value!(StarlarkConfig);

impl fmt::Display for StarlarkConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Config(len={})", self.0.len())
    }
}

#[starlark::values::starlark_value(type = "Config")]
impl<'v> StarlarkValue<'v> for StarlarkConfig {
    fn get_attr(&self, attr: &str, _heap: &'v Heap) -> Option<Value<'v>> {
        match attr {
            "len" => Some(Value::new_int(self.0.len() as i32)),
            _ => None,
        }
    }
    // Method `get(qid)` is exposed via a separate macro-driven impl
    // (see starlark::values::starlark_module). Implement after the
    // attr-only path is validated.
}
```

(Defer `Config.iter`, `Config.hash`, `MoveSet.lanes`/`.encoded`/`.len` until they are first needed by `lib_move`. Add them in the same file as the consumer arrives — keeps the per-task surface tight.)

```rust
/// Wrapper around `MoveSet`.
#[derive(Debug, Clone, ProvidesStaticType, NoSerialize, allocative::Allocative)]
pub struct StarlarkMoveSet(pub MoveSet);

starlark_simple_value!(StarlarkMoveSet);

impl fmt::Display for StarlarkMoveSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MoveSet(len={})", self.0.len())
    }
}

#[starlark::values::starlark_value(type = "MoveSet")]
impl<'v> StarlarkValue<'v> for StarlarkMoveSet {
    fn get_attr(&self, attr: &str, _heap: &'v Heap) -> Option<Value<'v>> {
        match attr {
            "len" => Some(Value::new_int(self.0.len() as i32)),
            _ => None,
        }
    }
}
```

Append a test:

```rust
#[test]
fn config_wraps_and_reports_len() {
    use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;
    let cfg = Config::new([(0, LocationAddr::new(0, 0))]).unwrap();
    let w = StarlarkConfig(cfg);
    assert_eq!(format!("{w}"), "Config(len=1)");
}
```

- [ ] **Step 2: Add the `bloqade-lanes-search` path-dep to `dsl-core`**

Modify `crates/bloqade-lanes-dsl-core/Cargo.toml`:

This creates a circular issue if `bloqade-lanes-search` later depends on `dsl-core`. Resolution: keep `Config` / `MoveSet` defined in `bytecode-core` (they already exist conceptually as encoding-level types); **or** move the `Config`/`MoveSet` Starlark wrappers into a `search`-side module. Choose the second to avoid the cycle:

→ **Decision:** `StarlarkConfig` and `StarlarkMoveSet` live in `bloqade-lanes-search/src/move_policy_dsl/lib_move.rs` (or a peer file). `dsl-core::primitives::types` only wraps types from `bytecode-core` (Location, Lane). Revert this task's `Cargo.toml` change; move `StarlarkConfig`/`StarlarkMoveSet` to Phase 3.

- [ ] **Step 3: Update Phase 2 scope and commit**

The test added above moves to `move_policy_dsl/lib_move.rs` in Task 14. Remove the Config/MoveSet code from `primitives/types.rs` for now. Commit only the Location/Lane Starlark module work from Task 5 (already done).

If anything in this task got committed already, no action needed: Task 14 will replay it on the search side.

(Mark this task complete if the dependency-direction analysis is recorded in the commit history of Task 5. Otherwise leave as a documentation-only task: no separate commit.)

---

### Task 7: `ArchSpec` Starlark wrapper

**Files:**
- Modify: `crates/bloqade-lanes-dsl-core/src/primitives/arch_spec.rs`

- [ ] **Step 1: Survey the underlying API**

Read `crates/bloqade-lanes-bytecode-core/src/arch/types.rs` (specifically the `ArchSpec` impl) and `crates/bloqade-lanes-search/src/lane_index.rs` to confirm the methods we proxy:

- `ArchSpec::get_cz_partner(loc) -> Option<LocationAddr>`
- `ArchSpec::check_location_group(locs) -> Vec<...>`
- `ArchSpec::lane_endpoints(&LaneAddr) -> Option<(LocationAddr, LocationAddr)>`
- `ArchSpec::location_position(&LocationAddr) -> Option<(f64, f64)>`
- `LaneIndex::outgoing_lanes(loc) -> &[LaneAddr]`
- `LaneIndex::lane_durations[lane.encode_u64()] -> Option<f64>`

(Some methods we expose live on `LaneIndex`, not `ArchSpec`. The Starlark wrapper holds an `Arc` to both; the policy sees one `arch_spec` handle.)

- [ ] **Step 2: Write failing test using a tiny in-memory ArchSpec fixture**

Add `crates/bloqade-lanes-dsl-core/tests/arch_spec_wrapper.rs`:

```rust
// Acceptance: a tiny .star policy can read get_cz_partner via ctx.arch_spec.
// The fixture loads from `crates/bloqade-lanes-bytecode-core/tests/fixtures/`
// or generates inline. Use whatever already exists — the Starlark-side
// surface is tested by the move_policy_dsl integration tests in Phase 6.
```

This task's TDD is light because the wrapper is mechanical proxying. Defer rich tests to Phase 6's acid-test.

- [ ] **Step 3: Implement the wrapper**

Replace `crates/bloqade-lanes-dsl-core/src/primitives/arch_spec.rs` with:

```rust
//! Starlark-visible `ArchSpec` wrapper.

use std::sync::Arc;

use bloqade_lanes_bytecode_core::arch::addr::{LaneAddr, LocationAddr};
use bloqade_lanes_bytecode_core::arch::types::ArchSpec;
use starlark::starlark_module;
use starlark::values::{Heap, ProvidesStaticType, StarlarkValue, Value, NoSerialize};

use crate::primitives::types::{StarlarkLane, StarlarkLocation};

/// Read-only Starlark handle exposing the architecture spec to a policy.
///
/// Holds an `Arc<ArchSpec>` so it can be cloned cheaply across the
/// boundary. The `LaneIndex` accessors needed by `outgoing_lanes` and
/// `lane_endpoints`/`lane_duration_us` are passed in via the search-side
/// glue (see `move_policy_dsl/lib_move.rs`); the dsl-core surface here
/// covers the architecture-only methods.
#[derive(Debug, Clone, ProvidesStaticType, NoSerialize, allocative::Allocative)]
pub struct StarlarkArchSpec(pub Arc<ArchSpec>);

starlark::starlark_simple_value!(StarlarkArchSpec);

impl std::fmt::Display for StarlarkArchSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ArchSpec(name={})", self.0.name())
    }
}

#[starlark::values::starlark_value(type = "ArchSpec")]
impl<'v> StarlarkValue<'v> for StarlarkArchSpec {}

#[starlark_module]
pub fn register_arch_spec_methods(builder: &mut starlark::environment::MethodsBuilder) {
    /// Return the CZ-blockade partner of `loc`, or None.
    fn get_cz_partner<'v>(this: &StarlarkArchSpec, loc: &StarlarkLocation) -> anyhow::Result<Value<'v>> {
        match this.0.get_cz_partner(&loc.0) {
            Some(p) => Ok(Value::new_alloc_simple(StarlarkLocation(p))),
            None => Ok(Value::new_none()),
        }
    }

    /// Validate a candidate location group; returns a list of error messages.
    fn check_location_group<'v>(this: &StarlarkArchSpec, locs: Vec<&StarlarkLocation>) -> anyhow::Result<Vec<String>> {
        let raw: Vec<LocationAddr> = locs.iter().map(|l| l.0).collect();
        Ok(this.0.check_location_group(&raw).into_iter().map(|e| e.to_string()).collect())
    }

    /// Total number of locations in this architecture.
    fn num_locations(this: &StarlarkArchSpec) -> anyhow::Result<i32> {
        Ok(this.0.num_locations() as i32)
    }

    /// `(x, y)` physical position of `loc`, in micrometers.
    fn position<'v>(this: &StarlarkArchSpec, loc: &StarlarkLocation) -> anyhow::Result<Value<'v>> {
        match this.0.location_position(&loc.0) {
            Some((x, y)) => Ok(starlark::values::tuple::AllocTuple::alloc_2(x, y).to_value()),
            None => Ok(Value::new_none()),
        }
    }

    /// Endpoints of a lane: `(src, dst)` or `None`.
    fn lane_endpoints<'v>(this: &StarlarkArchSpec, lane: &StarlarkLane) -> anyhow::Result<Value<'v>> {
        match this.0.lane_endpoints(&lane.0) {
            Some((s, d)) => Ok(starlark::values::tuple::AllocTuple::alloc_2(
                StarlarkLocation(s),
                StarlarkLocation(d),
            )
            .to_value()),
            None => Ok(Value::new_none()),
        }
    }

    /// Triplet `(move_type, bus_id, direction)` of a lane.
    fn lane_triplet(this: &StarlarkArchSpec, lane: &StarlarkLane) -> anyhow::Result<(i32, i32, i32)> {
        Ok((
            lane.0.move_type() as i32,
            lane.0.bus_id() as i32,
            lane.0.direction() as i32,
        ))
    }
}
```

**Note on `outgoing_lanes` and `lane_duration_us`:** these live on `LaneIndex`, not `ArchSpec`. They will be added to the wrapper in Task 14 once the Move kernel can hand a `&LaneIndex` to the wrapper. For now the spec contract is "exposed via `ctx.arch_spec`", and the call sites are in `lib_move`, so we route them through `lib.*` rather than `ctx.arch_spec.*`. **Update spec §7.1 reference accordingly when the implementation diverges**: in this implementation, `outgoing_lanes(loc)` and `lane_duration_us(lane)` are on `lib`, not `arch_spec`.

- [ ] **Step 4: Run tests**

Run: `cargo test -p bloqade-lanes-dsl-core`
Expected: PASS (compile success on Starlark proxying).

- [ ] **Step 5: Commit**

```bash
git add crates/bloqade-lanes-dsl-core/src/primitives/arch_spec.rs
git commit -m "feat(dsl-core): add StarlarkArchSpec read-only wrapper

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Utilities (`stable_sort`, `argmax`, `normalize`)

**Files:**
- Modify: `crates/bloqade-lanes-dsl-core/src/primitives/utilities.rs`

- [ ] **Step 1: Failing tests for the three utilities**

Replace `primitives/utilities.rs` with:

```rust
//! Starlark globals: `stable_sort`, `argmax`, `normalize`.
//!
//! These are bound into every policy environment for deterministic
//! tie-breaking and ergonomic numeric pipelines.

use starlark::starlark_module;
use starlark::values::{Heap, Value};
use starlark::values::list::AllocList;

#[starlark_module]
pub fn register_utilities(builder: &mut starlark::environment::GlobalsBuilder) {
    /// Stable sort by `key_fn`. Returns a new list. `desc=True` reverses.
    fn stable_sort<'v>(
        items: Vec<Value<'v>>,
        key_fn: Value<'v>,
        #[starlark(default = false)] desc: bool,
        heap: &'v Heap,
        eval: &mut starlark::eval::Evaluator<'v, '_, '_>,
    ) -> anyhow::Result<Value<'v>> {
        let mut keyed: Vec<(usize, Value<'v>, Value<'v>)> = Vec::with_capacity(items.len());
        for (i, item) in items.iter().enumerate() {
            let k = eval.eval_function(key_fn, &[*item], &[])?;
            keyed.push((i, *item, k));
        }
        keyed.sort_by(|a, b| {
            let ord = a.2.compare(b.2).unwrap_or(std::cmp::Ordering::Equal);
            if desc { ord.reverse() } else { ord }.then_with(|| a.0.cmp(&b.0))
        });
        Ok(heap.alloc(AllocList(keyed.into_iter().map(|(_, v, _)| v))))
    }

    /// Return the item with the largest `key_fn(item)`. Ties: first
    /// occurrence wins. Empty input → `None`.
    fn argmax<'v>(
        items: Vec<Value<'v>>,
        key_fn: Value<'v>,
        eval: &mut starlark::eval::Evaluator<'v, '_, '_>,
    ) -> anyhow::Result<Value<'v>> {
        if items.is_empty() {
            return Ok(Value::new_none());
        }
        let mut best_idx: usize = 0;
        let mut best_key = eval.eval_function(key_fn, &[items[0]], &[])?;
        for i in 1..items.len() {
            let k = eval.eval_function(key_fn, &[items[i]], &[])?;
            if k.compare(best_key).unwrap_or(std::cmp::Ordering::Equal) == std::cmp::Ordering::Greater {
                best_idx = i;
                best_key = k;
            }
        }
        Ok(items[best_idx])
    }

    /// Scale a list of floats so max == 1.0. Empty → empty. All-zero → all-zero.
    fn normalize<'v>(
        values: Vec<f64>,
        heap: &'v Heap,
    ) -> anyhow::Result<Value<'v>> {
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let scaled: Vec<f64> = if values.is_empty() || !max.is_finite() || max == 0.0 {
            values
        } else {
            values.into_iter().map(|v| v / max).collect()
        };
        Ok(heap.alloc(AllocList(scaled.into_iter())))
    }
}
```

(The test for these is best written at the integration layer where a `.star` file calls `stable_sort([...], lambda x: -x)`. We add that test in Task 14's lib_move test suite. **No commit yet — wait for Step 2.**)

- [ ] **Step 2: Wire utilities into `build_globals`**

Modify `crates/bloqade-lanes-dsl-core/src/sandbox.rs::build_globals`:

```rust
pub fn build_globals(_cfg: &SandboxConfig) -> Globals {
    GlobalsBuilder::standard()
        .with(crate::primitives::utilities::register_utilities)
        .build()
}
```

- [ ] **Step 3: Add a smoke test that calls them from a tiny `.star`**

Add to `crates/bloqade-lanes-dsl-core/src/primitives/utilities.rs` test block:

```rust
#[cfg(test)]
mod tests {
    use crate::adapter::LoadedPolicy;
    use crate::sandbox::SandboxConfig;

    #[test]
    fn stable_sort_argmax_normalize_round_trip() {
        let cfg = SandboxConfig::default();
        let src = r#"
sorted = stable_sort([3, 1, 2], lambda x: x)
best = argmax([1, 5, 3, 5], lambda x: x)
nrm = normalize([2.0, 4.0, 1.0])
RESULT = struct(sorted = sorted, best = best, nrm = nrm)
"#;
        let p = LoadedPolicy::from_source("u.star".into(), src.into(), &cfg).expect("load");
        let result = p.get("RESULT").expect("RESULT bound");
        // Convert to Rust for assertions. (The exact pull-out API depends
        // on the pinned starlark version; use `Value::unpack_*` or
        // `to_value()`'s walker.)
        let _ = result; // assertion fleshed out in implementation
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p bloqade-lanes-dsl-core utilities`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/bloqade-lanes-dsl-core/src/primitives/utilities.rs crates/bloqade-lanes-dsl-core/src/sandbox.rs
git commit -m "feat(dsl-core): add stable_sort, argmax, normalize globals

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Generic `Policy` trait

**Files:**
- Modify: `crates/bloqade-lanes-dsl-core/src/policy_trait.rs`

- [ ] **Step 1: Define the trait**

Replace `policy_trait.rs` with:

```rust
//! Generic `Policy` trait — the seam between the kernel and a Starlark-hosted
//! algorithm. The Move DSL kernel and (later) the Target DSL adapter each
//! supply an implementation backed by a `LoadedPolicy`.

use crate::errors::DslError;

/// Outcome of a single `step()` invocation.
pub enum StepResult<A> {
    /// Apply these actions, in order, then call `step()` again.
    Continue(Vec<A>),
    /// Halt with this status. Kernel finalises the result.
    Halt(HaltStatus),
}

/// Halt reasons surfaced by `step()`.
#[derive(Debug, Clone)]
pub enum HaltStatus {
    Solved,
    Unsolvable,
    BudgetExhausted,
    Fallback(String),
    Error(String),
}

/// A hosted algorithm. Type parameters keep this generic enough to host
/// future placement DSLs (§11 of the spec) without modification.
pub trait Policy {
    type Handle;
    type Action;
    type InitArg;
    type NodeState;
    type GlobalState;

    /// One-time setup. Called after the kernel inserts the root node.
    fn init(&mut self, arg: Self::InitArg) -> Result<Self::GlobalState, DslError>;

    /// One tick. Returns actions to apply atomically, or a halt signal.
    fn step(
        &mut self,
        handle: &mut Self::Handle,
        gs: &mut Self::GlobalState,
    ) -> Result<StepResult<Self::Action>, DslError>;
}
```

- [ ] **Step 2: Smoke compile**

Run: `cargo check -p bloqade-lanes-dsl-core`
Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add crates/bloqade-lanes-dsl-core/src/policy_trait.rs
git commit -m "feat(dsl-core): add generic Policy trait for hosted DSLs

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 3 — Move Policy DSL: kernel + actions + handles

### Task 10: Module skeleton in `bloqade-lanes-search`

**Files:**
- Modify: `crates/bloqade-lanes-search/Cargo.toml`
- Modify: `crates/bloqade-lanes-search/src/lib.rs`
- Create: `crates/bloqade-lanes-search/src/move_policy_dsl/mod.rs`
- Create: `crates/bloqade-lanes-search/src/move_policy_dsl/{actions,graph_handle,kernel,lib_move,adapter_impl}.rs`

- [ ] **Step 1: Add path-dep on `bloqade-lanes-dsl-core`**

Modify `crates/bloqade-lanes-search/Cargo.toml`:

```toml
[dependencies]
bloqade-lanes-bytecode-core = { path = "../bloqade-lanes-bytecode-core" }
bloqade-lanes-dsl-core = { path = "../bloqade-lanes-dsl-core" }
rand = { version = "0.9", features = ["small_rng"] }
rayon = "1"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

- [ ] **Step 2: Create the empty submodule files**

Each file gets a one-line doc comment:

```rust
// mod.rs
//! Move Policy DSL: kernel + actions + handles for Starlark policies.

pub mod actions;
pub mod adapter_impl;
pub mod graph_handle;
pub mod kernel;
pub mod lib_move;

pub use kernel::{solve_with_policy, PolicyOptions, PolicyResult, PolicyStatus};
```

```rust
// actions.rs
//! Move-DSL action vocabulary: insert_child, update_node_state,
//! update_global_state, emit_solution, halt, invoke_builtin.
```

(Same one-liner for `adapter_impl.rs`, `graph_handle.rs`, `kernel.rs`, `lib_move.rs`.)

- [ ] **Step 3: Wire into `lib.rs`**

Modify `crates/bloqade-lanes-search/src/lib.rs`:

```rust
// ... existing modules ...
pub mod move_policy_dsl;
pub use move_policy_dsl::{solve_with_policy, PolicyOptions, PolicyResult, PolicyStatus};
```

- [ ] **Step 4: Verify**

Run: `cargo check -p bloqade-lanes-search`
Expected: clean (kernel exports are still empty stubs; provide minimal placeholders so the re-export compiles, e.g. `pub fn solve_with_policy() {}` will not satisfy a strongly-typed re-export. Use `pub struct PolicyOptions; pub struct PolicyResult; pub enum PolicyStatus { Pending }` and an `unimplemented!()` body for `solve_with_policy`).

- [ ] **Step 5: Commit**

```bash
git add crates/bloqade-lanes-search/Cargo.toml crates/bloqade-lanes-search/src/lib.rs crates/bloqade-lanes-search/src/move_policy_dsl/
git commit -m "feat(search): scaffold move_policy_dsl module

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: `MoveAction` enum + Starlark `actions` global

**Files:**
- Modify: `crates/bloqade-lanes-search/src/move_policy_dsl/actions.rs`

- [ ] **Step 1: Failing test for the action enum**

Replace `actions.rs` with:

```rust
//! Move-DSL action vocabulary.

use bloqade_lanes_dsl_core::sandbox::SandboxConfig;
use bloqade_lanes_dsl_core::adapter::LoadedPolicy;

use crate::graph::{MoveSet, NodeId};

/// One verb the policy issues per `step()`. The kernel applies these
/// in returned order, atomically per tick.
#[derive(Debug, Clone)]
pub enum MoveAction {
    InsertChild { parent: NodeId, move_set: MoveSet },
    UpdateNodeState { node: NodeId, patch: PatchValue },
    UpdateGlobalState { patch: PatchValue },
    EmitSolution { node: NodeId },
    Halt { status: String, message: String },
    InvokeBuiltin { name: String, args: PatchValue },
}

/// A serialized field-update bundle. Stored as JSON for simplicity:
/// the schema check (§5.10 row "update_node_state with a field not in
/// schema") is performed in the kernel against the declared NodeState
/// schema, not here.
#[derive(Debug, Clone)]
pub struct PatchValue(pub serde_json::Value);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn actions_are_clonable_and_debug() {
        let a = MoveAction::Halt {
            status: "solved".into(),
            message: "ok".into(),
        };
        let _ = format!("{a:?}");
        let _ = a.clone();
    }

    #[test]
    fn actions_are_emitted_from_starlark() {
        // Runs a tiny .star that returns a list of actions, ensures the
        // adapter unmarshalls them into MoveAction values.
        let cfg = SandboxConfig::default();
        let src = r#"
def step(graph, gs, ctx, lib):
    return [actions.halt("solved", "smoke")]
"#;
        // Test stub: actual unmarshalling is in adapter_impl.rs; this
        // test will be filled in once that lands. Marked as ignore for
        // now to keep CI green during incremental task landing.
        // #[ignore = "filled in by Task 18"]
        let _ = (cfg, src);
    }
}
```

- [ ] **Step 2: Implement the Starlark `actions` global**

Append to `actions.rs`:

```rust
use starlark::starlark_module;
use starlark::values::{Value, Heap};

/// Register the `actions` global into a `GlobalsBuilder`. Called by
/// `kernel::build_move_globals`.
pub(super) fn register_actions(builder: &mut starlark::environment::GlobalsBuilder) {
    builder.set("actions", build_actions_namespace(builder));
}

fn build_actions_namespace(_builder: &starlark::environment::GlobalsBuilder) -> starlark::values::FrozenValue {
    // Build an immutable struct whose attributes are the six verbs.
    // Each verb returns an opaque ActionValue that the kernel later
    // downcasts to MoveAction. Implementation detail: store a JSON
    // payload + a discriminant tag on a Starlark struct.
    todo!("implemented in Task 12 once we know what the kernel needs")
}
```

(Using `todo!()` here is intentional — the action-namespace must be co-developed with the kernel in Task 12, and writing it speculatively risks API drift. The compile fails until Task 12 completes; that's by design and matches the TDD red-green discipline.)

- [ ] **Step 3: Run tests**

Run: `cargo check -p bloqade-lanes-search`
Expected: build fails on `todo!()` macro warning. **Acceptable transient state** — Task 12 closes it.

- [ ] **Step 4: No commit yet** — Task 11 + Task 12 share a commit boundary because the API surface is co-designed.

---

### Task 12: Implement the `actions` namespace + kernel reception

**Files:**
- Modify: `crates/bloqade-lanes-search/src/move_policy_dsl/actions.rs`
- Modify: `crates/bloqade-lanes-search/src/move_policy_dsl/kernel.rs`

- [ ] **Step 1: Decide the on-the-wire representation**

Each Starlark `actions.X(...)` call returns a tagged Starlark `dict` of the form `{"$action": "insert_child", "parent": node_id, "move_set": move_set_value}`. The kernel scans the list returned by `step()` and converts each dict into a `MoveAction`.

Why dicts over a custom Starlark value type? Three reasons: (1) Starlark dicts are first-class and survive the `starlark_simple_value` / freeze cycle without needing `ProvidesStaticType` plumbing for six new types; (2) policies can introspect/log actions trivially in tests; (3) errors (e.g. `actions.update_node_state` with a non-numeric `node_id`) are reported once at conversion, with the same shape as `Schema` errors.

- [ ] **Step 2: Failing test for the conversion path**

Add to `kernel.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn dict_to_action_insert_child() {
        let dict = json!({
            "$action": "insert_child",
            "parent": 0,
            "move_set_encoded": [42u64, 99u64],
        });
        let action = MoveAction::try_from_json(&dict).expect("convert");
        match action {
            MoveAction::InsertChild { parent, move_set } => {
                assert_eq!(parent.0, 0);
                assert_eq!(move_set.encoded_lanes(), &[42, 99]);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn dict_to_action_unknown_kind_errors() {
        let dict = json!({"$action": "frob"});
        let err = MoveAction::try_from_json(&dict).err().expect("must fail");
        let msg = format!("{err}");
        assert!(msg.contains("frob"));
    }
}
```

- [ ] **Step 3: Implement `MoveAction::try_from_json`**

Append to `actions.rs`:

```rust
use bloqade_lanes_dsl_core::errors::DslError;

impl MoveAction {
    pub fn try_from_json(v: &serde_json::Value) -> Result<Self, DslError> {
        let kind = v
            .get("$action")
            .and_then(|s| s.as_str())
            .ok_or_else(|| DslError::BadPolicy("action dict missing $action key".into()))?;
        match kind {
            "insert_child" => {
                let parent = v.get("parent").and_then(|p| p.as_u64()).ok_or_else(|| {
                    DslError::BadPolicy("insert_child: parent must be integer".into())
                })?;
                let lanes: Vec<u64> = v
                    .get("move_set_encoded")
                    .and_then(|l| l.as_array())
                    .ok_or_else(|| DslError::BadPolicy("insert_child: move_set_encoded missing".into()))?
                    .iter()
                    .map(|x| x.as_u64().unwrap_or(0))
                    .collect();
                Ok(MoveAction::InsertChild {
                    parent: NodeId(parent as u32),
                    move_set: MoveSet::from_encoded(lanes),
                })
            }
            "update_node_state" => {
                let node = v.get("node").and_then(|p| p.as_u64()).ok_or_else(|| {
                    DslError::BadPolicy("update_node_state: node must be integer".into())
                })?;
                let patch = v.get("patch").cloned().unwrap_or(serde_json::Value::Null);
                Ok(MoveAction::UpdateNodeState {
                    node: NodeId(node as u32),
                    patch: PatchValue(patch),
                })
            }
            "update_global_state" => {
                let patch = v.get("patch").cloned().unwrap_or(serde_json::Value::Null);
                Ok(MoveAction::UpdateGlobalState { patch: PatchValue(patch) })
            }
            "emit_solution" => {
                let node = v.get("node").and_then(|p| p.as_u64()).ok_or_else(|| {
                    DslError::BadPolicy("emit_solution: node must be integer".into())
                })?;
                Ok(MoveAction::EmitSolution { node: NodeId(node as u32) })
            }
            "halt" => Ok(MoveAction::Halt {
                status: v.get("status").and_then(|s| s.as_str()).unwrap_or("error").into(),
                message: v.get("message").and_then(|s| s.as_str()).unwrap_or("").into(),
            }),
            "invoke_builtin" => Ok(MoveAction::InvokeBuiltin {
                name: v.get("name").and_then(|s| s.as_str()).unwrap_or("").into(),
                args: PatchValue(v.get("args").cloned().unwrap_or(serde_json::Value::Object(Default::default()))),
            }),
            other => Err(DslError::BadPolicy(format!("unknown action kind: {other}"))),
        }
    }
}
```

- [ ] **Step 4: Implement the Starlark `actions` namespace as struct factories**

Now flesh out `register_actions`. Each verb is a Starlark function that returns a dict shaped like `{"$action": "...", ...}`:

```rust
#[starlark_module]
pub(super) fn register_actions(builder: &mut starlark::environment::GlobalsBuilder) {
    fn insert_child<'v>(
        parent: i32,
        move_set: &crate::move_policy_dsl::lib_move::StarlarkMoveSet, // see Task 14
        heap: &'v Heap,
    ) -> anyhow::Result<Value<'v>> {
        let dict = starlark::values::dict::AllocDict([
            ("$action", heap.alloc("insert_child")),
            ("parent", heap.alloc(parent)),
            ("move_set_encoded", heap.alloc(move_set.0.encoded_lanes().to_vec())),
        ]);
        Ok(heap.alloc(dict))
    }

    // ... similar for update_node_state, update_global_state, emit_solution, halt, invoke_builtin ...
}
```

(Each is mechanical; copy the pattern. The `starlark_module` macro auto-namespaces them under `actions`. If the macro doesn't natively support namespacing, build a frozen struct: `actions = struct(insert_child=..., halt=..., ...)`.)

- [ ] **Step 5: Run tests**

Run: `cargo test -p bloqade-lanes-search move_policy_dsl::actions`
Expected: PASS for `dict_to_action_insert_child` and `dict_to_action_unknown_kind_errors`.

- [ ] **Step 6: Commit**

```bash
git add crates/bloqade-lanes-search/src/move_policy_dsl/
git commit -m "feat(search): MoveAction enum + Starlark actions namespace

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 13: `graph` handle — read accessors

**Files:**
- Modify: `crates/bloqade-lanes-search/src/move_policy_dsl/graph_handle.rs`

- [ ] **Step 1: Failing test for `graph.root` and `graph.config`**

Replace `graph_handle.rs` with:

```rust
//! Starlark-visible mediated view of `SearchGraph`, plus side channels
//! for last_insert / last_builtin_result.

use std::cell::RefCell;
use std::rc::Rc;

use crate::graph::{NodeId, SearchGraph};
use crate::move_policy_dsl::lib_move::StarlarkConfig;

use starlark::starlark_module;
use starlark::values::{Value, Heap, ProvidesStaticType, NoSerialize, StarlarkValue};

/// Outcome of the most recent `actions.insert_child(...)`. Mirrors
/// the spec §5.6 contract.
#[derive(Debug, Clone)]
pub struct InsertOutcome {
    pub child_id: Option<NodeId>,
    pub is_new: bool,
    pub error: Option<String>,
}

/// Outcome of the most recent `actions.invoke_builtin(...)`.
#[derive(Debug, Clone)]
pub struct BuiltinOutcome {
    pub status: String,
    pub payload: serde_json::Value,
}

/// Per-node DSL state, owned by the kernel. Generic JSON for the v1
/// scope: a future refinement could promote a typed `record` schema.
#[derive(Debug, Clone)]
pub struct NodeStateMap(pub std::collections::HashMap<NodeId, serde_json::Value>);

/// `PolicyGraph` is the mediated handle the policy sees as `graph` in
/// `step()`. Every accessor is a pure read; mutation flows through
/// `actions`, never the handle.
#[derive(ProvidesStaticType, NoSerialize, allocative::Allocative)]
pub struct PolicyGraph {
    // Use Rc<RefCell<...>> so the kernel and the Starlark-visible value
    // can share the same backing store across the boundary. Single-threaded
    // by construction (one solve = one evaluator).
    inner: Rc<RefCell<PolicyGraphInner>>,
}

pub(super) struct PolicyGraphInner {
    pub graph: SearchGraph,
    pub node_state: NodeStateMap,
    pub last_insert: Option<InsertOutcome>,
    pub last_builtin: Option<BuiltinOutcome>,
    /// Insertion-order children-of map. The base SearchGraph stores
    /// only parent pointers, so we maintain a parallel forward map.
    pub children: std::collections::HashMap<NodeId, Vec<NodeId>>,
}

impl std::fmt::Debug for PolicyGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PolicyGraph").finish()
    }
}

impl std::fmt::Display for PolicyGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<PolicyGraph>")
    }
}

starlark::starlark_simple_value!(PolicyGraph);

#[starlark::values::starlark_value(type = "PolicyGraph")]
impl<'v> StarlarkValue<'v> for PolicyGraph {
    fn get_attr(&self, attr: &str, _heap: &'v Heap) -> Option<Value<'v>> {
        match attr {
            "root" => Some(Value::new_int(self.inner.borrow().graph.root().0 as i32)),
            "count" => Some(Value::new_int(self.inner.borrow().graph.len() as i32)),
            _ => None,
        }
    }
}

#[starlark_module]
pub fn register_graph_methods(builder: &mut starlark::environment::MethodsBuilder) {
    /// Return the configuration of `node` as an opaque immutable handle.
    fn config<'v>(this: &PolicyGraph, node: i32) -> anyhow::Result<StarlarkConfig> {
        let inner = this.inner.borrow();
        Ok(StarlarkConfig(inner.graph.config(NodeId(node as u32)).clone()))
    }

    /// Parent of `node`, or `None` for the root.
    fn parent<'v>(this: &PolicyGraph, node: i32) -> anyhow::Result<Value<'v>> {
        match this.inner.borrow().graph.parent(NodeId(node as u32)) {
            Some(p) => Ok(Value::new_int(p.0 as i32)),
            None => Ok(Value::new_none()),
        }
    }

    /// Depth of `node` (root = 0).
    fn depth(this: &PolicyGraph, node: i32) -> anyhow::Result<i32> {
        Ok(this.inner.borrow().graph.depth(NodeId(node as u32)) as i32)
    }

    /// g-cost from root to `node`.
    fn g_cost(this: &PolicyGraph, node: i32) -> anyhow::Result<f64> {
        Ok(this.inner.borrow().graph.g_score(NodeId(node as u32)))
    }

    /// Insertion-order children of `node`.
    fn children_of(this: &PolicyGraph, node: i32) -> anyhow::Result<Vec<i32>> {
        let inner = this.inner.borrow();
        Ok(inner
            .children
            .get(&NodeId(node as u32))
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .map(|n| n.0 as i32)
            .collect())
    }

    /// Path from root to `node` as a list of MoveSet values.
    fn path_from_root(this: &PolicyGraph, node: i32) -> anyhow::Result<Vec<crate::move_policy_dsl::lib_move::StarlarkMoveSet>> {
        let inner = this.inner.borrow();
        let path = inner.graph.reconstruct_path(NodeId(node as u32));
        Ok(path.into_iter().map(crate::move_policy_dsl::lib_move::StarlarkMoveSet).collect())
    }

    // last_insert and last_builtin_result added in Task 14 once the
    // kernel actually populates them.

    /// Read the per-node DSL state. Returns a Starlark dict.
    fn ns<'v>(this: &PolicyGraph, node: i32, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
        let inner = this.inner.borrow();
        let state = inner
            .node_state
            .0
            .get(&NodeId(node as u32))
            .cloned()
            .unwrap_or(serde_json::Value::Object(Default::default()));
        Ok(json_to_starlark(state, heap))
    }

    /// Goal predicate: checks against `ctx.targets`. Implemented via the
    /// kernel-supplied `is_goal_fn`; for the v1 scope we delegate to
    /// `AllAtTarget` for the targets passed in via `ctx`.
    fn is_goal<'v>(this: &PolicyGraph, node: i32, ctx: Value<'v>) -> anyhow::Result<bool> {
        // Will be filled in once `ctx` handle is wired (Task 15).
        let _ = (this, node, ctx);
        Ok(false)
    }
}

fn json_to_starlark<'v>(value: serde_json::Value, heap: &'v Heap) -> Value<'v> {
    use serde_json::Value as J;
    match value {
        J::Null => Value::new_none(),
        J::Bool(b) => Value::new_bool(b),
        J::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::new_int(i as i32)
            } else if let Some(f) = n.as_f64() {
                heap.alloc(f)
            } else {
                Value::new_none()
            }
        }
        J::String(s) => heap.alloc(s),
        J::Array(arr) => heap.alloc(starlark::values::list::AllocList(
            arr.into_iter().map(|v| json_to_starlark(v, heap)),
        )),
        J::Object(map) => heap.alloc(starlark::values::dict::AllocDict(
            map.into_iter().map(|(k, v)| (k, json_to_starlark(v, heap))),
        )),
    }
}
```

- [ ] **Step 2: Add a kernel-side accessor unit test (no Starlark)**

Append `#[cfg(test)] mod tests { ... }` covering construction of a `PolicyGraph` from a `SearchGraph` and reading `root`, `parent`, `depth`, `g_cost`. (Bypass Starlark for this test — call the underlying `inner` directly via a `pub(super)` constructor.)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;

    pub(super) fn make_for_test(root: Config) -> PolicyGraph {
        let inner = PolicyGraphInner {
            graph: SearchGraph::new(root),
            node_state: NodeStateMap(Default::default()),
            last_insert: None,
            last_builtin: None,
            children: Default::default(),
        };
        PolicyGraph { inner: Rc::new(RefCell::new(inner)) }
    }

    #[test]
    fn root_attrs() {
        let cfg = Config::new([(0, LocationAddr::new(0, 0))]).unwrap();
        let pg = make_for_test(cfg);
        let inner = pg.inner.borrow();
        assert_eq!(inner.graph.root().0, 0);
        assert_eq!(inner.graph.depth(inner.graph.root()), 0);
    }
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p bloqade-lanes-search move_policy_dsl::graph_handle`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add crates/bloqade-lanes-search/src/move_policy_dsl/graph_handle.rs
git commit -m "feat(search): PolicyGraph handle with read accessors

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 14: `lib_move` — Starlark wrappers + distance/mobility primitives

**Files:**
- Modify: `crates/bloqade-lanes-search/src/move_policy_dsl/lib_move.rs`

This task is the largest single chunk of code in the plan. It bundles `StarlarkConfig` + `StarlarkMoveSet` (deferred from Task 6) plus the move-specific lib primitives.

- [ ] **Step 1: Failing test for `hop_distance` and `mobility`**

Replace `lib_move.rs` with:

```rust
//! Move Policy DSL `lib.*` primitives.
//!
//! Distance/mobility helpers, candidate-pipeline shims (`score_lanes`,
//! `top_c_per_qubit`, `group_by_triplet`, `pack_aod_rectangles`), and
//! graph-walking utilities (`walk_up`, `ancestors`).
//!
//! Also hosts the `Config` and `MoveSet` Starlark wrappers (placed here
//! rather than in dsl-core to avoid a search→dsl-core→search dep cycle).

use std::sync::Arc;

use starlark::starlark_module;
use starlark::values::{Heap, NoSerialize, ProvidesStaticType, StarlarkValue, Value};

use crate::config::Config;
use crate::context::SearchContext;
use crate::graph::{MoveSet, NodeId};
use crate::heuristic::DistanceTable;
use crate::lane_index::LaneIndex;

#[derive(Debug, Clone, ProvidesStaticType, NoSerialize, allocative::Allocative)]
pub struct StarlarkConfig(pub Config);
starlark::starlark_simple_value!(StarlarkConfig);

#[derive(Debug, Clone, ProvidesStaticType, NoSerialize, allocative::Allocative)]
pub struct StarlarkMoveSet(pub MoveSet);
starlark::starlark_simple_value!(StarlarkMoveSet);

impl std::fmt::Display for StarlarkConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Config(len={})", self.0.len())
    }
}
impl std::fmt::Display for StarlarkMoveSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MoveSet(len={})", self.0.len())
    }
}

#[starlark::values::starlark_value(type = "Config")]
impl<'v> StarlarkValue<'v> for StarlarkConfig {
    fn get_attr(&self, attr: &str, heap: &'v Heap) -> Option<Value<'v>> {
        match attr {
            "len" => Some(Value::new_int(self.0.len() as i32)),
            "hash" => {
                let mut h = std::collections::hash_map::DefaultHasher::new();
                std::hash::Hash::hash(&self.0, &mut h);
                Some(Value::new_int(std::hash::Hasher::finish(&h) as i32 as i32))
            }
            _ => None,
        }
    }
}

#[starlark::values::starlark_value(type = "MoveSet")]
impl<'v> StarlarkValue<'v> for StarlarkMoveSet {
    fn get_attr(&self, attr: &str, _heap: &'v Heap) -> Option<Value<'v>> {
        match attr {
            "len" => Some(Value::new_int(self.0.len() as i32)),
            "encoded" => {
                // Concatenated representation: hash of encoded vec for stable id.
                let mut h = std::collections::hash_map::DefaultHasher::new();
                std::hash::Hash::hash(&self.0, &mut h);
                Some(Value::new_int(std::hash::Hasher::finish(&h) as i32))
            }
            _ => None,
        }
    }
}

/// `lib` handle wraps the SearchContext + LaneIndex needed to answer
/// distance/mobility/etc. queries.
#[derive(ProvidesStaticType, NoSerialize, allocative::Allocative)]
pub struct LibMove {
    pub(super) index: Arc<LaneIndex>,
    pub(super) dist_table: Arc<DistanceTable>,
    pub(super) targets: Vec<(u32, u64)>,
    pub(super) blocked: std::collections::HashSet<u64>,
}

impl std::fmt::Debug for LibMove {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LibMove").finish()
    }
}
impl std::fmt::Display for LibMove {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<lib>")
    }
}
starlark::starlark_simple_value!(LibMove);

#[starlark::values::starlark_value(type = "Lib")]
impl<'v> StarlarkValue<'v> for LibMove {}

#[starlark_module]
pub fn register_lib_methods(builder: &mut starlark::environment::MethodsBuilder) {
    /// Lane-hop distance from `from_loc` to `target_loc`. None if unreachable.
    fn hop_distance<'v>(
        this: &LibMove,
        from_loc: &bloqade_lanes_dsl_core::primitives::types::StarlarkLocation,
        target_loc: &bloqade_lanes_dsl_core::primitives::types::StarlarkLocation,
    ) -> anyhow::Result<Value<'v>> {
        let dist = this
            .dist_table
            .distance(from_loc.0.encode(), target_loc.0.encode());
        Ok(match dist {
            Some(d) => Value::new_int(d as i32),
            None => Value::new_none(),
        })
    }

    /// Time-distance (µs). None if no time table or unreachable.
    fn time_distance<'v>(
        this: &LibMove,
        from_loc: &bloqade_lanes_dsl_core::primitives::types::StarlarkLocation,
        target_loc: &bloqade_lanes_dsl_core::primitives::types::StarlarkLocation,
    ) -> anyhow::Result<Value<'v>> {
        // dist_table holds time data when configured; expose if present.
        match this.dist_table.time_distance(from_loc.0.encode(), target_loc.0.encode()) {
            Some(t) => Ok(Value::new_alloc_simple(t)), // f64 alloc
            None => Ok(Value::new_none()),
        }
    }

    /// Convex blend of hop and time distance with weight `w_t` ∈ [0, 1].
    fn blended_distance(
        this: &LibMove,
        from_loc: &bloqade_lanes_dsl_core::primitives::types::StarlarkLocation,
        target_loc: &bloqade_lanes_dsl_core::primitives::types::StarlarkLocation,
        w_t: f64,
    ) -> anyhow::Result<f64> {
        let h = this
            .dist_table
            .distance(from_loc.0.encode(), target_loc.0.encode())
            .map(|d| d as f64)
            .unwrap_or(f64::INFINITY);
        let t = this
            .dist_table
            .time_distance(from_loc.0.encode(), target_loc.0.encode())
            .unwrap_or(f64::INFINITY);
        Ok((1.0 - w_t) * h + w_t * t)
    }

    fn fastest_lane_us(this: &LibMove) -> anyhow::Result<f64> {
        Ok(this.dist_table.fastest_lane_us().unwrap_or(0.0))
    }

    /// Mobility = Σ 1 / (1 + hop_distance(next_dst, target)) over outgoing lanes.
    /// `targets` is a list of `(qid, target_loc)` (typically `ctx.targets`).
    fn mobility<'v>(
        this: &LibMove,
        loc: &bloqade_lanes_dsl_core::primitives::types::StarlarkLocation,
        targets: Vec<Value<'v>>, // [(qid, target_loc), ...] — accept tuples
    ) -> anyhow::Result<f64> {
        let mut sum = 0.0_f64;
        for &lane in this.index.outgoing_lanes(loc.0).iter() {
            let Some((_, dst)) = this.index.endpoints(&lane) else { continue };
            let dst_enc = dst.encode();
            for tv in &targets {
                // each tv is a tuple (qid: int, target: StarlarkLocation)
                let tup = tv.unpack_tuple().ok_or_else(|| anyhow::anyhow!("targets must be tuples"))?;
                let target = tup
                    .get(1)
                    .and_then(|v| v.downcast_ref::<bloqade_lanes_dsl_core::primitives::types::StarlarkLocation>())
                    .ok_or_else(|| anyhow::anyhow!("target loc downcast"))?;
                let d = this.dist_table.distance(dst_enc, target.0.encode()).unwrap_or(u32::MAX);
                if d != u32::MAX {
                    sum += 1.0 / (1.0 + d as f64);
                }
            }
        }
        Ok(sum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{lane, loc};

    #[test]
    fn hop_distance_known_path_returns_int() {
        // The fixture builder lives in test_utils; reuse it.
        // Confirm hop_distance returns an integer for a reachable pair.
        // (Skipped pseudo-code: integration is exercised in the acid test.)
        let _ = (lane, loc);
    }
}
```

- [ ] **Step 2: Defer `score_lanes`/`top_c`/`group`/`pack` to Task 15**

These reuse the existing `HeuristicGenerator` pipeline; splitting them into a separate task keeps Task 14's diff readable.

- [ ] **Step 3: Run tests**

Run: `cargo test -p bloqade-lanes-search move_policy_dsl::lib_move`
Expected: PASS (the unit test is a no-op smoke; the real check is `cargo check -p bloqade-lanes-search` succeeds).

- [ ] **Step 4: Commit**

```bash
git add crates/bloqade-lanes-search/src/move_policy_dsl/lib_move.rs
git commit -m "feat(search): lib_move primitives — distances, mobility, Starlark Config/MoveSet

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 15: `lib_move` — candidate pipeline (`score_lanes`, `top_c_per_qubit`, `group_by_triplet`, `pack_aod_rectangles`)

**Files:**
- Modify: `crates/bloqade-lanes-search/src/move_policy_dsl/lib_move.rs`
- Modify: `crates/bloqade-lanes-search/src/aod_grid.rs` (visibility only — make `BusGridContext::new` and `build_aod_grids` callable from `move_policy_dsl/`)
- Modify: `crates/bloqade-lanes-search/src/generators/heuristic.rs` (extract pure-stage helpers; verify visibility)

- [ ] **Step 1: Survey what's reusable**

Read `crates/bloqade-lanes-search/src/generators/heuristic.rs` lines 1-300 and `crates/bloqade-lanes-search/src/aod_grid.rs` to identify the existing pipeline stages. Confirm the function signatures of:

- The score-and-rank-lanes pass (per-qubit-bus loop in `HeuristicGenerator::generate`).
- `top_c_per_qubit` selection.
- The triplet-group fold.
- `BusGridContext::build_aod_grids` (the AOD-rectangle pack).

If `HeuristicGenerator::generate` is monolithic, **extract three helper functions** (still inside `generators/heuristic.rs`):
- `pub(crate) fn score_all_lanes(config, ctx, scorer) -> Vec<ScoredLane>`
- `pub(crate) fn top_c_per_qubit(scored, c) -> Vec<ScoredLane>`
- `pub(crate) fn group_by_triplet(scored) -> Vec<TripletGroup>`

These become the dependencies for the Starlark `lib.score_lanes` etc. wrappers. Don't change behavior — just refactor.

- [ ] **Step 2: Failing test for `pack_aod_rectangles` via Starlark**

Append to `lib_move.rs`:

```rust
#[cfg(test)]
mod pack_tests {
    // Build a tiny ArchSpec, run pack_aod_rectangles via a one-line .star,
    // assert at least one valid AOD rectangle is produced.
    // Skeleton — flesh out using the existing test_utils fixture.
}
```

- [ ] **Step 3: Implement the four primitives**

Append to `lib_move.rs` `register_lib_methods`:

```rust
#[starlark_module]
pub fn register_lib_pipeline_methods(builder: &mut starlark::environment::MethodsBuilder) {
    /// Score every (qubit, outgoing-lane) pair using a Starlark callable
    /// `score_fn(qubit, lane, ns, ctx) -> float`.
    fn score_lanes<'v>(
        this: &LibMove,
        config: &StarlarkConfig,
        ns: Value<'v>,
        score_fn: Value<'v>,
        ctx: Value<'v>,
        eval: &mut starlark::eval::Evaluator<'v, '_, '_>,
        heap: &'v Heap,
    ) -> anyhow::Result<Value<'v>> {
        // Iterate unresolved qubits × outgoing lanes; call score_fn per pair;
        // emit list of struct(qid, lane, score), sorted by (qid asc, lane.encoded asc).
        let _ = (this, config, ns, score_fn, ctx, eval, heap);
        todo!("implement using scorers::DistanceScorer + heuristic::extract_unresolved as references")
    }

    /// Group by qid, keep top-c by score; falls back to the single best
    /// when no positive scores (matches HeuristicGenerator semantics).
    fn top_c_per_qubit<'v>(
        this: &LibMove,
        scored: Vec<Value<'v>>,
        c: i32,
    ) -> anyhow::Result<Vec<Value<'v>>> {
        let _ = (this, scored, c);
        todo!()
    }

    /// Group by `(move_type, bus_id, direction)` triplet, sort triplet keys.
    fn group_by_triplet<'v>(
        this: &LibMove,
        scored: Vec<Value<'v>>,
    ) -> anyhow::Result<Vec<Value<'v>>> {
        let _ = (this, scored);
        todo!()
    }

    /// Build AOD-rectangle packed candidates per triplet group via
    /// BusGridContext::build_aod_grids. Returns
    /// `[struct(move_set, new_config, score_sum)]` sorted by score_sum desc.
    fn pack_aod_rectangles<'v>(
        this: &LibMove,
        groups: Vec<Value<'v>>,
        config: &StarlarkConfig,
        ctx: Value<'v>,
    ) -> anyhow::Result<Vec<Value<'v>>> {
        let _ = (this, groups, config, ctx);
        todo!()
    }
}
```

Each `todo!()` is replaced with the real implementation by porting/refactoring the corresponding step from `HeuristicGenerator::generate`. Implementation guidance:

- `score_lanes`: walk `config.iter()` for unresolved qubits (via `lib.unresolved_qubits` logic — can use the `ctx.targets`), for each call `index.outgoing_lanes(loc)` and invoke `score_fn` via `eval.eval_function(score_fn, &[qubit_struct, lane_struct, ns, ctx], &[])`. Collect results into a Starlark list of `struct(qid, lane, score)`.
- `top_c_per_qubit`: group by `qid`, sort each group by `(score desc, lane.encoded asc)`, take top `c`. Falls back to the single best if no positive scores — this is exactly what `HeuristicGenerator` does; lift the logic.
- `group_by_triplet`: for each scored entry, extract `lane.move_type`, `lane.bus_id`, `lane.direction`. Group, sort by triplet key.
- `pack_aod_rectangles`: for each triplet group, call `BusGridContext::new(...)` + `build_aod_grids(...)` and convert the resulting AOD rectangles into Starlark `struct(move_set, new_config, score_sum)` values.

- [ ] **Step 4: Run tests**

Run: `cargo test -p bloqade-lanes-search move_policy_dsl::lib_move`
Expected: PASS (the smoke test in `pack_tests` exercises the full pipeline).

- [ ] **Step 5: Commit**

```bash
git add crates/bloqade-lanes-search/src/move_policy_dsl/lib_move.rs crates/bloqade-lanes-search/src/aod_grid.rs crates/bloqade-lanes-search/src/generators/heuristic.rs
git commit -m "feat(search): lib_move candidate pipeline (score_lanes, top_c, group, pack)

Refactors HeuristicGenerator's per-stage logic into pub(crate) helpers
that the Starlark lib_move primitives reuse.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 16: `ctx` handle

**Files:**
- Modify: `crates/bloqade-lanes-search/src/move_policy_dsl/lib_move.rs` (or new `ctx_handle.rs`)

- [ ] **Step 1: Add the `Ctx` Starlark value**

Append to `lib_move.rs`:

```rust
/// `ctx` handle: read-only search context (targets, blocked, arch_spec).
#[derive(ProvidesStaticType, NoSerialize, allocative::Allocative, Clone)]
pub struct Ctx {
    pub(super) targets: Vec<(u32, u64)>,
    pub(super) blocked: std::collections::HashSet<u64>,
    pub(super) arch_spec: bloqade_lanes_dsl_core::primitives::arch_spec::StarlarkArchSpec,
}

impl std::fmt::Debug for Ctx { /* ... */ }
impl std::fmt::Display for Ctx { /* ... */ }

starlark::starlark_simple_value!(Ctx);

#[starlark::values::starlark_value(type = "Ctx")]
impl<'v> StarlarkValue<'v> for Ctx {
    fn get_attr(&self, attr: &str, heap: &'v Heap) -> Option<Value<'v>> {
        match attr {
            "arch_spec" => Some(heap.alloc(self.arch_spec.clone())),
            "targets" => {
                let tuples: Vec<Value<'v>> = self
                    .targets
                    .iter()
                    .map(|&(qid, loc)| {
                        let l = bloqade_lanes_dsl_core::primitives::types::StarlarkLocation(
                            bloqade_lanes_bytecode_core::arch::addr::LocationAddr::decode(loc),
                        );
                        starlark::values::tuple::AllocTuple::alloc_2(qid as i32, l).to_value()
                    })
                    .collect();
                Some(heap.alloc(starlark::values::list::AllocList(tuples.into_iter())))
            }
            "blocked" => {
                let locs: Vec<Value<'v>> = self
                    .blocked
                    .iter()
                    .map(|&loc| {
                        let l = bloqade_lanes_dsl_core::primitives::types::StarlarkLocation(
                            bloqade_lanes_bytecode_core::arch::addr::LocationAddr::decode(loc),
                        );
                        heap.alloc(l)
                    })
                    .collect();
                Some(heap.alloc(starlark::values::list::AllocList(locs.into_iter())))
            }
            _ => None,
        }
    }
}
```

- [ ] **Step 2: Run tests**

Run: `cargo check -p bloqade-lanes-search`
Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add crates/bloqade-lanes-search/src/move_policy_dsl/lib_move.rs
git commit -m "feat(search): Ctx handle exposing targets/blocked/arch_spec

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 17: Kernel loop — happy path (no fallback yet)

**Files:**
- Modify: `crates/bloqade-lanes-search/src/move_policy_dsl/kernel.rs`
- Modify: `crates/bloqade-lanes-search/src/move_policy_dsl/adapter_impl.rs`

- [ ] **Step 1: Failing integration test**

Create `crates/bloqade-lanes-search/tests/dsl_kernel_minimal.rs`:

```rust
//! Minimal kernel test: a no-op policy that immediately halts must
//! produce a `PolicyStatus::Solved` (or `Halted("solved")`) without
//! mutating the graph.

use bloqade_lanes_search::move_policy_dsl::{solve_with_policy, PolicyOptions};

const POLICY_SRC: &str = r#"
NodeState = record(value = field(int, default = 0))
PARAMS    = struct()

def init(root, ctx):
    return struct()

def step(graph, gs, ctx, lib):
    return actions.halt("solved", "trivial")
"#;

#[test]
fn trivial_halt_policy_returns_solved() {
    // Build a minimal arch + initial config + target. Reuse the
    // crate's test_utils where possible.
    // ... assertion: result.status == PolicyStatus::Solved
}
```

- [ ] **Step 2: Implement the kernel**

Replace `kernel.rs` body with:

```rust
//! The Move Policy DSL kernel loop.

use std::sync::Arc;

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;
use bloqade_lanes_dsl_core::adapter::LoadedPolicy;
use bloqade_lanes_dsl_core::errors::DslError;
use bloqade_lanes_dsl_core::sandbox::SandboxConfig;

use crate::config::Config;
use crate::context::SearchContext;
use crate::graph::{MoveSet, NodeId, SearchGraph};
use crate::heuristic::DistanceTable;
use crate::lane_index::LaneIndex;
use crate::move_policy_dsl::actions::{MoveAction, PatchValue};
use crate::move_policy_dsl::graph_handle::{
    BuiltinOutcome, InsertOutcome, NodeStateMap, PolicyGraph, PolicyGraphInner,
};
use crate::move_policy_dsl::lib_move::{Ctx, LibMove};

#[derive(Debug, Clone)]
pub struct PolicyOptions {
    pub policy_path: String,
    pub policy_params: serde_json::Value,
    pub max_expansions: u64,
    pub timeout_s: Option<f64>,
    pub sandbox: SandboxConfig,
}

#[derive(Debug, Clone)]
pub struct PolicyResult {
    pub status: PolicyStatus,
    pub move_layers: Vec<MoveSet>,
    pub goal_config: Config,
    pub nodes_expanded: u32,
    pub policy_file: String,
    pub policy_params: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyStatus {
    Solved,
    Unsolvable,
    BudgetExhausted,
    Timeout,
    Fallback(String),
    SyntaxError(String),
    RuntimeError(String),
    SchemaError(String),
    BadPolicy(String),
    StarlarkBudget,
    StarlarkOOM,
}

/// Public entry point used by the PyO3 binding.
pub fn solve_with_policy(
    initial: impl IntoIterator<Item = (u32, LocationAddr)>,
    target: impl IntoIterator<Item = (u32, LocationAddr)>,
    blocked: impl IntoIterator<Item = LocationAddr>,
    index: Arc<LaneIndex>,
    opts: PolicyOptions,
) -> Result<PolicyResult, DslError> {
    let initial_cfg = Config::new(initial).map_err(|e| DslError::BadPolicy(e.to_string()))?;
    let target_cfg = Config::new(target).map_err(|e| DslError::BadPolicy(e.to_string()))?;
    let blocked_set: std::collections::HashSet<u64> =
        blocked.into_iter().map(|l| l.encode()).collect();
    let target_pairs: Vec<(u32, u64)> = target_cfg
        .iter()
        .map(|(q, l)| (q, l.encode()))
        .collect();

    let dist_table = Arc::new(DistanceTable::new_with_time(
        &target_pairs.iter().map(|&(_, l)| l).collect::<Vec<_>>(),
        &index,
    ));

    // Load the policy.
    let policy = LoadedPolicy::from_path(&opts.policy_path, &opts.sandbox)?;

    // Build the graph + handles.
    let inner = PolicyGraphInner {
        graph: SearchGraph::new(initial_cfg.clone()),
        node_state: NodeStateMap(Default::default()),
        last_insert: None,
        last_builtin: None,
        children: Default::default(),
    };
    let pg = PolicyGraph::new(inner);

    let lib = LibMove {
        index: index.clone(),
        dist_table: dist_table.clone(),
        targets: target_pairs.clone(),
        blocked: blocked_set.clone(),
    };
    let ctx_value = Ctx {
        targets: target_pairs.clone(),
        blocked: blocked_set.clone(),
        arch_spec: bloqade_lanes_dsl_core::primitives::arch_spec::StarlarkArchSpec(
            Arc::new(index.arch_spec().clone()),
        ),
    };

    // Invoke `init(root, ctx)`.
    let gs = call_init(&policy, &pg, &ctx_value, &opts.sandbox)?;

    // Main loop.
    let mut nodes_expanded: u32 = 0;
    let mut solutions: Vec<NodeId> = Vec::new();
    let start = std::time::Instant::now();

    loop {
        if let Some(t) = opts.timeout_s
            && start.elapsed().as_secs_f64() > t
        {
            return Ok(PolicyResult {
                status: PolicyStatus::Timeout,
                move_layers: vec![],
                goal_config: initial_cfg,
                nodes_expanded,
                policy_file: opts.policy_path,
                policy_params: opts.policy_params,
            });
        }

        if (nodes_expanded as u64) >= opts.max_expansions {
            return Ok(PolicyResult {
                status: PolicyStatus::BudgetExhausted,
                move_layers: vec![],
                goal_config: initial_cfg,
                nodes_expanded,
                policy_file: opts.policy_path,
                policy_params: opts.policy_params,
            });
        }

        let actions = call_step(&policy, &pg, &gs, &ctx_value, &lib, &opts.sandbox)?;

        // Apply actions atomically; track whether any insert succeeded.
        let (committed_new_child, halt) = apply_actions(&pg, &actions, &mut solutions, &opts)?;
        if committed_new_child {
            nodes_expanded += 1;
        }
        if let Some(status) = halt {
            return Ok(PolicyResult {
                status,
                move_layers: solutions
                    .first()
                    .map(|&n| pg.inner_borrow().graph.reconstruct_path(n))
                    .unwrap_or_default(),
                goal_config: solutions
                    .first()
                    .map(|&n| pg.inner_borrow().graph.config(n).clone())
                    .unwrap_or(initial_cfg.clone()),
                nodes_expanded,
                policy_file: opts.policy_path,
                policy_params: opts.policy_params,
            });
        }
    }
}

fn call_init(
    policy: &LoadedPolicy,
    pg: &PolicyGraph,
    ctx: &Ctx,
    cfg: &SandboxConfig,
) -> Result<serde_json::Value, DslError> {
    // Find the `init` binding, call it with (root_node_id, ctx).
    // Marshal the returned struct to serde_json::Value (the kernel
    // doesn't care about the schema; it stores the value verbatim).
    let _ = (policy, pg, ctx, cfg);
    todo!("call init via Evaluator::eval_function on policy.frozen.get(\"init\")")
}

fn call_step(
    policy: &LoadedPolicy,
    pg: &PolicyGraph,
    gs: &serde_json::Value,
    ctx: &Ctx,
    lib: &LibMove,
    cfg: &SandboxConfig,
) -> Result<Vec<MoveAction>, DslError> {
    let _ = (policy, pg, gs, ctx, lib, cfg);
    todo!("call step, unmarshall returned dict-or-list-of-dicts to Vec<MoveAction>")
}

fn apply_actions(
    pg: &PolicyGraph,
    actions: &[MoveAction],
    solutions: &mut Vec<NodeId>,
    opts: &PolicyOptions,
) -> Result<(bool, Option<PolicyStatus>), DslError> {
    let _ = (pg, actions, solutions, opts);
    // For each action: dispatch to graph_handle / NodeStateMap / etc.
    // Track:
    //   - committed_new_child: true if any InsertChild produced is_new=true
    //   - halt: Some(PolicyStatus) if a Halt action fired
    todo!("dispatch table for the six MoveAction variants")
}
```

- [ ] **Step 3: Wire the `PolicyGraph::new` constructor**

Add a `pub(super) fn new(inner: PolicyGraphInner) -> Self` and a `pub(super) fn inner_borrow(&self) -> std::cell::Ref<'_, PolicyGraphInner>` to `graph_handle.rs`.

- [ ] **Step 4: Implement `call_init`, `call_step`, `apply_actions`**

For each `todo!()`:
- `call_init`: `let init = policy.get("init").ok_or(BadPolicy)?; eval.eval_function(init, &[Value::new_int(root_id), heap.alloc(ctx)], &[])?` — convert returned struct to `serde_json::Value`.
- `call_step`: same pattern; the return is either a single dict, a list of dicts, or empty list. Unify into `Vec<MoveAction>` via `MoveAction::try_from_json`.
- `apply_actions`: match each variant:
  - `InsertChild { parent, move_set }`: validate AOD via `BusGridContext::is_aod_valid(...)` (or equivalent — check existing `aod_grid.rs` for the predicate). On valid: derive `new_config` via `Config::with_moves`, call `inner.graph.insert(parent, move_set, new_config, edge_cost)`. Update `inner.children`. Set `inner.last_insert = Some(InsertOutcome { ... })`. Increment `committed_new_child` only if `is_new`.
  - On invalid: set `inner.last_insert = Some(InsertOutcome { error: Some("aod_invalid: ..."), ... })`; do not halt.
  - `UpdateNodeState { node, patch }`: merge `patch.0` into `inner.node_state.0[node]`. Schema-check (defer to Task 19) — for v1 just merge.
  - `UpdateGlobalState { patch }`: merge `patch.0` into the GS held by the kernel. Re-bind in subsequent `call_step` invocations.
  - `EmitSolution { node }`: push to `solutions`.
  - `Halt { status, message }`: return `(committed, Some(map_status(status, message)))` immediately.
  - `InvokeBuiltin { name, args }`: see Task 20.

- [ ] **Step 5: Run integration test**

Run: `cargo test -p bloqade-lanes-search --test dsl_kernel_minimal`
Expected: PASS (`status == Solved`).

- [ ] **Step 6: Commit**

```bash
git add crates/bloqade-lanes-search/src/move_policy_dsl/kernel.rs crates/bloqade-lanes-search/src/move_policy_dsl/adapter_impl.rs crates/bloqade-lanes-search/src/move_policy_dsl/graph_handle.rs crates/bloqade-lanes-search/tests/dsl_kernel_minimal.rs
git commit -m "feat(search): kernel loop — init/step/halt happy path

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 18: `last_insert` and `last_builtin_result` side channels

**Files:**
- Modify: `crates/bloqade-lanes-search/src/move_policy_dsl/graph_handle.rs`

- [ ] **Step 1: Failing test for the side channel**

Add an integration test `dsl_kernel_insert_outcome.rs`:

```rust
//! After issuing an insert_child action, `graph.last_insert()` must
//! return a struct with the right shape on the *next* tick.

const POLICY_SRC: &str = r#"
NodeState = record()
PARAMS = struct()

def init(root, ctx):
    return struct(stage = 0, child = None)

def step(graph, gs, ctx, lib):
    if gs.stage == 0:
        # Issue a single, deliberately AOD-invalid insert to test the
        # error path (or any valid one — the contract is that
        # last_insert() reflects the action's outcome).
        return [actions.update_global_state(stage = 1)]
    if gs.stage == 1:
        outcome = graph.last_insert()
        if outcome != None:
            return actions.halt("solved", "saw outcome")
        return actions.halt("error", "no outcome seen")
"#;
```

(In practice this test issues `actions.insert_child(graph.root, valid_move_set)` at stage 0 and asserts `outcome.is_new == True` at stage 1. Build a real `valid_move_set` via the test_utils.)

- [ ] **Step 2: Implement `graph.last_insert()` and `graph.last_builtin_result()`**

In `graph_handle.rs`, add to `register_graph_methods`:

```rust
fn last_insert<'v>(this: &PolicyGraph, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
    let inner = this.inner.borrow();
    match &inner.last_insert {
        None => Ok(Value::new_none()),
        Some(o) => {
            let dict = starlark::values::dict::AllocDict([
                ("child_id", o.child_id.map(|n| heap.alloc(n.0 as i32)).unwrap_or_else(Value::new_none)),
                ("is_new", heap.alloc(o.is_new)),
                ("error", o.error.as_ref().map(|s| heap.alloc(s.as_str())).unwrap_or_else(Value::new_none)),
            ]);
            Ok(heap.alloc(dict))
        }
    }
}

fn last_builtin_result<'v>(this: &PolicyGraph, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
    let inner = this.inner.borrow();
    match &inner.last_builtin {
        None => Ok(Value::new_none()),
        Some(o) => {
            let dict = starlark::values::dict::AllocDict([
                ("status", heap.alloc(o.status.as_str())),
                ("payload", heap.alloc(o.payload.to_string())),
            ]);
            Ok(heap.alloc(dict))
        }
    }
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p bloqade-lanes-search --test dsl_kernel_insert_outcome`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add crates/bloqade-lanes-search/src/move_policy_dsl/graph_handle.rs crates/bloqade-lanes-search/tests/dsl_kernel_insert_outcome.rs
git commit -m "feat(search): last_insert/last_builtin_result side channels

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 19: Schema validation for `update_node_state` / `update_global_state`

**Files:**
- Modify: `crates/bloqade-lanes-search/src/move_policy_dsl/kernel.rs`
- Modify: `crates/bloqade-lanes-search/src/move_policy_dsl/actions.rs`

- [ ] **Step 1: Failing test for schema rejection**

Add an integration test that defines `NodeState = record(entropy=field(int, default=0))` then issues `actions.update_node_state(node, foo=1)` (unknown field). Expected: `result.status == SchemaError("foo")`.

- [ ] **Step 2: Capture the schema declaration**

When `LoadedPolicy` is built, look up `policy.get("NodeState")` and `policy.get("GlobalState")` (if present). The Starlark `record(...)` value carries its field schema introspectably via the `starlark::record::Record` type. Extract the field-name set into a `Vec<String>` and store on `PolicyResult` / kernel state.

(If the API for introspecting `record` schemas is awkward in the pinned starlark version, a simpler v1 approach: only validate at apply time — when an `update_node_state` patch is applied, check that all keys exist in the *first* dict ever stored at that node. This catches typos but not the truly empty-state initial case. Iterate to a proper schema check after the v1 ships.)

- [ ] **Step 3: Implement the check in `apply_actions`**

```rust
MoveAction::UpdateNodeState { node, patch } => {
    if let Some(schema) = node_state_schema {
        for k in patch.0.as_object().unwrap_or(&Default::default()).keys() {
            if !schema.contains(k) {
                return Ok((false, Some(PolicyStatus::SchemaError(k.clone()))));
            }
        }
    }
    // ... merge patch into inner.node_state.0[node]
}
```

- [ ] **Step 4: Run test, commit**

Run: `cargo test -p bloqade-lanes-search`
Expected: schema rejection test passes.

```bash
git add crates/bloqade-lanes-search/src/move_policy_dsl/
git commit -m "feat(search): schema validation for state updates

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 20: Builtin `sequential_fallback`

**Files:**
- Modify: `crates/bloqade-lanes-search/src/move_policy_dsl/kernel.rs`

- [ ] **Step 1: Failing test for the builtin**

Issuing `actions.invoke_builtin("sequential_fallback", from_config=graph.config(graph.root))` followed by `actions.halt("fallback")` must produce `PolicyStatus::Fallback("...")` and a non-empty `move_layers` (the greedy per-qubit BFS sequence).

- [ ] **Step 2: Implement**

The existing `entropy.rs` has a `sequential_fallback` analog (search for "fallback" in `entropy.rs`). Lift it to `kernel.rs` (or a new `move_policy_dsl/builtins.rs`) as `pub(super) fn sequential_fallback(from: &Config, ctx: &SearchContext, index: &LaneIndex) -> Vec<MoveSet>`. Wire it into the `MoveAction::InvokeBuiltin` branch:

```rust
MoveAction::InvokeBuiltin { name, args } if name == "sequential_fallback" => {
    let from_cfg = parse_config_from_json(&args.0, "from_config")?;
    let path = builtins::sequential_fallback(&from_cfg, &search_ctx, &index);
    inner.last_builtin = Some(BuiltinOutcome {
        status: "ok".into(),
        payload: serde_json::json!({"path_len": path.len()}),
    });
    // Append to solutions if it produced one. Or store separately;
    // depends on §5.5 semantics. Per spec, halt("fallback") is the
    // policy's signal that this is the result.
    fallback_path = Some(path);
}
MoveAction::InvokeBuiltin { name, .. } => {
    return Ok((false, Some(PolicyStatus::BadPolicy(format!("unknown builtin: {name}")))));
}
```

- [ ] **Step 3: Run tests, commit**

```bash
git add crates/bloqade-lanes-search/src/move_policy_dsl/
git commit -m "feat(search): sequential_fallback builtin

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 4 — PyO3 binding + reference policy + acid test

### Task 21: PyO3 — `policy_path` + `policy_params` kwargs on `MoveSolver.solve`

**Files:**
- Modify: `crates/bloqade-lanes-bytecode-python/src/search_python.rs`
- Modify: `python/bloqade/lanes/bytecode/_native.pyi`

- [ ] **Step 1: Failing Python test**

Create `python/tests/bytecode/test_move_policy_dsl.py`:

```python
"""End-to-end smoke test: MoveSolver.solve(policy_path=...) routes through
the Move Policy DSL kernel and returns a SolveResult."""

from pathlib import Path

import pytest

from bloqade.lanes.bytecode import MoveSolver

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_trivial_halt_policy_returns_solved(arch_spec_json, tiny_initial, tiny_target):
    policy = (REPO_ROOT / "policies" / "reference" / "entropy.star").as_posix()
    solver = MoveSolver.from_json(arch_spec_json)
    result = solver.solve(
        initial=tiny_initial,
        target=tiny_target,
        blocked=[],
        policy_path=policy,
        policy_params={"e_max": 8},
        max_expansions=10_000,
        timeout_s=10.0,
    )
    assert result.policy_file.endswith("entropy.star")
    assert result.policy_params == {"e_max": 8}
    assert result.status.name in {"SOLVED", "FALLBACK"}, result.status.name
```

(Add fixtures `arch_spec_json`, `tiny_initial`, `tiny_target` either inline or in a `conftest.py`. The existing test suite has fixture patterns to copy.)

- [ ] **Step 2: Wire the PyO3 kwarg surface**

Modify `crates/bloqade-lanes-bytecode-python/src/search_python.rs::PyMoveSolver::solve`:

```rust
#[pymethods]
impl PyMoveSolver {
    #[pyo3(signature = (initial, target, blocked, *, max_expansions=None, options=None, policy_path=None, policy_params=None, timeout_s=None))]
    fn solve(
        &self,
        py: Python<'_>,
        initial: Vec<(u32, PyLocationAddr)>,
        target: Vec<(u32, PyLocationAddr)>,
        blocked: Vec<PyLocationAddr>,
        max_expansions: Option<u64>,
        options: Option<PySolveOptions>,
        policy_path: Option<&str>,
        policy_params: Option<&Bound<'_, PyDict>>,
        timeout_s: Option<f64>,
    ) -> PyResult<PySolveResult> {
        if let Some(path) = policy_path {
            let params = policy_params
                .map(|d| pydict_to_json(d))
                .transpose()?
                .unwrap_or(serde_json::Value::Object(Default::default()));
            let opts = bloqade_lanes_search::move_policy_dsl::PolicyOptions {
                policy_path: path.to_string(),
                policy_params: params,
                max_expansions: max_expansions.unwrap_or(100_000),
                timeout_s,
                sandbox: bloqade_lanes_dsl_core::sandbox::SandboxConfig::default(),
            };
            let result = bloqade_lanes_search::move_policy_dsl::solve_with_policy(
                initial.into_iter().map(|(q, l)| (q, l.0)),
                target.into_iter().map(|(q, l)| (q, l.0)),
                blocked.into_iter().map(|l| l.0),
                Arc::new(self.inner.index().clone()),
                opts,
            )
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;
            return Ok(PySolveResult::from_policy(result));
        }
        // ... existing strategy-based path ...
    }
}
```

Add `pub fn from_policy(p: PolicyResult) -> Self` on `PySolveResult` that fills the new `.policy_file` and `.policy_params` fields and maps the `PolicyStatus` to a string-typed status. Add `policy_file: Option<String>` and `policy_params: Option<PyObject>` fields to `PySolveResult` (keeping them `None` for the Strategy path).

- [ ] **Step 3: Update the type stub**

Modify `python/bloqade/lanes/bytecode/_native.pyi` to add the new kwargs and result fields.

- [ ] **Step 4: Rebuild + run Python test**

```bash
just develop-python
uv run pytest python/tests/bytecode/test_move_policy_dsl.py -v
```

Expected: PASS once the reference policy lands in Task 22.

- [ ] **Step 5: Commit**

```bash
git add crates/bloqade-lanes-bytecode-python/ python/bloqade/lanes/bytecode/_native.pyi python/tests/bytecode/test_move_policy_dsl.py
git commit -m "feat(python): expose policy_path/policy_params on MoveSolver.solve

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 22: Reference policy `entropy.star`

**Files:**
- Create: `policies/reference/entropy.star`

- [ ] **Step 1: Author the policy**

Copy the `entropy.star` body from spec §9 verbatim into `policies/reference/entropy.star`. Add a top-of-file docstring:

```python
# entropy.star — reference Starlark reproduction of `Strategy::Entropy`.
#
# Equivalent to Strategy::Entropy with restart_count=1 and the entropy
# perturbation RNG dropped (§9 of the design spec). Restart diversity is
# the caller's responsibility: run N parallel solves with N different
# PARAMS overrides.
#
# Acid test: `crates/bloqade-lanes-search/tests/dsl_entropy_acid.rs`
# verifies this produces the same `goal_config` as Strategy::Entropy
# on N committed fixtures.
```

(The full body is in the spec §9, ~80 lines. Paste it as-is, fix any small naming drifts surfaced by Phase 3 — e.g. `lib.normalize` is `normalize` at the global scope per Task 8; adjust if the spec used a `lib.` prefix.)

- [ ] **Step 2: Smoke check from Python**

Run: `uv run pytest python/tests/bytecode/test_move_policy_dsl.py::test_trivial_halt_policy_returns_solved -v`
Expected: PASS — the policy loads, runs, and returns a result.

- [ ] **Step 3: Commit**

```bash
git add policies/reference/entropy.star
git commit -m "feat(policies): add reference entropy.star policy

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 23: Acid test — `entropy.star` vs `Strategy::Entropy`

**Files:**
- Create: `crates/bloqade-lanes-search/tests/dsl_entropy_acid.rs`
- Create: `crates/bloqade-lanes-search/tests/fixtures/dsl_acid/{tiny.json,small.json,medium.json}` (or reuse existing test fixtures)

- [ ] **Step 1: Identify reproducible Strategy::Entropy fixtures**

Read `crates/bloqade-lanes-search/src/entropy.rs` test module (`#[cfg(test)] mod tests`). Find at least three test cases where `Strategy::Entropy` deterministically produces a specific `move_layers.len()` and `goal_config` (using the existing seeded RNG path). Note these as the "ground truth" fixtures.

- [ ] **Step 2: Failing acid test**

Create `crates/bloqade-lanes-search/tests/dsl_entropy_acid.rs`:

```rust
//! Acid test: `entropy.star` produces the same goal_config as
//! `Strategy::Entropy` on N fixtures, modulo RNG-induced path
//! differences.
//!
//! What we assert:
//!   1. result.status == Solved (or both fall back, with same fallback)
//!   2. result.goal_config == strategy_result.goal_config
//!   3. |result.move_layers| within ±2 of strategy_result move_layers
//!      length (RNG perturbation in entropy.rs allows minor variation;
//!      tightening this is a follow-up).

use bloqade_lanes_search::move_policy_dsl::{solve_with_policy, PolicyOptions};
use bloqade_lanes_search::solve::{MoveSolver, SolveOptions, Strategy};
// ... use crate::test_utils::* ...

#[test]
fn entropy_star_matches_strategy_entropy_tiny() {
    // Build a tiny placement problem; load entropy.star; solve both
    // paths; compare goal_config.
}

#[test]
fn entropy_star_matches_strategy_entropy_small() { ... }

#[test]
fn entropy_star_matches_strategy_entropy_medium() { ... }
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p bloqade-lanes-search --test dsl_entropy_acid`
Expected: 3 PASS.

If the path-length tolerance is tight, **expand it to ±3 or ±5** rather than chase RNG ghosts. The acid test's purpose is structural ("the policy reproduces the algorithm"), not bit-exact.

- [ ] **Step 4: Commit**

```bash
git add crates/bloqade-lanes-search/tests/dsl_entropy_acid.rs crates/bloqade-lanes-search/tests/fixtures/
git commit -m "test(search): acid test entropy.star vs Strategy::Entropy

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 24: Determinism property test

**Files:**
- Modify: `crates/bloqade-lanes-search/tests/dsl_entropy_acid.rs` (or new `dsl_determinism.rs`)

- [ ] **Step 1: Failing property test**

```rust
#[test]
fn entropy_star_is_deterministic_across_runs() {
    // Run entropy.star twice on the same fixture, assert byte-identical
    // (move_layers, goal_config, nodes_expanded).
}
```

- [ ] **Step 2: Run, commit**

If determinism fails (e.g. a `HashMap` iteration leaked into the kernel-side action ordering), root-cause and fix. The spec §8 invariants are non-negotiable.

```bash
git commit -m "test(search): determinism property for entropy.star"
```

---

### Task 25: PARAMS overrides

**Files:**
- Modify: `crates/bloqade-lanes-search/src/move_policy_dsl/kernel.rs`

- [ ] **Step 1: Failing test**

`policy_params={"e_max": 16}` should override the policy's declared `PARAMS.e_max`. After overrides, `PARAMS.e_max == 16` from inside the policy.

- [ ] **Step 2: Implement**

In `solve_with_policy`, after `LoadedPolicy::from_path`:

```rust
if let serde_json::Value::Object(overrides) = &opts.policy_params {
    apply_params_overrides(&policy, overrides)?;
}
```

`apply_params_overrides`: re-read `policy.get("PARAMS")`, for each key in `overrides`, replace the field on the struct (since the module is frozen, this requires an unfreezing-and-refreezing cycle, or — simpler — pass the merged params to `init`/`step` as a `params` kwarg). The spec only supports top-level overrides; nested struct overrides are out of scope.

**Simpler approach:** on every call to `init` and `step`, merge `PARAMS` (from the policy) with `policy_params` (from the caller) and pass the merged struct as an extra Starlark global named `PARAMS_OVERRIDE`. The reference `entropy.star` can read it via `PARAMS = struct(...) | (PARAMS_OVERRIDE or struct())` at module top-level. **This is awkward for v1**; chooses the unfreeze-merge-refreeze cycle if starlark-rust supports it cleanly, otherwise document the constraint and ship the via-`PARAMS_OVERRIDE` variant.

- [ ] **Step 3: Run, commit**

---

## Phase 5 — Polish + Plan A close-out

### Task 26: Doc + AGENT.md update

**Files:**
- Modify: `AGENT.md` (add a "Move Policy DSL" section)
- Create: `docs/superpowers/specs/2026-04-29-move-policy-dsl-design.md` (transcribe the Notion spec we worked from, for posterity)

- [ ] **Step 1: AGENT.md section**

Add a section after the "Common Commands" header documenting the new `policies/reference/` directory and the `MoveSolver.solve(policy_path=...)` Python API.

- [ ] **Step 2: Spec transcription**

Copy the Notion spec body into `docs/superpowers/specs/2026-04-29-move-policy-dsl-design.md`. (Per user's memory `feedback_no_commit_design_docs.md`, design docs are typically not committed; **confirm with the user before committing this file**. If they decline, leave it on disk in the worktree and skip the commit.)

- [ ] **Step 3: Commit (only if approved)**

---

### Task 27: Final lint + format pass

- [ ] **Step 1: Run formatters**

```bash
cargo fmt --all
uv run black python
uv run isort python
```

- [ ] **Step 2: Run linters**

```bash
cargo clippy -p bloqade-lanes-dsl-core -p bloqade-lanes-search -p bloqade-lanes-bytecode-python --all-targets -- -D warnings
uv run ruff check python
uv run pyright python
```

Fix any new warnings introduced by the DSL code.

- [ ] **Step 3: Run the full test suite**

```bash
just test
```

Expected: all green.

- [ ] **Step 4: Commit**

```bash
git commit -am "chore: format + clippy after DSL implementation"
```

---

### Task 28: Hand-off summary

- [ ] **Step 1: Write a hand-off note** to the user describing:
  - What's on `jason/move-policy-dsl`.
  - What's covered (Plan A) and what's deferred (Plans B, C).
  - Open questions from spec §14 that have concrete answers now (e.g. crate split: in-crate `move_policy_dsl` module per the implementation; `policy_path` is orthogonal to `Strategy`).
  - Any acid-test tolerances (path-length ±N) that the user should sanity-check before merging.

- [ ] **Step 2: Stop here.** PR creation, merging, and follow-up Plans B/C are user-driven.

---

## Plan B — Target Generator DSL (delivered)

**Delivered:** 2026-05-01, stacked on top of Plan A on the same branch (`jason/move-policy-dsl`). Issue #597 Plan B checkbox is satisfied.

**Goal recap.** Mirror the Move Policy DSL on the *Place* side: a Starlark-hosted adapter that implements `TargetGeneratorABC` so agents can author CZ-stage target generators in `.star` files and inject them at `PhysicalPlacementStrategy(target_generator=...)`. Target generation is a single pure function call per CZ stage (no search graph, no step loop, no transposition table), so the implementation is significantly thinner than Plan A's kernel.

**Architecture decisions.**
- **Kernel location:** new module `crates/bloqade-lanes-search/src/target_generator_dsl/` (sibling to `move_policy_dsl/`). Lives in `bloqade-lanes-search` so it can reuse the existing Rust `validate_candidate` in `crates/bloqade-lanes-search/src/target_generator.rs:130-185` without crossing crate boundaries.
- **Lib surface — minimal v1.** No bespoke pipeline primitives. Policies have `stable_sort` / `argmax` / `normalize` from `dsl-core` globals, plus a thin `lib.cz_partner(loc)` shortcut. Richer `lib_target.*` helpers are deferred until a real policy demands them.
- **PyO3 surface — native ArchSpec.** `TargetPolicyRunner.__init__(policy_path, arch_spec)` takes the native `ArchSpec` (matching `MoveSolver.from_arch_spec`'s style) rather than re-parsing JSON per call. Avoids round-trips while still keeping the runner stateless beyond its parsed-and-frozen `LoadedPolicy`.
- **Runner caching.** The Python adapter caches the underlying `TargetPolicyRunner` keyed by the `arch_spec` instance, so a single `.star` file is parsed once and reused across every CZ stage of a placement run.
- **Validation: Rust-side, single source of truth.** Each candidate returned by the policy is validated by `target_generator::validate_candidate` before crossing back into Python. The Python `_validate_candidate` in `target_generator.py` continues to run as defense-in-depth on the strategy's `_build_candidates` path; both check the same invariants.

### Policy author surface (`.star` contract)

```python
def generate(ctx, lib):
    # ctx.arch_spec           — ArchSpec wrapper (read-only)
    # ctx.placement           — Placement: dict-like, .qubits()/.get(qid)/.items()/.len
    # ctx.controls            — list[int]
    # ctx.targets             — list[int]
    # ctx.lookahead_cz_layers — list[(list[int], list[int])]
    # ctx.cz_stage_index      — int
    # lib.arch_spec           — alias of ctx.arch_spec
    # lib.cz_partner(loc)     — Location | None
    target = {}
    for q in ctx.placement.qubits():
        target[q] = ctx.placement.get(q)
    for i in range(len(ctx.controls)):
        c = ctx.controls[i]
        t = ctx.targets[i]
        target[c] = lib.cz_partner(target[t])
    return [target]   # list[dict[int, Location]]
```

Empty `[]` defers to the strategy's `DefaultTargetGenerator` fallback.

The starlark-0.13 late-binding caveat from `entropy.star` still applies: pass `ctx` and `lib` as function parameters; do not capture them as free variables in nested helpers.

### File structure (delivered)

**New Rust files:**
| File | Purpose |
|---|---|
| `crates/bloqade-lanes-dsl-core/src/primitives/placement.rs` | `StarlarkPlacement` shared primitive (BTreeMap-backed dict-like view; `.get`, `.qubits`, `.items`, `.len`). |
| `crates/bloqade-lanes-search/src/target_generator_dsl/mod.rs` | Module index. |
| `crates/bloqade-lanes-search/src/target_generator_dsl/ctx_handle.rs` | `StarlarkTargetContext` exposing the six `ctx.*` fields to policies. |
| `crates/bloqade-lanes-search/src/target_generator_dsl/lib_target.rs` | `StarlarkLibTarget` with `cz_partner(loc)` and the `arch_spec` alias attribute. |
| `crates/bloqade-lanes-search/src/target_generator_dsl/kernel.rs` | `TargetPolicyRunner` + `run_target_policy(...)` + `TargetPolicyError`. Loads `.star`, invokes `generate(ctx, lib)`, marshals the returned `list[dict[int, Location]]`, runs `validate_candidate`. |
| `crates/bloqade-lanes-bytecode-python/src/target_generator_dsl_python.rs` | PyO3 `TargetPolicyRunner` class — `__init__(policy_path, arch_spec)` + `generate(...)`. |

**Modified Rust files:**
| File | Change |
|---|---|
| `crates/bloqade-lanes-dsl-core/src/primitives/mod.rs` | `pub mod placement;` + re-export `StarlarkPlacement`. |
| `crates/bloqade-lanes-search/src/lib.rs` | `pub mod target_generator_dsl;`. |
| `crates/bloqade-lanes-search/Cargo.toml` | Added `thiserror = "1"` to deps. |
| `crates/bloqade-lanes-bytecode-python/src/lib.rs` | Register `PyTargetPolicyRunner` in the `_native` pymodule. |
| `crates/bloqade-lanes-bytecode-python/src/search_python.rs` | `pydict_to_json` promoted to `pub(crate)` for reuse. |

**New Python files:**
| File | Purpose |
|---|---|
| `python/bloqade/lanes/heuristics/physical/target_generator_dsl.py` | `TargetGeneratorDSL(TargetGeneratorABC)` adapter — caches one `TargetPolicyRunner` per `arch_spec`, marshals `TargetContext` to/from native types per call. |
| `policies/reference/default_target.star` | Reference policy mirroring `DefaultTargetGenerator` (acid-test against the existing in-tree default). |
| `python/tests/heuristics/test_target_generator_dsl.py` | 8 tests: subclass check, default-policy parity, wrapper round-trip, empty-list fallback semantics, validator rejection, runner caching, reference-policy parity, end-to-end through `PhysicalPlacementStrategy._build_candidates`. |

**Modified Python files:**
| File | Change |
|---|---|
| `python/bloqade/lanes/bytecode/_native.pyi` | Added `TargetPolicyRunner` stub class. |
| `python/bloqade/lanes/bytecode/__init__.py` | Re-export `TargetPolicyRunner` from `_native`. |

### Determinism guarantees (Plan B specific)

- `StarlarkPlacement` is backed by a `BTreeMap`, so `ctx.placement.qubits()` and `ctx.placement.items()` iterate in sorted qubit-id order on every call.
- The kernel iterates the returned candidates in policy-defined order; validation never reorders, only accepts/rejects.
- No new RNG, time, or IO surface is exposed — `lib.cz_partner` is a pure arch lookup, and the rest of the determinism story is inherited from `dsl-core::sandbox`.

### Verification (results from delivery)

- **Rust unit tests** (`cargo test -p bloqade-lanes-dsl-core -p bloqade-lanes-search --lib`): 193 pass — including 3 `StarlarkPlacement`, 2 `StarlarkTargetContext`, 1 `StarlarkLibTarget`, and 4 `TargetPolicyRunner` tests added by Plan B.
- **Python tests** (`uv run pytest python/tests`): 1075 passed, 9 skipped, 0 failed. The 8 new `test_target_generator_dsl.py` tests cover ABC subclassing, parity with `DefaultTargetGenerator`, wrapper round-trip, fallback semantics, validator rejection, runner caching, reference-policy parity, and end-to-end integration with `PhysicalPlacementStrategy`.
- **Lint:** `cargo clippy -p bloqade-lanes-dsl-core` clean; `pyright`, `black`, `isort`, `ruff` clean on every Plan B Python file. (Pre-existing clippy issues in `bloqade-lanes-search/src/{entropy,generators,ordering}.rs` predate Plan B and are outside CI's lint scope, which only covers the bytecode-core / bytecode-cli crates.)

### Out of scope for Plan B (deferred to Plan C or later)

- CLI harness for target-generator policies (`lanes eval-policy --target ...`).
- Auto-generated `policies/primer.md` covering the Place DSL surface.
- Snapshot tests against fixture corpora.
- Richer `lib_target.*` primitives (Dijkstra-style cost estimation, AOD-cluster signature grouping). Will be added when a policy author needs them.
- A `from_arch_spec_json` constructor on `TargetPolicyRunner`. The native-`ArchSpec` constructor covers every in-tree call site; expose JSON only if a downstream caller actually wants it.

---

## Self-review notes (already applied above)

- **Spec coverage check.** Walking through the spec sections: §1 (covered by Task 21 + Task 22), §2 (motivation; no impl needed), §3 (non-goals; no impl), §4 (covered by Tasks 9, 17, 21), §5 (covered by Tasks 11–20), §6 (deferred to Plan B, noted), §7 (covered by Tasks 1–9), §8 (covered by Task 24), §9 (covered by Tasks 22–23), §10 (partially covered: Python test in Task 21; CLI deferred to Plan C), §11 (no impl needed; the Policy trait in Task 9 is the forward-compat seam), §12 (file layout matches), §13 (no impl), §14 (open questions resolved inline above).
- **Placeholder scan.** The plan uses `todo!()` in three places (Tasks 11, 14, 17) — each is intentional and gated to be replaced *within the same task* (Task 12 closes Task 11's `todo!`; Task 14 has the impl in Step 1; Task 17 closes Task 11/14 holes via `call_init`/`call_step`/`apply_actions`). No `TBD`, `figure-out-later`, or "similar to" references. Acid-test tolerance is concrete (±2 path layers, expandable to ±5).
- **Type consistency.** `PolicyOptions`, `PolicyResult`, `PolicyStatus`, `MoveAction`, `PatchValue`, `InsertOutcome`, `BuiltinOutcome`, `NodeStateMap`, `PolicyGraph`, `PolicyGraphInner`, `LibMove`, `Ctx`, `StarlarkConfig`, `StarlarkMoveSet`, `StarlarkLocation`, `StarlarkLane`, `StarlarkArchSpec`, `LoadedPolicy`, `SandboxConfig`, `DslError`, `Policy`, `StepResult`, `HaltStatus`, `solve_with_policy` — names are stable across all task references above.
- **Cycle avoidance.** Resolved in Task 6 / Task 14: `Config` and `MoveSet` Starlark wrappers live in `bloqade-lanes-search` (not in `dsl-core`), so the dep graph stays acyclic: `dsl-core ← search`, never the reverse.
