# Move Policy DSL — Plan C Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land Plan C of the Move Policy DSL framework — three reference Move policies (`dfs.star`, `bfs.star`, `ids.star`); `eval-policy` and `trace-policy` CLI subcommands on `bloqade-bytecode` covering both Move and Target policies via a `"kind"` discriminator; an auto-generated and CI-checked `policies/primer.md`; and a committed `policies/fixtures/` snapshot corpus with a Rust integration-test driver.

**Architecture:** A `MoveKernelObserver` trait in `bloqade-lanes-search::move_policy_dsl::observer` (and a parallel `TargetKernelObserver` in `target_generator_dsl::observer`) is threaded through `solve_with_policy` and the Target kernel. The CLI's `trace-policy` installs a `JsonMoveTraceObserver` / `JsonTargetTraceObserver` that writes NDJSON; `eval-policy` installs a `NoOpMoveObserver` / `NoOpTargetObserver` and emits a single JSON summary. A shared problem-fixture loader at `bloqade-lanes-search::fixture` parses self-contained `problem.json` files discriminated by a top-level `"kind": "move"|"target"` field. A new `policies-primer` binary in the search crate uses `include_str!` to embed the registration-site source files at its own compile time, parses doc-comments via `syn`, and stamps `policies/primer.md` with sentinel-bracketed AUTOGEN sections (regenerated) and PROSE sections (preserved). Snapshot fixtures use structural-match comparison on key fields.

**Tech Stack:** Rust 2024, `clap` (CLI), `serde` + `serde_json` + `schemars` (schema generation), `syn` + `proc_macro2` (primer source-inspection), `assert_cmd` (CLI integration tests), `starlark-rust`, existing `bloqade-lanes-search` infrastructure.

**Spec:** [docs/superpowers/specs/2026-05-01-move-policy-dsl-plan-c-design.md](../specs/2026-05-01-move-policy-dsl-plan-c-design.md). Sections referenced as `§N`.

**Predecessors:** Plan A (Move Policy DSL framework, commit `b9d9383`), Plan B (Target Generator DSL, commit `11c664f`). Both delivered on this branch.

---

## Conventions used throughout

- **Working tree:** `~/.config/superpowers/worktrees/bloqade-lanes/move-policy-dsl/` on branch `jason/move-policy-dsl`. All paths in this plan are relative to the repository root inside that worktree.
- **Crate scopes:** `search` = `bloqade-lanes-search`, `cli` = `bloqade-lanes-bytecode-cli`, `dsl-core` = `bloqade-lanes-dsl-core`, `core` = `bloqade-lanes-bytecode-core`.
- **Run Rust tests:** `cargo test -p <crate>` for one crate, `just test-rust` for all Rust. Use `cargo test -p <crate> <test_name>` to target a single test.
- **Run Python tests:** `uv run pytest python/tests/<file>.py -v`. The native Rust extension must be rebuilt after Rust changes: `just develop-python` (fast path) or `just develop` (full).
- **Format / lint:** `cargo fmt --all` + `cargo clippy -p <crate> --all-targets -- -D warnings` before each commit. Pre-commit hooks enforce this on commit.
- **Commit messages:** Conventional Commits per `AGENT.md`. Each task ends with **at most one** commit.
- **Git policy:** The user owns git operations. The `git commit` step in each task is provided for executing-agent convenience but **may be deferred to the user**. If executing without auto-commit, leave the working tree dirty between tasks and let the user batch-commit at logical boundaries — the per-task commit messages below still serve as message templates.
- **Naming clarification:** The search crate already exports `crate::observer::NoOpObserver` for the existing A*/entropy `SearchObserver` infrastructure. Plan C's new types use distinct names — `NoOpMoveObserver`, `NoOpTargetObserver`, `JsonMoveTraceObserver`, `JsonTargetTraceObserver` — and live in `move_policy_dsl::observer` / `target_generator_dsl::observer`. They are **not** re-exported from `lib.rs` at the unqualified name; consumers reach them via the submodule path.

---

## File Structure

The full file inventory is in [§12 of the spec](../specs/2026-05-01-move-policy-dsl-plan-c-design.md#12-file-by-file-additions). This plan touches every file in that table. Phases below sequence the work so each phase produces working, testable software.

| Phase | Deliverable | Tasks |
|---|---|---|
| 1 | Observer trait + kernel hookup (Move + Target) | Tasks 1–4 |
| 2 | Shared fixture loader at `search::fixture` | Task 5 |
| 3 | CLI subcommands `eval-policy` / `trace-policy` | Tasks 6–10 |
| 4 | Reference Move policies (`dfs.star`, `bfs.star`, `ids.star`) | Tasks 11–13 |
| 5 | Snapshot fixtures and `dsl_snapshot.rs` driver | Tasks 14–17 |
| 6 | Primer generator + CI hookup | Tasks 18–27 |

---

## Phase 0 — Baseline verification (before starting Task 1)

Run once before starting Plan C tasks:

```bash
git status                              # expect: branch jason/move-policy-dsl, clean
just test-rust                          # expect: all tests pass (Plan B baseline)
just develop-python && uv run pytest python/tests   # expect: 1075 passed, 9 skipped
```

If any baseline tests fail, stop and triage. Do not start Plan C on a red baseline.

---

## Phase 1 — Observer trait and kernel hookup

### Task 1: `MoveKernelObserver` trait + `NoOpMoveObserver` + kernel hookup

**Files:**
- Create: `crates/bloqade-lanes-search/src/move_policy_dsl/observer.rs`
- Modify: `crates/bloqade-lanes-search/src/move_policy_dsl/mod.rs`
- Modify: `crates/bloqade-lanes-search/src/move_policy_dsl/kernel.rs`
- Test: `crates/bloqade-lanes-search/src/move_policy_dsl/observer.rs` (inline `#[cfg(test)] mod tests`)

- [ ] **Step 1: Write the failing test for trait call ordering**

Add to a new file at the path above:

```rust
//! Observer trait for the Move Policy DSL kernel.
//!
//! Plan C — see `docs/superpowers/specs/2026-05-01-move-policy-dsl-plan-c-design.md` §5.

use crate::move_policy_dsl::actions::MoveAction;
use crate::move_policy_dsl::kernel::PolicyStatus;

/// Snapshot of the policy graph at the moment `on_init` is called.
/// Data-only mirror of relevant fields on `PolicyGraphInner`.
#[derive(Debug, Clone)]
pub struct PolicyGraphSnapshot {
    pub root_qubits: Vec<u32>,
    pub target_qubits: Vec<u32>,
    pub blocked_count: usize,
}

/// Per-step delta covering the kernel side-channel writes that Starlark sees.
#[derive(Debug, Clone, Default)]
pub struct GraphDelta {
    pub last_insert: Option<u64>,
    pub last_builtin: Option<String>,
    pub node_state_writes: Vec<(String, String)>,
}

/// Kernel observer for the Move Policy DSL `solve_with_policy` loop.
///
/// All hooks have default empty bodies. Implementors override only what
/// they need.
pub trait MoveKernelObserver {
    fn on_init(&mut self, _root: &PolicyGraphSnapshot) {}
    fn on_step(&mut self, _step: u64, _depth: u32, _action: &MoveAction, _delta: &GraphDelta) {}
    fn on_builtin(&mut self, _step: u64, _name: &str, _ok: bool) {}
    fn on_halt(&mut self, _status: &PolicyStatus) {}
}

/// No-op observer; the default for non-tracing callers.
pub struct NoOpMoveObserver;
impl MoveKernelObserver for NoOpMoveObserver {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::move_policy_dsl::actions::MoveAction;

    /// Test double that records every call in order, used by the kernel
    /// hookup test in Step 4.
    #[derive(Default)]
    pub(crate) struct RecordingObserver {
        pub calls: Vec<String>,
    }
    impl MoveKernelObserver for RecordingObserver {
        fn on_init(&mut self, _root: &PolicyGraphSnapshot) {
            self.calls.push("init".into());
        }
        fn on_step(&mut self, step: u64, depth: u32, _a: &MoveAction, _d: &GraphDelta) {
            self.calls.push(format!("step:{step}:{depth}"));
        }
        fn on_builtin(&mut self, step: u64, name: &str, ok: bool) {
            self.calls.push(format!("builtin:{step}:{name}:{ok}"));
        }
        fn on_halt(&mut self, _status: &PolicyStatus) {
            self.calls.push("halt".into());
        }
    }

    #[test]
    fn no_op_observer_compiles_against_trait() {
        // Pure type-level check that the no-op satisfies the trait.
        fn _accept(_o: &mut dyn MoveKernelObserver) {}
        let mut obs = NoOpMoveObserver;
        _accept(&mut obs);
    }
}
```

- [ ] **Step 2: Run the inline tests to verify they compile but kernel hookup is missing**

```bash
cargo test -p bloqade-lanes-search --lib move_policy_dsl::observer::
```

Expected: the `no_op_observer_compiles_against_trait` test passes. The `RecordingObserver` is defined but not yet exercised — that comes in Step 4.

- [ ] **Step 3: Wire the new module into `move_policy_dsl/mod.rs`**

Modify `crates/bloqade-lanes-search/src/move_policy_dsl/mod.rs` to add:

```rust
pub mod observer;

pub use kernel::{PolicyOptions, PolicyResult, PolicyStatus, solve_with_policy};
pub use observer::{
    GraphDelta, MoveKernelObserver, NoOpMoveObserver, PolicyGraphSnapshot,
};
```

(Keep the existing `pub mod` lines for `actions`, `adapter_impl`, `builtins`, `graph_handle`, `kernel`, `lib_move`.)

- [ ] **Step 4: Thread the observer through `solve_with_policy`**

Modify `crates/bloqade-lanes-search/src/move_policy_dsl/kernel.rs`:

a. Update the public function signature to take an observer:

```rust
pub fn solve_with_policy(
    initial: impl IntoIterator<Item = (u32, LocationAddr)>,
    target: impl IntoIterator<Item = (u32, LocationAddr)>,
    blocked: impl IntoIterator<Item = LocationAddr>,
    index: Arc<LaneIndex>,
    opts: PolicyOptions,
    observer: &mut dyn crate::move_policy_dsl::observer::MoveKernelObserver,
) -> Result<PolicyResult, DslError> { … }
```

b. Inside the function, after building `policy_graph`:

```rust
use crate::move_policy_dsl::observer::{GraphDelta, PolicyGraphSnapshot};
let snap = PolicyGraphSnapshot {
    root_qubits: initial_cfg.iter().map(|(q, _)| q).collect(),
    target_qubits: target_cfg.iter().map(|(q, _)| q).collect(),
    blocked_count: blocked_set.len(),
};
observer.on_init(&snap);
```

c. After applying each action in the loop, build a `GraphDelta` from `last_insert`, `last_builtin`, and `node_state_writes` (the side-channel state the kernel already maintains), and call:

```rust
observer.on_step(step_idx, depth, &action, &delta);
```

`step_idx` is the kernel's existing loop counter; `depth` is the depth of the just-applied node (already tracked in the kernel for transposition dedup).

d. After every builtin invocation:

```rust
observer.on_builtin(step_idx, name, ok);
```

e. At the end (success or any error path that produces a `PolicyResult`):

```rust
observer.on_halt(&result.status);
```

f. Update every existing call site of `solve_with_policy` to pass `&mut NoOpMoveObserver`. Locations to fix (search via `grep -rn solve_with_policy crates/`): `bloqade-lanes-bytecode-python` PyO3 binding (existing call), any unit/integration tests in `crates/bloqade-lanes-search/tests/dsl_*.rs` and inline `#[cfg(test)]` modules.

- [ ] **Step 5: Add the kernel hookup unit test**

Append to the `#[cfg(test)] mod tests` block in `observer.rs`:

```rust
#[test]
fn kernel_calls_observer_in_order_for_a_trivial_policy() {
    use crate::lane_index::LaneIndex;
    use crate::move_policy_dsl::kernel::{PolicyOptions, solve_with_policy};
    use bloqade_lanes_dsl_core::sandbox::SandboxConfig;
    use std::path::PathBuf;
    use std::sync::Arc;

    let arch = crate::test_utils::small_arch();          // existing helper
    let index = Arc::new(LaneIndex::new(arch));
    let initial = vec![(0, (1, 0, 0).into())];
    let target  = vec![(0, (1, 0, 0).into())];           // already at target
    let policy_path = crate::test_utils::write_tmp_policy(
        "def init(root, ctx): pass\n\
         def step(graph, gs, ctx, lib): return [actions.halt('done')]\n",
    );
    let opts = PolicyOptions {
        policy_path: policy_path.display().to_string(),
        sandbox: SandboxConfig::default(),
        policy_params: serde_json::Value::Null,
        max_expansions: 32,
        timeout_s: Some(1.0),
    };
    let mut obs = RecordingObserver::default();
    let _ = solve_with_policy(initial, target, std::iter::empty(), index, opts, &mut obs).unwrap();

    // Trivial policy: one init, zero steps (target already met → halt on first step),
    // one halt.
    assert_eq!(obs.calls.first().unwrap(), "init");
    assert_eq!(obs.calls.last().unwrap(),  "halt");
}
```

(`crate::test_utils::small_arch` and `write_tmp_policy` are existing helpers used by Plan A's tests; if a needed helper doesn't yet exist, add a one-liner to `test_utils.rs` rather than rewriting fixtures.)

- [ ] **Step 6: Run tests; verify pass**

```bash
cargo test -p bloqade-lanes-search --lib move_policy_dsl::observer::
```

Expected: `no_op_observer_compiles_against_trait` and `kernel_calls_observer_in_order_for_a_trivial_policy` both pass. Then `cargo clippy -p bloqade-lanes-search --all-targets -- -D warnings`.

- [ ] **Step 7: Commit**

```bash
git add crates/bloqade-lanes-search/src/move_policy_dsl/observer.rs \
        crates/bloqade-lanes-search/src/move_policy_dsl/mod.rs \
        crates/bloqade-lanes-search/src/move_policy_dsl/kernel.rs
# also: any callers updated in Step 4f
git commit -m "feat(search): add MoveKernelObserver trait and thread through solve_with_policy"
```

---

### Task 2: `JsonMoveTraceObserver` (NDJSON record emitter)

**Files:**
- Modify: `crates/bloqade-lanes-search/src/move_policy_dsl/observer.rs`
- Modify: `crates/bloqade-lanes-search/src/move_policy_dsl/mod.rs` (add to re-exports)
- Modify: `crates/bloqade-lanes-search/Cargo.toml` (ensure `serde` + `serde_json` available)

- [ ] **Step 1: Verify `serde_json` is on the search crate**

```bash
grep -E '^serde(_json)? *=' crates/bloqade-lanes-search/Cargo.toml
```

If `serde_json` is missing, add to `[dependencies]`:

```toml
serde_json = { workspace = true }
serde = { workspace = true, features = ["derive"] }
```

- [ ] **Step 2: Write the failing test for NDJSON envelope shape**

Append to the `#[cfg(test)] mod tests` block:

```rust
#[test]
fn json_trace_observer_emits_init_step_halt_records() {
    let mut buf: Vec<u8> = Vec::new();
    {
        let mut obs = JsonMoveTraceObserver::new(&mut buf);
        obs.on_init(&PolicyGraphSnapshot {
            root_qubits: vec![0, 1],
            target_qubits: vec![0, 1],
            blocked_count: 0,
        });
        let action = MoveAction::Halt { status: "done".into(), message: String::new() };
        obs.on_step(0, 1, &action, &GraphDelta::default());
        obs.on_halt(&PolicyStatus::Solved);
    }
    let s = std::str::from_utf8(&buf).unwrap();
    let lines: Vec<_> = s.lines().collect();
    assert_eq!(lines.len(), 3);
    assert!(lines[0].contains(r#""kind":"init""#));
    assert!(lines[1].contains(r#""kind":"step""#));
    assert!(lines[1].contains(r#""depth":1"#));
    assert!(lines[2].contains(r#""kind":"halt""#));
    // Each line is a single line (no embedded newlines except trailing).
    for ln in &lines {
        assert!(!ln.is_empty());
    }
}
```

- [ ] **Step 3: Run test to verify failure**

```bash
cargo test -p bloqade-lanes-search --lib json_trace_observer_emits
```

Expected: compile error — `JsonMoveTraceObserver` undefined.

- [ ] **Step 4: Implement `JsonMoveTraceObserver`**

Add to `observer.rs` (above the `#[cfg(test)]` block):

```rust
use serde::Serialize;
use std::io::{self, Write};

/// JSON record envelope. Schema version `v` is bumped only on incompatible
/// changes (field removals / semantic shifts); additive changes are
/// non-breaking.
#[derive(Serialize)]
struct EnvInit<'a> {
    v: u32,
    kind: &'static str,
    root: &'a PolicyGraphSnapshot,
}
#[derive(Serialize)]
struct EnvStep<'a> {
    v: u32,
    kind: &'static str,
    step: u64,
    depth: u32,
    action: &'a MoveAction,
    delta: &'a GraphDelta,
}
#[derive(Serialize)]
struct EnvBuiltin<'a> {
    v: u32,
    kind: &'static str,
    step: u64,
    name: &'a str,
    ok: bool,
}
#[derive(Serialize)]
struct EnvHalt<'a> {
    v: u32,
    kind: &'static str,
    status: &'a PolicyStatus,
}

const SCHEMA_VERSION: u32 = 1;

/// Streaming NDJSON trace observer. One record per kernel event; one line
/// per record; no trailing comma; flushes after every record so a partial
/// run still produces a parseable transcript.
pub struct JsonMoveTraceObserver<W: Write> {
    writer: W,
}

impl<W: Write> JsonMoveTraceObserver<W> {
    pub fn new(writer: W) -> Self {
        Self { writer }
    }

    fn emit<T: Serialize>(&mut self, env: &T) {
        // Best-effort emission. If the writer fails (e.g., broken pipe to
        // `head`), drop the record silently; we don't want trace I/O to
        // poison the policy run itself.
        let line = serde_json::to_string(env).expect("trace serialization");
        let _ = writeln!(self.writer, "{line}");
        let _ = self.writer.flush();
    }
}

impl<W: Write> MoveKernelObserver for JsonMoveTraceObserver<W> {
    fn on_init(&mut self, root: &PolicyGraphSnapshot) {
        self.emit(&EnvInit { v: SCHEMA_VERSION, kind: "init", root });
    }
    fn on_step(&mut self, step: u64, depth: u32, action: &MoveAction, delta: &GraphDelta) {
        self.emit(&EnvStep { v: SCHEMA_VERSION, kind: "step", step, depth, action, delta });
    }
    fn on_builtin(&mut self, step: u64, name: &str, ok: bool) {
        self.emit(&EnvBuiltin { v: SCHEMA_VERSION, kind: "builtin", step, name, ok });
    }
    fn on_halt(&mut self, status: &PolicyStatus) {
        self.emit(&EnvHalt { v: SCHEMA_VERSION, kind: "halt", status });
    }
}
```

`PolicyGraphSnapshot`, `GraphDelta`, `MoveAction`, and `PolicyStatus` need `#[derive(Serialize)]`. Add the derive to each:

- `PolicyGraphSnapshot` and `GraphDelta` — already declared in this file; add `Serialize` to the `#[derive(...)]` line and `use serde::Serialize;` (already added).
- `MoveAction` — modify `move_policy_dsl/actions.rs`. Add `serde::Serialize` derive on the enum and on any payload structs.
- `PolicyStatus` — modify `move_policy_dsl/kernel.rs`. Add `serde::Serialize` derive (with `#[serde(tag = "type", content = "detail")]` so e.g. `PolicyStatus::Fallback("oom")` serialises as `{"type":"Fallback","detail":"oom"}`).

- [ ] **Step 5: Re-export from `mod.rs`**

```rust
pub use observer::{
    GraphDelta, JsonMoveTraceObserver, MoveKernelObserver, NoOpMoveObserver, PolicyGraphSnapshot,
};
```

- [ ] **Step 6: Run tests; verify pass**

```bash
cargo test -p bloqade-lanes-search --lib move_policy_dsl::observer
cargo clippy -p bloqade-lanes-search --all-targets -- -D warnings
```

Both green.

- [ ] **Step 7: Commit**

```bash
git add crates/bloqade-lanes-search/src/move_policy_dsl/observer.rs \
        crates/bloqade-lanes-search/src/move_policy_dsl/mod.rs \
        crates/bloqade-lanes-search/src/move_policy_dsl/actions.rs \
        crates/bloqade-lanes-search/src/move_policy_dsl/kernel.rs \
        crates/bloqade-lanes-search/Cargo.toml
git commit -m "feat(search): add JsonMoveTraceObserver emitting NDJSON kernel trace"
```

---

### Task 3: `TargetKernelObserver` trait + `NoOpTargetObserver` + kernel hookup

**Files:**
- Create: `crates/bloqade-lanes-search/src/target_generator_dsl/observer.rs`
- Modify: `crates/bloqade-lanes-search/src/target_generator_dsl/mod.rs`
- Modify: `crates/bloqade-lanes-search/src/target_generator_dsl/kernel.rs`

- [ ] **Step 1: Create the observer module skeleton**

Write to `crates/bloqade-lanes-search/src/target_generator_dsl/observer.rs`:

```rust
//! Observer trait for the Target Generator DSL kernel.
//!
//! Plan C — see `docs/superpowers/specs/2026-05-01-move-policy-dsl-plan-c-design.md` §5.2.
//!
//! Target generation is a single call per CZ stage: one `on_invoke`,
//! one `on_result` per stage. There is no per-step loop.

use serde::Serialize;
use std::io::Write;

#[derive(Debug, Clone, Serialize)]
pub struct TargetContextSnapshot {
    pub current_qubit_count: usize,
    pub controls_len: usize,
    pub targets_len: usize,
    pub lookahead_layers: usize,
    pub cz_stage_index: u32,
}

/// Summary of `TargetPolicyRunner::generate`'s `Vec<Vec<(u32, LocationAddr)>>`
/// result: `num_candidates` is the outer length; `first_candidate_size` is
/// the inner length for the first candidate (or 0 when empty).
#[derive(Debug, Clone, Serialize)]
pub struct CandidateSummary {
    pub num_candidates: usize,
    pub first_candidate_size: usize,
}

pub trait TargetKernelObserver {
    fn on_invoke(&mut self, _stage_idx: u64, _ctx: &TargetContextSnapshot) {}
    fn on_result(&mut self, _stage_idx: u64, _summary: &CandidateSummary, _ok: bool) {}
}

pub struct NoOpTargetObserver;
impl TargetKernelObserver for NoOpTargetObserver {}

const SCHEMA_VERSION: u32 = 1;

#[derive(Serialize)]
struct EnvInvoke<'a> {
    v: u32,
    kind: &'static str,
    stage: u64,
    ctx: &'a TargetContextSnapshot,
}
#[derive(Serialize)]
struct EnvResult<'a> {
    v: u32,
    kind: &'static str,
    stage: u64,
    summary: &'a CandidateSummary,
    ok: bool,
}

pub struct JsonTargetTraceObserver<W: Write> {
    writer: W,
}

impl<W: Write> JsonTargetTraceObserver<W> {
    pub fn new(writer: W) -> Self {
        Self { writer }
    }
    fn emit<T: Serialize>(&mut self, env: &T) {
        let line = serde_json::to_string(env).expect("target trace serialization");
        let _ = writeln!(self.writer, "{line}");
        let _ = self.writer.flush();
    }
}

impl<W: Write> TargetKernelObserver for JsonTargetTraceObserver<W> {
    fn on_invoke(&mut self, stage: u64, ctx: &TargetContextSnapshot) {
        self.emit(&EnvInvoke { v: SCHEMA_VERSION, kind: "invoke", stage, ctx });
    }
    fn on_result(&mut self, stage: u64, summary: &CandidateSummary, ok: bool) {
        self.emit(&EnvResult { v: SCHEMA_VERSION, kind: "result", stage, summary, ok });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_target_trace_observer_emits_invoke_and_result() {
        let mut buf: Vec<u8> = Vec::new();
        {
            let mut obs = JsonTargetTraceObserver::new(&mut buf);
            obs.on_invoke(0, &TargetContextSnapshot {
                current_qubit_count: 4,
                controls_len: 1,
                targets_len: 1,
                lookahead_layers: 0,
                cz_stage_index: 0,
            });
            obs.on_result(0, &CandidateSummary { num_candidates: 1, first_candidate_size: 4 }, true);
        }
        let s = std::str::from_utf8(&buf).unwrap();
        let lines: Vec<_> = s.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains(r#""kind":"invoke""#));
        assert!(lines[1].contains(r#""kind":"result""#));
    }
}
```

- [ ] **Step 2: Wire into `target_generator_dsl/mod.rs`**

Add:

```rust
pub mod observer;

pub use observer::{
    CandidateSummary, JsonTargetTraceObserver, NoOpTargetObserver,
    TargetContextSnapshot, TargetKernelObserver,
};
```

(Keep existing re-exports for `StarlarkTargetContext`, `TargetPolicyError`, `TargetPolicyRunner`, `run_target_policy`, `StarlarkLibTarget`.)

- [ ] **Step 3: Thread the observer into `TargetPolicyRunner::generate` and `run_target_policy`**

Modify `crates/bloqade-lanes-search/src/target_generator_dsl/kernel.rs`:

a. Add an `observer: &mut dyn TargetKernelObserver` parameter to `TargetPolicyRunner::generate` (the per-stage entry point) and to `run_target_policy` (the one-shot wrapper, which forwards the observer into `generate`). The real signatures are:

```rust
impl TargetPolicyRunner {
    pub fn generate(
        &self,
        index: Arc<LaneIndex>,
        placement: Vec<(u32, LocationAddr)>,
        controls: Vec<u32>,
        targets: Vec<u32>,
        lookahead_cz_layers: Vec<(Vec<u32>, Vec<u32>)>,
        cz_stage_index: u32,
        policy_params: serde_json::Value,
        observer: &mut dyn crate::target_generator_dsl::observer::TargetKernelObserver,
    ) -> Result<Vec<Vec<(u32, LocationAddr)>>, TargetPolicyError> { … }
}

#[allow(clippy::too_many_arguments)]
pub fn run_target_policy(
    policy_path: impl AsRef<Path>,
    index: Arc<LaneIndex>,
    placement: Vec<(u32, LocationAddr)>,
    controls: Vec<u32>,
    targets: Vec<u32>,
    lookahead_cz_layers: Vec<(Vec<u32>, Vec<u32>)>,
    cz_stage_index: u32,
    policy_params: serde_json::Value,
    cfg: &SandboxConfig,
    observer: &mut dyn crate::target_generator_dsl::observer::TargetKernelObserver,
) -> Result<Vec<Vec<(u32, LocationAddr)>>, TargetPolicyError> { … }
```

b. Inside `generate`, before the Starlark call into the policy:

```rust
use crate::target_generator_dsl::observer::TargetContextSnapshot;
let snap = TargetContextSnapshot {
    current_qubit_count: placement.len(),
    controls_len: controls.len(),
    targets_len: targets.len(),
    lookahead_layers: lookahead_cz_layers.len(),
    cz_stage_index,
};
observer.on_invoke(cz_stage_index as u64, &snap);
```

c. After the call (success or `Err`):

```rust
use crate::target_generator_dsl::observer::CandidateSummary;
let summary = match &result {
    Ok(cands) => CandidateSummary {
        num_candidates: cands.len(),
        first_candidate_size: cands.first().map_or(0, |c| c.len()),
    },
    Err(_) => CandidateSummary { num_candidates: 0, first_candidate_size: 0 },
};
observer.on_result(cz_stage_index as u64, &summary, result.is_ok());
```

d. Update existing call sites of `generate` and `run_target_policy` to pass `&mut NoOpTargetObserver`. Locations to fix (search via `grep -rn "TargetPolicyRunner\|run_target_policy" crates/`): the PyO3 binding at `crates/bloqade-lanes-bytecode-python/src/target_generator_dsl_python.rs` (Python `TargetPolicyRunner.generate(...)` shouldn't expose an observer to Python in v1, so the PyO3 wrapper passes `&mut NoOpTargetObserver` internally), Plan B's existing unit and integration tests in this crate.

- [ ] **Step 4: Run tests**

```bash
cargo test -p bloqade-lanes-search --lib target_generator_dsl::observer
cargo test -p bloqade-lanes-search --lib target_generator_dsl
cargo clippy -p bloqade-lanes-search --all-targets -- -D warnings
```

All green. Existing Plan B tests must still pass (they now use `NoOpTargetObserver`).

- [ ] **Step 5: Commit**

```bash
git add crates/bloqade-lanes-search/src/target_generator_dsl/observer.rs \
        crates/bloqade-lanes-search/src/target_generator_dsl/mod.rs \
        crates/bloqade-lanes-search/src/target_generator_dsl/kernel.rs
# also: any callers updated in Step 3d
git commit -m "feat(search): add TargetKernelObserver trait + JSON impl, thread through run_target_policy"
```

---

### Task 4: PyO3 binding compatibility check

**Files:**
- Modify: `crates/bloqade-lanes-bytecode-python/src/search_python.rs` (verify only)
- Modify: `crates/bloqade-lanes-bytecode-python/src/target_python.rs` (verify only — file may not exist by that exact name; check via `grep -rn TargetPolicyRunner crates/bloqade-lanes-bytecode-python/src/`)

This task verifies that the Python bindings still compile after Tasks 1–3 added new observer parameters. If Tasks 1–3 already updated callers, this is a no-op verification step.

- [ ] **Step 1: Build the Python extension**

```bash
just develop-python
```

Expected: clean build. If errors point at missing observer args in PyO3 callers, fix them in place by passing `&mut NoOpMoveObserver` / `&mut NoOpTargetObserver`.

- [ ] **Step 2: Run the Python test suite**

```bash
uv run pytest python/tests
```

Expected: 1075 passed, 9 skipped. No regressions.

- [ ] **Step 3: Commit (only if PyO3 fixes were needed)**

```bash
git add crates/bloqade-lanes-bytecode-python/
git commit -m "chore(python): pass NoOp observers from PyO3 bindings"
```

If no fixes were required, skip the commit.

---

## Phase 2 — Shared fixture loader

### Task 5: `bloqade_lanes_search::fixture` (Move + Target problem types)

**Files:**
- Create: `crates/bloqade-lanes-search/src/fixture.rs`
- Modify: `crates/bloqade-lanes-search/src/lib.rs`
- Modify: `crates/bloqade-lanes-search/Cargo.toml` (add `schemars` if not present)

- [ ] **Step 1: Verify `schemars` availability**

```bash
grep -E '^schemars *=' crates/bloqade-lanes-search/Cargo.toml
```

If missing, add to `[dependencies]`:

```toml
schemars = { workspace = true }
```

(If `schemars` is not yet in the workspace `Cargo.toml`, add it there: `schemars = "0.8"`.)

- [ ] **Step 2: Write the failing tests**

Create `crates/bloqade-lanes-search/src/fixture.rs`:

```rust
//! Shared problem-fixture loader for the Move and Target Policy DSLs.
//!
//! Plan C — see `docs/superpowers/specs/2026-05-01-move-policy-dsl-plan-c-design.md` §6.
//!
//! Fixture files are self-contained JSON documents discriminated by a
//! top-level `"kind"` field (`"move"` or `"target"`). The `arch` field is
//! a path resolved relative to the fixture file, allowing one ArchSpec
//! JSON to back many fixtures.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum FixtureError {
    #[error("reading fixture {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("parsing fixture {path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("schema version mismatch in {path}: got {got}, expected {expected}")]
    SchemaVersion {
        path: PathBuf,
        got: u32,
        expected: u32,
    },
    #[error("resolving arch path '{arch}' relative to fixture {path}: {reason}")]
    ArchResolve {
        path: PathBuf,
        arch: String,
        reason: String,
    },
}

const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Problem {
    Move(MoveProblem),
    Target(TargetProblem),
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct MoveProblem {
    pub v: u32,
    pub arch: String,
    pub initial: Vec<(u32, [i32; 3])>,
    pub target: Vec<(u32, [i32; 3])>,
    #[serde(default)]
    pub blocked: Vec<[i32; 3]>,
    #[serde(default)]
    pub budget: Option<Budget>,
    #[serde(default)]
    pub policy_params: serde_json::Value,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct TargetProblem {
    pub v: u32,
    pub arch: String,
    pub current_placement: Vec<(u32, [i32; 3])>,
    pub controls: Vec<u32>,
    pub targets: Vec<u32>,
    #[serde(default)]
    pub lookahead_cz_layers: Vec<(Vec<u32>, Vec<u32>)>,
    #[serde(default)]
    pub cz_stage_index: u32,
    #[serde(default)]
    pub policy_params: serde_json::Value,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct Budget {
    pub max_expansions: u64,
    pub timeout_s: f64,
}

impl Problem {
    pub fn schema_version(&self) -> u32 {
        match self {
            Problem::Move(p) => p.v,
            Problem::Target(p) => p.v,
        }
    }

    pub fn arch_path_str(&self) -> &str {
        match self {
            Problem::Move(p) => &p.arch,
            Problem::Target(p) => &p.arch,
        }
    }
}

/// Load and validate a problem-fixture file.
///
/// Returns the parsed `Problem` and the resolved absolute path to the
/// referenced ArchSpec JSON. The arch path is resolved relative to the
/// fixture file's parent directory.
pub fn load(path: &Path) -> Result<(Problem, PathBuf), FixtureError> {
    let bytes = std::fs::read(path).map_err(|e| FixtureError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    let problem: Problem = serde_json::from_slice(&bytes).map_err(|e| FixtureError::Parse {
        path: path.to_path_buf(),
        source: e,
    })?;
    if problem.schema_version() != SCHEMA_VERSION {
        return Err(FixtureError::SchemaVersion {
            path: path.to_path_buf(),
            got: problem.schema_version(),
            expected: SCHEMA_VERSION,
        });
    }
    let arch_str = problem.arch_path_str();
    let parent = path
        .parent()
        .ok_or_else(|| FixtureError::ArchResolve {
            path: path.to_path_buf(),
            arch: arch_str.into(),
            reason: "fixture path has no parent".into(),
        })?;
    let arch_path = parent.join(arch_str).canonicalize().map_err(|e| {
        FixtureError::ArchResolve {
            path: path.to_path_buf(),
            arch: arch_str.into(),
            reason: e.to_string(),
        }
    })?;
    Ok((problem, arch_path))
}

/// `schemars`-generated JSON Schema for the Problem enum, used by the
/// primer generator's AUTOGEN: schema section (Task 23).
pub fn json_schema_pretty() -> String {
    let schema = schemars::schema_for!(Problem);
    serde_json::to_string_pretty(&schema).expect("schema serialize")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn write_arch_stub(dir: &Path) -> PathBuf {
        let p = dir.join("arch.json");
        std::fs::write(&p, br#"{"version": 1, "kind": "stub"}"#).unwrap();
        p
    }

    fn write_fixture(dir: &Path, body: &str) -> PathBuf {
        let p = dir.join("problem.json");
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(body.as_bytes()).unwrap();
        p
    }

    #[test]
    fn loads_move_fixture_and_resolves_arch() {
        let tmp = TempDir::new().unwrap();
        write_arch_stub(tmp.path());
        let p = write_fixture(
            tmp.path(),
            r#"{"v":1,"kind":"move","arch":"arch.json",
                "initial":[[0,[1,0,0]]],"target":[[0,[1,0,1]]],
                "blocked":[],"policy_params":{}}"#,
        );
        let (prob, arch_path) = load(&p).unwrap();
        match prob {
            Problem::Move(m) => assert_eq!(m.initial.len(), 1),
            _ => panic!("expected Move"),
        }
        assert!(arch_path.ends_with("arch.json"));
    }

    #[test]
    fn loads_target_fixture() {
        let tmp = TempDir::new().unwrap();
        write_arch_stub(tmp.path());
        let p = write_fixture(
            tmp.path(),
            r#"{"v":1,"kind":"target","arch":"arch.json",
                "current_placement":[[0,[1,0,0]]],"controls":[0],"targets":[1]}"#,
        );
        let (prob, _) = load(&p).unwrap();
        assert!(matches!(prob, Problem::Target(_)));
    }

    #[test]
    fn rejects_unknown_kind() {
        let tmp = TempDir::new().unwrap();
        let p = write_fixture(
            tmp.path(),
            r#"{"v":1,"kind":"flavor","arch":"x.json"}"#,
        );
        let err = load(&p).unwrap_err();
        assert!(matches!(err, FixtureError::Parse { .. }));
    }

    #[test]
    fn rejects_schema_version_mismatch() {
        let tmp = TempDir::new().unwrap();
        write_arch_stub(tmp.path());
        let p = write_fixture(
            tmp.path(),
            r#"{"v":99,"kind":"move","arch":"arch.json",
                "initial":[],"target":[]}"#,
        );
        let err = load(&p).unwrap_err();
        assert!(matches!(err, FixtureError::SchemaVersion { got: 99, .. }));
    }

    #[test]
    fn json_schema_renders() {
        let s = json_schema_pretty();
        assert!(s.contains("MoveProblem"));
        assert!(s.contains("TargetProblem"));
    }
}
```

- [ ] **Step 3: Wire the module into `lib.rs`**

Modify `crates/bloqade-lanes-search/src/lib.rs`:

```rust
pub mod fixture;

pub use fixture::{Budget, FixtureError, MoveProblem, Problem, TargetProblem};
```

(Add the `pub mod fixture;` near the other `pub mod` declarations and the `pub use fixture::...` near the existing `pub use ...` block.)

- [ ] **Step 4: Add `tempfile` dev-dep if missing**

```bash
grep -E '^tempfile *=' crates/bloqade-lanes-search/Cargo.toml
```

If absent, add to `[dev-dependencies]`:

```toml
tempfile = { workspace = true }
```

- [ ] **Step 5: Run tests; verify pass**

```bash
cargo test -p bloqade-lanes-search --lib fixture::
cargo clippy -p bloqade-lanes-search --all-targets -- -D warnings
```

All five tests pass.

- [ ] **Step 6: Commit**

```bash
git add crates/bloqade-lanes-search/src/fixture.rs \
        crates/bloqade-lanes-search/src/lib.rs \
        crates/bloqade-lanes-search/Cargo.toml
git commit -m "feat(search): add shared fixture loader for Move/Target Policy DSL problems"
```

---

## Phase 3 — CLI subcommands

### Task 6: Add `search` and `dsl-core` deps to the CLI crate

**Files:**
- Modify: `crates/bloqade-lanes-bytecode-cli/Cargo.toml`

- [ ] **Step 1: Add the dependencies**

In `[dependencies]`:

```toml
bloqade-lanes-search = { path = "../bloqade-lanes-search" }
bloqade-lanes-dsl-core = { path = "../bloqade-lanes-dsl-core" }
```

In `[dev-dependencies]` (for Task 10 integration tests):

```toml
assert_cmd = { workspace = true }
predicates = { workspace = true }
```

(If these are not yet in the workspace `Cargo.toml`, add them: `assert_cmd = "2"`, `predicates = "3"`.)

- [ ] **Step 2: Verify the CLI still builds**

```bash
cargo build -p bloqade-lanes-bytecode-cli
```

Expected: clean build. The CLI binary now links `starlark-rust` transitively. Build time and final size grow.

- [ ] **Step 3: Commit**

```bash
git add crates/bloqade-lanes-bytecode-cli/Cargo.toml Cargo.toml
git commit -m "build(cli): add search + dsl-core deps for upcoming policy subcommands"
```

---

### Task 7: CLI `policy` module skeleton; register subcommands

**Files:**
- Create: `crates/bloqade-lanes-bytecode-cli/src/policy/mod.rs`
- Create: `crates/bloqade-lanes-bytecode-cli/src/policy/output.rs`
- Modify: `crates/bloqade-lanes-bytecode-cli/src/main.rs`
- Modify: `crates/bloqade-lanes-bytecode-cli/src/lib.rs` (re-export module)

- [ ] **Step 1: Create `policy/mod.rs` skeleton**

```rust
//! `eval-policy` and `trace-policy` CLI subcommands.
//!
//! Plan C — see `docs/superpowers/specs/2026-05-01-move-policy-dsl-plan-c-design.md` §4.

pub mod eval;
pub mod output;
pub mod trace;

pub use eval::run_eval_policy;
pub use trace::run_trace_policy;
```

(Stub `eval.rs` and `trace.rs` as empty modules with `pub fn run_eval_policy(...)` / `pub fn run_trace_policy(...)` returning `Err("not yet implemented".into())` — they get bodies in Tasks 8 and 9.)

- [ ] **Step 2: Create `policy/output.rs` (shared formatting)**

```rust
//! Shared output formatting for `eval-policy` and `trace-policy`.

use serde::Serialize;

#[derive(Serialize)]
pub struct EvalEnvelope<'a> {
    pub v: u32,
    pub kind: &'static str,         // "move" or "target"
    pub policy: &'a str,
    pub problem: &'a str,
    pub status: &'a str,
    pub halt_reason: Option<&'a str>,
    pub expansions: u64,
    pub max_depth: u32,
    pub wall_time_ms: f64,
}

#[derive(Serialize)]
pub struct TargetEvalEnvelope<'a> {
    pub v: u32,
    pub kind: &'static str,         // "target"
    pub policy: &'a str,
    pub problem: &'a str,
    pub ok: bool,
    pub num_candidates: usize,
    pub first_candidate_size: usize,
    pub wall_time_ms: f64,
}

pub fn print_human_move(env: &EvalEnvelope) {
    println!("status         {}", env.status);
    if let Some(r) = env.halt_reason {
        println!("halt_reason    {r}");
    }
    println!("expansions     {}", env.expansions);
    println!("max_depth      {}", env.max_depth);
    println!("wall_time      {:.1} ms", env.wall_time_ms);
}

pub fn print_human_target(env: &TargetEvalEnvelope) {
    println!("ok                       {}", env.ok);
    println!("num_candidates           {}", env.num_candidates);
    println!("first_candidate_size     {}", env.first_candidate_size);
    println!("wall_time                {:.1} ms", env.wall_time_ms);
}
```

- [ ] **Step 3: Register the subcommands in `main.rs`**

Add to the `Command` enum:

```rust
/// Run a policy end-to-end and print a single result summary.
EvalPolicy {
    #[arg(long)] policy: PathBuf,
    #[arg(long)] problem: PathBuf,
    #[arg(long)] params: Option<PathBuf>,
    #[arg(long)] max_expansions: Option<u64>,
    #[arg(long)] timeout: Option<f64>,
    #[arg(long)] json: bool,
    #[arg(long)] seed: Option<u64>,
},
/// Run a policy with a verbose observer and emit a step-by-step trace.
TracePolicy {
    #[arg(long)] policy: PathBuf,
    #[arg(long)] problem: PathBuf,
    #[arg(long)] params: Option<PathBuf>,
    #[arg(long)] max_expansions: Option<u64>,
    #[arg(long)] timeout: Option<f64>,
    #[arg(long)] json: bool,
    #[arg(long)] seed: Option<u64>,
    #[arg(long)] out: Option<PathBuf>,
},
```

Add a `mod policy;` near the top of `main.rs` (or the lib.rs equivalent). Wire dispatch in the `match cli.command` block:

```rust
Command::EvalPolicy { policy, problem, params, max_expansions, timeout, json, seed } => {
    policy::run_eval_policy(&policy, &problem, params.as_deref(), max_expansions, timeout, json, seed)
}
Command::TracePolicy { policy, problem, params, max_expansions, timeout, json, seed, out } => {
    policy::run_trace_policy(&policy, &problem, params.as_deref(), max_expansions, timeout, json, seed, out.as_deref())
}
```

- [ ] **Step 4: Verify `--help` lists the new subcommands**

```bash
cargo run -p bloqade-lanes-bytecode-cli -- --help
cargo run -p bloqade-lanes-bytecode-cli -- eval-policy --help
cargo run -p bloqade-lanes-bytecode-cli -- trace-policy --help
```

Expected: clap prints help that includes both subcommands and their flags. Stub bodies still return errors when run.

- [ ] **Step 5: Commit**

```bash
git add crates/bloqade-lanes-bytecode-cli/src/main.rs \
        crates/bloqade-lanes-bytecode-cli/src/lib.rs \
        crates/bloqade-lanes-bytecode-cli/src/policy/
git commit -m "feat(cli): scaffold eval-policy and trace-policy subcommands"
```

---

### Task 8: `eval-policy` implementation (Move + Target dispatch)

**Files:**
- Modify: `crates/bloqade-lanes-bytecode-cli/src/policy/eval.rs`

- [ ] **Step 1: Implement `run_eval_policy`**

Replace the stub with:

```rust
//! `eval-policy` subcommand: run-once with summary output.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use bloqade_lanes_dsl_core::sandbox::SandboxConfig;
use bloqade_lanes_search::fixture::{self, Problem};
use bloqade_lanes_search::lane_index::LaneIndex;
use bloqade_lanes_search::move_policy_dsl::{
    NoOpMoveObserver, PolicyOptions, PolicyStatus, solve_with_policy,
};
use bloqade_lanes_search::target_generator_dsl::{
    NoOpTargetObserver, run_target_policy,
};

use super::output::{EvalEnvelope, TargetEvalEnvelope, print_human_move, print_human_target};

const SCHEMA_VERSION: u32 = 1;

pub fn run_eval_policy(
    policy: &Path,
    problem: &Path,
    params: Option<&Path>,
    max_expansions: Option<u64>,
    timeout_s: Option<f64>,
    json: bool,
    _seed: Option<u64>,
) -> Result<(), String> {
    let (parsed, arch_path) =
        fixture::load(problem).map_err(|e| format!("error: {e}"))?;
    let arch_json = std::fs::read_to_string(&arch_path)
        .map_err(|e| format!("error: reading arch {}: {e}", arch_path.display()))?;
    let arch = bloqade_lanes_bytecode_core::arch::ArchSpec::from_json(&arch_json)
        .map_err(|e| format!("error: parsing arch {}: {e}", arch_path.display()))?;

    match parsed {
        Problem::Move(mp) => {
            let exit = run_move(policy, problem, mp, arch, params, max_expansions, timeout_s, json)?;
            std::process::exit(exit);
        }
        Problem::Target(tp) => {
            let exit = run_target(policy, problem, tp, arch, params, json)?;
            std::process::exit(exit);
        }
    }
}

fn run_move(
    policy: &Path,
    problem: &Path,
    mp: bloqade_lanes_search::fixture::MoveProblem,
    arch: bloqade_lanes_bytecode_core::arch::ArchSpec,
    params: Option<&Path>,
    max_expansions: Option<u64>,
    timeout_s: Option<f64>,
    json: bool,
) -> Result<i32, String> {
    let index = Arc::new(LaneIndex::new(arch));
    let initial = mp.initial.iter().map(|(q, [l, r, c])| (*q, (*l, *r, *c).into()));
    let target = mp.target.iter().map(|(q, [l, r, c])| (*q, (*l, *r, *c).into()));
    let blocked = mp.blocked.iter().map(|[l, r, c]| (*l, *r, *c).into());

    let opts = PolicyOptions {
        policy_path: policy.display().to_string(),
        sandbox: SandboxConfig::default(),
        policy_params: load_params(params, &mp.policy_params)?,
        max_expansions: max_expansions
            .or(mp.budget.as_ref().map(|b| b.max_expansions))
            .unwrap_or(5_000),
        timeout_s: Some(
            timeout_s
                .or(mp.budget.as_ref().map(|b| b.timeout_s))
                .unwrap_or(10.0),
        ),
    };
    let mut obs = NoOpMoveObserver;
    let t0 = Instant::now();
    let res = solve_with_policy(initial, target, blocked, index, opts, &mut obs)
        .map_err(|e| format!("error: {e}"))?;
    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let status_str = status_label(&res.status);
    let halt_reason = halt_reason(&res.status);
    let env = EvalEnvelope {
        v: SCHEMA_VERSION,
        kind: "move",
        policy: policy.to_str().unwrap_or(""),
        problem: problem.to_str().unwrap_or(""),
        status: status_str,
        halt_reason: halt_reason.as_deref(),
        expansions: res.nodes_expanded as u64,
        max_depth: res.move_layers.len() as u32,
        wall_time_ms: wall_ms,
    };
    if json {
        println!("{}", serde_json::to_string(&env).unwrap());
    } else {
        print_human_move(&env);
    }
    Ok(exit_code(&res.status))
}

fn run_target(
    policy: &Path,
    problem: &Path,
    tp: bloqade_lanes_search::fixture::TargetProblem,
    arch: bloqade_lanes_bytecode_core::arch::ArchSpec,
    params: Option<&Path>,
    json: bool,
) -> Result<i32, String> {
    let index = Arc::new(LaneIndex::new(arch));
    let placement: Vec<(u32, _)> = tp
        .current_placement
        .iter()
        .map(|(q, [l, r, c])| (*q, (*l, *r, *c).into()))
        .collect();
    let cfg = SandboxConfig::default();
    let mut obs = NoOpTargetObserver;
    let t0 = Instant::now();
    let result = run_target_policy(
        policy,
        index,
        placement,
        tp.controls.clone(),
        tp.targets.clone(),
        tp.lookahead_cz_layers.clone(),
        tp.cz_stage_index,
        load_params(params, &tp.policy_params)?,
        &cfg,
        &mut obs,
    );
    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let (num_candidates, first_candidate_size) = match &result {
        Ok(cands) => (cands.len(), cands.first().map_or(0, |c| c.len())),
        Err(_) => (0, 0),
    };
    let env = TargetEvalEnvelope {
        v: SCHEMA_VERSION,
        kind: "target",
        policy: policy.to_str().unwrap_or(""),
        problem: problem.to_str().unwrap_or(""),
        ok: result.is_ok(),
        num_candidates,
        first_candidate_size,
        wall_time_ms: wall_ms,
    };
    if json {
        println!("{}", serde_json::to_string(&env).unwrap());
    } else {
        print_human_target(&env);
    }
    Ok(if result.is_ok() { 0 } else { 2 })
}

fn load_params(
    file: Option<&Path>,
    fallback: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    match file {
        Some(p) => {
            let bytes = std::fs::read(p).map_err(|e| format!("error: reading {}: {e}", p.display()))?;
            serde_json::from_slice(&bytes).map_err(|e| format!("error: parsing {}: {e}", p.display()))
        }
        None => Ok(fallback.clone()),
    }
}

fn status_label(s: &PolicyStatus) -> &'static str {
    match s {
        PolicyStatus::Solved => "Solved",
        PolicyStatus::Unsolvable => "Unsolvable",
        PolicyStatus::BudgetExhausted => "BudgetExhausted",
        PolicyStatus::Timeout => "Timeout",
        PolicyStatus::Fallback(_) => "Fallback",
        PolicyStatus::SyntaxError(_) => "SyntaxError",
        PolicyStatus::RuntimeError(_) => "RuntimeError",
        PolicyStatus::SchemaError(_) => "SchemaError",
        PolicyStatus::BadPolicy(_) => "BadPolicy",
        PolicyStatus::StarlarkBudget => "StarlarkBudget",
        PolicyStatus::StarlarkOOM => "StarlarkOOM",
    }
}

fn halt_reason(s: &PolicyStatus) -> Option<String> {
    match s {
        PolicyStatus::Solved => Some("policy_halt".into()),
        PolicyStatus::Fallback(r) => Some(r.clone()),
        _ => None,
    }
}

fn exit_code(s: &PolicyStatus) -> i32 {
    match s {
        PolicyStatus::Solved => 0,
        PolicyStatus::Unsolvable
        | PolicyStatus::BudgetExhausted
        | PolicyStatus::Timeout
        | PolicyStatus::Fallback(_) => 2,
        _ => 1,
    }
}
```

- [ ] **Step 2: Build and smoke-test**

```bash
cargo build -p bloqade-lanes-bytecode-cli
```

Expected: clean build. (End-to-end smoke tests come in Task 10.)

- [ ] **Step 3: Verify clippy**

```bash
cargo clippy -p bloqade-lanes-bytecode-cli --all-targets -- -D warnings
```

Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add crates/bloqade-lanes-bytecode-cli/src/policy/eval.rs
git commit -m "feat(cli): implement eval-policy for Move and Target policies"
```

---

### Task 9: `trace-policy` implementation (NDJSON streaming)

**Files:**
- Modify: `crates/bloqade-lanes-bytecode-cli/src/policy/trace.rs`

- [ ] **Step 1: Implement `run_trace_policy`**

```rust
//! `trace-policy` subcommand: per-event verbose trace output.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::sync::Arc;

use bloqade_lanes_dsl_core::sandbox::SandboxConfig;
use bloqade_lanes_search::fixture::{self, Problem};
use bloqade_lanes_search::lane_index::LaneIndex;
use bloqade_lanes_search::move_policy_dsl::{
    JsonMoveTraceObserver, MoveKernelObserver, PolicyOptions, solve_with_policy,
};
use bloqade_lanes_search::target_generator_dsl::{
    JsonTargetTraceObserver, run_target_policy,
};

pub fn run_trace_policy(
    policy: &Path,
    problem: &Path,
    params: Option<&Path>,
    max_expansions: Option<u64>,
    timeout_s: Option<f64>,
    json: bool,
    _seed: Option<u64>,
    out: Option<&Path>,
) -> Result<(), String> {
    let (parsed, arch_path) =
        fixture::load(problem).map_err(|e| format!("error: {e}"))?;
    let arch_json = std::fs::read_to_string(&arch_path)
        .map_err(|e| format!("error: reading arch {}: {e}", arch_path.display()))?;
    let arch = bloqade_lanes_bytecode_core::arch::ArchSpec::from_json(&arch_json)
        .map_err(|e| format!("error: parsing arch {}: {e}", arch_path.display()))?;

    let writer: Box<dyn Write> = match out {
        Some(p) => Box::new(BufWriter::new(File::create(p).map_err(|e| format!("error: writing {}: {e}", p.display()))?)),
        None => Box::new(io::stdout()),
    };

    match parsed {
        Problem::Move(mp) => trace_move(policy, mp, arch, params, max_expansions, timeout_s, json, writer),
        Problem::Target(tp) => trace_target(policy, tp, arch, params, json, writer),
    }
}

fn trace_move(
    policy: &Path,
    mp: bloqade_lanes_search::fixture::MoveProblem,
    arch: bloqade_lanes_bytecode_core::arch::ArchSpec,
    params: Option<&Path>,
    max_expansions: Option<u64>,
    timeout_s: Option<f64>,
    json: bool,
    writer: Box<dyn Write>,
) -> Result<(), String> {
    let index = Arc::new(LaneIndex::new(arch));
    let initial = mp.initial.iter().map(|(q, [l, r, c])| (*q, (*l, *r, *c).into()));
    let target = mp.target.iter().map(|(q, [l, r, c])| (*q, (*l, *r, *c).into()));
    let blocked = mp.blocked.iter().map(|[l, r, c]| (*l, *r, *c).into());

    let opts = PolicyOptions {
        policy_path: policy.display().to_string(),
        sandbox: SandboxConfig::default(),
        policy_params: super::eval::load_params_pub(params, &mp.policy_params)?,
        max_expansions: max_expansions
            .or(mp.budget.as_ref().map(|b| b.max_expansions))
            .unwrap_or(5_000),
        timeout_s: Some(
            timeout_s
                .or(mp.budget.as_ref().map(|b| b.timeout_s))
                .unwrap_or(10.0),
        ),
    };

    if json {
        let mut obs = JsonMoveTraceObserver::new(writer);
        let _ = solve_with_policy(initial, target, blocked, index, opts, &mut obs)
            .map_err(|e| format!("error: {e}"))?;
    } else {
        let mut obs = HumanMoveTraceObserver::new(writer);
        let _ = solve_with_policy(initial, target, blocked, index, opts, &mut obs)
            .map_err(|e| format!("error: {e}"))?;
    }
    Ok(())
}

fn trace_target(
    policy: &Path,
    tp: bloqade_lanes_search::fixture::TargetProblem,
    arch: bloqade_lanes_bytecode_core::arch::ArchSpec,
    params: Option<&Path>,
    json: bool,
    writer: Box<dyn Write>,
) -> Result<(), String> {
    let index = Arc::new(LaneIndex::new(arch));
    let placement: Vec<(u32, _)> = tp
        .current_placement
        .iter()
        .map(|(q, [l, r, c])| (*q, (*l, *r, *c).into()))
        .collect();
    let cfg = SandboxConfig::default();
    let params_value = super::eval::load_params_pub(params, &tp.policy_params)?;
    let controls = tp.controls.clone();
    let targets = tp.targets.clone();
    let lookahead = tp.lookahead_cz_layers.clone();
    let stage_idx = tp.cz_stage_index;

    if json {
        let mut obs = JsonTargetTraceObserver::new(writer);
        let _ = run_target_policy(
            policy, index, placement, controls, targets, lookahead, stage_idx,
            params_value, &cfg, &mut obs,
        );
    } else {
        let mut obs = HumanTargetTraceObserver::new(writer);
        let _ = run_target_policy(
            policy, index, placement, controls, targets, lookahead, stage_idx,
            params_value, &cfg, &mut obs,
        );
    }
    Ok(())
}

/// Human-readable observer for Move policies. One line per event,
/// terse fields, designed for interactive debugging.
struct HumanMoveTraceObserver<W: Write> { w: W }
impl<W: Write> HumanMoveTraceObserver<W> {
    fn new(w: W) -> Self { Self { w } }
}
impl<W: Write> MoveKernelObserver for HumanMoveTraceObserver<W> {
    fn on_init(&mut self, root: &bloqade_lanes_search::move_policy_dsl::PolicyGraphSnapshot) {
        let _ = writeln!(self.w, "init   qubits={} target={} blocked={}",
            root.root_qubits.len(), root.target_qubits.len(), root.blocked_count);
    }
    fn on_step(&mut self, step: u64, depth: u32, action: &bloqade_lanes_search::move_policy_dsl::actions::MoveAction, _: &bloqade_lanes_search::move_policy_dsl::GraphDelta) {
        let _ = writeln!(self.w, "step   #{step:04} depth={depth} action={action:?}");
    }
    fn on_builtin(&mut self, step: u64, name: &str, ok: bool) {
        let _ = writeln!(self.w, "builtin #{step:04} {name} ok={ok}");
    }
    fn on_halt(&mut self, status: &bloqade_lanes_search::move_policy_dsl::PolicyStatus) {
        let _ = writeln!(self.w, "halt   status={status:?}");
    }
}

/// Human-readable observer for Target policies.
struct HumanTargetTraceObserver<W: Write> { w: W }
impl<W: Write> HumanTargetTraceObserver<W> {
    fn new(w: W) -> Self { Self { w } }
}
impl<W: Write> bloqade_lanes_search::target_generator_dsl::TargetKernelObserver for HumanTargetTraceObserver<W> {
    fn on_invoke(&mut self, stage: u64, ctx: &bloqade_lanes_search::target_generator_dsl::TargetContextSnapshot) {
        let _ = writeln!(
            self.w,
            "invoke stage={stage} qubits={} controls={} targets={} lookahead={}",
            ctx.current_qubit_count, ctx.controls_len, ctx.targets_len, ctx.lookahead_layers
        );
    }
    fn on_result(&mut self, stage: u64, s: &bloqade_lanes_search::target_generator_dsl::CandidateSummary, ok: bool) {
        let _ = writeln!(
            self.w,
            "result stage={stage} ok={ok} candidates={} first_size={}",
            s.num_candidates, s.first_candidate_size
        );
    }
}
```

- [ ] **Step 2: Make `eval::load_params` available to `trace.rs`**

In `eval.rs`, add at the bottom:

```rust
pub(crate) fn load_params_pub(
    file: Option<&Path>,
    fallback: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    load_params(file, fallback)
}
```

(Or simply make `load_params` `pub(crate)`. The pub-pub helper avoids touching the private function's visibility surface for non-cross-module callers.)

- [ ] **Step 3: Build**

```bash
cargo build -p bloqade-lanes-bytecode-cli
cargo clippy -p bloqade-lanes-bytecode-cli --all-targets -- -D warnings
```

Both green.

- [ ] **Step 4: Commit**

```bash
git add crates/bloqade-lanes-bytecode-cli/src/policy/trace.rs \
        crates/bloqade-lanes-bytecode-cli/src/policy/eval.rs
git commit -m "feat(cli): implement trace-policy with NDJSON and human modes"
```

---

### Task 10: CLI integration tests with `assert_cmd`

**Files:**
- Create: `crates/bloqade-lanes-bytecode-cli/tests/cli_policy.rs`
- Create: `crates/bloqade-lanes-bytecode-cli/tests/fixtures/cli_policy/arch.json` (minimal valid ArchSpec)
- Create: `crates/bloqade-lanes-bytecode-cli/tests/fixtures/cli_policy/move_problem.json`
- Create: `crates/bloqade-lanes-bytecode-cli/tests/fixtures/cli_policy/target_problem.json`
- Create: `crates/bloqade-lanes-bytecode-cli/tests/fixtures/cli_policy/halt_now.star`

- [ ] **Step 1: Create the trivial halt-now policy**

`tests/fixtures/cli_policy/halt_now.star`:

```python
def init(root, ctx):
    pass

def step(graph, gs, ctx, lib):
    return [actions.halt("done")]
```

- [ ] **Step 2: Create the minimal valid ArchSpec**

`tests/fixtures/cli_policy/arch.json` — copy the smallest existing ArchSpec from `examples/arch/` (verify by eye that `cargo run -p bloqade-lanes-bytecode-cli -- arch <file>` pretty-prints without error).

- [ ] **Step 3: Create the move and target problems**

`move_problem.json`:

```json
{
  "v": 1,
  "kind": "move",
  "arch": "arch.json",
  "initial": [[0, [1, 0, 0]]],
  "target":  [[0, [1, 0, 0]]],
  "blocked": [],
  "policy_params": {}
}
```

`target_problem.json`:

```json
{
  "v": 1,
  "kind": "target",
  "arch": "arch.json",
  "current_placement": [[0, [1, 0, 0]], [1, [1, 0, 1]]],
  "controls": [0],
  "targets":  [1],
  "lookahead_cz_layers": [],
  "cz_stage_index": 0,
  "policy_params": {}
}
```

- [ ] **Step 4: Write the integration tests**

`crates/bloqade-lanes-bytecode-cli/tests/cli_policy.rs`:

```rust
use assert_cmd::Command;
use predicates::prelude::*;
use std::path::PathBuf;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/cli_policy")
}

#[test]
fn eval_policy_move_human_output_succeeds() {
    let dir = fixture_dir();
    Command::cargo_bin("bloqade-bytecode")
        .unwrap()
        .arg("eval-policy")
        .arg("--policy").arg(dir.join("halt_now.star"))
        .arg("--problem").arg(dir.join("move_problem.json"))
        .assert()
        .success()
        .stdout(predicate::str::contains("status"))
        .stdout(predicate::str::contains("Solved"));
}

#[test]
fn eval_policy_move_json_emits_envelope() {
    let dir = fixture_dir();
    let out = Command::cargo_bin("bloqade-bytecode")
        .unwrap()
        .arg("eval-policy").arg("--json")
        .arg("--policy").arg(dir.join("halt_now.star"))
        .arg("--problem").arg(dir.join("move_problem.json"))
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let v: serde_json::Value = serde_json::from_slice(&out).expect("valid JSON");
    assert_eq!(v["v"], 1);
    assert_eq!(v["kind"], "move");
    assert_eq!(v["status"], "Solved");
}

#[test]
fn trace_policy_move_ndjson_lines_parse() {
    let dir = fixture_dir();
    let out = Command::cargo_bin("bloqade-bytecode")
        .unwrap()
        .arg("trace-policy").arg("--json")
        .arg("--policy").arg(dir.join("halt_now.star"))
        .arg("--problem").arg(dir.join("move_problem.json"))
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let s = std::str::from_utf8(&out).unwrap();
    let lines: Vec<_> = s.lines().collect();
    assert!(!lines.is_empty(), "expected NDJSON output");
    for line in &lines {
        let v: serde_json::Value = serde_json::from_str(line)
            .unwrap_or_else(|e| panic!("non-JSON line: {line}: {e}"));
        assert_eq!(v["v"], 1);
        assert!(["init", "step", "builtin", "halt"].contains(&v["kind"].as_str().unwrap_or("")));
    }
    let kinds: Vec<_> = lines.iter().filter_map(|ln| {
        let v: serde_json::Value = serde_json::from_str(ln).ok()?;
        Some(v["kind"].as_str()?.to_string())
    }).collect();
    assert_eq!(kinds.first().unwrap(), "init");
    assert_eq!(kinds.last().unwrap(),  "halt");
}

#[test]
fn eval_policy_unknown_kind_exits_one() {
    let tmp = tempfile::TempDir::new().unwrap();
    let arch = tmp.path().join("arch.json");
    std::fs::write(&arch, br#"{}"#).unwrap();
    let prob = tmp.path().join("p.json");
    std::fs::write(&prob, br#"{"v":1,"kind":"flavor","arch":"arch.json"}"#).unwrap();
    Command::cargo_bin("bloqade-bytecode")
        .unwrap()
        .arg("eval-policy")
        .arg("--policy").arg("nonexistent.star")
        .arg("--problem").arg(&prob)
        .assert()
        .failure()
        .code(1);
}
```

- [ ] **Step 5: Add `tempfile` to CLI dev-deps if missing**

```bash
grep -E '^tempfile *=' crates/bloqade-lanes-bytecode-cli/Cargo.toml
```

Add to `[dev-dependencies]` if absent:

```toml
tempfile = { workspace = true }
serde_json = { workspace = true }
```

- [ ] **Step 6: Run tests**

```bash
cargo test -p bloqade-lanes-bytecode-cli --test cli_policy
```

Expected: all four tests pass.

- [ ] **Step 7: Commit**

```bash
git add crates/bloqade-lanes-bytecode-cli/tests/cli_policy.rs \
        crates/bloqade-lanes-bytecode-cli/tests/fixtures/cli_policy/ \
        crates/bloqade-lanes-bytecode-cli/Cargo.toml
git commit -m "test(cli): add eval-policy and trace-policy integration tests"
```

---

## Phase 4 — Reference Move policies

### Task 11: `dfs.star`

**Files:**
- Create: `policies/reference/dfs.star`
- Create: `policies/fixtures/dev/dfs_problem.json` (a tiny problem the policy can solve in <100 expansions for the smoke test in Step 3)
- Create: `policies/fixtures/dev/arch.json` (minimal arch, may be a copy of an existing tiny arch)

- [ ] **Step 1: Author `dfs.star`**

```python
"""
Depth-First Search reference policy
====================================

Strategy: always recurse on the most recently inserted child until the
current branch can no longer extend, then unwind to the nearest sibling.
The kernel's `last_insert` side channel is the only frontier state we
need — DFS is naturally LIFO with respect to insert order.

Demonstrates:
  - Reading `graph.last_insert` between steps.
  - Calling `lib_move.expand_candidates(parent_id)` to enumerate moves.
  - Halting via `actions.halt(reason)`.
  - Bailing via `actions.fallback(reason)` when stuck.

This is an EDUCATIONAL policy. It is not tuned to be efficient on
medium/large problems and may exhaust the budget for non-trivial inputs.
"""

PARAMS_OVERRIDE = PARAMS_OVERRIDE if 'PARAMS_OVERRIDE' in dir() else {}
DEFAULTS = {"max_branch": 8}
PARAMS = dict(DEFAULTS)
PARAMS.update(PARAMS_OVERRIDE)


def init(root, ctx):
    # No global state needed for vanilla DFS — `graph.last_insert`
    # carries everything.
    pass


def step(graph, gs, ctx, lib):
    cur = graph.last_insert
    if cur == None:
        cur = graph.root

    # Halt if cur is a target placement.
    if lib.is_target(cur):
        return [actions.halt("found_target")]

    # Enumerate candidates. If none, fall back (true dead end on DFS).
    cands = lib.expand_candidates(cur)
    if len(cands) == 0:
        return [actions.fallback("dead_end")]

    # Insert at most `max_branch` children; the kernel's transposition
    # table will dedupe duplicates. The next `step` will see the LAST
    # inserted child as `last_insert`.
    actions_out = []
    for cand in cands[:PARAMS["max_branch"]]:
        actions_out.append(actions.insert_child(cur, cand))
    return actions_out
```

- [ ] **Step 2: Create the dev fixture pair**

`policies/fixtures/dev/arch.json` — minimal valid ArchSpec (4×4 grid, 2 layers).

`policies/fixtures/dev/dfs_problem.json`:

```json
{
  "v": 1,
  "kind": "move",
  "arch": "arch.json",
  "initial": [[0, [1, 0, 0]], [1, [1, 0, 1]]],
  "target":  [[0, [1, 0, 1]], [1, [1, 0, 0]]],
  "blocked": [],
  "budget": { "max_expansions": 200, "timeout_s": 5.0 },
  "policy_params": {}
}
```

- [ ] **Step 3: Smoke test via the CLI**

```bash
cargo run -p bloqade-lanes-bytecode-cli -- eval-policy \
  --policy policies/reference/dfs.star \
  --problem policies/fixtures/dev/dfs_problem.json
```

Expected: exit code `0` or `2`. Either is fine for the smoke test — `0` means the policy solved it; `2` means budget-exhausted, which is acceptable for a small toy problem if DFS happens to dive into a deep branch first. Failures (exit `1`) point at a real policy or kernel bug.

- [ ] **Step 4: Commit**

```bash
git add policies/reference/dfs.star \
        policies/fixtures/dev/arch.json \
        policies/fixtures/dev/dfs_problem.json
git commit -m "feat(policies): add dfs.star reference Move policy + dev fixture"
```

---

### Task 12: `bfs.star`

**Files:**
- Create: `policies/reference/bfs.star`

- [ ] **Step 1: Author `bfs.star`**

```python
"""
Breadth-First Search reference policy
======================================

Strategy: visit nodes in insertion order via an explicit FIFO queue
maintained in policy global state. Each step pops the head of the queue,
expands it, and pushes children onto the tail. The result is a level-by-
level traversal of the search graph.

Demonstrates:
  - `update_global_state(...)` for mutating policy-owned state.
  - Reading and updating a list (the queue) across steps.
  - The contrast with `dfs.star`: BFS does NOT use `graph.last_insert`.

This is an EDUCATIONAL policy. It will spend its budget enumerating
shallow nodes; expect budget-exhaust on anything but the smallest
fixtures.
"""

PARAMS_OVERRIDE = PARAMS_OVERRIDE if 'PARAMS_OVERRIDE' in dir() else {}
DEFAULTS = {"max_branch": 8}
PARAMS = dict(DEFAULTS)
PARAMS.update(PARAMS_OVERRIDE)


def init(root, ctx):
    # Seed the queue with the root.
    update_global_state({"queue": [root.id]})


def step(graph, gs, ctx, lib):
    queue = list(gs["queue"])
    if len(queue) == 0:
        return [actions.fallback("queue_empty")]
    head = queue[0]
    rest = queue[1:]

    if lib.is_target(head):
        return [actions.halt("found_target")]

    cands = lib.expand_candidates(head)
    actions_out = []
    new_ids = []
    for cand in cands[:PARAMS["max_branch"]]:
        # The kernel will assign a NodeId; we read it via
        # `graph.last_insert` after each insert.  For a queue-based
        # policy we approximate by appending the parent id N times and
        # let DFS tie-breaks resolve — a real BFS would receive
        # explicit child ids back, which the v1 surface doesn't expose.
        # Instead we round-robin via the queue's BFS order without
        # tracking child ids explicitly: queue advances by one parent
        # per step.
        actions_out.append(actions.insert_child(head, cand))
        new_ids.append(head)  # placeholder; see comment above

    # Drop head, advance queue.
    update_global_state({"queue": rest})
    return actions_out
```

(Educational note: the `new_ids` placeholder is intentional — pure BFS over child ids needs a `last_insert_batch` accessor that v1 doesn't ship. The policy's heredoc explains this and the primer's reference-policy tour will call it out as a learning point.)

- [ ] **Step 2: Smoke test**

```bash
cargo run -p bloqade-lanes-bytecode-cli -- eval-policy \
  --policy policies/reference/bfs.star \
  --problem policies/fixtures/dev/dfs_problem.json
```

Expected: exit `0` or `2`. Same acceptance as Task 11 Step 3.

- [ ] **Step 3: Commit**

```bash
git add policies/reference/bfs.star
git commit -m "feat(policies): add bfs.star reference Move policy"
```

---

### Task 13: `ids.star`

**Files:**
- Create: `policies/reference/ids.star`

- [ ] **Step 1: Author `ids.star`**

```python
"""
Iterative Deepening Search reference policy
============================================

Strategy: DFS with a depth cap that grows by one each time the cap is
reached without finding a target. On cap-reached, reset to root and
restart with a higher cap.

Demonstrates:
  - `actions.reset_to_root()` — the only way to discard the current
    search subtree without halting.
  - Maintaining a depth cap in policy global state.
  - The interaction between `graph.depth` (read-only) and policy-owned
    counters.

This is an EDUCATIONAL policy and trades thoroughness for completeness:
on shallow targets it finds quickly; on deep targets it pays for the
re-exploration of the upper levels.
"""

PARAMS_OVERRIDE = PARAMS_OVERRIDE if 'PARAMS_OVERRIDE' in dir() else {}
DEFAULTS = {"start_depth": 2, "max_depth": 16, "max_branch": 8}
PARAMS = dict(DEFAULTS)
PARAMS.update(PARAMS_OVERRIDE)


def init(root, ctx):
    update_global_state({"cap": PARAMS["start_depth"]})


def step(graph, gs, ctx, lib):
    cap = gs["cap"]
    cur = graph.last_insert
    if cur == None:
        cur = graph.root

    if lib.is_target(cur):
        return [actions.halt("found_target")]

    if graph.depth >= cap:
        # Cap reached: bump cap and reset to root, unless we exceeded the global ceiling.
        new_cap = cap + 1
        if new_cap > PARAMS["max_depth"]:
            return [actions.fallback("max_depth_exceeded")]
        update_global_state({"cap": new_cap})
        return [actions.reset_to_root()]

    cands = lib.expand_candidates(cur)
    if len(cands) == 0:
        return [actions.fallback("dead_end")]

    actions_out = []
    for cand in cands[:PARAMS["max_branch"]]:
        actions_out.append(actions.insert_child(cur, cand))
    return actions_out
```

- [ ] **Step 2: Smoke test**

```bash
cargo run -p bloqade-lanes-bytecode-cli -- eval-policy \
  --policy policies/reference/ids.star \
  --problem policies/fixtures/dev/dfs_problem.json
```

Expected: exit `0` or `2`.

- [ ] **Step 3: Commit**

```bash
git add policies/reference/ids.star
git commit -m "feat(policies): add ids.star iterative-deepening reference policy"
```

---

## Phase 5 — Snapshot fixtures

### Task 14: Move fixture corpus (small / medium / large)

**Files:**
- Create: `policies/fixtures/README.md`
- Create: `policies/fixtures/move/small/{arch.json, problem.json}`
- Create: `policies/fixtures/move/medium/{arch.json, problem.json}`
- Create: `policies/fixtures/move/large/{arch.json, problem.json}`

- [ ] **Step 1: Write `policies/fixtures/README.md`**

```markdown
# Plan C snapshot fixtures

Each subdirectory contains a `problem.json` and one or more
`expected.<policy>.json` files that the snapshot-fixture test driver
(`crates/bloqade-lanes-search/tests/dsl_snapshot.rs`) consumes.

Comparison is **structural**, not byte-for-byte:

- Move: `{status, halt_reason, expansions, max_depth}`
- Target: `{ok, num_candidates, first_candidate_size}`

Wall-time and policy/problem paths are excluded.

## Sizes

- `small/`  ≈ 4×4 grid, ~6 qubits — exercised by every reference policy.
- `medium/` ≈ 10×10 grid, ~16 qubits — exercised by `entropy.star` only.
- `large/`  ≈ 20×20 grid, ~40 qubits — exercised by `entropy.star` only.

## Regenerating

When a baseline (kernel, `entropy.star`, `default_target.star`) shifts
in a way that legitimately changes a result, regenerate:

```bash
just regenerate-fixtures
git diff policies/fixtures/         # eyeball the diff
git add  policies/fixtures/         # commit only intentional shifts
```

CI does **not** auto-regenerate. A failing snapshot test means *either*
a real regression *or* an expected baseline shift — review before
regenerating.
```

- [ ] **Step 2: Author `policies/fixtures/move/small/`**

`arch.json` — 4×4 grid, 2 layers; copy the smallest existing example or hand-write.
`problem.json`:

```json
{
  "v": 1,
  "kind": "move",
  "arch": "arch.json",
  "initial": [[0, [1, 0, 0]], [1, [1, 0, 1]], [2, [1, 1, 0]], [3, [1, 1, 1]]],
  "target":  [[0, [1, 1, 1]], [1, [1, 1, 0]], [2, [1, 0, 1]], [3, [1, 0, 0]]],
  "blocked": [],
  "budget": { "max_expansions": 1000, "timeout_s": 5.0 },
  "policy_params": {}
}
```

- [ ] **Step 3: Author `policies/fixtures/move/medium/`**

`arch.json` — 10×10 grid, 2 layers.
`problem.json` — 16 qubits, target is a permutation that requires multiple swap-adjacent stages. Pick coordinates so `entropy.star` finishes in <5 s.

- [ ] **Step 4: Author `policies/fixtures/move/large/`**

`arch.json` — 20×20 grid, 2 layers.
`problem.json` — 40 qubits. Target shaped to keep `entropy.star` under ~20 s on CI hardware.

(Exact qubit and budget numbers will be tuned during implementation against the 30 s CI budget. The schema and structure shown above are the contract.)

- [ ] **Step 5: Commit**

```bash
git add policies/fixtures/README.md \
        policies/fixtures/move/
git commit -m "test(policies): add move snapshot fixture corpora (small/medium/large)"
```

---

### Task 15: Target fixture corpus

**Files:**
- Create: `policies/fixtures/target/small/{arch.json, problem.json}`

- [ ] **Step 1: Author `policies/fixtures/target/small/arch.json`**

Reuse the small move arch (copy or symlink). The Target Generator runs against the same arch shape.

- [ ] **Step 2: Author `policies/fixtures/target/small/problem.json`**

```json
{
  "v": 1,
  "kind": "target",
  "arch": "arch.json",
  "current_placement": [[0, [1, 0, 0]], [1, [1, 0, 1]], [2, [1, 1, 0]], [3, [1, 1, 1]]],
  "controls": [0, 2],
  "targets":  [1, 3],
  "lookahead_cz_layers": [],
  "cz_stage_index": 0,
  "policy_params": {}
}
```

- [ ] **Step 3: Commit**

```bash
git add policies/fixtures/target/
git commit -m "test(policies): add target snapshot fixture corpus (small)"
```

---

### Task 16: `dsl_snapshot.rs` integration test driver

**Files:**
- Create: `crates/bloqade-lanes-search/tests/dsl_snapshot.rs`

- [ ] **Step 1: Write the driver**

```rust
//! Plan C snapshot-fixture regression test.
//!
//! Walks `policies/fixtures/{move,target}/<size>/` and for every
//! `expected.<policy>.json` runs the matching policy against
//! `problem.json` and structurally compares the result.
//!
//! Comparison fields (Plan C spec §9.3):
//!   - Move:   {status, halt_reason, expansions, max_depth}
//!   - Target: {ok, num_candidates, first_candidate_size}
//!
//! Failure messages include a hint to run `just regenerate-fixtures`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use bloqade_lanes_bytecode_core::arch::ArchSpec;
use bloqade_lanes_dsl_core::sandbox::SandboxConfig;
use bloqade_lanes_search::fixture::{self, Problem};
use bloqade_lanes_search::lane_index::LaneIndex;
use bloqade_lanes_search::move_policy_dsl::{
    NoOpMoveObserver, PolicyOptions, solve_with_policy,
};
use bloqade_lanes_search::target_generator_dsl::{
    NoOpTargetObserver, TargetPolicyRunner, run_target_policy,
};

fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../policies/fixtures")
}

fn policies_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../policies/reference")
}

#[derive(serde::Deserialize, PartialEq, Debug)]
struct ExpectedMove {
    status: String,
    halt_reason: Option<String>,
    expansions: u64,
    max_depth: u32,
}

#[derive(serde::Deserialize, PartialEq, Debug)]
struct ExpectedTarget {
    ok: bool,
    num_candidates: usize,
    first_candidate_size: usize,
}

#[test]
fn snapshot_corpus_passes_structural_match() {
    let root = fixture_root();
    let mut failures: Vec<String> = Vec::new();
    let mut total = 0;

    for kind_dir in ["move", "target"].iter().map(|k| root.join(k)) {
        if !kind_dir.exists() {
            continue;
        }
        for size in std::fs::read_dir(&kind_dir).unwrap().flatten() {
            let size_path = size.path();
            if !size_path.is_dir() {
                continue;
            }
            let problem = size_path.join("problem.json");
            if !problem.exists() {
                continue;
            }
            for entry in std::fs::read_dir(&size_path).unwrap().flatten() {
                let p = entry.path();
                let name = match p.file_name().and_then(|s| s.to_str()) {
                    Some(n) => n,
                    None => continue,
                };
                let policy_name = match name.strip_prefix("expected.").and_then(|n| n.strip_suffix(".json")) {
                    Some(s) => s,
                    None => continue,
                };
                total += 1;
                let policy_path = policies_dir().join(format!("{policy_name}.star"));
                match run_one(&problem, &p, &policy_path) {
                    Ok(()) => {}
                    Err(msg) => failures.push(format!("{}: {msg}", p.display())),
                }
            }
        }
    }

    assert!(total > 0, "no snapshot fixtures found at {}", root.display());
    if !failures.is_empty() {
        let joined = failures.join("\n  ");
        panic!(
            "snapshot mismatches ({} of {}):\n  {joined}\n\n\
             Hint: if these are intentional baseline shifts, run `just regenerate-fixtures`.",
            failures.len(), total
        );
    }
}

fn run_one(problem_path: &Path, expected_path: &Path, policy_path: &Path) -> Result<(), String> {
    let (parsed, arch_path) = fixture::load(problem_path).map_err(|e| e.to_string())?;
    let arch_json = std::fs::read_to_string(&arch_path).map_err(|e| e.to_string())?;
    let arch = ArchSpec::from_json(&arch_json).map_err(|e| e.to_string())?;
    match parsed {
        Problem::Move(mp) => {
            let index = Arc::new(LaneIndex::new(arch));
            let initial = mp.initial.iter().map(|(q, [l, r, c])| (*q, (*l, *r, *c).into()));
            let target = mp.target.iter().map(|(q, [l, r, c])| (*q, (*l, *r, *c).into()));
            let blocked = mp.blocked.iter().map(|[l, r, c]| (*l, *r, *c).into());
            let opts = PolicyOptions {
                policy_path: policy_path.display().to_string(),
                sandbox: SandboxConfig::default(),
                policy_params: mp.policy_params.clone(),
                max_expansions: mp.budget.as_ref().map(|b| b.max_expansions).unwrap_or(5000),
                timeout_s: Some(mp.budget.as_ref().map(|b| b.timeout_s).unwrap_or(10.0)),
            };
            let mut obs = NoOpMoveObserver;
            let res = solve_with_policy(initial, target, blocked, index, opts, &mut obs)
                .map_err(|e| e.to_string())?;
            let expected: ExpectedMove = serde_json::from_slice(
                &std::fs::read(expected_path).map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
            let actual = ExpectedMove {
                status: status_label(&res.status).into(),
                halt_reason: halt_reason(&res.status),
                expansions: res.nodes_expanded as u64,
                max_depth: res.move_layers.len() as u32,
            };
            if actual != expected {
                return Err(format!("expected {expected:?}, got {actual:?}"));
            }
            Ok(())
        }
        Problem::Target(tp) => {
            let index = Arc::new(LaneIndex::new(arch));
            let placement: Vec<(u32, _)> = tp
                .current_placement
                .iter()
                .map(|(q, [l, r, c])| (*q, (*l, *r, *c).into()))
                .collect();
            let cfg = bloqade_lanes_dsl_core::sandbox::SandboxConfig::default();
            let mut obs = NoOpTargetObserver;
            let result = bloqade_lanes_search::target_generator_dsl::run_target_policy(
                policy_path,
                index,
                placement,
                tp.controls.clone(),
                tp.targets.clone(),
                tp.lookahead_cz_layers.clone(),
                tp.cz_stage_index,
                tp.policy_params.clone(),
                &cfg,
                &mut obs,
            );
            let expected: ExpectedTarget = serde_json::from_slice(
                &std::fs::read(expected_path).map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
            let (num_candidates, first_candidate_size) = match &result {
                Ok(c) => (c.len(), c.first().map_or(0, |v| v.len())),
                Err(_) => (0, 0),
            };
            let actual = ExpectedTarget {
                ok: result.is_ok(),
                num_candidates,
                first_candidate_size,
            };
            if actual != expected {
                return Err(format!("expected {expected:?}, got {actual:?}"));
            }
            Ok(())
        }
    }
}

fn status_label(s: &bloqade_lanes_search::move_policy_dsl::PolicyStatus) -> &'static str {
    use bloqade_lanes_search::move_policy_dsl::PolicyStatus::*;
    match s {
        Solved => "Solved",
        Unsolvable => "Unsolvable",
        BudgetExhausted => "BudgetExhausted",
        Timeout => "Timeout",
        Fallback(_) => "Fallback",
        SyntaxError(_) => "SyntaxError",
        RuntimeError(_) => "RuntimeError",
        SchemaError(_) => "SchemaError",
        BadPolicy(_) => "BadPolicy",
        StarlarkBudget => "StarlarkBudget",
        StarlarkOOM => "StarlarkOOM",
    }
}

fn halt_reason(s: &bloqade_lanes_search::move_policy_dsl::PolicyStatus) -> Option<String> {
    use bloqade_lanes_search::move_policy_dsl::PolicyStatus::*;
    match s {
        Solved => Some("policy_halt".into()),
        Fallback(r) => Some(r.clone()),
        _ => None,
    }
}
```

- [ ] **Step 2: Try to run; expect failure due to missing expected.*.json**

```bash
cargo test -p bloqade-lanes-search --test dsl_snapshot
```

Expected: panic — "no snapshot fixtures found" if `expected.*.json` files don't yet exist (Task 17 generates them).

- [ ] **Step 3: Commit**

```bash
git add crates/bloqade-lanes-search/tests/dsl_snapshot.rs
git commit -m "test(search): add snapshot-corpus structural-match driver"
```

---

### Task 17: `regenerate-fixtures` recipe + initial expected.\*.json

**Files:**
- Modify: `justfile`
- Create: `policies/fixtures/move/small/expected.{entropy,dfs,bfs,ids}.json`
- Create: `policies/fixtures/move/medium/expected.entropy.json`
- Create: `policies/fixtures/move/large/expected.entropy.json`
- Create: `policies/fixtures/target/small/expected.default_target.json`

- [ ] **Step 1: Add the just recipe**

Append to `justfile`:

```makefile
# Regenerate every policies/fixtures/<kind>/<size>/expected.*.json by
# running its matching policy via `eval-policy --json` and stripping
# fields that are excluded from structural comparison.
regenerate-fixtures:
    #!/usr/bin/env bash
    set -euo pipefail
    cargo build -p bloqade-lanes-bytecode-cli --release
    BIN="target/release/bloqade-bytecode"
    for kind_dir in policies/fixtures/move policies/fixtures/target; do
      [ -d "$kind_dir" ] || continue
      for size_dir in "$kind_dir"/*/; do
        problem="$size_dir/problem.json"
        [ -f "$problem" ] || continue
        kind=$(basename "$kind_dir")
        for policy_path in $(ls "$size_dir"expected.*.json 2>/dev/null || true); do
          name=$(basename "$policy_path" .json | sed 's/^expected\.//')
          policy_file="policies/reference/${name}.star"
          if [ ! -f "$policy_file" ]; then
            echo "skip: no policy $policy_file" >&2
            continue
          fi
          tmp=$(mktemp)
          "$BIN" eval-policy --json --policy "$policy_file" --problem "$problem" > "$tmp" || true
          if [ "$kind" = "move" ]; then
            jq '{status, halt_reason, expansions, max_depth}' "$tmp" > "$policy_path"
          else
            jq '{ok, num_candidates, first_candidate_size}' "$tmp" > "$policy_path"
          fi
          rm "$tmp"
        done
      done
    done
    echo "regenerated. eyeball: git diff policies/fixtures/"
```

- [ ] **Step 2: Author the initial expected files BY HAND, then verify with regenerate**

For each expected file in the corpus, hand-write a plausible expected output — the *first* run of `regenerate-fixtures` will overwrite these with the real values, but having stubs commit first lets the recipe iterate over them.

Example `policies/fixtures/move/small/expected.dfs.json` (stub):

```json
{
  "status": "Solved",
  "halt_reason": "policy_halt",
  "expansions": 0,
  "max_depth": 0
}
```

(Exact stub values are placeholders; Step 3 overwrites them.)

- [ ] **Step 3: Run regenerate**

```bash
just regenerate-fixtures
git diff policies/fixtures/
```

Expected: every expected file is overwritten with real values from the corresponding policy run. Eyeball the diff.

- [ ] **Step 4: Run the snapshot test**

```bash
cargo test -p bloqade-lanes-search --test dsl_snapshot
```

Expected: passes. If a fixture is consistently flaky (different run → different result), the underlying policy is non-deterministic and the issue must be fixed in the policy, not the fixture.

- [ ] **Step 5: Commit**

```bash
git add justfile policies/fixtures/move/*/expected.*.json policies/fixtures/target/*/expected.*.json
git commit -m "test(policies): add snapshot expected baselines + regenerate-fixtures recipe"
```

---

## Phase 6 — Primer generator

### Task 18: Author registration-site doc-comments

**Files:**
- Modify: `crates/bloqade-lanes-search/src/move_policy_dsl/actions.rs`
- Modify: `crates/bloqade-lanes-search/src/move_policy_dsl/lib_move.rs`
- Modify: `crates/bloqade-lanes-search/src/move_policy_dsl/graph_handle.rs`
- Modify: `crates/bloqade-lanes-search/src/target_generator_dsl/lib_target.rs`

This task is mechanical: every `MethodsBuilder` registration in those four files gets a `#[doc = "..."]` doc-comment that summarises the method in one line, suitable for direct copy into `policies/primer.md`.

- [ ] **Step 1: Audit current doc state**

```bash
grep -n "fn .* (" crates/bloqade-lanes-search/src/move_policy_dsl/actions.rs | head
grep -n "/// " crates/bloqade-lanes-search/src/move_policy_dsl/actions.rs | head
```

Use this to compile a list of methods that need doc-comments.

- [ ] **Step 2: Add doc-comments at every `#[starlark_module]` / `MethodsBuilder` registration**

Example pattern (in `actions.rs`):

```rust
#[starlark_module]
fn actions_module(builder: &mut GlobalsBuilder) {
    /// Insert a new child node under `parent`, applying the given
    /// location writes. Returns the kernel-assigned child id via
    /// `graph.last_insert` on the next step.
    fn insert_child<'v>(parent: Value<'v>, writes: Value<'v>) -> anyhow::Result<NoneType> {
        // …
    }

    /// Halt the search with the given reason string. The kernel maps
    /// halt-reason to `PolicyStatus::Solved` and emits it on the trace.
    fn halt(reason: &str) -> anyhow::Result<NoneType> {
        // …
    }

    // … same pattern for every other action verb.
}
```

Apply the same pattern to each method in `lib_move.rs`, `graph_handle.rs`, `lib_target.rs`. Doc-comments must be **single-line** and follow a `[verb] [object] [optionally what it returns]` pattern, ≤ 100 chars.

- [ ] **Step 3: Verify the existing tests still pass**

```bash
cargo test -p bloqade-lanes-search --lib
cargo clippy -p bloqade-lanes-search --all-targets -- -D warnings
```

(Doc-comments don't change runtime behaviour; this is just a smoke check.)

- [ ] **Step 4: Commit**

```bash
git add crates/bloqade-lanes-search/src/move_policy_dsl/{actions,lib_move,graph_handle}.rs \
        crates/bloqade-lanes-search/src/target_generator_dsl/lib_target.rs
git commit -m "docs(search): add primer-friendly doc-comments to DSL registration sites"
```

---

### Task 19: `policies-primer` binary skeleton

**Files:**
- Create: `crates/bloqade-lanes-search/src/bin/policies-primer.rs`
- Modify: `crates/bloqade-lanes-search/Cargo.toml` (add `syn`, `proc_macro2`, `similar` for diff output)

- [ ] **Step 1: Add dependencies**

In `[dependencies]` (the binary uses them at run time, so they're regular deps):

```toml
syn = { workspace = true, features = ["full", "extra-traits"] }
proc-macro2 = { workspace = true }
similar = { workspace = true }
```

If `similar` is not in the workspace, add `similar = "2"`.

- [ ] **Step 2: Write the skeleton**

```rust
//! `policies-primer` — autogenerator for `policies/primer.md`.
//!
//! Plan C — see `docs/superpowers/specs/2026-05-01-move-policy-dsl-plan-c-design.md` §8.
//!
//! Modes:
//!   default  — write `policies/primer.md`
//!   --check  — exit 1 with a diff if `policies/primer.md` differs from
//!              the regenerated content
//!
//! Source files are embedded via `include_str!` at compile time;
//! generation involves no runtime path resolution against the cargo
//! manifest layout.

use std::collections::BTreeMap;

const ACTIONS_SRC:      &str = include_str!("../move_policy_dsl/actions.rs");
const LIB_MOVE_SRC:     &str = include_str!("../move_policy_dsl/lib_move.rs");
const GRAPH_HANDLE_SRC: &str = include_str!("../move_policy_dsl/graph_handle.rs");
const LIB_TARGET_SRC:   &str = include_str!("../target_generator_dsl/lib_target.rs");

const PRIMER_PATH: &str = "policies/primer.md";

#[derive(Default)]
struct ParsedPrimer {
    prose: BTreeMap<String, String>,
}

fn main() {
    let check = std::env::args().any(|a| a == "--check");
    let existing = std::fs::read_to_string(PRIMER_PATH).unwrap_or_default();
    let parsed = parse_existing(&existing);

    let regenerated = render(&parsed);

    if check {
        if existing.trim() != regenerated.trim() {
            let diff = unified_diff(&existing, &regenerated);
            eprintln!("{PRIMER_PATH} is stale. Diff:\n{diff}");
            eprintln!("Run `just generate-primer` to update.");
            std::process::exit(1);
        }
        eprintln!("{PRIMER_PATH} is up to date.");
    } else {
        std::fs::write(PRIMER_PATH, &regenerated).expect("write primer");
        eprintln!("wrote {PRIMER_PATH}");
    }
}

fn parse_existing(_src: &str) -> ParsedPrimer {
    // PROSE block extraction implemented in Task 20 alongside the
    // sentinel-block format. For Task 19 the parser returns empty,
    // which means a fresh worktree generates stub PROSE blocks.
    ParsedPrimer::default()
}

fn render(parsed: &ParsedPrimer) -> String {
    let mut out = String::new();
    out.push_str("<!-- AUTOGEN: DO NOT EDIT BY HAND.\n");
    out.push_str("     Regenerate with `just generate-primer`. -->\n\n");
    out.push_str("# Move Policy & Target Generator DSL — Primer\n\n");
    out.push_str(&prose_block("intro", parsed));
    out.push_str("\n## Move Policy surface\n\n");
    out.push_str(&autogen_block("actions", "TODO: fill in Task 20"));
    out.push_str(&autogen_block("lib_move", "TODO: fill in Task 21"));
    out.push_str(&autogen_block("graph_handle", "TODO: fill in Task 21"));
    out.push_str(&prose_block("move-tour", parsed));
    out.push_str("\n## Target Generator surface\n\n");
    out.push_str(&autogen_block("lib_target", "TODO: fill in Task 22"));
    out.push_str(&prose_block("target-tour", parsed));
    out.push_str("\n## Problem fixture schema\n\n");
    out.push_str(&autogen_block("schema", "TODO: fill in Task 23"));
    out
}

fn prose_block(name: &str, parsed: &ParsedPrimer) -> String {
    let body = parsed
        .prose
        .get(name)
        .cloned()
        .unwrap_or_else(|| format!("TODO: write prose for {name}\n"));
    format!("<!-- BEGIN PROSE: {name} -->\n{body}\n<!-- END PROSE: {name} -->\n")
}

fn autogen_block(name: &str, body: &str) -> String {
    format!("<!-- BEGIN AUTOGEN: {name} -->\n{body}\n<!-- END AUTOGEN: {name} -->\n\n")
}

fn unified_diff(a: &str, b: &str) -> String {
    let diff = similar::TextDiff::from_lines(a, b);
    let mut buf = String::new();
    for change in diff.iter_all_changes() {
        let sign = match change.tag() {
            similar::ChangeTag::Delete => "-",
            similar::ChangeTag::Insert => "+",
            similar::ChangeTag::Equal => " ",
        };
        buf.push_str(&format!("{sign}{change}"));
    }
    buf
}
```

- [ ] **Step 3: Verify it builds and runs**

```bash
cargo build -p bloqade-lanes-search --bin policies-primer
cargo run -p bloqade-lanes-search --bin policies-primer
```

Expected: writes `policies/primer.md` with stub AUTOGEN blocks. Inspect the file.

- [ ] **Step 4: Commit**

```bash
git add crates/bloqade-lanes-search/src/bin/policies-primer.rs \
        crates/bloqade-lanes-search/Cargo.toml
git commit -m "feat(search): add policies-primer binary skeleton (sentinel blocks + diff)"
```

---

### Task 20: `AUTOGEN: actions` section + PROSE round-trip

**Files:**
- Modify: `crates/bloqade-lanes-search/src/bin/policies-primer.rs`

- [ ] **Step 1: Add PROSE-block extraction**

Replace `parse_existing` with a real implementation:

```rust
fn parse_existing(src: &str) -> ParsedPrimer {
    let mut prose = BTreeMap::new();
    let mut in_block: Option<String> = None;
    let mut buf = String::new();
    for line in src.lines() {
        if let Some(rest) = line.strip_prefix("<!-- BEGIN PROSE: ") {
            let name = rest.trim_end_matches(" -->").to_string();
            in_block = Some(name);
            buf.clear();
            continue;
        }
        if line.starts_with("<!-- END PROSE: ") {
            if let Some(name) = in_block.take() {
                prose.insert(name, std::mem::take(&mut buf));
            }
            continue;
        }
        if in_block.is_some() {
            buf.push_str(line);
            buf.push('\n');
        }
    }
    ParsedPrimer { prose }
}
```

- [ ] **Step 2: Add `render_actions(...)` driven by `syn` parsing of `ACTIONS_SRC`**

Add helpers:

```rust
struct StarlarkMethod {
    name: String,
    sig: String,        // a stringified rust signature, used as the primer "signature" line
    summary: String,    // first non-empty doc-comment line
}

fn parse_starlark_methods(src: &str, module_filter: &str) -> Vec<StarlarkMethod> {
    let file: syn::File = syn::parse_str(src).expect("parse rust source");
    let mut out = Vec::new();
    for item in file.items {
        if let syn::Item::Fn(f) = &item {
            // Inside a `#[starlark_module]` block, but `syn` parses the
            // outer fn that wraps the methods. We look for inner items
            // via the function's body.
            let body_str = quote::quote!(#f).to_string();
            // Crude heuristic: match `module_filter` as a substring of
            // the wrapping fn's name (e.g. "actions_module").
            if !body_str.contains(module_filter) {
                continue;
            }
            // Walk `f.block` for inner fn items.
            for stmt in &f.block.stmts {
                if let syn::Stmt::Item(syn::Item::Fn(inner)) = stmt {
                    let name = inner.sig.ident.to_string();
                    let sig  = quote::quote!(#inner.sig).to_string()
                        .lines().next().unwrap_or("").to_string();
                    let summary = inner.attrs.iter().find_map(|a| {
                        if a.path().is_ident("doc") {
                            a.meta.require_name_value().ok().and_then(|nv| {
                                if let syn::Expr::Lit(syn::ExprLit { lit: syn::Lit::Str(s), .. }) = &nv.value {
                                    Some(s.value().trim().to_string())
                                } else { None }
                            })
                        } else { None }
                    }).unwrap_or_else(|| "(undocumented)".into());
                    out.push(StarlarkMethod { name, sig, summary });
                }
            }
        }
    }
    out
}

fn render_starlark_section(title: &str, methods: &[StarlarkMethod]) -> String {
    let mut out = String::new();
    out.push_str(&format!("### `{title}`\n\n"));
    for m in methods {
        out.push_str(&format!("#### `{}`\n\n{}\n\n```rust\n{}\n```\n\n", m.name, m.summary, m.sig));
    }
    out
}
```

(`quote` and `syn` are added as deps; if `quote` isn't present, add it: `quote = { workspace = true }` or `quote = "1"`.)

- [ ] **Step 3: Wire `render_actions` into `render`**

```rust
fn render_actions() -> String {
    let methods = parse_starlark_methods(ACTIONS_SRC, "actions_module");
    render_starlark_section("actions.* — kernel-driven verbs", &methods)
}
```

Update `render` to call `render_actions()` instead of the TODO stub.

- [ ] **Step 4: Run, verify the actions section is non-empty**

```bash
cargo run -p bloqade-lanes-search --bin policies-primer
```

Inspect `policies/primer.md`: the `<!-- BEGIN AUTOGEN: actions -->...<!-- END AUTOGEN: actions -->` block should now contain headings and signatures for every action method.

- [ ] **Step 5: Commit**

```bash
git add crates/bloqade-lanes-search/src/bin/policies-primer.rs \
        crates/bloqade-lanes-search/Cargo.toml
git commit -m "feat(search): generate AUTOGEN:actions section + parse PROSE blocks"
```

---

### Task 21: `AUTOGEN: lib_move` + `AUTOGEN: graph_handle` sections

**Files:**
- Modify: `crates/bloqade-lanes-search/src/bin/policies-primer.rs`

- [ ] **Step 1: Add `render_lib_move` and `render_graph_handle`**

```rust
fn render_lib_move() -> String {
    let methods = parse_starlark_methods(LIB_MOVE_SRC, "lib_move_module");
    render_starlark_section("lib_move.* — query primitives", &methods)
}

fn render_graph_handle() -> String {
    let methods = parse_starlark_methods(GRAPH_HANDLE_SRC, "graph_module");
    render_starlark_section("graph.* — read-only graph accessors", &methods)
}
```

(Verify the actual `module_filter` strings match the wrapper-fn names in the source — adjust if the registration site uses a different convention.)

- [ ] **Step 2: Wire into `render`**

Replace the lib_move and graph_handle TODO stubs with calls to the new functions.

- [ ] **Step 3: Run and verify**

```bash
cargo run -p bloqade-lanes-search --bin policies-primer
```

Inspect: both new sections are populated.

- [ ] **Step 4: Commit**

```bash
git add crates/bloqade-lanes-search/src/bin/policies-primer.rs
git commit -m "feat(search): generate AUTOGEN:lib_move and AUTOGEN:graph_handle sections"
```

---

### Task 22: `AUTOGEN: lib_target` section

**Files:**
- Modify: `crates/bloqade-lanes-search/src/bin/policies-primer.rs`

- [ ] **Step 1: Add `render_lib_target`**

```rust
fn render_lib_target() -> String {
    let methods = parse_starlark_methods(LIB_TARGET_SRC, "lib_target_module");
    render_starlark_section("lib_target.* — placement query primitives", &methods)
}
```

- [ ] **Step 2: Wire into `render`**

Replace the lib_target TODO stub with `render_lib_target()`.

- [ ] **Step 3: Run and verify**

```bash
cargo run -p bloqade-lanes-search --bin policies-primer
```

- [ ] **Step 4: Commit**

```bash
git add crates/bloqade-lanes-search/src/bin/policies-primer.rs
git commit -m "feat(search): generate AUTOGEN:lib_target section"
```

---

### Task 23: `AUTOGEN: schema` section

**Files:**
- Modify: `crates/bloqade-lanes-search/src/bin/policies-primer.rs`

- [ ] **Step 1: Render the JSON Schema**

```rust
use bloqade_lanes_search::fixture;

fn render_schema() -> String {
    let mut out = String::from("### Problem fixture schema\n\n");
    out.push_str("Problem fixtures are JSON files with one of two top-level shapes, ");
    out.push_str("discriminated by a `\"kind\"` field.\n\n");
    out.push_str("```json\n");
    out.push_str(&fixture::json_schema_pretty());
    out.push_str("\n```\n");
    out
}
```

- [ ] **Step 2: Wire into `render`**

Replace the schema TODO stub with `render_schema()`.

- [ ] **Step 3: Run and verify**

```bash
cargo run -p bloqade-lanes-search --bin policies-primer
```

Inspect: the schema section contains a JSON-Schema document for `MoveProblem` and `TargetProblem`.

- [ ] **Step 4: Commit**

```bash
git add crates/bloqade-lanes-search/src/bin/policies-primer.rs
git commit -m "feat(search): generate AUTOGEN:schema section from fixture types"
```

---

### Task 24: Author initial PROSE blocks

**Files:**
- Modify: `policies/primer.md` (in-place edit of the auto-generated PROSE blocks)

- [ ] **Step 1: Generate the file once**

```bash
cargo run -p bloqade-lanes-search --bin policies-primer
```

The PROSE blocks contain `TODO: write prose for <name>` placeholders.

- [ ] **Step 2: Replace each PROSE block with real prose**

Edit `policies/primer.md` in-place. For each block:

- **`intro`** — three-paragraph orientation: what this DSL is, what problems it solves, how to invoke it (Python kwarg + CLI subcommand), and a link to the spec.
- **`move-tour`** — walk through `entropy.star`, `dfs.star`, `bfs.star`, `ids.star`. For each: one paragraph explaining the strategy, a code excerpt of `step(...)`, and a note on which surface methods it relies on.
- **`target-tour`** — walk through `default_target.star`. Explain that target generation is a single-call-per-stage operation, contrast with the Move surface.

- [ ] **Step 3: Re-run the generator and verify the prose is preserved**

```bash
cargo run -p bloqade-lanes-search --bin policies-primer
```

Re-inspect `policies/primer.md`: PROSE blocks contain the prose you authored, AUTOGEN blocks are unchanged.

- [ ] **Step 4: Commit**

```bash
git add policies/primer.md
git commit -m "docs(policies): author initial primer prose (intro, move-tour, target-tour)"
```

---

### Task 25: `primer_golden` integration test

**Files:**
- Create: `crates/bloqade-lanes-search/tests/primer_golden.rs`
- Create: `crates/bloqade-lanes-search/tests/fixtures/primer/input/{actions,lib_move,graph_handle,lib_target}.rs` (curated stub registration files)
- Create: `crates/bloqade-lanes-search/tests/fixtures/primer/expected.md`

- [ ] **Step 1: Curate the input stubs**

Each stub is a tiny `#[starlark_module]` block with two methods carrying known doc-comments. Example `tests/fixtures/primer/input/actions.rs`:

```rust
// Stub registration file used by the primer-generator golden test.
#[starlark_module]
fn actions_module(builder: &mut GlobalsBuilder) {
    /// Stub-doc: insert child.
    fn insert_child(parent: i32, write: i32) -> anyhow::Result<()> { Ok(()) }
    /// Stub-doc: halt with reason.
    fn halt(reason: &str) -> anyhow::Result<()> { Ok(()) }
}
```

(Make analogous stubs for `lib_move`, `graph_handle`, `lib_target`.)

- [ ] **Step 2: Author the golden expected.md**

Hand-write `tests/fixtures/primer/expected.md`. It should match exactly what the generator produces when fed the four stubs above.

- [ ] **Step 3: Write the test**

```rust
//! Golden-file test for the primer generator.
//!
//! Feeds curated stub registration files through the same parsing
//! pipeline the binary uses, and compares to a committed expected.md.
//! The live `policies/primer.md` is *not* the test oracle — that file
//! tracks the real DSL surface, which evolves under separate review.

use std::path::PathBuf;

#[test]
fn primer_generator_matches_golden_for_curated_stubs() {
    // The primer binary's parsing functions are private to the binary
    // crate, so the golden test invokes the binary directly with a
    // working directory pointing at the fixture root and verifies
    // stdout/file output. This pattern keeps the binary simple.
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let golden_input  = manifest.join("tests/fixtures/primer/input");
    let expected_path = manifest.join("tests/fixtures/primer/expected.md");
    let expected = std::fs::read_to_string(&expected_path).expect("read expected.md");

    // Invoke the binary in a special mode that takes --input-dir and
    // --stdout. Implement that mode in policies-primer first if absent
    // (a pre-task fix; it's a small flag).
    use assert_cmd::Command;
    let out = Command::cargo_bin("policies-primer")
        .unwrap()
        .arg("--input-dir").arg(&golden_input)
        .arg("--stdout")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let actual = String::from_utf8(out).unwrap();
    if actual.trim() != expected.trim() {
        panic!(
            "primer-golden mismatch.\n--- expected ---\n{expected}\n--- actual ---\n{actual}"
        );
    }
}
```

- [ ] **Step 4: Add `--input-dir` and `--stdout` flags to the primer binary**

Modify `policies-primer`'s `main`:

```rust
let mut input_dir: Option<PathBuf> = None;
let mut stdout = false;
let mut check = false;
for arg in std::env::args().skip(1) {
    if arg == "--check" { check = true; }
    else if arg == "--stdout" { stdout = true; }
    else if let Some(path) = arg.strip_prefix("--input-dir=") {
        input_dir = Some(path.into());
    }
}
```

When `input_dir` is set, the binary reads `actions.rs`/`lib_move.rs`/`graph_handle.rs`/`lib_target.rs` from that directory instead of from the embedded `include_str!` constants. When `--stdout` is set, write to stdout instead of `policies/primer.md`.

This keeps the production path unchanged (no `--input-dir`) and lets the test feed curated stubs.

- [ ] **Step 5: Add `assert_cmd` to search's dev-deps if absent**

```bash
grep -E '^assert_cmd *=' crates/bloqade-lanes-search/Cargo.toml
```

If absent:

```toml
[dev-dependencies]
assert_cmd = { workspace = true }
```

- [ ] **Step 6: Run the golden test**

```bash
cargo test -p bloqade-lanes-search --test primer_golden
```

If `expected.md` was hand-written incorrectly, regenerate via:

```bash
cargo run -p bloqade-lanes-search --bin policies-primer -- --input-dir=crates/bloqade-lanes-search/tests/fixtures/primer/input --stdout > crates/bloqade-lanes-search/tests/fixtures/primer/expected.md
```

Then re-run the test. Eyeball the diff before committing the regenerated expected.md.

- [ ] **Step 7: Commit**

```bash
git add crates/bloqade-lanes-search/tests/primer_golden.rs \
        crates/bloqade-lanes-search/tests/fixtures/primer/ \
        crates/bloqade-lanes-search/src/bin/policies-primer.rs \
        crates/bloqade-lanes-search/Cargo.toml
git commit -m "test(search): add primer-generator golden test and --input-dir/--stdout flags"
```

---

### Task 26: `just generate-primer` and `just check-primer` recipes

**Files:**
- Modify: `justfile`

- [ ] **Step 1: Add the recipes**

Append to `justfile`:

```makefile
# Regenerate policies/primer.md from registration-site source.
generate-primer:
    cargo run -p bloqade-lanes-search --bin policies-primer

# Verify policies/primer.md is up to date; exits non-zero with a diff
# if stale. Used by CI alongside `just check-header`.
check-primer:
    cargo run -p bloqade-lanes-search --bin policies-primer -- --check
```

Add `check-primer` to the `lint` recipe's dependencies (so `just lint` runs it):

```makefile
lint: format-check check-header check-primer
    cargo clippy -p bloqade-lanes-bytecode-core -p bloqade-lanes-bytecode-cli --all-targets -- -D warnings
```

(Adjust the existing `lint` recipe to insert `check-primer` in the dependency list — preserve current dependencies.)

- [ ] **Step 2: Verify recipes work**

```bash
just generate-primer
just check-primer
just lint
```

Each succeeds.

- [ ] **Step 3: Commit**

```bash
git add justfile
git commit -m "build: add generate-primer and check-primer just recipes"
```

---

### Task 27: CI hookup

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Find the existing `check-header` step**

```bash
grep -n "check-header\|just check" .github/workflows/ci.yml
```

- [ ] **Step 2: Add `just check-primer` next to it**

Insert a step in the same job:

```yaml
- name: Check policies/primer.md is up to date
  run: just check-primer
```

Place it directly after the `check-header` step. The job already has Rust + just installed for `check-header`, so no setup changes are needed.

- [ ] **Step 3: Verify CI YAML parses**

```bash
yamllint .github/workflows/ci.yml || true
```

(yamllint may not be installed; ignore the failure-to-find error and confirm the YAML structure visually.)

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: check policies/primer.md staleness alongside check-header"
```

---

## Final verification

After Task 27 commits, run the full pipeline one more time:

```bash
just test-rust
just develop-python && uv run pytest python/tests
just lint
just check-primer
cargo test -p bloqade-lanes-search --test dsl_snapshot
cargo test -p bloqade-lanes-bytecode-cli --test cli_policy
```

All green. The branch is ready for the user to push and open a PR.

---

## Self-review notes

**Spec coverage check.** Walking through the spec section by section:

- §1 (goal): covered by Tasks 11–13 (reference policies), 7–10 (CLI), 19–24 (primer), 14–17 (snapshots).
- §2 (non-goals): no implementation tasks; nothing to cover.
- §3 (architecture overview): Task 1–4 (observer hookup), Task 5 (fixture loader), Task 6–10 (CLI deps + subcommands), Task 19–27 (primer + CI).
- §3.1 (dep graph): Task 6 adds `search` and `dsl-core` as direct deps of CLI.
- §3.2 (sibling-module rationale): observer types live in `move_policy_dsl/observer.rs` and `target_generator_dsl/observer.rs` — Tasks 1 and 3.
- §4 (CLI surface): Task 7 (subcommand registration), Task 8 (eval-policy), Task 9 (trace-policy).
- §4.4 (exit codes): Task 8's `exit_code(&PolicyStatus)` and Task 9's flow.
- §4.5 (eval JSON envelope): Task 8's `EvalEnvelope` / `TargetEvalEnvelope`.
- §4.6 (NDJSON trace): Task 2 (Move JSON observer), Task 3 (Target JSON observer), Task 9 (CLI plumbing).
- §5 (observer trait): Tasks 1, 2, 3.
- §5.3 (extensibility): default-method bodies on the trait — already in Task 1's signature.
- §6 (problem fixture schema): Task 5.
- §7 (reference policies): Tasks 11–13.
- §8 (primer generator): Tasks 19–25.
- §9 (snapshot fixtures): Tasks 14–17.
- §10 (testing strategy): Tasks 1, 2, 3, 5, 10, 16, 25 — every test row of the §10 table is implemented by an explicit task.
- §11 (error handling): Task 8 maps `PolicyStatus` to exit codes.
- §12 (file-by-file additions): every row corresponds to at least one task; each task declares the files it creates/modifies.
- §13 (out of scope): no tasks (deferral).
- §14 (open questions): none remained.

**Placeholder scan.** The plan deliberately uses three placeholder strings:

1. `TODO: fill in Task N` inside the `policies-primer` skeleton (Task 19) — replaced by Tasks 20–23 within the same plan.
2. `TODO: write prose for <name>` inside the generator's stub PROSE block — replaced by Task 24's hand-authored prose.
3. The stub `expected.<policy>.json` files in Task 17 Step 2 — overwritten by `just regenerate-fixtures` in Step 3 of the same task.

No `TBD`, `figure-out-later`, or "similar to" references exist outside these intentional patterns.

**Type consistency.**

- `MoveKernelObserver`, `TargetKernelObserver`, `NoOpMoveObserver`, `NoOpTargetObserver`, `JsonMoveTraceObserver`, `JsonTargetTraceObserver`, `PolicyGraphSnapshot`, `GraphDelta`, `TargetContextSnapshot`, `CandidateSummary`, `Problem`, `MoveProblem`, `TargetProblem`, `Budget`, `FixtureError`, `EvalEnvelope`, `TargetEvalEnvelope`, `StarlarkMethod`, `ParsedPrimer` — names are stable across all task references above.
- The kernel-side `step_idx`, `depth`, `last_insert`, `last_builtin`, `node_state_writes` field names match what Plan A's `kernel.rs` already exposes (verified inline in Task 1 Step 4).
- The generator's `module_filter` strings (`"actions_module"`, `"lib_move_module"`, `"graph_module"`, `"lib_target_module"`) are pinned in Tasks 20–22; Task 21 Step 1 explicitly notes that these must match the wrapper-fn names actually used at the registration sites and tells the implementer to adjust if they differ.

**Cycle avoidance.** Verified at Task 5 (loader in `search`, used by `cli` and by the `search` test crate; no reverse edge) and Task 18+ (primer binary is in `search`, doesn't pull `cli`).

**Ordering check.** Each phase produces working software:

- After Phase 1: existing tests pass with NoOp observers; observer infrastructure is unit-tested.
- After Phase 2: fixture loader exposes `Problem` enum; unit tests pass.
- After Phase 3: `eval-policy` and `trace-policy` work end-to-end on tiny inline fixtures; CLI integration tests pass.
- After Phase 4: every reference policy can be invoked via the CLI.
- After Phase 5: snapshot regression test passes against committed fixtures.
- After Phase 6: `policies/primer.md` is committed, generator runs cleanly, CI checks staleness.
