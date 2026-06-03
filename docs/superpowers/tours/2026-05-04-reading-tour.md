# Reading tour of the Move Policy DSL branch

**Status:** Curriculum draft (2026-05-04). Uncommitted, like the spec and plan files — a working document, not a normative artifact.

**Branch:** `jason/move-policy-dsl` (worktree at `~/.config/superpowers/worktrees/bloqade-lanes/move-policy-dsl/`)

**Audience:** Jason, who knows the design (was the brainstorming partner) but has limited Rust knowledge and now wants to walk through the actual code that landed.

**Related artifacts:**
- Plan A spec/plan: `docs/superpowers/plans/2026-04-29-move-policy-dsl.md` (committed in repo)
- Plan C design: `docs/superpowers/specs/2026-05-01-move-policy-dsl-plan-c-design.md` (uncommitted)
- Plan C implementation plan: `docs/superpowers/plans/2026-05-01-move-policy-dsl-plan-c.md` (uncommitted)

Eight sessions, ordered by load-bearing importance. Each session aims for ~30–60 min and ends with a "you should now be able to..." checkpoint. Sessions are interactive — Jason asks questions, Claude leads. Rust concepts get explained inline as they come up.

---

## Session 0 — Architecture refresher (no code, ~15 min)

**Goal:** Re-anchor the big picture so the code reading lands in context.

**What to cover:**
- Why a DSL exists at all (the "policy-as-data" problem: people want to author search heuristics without forking the Rust compiler).
- The two DSL surfaces and how they differ:
  - **Move Policy DSL** drives a search loop (many ticks, transposition table, frontier).
  - **Target Generator DSL** is one-shot per CZ stage (no loop).
- The shared substrate (`bloqade-lanes-dsl-core`) versus the per-DSL kernels.
- The data flow: Python user → `.star` file → Starlark evaluator (sandboxed) → Rust kernel calls back into evaluator each tick → kernel mutates graph → returns result.
- The crate graph: `dsl-core` ← `search` ← `bytecode-cli` and `← bytecode-python`.

**Checkpoint:** You can explain which crate "owns" each major concept (sandbox, kernel, observers, CLI, Python bindings).

---

## Session 1 — The Move kernel run loop *(THE heart, ~60 min)*

**Goal:** Understand the single most important function in the whole branch.

**Files (read in this order):**
1. `crates/bloqade-lanes-search/src/move_policy_dsl/kernel.rs` — `solve_with_policy(...)` (~700 lines)
2. `crates/bloqade-lanes-search/src/move_policy_dsl/mod.rs` (10 lines, just to see the module shape)

**Rust concepts introduced (taught as we hit them):**
- **Functions with generic `impl Trait` parameters** — `initial: impl IntoIterator<Item = (u32, LocationAddr)>` is the equivalent of Python's "anything iterable yielding pairs."
- **`Arc<T>`** — reference-counted pointer for sharing immutable data across threads. Used here for `LaneIndex` (we don't want to copy the whole arch spec on every call).
- **`Mutex<T>` / `Arc<Mutex<T>>`** — interior mutability for the policy graph that's read by both Rust and Starlark. The lock disciplines explain why some sections look weirdly indented.
- **`Result<T, E>` and the `?` operator** — Rust's "raise on error" idiom.
- **Trait objects (`&mut dyn MoveKernelObserver`)** — dynamic dispatch (like a Python interface).
- **Pattern matching** (`match action { MoveAction::Halt { .. } => ... }`) — exhaustive enum dispatch.

**What to trace through:**
1. Function signature and what each parameter means.
2. Setup phase — building `Config`, `LaneIndex`, `DistanceTable`.
3. Loading the `.star` file (single line — the heavy lifting is in `dsl-core::adapter`).
4. The main tick loop:
   - Call into Starlark `step(graph, gs, ctx, lib)` and get a list of actions back.
   - For each action: `apply_action(...)` mutates the graph; observer fires.
   - Halt / budget / timeout checks.
5. Result construction.

**Checkpoint:** You can describe the lifecycle of a single `step(...)` invocation: from Rust calling into Starlark, getting actions back, applying them, and what the policy sees on the next tick.

---

## Session 2 — The action contract *(~30 min)*

**Goal:** Understand the only language policies have for *changing things*.

**Files:**
1. `crates/bloqade-lanes-search/src/move_policy_dsl/actions.rs` — `MoveAction` enum + the `register_actions` Starlark module
2. `policies/reference/entropy.star` lines 240-252 — see actions composed in the wild

**Rust concepts:**
- **Sum types (`enum`)** — Rust enums carry data per variant, like tagged unions. `MoveAction::InsertChild { parent, move_set }` is a struct-shaped variant; `Halt { status, message }` is another.
- **`#[derive(Serialize)]`** — proc macros that auto-generate trait implementations. We use this to make actions emit as JSON.
- **`#[starlark_module]`** — a proc macro from `starlark-rust` that turns a Rust function into a registry of Starlark-callable verbs.

**What to trace:**
- The 6 verbs: `insert_child`, `update_node_state`, `update_global_state`, `emit_solution`, `halt`, `invoke_builtin`.
- How the kernel maps each variant to a graph mutation in `apply_action`.
- The `halt` status mapping (`"solved"`/`"unsolvable"`/`"fallback"`/`"error"` → `PolicyStatus`).

**Checkpoint:** You can read `entropy.star`'s `step(...)` function and predict what graph mutations each return statement causes.

---

## Session 3 — What the policy can read *(~45 min)*

**Goal:** Understand the read-side surface that policies query each tick.

**Files:**
1. `crates/bloqade-lanes-search/src/move_policy_dsl/graph_handle.rs` — `graph.*` (parent, depth, children_of, is_goal, last_insert, etc.)
2. `crates/bloqade-lanes-search/src/move_policy_dsl/lib_move.rs` — `lib.*` (strategy-neutral queries: unresolved_qubits / legal_lanes / scored_lane, plus the Rust-backed pack_aod_rectangles)
3. `crates/bloqade-lanes-search/src/graph.rs` — the underlying `SearchGraph` data structure (just the type definitions — we won't read the whole thing)

**Rust concepts:**
- **Lifetimes (`'v`)** — Starlark values are tied to the evaluator's heap; the `'v` lifetime says "this value lives as long as the evaluator does." It's plumbed through every accessor.
- **Newtype wrappers** — `NodeId(u32)` is just a `u32` with a different name to prevent confusing it with other indices.
- **`HashMap` / `BTreeMap`** — ordered vs unordered associative containers.
- **`Result<Value<'v>, anyhow::Error>`** — the canonical "may-fail returning a Starlark value" signature.

**What to trace:**
- `graph.depth(node)` — read access into the locked `PolicyGraphInner`.
- The candidate pipeline as four composable stages.
- Why the pipeline returns "lane sets" not "qubit moves" (AOD parallelism).

**Checkpoint:** You can read a policy-owned scoring loop over `lib.unresolved_qubits(...)` and `lib.legal_lanes(...)`, then explain how `lib.pack_aod_rectangles(...)` converts selected scored lanes into legal move sets.

---

## Session 4 — The Starlark host *(~30 min)*

**Goal:** Understand how a `.star` file becomes a callable Rust function.

**Files:**
1. `crates/bloqade-lanes-dsl-core/src/sandbox.rs` — `SandboxConfig` + `make_evaluator()`
2. `crates/bloqade-lanes-dsl-core/src/adapter.rs` — `LoadedPolicy::from_path(...)` + freezing
3. `crates/bloqade-lanes-dsl-core/src/policy_trait.rs` — the generic `Policy<H,A,I,N,G>` trait

**Rust concepts:**
- **Generic trait parameters** — the `Policy` trait abstracts over five associated types so it's reusable for both Move (rich) and Target (simple) DSLs.
- **Frozen modules** — Starlark loads, evaluates the top of the file, then "freezes" the module so all globals (`PARAMS`, `DEFAULTS`, the `step` function) become read-only. This is what makes calling `step` 1,000 times per solve cheap.

**What to trace:**
- Reading a `.star` file from disk to running its `init` function once.
- The sandbox: no I/O, no `load()`, no recursion.
- The policy-params override mechanism (`PARAMS_OVERRIDE` global injected pre-freeze).

**Checkpoint:** You understand why authoring a policy costs almost nothing per call once the module is loaded once.

---

## Session 5 — The Target side *(~20 min)*

**Goal:** See how the Move pattern simplifies when there's no loop.

**Files:**
1. `crates/bloqade-lanes-search/src/target_generator_dsl/kernel.rs` — `TargetPolicyRunner::generate(...)` and `run_target_policy(...)`
2. `crates/bloqade-lanes-search/src/target_generator_dsl/lib_target.rs` — the (single) query primitive
3. `crates/bloqade-lanes-search/src/target_generator_dsl/ctx_handle.rs` — the read-only context the policy sees
4. `policies/reference/default_target.star` — the only target policy

**What to cover:**
- One call per CZ stage = no transposition table, no frontier.
- The validator (`validate_candidate`) re-uses the existing Rust safety enforcement — the policy is responsible for *generating* candidates; the kernel is responsible for *rejecting* unsafe ones.
- Why this side is so much smaller than Move.

**Checkpoint:** You can articulate the architectural reasons the Target DSL is ~30% the code of the Move DSL.

---

## Session 6 — The Python bridge *(~20 min)*

**Goal:** Understand how Python users invoke the DSL and where the marshaling happens.

**Files:**
1. `crates/bloqade-lanes-bytecode-python/src/search_python.rs` — focus on the `MoveSolver.solve(...)` method's `policy_path` branch
2. `crates/bloqade-lanes-bytecode-python/src/target_generator_dsl_python.rs` — `PyTargetPolicyRunner` class
3. `python/bloqade/lanes/bytecode/_native.pyi` — the Python type stubs (what users see)

**Rust concepts:**
- **`#[pyclass]` / `#[pymethods]`** — `pyo3` macros that expose Rust types as Python classes.
- **`PyResult<T>`** — wraps the GIL release/reacquire dance into normal `Result`.

**What to trace:**
- A Python call `solver.solve(initial, target, blocked, policy_path="x.star")` end-to-end into `solve_with_policy`.
- How errors map: Rust `DslError` → Python exception class hierarchy.

**Checkpoint:** You can point at the file:line in `search_python.rs` where the Python-to-Rust handoff happens.

---

## Session 7 — The Plan C CLI + observers *(~30 min)*

**Goal:** Understand what was added on top of A+B to make the framework usable from the shell and debuggable.

**Files:**
1. `crates/bloqade-lanes-search/src/move_policy_dsl/observer.rs` — the trait, NoOp, JSON impl
2. `crates/bloqade-lanes-search/src/fixture.rs` — JSON schema for problem files
3. `crates/bloqade-lanes-bytecode-cli/src/main.rs` — clap subcommand registration
4. `crates/bloqade-lanes-bytecode-cli/src/policy/eval.rs` — the `eval-policy` subcommand
5. `crates/bloqade-lanes-bytecode-cli/src/policy/trace.rs` — the `trace-policy` subcommand (with NDJSON streaming via the JSON observer)

**Rust concepts:**
- **`thiserror`** — derive-based error enums.
- **`schemars`** — derive-based JSON Schema generation (used to keep fixture schema and primer in sync).
- **`clap` derive** — subcommand definitions via struct attributes.

**What to trace:**
- The observer trait's "default empty body" pattern that makes adding hooks non-breaking.
- The JSON envelope versioning (`"v": 1`) that shields downstream tooling from future schema changes.
- How `trace-policy --json` gives you a streaming NDJSON transcript.

**Checkpoint:** You can describe what each kernel hook (`on_init`, `on_step`, `on_builtin`, `on_halt`) emits and when.

---

## Session 8 — The supporting cast *(~30 min)*

**Goal:** Understand the meta-tooling — the parts that keep the documentation honest.

**Files:**
1. `crates/bloqade-lanes-search/src/bin/policies-primer.rs` — auto-generates `policies/primer.md` by parsing its own crate's source via `syn`
2. `crates/bloqade-lanes-search/tests/dsl_snapshot.rs` — snapshot regression test driver
3. `policies/fixtures/` — the corpus structure
4. `crates/bloqade-lanes-search/tests/primer_golden.rs` — golden-file test for the generator itself

**Rust concepts:**
- **Procedural macros vs. compile-time codegen vs. source inspection** — three different "metaprogramming" categories. The primer uses the third (read its own source via `include_str!`, parse via `syn`, render markdown).
- **Binary targets in a library crate** — `src/bin/policies-primer.rs` produces a separate executable that ships with the search crate.
- **Integration tests** (`tests/*.rs`) vs. unit tests (inline `#[cfg(test)] mod tests`).

**What to cover:**
- Why `include_str!` instead of runtime file reads (deterministic, no path-resolution bugs).
- The structural-match comparison (`{status, halt_reason, expansions, max_depth}`) and why it's robust against benign refactors.
- The "PROSE blocks survive regeneration" trick (sentinel comments).

**Checkpoint:** You can explain the loop "edit `actions.rs` → `just generate-primer` → primer.md updates → CI's `check-primer` catches stale primers."

---

## Suggested cadence

- **One session per work session.** No rush. Each one stands alone — you can stop after Session 1 and have a useful mental model.
- **Sessions 1–3 are the load-bearing ones.** If you only do those, you'll understand 80% of the engine.
- **Sessions 4–8 are nice-to-haves.** They cover the integration glue (Python, CLI, tests, docs). You can cherry-pick as questions come up.
