# Consolidate & dedupe `bloqade.lanes` transform APIs

**Issue:** [#801](https://github.com/QuEraComputing/bloqade-lanes/issues/801) (Goal 1 only)
**Date:** 2026-07-21
**Branch:** `refactor/consolidate-transforms-801`

## Context

`bloqade.lanes/__init__` was emptied of eager re-exports on the movement-dialect
branch to break a `lanes ↔ gemini` import cycle. (Note: that emptying is **not** yet
on `main`; this branch starts from `main`, where `__init__.py` still re-exports
`gemini.device`, `metrics`, and `steane_defaults`.) That work exposed how tangled and
duplicated the top-level `bloqade.lanes` module surface is.

The compilation stack currently has **two overlapping, behaviorally-different
implementations** of the squin→move direction:

- **Legacy** (`bloqade/lanes/upstream.py`): public `NativeToPlace`, `PlaceToMove`,
  and the `squin_to_move()` function. Monolithic; a single `logical_initialize` flag
  toggles the logical-vs-physical behavior.
- **Current** (`bloqade/lanes/pipeline/`): template-method `_NativeToPlaceBase` +
  `_PhysicalNativeToPlace`/`_LogicalNativeToPlace`, `_PlaceToMove`, and the
  `PhysicalPipeline`/`LogicalPipeline` orchestrators. `arch_spec` is the single
  source of truth; validation/rewrite steps differ per subclass.

The two forks are **not output-equivalent** (see "Behavioral gaps" below), so this is
not a mechanical delete-one-fork.

On top of these sit two duplicate wrapper modules with colliding function names:

- `bloqade/lanes/compile.py` — physical variants (`PhysicalPipeline`).
- `bloqade/lanes/logical_mvp.py` — logical variants (`LogicalPipeline`) plus
  Gemini-level orchestration (`compile_task`, measurement annotation, Steane
  defaults, cudaq conversion, stim emission).

The `move→squin` side (`bloqade/lanes/transform.py`: `MoveToSquin{Base,Logical,Physical}`,
`MoveToStackMove`) is already clean and well-factored — it is **not** a source of
duplication.

### Layering principle

`bloqade.lanes` = machine-agnostic movement compilation (lower layer);
`bloqade.gemini` = Gemini-machine specifics (higher layer, builds on lanes).
Long-term, `lanes` must not import `gemini`.

## Goals (this spec)

**Goal 1 — Simplify/dedupe the `bloqade.lanes` transform APIs into a single canonical
set of transformations**, and relocate the Gemini-level entry points up into
`bloqade.gemini`.

### Non-goals (explicitly deferred to a future issue — "Goal 2")

Achieving **zero `lanes → gemini` imports** is **out of scope**. The deep coupling in
the compilation core remains after this work:

- `rewrite/circuit2place.py` rewrites Gemini logical-dialect statements
  (`Initialize`, `StarRz`, `TerminalLogicalMeasurement`, `NewAt`).
- `dialects/move.py` + `place.py` call `gemini.star.validate_steane_star_support`.
- The pipelines wire in `GeminiLogicalValidation`, `_RewriteU3ToInitialize`,
  `RewriteSteaneTransversalCliffordAdjoints`, and terminal-measurement validation.

Severing these requires either relocating Gemini dialects down into lanes (violates
the layering principle) or inverting the dependency via rewrite/validation
registration hooks — a separate redesign of the squin→place stage. Also out of scope:
converting `metrics.Metrics` into an analysis pass (tracked separately).

## Design

### 1. Canonical transform sub-package in lanes

`bloqade/lanes/transform.py` (a module) becomes `bloqade/lanes/transform/` (a package),
absorbing today's `upstream.py`, `pipeline/`, and `transform.py`. The template-method
stage classes become the **single implementation per stage**.

```
bloqade/lanes/transform/
  __init__.py          # re-exports the canonical set (see below)
  native_to_place.py   # NativeToPlace (base) + _PhysicalNativeToPlace, _LogicalNativeToPlace
                        #   <- pipeline/base.py + pipeline/physical.py + pipeline/logical.py (native parts)
  place_to_move.py     # PlaceToMove                        <- pipeline/base.py (_PlaceToMove)
  pipeline.py          # PhysicalPipeline, LogicalPipeline, transversal_rewrites
                        #   <- pipeline/physical.py + pipeline/logical.py
  move_to_squin.py     # MoveToSquin{Base,Logical,Physical} <- transform.py
  move_to_stack.py     # MoveToStackMove                    <- transform.py
```

Deleted: `bloqade/lanes/upstream.py`, `bloqade/lanes/pipeline/` (whole package),
`bloqade/lanes/compile.py`.

**Canonical public set** (re-exported from `bloqade.lanes.transform`):

- `PhysicalPipeline`, `LogicalPipeline` — high-level orchestration (squin→move).
- `NativeToPlace` stage classes (public), `PlaceToMove` — for consumers that need
  stage-level control (was `upstream.NativeToPlace`/`PlaceToMove`, and internal
  `_NativeToPlaceBase` etc.). Naming of the public stage surface is settled during
  implementation; the private `_`-prefixed subclasses are promoted only as far as
  actual consumers (tracer) require.
- `MoveToSquinLogical`, `MoveToSquinPhysical` — move→squin.
- `MoveToStackMove` — move→stack_move (+ `emit_bytecode`).
- `transversal_rewrites` — Steane transversal rewrite helper.
- The noise-model ABCs currently surfaced via `transform.py`
  (`SimpleNoiseModel`, `SimpleLogicalNoiseModel`, `NoiseModelABC`,
  `LogicalNoiseModelABC`, `LogicalInitKernel`) continue to be importable where
  `noise_model.py` expects them.

The legacy `squin_to_move()` free function is removed; its consumers move to the
pipelines or stage classes (see the migration map). If implementation reveals a
genuine need for a bare NativeToPlace+PlaceToMove composition helper, it may be kept
as a thin convenience over the canonical stage classes — but only if a consumer
actually requires it after migration.

### 2. Gemini relocation (entry-point / orchestration layer)

Gemini-level orchestration moves up into `bloqade.gemini`, importing **downward** from
`lanes.transform` (no cycle):

```
bloqade/gemini/compile/__init__.py   # re-exports compile_task, compile_to_stim_program
bloqade/gemini/compile/task.py       # compile_task, append_measurements_and_annotations,
                                     #   run_squin_kernel_validation,
                                     #   _find_qubit_ssas, _find_return_stmt, _insert_before
bloqade/gemini/compile/stim.py       # compile_to_stim_program (RemoveReturn -> squin_to_stim -> EmitStim)
bloqade/gemini/steane_defaults.py    # moved verbatim from lanes/steane_defaults.py
bloqade/gemini/cudaq.py              # moved from lanes/cudaq_integration.py
```

`compile_to_stim_program`: the logical variant (from `logical_mvp.py`) is the one that
survives, since its only real consumers are Gemini-level. If a physical stim path is
still needed anywhere, it is expressed by composing `PhysicalPipeline` +
`MoveToSquinPhysical` + the stim emission in `stim.py`; this is confirmed during
implementation by the consumer audit.

### 3. Wrappers: delete the thin ones, keep real orchestration

**Deleted** (consumers switch to the canonical transform classes directly):

- `compile_squin_to_move` (both files) — one line over a Pipeline.
- `compile_to_physical_squin_noise_model` (both) — Pipeline + `MoveToSquin*`.
- `compile_squin_to_move_and_visualize` (both) — Pipeline + `visualize.debugger`.
- `compile_squin_to_move_best` (physical) — only `test_compile_api_split.py` uses it;
  retire with that test.

**Kept, relocated to `gemini.compile`:**

- `compile_task` + its helpers (`append_measurements_and_annotations`,
  `run_squin_kernel_validation`, `_find_qubit_ssas`, `_find_return_stmt`,
  `_insert_before`).
- `compile_to_stim_program`.

### 4. Consumer migration map

| Consumer | Now | After |
|---|---|---|
| `gemini/device/simulator.py`, `physical_simulator.py` | `logical_mvp.compile_task`, `logical_mvp._find_qubit_ssas` etc., `MoveToSquin*`, `PhysicalPipeline` | `gemini.compile.compile_task` (+ helpers), `lanes.transform.*` |
| demos: `ghz_moves_demo`, `msd`, `pipeline_demo`, `steane_demo`, `pipeline_details`, `move_lang`, `community_call_jan_28_2026`, `explicit_allocation` | wrappers + `upstream.*` | pipelines / transform classes directly (visualize inlined where `_and_visualize` was used) |
| `metrics.py` | `squin_to_move` + `transversal_rewrites` + `MoveToSquinLogical` | `LogicalPipeline` (behavior-preserving) — **flagged** |
| `visualize/entropy_tree/tracer.py` | `upstream.NativeToPlace(logical_initialize=False)` + raw `squin_to_move(logical_initialize=False)` | public stage classes / a stop-at-place path — **flagged** |
| `benchmarks/harness/runner.py` | `squin_to_move`, `transversal_rewrites`, `MoveToSquinLogical` | `PhysicalPipeline`/`LogicalPipeline`, `lanes.transform.*` |
| tests: `test_upstream`, `test_integration`, `test_validation_squin_kernels`, `test_explicit_allocation`, `test_e3_pipeline_validation`, `test_execute_cz_return`, `test_noise_init_generator`, `test_cudaq_integration`, `test_device`, `test_star_rz`, `test_shot_remapping`, `pipeline/*`, `test_measure_permutation`, `test_transform_move_to_stack_move` | mix of wrappers / `upstream` / `pipeline` / `transform` imports | canonical set; **retire `test_compile_api_split.py`** |
| docs / plans referencing `lanes._prelude`, `upstream.py` | old paths | new paths |

### 5. Behavioral-gap rule

Migrations MUST be output-preserving, verified by existing and new tests (TDD). The two
known gaps:

1. **`metrics.py`**: today it runs `squin_to_move(logical_initialize=True)` then
   `transversal_rewrites(move_mt)` *afterward*. `LogicalPipeline(transversal_rewrite=True)`
   additionally runs `RewriteSteaneTransversalCliffordAdjoints` at the native stage, so
   it is not equivalent. The migration must reproduce metrics' exact sequence — either
   by driving the stage classes in the same order, or by adding an explicit option to
   the pipeline. Do not silently adopt the pipeline's transversal ordering.
2. **`tracer.py`**: needs a raw "stop at place" / no-pinned-lowering path to capture
   placement entropy traces. Expose this explicitly on the canonical stage classes
   (e.g. a public `NativeToPlace` usable standalone), rather than reintroducing
   `upstream`.

Where a pipeline genuinely cannot express a legacy mode, extend the canonical class
with a minimal, explicit option. Any unavoidable output change is called out in the PR
description.

### 6. Prelude consolidation (issue WS3)

- Delete `bloqade/lanes/prelude.py` (legacy; uses `kirin.prelude.basic`).
- Point `rewrite/transversal.py` at the surviving prelude.
- Promote `_prelude.py` → public `prelude.py` (uses `kirin.prelude.structural`).

This changes `transversal.py`'s dialect-group base from `basic` to `structural`; it is
gated on `transversal.py`'s tests (and the transversal-dependent pipeline tests)
passing. Update the other `_prelude` importers (tests + demos) to the new `prelude`
path.

### 7. `__init__.py` cleanup

Empty the top-level `bloqade/lanes/__init__.py` re-exports (they still import
`gemini.device` upward on `main`, plus `metrics`, `steane_defaults`, `noise_model`).
Consumers import from submodules / `bloqade.gemini` directly. `metrics.py`'s
`transversal_rewrites` import repoints from `lanes.logical_mvp` to
`lanes.transform.pipeline`, removing that lanes→(to-be-gemini) edge.

## Testing strategy

- **TDD, behavior-preserving.** The existing test suite is the primary regression gate;
  run `just test-python` (or `uv run coverage run -m pytest python/tests`) throughout.
- Retire `python/tests/test_compile_api_split.py` (it asserts the old two-module split
  and the deleted `compile_squin_to_move_best`). Any still-relevant assertions in it
  (e.g. LogicalPipeline arch-spec default propagation) move into `pipeline`/`transform`
  tests.
- Add/keep tests that pin the two flagged behaviors (metrics transversal ordering;
  tracer stop-at-place path) so the migration is provably output-preserving.
- Demos are smoke-tested via `just demo`.
- Lint/type gates: `uv run black python`, `uv run isort python`,
  `uv run ruff check python`, `uv run pyright python`.

## Risks & mitigations

- **Silent behavior change on migration** (metrics, tracer, physical raw path).
  Mitigation: the behavioral-gap rule (§5) + explicit tests.
- **Wide consumer churn** (gemini device, ~12 test files, ~8 demos, benchmarks).
  Mitigation: the migration map (§4) enumerates every site; work is mechanical once
  the canonical API names are fixed.
- **Prelude `basic`→`structural` change** for `transversal.py`. Mitigation: gate on
  transversal tests.
- **Import-order regressions** while moving modules across the `lanes`/`gemini`
  boundary. Mitigation: relocations are downward-only (`gemini` → `lanes`); run the
  full import/test suite after each relocation.

## Success criteria

1. A single canonical transform sub-package `bloqade.lanes.transform` with one
   implementation per stage; `upstream.py`, `pipeline/`, and `compile.py` deleted.
2. `compile_task` + `compile_to_stim_program` (+ helpers), `steane_defaults`, and
   cudaq integration live under `bloqade.gemini`; thin wrappers deleted.
3. `prelude.py` removed; `_prelude` promoted to public `prelude`; `transversal.py`
   migrated.
4. `bloqade/lanes/__init__.py` no longer re-exports; `metrics` no longer imports via
   `logical_mvp`.
5. Full Python test suite green; demos smoke-test clean; lint/type gates pass.
6. No new `lanes → gemini` edges introduced (existing Tier-2 edges may remain; Goal 2
   is deferred).
