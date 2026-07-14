# Unified Measurement Annotations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace duplicated logical and physical annotation implementations with one level-aware API while retaining compatibility imports and behavior.

**Architecture:** A new `bloqade.gemini.measurement_annotations` module owns matrix validation, address-based qubit discovery, measurement resolution, and shared detector/observable emission. Logical and physical setup remain explicit branches selected by `level`; `logical_mvp` re-exports the unified function and the physical module keeps a thin legacy wrapper.

**Tech Stack:** Python 3.10+, Kirin IR, Bloqade SQuIn/Gemini dialects, pytest, uv.

---

### Task 1: Add the unified level-aware API

**Files:**
- Create: `python/bloqade/gemini/measurement_annotations.py`
- Modify: `python/bloqade/lanes/logical_mvp.py`
- Modify: `python/bloqade/gemini/device/physical_simulator.py`
- Test: `python/tests/test_cudaq_integration.py`
- Test: `python/tests/gemini/test_physical_simulator.py`

- [ ] **Step 1: Write failing unified-API tests**

Add tests that call both the canonical
`bloqade.gemini.measurement_annotations` API and the existing `logical_mvp`
re-export with `level="physical"`. Require an invalid level, empty matrix, and
ragged matrix to raise `ValueError`; require detector/observable matrices with
unequal row counts to raise `ValueError`; and verify `[[]]` remains a valid
zero-column matrix.

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
uv run pytest \
  python/tests/test_cudaq_integration.py -k "annotation_matrix or annotation_level" \
  python/tests/gemini/test_physical_simulator.py::test_unified_append_measurements_and_annotations_supports_physical_level \
  -q
```

Expected: failures because the current logical function has no `level`
parameter and does not consistently validate both matrices.

- [ ] **Step 3: Implement the shared module**

Move `_find_qubit_ssas`, `_find_return_stmt`, `_insert_before`, and matrix
validation into `measurement_annotations.py`. Implement
`append_measurements_and_annotations(..., level="logical")` with:

- logical setup that finds/creates `TerminalLogicalMeasurement` and resolves
  global matrix rows through nested results;
- physical setup that unrolls, validates physical block size, requires one
  `qubit.Measure`, and resolves flat measurement results;
- one shared loop for `SetDetector` and `SetObservable` emission.

Keep `append_measurements_and_annotations_physical` as a delegating wrapper.
Re-export the shared function and private compatibility helpers from the old
modules, removing their duplicate bodies and now-unused imports.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the command from Step 2. Expected: all selected tests pass.

- [ ] **Step 5: Run both affected test modules**

Run:

```bash
uv run pytest python/tests/test_cudaq_integration.py python/tests/gemini/test_physical_simulator.py -q
```

Expected: all tests pass.

### Task 2: Route `PhysicalSimulator.task` through the unified API

**Files:**
- Modify: `python/bloqade/gemini/device/physical_simulator.py`
- Test: `python/tests/gemini/test_physical_simulator.py`

- [ ] **Step 1: Update the task-delegation test first**

Patch the module's `append_measurements_and_annotations` binding, call
`PhysicalSimulator.task`, and assert it receives the kernel, matrices, and
`level="physical"`.

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
uv run pytest python/tests/gemini/test_physical_simulator.py::test_physical_simulator_task_passes_placement_strategy -q
```

Expected: failure because `task()` still invokes the legacy physical wrapper.

- [ ] **Step 3: Change the production call**

Have `PhysicalSimulator.task` call the unified function with
`level="physical"`; leave the old wrapper available only for compatibility.

- [ ] **Step 4: Run the affected test modules**

Run:

```bash
uv run pytest python/tests/test_cudaq_integration.py python/tests/gemini/test_physical_simulator.py -q
```

Expected: all tests pass.

### Task 3: Format and verify the refactor

**Files:**
- Modify only files changed by Tasks 1-2 if formatters require it.

- [ ] **Step 1: Run Python formatting and static checks**

Run:

```bash
uv run isort python/bloqade/gemini/measurement_annotations.py python/bloqade/lanes/logical_mvp.py python/bloqade/gemini/device/physical_simulator.py python/tests/test_cudaq_integration.py python/tests/gemini/test_physical_simulator.py
uv run black python/bloqade/gemini/measurement_annotations.py python/bloqade/lanes/logical_mvp.py python/bloqade/gemini/device/physical_simulator.py python/tests/test_cudaq_integration.py python/tests/gemini/test_physical_simulator.py
uv run ruff check python/bloqade/gemini/measurement_annotations.py python/bloqade/lanes/logical_mvp.py python/bloqade/gemini/device/physical_simulator.py python/tests/test_cudaq_integration.py python/tests/gemini/test_physical_simulator.py
uv run pyright python/bloqade/gemini/measurement_annotations.py python/bloqade/lanes/logical_mvp.py python/bloqade/gemini/device/physical_simulator.py
```

Expected: all commands exit zero.

- [ ] **Step 2: Run the full Python test suite**

Run:

```bash
just test-python
```

Expected: all Python tests pass.

- [ ] **Step 3: Review the final diff and commit**

Confirm only the shared module, compatibility modules, tests, and planning
documentation changed. Commit with:

```bash
git add python/bloqade/gemini/measurement_annotations.py \
  python/bloqade/lanes/logical_mvp.py \
  python/bloqade/gemini/device/physical_simulator.py \
  python/tests/test_cudaq_integration.py \
  python/tests/gemini/test_physical_simulator.py \
  docs/superpowers/plans/2026-07-14-unified-measurement-annotations.md
git commit -m "refactor(python): unify measurement annotation helpers"
```
