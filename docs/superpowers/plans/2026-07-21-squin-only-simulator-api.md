# Squin-Only Simulator API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Gemini logical and physical simulator objects accept prepared Squin `ir.Method` kernels only, removing constructor-driven annotation and implicit CUDA-Q conversion.

**Architecture:** Simulator task creation validates the public Squin-only boundary, then delegates to the existing logical or physical compilation pipeline. CUDA-Q conversion and detector/observable annotation remain standalone preprocessing utilities used explicitly by callers and demos.

**Tech Stack:** Python 3.12, Kirin IR, Bloqade Circuit/Gemini, pytest, Ruff, Black, Pyright.

---

## File Map

- Modify `python/bloqade/gemini/device/simulator.py`: remove matrix fields and the `LogicalKernel` union, enforce `ir.Method` input, and type all task-based helpers as Squin-only.
- Modify `python/bloqade/gemini/device/physical_simulator.py`: remove matrix fields and automatic physical annotation, and enforce `ir.Method` input.
- Modify `python/tests/test_device.py`: add logical API contract tests and migrate explicit annotation/CUDA-Q workflows.
- Modify `python/tests/gemini/test_physical_simulator.py`: add physical API contract tests and migrate constructor-driven annotation tests to explicit preprocessing.
- Modify `demo/cudaq_demo.py`: demonstrate explicit CUDA-Q conversion and annotation before simulator task creation.
- Preserve `python/bloqade/lanes/cudaq_integration.py` and annotation helper implementations.

### Task 1: Add Failing Squin-Only API Contract Tests

**Files:**
- Modify: `python/tests/test_device.py`
- Modify: `python/tests/gemini/test_physical_simulator.py`

- [ ] **Step 1: Test removed constructor arguments**

Add parametrized tests showing both simulator constructors reject the removed fields:

```python
@pytest.mark.parametrize("simulator_type", [GeminiLogicalSimulator, PhysicalSimulator])
@pytest.mark.parametrize("kwargs", [{"m2dets": [[1]]}, {"m2obs": [[1]]}])
def test_simulator_constructor_rejects_measurement_matrices(simulator_type, kwargs):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        simulator_type(**kwargs)
```

- [ ] **Step 2: Test non-Squin task rejection before compilation**

For each simulator, pass a plain callable and assert a clear `TypeError` containing
`Squin ir.Method`. Monkeypatch the logical compiler and physical pipeline constructor
to prove neither compilation path runs.

- [ ] **Step 3: Test delegation and logical source preservation**

For both simulator types, parameterize `run()`, `run_async()`, and `visualize()` with
a plain callable and assert they delegate through `task()` and expose the same clear
`TypeError`. Add a logical task test that records `kernel.print_str()`, creates a task,
and verifies the caller-owned method remains structurally unchanged and distinct from
`task.logical_squin_kernel`.

- [ ] **Step 4: Run the new tests and verify RED**

Run:

```bash
uv run pytest -q \
  python/tests/test_device.py \
  python/tests/gemini/test_physical_simulator.py \
  -k 'constructor_rejects_measurement_matrices or rejects_non_squin or preserves_source'
```

Expected: failures because matrix fields and implicit logical callable support still exist, and physical input currently fails with an incidental attribute error.

### Task 2: Implement the Squin-Only Simulator Boundary

**Files:**
- Modify: `python/bloqade/gemini/device/simulator.py`
- Modify: `python/bloqade/gemini/device/physical_simulator.py`

- [ ] **Step 1: Simplify logical simulator types and fields**

Remove `Any`, `Callable`, `Union`, and `LogicalKernel`. Remove `m2dets` and `m2obs`
from `GeminiLogicalSimulator`. Change every kernel parameter in `task()`, `run()`,
`run_async()`, `visualize()`, `physical_squin_kernel()`, `physical_move_kernel()`,
`tsim_circuit()`, and `fidelity_bounds()` to `ir.Method[[], RetType]`.

- [ ] **Step 2: Enforce the logical runtime boundary**

At the beginning of `GeminiLogicalSimulator.task()` add:

```python
if not isinstance(logical_kernel, ir.Method):
    raise TypeError("GeminiLogicalSimulator.task() requires a Squin ir.Method")
```

Call `compile_task(logical_kernel)` without matrices and update the docstring to state
that CUDA-Q conversion and annotation must happen before task creation.

- [ ] **Step 3: Remove physical constructor annotation behavior**

Remove `m2dets` and `m2obs` from `GeminiPhysicalSimulator`. Delete the conditional
call to `append_measurements_and_annotations_physical()` from `task()` while retaining
the standalone helper itself.

- [ ] **Step 4: Enforce the physical runtime boundary**

Before `physical_kernel.similar()` add:

```python
if not isinstance(physical_kernel, ir.Method):
    raise TypeError("GeminiPhysicalSimulator.task() requires a Squin ir.Method")
```

- [ ] **Step 5: Run contract tests and verify GREEN**

Run the Task 1 command. Expected: all selected tests pass.

### Task 3: Migrate Logical Annotation and CUDA-Q Tests

**Files:**
- Modify: `python/tests/test_device.py`

- [ ] **Step 1: Write explicit-preprocessing expectations**

Update logical detector/observable tests to call
`append_measurements_and_annotations(kernel, m2dets, m2obs)` before constructing the
simulator task. Preserve coverage for both matrices, detector-only, and
observable-only cases.

- [ ] **Step 2: Migrate CUDA-Q integration**

Change the CUDA-Q test workflow to:

```python
squin_kernel = cudaq_to_squin(bell_pair)
append_measurements_and_annotations(
    squin_kernel,
    m2dets if use_dets else None,
    m2obs if use_obs else None,
)
task = GeminiLogicalSimulator().task(squin_kernel)
```

Also assert `GeminiLogicalSimulator().task(bell_pair)` raises the public `TypeError`
in the contract test, without depending on CUDA-Q being installed.

- [ ] **Step 3: Run logical tests**

Run:

```bash
uv run pytest -q python/tests/test_device.py python/tests/test_cudaq_integration.py
```

Expected: all tests pass; existing unknown `slow` marker warnings may remain.

### Task 4: Migrate Physical Annotation Tests

**Files:**
- Modify: `python/tests/gemini/test_physical_simulator.py`

- [ ] **Step 1: Update task configuration test**

Remove matrix arguments and annotation monkeypatching from the placement-strategy
test. Replace its plain `MagicMock` input with a real minimal Squin `ir.Method` (and
monkeypatch that method's `similar()` result only if needed), so the new runtime type
guard is exercised without stopping the pipeline assertions. Continue asserting kernel
ownership, pipeline options, postprocessing, and task composition.

- [ ] **Step 2: Preserve explicit annotation coverage**

For source-preservation and PyQrack detector-fallback tests, create an owned test
kernel with `.similar()`, call `append_measurements_and_annotations_physical()`
explicitly, then pass that prepared kernel to `PhysicalSimulator().task()` or `.run()`.
Assert task creation does not add a second set of annotations and does not mutate the
prepared source.

- [ ] **Step 3: Run physical tests**

Run:

```bash
uv run pytest -q python/tests/gemini/test_physical_simulator.py
```

Expected: all tests pass.

### Task 5: Migrate the CUDA-Q Demo and Documentation Strings

**Files:**
- Modify: `demo/cudaq_demo.py`
- Modify: `python/bloqade/gemini/device/simulator.py`
- Modify: `python/bloqade/gemini/device/physical_simulator.py`

- [ ] **Step 1: Update the demo**

Import `cudaq_to_squin` and `append_measurements_and_annotations`, convert and annotate
`main_cuda`, then call `GeminiLogicalSimulator().task(squin_kernel)`.

- [ ] **Step 2: Update public docstrings**

State that simulator APIs require prepared Squin `ir.Method` kernels. Mention external
preprocessing in `task()` docstrings without adding new convenience APIs.

- [ ] **Step 3: Run the demo smoke test**

Run:

```bash
uv run python demo/cudaq_demo.py
```

Expected: the demo completes and prints detector results.

### Task 6: Format and Verify the Breaking Change

**Files:**
- All modified production, test, demo, and design/plan files.

- [ ] **Step 1: Format**

Run:

```bash
uv run black python/bloqade/gemini/device/simulator.py \
  python/bloqade/gemini/device/physical_simulator.py \
  python/tests/test_device.py \
  python/tests/gemini/test_physical_simulator.py \
  demo/cudaq_demo.py
uv run isort \
  python/bloqade/gemini/device/simulator.py \
  python/bloqade/gemini/device/physical_simulator.py \
  python/tests/test_device.py \
  python/tests/gemini/test_physical_simulator.py \
  demo/cudaq_demo.py
```

- [ ] **Step 2: Run static checks**

Run:

```bash
uv run ruff check \
  python/bloqade/gemini/device/simulator.py \
  python/bloqade/gemini/device/physical_simulator.py \
  python/tests/test_device.py \
  python/tests/gemini/test_physical_simulator.py \
  demo/cudaq_demo.py
uv run pyright \
  python/bloqade/gemini/device/simulator.py \
  python/bloqade/gemini/device/physical_simulator.py \
  python/tests/test_device.py \
  python/tests/gemini/test_physical_simulator.py \
  demo/cudaq_demo.py
```

Expected: no errors.

- [ ] **Step 3: Run focused regression tests**

Run:

```bash
uv run pytest -q \
  python/tests/test_device.py \
  python/tests/test_cudaq_integration.py \
  python/tests/gemini/test_physical_simulator.py \
  python/tests/gemini/test_optional_tsim_import.py \
  python/tests/gemini/test_simulator_backend.py
```

Expected: all tests pass except explicitly documented pre-existing failures, if any.

- [ ] **Step 4: Run the broader Python suite if focused tests are clean**

Run:

```bash
uv run pytest -q python/tests
```

Expected: all tests pass.

- [ ] **Step 5: Check patch integrity**

Run `git diff --check`, inspect `git status --short`, and verify unrelated user-owned
files remain unchanged.
