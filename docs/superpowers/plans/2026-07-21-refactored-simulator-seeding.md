# Refactored Simulator Seeding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port `origin/main` per-call seed behavior into the backend-based logical and physical simulator framework and make PyQrack sampling fully reproducible.

**Architecture:** `_SimulatorTaskBase` is the sole per-call seed validation boundary and forwards valid seeds to backend sampling. Tsim, CliffT, and PyQrack own their seed-specific execution details; PyQrack seeds both its NumPy noise stream and each fresh native Qrack simulator. Simulator convenience APIs only forward seed values.

**Tech Stack:** Python 3.10+, Kirin IR, NumPy, Stim/Tsim, CliffT, PyQrack, pytest, Black, Ruff, Pyright

---

The worktree already contains staged user edits. Do not reset, unstage, or
commit them. Add implementation edits in the working tree and report the final
staged/unstaged split. Before editing, record the output of
`git diff --cached --binary | shasum -a 256`; the same command must produce the
same checksum after implementation.

### Task 1: Break the import cycle and restore property contracts

**Files:**
- Modify: `python/bloqade/gemini/device/_task_runtime.py`
- Modify: `python/bloqade/gemini/device/simulator_backend.py`
- Test: `python/tests/gemini/test_optional_tsim_import.py`
- Test: `python/tests/gemini/test_physical_simulator.py`

- [ ] **Step 1: Add an import-boundary regression test and verify RED**

Add a subprocess test to `test_optional_tsim_import.py` which imports
`bloqade.gemini.device` and fails if that import loads `tests` or a `tests.*`
module. This models an installed wheel where the repository test package is
not available.

Run:

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m pytest \
  python/tests/gemini/test_optional_tsim_import.py::test_simulator_runtime_does_not_import_test_modules -q
```

Expected: FAIL because `_task_runtime.py` currently imports
`tests.gemini.validation.test_physical_terminal_measure`.

- [ ] **Step 2: Add property-contract tests and verify RED**

Using mock Tsim circuits in `test_physical_simulator.py`, access each of the
four cached properties and assert that `compile_sampler()` or
`compile_detector_sampler()` is called with no arguments—not `seed=None`.

Run:

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m pytest \
  python/tests/gemini/test_physical_simulator.py \
  -k 'task_cached_sampler_properties_compile_without_seed_argument' -q
```

Expected: FAIL because the current parameterized cached-property getters pass
`seed=None` to the compilers.

- [ ] **Step 3: Apply the minimal import and property fixes**

Remove the accidental production import from
`tests.gemini.validation.test_physical_terminal_measure`. Keep `_validate_seed`
defined in `_task_runtime.py`. Confirm `simulator_backend.py` imports neither
`simulator.py` nor `_task_runtime.py`; backends do not validate per-call seeds
directly.

Restore the four task sampler cached properties to parameterless getters:

```python
@cached_property
def measurement_sampler(self):
    return self.tsim_circuit.compile_sampler()
```

Apply the same pattern to the noiseless measurement and detector properties.

- [ ] **Step 4: Verify GREEN**

Run the import test above and:

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -c "import bloqade.gemini.device"

MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m pytest \
  python/tests/gemini/test_physical_simulator.py \
  -k 'task_cached_sampler_properties_compile_without_seed_argument' -q
```

Expected: PASS with no optional simulator backend imported eagerly.

### Task 2: Specify the shared task seed contract

**Files:**
- Modify: `python/tests/test_device.py`
- Modify: `python/tests/gemini/test_physical_simulator.py`
- Modify if required: `python/bloqade/gemini/device/_task_runtime.py`

- [ ] **Step 1: Update existing backend-routing expectations**

Because `_SimulatorTaskBase` now always forwards the keyword, update existing
assertions to expect:

```python
backend.sample.assert_called_once_with(
    "noisy-kernel",
    shots=1,
    run_detectors=False,
    seed=None,
)
```

- [ ] **Step 2: Add task forwarding and validation characterization tests**

Adapt the applicable `origin/main` tests to `_SimulatorTaskBase` through the
existing logical and physical mock-task helpers. Cover:

```python
@pytest.mark.parametrize("seed", [0, 2**63 - 1])
def test_task_run_accepts_valid_seed_values(seed): ...

@pytest.mark.parametrize("seed", [True, -1, 2**63, 1.5, "1"])
def test_task_run_rejects_invalid_seed_before_dem_or_sampling(seed): ...
```

The invalid test must assert that neither `detector_error_model` nor `sample`
was called. Add `run_async` coverage asserting validation happens before the
executor's `submit()` and valid seeds are forwarded to `run()` for detector and
measurement paths. Add a separate `task.run(seed=None)` test so explicit `None`
is covered independently from omission of the argument.

- [ ] **Step 3: Run the focused characterization tests**

Run:

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m pytest \
  python/tests/test_device.py \
  -k 'seed or samples_noisy or samples_noiseless or detectors_uses_native' -q

MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m pytest \
  python/tests/gemini/test_physical_simulator.py \
  -k 'seed or task_run_routes' -q
```

These tests may begin GREEN because much of the behavior is present in the
user's staged implementation. Record passing cases as characterization
coverage. Treat only a failure against the approved contract as RED requiring a
production correction.

- [ ] **Step 4: Make the minimal runtime corrections**

Keep `_validate_seed(seed)` as the first operation in both `run()` and
`run_async()`. Ensure all overloads have a trailing comma after the seed
annotation and both paths forward `seed=seed` to the backend or executor.

- [ ] **Step 5: Verify GREEN**

Re-run the focused command and expect PASS.

### Task 3: Port Tsim and CliffT seed behavior to backend tests

**Files:**
- Modify: `python/tests/gemini/test_simulator_backend.py`
- Modify if required: `python/bloqade/gemini/device/simulator_backend.py`

- [ ] **Step 1: Add Tsim seed-path characterization tests**

Update unseeded mock expectations to the chosen backend call contract and add
`seed=0` coverage for:

- Clifford measurement sampler compilation;
- Clifford detector measurement sampling plus M2D conversion;
- non-Clifford measurement sampler compilation;
- non-Clifford detector sampler compilation.

Add a real Stim fixed-batch test using `_random_physical_kernel` showing two
independent `sample(..., shots=64, seed=17)` calls are equal.

- [ ] **Step 2: Add CliffT precedence characterization tests**

Adapt the deleted `_CliffTSimulatorTask` tests to
`CliffTSimulatorBackend.sample()`:

```python
backend = CliffTSimulatorBackend(seed=123, tsim_backend=tsim_backend)
backend.sample(kernel, shots=2, seed=0)
clifft.sample.assert_called_once_with("program", shots=2, seed=0)
```

Also verify `seed=None` falls back to `backend.seed`, and both missing seeds omit
the keyword.

- [ ] **Step 3: Run focused backend tests**

Run:

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m pytest \
  python/tests/gemini/test_simulator_backend.py -k 'tsim or clifft' -q
```

These may begin GREEN because they characterize the staged implementation.
Only contract failures proceed to a production correction.

- [ ] **Step 4: Correct backend seed forwarding and verify GREEN**

Preserve `seed=0`, pass the seed into each Tsim sampler compiler, and compute
CliffT's effective seed as the per-call value when present and backend value
otherwise. Re-run the focused tests.

### Task 4: Test logical and physical convenience forwarding

**Files:**
- Modify: `python/tests/test_device.py`
- Modify: `python/tests/gemini/test_physical_simulator.py`
- Modify if required: `python/bloqade/gemini/device/simulator.py`
- Modify if required: `python/bloqade/gemini/device/physical_simulator.py`

- [ ] **Step 1: Add synchronous forwarding characterization tests**

Mock each simulator's `task()` result and verify `run(..., seed=0)` forwards the
seed unchanged for both `run_detectors=False` and `True`. Parameterize the
logical simulator test over `[None, 0]` to preserve `origin/main` explicit-None
coverage. These tests may begin GREEN because they characterize the user's
existing implementation.

- [ ] **Step 2: Run the synchronous tests**

If a test fails for a behavior required by the spec, make the minimal
production correction and re-run it. If it starts GREEN, record it as
characterization coverage and do not change production code.

Run:

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m pytest \
  python/tests/test_device.py::test_logical_simulator_run_forwards_seed \
  python/tests/gemini/test_physical_simulator.py::test_physical_simulator_run_forwards_seed -q
```

- [ ] **Step 3: Add asynchronous forwarding characterization tests**

Return an already-resolved `Future` from the mock task and verify
`run_async(..., seed=0).result()` forwards the seed for both detector modes.

- [ ] **Step 4: Run the asynchronous tests**

If a required behavior fails, make the minimal production correction and
re-run. Otherwise retain the passing characterization tests.

Run:

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m pytest \
  python/tests/test_device.py::test_logical_simulator_run_async_forwards_seed \
  python/tests/gemini/test_physical_simulator.py::test_physical_simulator_run_async_forwards_seed -q
```

- [ ] **Step 5: Verify the full convenience-method subset**

Run:

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m pytest \
  python/tests/test_device.py -k 'simulator and seed' \
  python/tests/gemini/test_physical_simulator.py -k 'simulator and seed' -q
```

### Task 5: Verify both PyQrack random streams

**Files:**
- Modify: `python/tests/gemini/test_simulator_backend.py`
- Modify if required: `python/bloqade/gemini/device/simulator_backend.py`

- [ ] **Step 1: Add native measurement reproducibility characterization test**

Use `_random_physical_kernel`, which applies `H` before measurement:

```python
backend = PyQrackSimulatorBackend()
first = backend.sample(kernel, shots=64, seed=441)
second = backend.sample(kernel, shots=64, seed=441)
assert np.array_equal(first.measurements, second.measurements)
```

Assert the batch contains both outcomes so a bug that gives every fresh native
simulator the same first random draw is detected.

- [ ] **Step 2: Run the reproducibility test**

If it fails because only NumPy noise is seeded, retain the RED result and make
the minimal native-Qrack seeding correction after Step 3 specifies the exact
seed calls. If it starts GREEN, treat it as coverage of the user's existing
implementation.

Run:

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m pytest \
  python/tests/gemini/test_simulator_backend.py::test_pyqrack_per_call_seed_reproduces_native_measurements -q
```

- [ ] **Step 3: Add derived native-seed correction test**

Patch `pyqrack.QrackSimulator.seed`, record one call per shot, and compare the
recorded arguments with the fresh deterministic draws expected from
`np.random.default_rng(call_seed)`. Do not require mathematical uniqueness.
Run this test and confirm whether the current implementation satisfies it.

Run:

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m pytest \
  python/tests/gemini/test_simulator_backend.py::test_pyqrack_derives_fresh_native_seed_for_each_shot -q
```

- [ ] **Step 4: Add persistent-stream isolation characterization test**

Create control and subject backends with the same constructor seed. Advance
both once, execute an explicit per-call seeded request only on the subject,
then verify their next unseeded fixed-size batches still match. This proves the
per-call generator does not advance `_rng_state`.

- [ ] **Step 5: Run the PyQrack subset**

Run:

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m pytest \
  python/tests/gemini/test_simulator_backend.py -k pyqrack -q
```

For any failing required behavior, make only the changes required to seed the
native register after `super().initialize()` and to select a local generator
for explicit call seeds, then re-run the subset to GREEN.

### Task 6: Format and verify the complete change

**Files:**
- Verify all modified production and test files

- [ ] **Step 1: Format the touched Python files**

Run:

```bash
.venv/bin/python -m black \
  python/bloqade/gemini/device/_task_runtime.py \
  python/bloqade/gemini/device/simulator_backend.py \
  python/bloqade/gemini/device/simulator.py \
  python/bloqade/gemini/device/physical_simulator.py \
  python/tests/gemini/test_optional_tsim_import.py \
  python/tests/gemini/test_simulator_backend.py \
  python/tests/gemini/test_physical_simulator.py \
  python/tests/test_device.py

.venv/bin/python -m isort \
  python/bloqade/gemini/device/_task_runtime.py \
  python/bloqade/gemini/device/simulator_backend.py \
  python/bloqade/gemini/device/simulator.py \
  python/bloqade/gemini/device/physical_simulator.py \
  python/tests/gemini/test_optional_tsim_import.py \
  python/tests/gemini/test_simulator_backend.py \
  python/tests/gemini/test_physical_simulator.py \
  python/tests/test_device.py
```

- [ ] **Step 2: Run focused tests**

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m pytest \
  python/tests/gemini/test_optional_tsim_import.py \
  python/tests/gemini/test_simulator_backend.py \
  python/tests/gemini/test_physical_simulator.py \
  python/tests/test_device.py -q
```

- [ ] **Step 3: Run static checks**

```bash
.venv/bin/python -m ruff check \
  python/bloqade/gemini/device/_task_runtime.py \
  python/bloqade/gemini/device/simulator_backend.py \
  python/bloqade/gemini/device/simulator.py \
  python/bloqade/gemini/device/physical_simulator.py \
  python/tests/gemini/test_optional_tsim_import.py \
  python/tests/gemini/test_simulator_backend.py \
  python/tests/gemini/test_physical_simulator.py \
  python/tests/test_device.py
.venv/bin/python -m pyright \
  python/bloqade/gemini/device \
  python/tests/gemini/test_simulator_backend.py \
  python/tests/gemini/test_physical_simulator.py \
  python/tests/test_device.py
```

- [ ] **Step 4: Run the broader Python suite if focused checks pass**

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m pytest python/tests -q
```

- [ ] **Step 5: Inspect the final diff**

Run `git diff --check`, `git diff --cached --check`, `git diff --cached --stat`,
`git diff --stat`, and `git status --short`. Re-run
`git diff --cached --binary | shasum -a 256` and require it to match the initial
checksum. Confirm that unrelated notebook and user files were not modified and
report any pre-existing failures separately.
