# Backend-Configured Simulator Sampling Results Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove per-call detector sampling flags, configure detector-only sampling on Tsim, and expose one structural result API over measurement-backed and detector-backed results.

**Architecture:** Backends return exactly one of two `BackendSample` representations: measurements only, or detectors plus observables only. The shared task runtime validates the payload without inspecting the backend type and returns either `Result` or `DetectorResult`, both structurally satisfying a public generic `SimulatorResult` protocol.

**Tech Stack:** Python 3.10+, dataclasses, `typing.Protocol`, NumPy, Kirin IR, Tsim/Stim, CliffT, PyQrack, pytest, Pyright, Ruff, Black, isort.

---

## File Structure

- Modify `python/bloqade/gemini/device/_task_runtime.py`: define the public result protocol, complete `DetectorResult`, validate backend payloads, and simplify task run methods.
- Modify `python/bloqade/gemini/device/simulator_backend.py`: simplify the backend contract and move Tsim detector selection into backend state.
- Modify `python/bloqade/gemini/device/simulator.py`: simplify logical simulator forwarding and return annotations.
- Modify `python/bloqade/gemini/device/physical_simulator.py`: simplify physical simulator forwarding and return annotations.
- Modify `python/bloqade/gemini/device/__init__.py` and `python/bloqade/gemini/__init__.py`: export `SimulatorResult`.
- Modify `python/bloqade/gemini/decoding/experiments.py`: consume the protocol and stop passing the removed flag.
- Modify `python/tests/gemini/test_simulator_backend.py`: test backend configuration and payload contracts.
- Modify `python/tests/test_device.py`: test logical task routing, protocol typing, and integration.
- Modify `python/tests/gemini/test_physical_simulator.py`: test physical simulator forwarding and typing.
- Modify `python/tests/gemini/decoding/test_experiment.py`: update the decoding call contract.
- Do not modify `demo/simulator_device_demo.ipynb` or unrelated untracked files.

### Task 1: Define the Common Result Protocol

**Files:**
- Modify: `python/bloqade/gemini/device/_task_runtime.py:31-101`
- Modify: `python/bloqade/gemini/device/__init__.py:8-18`
- Modify: `python/bloqade/gemini/__init__.py:1-24`
- Test: `python/tests/test_device.py`

- [ ] **Step 1: Write failing result-contract tests**

Add tests that construct `DetectorResult[None]`, verify its existing detector,
observable, DEM, and fidelity members, and assert that unavailable values fail:

```python
def test_detector_result_rejects_unavailable_measurements_and_return_values():
    result = DetectorResult[None](
        _detector_error_model=MagicMock(),
        _fidelity_min=0.5,
        _fidelity_max=0.9,
        _detectors=[[True]],
        _observables=[[False]],
    )

    with pytest.raises(ValueError, match="Raw measurements are unavailable"):
        _ = result.measurements
    with pytest.raises(ValueError, match="return values.*unavailable"):
        _ = result.return_values
```

Under the existing `TYPE_CHECKING` block, add structural assignments for both
concrete classes:

```python
measurement_result: SimulatorResult[Any] = cast(Result[Any], task.run())
detector_result: SimulatorResult[Any] = DetectorResult[Any](
    _detector_error_model=MagicMock(),
    _fidelity_min=0.5,
    _fidelity_max=0.9,
    _detectors=[[True]],
    _observables=[[False]],
)
```

Add import tests confirming `SimulatorResult` is exported from
`bloqade.gemini.device` and `bloqade.gemini`.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
MPLCONFIGDIR=/tmp/matplotlib UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  python/tests/test_device.py -k "detector_result_rejects or simulator_result_exports"
```

Expected: failures because `DetectorResult` is not generic, lacks the two
properties, and `SimulatorResult` is not exported.

- [ ] **Step 3: Implement the protocol and detector-only error properties**

In `_task_runtime.py`, replace the unused `Literal`/`overload` imports as later
tasks permit and add `Protocol`, `Sequence`, and a covariant return type:

```python
RetType = TypeVar("RetType")
RetType_co = TypeVar("RetType_co", covariant=True)


class SimulatorResult(Protocol[RetType_co]):
    def fidelity_bounds(self) -> tuple[float, float]: ...

    @property
    def detector_error_model(self) -> DetectorErrorModel: ...

    @property
    def return_values(self) -> Sequence[RetType_co]: ...

    @property
    def detectors(self) -> Sequence[Sequence[bool]]: ...

    @property
    def measurements(self) -> Sequence[Sequence[bool]]: ...

    @property
    def observables(self) -> Sequence[Sequence[bool]]: ...
```

Make `DetectorResult` generic with a covariant phantom return type and add
ordinary properties:

```python
@property
def measurements(self) -> Sequence[Sequence[bool]]:
    raise ValueError("Raw measurements are unavailable for detector-only results")

@property
def return_values(self) -> Sequence[RetType_co]:
    raise ValueError("Kernel return values are unavailable for detector-only results")
```

Do not alter `Result`. Export `SimulatorResult` from both public namespaces.
Do not add `@runtime_checkable`; conformance is static and unavailable
properties must not be evaluated by runtime protocol inspection.

- [ ] **Step 4: Run the focused tests and Pyright**

Run:

```bash
MPLCONFIGDIR=/tmp/matplotlib UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  python/tests/test_device.py -k "detector_result_rejects or simulator_result_exports"
UV_CACHE_DIR=/tmp/uv-cache uv run pyright \
  python/bloqade/gemini/device/_task_runtime.py \
  python/bloqade/gemini/device/__init__.py \
  python/bloqade/gemini/__init__.py \
  python/tests/test_device.py
```

Expected: focused tests pass and Pyright reports zero errors.

- [ ] **Step 5: Commit the result contract**

```bash
git add python/bloqade/gemini/device/_task_runtime.py \
  python/bloqade/gemini/device/__init__.py \
  python/bloqade/gemini/__init__.py \
  python/tests/test_device.py
git commit -m "feat!: add common simulator result protocol"
```

### Task 2: Move Detector Sampling into the Tsim Backend Configuration

**Files:**
- Modify: `python/bloqade/gemini/device/simulator_backend.py:30-44,80-146,205-238,285-360`
- Test: `python/tests/gemini/test_simulator_backend.py`
- Test: `python/tests/gemini/test_optional_tsim_import.py`

- [ ] **Step 1: Rewrite backend tests for the new contract**

Change Tsim detector tests to construct the backend in detector mode:

```python
backend = TsimSimulatorBackend(run_detectors=True)
backend._tsim_circuit = MagicMock(return_value=circuit)
sample = backend.sample(_physical_kernel, shots=1, seed=0)
```

Add an explicit default-field assertion:

```python
assert TsimSimulatorBackend().run_detectors is False
assert TsimSimulatorBackend(run_detectors=True).run_detectors is True
```

Change the CliffT native-detector test into a measurement-only contract test:

```python
sample = backend.sample(_physical_kernel, shots=2, seed=None)
assert sample.measurements is not None
assert sample.detectors is None
assert sample.observables is None
```

Remove `run_detectors` from every PyQrack test call. Add signature assertions
using `inspect.signature` to ensure `sample()` on the abstract and all concrete
backends has no `run_detectors` parameter.

- [ ] **Step 2: Run backend tests and verify RED**

Run:

```bash
MPLCONFIGDIR=/tmp/matplotlib UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  python/tests/gemini/test_simulator_backend.py \
  python/tests/gemini/test_optional_tsim_import.py
```

Expected: failures show that Tsim does not yet have the field and backend sample
signatures/CliffT payload behavior still follow the old flag.

- [ ] **Step 3: Implement the backend contract**

Update the abstract signature to:

```python
def sample(
    self,
    physical_squin_kernel: ir.Method,
    *,
    shots: int,
    seed: int | None = None,
) -> BackendSample:
```

Add the Tsim field before its non-init cache field:

```python
run_detectors: bool = False
```

In Tsim `sample()`, use `if self.run_detectors:`. Remove the argument from
CliffT and PyQrack signatures. Delete CliffT's conditional detector branch and
always normalize `sample_result.measurements` into a measurement-only
`BackendSample`. Preserve all seed behavior and private Tsim composition.

- [ ] **Step 4: Run backend tests and linters**

Run:

```bash
MPLCONFIGDIR=/tmp/matplotlib UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  python/tests/gemini/test_simulator_backend.py \
  python/tests/gemini/test_optional_tsim_import.py
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check \
  python/bloqade/gemini/device/simulator_backend.py \
  python/tests/gemini/test_simulator_backend.py
```

Expected: tests and Ruff pass.

- [ ] **Step 5: Commit the backend change**

```bash
git add python/bloqade/gemini/device/simulator_backend.py \
  python/tests/gemini/test_simulator_backend.py \
  python/tests/gemini/test_optional_tsim_import.py
git commit -m "feat!: configure detector sampling on Tsim backend"
```

### Task 3: Route Exact Backend Payload Shapes in the Shared Task Runtime

**Files:**
- Modify: `python/bloqade/gemini/device/_task_runtime.py:190-360`
- Test: `python/tests/test_device.py:475-675`
- Test: `python/tests/gemini/test_physical_simulator.py:75-105`

- [ ] **Step 1: Rewrite task-runtime tests around payload shape**

Update mock assertions so backend calls are exactly:

```python
backend.sample.assert_called_once_with("noisy-kernel", shots=1, seed=None)
```

Test measurement-only payloads return unchanged `Result`. Test detector plus
observable payloads return `DetectorResult`. Parametrize invalid shapes:

```python
@pytest.mark.parametrize(
    "sample",
    [
        BackendSample(),
        BackendSample(detectors=np.array([[True]])),
        BackendSample(observables=np.array([[False]])),
        BackendSample(
            measurements=np.array([[True]]),
            detectors=np.array([[True]]),
            observables=np.array([[False]]),
        ),
    ],
)
def test_run_rejects_invalid_backend_payload_shapes(sample):
    task = _mock_task()
    task.simulator.backend.sample.return_value = sample

    with pytest.raises(ValueError, match="either measurement samples or both"):
        GeminiLogicalSimulatorTask.run(task, shots=1)
```

Retain dimensional and shot-count normalization tests for both valid forms.
Delete the old measurement fallback test for detector mode: measurement payloads
now directly create `Result`, whose properties already perform postprocessing.

- [ ] **Step 2: Run task-runtime tests and verify RED**

Run:

```bash
MPLCONFIGDIR=/tmp/matplotlib UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  python/tests/test_device.py -k "backend or payload or task_run" \
  python/tests/gemini/test_physical_simulator.py -k "task_run_routes"
```

Expected: failures because task methods still pass and branch on
`run_detectors`, and mixed payloads are not rejected.

- [ ] **Step 3: Simplify synchronous routing**

Replace the overload set with one method:

```python
def run(
    self,
    shots: int = 1,
    with_noise: bool = True,
    *,
    seed: int | None = None,
) -> SimulatorResult[RetType]:
```

Call `sample(physical_kernel, shots=shots, seed=seed)`. Validate these booleans:

```python
has_measurements = sample.measurements is not None
has_detectors = sample.detectors is not None
has_observables = sample.observables is not None
```

- If `has_measurements` and neither direct array is present, normalize and
  construct existing `Result`.
- If no measurements and both direct arrays are present, normalize and
  construct `DetectorResult[RetType]`.
- Otherwise raise `ValueError` describing the two accepted payload shapes.

Delete `_detector_result`, including its measurement fallback.

- [ ] **Step 4: Simplify asynchronous routing**

Replace the overloads and branch with:

```python
def run_async(
    self,
    shots: int = 1,
    with_noise: bool = True,
    *,
    seed: int | None = None,
) -> Future[SimulatorResult[RetType]]:
    _validate_seed(seed)
    return self._thread_pool_executor.submit(
        self.run, shots, with_noise, seed=seed
    )
```

Preserve validation before executor submission.

- [ ] **Step 5: Run focused tests and Ruff**

Run:

```bash
MPLCONFIGDIR=/tmp/matplotlib UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  python/tests/test_device.py -k "backend or payload or task_run" \
  python/tests/gemini/test_physical_simulator.py -k "task_run_routes"
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check \
  python/bloqade/gemini/device/_task_runtime.py \
  python/tests/test_device.py \
  python/tests/gemini/test_physical_simulator.py
```

Expected: focused tests and Ruff pass. Defer Pyright until Task 4 has migrated
the logical and physical simulator forwarders and all static type assertions to
the new task signature.

- [ ] **Step 6: Commit task routing**

```bash
git add python/bloqade/gemini/device/_task_runtime.py \
  python/tests/test_device.py \
  python/tests/gemini/test_physical_simulator.py
git commit -m "refactor!: select simulator results from backend payloads"
```

### Task 4: Simplify Logical and Physical Simulator Public APIs

**Files:**
- Modify: `python/bloqade/gemini/device/simulator.py:1-205`
- Modify: `python/bloqade/gemini/device/physical_simulator.py:1-350`
- Test: `python/tests/test_device.py:260-315,680-715`
- Test: `python/tests/gemini/test_physical_simulator.py:280-325,500-535`

- [ ] **Step 1: Rewrite forwarding and static typing tests**

Remove boolean parametrization from seed-forwarding tests. Assert simulator
forwarders call tasks without a detector flag:

```python
task.run.assert_called_once_with(3, False, seed=seed)
task.run_async.assert_called_once_with(3, False, seed=0)
```

Replace overload type checks with:

```python
assert_type(task.run(), SimulatorResult[Any])
assert_type(task.run_async(), Future[SimulatorResult[Any]])
assert_type(simulator.run(kernel), SimulatorResult[Any])
assert_type(
    simulator.run_async(kernel),
    Future[SimulatorResult[Any]],
)
```

Add `inspect.signature` assertions that task, logical simulator, and physical
simulator run methods do not expose `run_detectors`.

- [ ] **Step 2: Run forwarding tests and verify RED**

Run:

```bash
MPLCONFIGDIR=/tmp/matplotlib UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  python/tests/test_device.py -k "forwards_seed or public_run_signature" \
  python/tests/gemini/test_physical_simulator.py -k "forwards_seed or run_signature"
```

Expected: old forwarding calls and overload signatures fail the assertions.

- [ ] **Step 3: Remove logical simulator overloads and flags**

Delete `Literal` and `overload` imports and overload blocks from `simulator.py`.
Annotate `run()` as `SimulatorResult[RetType]` and `run_async()` as
`Future[SimulatorResult[RetType]]`. Both directly forward `seed` without a
conditional branch.

- [ ] **Step 4: Remove physical simulator overloads and flags**

Apply the same change to `physical_simulator.py`. Preserve kernel validation,
compilation, noise selection, and all non-run public APIs.

- [ ] **Step 5: Run logical/physical tests and Pyright**

Run:

```bash
MPLCONFIGDIR=/tmp/matplotlib UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  python/tests/test_device.py \
  python/tests/gemini/test_physical_simulator.py
UV_CACHE_DIR=/tmp/uv-cache uv run pyright \
  python/bloqade/gemini/device/simulator.py \
  python/bloqade/gemini/device/physical_simulator.py \
  python/tests/test_device.py \
  python/tests/gemini/test_physical_simulator.py
```

Expected: both suites pass and Pyright reports zero errors.

- [ ] **Step 6: Commit the public API simplification**

```bash
git add python/bloqade/gemini/device/simulator.py \
  python/bloqade/gemini/device/physical_simulator.py \
  python/tests/test_device.py \
  python/tests/gemini/test_physical_simulator.py
git commit -m "feat!: remove per-run detector sampling flags"
```

### Task 5: Migrate Integration Workflows

**Files:**
- Modify: `python/bloqade/gemini/decoding/experiments.py:8-18,120-135,350-370`
- Modify: `python/tests/gemini/decoding/test_experiment.py:440-460`
- Modify: `python/tests/test_device.py:100-190,245-270,360-390`
- Modify: `python/tests/gemini/test_physical_simulator.py:250-280`

- [ ] **Step 1: Rewrite integration tests for backend configuration**

For native Tsim detector-result tests, construct:

```python
simulator = GeminiLogicalSimulator(
    backend=TsimSimulatorBackend(run_detectors=True)
)
result = simulator.run(kernel, shots=10, with_noise=False)
assert isinstance(result, DetectorResult)
```

For PyQrack tests, call `run()` without a flag and assert `Result`; detectors and
observables remain available through atom postprocessing. Apply the same rules
to physical simulator integration tests.

Change the experiment mock expectation to:

```python
task.run_async.assert_called_once_with(100)
```

- [ ] **Step 2: Run integration tests and verify RED**

Run:

```bash
MPLCONFIGDIR=/tmp/matplotlib UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  python/tests/test_device.py \
  python/tests/gemini/test_physical_simulator.py \
  python/tests/gemini/decoding/test_experiment.py
```

Expected: stale `run_detectors` call sites or expectations fail.

- [ ] **Step 3: Update decoding types and calls**

In `experiments.py`, replace the concrete result union with
`SimulatorResult[_LogicalTomographyReturn]` where the helper only consumes
detectors and observables. Change `task.run_async(num_shots,
run_detectors=True)` to `task.run_async(num_shots)`. Do not force the experiment
to use a particular backend; its task's configured backend controls the payload.

- [ ] **Step 4: Update all remaining tracked callers and assert no stale calls**

Migrate remaining tracked tests to backend construction or measurement-backed
results as appropriate. Run:

```bash
git grep -n "run_detectors" -- '*.py' '*.ipynb'
```

Expected: matches only the intentional public field and tests/specifications
that refer to that field; no method signature or call passes `run_detectors`.
Do not edit untracked scratch demos or the user's modified notebook.

- [ ] **Step 5: Run integration tests and Ruff**

Run:

```bash
MPLCONFIGDIR=/tmp/matplotlib UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  python/tests/test_device.py \
  python/tests/gemini/test_physical_simulator.py \
  python/tests/gemini/decoding/test_experiment.py
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check \
  python/bloqade/gemini \
  python/tests/test_device.py \
  python/tests/gemini/test_physical_simulator.py \
  python/tests/gemini/test_simulator_backend.py \
  python/tests/gemini/decoding/test_experiment.py
```

Expected: integration tests and Ruff pass.

- [ ] **Step 6: Commit integration migration**

```bash
git add python/bloqade/gemini/decoding/experiments.py \
  python/tests/gemini/decoding/test_experiment.py \
  python/tests/test_device.py \
  python/tests/gemini/test_physical_simulator.py
git commit -m "refactor: migrate simulator detector sampling configuration"
```

### Task 6: Final Formatting, Static Analysis, and Regression Verification

**Files:**
- Verify all files modified in Tasks 1-5

- [ ] **Step 1: Format changed Python files**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run black \
  python/bloqade/gemini/device/_task_runtime.py \
  python/bloqade/gemini/device/simulator_backend.py \
  python/bloqade/gemini/device/simulator.py \
  python/bloqade/gemini/device/physical_simulator.py \
  python/bloqade/gemini/device/__init__.py \
  python/bloqade/gemini/__init__.py \
  python/bloqade/gemini/decoding/experiments.py \
  python/tests/gemini/test_simulator_backend.py \
  python/tests/gemini/test_physical_simulator.py \
  python/tests/gemini/decoding/test_experiment.py \
  python/tests/test_device.py
UV_CACHE_DIR=/tmp/uv-cache uv run isort \
  python/bloqade/gemini/device/_task_runtime.py \
  python/bloqade/gemini/device/simulator_backend.py \
  python/bloqade/gemini/device/simulator.py \
  python/bloqade/gemini/device/physical_simulator.py \
  python/bloqade/gemini/device/__init__.py \
  python/bloqade/gemini/__init__.py \
  python/bloqade/gemini/decoding/experiments.py \
  python/tests/gemini/test_simulator_backend.py \
  python/tests/gemini/test_physical_simulator.py \
  python/tests/gemini/decoding/test_experiment.py \
  python/tests/test_device.py
```

Expected: files are formatted without touching unrelated paths.

- [ ] **Step 2: Run focused simulator and decoding tests**

Run:

```bash
MPLCONFIGDIR=/tmp/matplotlib UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  python/tests/gemini/test_simulator_backend.py \
  python/tests/gemini/test_optional_tsim_import.py \
  python/tests/gemini/test_physical_simulator.py \
  python/tests/gemini/decoding/test_experiment.py \
  python/tests/test_device.py
```

Expected: all selected tests pass.

- [ ] **Step 3: Run Ruff, Pyright, and diff hygiene checks**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check python/bloqade/gemini python/tests
UV_CACHE_DIR=/tmp/uv-cache uv run pyright python
git diff --check
```

Expected: Ruff and Pyright report no errors and `git diff --check` is silent.

- [ ] **Step 4: Run the complete Python test suite**

Run:

```bash
MPLCONFIGDIR=/tmp/matplotlib UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q python/tests
```

Expected: the full suite passes, aside from explicitly skipped tests and known
non-failing warnings.

- [ ] **Step 5: Review the final diff and workspace boundaries**

Run:

```bash
git status --short
git diff --stat 54d49291..HEAD
git diff 54d49291..HEAD -- \
  python/bloqade/gemini \
  python/tests/test_device.py \
  python/tests/gemini
```

Expected: only planned production/test files and the design/plan documents are
part of this change. `demo/simulator_device_demo.ipynb` and unrelated untracked
files remain untouched.

- [ ] **Step 6: Commit any formatter-only residue**

If formatting changed planned files after their task commits:

```bash
git add python/bloqade/gemini python/tests
git commit -m "style: format simulator sampling refactor"
```

Skip this commit when there is no formatter residue.
