# Private Backend Detector Error Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make detector-error-model generation a private Gemini backend protocol method while preserving public task/result APIs.

**Architecture:** Rename the abstract method and all backend overrides to `_detector_error_model`, update the task runtime to call it, and migrate internal tests/mocks. Leave the native Tsim circuit method and public task/result properties unchanged.

**Tech Stack:** Python ABCs/dataclasses, pytest, Ruff, Black, Pyright.

---

### Task 1: Add the Backend Privacy Contract

**Files:**
- Modify: `python/tests/gemini/test_simulator_backend.py`

- [ ] Add this privacy contract:

```python
@pytest.mark.parametrize(
    "backend_type",
    [TsimSimulatorBackend, CliffTSimulatorBackend, PyQrackSimulatorBackend],
)
def test_backend_exposes_only_private_detector_error_model(backend_type):
    backend = backend_type()

    assert callable(backend._detector_error_model)
    assert not hasattr(backend, "detector_error_model")
```

- [ ] Change the expected abstract operation name in
  `test_backend_contract_has_exactly_two_abstract_operations` from
  `"detector_error_model"` to `"_detector_error_model"`.
- [ ] Run RED:

```bash
uv run pytest -q \
  python/tests/gemini/test_simulator_backend.py::test_backend_exposes_only_private_detector_error_model \
  python/tests/gemini/test_simulator_backend.py::test_backend_contract_has_exactly_two_abstract_operations
```

Expected: four failures—three because `_detector_error_model` is absent and one
because the abstract method is still public.

### Task 2: Rename the Protocol and Call Sites

**Files:**
- Modify: `python/bloqade/gemini/device/simulator_backend.py`
- Modify: `python/bloqade/gemini/device/_task_runtime.py`
- Modify: `python/tests/gemini/test_simulator_backend.py`
- Modify: `python/tests/gemini/test_optional_tsim_import.py`
- Modify: `python/tests/gemini/test_physical_simulator.py`
- Modify: `python/tests/test_device.py`

- [ ] Rename the abstract method and all backend implementations to
  `_detector_error_model`.

```python
@abc.abstractmethod
def _detector_error_model(
    self, physical_squin_kernel: ir.Method
) -> DetectorErrorModel:
    """Build the detector error model for a physical SQuIn kernel."""
```

- [ ] Update composite backends to delegate to
  `self._tsim_backend._detector_error_model(...)`.
- [ ] Update `_SimulatorTaskBase.detector_error_model` to call the private
  backend method while keeping the property public.
- [ ] Update backend/task mocks, assertions, optional-import patches, and direct
  backend calls in tests.
- [ ] Add no compatibility alias and do not rename the native Tsim circuit
  method.
- [ ] Run GREEN:

```bash
uv run pytest -q \
  python/tests/gemini/test_simulator_backend.py \
  python/tests/gemini/test_optional_tsim_import.py \
  python/tests/gemini/test_physical_simulator.py \
  python/tests/test_device.py
```

Expected: all selected tests pass.

### Task 3: Verify

- [ ] Run formatting:

```bash
uv run black \
  python/bloqade/gemini/device/simulator_backend.py \
  python/bloqade/gemini/device/_task_runtime.py \
  python/tests/gemini/test_simulator_backend.py \
  python/tests/gemini/test_optional_tsim_import.py \
  python/tests/gemini/test_physical_simulator.py \
  python/tests/test_device.py
uv run isort \
  python/bloqade/gemini/device/simulator_backend.py \
  python/bloqade/gemini/device/_task_runtime.py \
  python/tests/gemini/test_simulator_backend.py \
  python/tests/gemini/test_optional_tsim_import.py \
  python/tests/gemini/test_physical_simulator.py \
  python/tests/test_device.py
```

- [ ] Run static checks with the same six file paths:

```bash
uv run ruff check \
  python/bloqade/gemini/device/simulator_backend.py \
  python/bloqade/gemini/device/_task_runtime.py \
  python/tests/gemini/test_simulator_backend.py \
  python/tests/gemini/test_optional_tsim_import.py \
  python/tests/gemini/test_physical_simulator.py \
  python/tests/test_device.py
uv run pyright \
  python/bloqade/gemini/device/simulator_backend.py \
  python/bloqade/gemini/device/_task_runtime.py \
  python/tests/gemini/test_simulator_backend.py \
  python/tests/gemini/test_optional_tsim_import.py \
  python/tests/gemini/test_physical_simulator.py \
  python/tests/test_device.py
```

Expected: no errors.

- [ ] Search for stale backend-protocol calls:

```bash
git grep -n -E '(backend|_tsim_backend)\.detector_error_model' -- 'python/**/*.py'
```

Expected: no matches. Public `task.detector_error_model`, result properties,
decoding code, and the native Tsim circuit method are intentionally permitted.

- [ ] Run the full suite:

```bash
uv run pytest -q python/tests
```

Expected: all tests pass, with configured skips and existing warnings allowed.

- [ ] Run `git diff --check`, inspect `git status --short`, and dispatch a final
  independent whole-diff review.

No commits are made by this plan because the shared feature worktree contains
unrelated user-owned changes; hand off the verified changes unstaged.
