# Private Tsim Backend Field Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the embedded Tsim dependency on CliffT and PyQrack backends to the private-by-convention `_tsim_backend` field without changing behavior.

**Architecture:** Keep ordinary dataclass constructor injection, default construction, and hidden repr behavior. Update internal delegation and repository tests to use the underscored field, with no public compatibility alias.

**Tech Stack:** Python dataclasses, pytest, Ruff, Black, Pyright.

---

### Task 1: Add the Private-Field Contract

**Files:**
- Modify: `python/tests/gemini/test_simulator_backend.py`

- [ ] Add this private-field contract test:

```python
@pytest.mark.parametrize(
    "backend_type", [CliffTSimulatorBackend, PyQrackSimulatorBackend]
)
def test_composite_backend_exposes_only_private_tsim_backend(backend_type):
    backend = backend_type()

    assert isinstance(backend._tsim_backend, TsimSimulatorBackend)
    assert not hasattr(backend, "tsim_backend")
```

- [ ] Run and verify RED:

```bash
uv run pytest -q \
  python/tests/gemini/test_simulator_backend.py::test_composite_backend_exposes_only_private_tsim_backend
```

Expected: both cases fail because `_tsim_backend` does not yet exist.

### Task 2: Rename the Production Field and Test Call Sites

**Files:**
- Modify: `python/bloqade/gemini/device/simulator_backend.py`
- Modify: `python/tests/gemini/test_simulator_backend.py`
- Modify: `python/tests/gemini/test_optional_tsim_import.py`

- [ ] Rename both dataclass fields to `_tsim_backend`, retaining
  `default_factory=TsimSimulatorBackend` and `repr=False`.
- [ ] Change all internal accesses to `self._tsim_backend`.
- [ ] Change test constructor injection keywords and direct attribute patches to
  `_tsim_backend`; keep local fixture variable names unchanged where helpful.
- [ ] Run and verify GREEN:

```bash
uv run pytest -q \
  python/tests/gemini/test_simulator_backend.py \
  python/tests/gemini/test_optional_tsim_import.py
```

Expected: all tests pass.

### Task 3: Verify

- [ ] Run Black and isort:

```bash
uv run black \
  python/bloqade/gemini/device/simulator_backend.py \
  python/tests/gemini/test_simulator_backend.py \
  python/tests/gemini/test_optional_tsim_import.py
uv run isort \
  python/bloqade/gemini/device/simulator_backend.py \
  python/tests/gemini/test_simulator_backend.py \
  python/tests/gemini/test_optional_tsim_import.py
```

- [ ] Run Ruff and Pyright:

```bash
uv run ruff check \
  python/bloqade/gemini/device/simulator_backend.py \
  python/tests/gemini/test_simulator_backend.py \
  python/tests/gemini/test_optional_tsim_import.py
uv run pyright \
  python/bloqade/gemini/device/simulator_backend.py \
  python/tests/gemini/test_simulator_backend.py \
  python/tests/gemini/test_optional_tsim_import.py
```

Expected: all commands exit successfully with no errors.
- [ ] Run:

```bash
uv run pytest -q \
  python/tests/gemini/test_simulator_backend.py \
  python/tests/gemini/test_optional_tsim_import.py \
  python/tests/test_device.py \
  python/tests/gemini/test_physical_simulator.py
```

- [ ] Search tracked Python sources for stale public field accesses or constructor
  keywords:

```bash
git grep -n -E '\.tsim_backend|[^_]tsim_backend=' -- 'python/**/*.py'
```

Expected: no matches. Local variables named `tsim_backend` are permitted.
- [ ] Run `git diff --check` and inspect scope, preserving all unrelated files.
