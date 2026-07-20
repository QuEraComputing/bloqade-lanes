# Per-call tsim sampling seed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional per-call sampling seed to Gemini logical simulator task and convenience APIs.

**Architecture:** Validate the seed once at the logical-task boundary, then select cached samplers for unseeded tsim calls and fresh compiled samplers for seeded calls. Forward the same keyword through asynchronous, simulator convenience, and CliffT override paths; a per-call CliffT seed overrides its existing task-level seed.

**Tech Stack:** Python 3.10+, pytest, Stim, tsim, CliffT test doubles, Kirin.

---

### Task 1: Specify seeded task sampling behavior with tests

**Files:**
- Modify: `python/tests/test_device.py:316-383`
- Modify: `python/tests/gemini/decoding/test_tasks.py:19-76`

- [ ] **Step 1: Write failing logical-task tests**

Add mocked tests that call `GeminiLogicalSimulatorTask.run(..., seed=0)` for:

```python
def test_run_clifford_compiles_seeded_stim_sampler():
    task = _mock_task(is_clifford=True)
    GeminiLogicalSimulatorTask.run(task, seed=0)
    task.tsim_circuit.stim_circuit.compile_sampler.assert_called_once_with(seed=0)

def test_run_non_clifford_compiles_fresh_seeded_sampler():
    task = _mock_task(is_clifford=False)
    GeminiLogicalSimulatorTask.run(task, seed=123)
    task.tsim_circuit.compile_sampler.assert_called_once_with(seed=123)
    task.measurement_sampler.sample.assert_not_called()
```

Cover detector equivalents, noisy/noiseless circuit selection, and no-seed
cached sampler behavior. Add an explicit validation matrix that accepts `0`
and `2**63 - 1` and rejects `True`, negative values, `2**63`, and a
non-integer. Run the task's seeded `run_async(...).result()` path, then add a
real fixed-batch non-Clifford tsim test showing two independent same-seed calls
produce the same samples and that a seeded call does not change a following
unseeded call's use of its cached sampler.

- [ ] **Step 2: Write failing CliffT tests**

Extend `_task_with_cached_programs` tests to show that `seed=` overrides the
task field, `seed=None` falls back to it, the same validation matrix fails
before sampling, and `run_async(..., seed=...).result()` forwards to `run`.

- [ ] **Step 3: Run focused tests and confirm they fail**

Run: `uv run pytest python/tests/test_device.py python/tests/gemini/decoding/test_tasks.py -q`

Expected: failures reporting that `run` and/or `_sample_clifft` do not accept `seed`.

### Task 2: Implement seed validation and task propagation

**Files:**
- Modify: `python/bloqade/gemini/device/simulator.py:305-457`
- Modify: `python/bloqade/gemini/decoding/tasks.py:43-198`

- [ ] **Step 1: Add a shared seed validator in `simulator.py`**

Implement a small helper accepting `None` or non-boolean `int` in `[0, 2**63)` and raising `ValueError` otherwise. Preserve `seed=0`.

- [ ] **Step 2: Add seeded sampler selection to `GeminiLogicalSimulatorTask`**

Add `seed` to all task `run` / `run_async` overloads and implementations. For a provided seed, compile a fresh sampler with `seed=seed`: Stim for Clifford circuits, tsim measurement or detector sampler for non-Clifford circuits. For `None`, retain the existing cached/non-cached code paths. Pass seed to `_run_detectors` and executor calls.

- [ ] **Step 3: Update `_CliffTSimulatorTask`**

Add the same `seed` keyword to its sampling, `run`, and `run_async` methods. `_sample_clifft` uses the per-call seed when non-`None`; otherwise it retains `self.seed` fallback. Validate with the shared helper.

- [ ] **Step 4: Run focused tests and confirm they pass**

Run: `uv run pytest python/tests/test_device.py python/tests/gemini/decoding/test_tasks.py -q`

Expected: PASS.

### Task 3: Add simulator convenience forwarding and verification

**Files:**
- Modify: `python/bloqade/gemini/device/simulator.py:540-641`
- Modify: `python/tests/test_device.py:316-383`

- [ ] **Step 1: Write failing forwarding tests**

Use a mock task returned by `GeminiLogicalSimulator.task` to verify:

```python
sim.run(main, shots=2, seed=123)
task.run.assert_called_once_with(2, True, run_detectors=False, seed=123)

sim.run_async(main, shots=2, run_detectors=True, seed=123)
task.run_async.assert_called_once_with(2, True, run_detectors=True, seed=123)
```

Also use a task double returning an already-completed future so the test calls
the simulator's `run_async(..., seed=123).result()` successfully for both
measurement and detector branches.

- [ ] **Step 2: Run forwarding tests and confirm they fail**

Run: `uv run pytest python/tests/test_device.py -k seed -q`

Expected: failure because the simulator methods do not accept or forward `seed`.

- [ ] **Step 3: Add optional seed to simulator overloads and implementations**

Add `seed: int | None = None` as a keyword-only argument and document it. Forward it unchanged to each newly created task's `run` or `run_async` call, including the detector branch.

- [ ] **Step 4: Run all focused tests and static checks**

Run:
`uv run pytest python/tests/test_device.py python/tests/gemini/decoding/test_tasks.py -q`

`uv run pyright python/bloqade/gemini/device/simulator.py python/bloqade/gemini/decoding/tasks.py`

Expected: PASS.

- [ ] **Step 5: Run the full Python suite**

Run: `just test-python`

Expected: PASS.

- [ ] **Step 6: Commit the implementation**

```bash
git add python/bloqade/gemini/device/simulator.py \
    python/bloqade/gemini/decoding/tasks.py \
    python/tests/test_device.py \
    python/tests/gemini/decoding/test_tasks.py
git commit -m "feat(python): add per-call simulator sampling seeds"
```
