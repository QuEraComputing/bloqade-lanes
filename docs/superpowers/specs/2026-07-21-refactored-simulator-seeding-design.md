# Refactored Simulator Seeding Design

## Goal

Port the per-call simulator seed behavior from `origin/main` into the
backend-based simulator architecture, while extending reproducible sampling to
the physical simulator and PyQrack backend.

## Validation boundary

`_validate_seed` remains in `python/bloqade/gemini/device/_task_runtime.py`.
`_SimulatorTaskBase.run()` and `_SimulatorTaskBase.run_async()` are the sole
validation boundary for per-call seeds. A valid seed is either `None` or a
non-boolean Python `int` in the range `0 <= seed < 2**63`.

Invalid values raise `ValueError` synchronously. `run()` validates before DEM
construction or backend sampling, and `run_async()` validates before submitting
work to its executor.

Simulator convenience methods forward the seed to their task. Backends assume
that task-mediated calls already received a valid per-call seed. Consequently,
direct calls to `backend.sample()` do not promise seed validation, and an
invalid seed supplied to a simulator convenience method can compile the task
before `_SimulatorTaskBase` rejects it. This deliberately differs from
`origin/main`, which validates in additional simulator and CliffT-specific
locations.

`simulator_backend.py` must not import `_task_runtime.py`: `_task_runtime.py`
already imports backend types, so the reverse dependency would create a module
cycle.

## Seed propagation

Both `GeminiLogicalSimulator` and `GeminiPhysicalSimulator` accept a keyword-only
`seed` in synchronous and asynchronous execution APIs and forward it unchanged.
`_SimulatorTaskBase` validates and forwards it to
`AbstractSimulatorBackend.sample()`.

The existing Tsim sampler properties on `_SimulatorTaskBase` remain
parameterless cached properties. Backend sampling owns seeded sampler
construction, so those properties do not accept seed arguments.

## Backend behavior

### Tsim

The Tsim backend forwards the per-call seed to the appropriate sampler compiler
for all four sampling paths:

- Clifford measurements through Stim;
- Clifford detectors through Stim plus the measurement-to-detector converter;
- non-Clifford measurements through Tsim;
- non-Clifford detectors through Tsim.

`seed=0` must be preserved rather than treated as missing.

### CliffT

A non-`None` per-call seed overrides `CliffTSimulatorBackend.seed`. When the
per-call seed is `None`, the backend-level seed remains the fallback. If both
are `None`, the `seed` keyword is omitted from `clifft.sample()`.

### PyQrack

PyQrack has two random sources: the NumPy generator used for Squin noise and
loss, and the native Qrack generator used for quantum measurement. A supplied
per-call seed creates a fresh NumPy generator for that sampling request. A
missing per-call seed preserves the backend's existing persistent
`_rng_state`, including the existing advancing-stream behavior of
`PyQrackSimulatorBackend(seed=...)`.

Each shot creates a fresh native Qrack simulator. Immediately after that native
simulator is created, the interpreter makes a fresh deterministic integer draw
from its NumPy generator and calls `sim_reg.seed(...)`. Thus shots do not all
deliberately restart the native measurement RNG from one fixed seed, while
repeated calls with the same explicit per-call seed reproduce the complete
fixed-size batch. Independent random draws are not specified to be
mathematically collision-free.

## Tests

Applicable tests from the `origin/main` seed change are adapted to the new
ownership boundaries instead of copied literally:

- valid endpoints, invalid values, explicit `None`, and `seed=0` at the shared
  task runtime;
- synchronous and asynchronous task forwarding;
- logical simulator synchronous and asynchronous forwarding;
- seeded Clifford/non-Clifford measurement and detector paths;
- fixed-batch reproducibility;
- CliffT per-call override and backend-seed fallback.

Refactor-specific coverage adds:

- physical simulator synchronous and asynchronous forwarding;
- an import smoke test that catches simulator/backend cycles;
- PyQrack fixed-batch reproduction for native measurement randomness;
- PyQrack use of fresh derived native-seed draws across shots;
- preservation of the persistent unseeded backend stream after an isolated
  per-call seeded request.

Invalid-seed tests also verify that `run()` performs no DEM/backend work and
that `run_async()` submits no executor work before raising.

Tests from `origin/main` that specifically assert task-owned cached sampler
selection or the deleted `_CliffTSimulatorTask` implementation are translated
to backend-level assertions. Direct backend invalid-seed tests are excluded
because validation is intentionally owned only by `_SimulatorTaskBase`.
