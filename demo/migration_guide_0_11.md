# Migration guide to v0.11

Version 0.11 refactors the Gemini simulator API around explicit simulator
backends. Configure compilation and sampling when constructing a simulator,
compile a kernel with `task()`, then execute the resulting task with `run()` or
`run_async()`.

`PhysicalSimulator` remains an alias for `GeminiPhysicalSimulator`.

Each v0.11 example imports the new public APIs it introduces. Variables such
as `logical_kernel`, `physical_kernel`, `noise_model`, `arch_spec`, and
`placement_strategy` represent the corresponding objects already defined by
your application.

## Core API change: configure the simulator; give `task()` a prepared kernel

Previously, task creation mixed the kernel with backend, placement, and
annotation options. In v0.11, both simulator types own their stable
configuration and `task()` accepts only a kernel:

- `GeminiLogicalSimulator` owns its backend configuration. Its task must be a
  logical SQuIN `ir.Method`; CUDA-Q conversion and detector/observable
  annotations happen before `task()`.
- `GeminiPhysicalSimulator` owns backend, architecture, noise, and placement
  configuration. Its task must be a prepared physical SQuIN method containing
  its terminal measurement and any desired annotations.

The remaining sections show the concrete old-to-new replacements.

## Choose a backend object instead of a backend name

Simulator constructors no longer accept backend-name strings such as `"clifft"`.
Pass an explicit backend object instead.

```python
# v0.10
simulator = GeminiLogicalSimulator(backend="clifft", seed=10)

# v0.11
from bloqade.gemini import CliffTSimulatorBackend, GeminiLogicalSimulator

simulator = GeminiLogicalSimulator(backend=CliffTSimulatorBackend(seed=10))
```

Use `TsimSimulatorBackend()` for the default Tsim behavior explicitly when you
need to configure it:

```python
from bloqade.gemini import GeminiLogicalSimulator, TsimSimulatorBackend

simulator = GeminiLogicalSimulator(backend=TsimSimulatorBackend())
```

The same pattern applies to `GeminiPhysicalSimulator` / `PhysicalSimulator`.

## Move simulator configuration out of `task()`

`task()` now accepts only the kernel. Configure physical compilation options on
the simulator itself.

```python
# v0.10
simulator = PhysicalSimulator(noise_model=noise_model, arch_spec=arch_spec)
task = simulator.task(
    physical_kernel,
    place_opt_type=ASAPPlacePass,
    placement_strategy=placement_strategy,
)

# v0.11
from bloqade.gemini import PhysicalSimulator
from bloqade.lanes.passes import ASAPPlacePass

simulator = PhysicalSimulator(
    noise_model=noise_model,
    arch_spec=arch_spec,
    place_opt_type=ASAPPlacePass,
    placement_strategy=placement_strategy,
)
task = simulator.task(physical_kernel)
```

This also applies to backend selection. A simulator's backend is shared by the
tasks it creates, so construct a separate simulator when a workflow needs a
different backend configuration.

## Convert and annotate CUDA-Q kernels before creating a logical task

`GeminiLogicalSimulator.task()` now requires a SQuIN `ir.Method`; it no longer
accepts CUDA-Q kernels or the `m2dets` and `m2obs` arguments. Convert CUDA-Q
kernels and append the desired annotations explicitly.

```python
# v0.10
task = GeminiLogicalSimulator().task(
    main_cuda,
    m2dets=m2dets,
    m2obs=m2obs,
)

# v0.11
from bloqade.gemini import GeminiLogicalSimulator
from bloqade.gemini.compile import append_measurements_and_annotations
from bloqade.gemini.cudaq import cudaq_to_squin

prepared_squin = cudaq_to_squin(main_cuda)
append_measurements_and_annotations(prepared_squin, m2dets, m2obs)
task = GeminiLogicalSimulator().task(prepared_squin)
```

`append_measurements_and_annotations` mutates the SQuIN method in place and
requires at least one of `m2dets` or `m2obs`. If older CUDA-Q code relied on
the default Steane annotations, supply those matrices explicitly before calling
`task()`.

The same change applies when the starting point is already a logical SQuIN
kernel:

```python
# v0.10
task = GeminiLogicalSimulator().task(
    logical_kernel,
    m2dets=m2dets,
    m2obs=m2obs,
)

# v0.11
from bloqade.gemini.compile import append_measurements_and_annotations

prepared_squin = logical_kernel.similar()
append_measurements_and_annotations(prepared_squin, m2dets, m2obs)
task = GeminiLogicalSimulator().task(prepared_squin)
```

Physical tasks likewise no longer accept annotation matrices. Add terminal
physical measurement and annotations before calling `task()`. The migration
helper below preserves the old matrix-based workflow:

```python
# v0.10
task = PhysicalSimulator().task(
    physical_kernel,
    m2dets=m2dets,
    m2obs=m2obs,
)

# v0.11
from bloqade.gemini import PhysicalSimulator
from bloqade.gemini.device.physical_simulator import (
    append_measurements_and_annotations_physical,
)

prepared_kernel = physical_kernel.similar()
append_measurements_and_annotations_physical(
    prepared_kernel,
    m2dets,
    m2obs,
)
task = PhysicalSimulator().task(prepared_kernel)
```

## Configure detector sampling on `TsimSimulatorBackend`

`run_detectors` is no longer an argument to `run()` or `run_async()`. It is a
Tsim-specific backend setting.

```python
# v0.10
result = task.run(shots=1_000, run_detectors=True)

# v0.11
from bloqade.gemini import GeminiLogicalSimulator, TsimSimulatorBackend

simulator = GeminiLogicalSimulator(
    backend=TsimSimulatorBackend(run_detectors=True),
)
task = simulator.task(logical_kernel)
result = task.run(shots=1_000)
```

With `run_detectors=True`, Tsim returns a `DetectorResult` containing detector
and observable samples. Use `TsimSimulatorBackend(run_detectors=False)` for
the raw-measurement path and its post-processing result.

## Seed the backend, not the simulator or an individual task run

The `seed` field was removed from `GeminiLogicalSimulator`, and the `seed`
keyword was removed from `task.run()` and `task.run_async()`. Set a root seed
when constructing a supported backend instead.

```python
# v0.10
simulator = GeminiLogicalSimulator(backend="clifft", seed=1234)
task = simulator.task(logical_kernel)
result = task.run(shots=1_000, seed=1234)

# v0.11
from bloqade.gemini import GeminiLogicalSimulator, TsimSimulatorBackend

simulator = GeminiLogicalSimulator(backend=TsimSimulatorBackend(seed=1234))
result = simulator.task(logical_kernel).run(shots=1_000)
```

`TsimSimulatorBackend` and `CliffTSimulatorBackend` derive a fresh native
sampler seed for each sampling request from this root seed. To replay the same
sequence, create a fresh backend with the same root seed; do not expect
repeated calls on one task to reuse the same per-run seed.

## Create a task before running, visualizing, or inspecting a circuit

Convenience methods on simulator objects were removed. Create a task first,
then use the corresponding task API.

```python
# v0.10
result = simulator.run(kernel, shots=100)
future = simulator.run_async(kernel, shots=100)
simulator.visualize(kernel)
circuit = simulator.tsim_circuit(kernel)
bounds = simulator.fidelity_bounds(kernel)

# v0.11
task = simulator.task(kernel)
result = task.run(shots=100)
future = task.run_async(shots=100)
task.visualize()
circuit = task.tsim_circuit
bounds = task.fidelity_bounds()
```

`physical_squin_kernel`, `physical_move_kernel`, and
`noiseless_tsim_circuit` are similarly accessed from the task.

## Do not depend on backend implementation details

The backend composition fields and backend DEM method are now private. Do not
mechanically rename public API uses to their underscored counterparts:

- `backend.tsim_backend` has no public replacement; a non-Tsim backend is not
  required to be composed from Tsim in future versions.
- `backend.detector_error_model(kernel)` has no public replacement. Obtain the
  guaranteed DEM from the compiled task instead: `task.detector_error_model`.
- `task.measurement_sampler`, `task.noiseless_measurement_sampler`,
  `task.detector_sampler`, and `task.noiseless_detector_sampler` were removed.
  Use `task.run()` / `task.run_async()` and configure detector sampling through
  `TsimSimulatorBackend(run_detectors=True)`. For an intentionally
  Tsim-specific integration, obtain the explicit circuit from
  `task.tsim_circuit` and use Tsim's API directly.
- Simulator tasks store their backend privately. Create tasks through
  `simulator.task(kernel)` instead of constructing task classes directly.
- The PyQrack backend implementation is private and is not a supported public
  backend-construction API in v0.11.

## Additional v0.11 breaking changes outside the simulator refactor

The simulator API changes above came from the simulator refactor. Version 0.11
also contains the following release-wide migrations.

### Import concrete modules instead of `bloqade.lanes` package re-exports

`bloqade.lanes` no longer re-exports simulator types, metrics, noise-model
helpers, or Steane defaults. Import each symbol from its owning module.

```python
# v0.10
from bloqade.lanes import (
    GeminiLogicalSimulator,
    Metrics,
    generate_logical_noise_model,
    steane7_m2dets,
)

# v0.11
from bloqade.gemini import GeminiLogicalSimulator
from bloqade.gemini.steane_defaults import steane7_m2dets
from bloqade.lanes.metrics import Metrics
from bloqade.lanes.noise_model import generate_logical_noise_model
```

`NoiseModelABC` is available from
`bloqade.lanes.rewrite.move2squin.noise`.

### Update relocated compiler and CUDA-Q imports

Several old `bloqade.lanes` modules were removed or relocated:

| v0.10 import or API | v0.11 replacement |
| --- | --- |
| `bloqade.lanes.cudaq_integration` | `bloqade.gemini.cudaq` |
| `bloqade.lanes.steane_defaults` | `bloqade.gemini.steane_defaults` |
| `bloqade.lanes.logical_mvp` | `bloqade.gemini.compile` |
| `bloqade.lanes.pipeline` | `bloqade.lanes.transform` |
| `bloqade.lanes.compile`, `bloqade.lanes.upstream` | Compose the public transform stages in `bloqade.lanes.transform` |

In particular, the `compile_squin_to_move(...)` convenience wrappers were
removed. Use a pipeline directly:

```python
# v0.10
from bloqade.lanes.compile import compile_squin_to_move

move_kernel = compile_squin_to_move(logical_kernel)

# v0.11
from bloqade.lanes.transform import LogicalPipeline

move_kernel = LogicalPipeline(transversal_rewrite=True).emit(logical_kernel)
```

For physical compilation, use `PhysicalPipeline(...).emit(...)`; for a custom
stage-by-stage flow, compose `NativeToPlace`, `PlaceToMove`, and a
`MoveToSquin*` transform as appropriate.

### Replace `MoveSolver` with the typed placement API

`MoveSolver` / `PyMoveSolver` were removed. Fixed-target routing now composes a
shared `SearchEngine`, a `MoveSearch` configuration, and a `TargetSolver`:

```python
# v0.10
from bloqade.lanes.bytecode import MoveSolver

solver = MoveSolver.from_arch_spec(arch_spec)
result = solver.solve(initial, target, blocked)

# v0.11
from bloqade.lanes.bytecode import SearchEngine, MoveSearch, TargetSolver

engine = SearchEngine.from_arch_spec(arch_spec)
search = MoveSearch.entropy()
result = TargetSolver(engine, search).solve(initial, target, blocked)
```

For CZ placement, use the strategy-specific types:
`SingleHeuristicCzPlacement`, `LooseGoalCzPlacement`,
`RecedingHorizonCzPlacement`, or `NoHomeCzPlacement`. The legacy
`MoveSolver.solve_with_generator()` and `MoveSolver.generate_candidates()`
methods have no direct typed-surface equivalent.

### Update bytecode consumers and C FFI integrations

This section matters only if your application persists lane bytecode, parses
or emits bytecode text, or links the native C library.

- The binary program container is incompatible: its magic changes from `BLQD`
  to `LANES`.
- `Program.to_text()` and `Program.from_text()` use the new module/function
  grammar rather than the old flat `.version` format.
- `Instruction.opcode` now exposes the Vihaco opcode byte, not the former
  packed opcode value. Prefer `Instruction.op_name()` when comparing opcode
  identities.
- Invalid, unaligned binary code is reported as `UnalignedCodeError` rather
  than `InvalidCodeSectionLengthError`.
- Every C API identifier was renamed: `blqd_*` becomes `lanes_*`, `BLQD*`
  becomes `LANES*`, and `BlqdStatus` becomes `LanesStatus`.

The unused `bloqade.lanes.rewrite.split_static_placement` module was also
removed. Migrate an explicit import of that module by removing the dependency;
the current placement passes do not use it.
