# Squin-Only Simulator API Design

## Summary

`GeminiLogicalSimulator` and `GeminiPhysicalSimulator` will consume prepared
Squin kernels only. They will no longer accept CUDA-Q callables or
measurement-to-detector/observable matrices. CUDA-Q conversion and annotation
insertion remain explicit preprocessing operations outside the simulator
objects.

This is an intentional breaking API change.

## Public API

Remove `m2dets` and `m2obs` from both simulator dataclasses. Do not add a
`task_cudaq()` method.

The following methods accept `ir.Method[[], RetType]` Squin kernels only:

- `task()`
- `run()`
- `run_async()`
- `visualize()`
- task-based compilation and inspection helpers such as
  `physical_squin_kernel()`, `physical_move_kernel()`, `tsim_circuit()`, and
  `fidelity_bounds()`

Remove the `LogicalKernel` union alias and update all overloads, docstrings, and
the `PhysicalSimulator` alias-facing documentation to describe Squin-only input.

Passing a CUDA-Q callable or another non-`ir.Method` value to `task()` must fail
with a clear `TypeError`. This runtime check prevents Python callers from
bypassing the type-only API boundary. Methods that delegate through `task()`
inherit the same behavior.

## Responsibilities

Simulator objects are responsible for:

1. Copying the supplied Squin kernel before mutating compilation passes.
2. Compiling the Squin kernel through the selected logical or physical
   pipeline.
3. Creating and running simulator tasks through the configured backend.

Simulator objects are not responsible for:

1. Detecting or converting CUDA-Q kernels.
2. Adding terminal measurements for CUDA-Q kernels.
3. Inserting detector or observable annotations from matrices.

The existing standalone `cudaq_to_squin()`,
`append_measurements_and_annotations()`, and
`append_measurements_and_annotations_physical()` utilities remain available.
Users may compose those utilities before passing the resulting `ir.Method` to a
simulator.

## Logical Compilation

`GeminiLogicalSimulator.task()` accepts an `ir.Method`, makes or receives an
owned copy through the logical compilation helper, performs Squin validation,
and builds a `GeminiLogicalSimulatorTask`. It passes no measurement matrices to
the compiler.

The lower-level `compile_task()` helper may retain its existing broader API for
compatibility with non-simulator callers. The simulator must nevertheless
enforce its Squin-only boundary before calling it.

## Physical Compilation

`GeminiPhysicalSimulator.task()` accepts an `ir.Method`, verifies the runtime
type, copies it with `similar()`, and compiles that copy. It does not call
`append_measurements_and_annotations_physical()`.

A caller-provided physical Squin kernel must already satisfy physical terminal
measurement validation and contain any desired detector or observable
annotations.

## User Workflow

Logical CUDA-Q users explicitly prepare the Squin kernel:

```python
squin_kernel = cudaq_to_squin(cudaq_kernel)
append_measurements_and_annotations(squin_kernel, m2dets, m2obs)
task = GeminiLogicalSimulator().task(squin_kernel)
```

Physical users are responsible for producing a physical-pipeline-compatible
Squin kernel, including terminal physical measurement and annotations. The
simulator does not promise that raw `cudaq_to_squin()` output is directly
physical-pipeline compatible.

## Errors

- Constructor calls containing `m2dets` or `m2obs` fail under normal dataclass
  argument checking because those fields no longer exist.
- `task(non_method)` raises `TypeError` with a message stating that a Squin
  `ir.Method` is required.
- Existing pipeline validation errors remain unchanged for invalid Squin
  methods.

## Tests

Tests will verify:

1. Both simulator constructors no longer expose matrix fields.
2. Logical and physical `task()` accept Squin `ir.Method` kernels.
3. Both `task()` implementations reject CUDA-Q-like callables/non-methods
   before compilation.
4. Simulator task creation does not insert matrix-derived annotations.
5. Logical and physical `run()`, `run_async()`, and `visualize()` delegate only
   through the Squin-only `task()` path.
6. Existing CUDA-Q integration tests are migrated to explicit
   conversion/annotation preprocessing where they remain useful.
7. Source Squin kernels remain unmutated by task creation.
8. Repository demos using simulator constructor matrices or CUDA-Q callables
   are migrated to explicit preprocessing. In the current tree this includes
   `demo/cudaq_demo.py`; unrelated untracked demos are not part of the change.

## Out of Scope

- Native CUDA-Q measurement support.
- A physical-specific CUDA-Q converter.
- `task_cudaq()`, `run_cudaq()`, or similar convenience methods.
- Removal of standalone CUDA-Q or annotation utilities.
- Redesign of the lower-level `compile_task()` helper beyond what is necessary
  for the simulator boundary.
