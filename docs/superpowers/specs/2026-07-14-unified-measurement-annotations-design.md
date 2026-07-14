# Unified Measurement Annotations

**Status:** approved (design)
**Date:** 2026-07-14

## Goal

Replace the duplicated logical and physical measurement-annotation implementations
with one public `append_measurements_and_annotations` API while preserving the two
existing matrix-layout contracts and existing callers.

## Public API

Add the shared implementation in `bloqade.gemini.measurement_annotations`:

```python
def append_measurements_and_annotations(
    mt: ir.Method,
    m2dets: list[list[int]] | None,
    m2obs: list[list[int]] | None,
    *,
    level: Literal["logical", "physical"] = "logical",
) -> None: ...
```

The default remains `"logical"` so imports from `bloqade.lanes.logical_mvp` stay
source-compatible. `append_measurements_and_annotations_physical` remains as a
thin compatibility wrapper and delegates with `level="physical"`.

## Behavior

Both levels validate that at least one matrix is present, matrices have at least
one row and are rectangular, and detector/observable matrices have equal row
counts when both are supplied. Zero-column matrices such as `[[]]` remain valid.

Logical mode:

- Treats the rows as one global measurement map across all logical qubits.
- Finds or creates `TerminalLogicalMeasurement`.
- Resolves a matrix row through the nested logical-qubit/physical-measurement
  result.
- Emits detector coordinates `(0, detector_index)`.

Physical mode:

- Aggressively unrolls the kernel before resolving allocations.
- Treats the rows as a per-logical-block map and repeats annotations once per
  physical-qubit block.
- Requires exactly one physical `qubit.Measure` statement and indexes its flat
  result.
- Emits detector coordinates `(logical_block_index, detector_index)`.
- Leaves the kernel's existing return value unchanged.

After each mode prepares its measurement resolver, a shared emitter creates all
`SetDetector` and `SetObservable` statements. Address-based qubit discovery and
IR insertion helpers also live in the shared module. The old modules continue to
expose `_find_qubit_ssas`; `logical_mvp` also continues to expose
`_find_return_stmt`, matching existing test and consumer imports.

## Errors and Compatibility

Invalid levels raise `ValueError`. Existing level-specific allocation and
measurement errors retain their current messages where practical. No deprecation
warning is emitted by the physical compatibility wrapper, avoiding new runtime
noise while callers migrate.

## Testing

- Exercise physical behavior through the unified API.
- Verify the legacy physical wrapper delegates correctly.
- Retain logical annotation-count, terminal-measurement, allocation, and return
  tests.
- Retain physical block repetition, divisibility, `new_at`, post-processing, and
  return-preservation tests.
- Add validation coverage for an invalid `level` and mismatched matrix row counts.

## Non-goals

- Changing the meaning or orientation of either matrix format.
- Automatically inferring logical versus physical mode from kernel contents.
- Changing detector or observable post-processing semantics.
