# Shot Remapping: Map Hardware Shots to Logical Qubit Measurements

**Date:** 2026-04-02
**Issue:** #98
**Status:** Draft

## Problem

Neutral atom hardware returns a measurement bitstring for every `LocationAddress` in `ZoneAddress(0)`, including sites with no atom. The compiler service needs a static index mapping (`list[list[int]]`) that tells it which positions in the raw shot bitstring correspond to the physical qubits of each logical qubit.

This mapping must be congruent with the `ArchSpec` used by both bloqade-lanes (compiler) and bloqade-flair (executor), since the `ArchSpec` defines the site ordering that determines the bitstring layout.

## Scope

This design covers only the **atom-position-based shot remapping** within bloqade-lanes. It does not cover:

- Detector/observable post-processing (stays in `bloqade.gemini.logical`)
- Stripping `SetDetector`/`SetObservable` from kernels (tracked separately, depends on QuEraComputing/bloqade-internal#222)
- Deprecation of the callable-based `PostProcessing` (tracked separately)

## Design

### Change 1: Add `location_address` to `MeasureResult`

**File:** `python/bloqade/lanes/analysis/atom/lattice.py`

Add a `location_address: LocationAddress` field to `MeasureResult`:

```python
@final
@dataclass
class MeasureResult(MoveExecution):
    qubit_id: int
    location_address: LocationAddress

    def copy(self):
        return MeasureResult(self.qubit_id, self.location_address)

    def is_subseteq_MeasureResult(self, elem: "MeasureResult") -> bool:
        return self.qubit_id == elem.qubit_id and self.location_address == elem.location_address
```

**File:** `python/bloqade/lanes/analysis/atom/impl.py`

Update `GetFutureResult` handler to pass `location_address`:

```python
@interp.impl(move.GetFutureResult)
def get_future_result_impl(self, interp_, frame, stmt):
    future = frame.get(stmt.measurement_future)
    if not isinstance(future, MeasureFuture):
        return (Bottom(),)

    result = future.results.get(stmt.zone_address)
    if result is None:
        return (Bottom(),)

    qubit_id = result.get(stmt.location_address)
    if qubit_id is None:
        return (Bottom(),)

    return (MeasureResult(qubit_id, stmt.location_address),)
```

### Change 2: New `get_shot_remapping` function

**File:** `python/bloqade/lanes/analysis/atom/_shot_remapping.py` (new)

```python
def get_shot_remapping(
    return_value: MoveExecution,
    arch_spec: ArchSpec,
) -> list[list[int]]:
```

**Algorithm:**

1. Build `LocationAddress -> zone_index` from `arch_spec.yield_zone_locations(ZoneAddress(0))`
2. Validate `return_value` is `IListResult[IListResult[MeasureResult]]`
3. For each outer element (logical qubit), collect the `zone_index` of each inner `MeasureResult.location_address`
4. Return `list[list[int]]`

**Error cases:**

- Return value structure is not `IListResult[IListResult[MeasureResult]]` -> `ValueError`
- A `MeasureResult.location_address` is not found in Zone 0 -> `ValueError`

### Change 3: `AtomInterpreter.get_shot_remapping` method

**File:** `python/bloqade/lanes/analysis/atom/analysis.py`

Add a convenience method that runs the interpreter and calls the standalone function:

```python
def get_shot_remapping(self, method: ir.Method) -> list[list[int]]:
    _, output = self.run(method)
    return _get_shot_remapping(output, self.arch_spec)
```

### Change 4: Export from `analysis/atom/__init__.py`

Export `get_shot_remapping` (the standalone function) and make it available from the package.

### Impact on existing code

- `_post_processing.py:constructor_function` uses `MeasureResult.qubit_id` as an index into the measurement sequence. This still works unchanged since `qubit_id` remains present.
- The `MeasureResult.copy()` and `is_subseteq` methods are updated to include the new field.
- No other lattice types are affected.

## Cross-Package Dependencies

1. **bloqade-flair:** Must use `ArchSpec` to define the full set of Zone 0 measurement sites, so the shot bitstring ordering is congruent with what bloqade-lanes produces.
2. **bloqade-lanes (gemini submodule):** After QuEraComputing/bloqade-internal#222 merges, port the detector/observable stripping rewrite into `bloqade.gemini.logical`, preserving `TerminalLogicalMeasurement`.
3. **bloqade-lanes:** Deprecate callable-based `PostProcessing` in favor of `bloqade.gemini.logical` measurement post-processing.

## Work Plan

### Task 1: Add `location_address` to `MeasureResult`
- Modify `lattice.py`: add field, update `copy()` and `is_subseteq`
- Modify `impl.py`: pass `stmt.location_address` in `get_future_result_impl`
- Run existing tests to confirm no regressions

### Task 2: Implement `get_shot_remapping` function
- Create `_shot_remapping.py` with the standalone function
- Add convenience method to `AtomInterpreter`
- Export from `__init__.py`

### Task 3: Tests
- Unit test `get_shot_remapping` with hand-built `IListResult`/`MeasureResult` lattice values and a small `ArchSpec`
- Integration test via `compile_task()` with a known circuit: verify indices correctly remap a synthetic full shot to per-logical-qubit measurements

### Task 4: Open cross-package tracking issues
- bloqade-flair: ArchSpec congruence for shot bitstring ordering
- bloqade-lanes: port stripping rewrite after bloqade-internal#222
- bloqade-lanes: deprecate `PostProcessing`
