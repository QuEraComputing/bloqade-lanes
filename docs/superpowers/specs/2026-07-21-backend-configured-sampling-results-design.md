# Backend-Configured Simulator Sampling Results

## Summary

Remove `run_detectors` from simulator, simulator-task, and backend `sample()`
method signatures. Sampling representation becomes a backend configuration
choice. `TsimSimulatorBackend` exposes `run_detectors: bool = False`; CliffT and
PyQrack always return raw measurements.

Task execution inspects the returned `BackendSample` and constructs either the
existing measurement-backed `Result` or a detector-backed `DetectorResult`.
Both concrete result types satisfy one public `SimulatorResult` protocol.

This is a breaking public API change.

## Public API

### Simulator result protocol

Add a generic, public `SimulatorResult[RetType]` protocol with:

- `fidelity_bounds()`
- `detector_error_model`
- `return_values`
- `detectors`
- `measurements`
- `observables`

Collection-valued properties use read-only `Sequence` return types in the
protocol so that the existing list-returning `Result` and tuple-returning
`DetectorResult` both satisfy it.

`Result` remains unchanged and continues to derive return values, detectors,
and observables from raw measurements through atom postprocessing.

`DetectorResult` becomes generic in `RetType` so it can satisfy the generic
protocol, although it stores no values of that type. It gains `measurements` and
`return_values` properties. Both raise a clear `ValueError`, because neither raw
measurements nor arbitrary kernel return values can be reconstructed from
detector and observable samples. These are ordinary properties rather than
cached properties because an operation that always raises produces no value to
cache.

### Run methods

Remove `run_detectors` from:

- logical and physical simulator `run()` methods
- logical and physical simulator `run_async()` methods
- `_SimulatorTaskBase.run()`
- `_SimulatorTaskBase.run_async()`

Remove the corresponding overloads. Synchronous methods return
`SimulatorResult[RetType]`; asynchronous methods return
`Future[SimulatorResult[RetType]]`.

The concrete runtime value remains either `Result[RetType]` or
`DetectorResult[RetType]`.

## Backend Contract

Remove `run_detectors` from `AbstractSimulatorBackend.sample()` and every
implementation.

Backend behavior is:

| Backend | Configuration | Payload |
| --- | --- | --- |
| `TsimSimulatorBackend` | `run_detectors=False` | measurements only |
| `TsimSimulatorBackend` | `run_detectors=True` | detectors and observables only |
| `CliffTSimulatorBackend` | none | measurements only |
| `PyQrackSimulatorBackend` | none | measurements only |

`TsimSimulatorBackend` gains the public dataclass field
`run_detectors: bool = False`. Its `sample()` method reads this field to select
its sampler. CliffT no longer conditionally exposes its native detector arrays;
it consistently returns raw measurements.

## Task Data Flow and Validation

The task runtime remains backend-agnostic. It does not inspect the backend type
or read `TsimSimulatorBackend.run_detectors`. Instead, it validates the
`BackendSample` representation returned by `sample()`.

Exactly two payload shapes are valid:

1. `measurements` is present, while `detectors` and `observables` are absent.
   Normalize the measurement matrix and construct `Result`.
2. `measurements` is absent, while both `detectors` and `observables` are
   present. Normalize both matrices and construct `DetectorResult`.

Reject empty, partial, and mixed payloads with `ValueError`. In particular, a
payload containing all three arrays is invalid rather than assigning an
implicit precedence.

This keeps the task/backend boundary independent of specific backend classes
and leaves `Result` unchanged.

## Internal Consumers and Compatibility

All internal calls stop passing `run_detectors`. Code that requires native Tsim
detector sampling constructs its simulator with:

```python
GeminiLogicalSimulator(
    backend=TsimSimulatorBackend(run_detectors=True),
)
```

Existing code that calls `run(..., run_detectors=True)` or
`sample(..., run_detectors=True)` must migrate to backend construction. This is
intentional and receives no compatibility alias.

`DetectorResult`, `Result`, and the new `SimulatorResult` protocol remain public
exports from both `bloqade.gemini.device` and `bloqade.gemini`. Decoding helpers
accept the protocol (or the narrow structural members they consume) instead of
spelling a concrete-result union.

## Testing

Tests cover:

- absence of `run_detectors` from public run, async, and sample signatures;
- Tsim measurement mode by default and detector mode when configured;
- CliffT and PyQrack measurement-only backend payloads;
- exact task validation of measurement-only and detector-only payloads;
- rejection of empty, partial, and mixed payloads;
- concrete result selection from valid payloads;
- successful protocol conformance for both concrete result types;
- clear exceptions from detector-only `measurements` and `return_values`;
- seed forwarding through the simplified synchronous and asynchronous APIs;
- logical and physical simulator behavior and decoding integration;
- static return-type assertions for `SimulatorResult` and its future.

Run focused tests first, then formatting, Ruff, Pyright, and the complete Python
test suite.
