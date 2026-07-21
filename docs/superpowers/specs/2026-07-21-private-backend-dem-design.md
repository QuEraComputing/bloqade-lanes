# Private Backend Detector Error Model Design

## Summary

Rename the detector-error-model method on the Gemini simulator backend protocol
from `detector_error_model()` to `_detector_error_model()`.

## Scope

Rename the abstract method and every implementation on:

- `AbstractSimulatorBackend`
- `TsimSimulatorBackend`
- `CliffTSimulatorBackend`
- `PyQrackSimulatorBackend`

Update `_SimulatorTaskBase` and repository tests to call the private method. Do
not add a public compatibility alias.

The public task/result APIs remain unchanged:

- `_SimulatorTaskBase.detector_error_model`
- `Result.detector_error_model`
- `DetectorResult.detector_error_model`

The native Tsim circuit call
`circuit.detector_error_model(approximate_disjoint_errors=True)` also remains
unchanged because it is not part of the Gemini backend interface.

## Behavior

Only the Python method name changes. Tsim conversion, DEM generation, error
reframing, caching, sampling, and optional-dependency behavior remain unchanged.

## Tests

Tests verify that backend instances expose `_detector_error_model` and not the
public method, that task-level DEM access delegates privately, and that Tsim,
CliffT, PyQrack, and optional-import behaviors remain intact.
