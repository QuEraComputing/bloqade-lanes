# Private Tsim Backend Field Design

## Summary

Rename the embedded `TsimSimulatorBackend` dependency on
`CliffTSimulatorBackend` and `PyQrackSimulatorBackend` from `tsim_backend` to
`_tsim_backend`.

## API

Both dataclasses retain constructor injection through the private-by-convention
keyword:

```python
CliffTSimulatorBackend(_tsim_backend=backend)
PyQrackSimulatorBackend(_tsim_backend=backend)
```

The field keeps `default_factory=TsimSimulatorBackend` and `repr=False`. It is
not changed to `init=False`. No public `tsim_backend` compatibility property is
added.

## Implementation

All internal circuit-conversion and detector-error-model delegation uses
`self._tsim_backend`. Existing seed, sampling, caching, and error-reframing
behavior is unchanged.

Repository tests that inject or patch this dependency use `_tsim_backend`.
Local variable names may remain `tsim_backend` because they are not public
attributes.

## Tests

Tests verify:

1. Both backends expose `_tsim_backend` and no longer expose `tsim_backend`.
2. Private constructor injection delegates circuit conversion and detector error
   model generation to the injected backend.
3. Optional-Tsim import behavior continues to work when the private dependency
   is patched.
4. Existing CliffT and PyQrack backend tests continue to pass.

## Out of Scope

- Removing constructor injection with `init=False`.
- Adding a compatibility alias or deprecation period.
- Changing `TsimSimulatorBackend` behavior.
- Changing backend sampling, seed, or cache semantics.
