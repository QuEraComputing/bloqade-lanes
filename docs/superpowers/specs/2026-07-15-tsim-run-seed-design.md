# Per-call tsim sampling seed

## Goal

Allow callers to seed one simulation invocation without changing the task's
default cached sampler or the simulator's existing configuration.

## API

Add an optional keyword-only `seed: int | None = None` argument to
`GeminiLogicalSimulatorTask.run`, `GeminiLogicalSimulatorTask.run_async`,
`GeminiLogicalSimulator.run`, and `GeminiLogicalSimulator.run_async`. The
same arguments will be added to the `_CliffTSimulatorTask` overrides so that
`GeminiLogicalSimulator(backend="clifft")` supports the advertised API.

The physical simulator is out of scope: this change is limited to the Gemini
logical simulator and its task types.

## Behavior

When `seed` is `None`, retain the current paths and cached sampler behavior.
When it is provided, compile a fresh seeded sampler for the specific call and
pass it to both measurement and detector execution paths. This applies to the
Stim path used for Clifford circuits and the tsim path used for non-Clifford
circuits. The simulator convenience methods forward the argument to their
newly created task.

For CliffT, the per-call seed takes precedence over the existing task-level
`GeminiLogicalSimulator(seed=...)` value; when the per-call seed is `None`,
the existing task-level value continues to apply. `seed=0` is a valid seed and
must be forwarded. Valid seeds are non-boolean Python integers in the range
`0 <= seed < 2**63`; invalid values raise `ValueError` before compilation so
all supported backends have the same contract.

## Validation

Unit tests will confirm that seeded calls pass the value to each backend's
sampler compiler and that the simulator `run` and `run_async` methods forward
the seed unchanged. They will cover Stim measurement, Stim detector
conversion, tsim measurement, and tsim detector sampling with both noisy and
noiseless circuits. Async task and simulator calls will be checked through
`.result()`. Tests will also confirm that a seeded call compiles a fresh
sampler every time, `seed=None` continues to use the cached tsim samplers,
two independent calls with the same seed reproduce samples at a fixed batch
size, and a seeded call does not leak into a later unseeded call.
