# Per-call tsim sampling seed

## Goal

Allow callers to seed one simulation invocation without changing the task's
default cached sampler or the simulator's existing configuration.

## API

Add an optional keyword-only `seed: int | None = None` argument to
`GeminiLogicalSimulatorTask.run`, `GeminiLogicalSimulatorTask.run_async`,
`GeminiLogicalSimulator.run`, and `GeminiLogicalSimulator.run_async`.

## Behavior

When `seed` is `None`, retain the current paths and cached sampler behavior.
When it is provided, compile a fresh seeded sampler for the specific call and
pass it to both measurement and detector execution paths. This applies to the
Stim path used for Clifford circuits and the tsim path used for non-Clifford
circuits. The simulator convenience methods forward the argument to their
newly created task.

## Validation

Unit tests will confirm that seeded calls pass the value to each backend's
sampler compiler and that the simulator `run` and `run_async` methods forward
the seed unchanged. Existing no-seed tests preserve the default behavior.
