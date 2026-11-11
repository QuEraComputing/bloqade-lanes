# Gemini namespace ownership

The `bloqade-lanes` distribution now ships only `bloqade.lanes`, from
`python/bloqade/lanes`. The Gemini frontend is moving to `src/bloqade/gemini`
in the `bloqade` repository ([PR #392](https://github.com/QuEraComputing/bloqade/pull/392)).
Install the corresponding bloqade release to use `bloqade.gemini.logical`,
`bloqade.gemini.physical`, devices, simulation, and decoding. Their public
import paths are unchanged by the distribution move; installing lanes alone
no longer supplies them.

Compiler definitions belong to `bloqade.lanes`:

| Previous definition | Canonical definition |
| --- | --- |
| `gemini.common.dialects.qubit` | `lanes.dialects.arch` (`NewAt`, `new_at`) |
| `gemini.common.dialects.arrange` | `lanes.dialects.arrange` |
| `gemini.logical.dialects.operations` | `lanes.dialects.steane713` |
| `gemini.logical.dialects.extensions` | `lanes.dialects.steane713.extensions` |
| `gemini.star` | `lanes.dialects.steane713.star` |
| `GeminiLogicalValidation` | `lanes.validation.steane.SteaneLogicalValidation` |
| `GeminiTerminalMeasurementValidation` | `lanes.validation.terminal_measurement.SteaneTerminalMeasurementValidation` |

This is a breaking change: the previous module paths are removed, with no
deprecated re-exports. Import these definitions from their lanes paths. The
frontend re-exports that are part of the kernel API (for example
`bloqade.gemini.logical.terminal_measure`, `loc` and `move_to`) remain.
Device-only extension rejection and kernel argument policies remain in Gemini.
The experimental STAR dialect remains separate from the ordinary Steane dialect.

## Method-table registration

Lanes owns the analysis and validation drivers. Gemini supplies and registers
the method tables for `NewAt` address analysis, allocation/movement validation,
Steane terminal-measurement type inference and measurement IDs, logical gate
validation, and physical/logical terminal-measurement validation. These tables
already exist in bloqade PR #392 and are removed from lanes to avoid duplicate
registration. Importing `bloqade.gemini` loads them before its kernel groups run.

Lanes retains its other method tables, including architecture constant folding
and concrete evaluation, Move validation, and compiler analyses. Lanes never
imports Gemini. Generic lanes compilation remains independent; callers using
Gemini-specific validation must load the frontend first. Registration attaches
tables to shared lanes/Kirin dialect objects and is process-wide.

## Packaging and release coordination

The lanes wheel excludes `bloqade.gemini` and generated caches. Gemini tests
belong with the frontend in the bloqade repository; lanes retains compiler-only
tests, including compilation without importing Gemini. Lanes CI no longer runs
Gemini demos or the MSD numerical regression. The retained Gemini demo scripts
require a separately installed compatible bloqade frontend.

Publish a compatible lanes release before the bloqade frontend release, then
update bloqade's dependency and lockfile. Do not combine the new bloqade frontend
with an older lanes wheel that also ships `bloqade.gemini`: both distributions
would own the same files.

## Saved IR and backend compatibility

Statement module paths and dialect names changed (`lanes.arch`, `lanes.arrange`,
`lanes.steane713`, and `lanes.steane713.extensions`). Recompile saved kernels from
source. Producers and consumers of serialized pre-lowering IR must upgrade
together; old serialized kernels are not promised to load. Compiled move/bytecode
formats are unchanged. New IR is covered by serialization round-trip tests.

The recursion, static-call, and call-site utilities currently live in lanes.
Upstreaming them to Kirin remains a separate proposal requiring its maintainer's
review; this change does not implement that proposal.
