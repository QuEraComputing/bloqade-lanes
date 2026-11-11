# Gemini namespace ownership

Both `bloqade.lanes` and `bloqade.gemini` continue to ship in the
`bloqade-lanes` distribution, from `python/bloqade`. This refactor separates
compiler and frontend responsibilities inside this repository; it does not
move the Gemini package to another distribution, and installation instructions
are unchanged. Gemini's frontend API (`bloqade.gemini.logical`,
`bloqade.gemini.physical`, devices, simulation, decoding) keeps its import paths;
the relocated compiler modules below do not.

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

Gemini owns its qubit-address, Steane type-inference/measurement, and validation
method tables. Importing its logical or physical kernel groups loads these
implementations before constructing the groups. Registration attaches them to
the shared lanes/Kirin dialect objects; it is process-wide, not per kernel group.
Validation passes and analysis classes remain in lanes, which never imports
Gemini. Direct compiler callers must first load their frontend's tables;
missing required registrations raise a clear error. Generic move/place and
architecture evaluation/constant-propagation tables remain in lanes.

## Packaging and a future distribution split

The lanes wheel includes both namespaces and excludes generated caches. No
coordinated bloqade release or local path dependency is needed for this refactor.
The prepared Gemini relocation in the bloqade repository is a separate future
change. Do not release that copy alongside this lanes wheel: both distributions
would own `bloqade.gemini`. A future split must remove Gemini from the lanes
wheel and update bloqade's dependency and lockfile together.

## Saved IR and backend compatibility

Statement module paths and dialect names changed (`lanes.arch`, `lanes.arrange`,
`lanes.steane713`, and `lanes.steane713.extensions`). Recompile saved kernels from
source. Producers and consumers of serialized pre-lowering IR must upgrade
together; old serialized kernels are not promised to load. Compiled move/bytecode
formats are unchanged. New IR is covered by serialization round-trip tests.

The recursion, static-call, and call-site utilities currently live in lanes.
Upstreaming them to Kirin remains a separate proposal requiring its maintainer's
review; this change does not implement that proposal.
