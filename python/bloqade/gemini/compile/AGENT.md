# AGENT.md — `bloqade.gemini.compile`

## Purpose

The one home for Gemini-level compile entry points and orchestration. These functions take
a user kernel and produce something a Gemini backend can consume:

| Module | Entry points |
| --- | --- |
| `task.py` | `compile_task`, `append_measurements_and_annotations`, `run_squin_kernel_validation` |
| `stim.py` | `compile_to_stim_program` |

## What goes here vs. next door

- **Machine-agnostic squin → place → move** is `bloqade.lanes.transform`. This package
  *calls* `LogicalPipeline` / `PhysicalPipeline`; it does not re-implement them.
- **Reusable Gemini pieces** — validation suites, rewrite rules, dialect statements,
  Steane defaults — live in `gemini/{common,logical,physical}` and `steane_defaults.py`.
  Keep this package thin: sequencing and artifact emission, not new primitives.
- **Execution** (submitting, sampling, results) belongs in `gemini/device/`; **decoding**
  in `gemini/decoding/`.

## Canonical API — do not fork

This package exists because compile entry points had previously forked into two
independent implementations with duplicate names. Adding a third variant is the regression
to avoid.

- New compile target (a new artifact kind)? Add a module here and export it from
  `__init__.py`.
- Variation on an existing target? Add a parameter to the existing entry point rather than
  a near-copy under a new name.
- Everything public goes through `bloqade.gemini.compile.__init__`; keep helpers private
  (`_` prefix) so the public surface stays readable.
