# AGENT.md — `bloqade.lanes.transform`

## Purpose

The single canonical home for machine-agnostic compilation *stages* and the pipelines
that compose them. A stage is an IR-level → IR-level conversion with a name:

| Module | Stage |
| --- | --- |
| `native_to_place.py` | squin (native gates) → `place` |
| `place_to_move.py` | `place` → `move` |
| `move_to_squin.py` | `move` → squin (+ noise model injection) |
| `move_to_stack.py` | `move` → `stack_move` |
| `pipeline.py` | `PhysicalPipeline`, `LogicalPipeline`, `transversal_rewrites` |

## What goes here vs. next door

- A **new compilation stage** (X → Y over whole methods) goes here.
- A **rewrite rule** (a `RewriteRule` matching one statement shape) goes in `../rewrite/`;
  a stage in this folder drives it.
- A reusable `kirin.passes.Pass` goes in `../passes.py`.
- Gemini-level orchestration (stim emission, task compilation, device wiring) goes in
  `bloqade.gemini.compile`.

## Canonical API — do not fork

`PhysicalPipeline` and `LogicalPipeline` are *the* compile entry points for this layer.
The duplicate-compile-wrapper drift this folder consolidated (two parallel squin→move
implementations) is exactly the failure mode to avoid.

When you need different behaviour:

1. Prefer configuring the existing pipeline — `arch_spec`, `layout_heuristic`,
   `placement_strategy`, `place_opt_type` are injection points, and `arch_spec` is the
   single source of truth when the others are left `None`.
2. Otherwise add a new stage module here and wire it into the existing pipeline.
3. Do **not** add a new `compile_*` convenience function in a sibling package that
   re-implements the sequence.

## Layering rule

Same as `bloqade.lanes`: no new imports of `bloqade.gemini`. `native_to_place.py` has a
tolerated legacy edge to `gemini.common.validation` / `gemini.logical.*`; do not copy it.
Machine specifics arrive via `ArchSpec` and injected strategies.
