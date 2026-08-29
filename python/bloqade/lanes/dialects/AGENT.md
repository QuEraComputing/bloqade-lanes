# AGENT.md — `bloqade.lanes.dialects`

## Purpose

Kirin dialect definitions for the movement compilation IRs — statements, types, and their
directly-attached interfaces:

| Dialect | Level |
| --- | --- |
| `place` | logical placement: qubits pinned to abstract locations, gates over regions |
| `move` | physical moves: lanes, zones, concrete location addresses |
| `stack_move` | stack-oriented form of `move` |
| `arch/` | architecture statements + their const-prop, impl, and rewrite support |

## What goes here vs. next door

- **Statement / type / attribute definitions and their signatures** — here.
- **Rewrites between dialects** — `../rewrite/`.
- **Analyses interpreting these statements** — `../analysis/` (method tables live with the
  analysis, not with the dialect).
- **Stage drivers and pipelines** — `../transform/`.

Keep these modules declarative. Validation logic that needs more than a statement-local
check belongs in an analysis or a validation pass, not inlined in a statement definition.
Adding a statement is usually a three-place change: define it here, teach the relevant
analysis method table about it in `../analysis/`, and add the rewrite that produces or
consumes it in `../rewrite/`.

## Layering rule

No new `bloqade.gemini` imports. `place.py` and `move.py` each carry one tolerated legacy
edge (`gemini.star.validate_steane_star_support`); do not add more. Statements here must
be meaningful for any `ArchSpec`-described machine — Gemini-only statements belong in
`bloqade.gemini.{common,logical}.dialects`.
