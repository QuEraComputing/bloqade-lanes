# AGENT.md — `bloqade.lanes.rewrite`

## Purpose

`kirin.rewrite.abc.RewriteRule` subclasses: local, pattern-matched IR edits. A rule
matches a statement (or small window) and rewrites it. Rules do not own a compilation
stage and do not decide their own iteration order.

Modules are named after the direction they rewrite (`circuit2place`, `place2move`,
`clifford2native`, `move2squin/`, `stack_move2move`, `move2stack_move`) plus a few
standalone cleanups (`fuse_gates`, `measure_lower`, `resolve_pinned`, `remove_debug`,
`stackify`, `transversal`, `reorder_static_placement/`).

## What goes here vs. next door

| You are adding | Put it in |
| --- | --- |
| A `RewriteRule` | here, in the `X2Y` module matching its direction |
| A `kirin.passes.Pass` that sequences rules + analysis | `../passes.py` |
| A whole-method stage (X IR → Y IR) that drives rules | `../transform/` |
| An abstract interpretation / lattice analysis | `../analysis/` |
| A statement or dialect definition | `../dialects/` |

Rules stay free of pipeline policy: no arch-spec construction, no strategy selection, no
top-level `Method` orchestration. If a rule needs a decision made once per method, take
it as a constructor field and let the caller in `transform/` or `passes.py` supply it.

## Layering rule

No new `bloqade.gemini` imports. `circuit2place.py` has a tolerated legacy edge to
`gemini.common.dialects` / `gemini.logical.dialects.operations.stmts`; it is not
precedent. A rule that only makes sense for Steane/Gemini logical encoding belongs in
`bloqade.gemini.logical.rewrite`.
