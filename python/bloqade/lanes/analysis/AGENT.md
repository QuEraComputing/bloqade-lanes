# AGENT.md — `bloqade.lanes.analysis`

## Purpose

Kirin abstract-interpretation analyses over the movement IRs. Each subpackage follows the
same shape: a lattice, an interpreter/analysis entry point, and a method table of
per-statement implementations.

| Subpackage | Answers |
| --- | --- |
| `atom/` | where atoms are and what executing a `move` kernel does (`AtomInterpreter`, `MoveExecution` lattice, measurement/detector results, post-processing) |
| `layout/` | layout analysis + the `LayoutHeuristicABC` interface |
| `placement/` | placement analysis, `AtomState`/`ConcreteState` lattice, `PlacementStrategyABC` |

## Conventions

- Keep the split: `lattice.py` (lattice elements) · `analysis.py` (interpreter / public
  entry) · `impl.py` (`interp.MethodTable` per dialect) · `exceptions.py`.
- Adding a statement to a dialect means adding its method-table entry here; a missing
  entry shows up as an unknown/bottom lattice value, not a clean error.
- This package defines the **strategy/heuristic ABCs**; concrete implementations live in
  `../heuristics/`. Keep it that way — an analysis must not import a specific heuristic.

## What goes here vs. next door

- **Read-only interpretation producing facts** — here.
- **Anything that mutates IR** — `../rewrite/` (rules) or `../passes.py` (passes that run
  an analysis, then rewrite with its results).
- **Concrete layout/placement strategies** — `../heuristics/`.

`../metrics.py` (`Metrics`) is the known outlier: it computes fidelity/move statistics by
running pipelines by hand and should be reworked into an analysis pass here. Add new
measurement code as an analysis in this package rather than extending `metrics.py`.

## Layering rule

No `bloqade.gemini` imports — this package currently has none, and it should stay that
way. Analyses are parameterized by `ArchSpec`, never by machine identity.
