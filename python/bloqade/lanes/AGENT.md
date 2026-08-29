# AGENT.md — `bloqade.lanes`

## Purpose

Machine-agnostic movement compilation: squin → place → move → squin/stim. Everything
here should make sense for *any* neutral-atom architecture described by an `ArchSpec`,
not just Gemini.

## Layering rule

`bloqade.gemini` sits **on top of** `bloqade.lanes`.

- `gemini` may import `lanes` freely.
- `lanes` **must not** import `gemini`. New code in this tree gets no new upward edge.

Machine specifics are expressed through data (`arch/`, `ArchSpec`) and injected
strategies, not through importing the Gemini package. If you need a Gemini concept in a
lanes module, that is the signal the code belongs in `bloqade.gemini` instead.

### Currently tolerated exceptions

These four modules still import `gemini`. They are legacy, tracked for removal, and must
not be used as precedent:

| Module | Upward import |
| --- | --- |
| `dialects/place.py`, `dialects/move.py` | `gemini.star.validate_steane_star_support` |
| `rewrite/circuit2place.py` | `gemini.common.dialects.{arrange,qubit}`, `gemini.logical.dialects.operations.stmts` |
| `transform/native_to_place.py` | `gemini.common.validation.*`, `gemini.logical.{rewrite,validation}.*` |

Note the practical consequence: because those edges exist, `bloqade/lanes/__init__.py`
deliberately has **no eager re-exports** — adding one can re-enter an in-progress
`bloqade.gemini` import and break initialization. Import from submodules.

## What goes where

| Folder | Holds |
| --- | --- |
| `dialects/` | Kirin dialect + statement definitions (`place`, `move`, `stack_move`, `arch`) |
| `rewrite/` | `RewriteRule` subclasses — local IR pattern rewrites |
| `passes.py` | `kirin.passes.Pass` subclasses composing rewrites + analysis |
| `transform/` | Stage-level transformations and the `*Pipeline` objects that compose them — the canonical compile surface |
| `analysis/` | Kirin abstract-interpretation analyses (atom state, layout, placement) |
| `arch/` | `ArchSpec` data, geometry, path finding, move metrics; bundled Gemini specs live under `arch/gemini*` as *data* |
| `heuristics/` | Pluggable layout heuristics and placement/movement search strategies |
| `bytecode/` | Rust-backed bytecode bindings (`_native`) and thin Python wrappers |
| `validation/`, `visualize/` | Validation helpers and debugging/visualization tooling |
| `types.py`, `utils.py`, `noise_model.py` | Cross-cutting core types and helpers |

`metrics.py` (`Metrics`) is a known outlier — it should become an analysis pass under
`analysis/`. Do not grow it; add new measurement code as an analysis.

## Canonical API

- Compiling a squin kernel to move IR: `bloqade.lanes.transform.{PhysicalPipeline,
  LogicalPipeline}`. Do **not** add another compile wrapper or fork a pipeline — extend
  these, or add a stage under `transform/`.
- Gemini-level entry points (stim programs, task compilation, device artifacts) belong in
  `bloqade.gemini.compile`, not here.
