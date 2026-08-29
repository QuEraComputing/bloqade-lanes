# AGENT.md — `bloqade.gemini`

## Purpose

Gemini-machine specifics, layered **on top of** `bloqade.lanes`. Anything that assumes the
Gemini architecture — Steane [[7,1,3]] encoding, the logical dialect, stim/noise
artifacts, decoding, device and simulator surfaces — lives here.

## Layering rule

- `gemini` **may** import `lanes` freely. That is the intended direction.
- `lanes` **must not** import `gemini`. Four legacy modules still do
  (`lanes/dialects/{place,move}.py`, `lanes/rewrite/circuit2place.py`,
  `lanes/transform/native_to_place.py`); they are tracked for removal. When you touch code
  on either side of one of those edges, move the Gemini-specific piece here rather than
  deepening it.
- Because those edges exist, `bloqade.lanes.__init__` has no eager re-exports. Import
  lanes symbols from their submodules (`from bloqade.lanes.transform import
  LogicalPipeline`), not from the package root.

## What goes where

| Folder | Holds |
| --- | --- |
| `common/` | shared across logical + physical: `dialects/` (`qubit`, `arrange`), `validation/`, `impl/` |
| `logical/` | Steane logical layer: `dialects/operations`, `rewrite/`, `validation/`, `stdlib/`, `impl/`, `group.py` |
| `physical/` | physical-level Gemini surface (`group.py`) |
| `compile/` | Gemini-level compile entry points and orchestration — see `compile/AGENT.md` |
| `device/` | device + simulator classes and task runtimes (the user-facing execution surface) |
| `decoding/` | decoders, DEM construction, sampling, postselection, tomography |
| `steane_defaults.py`, `star.py`, `cudaq.py`, `post_processing.py` | Steane detector/observable defaults, star-support validation, CUDA-Q bridge, result post-processing |

New Gemini-specific validation goes in `common/validation/` if it applies to both layers,
`logical/validation/` if it is encoding-specific.

## Canonical API

- **Compiling a kernel end-to-end**: `bloqade.gemini.compile` (`compile_task`,
  `compile_to_stim_program`). Do not add a parallel `compile_*` module elsewhere in this
  tree — the duplicated-entry-point drift this layout replaced is the failure mode to
  avoid.
- **The machine-agnostic squin→move step underneath it**:
  `bloqade.lanes.transform.{PhysicalPipeline,LogicalPipeline}`. Wrap it; don't re-implement
  it here.
- **Public execution surface**: the device/simulator classes re-exported from
  `bloqade.gemini.__init__`.
