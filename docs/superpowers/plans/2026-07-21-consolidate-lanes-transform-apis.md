# Consolidate & dedupe `bloqade.lanes` transform APIs — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse the two overlapping squin→move implementations into one canonical `bloqade.lanes.transform` package, relocate the Gemini-level compile entry points into `bloqade.gemini`, consolidate the prelude, and empty the `lanes` top-level re-exports.

**Architecture:** The template-method `pipeline/` stage classes + `PhysicalPipeline`/`LogicalPipeline` become the single canonical implementation; the legacy `upstream.py` trio is deleted and its consumers migrated. Gemini-level orchestration (`compile_task`, `compile_to_stim_program`, Steane defaults, cudaq) moves up into `bloqade.gemini.compile`, importing downward from `lanes.transform`. This is a **behavior-preserving refactor** — the existing pytest suite is the primary regression gate.

**Tech Stack:** Python ≥3.10, Kirin IR framework, `uv`, `pytest`/`coverage`, `just`. Lint/type: black (line-length 88), isort (profile black), ruff (py312), pyright.

## Global Constraints

- Absolute imports from the `bloqade.lanes` / `bloqade.gemini` namespace; snake_case files, PascalCase classes; extensive type annotations (pyright-enforced).
- **Do not introduce new `lanes → gemini` imports.** Existing Tier-2 edges (`circuit2place.py`, `dialects/{move,place}.py`, pipeline validations) are tolerated and out of scope; relocations must only ever add `gemini → lanes` (downward) edges.
- Every task ends green: `uv run coverage run -m pytest python/tests` (alias: `just test-python`) must pass. Fast single-file runs use `uv run pytest <path> -v`.
- Behavior-preserving: a migration that changes compiled IR output must be caught by a test and explicitly justified, not silently accepted.
- Commit after every task (Conventional Commits). Commit message footer:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- Branch: `refactor/consolidate-transforms-801` (already created).
- Full spec: `docs/superpowers/specs/2026-07-21-consolidate-lanes-transform-apis-design.md`.

---

## File Structure (end state)

```
python/bloqade/lanes/
  transform/                 # NEW package (was transform.py module)
    __init__.py              # re-exports canonical set + noise ABCs
    native_to_place.py       # NativeToPlaceBase + PhysicalNativeToPlace, LogicalNativeToPlace  (was pipeline/base+physical+logical)
    place_to_move.py         # PlaceToMove                                                        (was pipeline/base._PlaceToMove)
    pipeline.py              # PhysicalPipeline, LogicalPipeline, transversal_rewrites            (was pipeline/physical+logical)
    move_to_squin.py         # MoveToSquin{Base,Logical,Physical}                                 (was transform.py)
    move_to_stack.py         # MoveToStackMove                                                    (was transform.py)
  prelude.py                 # promoted from _prelude.py (structural); old basic prelude deleted
  __init__.py                # emptied of re-exports
  # DELETED: upstream.py, pipeline/ (package), compile.py, logical_mvp.py,
  #          _prelude.py, steane_defaults.py, cudaq_integration.py

python/bloqade/gemini/
  compile/                   # NEW package
    __init__.py              # re-exports compile_task, compile_to_stim_program
    task.py                  # compile_task + append_measurements_and_annotations +
                             #   run_squin_kernel_validation + _find_qubit_ssas/_find_return_stmt/_insert_before
    stim.py                  # compile_to_stim_program
  steane_defaults.py         # moved verbatim from lanes/
  cudaq.py                   # moved from lanes/cudaq_integration.py
```

---

## Task 1: Prelude consolidation (issue WS3)

**Files:**
- Delete: `python/bloqade/lanes/prelude.py`
- Rename: `python/bloqade/lanes/_prelude.py` → `python/bloqade/lanes/prelude.py`
- Modify: `python/bloqade/lanes/rewrite/transversal.py:11`
- Modify importers of `_prelude`: `python/tests/test_stack_move_e2e.py:9`, `python/tests/rewrite/test_measure_lower.py:6`, `python/tests/rewrite/move2squin/test_gate.py:10`, `python/tests/rewrite/move2squin/test_base.py:5`, `python/tests/analysis/atom/test_atom_interpreter.py:6`, `python/tests/benchmarks/test_runner.py:11`, `python/tests/validation/test_address.py:6`, `demo/move_lang.py:1`, `demo/community_call_jan_28_2026.py:27`
- Test: existing `python/tests/rewrite/*` and `python/tests/pipeline/*` (transversal-dependent)

**Interfaces:**
- Produces: `bloqade.lanes.prelude.kernel` (structural-based dialect group; replaces both the old `_prelude.kernel` and the deleted `prelude.kernel`).

- [ ] **Step 1: Delete the legacy basic prelude and promote the structural one**

```bash
cd /Users/pweinberg/Documents/Work/compiler_dev/bloqade-lanes
git rm python/bloqade/lanes/prelude.py
git mv python/bloqade/lanes/_prelude.py python/bloqade/lanes/prelude.py
```

- [ ] **Step 2: Repoint `rewrite/transversal.py` to the promoted prelude**

In `python/bloqade/lanes/rewrite/transversal.py` change line 11 from:

```python
from bloqade.lanes.prelude import kernel
```

(no textual change to the import line — it already reads `bloqade.lanes.prelude`; after the rename it now resolves to the structural kernel). Verify the file still imports `kernel` from `bloqade.lanes.prelude`. No edit needed beyond confirming; the behavior change is `basic` → `structural`.

- [ ] **Step 3: Repoint all `_prelude` importers to `prelude`**

Apply this exact replacement in each listed file:

```
from bloqade.lanes._prelude import kernel        ->  from bloqade.lanes.prelude import kernel
from bloqade.lanes._prelude import kernel as X   ->  from bloqade.lanes.prelude import kernel as X   (preserve the alias: move_kernel, lanes_kernel)
```

Files: `python/tests/test_stack_move_e2e.py`, `python/tests/rewrite/test_measure_lower.py`, `python/tests/rewrite/move2squin/test_gate.py`, `python/tests/rewrite/move2squin/test_base.py`, `python/tests/analysis/atom/test_atom_interpreter.py`, `python/tests/benchmarks/test_runner.py` (alias `move_kernel`), `python/tests/validation/test_address.py` (alias `lanes_kernel`), `demo/move_lang.py`, `demo/community_call_jan_28_2026.py`.

- [ ] **Step 4: Grep to confirm no `_prelude` references remain**

Run: `grep -rn "_prelude" python/ demo/`
Expected: only matches inside `docs/superpowers/plans/2026-04-21-stack-move-dialect.md` (historical doc, leave as-is). No matches under `python/` or `demo/`.

- [ ] **Step 5: Run the transversal-dependent tests**

Run: `uv run pytest python/tests/rewrite python/tests/pipeline python/tests/analysis/atom/test_atom_interpreter.py python/tests/validation/test_address.py python/tests/test_stack_move_e2e.py -v`
Expected: PASS. (If `transversal.py` breaks under `structural`, that is the flagged risk — investigate before proceeding; do not weaken the change.)

- [ ] **Step 6: Full suite + commit**

Run: `uv run coverage run -m pytest python/tests`
Expected: PASS.

```bash
git add -A
git commit -m "refactor(lanes): consolidate prelude onto structural, remove legacy basic prelude

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Turn `transform.py` into a package; split out move→squin / move→stack

**Files:**
- Delete: `python/bloqade/lanes/transform.py`
- Create: `python/bloqade/lanes/transform/__init__.py`
- Create: `python/bloqade/lanes/transform/move_to_squin.py`
- Create: `python/bloqade/lanes/transform/move_to_stack.py`
- Test: `python/tests/test_transform_move_to_stack_move.py`, `python/tests/test_noise_init_generator.py`, `python/tests/test_integration.py`

**Interfaces:**
- Produces (all importable as `from bloqade.lanes.transform import X`): `MoveToSquinBase`, `MoveToSquinLogical`, `MoveToSquinPhysical`, `MoveToStackMove`, `InitKernel`, and the re-exported noise ABCs `LogicalInitKernel`, `LogicalNoiseModelABC`, `NoiseModelABC`, `SimpleLogicalNoiseModel`, `SimpleNoiseModel` (preserving `noise_model.py:6`'s import).

- [ ] **Step 1: Create `move_to_squin.py` with the MoveToSquin classes**

```bash
mkdir -p python/bloqade/lanes/transform
```

Create `python/bloqade/lanes/transform/move_to_squin.py` containing the current `transform.py` lines 1–169 (the imports block, `InitKernel`, `MoveToSquinBase`, `MoveToSquinLogical`, `MoveToSquinPhysical`) verbatim, EXCEPT drop the `MoveToStackMove`-only imports. The import header for this file is exactly:

```python
import abc
from dataclasses import dataclass, field

from bloqade.squin.rewrite import SquinU3ToClifford
from bloqade.rewrite.passes import aggressive_unroll as agg
from kirin import ir, rewrite
from kirin.dialects import scf
from kirin.passes import TypeInfer

from bloqade import squin
from bloqade.lanes.analysis import atom
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.dialects import move
from bloqade.lanes.rewrite import move2squin
from bloqade.lanes.rewrite.move2squin import (
    LogicalInitKernel as LogicalInitKernel,
    LogicalNoiseModelABC as LogicalNoiseModelABC,
    NoiseModelABC as NoiseModelABC,
    SimpleLogicalNoiseModel as SimpleLogicalNoiseModel,
    SimpleNoiseModel as SimpleNoiseModel,
)

InitKernel = LogicalInitKernel | None
```

Then the three classes (`MoveToSquinBase`, `MoveToSquinLogical`, `MoveToSquinPhysical`) exactly as in the original `transform.py`.

- [ ] **Step 2: Create `move_to_stack.py` with MoveToStackMove**

Create `python/bloqade/lanes/transform/move_to_stack.py` containing the current `transform.py` `MoveToStackMove` class (lines 172–251) with this import header:

```python
from dataclasses import dataclass

from kirin import ir, rewrite
from kirin.passes import TypeInfer  # only if referenced; MoveToStackMove uses rewrite + stackify

from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode import Program
from bloqade.lanes.bytecode.encode import dump_program
from bloqade.lanes.dialects import move, stack_move
from bloqade.lanes.rewrite.move2stack_move import RewriteMoveToStackMove
from bloqade.lanes.rewrite.stackify import stackify
from bloqade.lanes.utils import statements_outside_dialect_group
```

(Remove the `TypeInfer` line if pyright/ruff flags it unused — `MoveToStackMove` does not call `TypeInfer`.)

- [ ] **Step 3: Create the package `__init__.py` re-exporting the canonical set so far**

Create `python/bloqade/lanes/transform/__init__.py`:

```python
from bloqade.lanes.transform.move_to_squin import (
    InitKernel as InitKernel,
    LogicalInitKernel as LogicalInitKernel,
    LogicalNoiseModelABC as LogicalNoiseModelABC,
    MoveToSquinBase as MoveToSquinBase,
    MoveToSquinLogical as MoveToSquinLogical,
    MoveToSquinPhysical as MoveToSquinPhysical,
    NoiseModelABC as NoiseModelABC,
    SimpleLogicalNoiseModel as SimpleLogicalNoiseModel,
    SimpleNoiseModel as SimpleNoiseModel,
)
from bloqade.lanes.transform.move_to_stack import MoveToStackMove as MoveToStackMove
```

- [ ] **Step 4: Delete the old module**

```bash
git rm python/bloqade/lanes/transform.py
```

- [ ] **Step 5: Run the transform-facing tests**

Run: `uv run pytest python/tests/test_transform_move_to_stack_move.py python/tests/test_noise_init_generator.py -v`
Expected: PASS (imports `from bloqade.lanes.transform import MoveToStackMove / MoveToSquinLogical` still resolve via `__init__`).

- [ ] **Step 6: Full suite + commit**

Run: `uv run coverage run -m pytest python/tests`
Expected: PASS.

```bash
git add -A
git commit -m "refactor(lanes): split transform.py into transform/ package (move->squin, move->stack)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Absorb `pipeline/` into `transform/`; delete `pipeline/`; migrate importers

**Files:**
- Create: `python/bloqade/lanes/transform/native_to_place.py`, `python/bloqade/lanes/transform/place_to_move.py`, `python/bloqade/lanes/transform/pipeline.py`
- Modify: `python/bloqade/lanes/transform/__init__.py`
- Delete: `python/bloqade/lanes/pipeline/` (whole package: `__init__.py`, `base.py`, `physical.py`, `logical.py`)
- Modify importers of `bloqade.lanes.pipeline`: `python/bloqade/lanes/logical_mvp.py:27-28`, `python/bloqade/lanes/compile.py:14`, `python/bloqade/gemini/device/physical_simulator.py:416`, `python/tests/pipeline/test_logical_pipeline.py:11-12,84`, `python/tests/pipeline/test_physical_pipeline.py:12,100`, `python/tests/pipeline/test_measure_permutation.py:20`, `python/tests/gemini/test_physical_simulator.py:22,75`, `python/tests/test_compile_api_split.py:39,46`, `demo/steane_demo.py:15`
- Modify: `python/bloqade/lanes/metrics.py:11` (`transversal_rewrites` import)

**Interfaces:**
- Produces (importable as `from bloqade.lanes.transform import X`): `PhysicalPipeline`, `LogicalPipeline`, `transversal_rewrites`, and the (now public) stage classes `NativeToPlaceBase`, `PhysicalNativeToPlace`, `LogicalNativeToPlace`, `PlaceToMove`.
- Naming decision: promote the `_`-prefixed stage classes to public names (drop the leading underscore) since Task 5/7 consumers (tracer) import them directly. Keep the class bodies otherwise identical.

- [ ] **Step 1: Create `place_to_move.py`**

Create `python/bloqade/lanes/transform/place_to_move.py` = current `pipeline/base.py` restricted to `_PlaceToMove`, renamed to public `PlaceToMove`. Import header:

```python
from __future__ import annotations

from dataclasses import dataclass

from bloqade.analysis import address
from kirin import passes, rewrite
from kirin.ir.method import Method
from kirin.rewrite.abc import RewriteRule

from bloqade.lanes.analysis import layout, placement
from bloqade.lanes.dialects import move
from bloqade.lanes.rewrite import place2move, resolve_pinned, state
```

Body: the `_PlaceToMove` dataclass from `pipeline/base.py:112-201`, renamed `class PlaceToMove:`.

- [ ] **Step 2: Create `native_to_place.py`**

Create `python/bloqade/lanes/transform/native_to_place.py` combining `pipeline/base.py._NativeToPlaceBase` (renamed `NativeToPlaceBase`) with the two subclasses from `pipeline/physical.py._PhysicalNativeToPlace` (→ `PhysicalNativeToPlace`) and `pipeline/logical.py._LogicalNativeToPlace` (→ `LogicalNativeToPlace`). Import header (union of the three files' native-stage imports):

```python
from __future__ import annotations

from dataclasses import dataclass, field

import bloqade.qubit as squin_qubit
from bloqade.analysis.validation.simple_nocloning import FlatKernelNoCloningValidation
from bloqade.native.dialects import gate as native_gate
from bloqade.native.upstream.squin2native import SquinToNative
from bloqade.rewrite.passes import AggressiveUnroll
from bloqade.rewrite.passes.callgraph import CallGraphPass
from bloqade.squin.rewrite.non_clifford_to_U3 import RewriteNonCliffordToU3
from kirin import ir, passes, rewrite
from kirin.dialects.scf import scf2cf
from kirin.ir.exception import ValidationErrorGroup
from kirin.ir.method import Method
from kirin.rewrite.abc import RewriteRule
from kirin.validation import ValidationSuite

from bloqade.gemini.common.dialects import qubit as gemini_qubit
from bloqade.gemini.common.validation.duplicate_address import DuplicateAddressValidation
from bloqade.gemini.common.validation.terminal_measure import (
    PhysicalTerminalMeasurementValidation,
)
from bloqade.gemini.logical.rewrite.initialize import _RewriteU3ToInitialize
from bloqade.gemini.logical.rewrite.steane_transversal import (
    RewriteSteaneTransversalCliffordAdjoints,
)
from bloqade.gemini.logical.validation.clifford.analysis import GeminiLogicalValidation
from bloqade.gemini.logical.validation.measurement.analysis import (
    GeminiTerminalMeasurementValidation,
)
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.dialects import place
from bloqade.lanes.rewrite import circuit2place
from bloqade.lanes.validation.address import Validation as AddressValidation
```

Body: `NativeToPlaceBase` (from `pipeline/base.py:27-109`), then `PhysicalNativeToPlace` (from `pipeline/physical.py:26-38`), then `LogicalNativeToPlace` (from `pipeline/logical.py:58-94`). Update the subclass base-class references to `NativeToPlaceBase`. (These retain the tolerated `gemini` imports — Goal 2 is out of scope.)

- [ ] **Step 3: Create `pipeline.py`**

Create `python/bloqade/lanes/transform/pipeline.py` = `PhysicalPipeline` (from `pipeline/physical.py:41-114`) + `LogicalPipeline` + `transversal_rewrites` (from `pipeline/logical.py:35-175`). Import header:

```python
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

from kirin import passes
from kirin.ir.method import Method

from bloqade.lanes.analysis import layout, placement
from bloqade.lanes.analysis.placement import PalindromePlacementStrategy
from bloqade.lanes.arch.gemini.logical import get_arch_spec as get_logical_arch_spec
from bloqade.lanes.arch.gemini.logical.upstream import steane7_transversal_map
from bloqade.lanes.arch.gemini.physical import get_arch_spec as get_physical_arch_spec
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.heuristics.logical.layout import LogicalLayoutHeuristic
from bloqade.lanes.heuristics.logical.placement import LogicalPlacementStrategyNoHome
from bloqade.lanes.heuristics.physical.layout import (
    PhysicalLayoutHeuristicGraphPartitionCenterOut,
)
from bloqade.lanes.heuristics.physical.movement import make_physical_placement_strategy
from bloqade.lanes.passes import SequentialPlacePass, TransversalRewritePass
from bloqade.lanes.transform.native_to_place import (
    LogicalNativeToPlace,
    PhysicalNativeToPlace,
)
from bloqade.lanes.transform.place_to_move import PlaceToMove
```

Body: `transversal_rewrites`, `PhysicalPipeline`, `LogicalPipeline`. Inside their `emit` methods replace `_PhysicalNativeToPlace`/`_LogicalNativeToPlace`/`_PlaceToMove` with the public names `PhysicalNativeToPlace`/`LogicalNativeToPlace`/`PlaceToMove`.

- [ ] **Step 4: Extend the package `__init__.py`**

Append to `python/bloqade/lanes/transform/__init__.py`:

```python
from bloqade.lanes.transform.native_to_place import (
    LogicalNativeToPlace as LogicalNativeToPlace,
    NativeToPlaceBase as NativeToPlaceBase,
    PhysicalNativeToPlace as PhysicalNativeToPlace,
)
from bloqade.lanes.transform.place_to_move import PlaceToMove as PlaceToMove
from bloqade.lanes.transform.pipeline import (
    LogicalPipeline as LogicalPipeline,
    PhysicalPipeline as PhysicalPipeline,
    transversal_rewrites as transversal_rewrites,
)
```

- [ ] **Step 5: Delete the `pipeline/` package**

```bash
git rm -r python/bloqade/lanes/pipeline
```

- [ ] **Step 6: Migrate `bloqade.lanes.pipeline` importers**

Apply these exact edits:

- `python/bloqade/lanes/logical_mvp.py:27-28`:
  `from bloqade.lanes.pipeline import LogicalPipeline` → `from bloqade.lanes.transform import LogicalPipeline`
  `from bloqade.lanes.pipeline.logical import transversal_rewrites` → `from bloqade.lanes.transform import transversal_rewrites`
- `python/bloqade/lanes/compile.py:14`: `from bloqade.lanes.pipeline import PhysicalPipeline` → `from bloqade.lanes.transform import PhysicalPipeline`
- `python/bloqade/lanes/metrics.py:11`: `from bloqade.lanes.logical_mvp import transversal_rewrites` → `from bloqade.lanes.transform import transversal_rewrites`
- `python/bloqade/gemini/device/physical_simulator.py:416`: `from bloqade.lanes.pipeline import PhysicalPipeline` → `from bloqade.lanes.transform import PhysicalPipeline`
- `demo/steane_demo.py:15`: `from bloqade.lanes.pipeline import PhysicalPipeline` → `from bloqade.lanes.transform import PhysicalPipeline`
- Tests — replace `bloqade.lanes.pipeline`/`bloqade.lanes.pipeline.base`/`bloqade.lanes.pipeline.logical` with `bloqade.lanes.transform` and the promoted class names:
  - `python/tests/pipeline/test_logical_pipeline.py`: `from bloqade.lanes.pipeline import LogicalPipeline` → `from bloqade.lanes.transform import LogicalPipeline`; `from bloqade.lanes.pipeline.logical import _LogicalNativeToPlace, transversal_rewrites` → `from bloqade.lanes.transform import LogicalNativeToPlace, transversal_rewrites` (and update `_LogicalNativeToPlace` usage at line 45 to `LogicalNativeToPlace`); `from bloqade.lanes.pipeline.base import _PlaceToMove` → `from bloqade.lanes.transform import PlaceToMove` (update `_PlaceToMove` usages at 87,93 to `PlaceToMove`).
  - `python/tests/pipeline/test_physical_pipeline.py`: `from bloqade.lanes.pipeline import PhysicalPipeline` → `from bloqade.lanes.transform import PhysicalPipeline`; `from bloqade.lanes.pipeline.base import _PlaceToMove` → `from bloqade.lanes.transform import PlaceToMove` (update usages at 103,110).
  - `python/tests/pipeline/test_measure_permutation.py:20`: → `from bloqade.lanes.transform import PhysicalPipeline`.
  - `python/tests/gemini/test_physical_simulator.py:22`: → `from bloqade.lanes.transform import PhysicalPipeline`; line 75 `import bloqade.lanes.pipeline as pipeline_module` → `import bloqade.lanes.transform as pipeline_module` and the `monkeypatch.setattr(pipeline_module, "PhysicalPipeline", ...)` at 105 still targets the `PhysicalPipeline` name re-exported from `transform` — but `physical_simulator.py` imports it via `from bloqade.lanes.transform import PhysicalPipeline` at call time (line 416), so patch the name where it is looked up: set `monkeypatch.setattr("bloqade.lanes.transform.pipeline.PhysicalPipeline", FakePhysicalPipeline)` instead. Verify the test's patch target matches the lookup site.
  - `python/tests/test_compile_api_split.py`: this test is retired in Task 11 — leave it for now but update its imports minimally so the suite stays green: line 39 `import bloqade.lanes.pipeline.logical as logical_module` → `import bloqade.lanes.transform.pipeline as logical_module`; line 46 `from bloqade.lanes.pipeline.logical import LogicalPipeline` → `from bloqade.lanes.transform import LogicalPipeline`; and the `monkeypatch.setattr(logical_module, "_LogicalNativeToPlace", ...)`/`"_PlaceToMove"` at 82-83 → target `"LogicalNativeToPlace"`/`"PlaceToMove"` on `logical_module` (now `transform.pipeline`, which references those names).

- [ ] **Step 7: Confirm no stale references**

Run: `grep -rn "lanes\.pipeline\|lanes import pipeline\|_LogicalNativeToPlace\|_PhysicalNativeToPlace\|_NativeToPlaceBase\|_PlaceToMove" python/ demo/`
Expected: no matches under `python/` or `demo/`.

- [ ] **Step 8: Run pipeline + physical-simulator tests, then full suite**

Run: `uv run pytest python/tests/pipeline python/tests/gemini/test_physical_simulator.py python/tests/test_compile_api_split.py -v`
Expected: PASS.
Run: `uv run coverage run -m pytest python/tests`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "refactor(lanes): fold pipeline/ into transform/ package; promote stage classes

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Characterization tests pinning the two flagged behaviors (metrics, tracer)

Locks current output of the behaviorally-divergent consumers BEFORE migrating them off `upstream.py`, so Task 5 is provably behavior-preserving.

**Files:**
- Create: `python/tests/test_upstream_migration_characterization.py`
- Reference: `python/bloqade/lanes/metrics.py`, `python/bloqade/lanes/visualize/entropy_tree/tracer.py`, `python/bloqade/lanes/upstream.py`

**Interfaces:**
- Consumes: `bloqade.lanes.upstream.squin_to_move`, `bloqade.lanes.upstream.NativeToPlace`, `bloqade.lanes.transform.transversal_rewrites`, `bloqade.lanes.transform.LogicalNativeToPlace`, `bloqade.lanes.transform.PhysicalNativeToPlace`.
- Produces: two tests asserting that the canonical stage classes reproduce the legacy `upstream` output for (a) the metrics logical path and (b) the tracer physical stop-at-place path.

- [ ] **Step 1: Write the metrics-path equivalence test**

Create `python/tests/test_upstream_migration_characterization.py`:

```python
"""Pin that the canonical transform classes reproduce legacy upstream output
for the two behaviorally-divergent consumers migrated in the upstream removal
(metrics logical path, tracer physical stop-at-place path)."""

from kirin import ir

from bloqade import squin
from bloqade.lanes.analysis.placement import PalindromePlacementStrategy
from bloqade.lanes.heuristics.logical.layout import LogicalLayoutHeuristic
from bloqade.lanes.heuristics.logical.placement import LogicalPlacementStrategyNoHome


def _bell() -> ir.Method:
    @squin.kernel
    def main():
        q = squin.qubit.new(2)
        squin.gate.h(q[0])
        squin.gate.cx(q[0], q[1])
        return

    return main


def _stmt_signature(mt: ir.Method) -> list[str]:
    return [type(s).__name__ for s in mt.callable_region.walk()]


def test_metrics_logical_path_matches_upstream():
    """metrics.py does squin_to_move(logical) then transversal_rewrites().
    The canonical LogicalNativeToPlace + PlaceToMove + transversal_rewrites
    must produce the same statement sequence."""
    from bloqade.lanes.transform import (
        LogicalNativeToPlace,
        PlaceToMove,
        transversal_rewrites,
    )
    from bloqade.lanes.upstream import squin_to_move

    strategy = PalindromePlacementStrategy(inner=LogicalPlacementStrategyNoHome())
    heuristic = LogicalLayoutHeuristic()

    legacy = squin_to_move(
        _bell(), layout_heuristic=heuristic, placement_strategy=strategy
    )
    legacy = transversal_rewrites(legacy)

    # canonical path (what metrics migrates to)
    place = LogicalNativeToPlace().emit(_bell())
    SequentialPlacePassRun = __import__(
        "bloqade.lanes.passes", fromlist=["SequentialPlacePass"]
    ).SequentialPlacePass
    SequentialPlacePassRun(place.dialects)(place)
    canonical = PlaceToMove(
        layout_heuristic=heuristic,
        placement_strategy=strategy,
        insert_initialize=True,
    ).emit(place)
    canonical = transversal_rewrites(canonical)

    assert _stmt_signature(legacy) == _stmt_signature(canonical)
```

- [ ] **Step 2: Write the tracer stop-at-place equivalence test**

Append to the same file:

```python
def test_tracer_physical_place_stage_matches_upstream():
    """tracer.py uses upstream.NativeToPlace(logical_initialize=False).emit for
    a raw physical place stage. Pin that the canonical stage reproduces the same
    place-dialect statement sequence for a physical kernel."""
    from bloqade.lanes.transform import PhysicalNativeToPlace
    from bloqade.lanes.upstream import NativeToPlace

    legacy = NativeToPlace(logical_initialize=False).emit(_bell(), no_raise=False)
    canonical = PhysicalNativeToPlace().emit(_bell(), no_raise=False)

    assert _stmt_signature(legacy) == _stmt_signature(canonical)
```

- [ ] **Step 3: Run the characterization tests**

Run: `uv run pytest python/tests/test_upstream_migration_characterization.py -v`
Expected: Either PASS (canonical == legacy, migration is safe) or FAIL with a concrete statement-sequence diff.

**Decision gate (record the outcome in the commit message):**
- If both PASS → Task 5 migrations are pure swaps.
- If `test_metrics_logical_path_matches_upstream` FAILS → the divergence is `RewriteSteaneTransversalCliffordAdjoints` (only fired when `transversal_rewrite=True` on the native stage; `LogicalNativeToPlace()` defaults to `False`, so it should match). If it still differs, adjust the canonical invocation in Task 5 to match legacy exactly (e.g. keep applying `transversal_rewrites` as a separate step, do NOT set `transversal_rewrite=True` on the stage).
- If `test_tracer_physical_place_stage_matches_upstream` FAILS → the physical stage genuinely differs (pinned-qubit lowering vs `InitializeNewQubits`). In that case Task 5 keeps tracer on a NativeToPlace configured to match: add an explicit boolean field to `NativeToPlaceBase` / a dedicated stage rather than forcing `PhysicalNativeToPlace`. Update this test to assert the chosen equivalent.

- [ ] **Step 4: Commit (whatever the outcome — the tests document reality)**

```bash
git add python/tests/test_upstream_migration_characterization.py
git commit -m "test(lanes): characterize upstream vs canonical output before migration

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Migrate the non-test `upstream` consumers (metrics, tracer, benchmarks)

**Files:**
- Modify: `python/bloqade/lanes/metrics.py` (3 `squin_to_move` calls @ 93,133,153; import @ 15)
- Modify: `python/bloqade/lanes/visualize/entropy_tree/tracer.py` (@ 151,175,196)
- Modify: `python/benchmarks/harness/runner.py` (@ 30 import, 136,160 calls)
- Test: `python/tests/test_upstream_migration_characterization.py`, `python/tests/benchmarks/test_runner.py`, benchmark/metrics-touching tests

**Interfaces:**
- Consumes: `bloqade.lanes.transform.{LogicalNativeToPlace, PhysicalNativeToPlace, PlaceToMove, transversal_rewrites}`, `bloqade.lanes.passes.SequentialPlacePass`.

- [ ] **Step 1: Migrate `metrics.py`**

Replace the `from bloqade.lanes.upstream import squin_to_move` import (line 15) and each of the three `squin_to_move(mt, layout_heuristic=..., placement_strategy=...)` call sites with the canonical equivalent confirmed by Task 4. If Task 4 Step 3 passed, use a small local helper in `metrics.py`:

```python
from bloqade.lanes.passes import SequentialPlacePass
from bloqade.lanes.transform import LogicalNativeToPlace, PlaceToMove


def _logical_squin_to_move(mt, *, layout_heuristic, placement_strategy):
    place = LogicalNativeToPlace().emit(mt)
    SequentialPlacePass(place.dialects)(place)
    return PlaceToMove(
        layout_heuristic=layout_heuristic,
        placement_strategy=placement_strategy,
        insert_initialize=True,
    ).emit(place)
```

Replace the three `squin_to_move(...)` calls with `_logical_squin_to_move(...)`. (Behavior pinned by Task 4's metrics test.)

- [ ] **Step 2: Migrate `tracer.py`**

At line 151 change `from bloqade.lanes.upstream import NativeToPlace, squin_to_move` to import the canonical equivalents chosen in Task 4:

```python
from bloqade.lanes.passes import SequentialPlacePass
from bloqade.lanes.transform import PhysicalNativeToPlace, PlaceToMove
```

Replace the `squin_to_move(kernel, layout_heuristic=..., placement_strategy=..., no_raise=False, logical_initialize=False)` call (line 175) with:

```python
place_kernel = PhysicalNativeToPlace().emit(kernel, no_raise=False)
SequentialPlacePass(place_kernel.dialects)(place_kernel)
move_main = PlaceToMove(
    layout_heuristic=layout_heuristic,
    placement_strategy=placement_strategy,
    insert_initialize=False,
).emit(place_kernel, no_raise=False)
```

Replace the `NativeToPlace(logical_initialize=False).emit(kernel, no_raise=False)` at line 196 with `PhysicalNativeToPlace().emit(kernel, no_raise=False)`. (If Task 4's tracer test required a different equivalent, use that instead.)

- [ ] **Step 3: Migrate `benchmarks/harness/runner.py`**

Line 30 `from bloqade.lanes.upstream import squin_to_move` → `from bloqade.lanes.transform import LogicalNativeToPlace, PhysicalNativeToPlace, PlaceToMove` and add `from bloqade.lanes.passes import SequentialPlacePass`. Replace the two `squin_to_move(...)` calls (lines 136, 160) with a stage-composition helper mirroring the `logical_initialize` flag:

```python
def _squin_to_move(mt, *, layout_heuristic, placement_strategy, logical_initialize):
    NativeStage = LogicalNativeToPlace if logical_initialize else PhysicalNativeToPlace
    place = NativeStage().emit(mt)
    SequentialPlacePass(place.dialects)(place)
    return PlaceToMove(
        layout_heuristic=layout_heuristic,
        placement_strategy=placement_strategy,
        insert_initialize=logical_initialize,
    ).emit(place)
```

Note: `python/tests/benchmarks/test_runner.py` monkeypatches `benchmarks.harness.runner.squin_to_move` (lines 73,180,246) and asserts kwargs (`logical_initialize`, absence of `insert_return_moves`). Update those patches/asserts to target the new `_squin_to_move` helper name and its kwargs.

- [ ] **Step 4: Run the affected tests**

Run: `uv run pytest python/tests/test_upstream_migration_characterization.py python/tests/benchmarks/test_runner.py -v`
Expected: PASS.

- [ ] **Step 5: Full suite + commit**

Run: `uv run coverage run -m pytest python/tests`
Expected: PASS.

```bash
git add -A
git commit -m "refactor(lanes): migrate metrics/tracer/benchmarks off upstream to transform stages

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Migrate the `upstream` test + demo consumers

**Files (tests):** `python/tests/test_integration.py:18,58,84,119`, `python/tests/test_validation_squin_kernels.py:14,27`, `python/tests/test_upstream.py:11,28`, `python/tests/analysis/placement/test_execute_cz_return.py:28,159`, `python/tests/integration/test_explicit_allocation.py:21,29`, `python/tests/gemini/validation/test_e3_pipeline_validation.py:16,28`
**Files (demos):** `demo/explicit_allocation.py:17,33`, `demo/pipeline_details.py:52,54,135,137`

**Interfaces:**
- Consumes: the same canonical stage-composition helper pattern as Task 5. These tests exercise `squin_to_move`'s explicit-heuristic + `logical_initialize` behavior.

- [ ] **Step 1: Add a shared test helper for stage composition**

Create `python/tests/_squin_to_move_helper.py`:

```python
"""Test-only replacement for the removed bloqade.lanes.upstream.squin_to_move,
composing the canonical transform stage classes."""

from kirin import ir

from bloqade.lanes.passes import SequentialPlacePass
from bloqade.lanes.transform import (
    LogicalNativeToPlace,
    PhysicalNativeToPlace,
    PlaceToMove,
)


def squin_to_move(
    mt: ir.Method,
    *,
    layout_heuristic,
    placement_strategy,
    no_raise: bool = True,
    logical_initialize: bool = True,
) -> ir.Method:
    NativeStage = LogicalNativeToPlace if logical_initialize else PhysicalNativeToPlace
    place = NativeStage().emit(mt, no_raise=no_raise)
    SequentialPlacePass(place.dialects, no_raise=no_raise)(place)
    return PlaceToMove(
        layout_heuristic=layout_heuristic,
        placement_strategy=placement_strategy,
        insert_initialize=logical_initialize,
    ).emit(place, no_raise=no_raise)
```

- [ ] **Step 2: Repoint the test imports**

In each test file listed above replace `from bloqade.lanes.upstream import squin_to_move` with `from tests._squin_to_move_helper import squin_to_move` (adjust to the project's test import root; if tests import as top-level `python/tests`, use `from _squin_to_move_helper import squin_to_move` matching the existing sibling-import style in that file). Keep the call sites unchanged if they already pass `layout_heuristic=`/`placement_strategy=` as keywords; if any pass positionally, convert to keywords.

`test_upstream.py` is specifically a test *of* `upstream`; rename it to `python/tests/test_squin_to_move_stages.py` and rewrite its body to exercise the canonical stage classes directly (the behavior it asserts — full squin→move lowering — now belongs to the stage composition).

- [ ] **Step 3: Migrate the demos**

- `demo/explicit_allocation.py:17`: `from bloqade.lanes.upstream import squin_to_move` → compose stages inline:
  ```python
  from bloqade.lanes.passes import SequentialPlacePass
  from bloqade.lanes.transform import LogicalNativeToPlace, PlaceToMove
  ```
  Replace the `squin_to_move(...)` call at line 33 with the `LogicalNativeToPlace().emit` + `SequentialPlacePass` + `PlaceToMove(...).emit` sequence (matching the demo's current heuristic/strategy args).
- `demo/pipeline_details.py`: lines 52/54 use `NativeToPlace().emit(...)` and 135/137 use `PlaceToMove(...)`. Change `from bloqade.lanes.upstream import NativeToPlace` (52) → `from bloqade.lanes.transform import LogicalNativeToPlace as NativeToPlace` and `from bloqade.lanes.upstream import PlaceToMove` (135) → `from bloqade.lanes.transform import PlaceToMove`. Confirm the demo still runs (it is narrative; `LogicalNativeToPlace` default reproduces the old `NativeToPlace()` logical default).

- [ ] **Step 4: Confirm no `upstream` references remain except the module itself**

Run: `grep -rn "lanes\.upstream\|lanes import upstream" python/ demo/`
Expected: matches only in `python/bloqade/lanes/upstream.py` (deleted next task) and `python/tests/test_upstream_migration_characterization.py` (which still imports legacy for the equivalence assertion — that test is removed in Task 7).

- [ ] **Step 5: Run affected tests + full suite**

Run: `uv run pytest python/tests/test_integration.py python/tests/test_validation_squin_kernels.py python/tests/test_squin_to_move_stages.py python/tests/analysis/placement/test_execute_cz_return.py python/tests/integration/test_explicit_allocation.py python/tests/gemini/validation/test_e3_pipeline_validation.py -v`
Expected: PASS.
Run: `uv run coverage run -m pytest python/tests`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor(lanes): migrate upstream test+demo consumers to canonical stages

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Delete `upstream.py` and the legacy-comparison characterization test

**Files:**
- Delete: `python/bloqade/lanes/upstream.py`
- Delete: `python/tests/test_upstream_migration_characterization.py` (its job — proving equivalence pre-removal — is done; the equivalence is now enforced by the migrated consumers' own tests)

- [ ] **Step 1: Delete**

```bash
git rm python/bloqade/lanes/upstream.py python/tests/test_upstream_migration_characterization.py
```

- [ ] **Step 2: Confirm nothing imports it**

Run: `grep -rn "lanes\.upstream\|lanes import upstream\|from bloqade.lanes.upstream" python/ demo/`
Expected: no matches.

- [ ] **Step 3: Full suite + commit**

Run: `uv run coverage run -m pytest python/tests`
Expected: PASS.

```bash
git add -A
git commit -m "refactor(lanes): delete legacy upstream.py (superseded by transform stages)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Relocate `steane_defaults` → `bloqade.gemini`

**Files:**
- Move: `python/bloqade/lanes/steane_defaults.py` → `python/bloqade/gemini/steane_defaults.py`
- Modify importers: `python/bloqade/lanes/logical_mvp.py:31`, `python/bloqade/lanes/__init__.py:12-15`, `python/tests/test_device.py:20`

- [ ] **Step 1: Move the module**

```bash
git mv python/bloqade/lanes/steane_defaults.py python/bloqade/gemini/steane_defaults.py
```

- [ ] **Step 2: Repoint importers**

- `python/bloqade/lanes/logical_mvp.py:31`: `from bloqade.lanes.steane_defaults import steane7_m2dets, steane7_m2obs` → `from bloqade.gemini.steane_defaults import steane7_m2dets, steane7_m2obs` (transitional — `logical_mvp` is deleted in Task 11).
- `python/tests/test_device.py:20`: `from bloqade.lanes.steane_defaults import ...` → `from bloqade.gemini.steane_defaults import ...`.
- `python/bloqade/lanes/__init__.py`: remove the `from .steane_defaults import (...)` re-export block (lines 12–15). (Full `__init__` emptying happens in Task 12; removing this block now is required because the module is gone.)

- [ ] **Step 3: Confirm + test + commit**

Run: `grep -rn "lanes\.steane_defaults" python/ demo/`
Expected: no matches.
Run: `uv run pytest python/tests/test_device.py -v && uv run coverage run -m pytest python/tests`
Expected: PASS.

```bash
git add -A
git commit -m "refactor(gemini): relocate steane_defaults from lanes to gemini

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Relocate `cudaq_integration` → `bloqade.gemini.cudaq`

**Files:**
- Move: `python/bloqade/lanes/cudaq_integration.py` → `python/bloqade/gemini/cudaq.py`
- Modify importers: `python/bloqade/lanes/logical_mvp.py:25`, `python/tests/test_cudaq_integration.py:13` (imports from `logical_mvp`, addressed in Task 10/11)

- [ ] **Step 1: Move the module**

```bash
git mv python/bloqade/lanes/cudaq_integration.py python/bloqade/gemini/cudaq.py
```

Inside `python/bloqade/gemini/cudaq.py`, the existing `from bloqade.gemini import logical` import stays (now a gemini-internal import — no longer an upward `lanes → gemini` edge).

- [ ] **Step 2: Repoint the `logical_mvp` importer (transitional)**

`python/bloqade/lanes/logical_mvp.py:25`: `from bloqade.lanes.cudaq_integration import cudaq_to_squin, is_cudaq_kernel` → `from bloqade.gemini.cudaq import cudaq_to_squin, is_cudaq_kernel`.

- [ ] **Step 3: Confirm + test + commit**

Run: `grep -rn "cudaq_integration" python/ demo/`
Expected: matches only in `python/tests/test_cudaq_integration.py` file NAME references (not imports) — the import at line 13 is from `logical_mvp`, not `cudaq_integration`, so it is unaffected here.
Run: `uv run coverage run -m pytest python/tests`
Expected: PASS.

```bash
git add -A
git commit -m "refactor(gemini): relocate cudaq integration from lanes to gemini.cudaq

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Create `bloqade.gemini.compile` (task.py + stim.py) from `logical_mvp`

**Files:**
- Create: `python/bloqade/gemini/compile/__init__.py`, `python/bloqade/gemini/compile/task.py`, `python/bloqade/gemini/compile/stim.py`
- Modify consumers to import from the new location: `python/bloqade/gemini/device/simulator.py:551`, `python/bloqade/gemini/device/physical_simulator.py:83`, and the `_find_qubit_ssas/_find_return_stmt/_insert_before` imports at `physical_simulator.py:83`

**Interfaces:**
- Produces: `bloqade.gemini.compile.compile_task(logical_kernel, m2dets=None, m2obs=None) -> tuple[ir.Method, ArchSpec, ir.Method, PostProcessing]`; `bloqade.gemini.compile.append_measurements_and_annotations(mt, m2dets, m2obs) -> None`; `bloqade.gemini.compile.run_squin_kernel_validation(mt)`; helpers `_find_qubit_ssas`, `_find_return_stmt`, `_insert_before`; `bloqade.gemini.compile.stim.compile_to_stim_program(mt, noise_model=None, no_raise=True, layout_heuristic=None) -> str`.

- [ ] **Step 1: Create `gemini/compile/task.py`**

```bash
mkdir -p python/bloqade/gemini/compile
```

Create `python/bloqade/gemini/compile/task.py` containing, from the current `logical_mvp.py`: `run_squin_kernel_validation` (46–67), `_find_qubit_ssas`/`_find_return_stmt`/`_insert_before` + `_S` TypeVar (217–252), `append_measurements_and_annotations` (255–348), `compile_task` (351–405). Import header:

```python
from functools import cache
from typing import Any, Callable, TypeVar

from bloqade.analysis.address import AddressAnalysis, AddressQubit
from bloqade.analysis.validation.simple_nocloning import FlatKernelNoCloningValidation
from bloqade.decoders.dialects.annotate.stmts import SetDetector, SetObservable
from kirin import ir
from kirin.dialects import func, ilist, py
from kirin.validation import ValidationSuite

from bloqade.gemini.compile.stim import compile_to_stim_program as compile_to_stim_program  # noqa: F401  (re-export convenience; remove if unused)
from bloqade.gemini.cudaq import cudaq_to_squin, is_cudaq_kernel
from bloqade.gemini.logical.dialects.operations.stmts import TerminalLogicalMeasurement
from bloqade.gemini.logical.validation.clifford.analysis import GeminiLogicalValidation
from bloqade.gemini.logical.validation.measurement.analysis import (
    GeminiTerminalMeasurementValidation,
)
from bloqade.gemini.steane_defaults import steane7_m2dets, steane7_m2obs
from bloqade.lanes.analysis import atom
from bloqade.lanes.arch.gemini import physical
from bloqade.lanes.transform import LogicalPipeline
```

Replace the body's `compile_squin_to_move(logical_squin_kernel, transversal_rewrite=True)` call (was `logical_mvp.py:393-395`) with the canonical pipeline:

```python
physical_move_kernel = LogicalPipeline(transversal_rewrite=True).emit(
    logical_squin_kernel
)
```

(Do NOT import the deleted `compile_to_stim_program` from task.py if it creates a cycle; keep it in stim.py only and let `__init__` re-export both. Remove the convenience re-export line above if pyright flags an unused import.)

- [ ] **Step 2: Create `gemini/compile/stim.py`**

Create `python/bloqade/gemini/compile/stim.py` with the logical `compile_to_stim_program` + its dependency `compile_to_physical_squin_noise_model` (inlined as a private helper, since the standalone wrapper is being deleted). From `logical_mvp.py` lines 144–214:

```python
import io

from kirin import ir

from bloqade.stim.emit.stim_str import EmitStimMain
from bloqade.stim.upstream.from_squin import squin_to_stim

from bloqade.lanes.arch.gemini import physical
from bloqade.lanes.noise_model import generate_logical_noise_model
from bloqade.lanes.rewrite.move2squin.noise import LogicalNoiseModelABC
from bloqade.lanes.rewrite.squin2stim import RemoveReturn
from bloqade.lanes.transform import LogicalPipeline, MoveToSquinLogical


def _to_physical_squin_noise_model(
    mt: ir.Method,
    noise_model: LogicalNoiseModelABC | None = None,
    no_raise: bool = True,
    layout_heuristic=None,
) -> ir.Method:
    if noise_model is None:
        noise_model = generate_logical_noise_model()
    move_mt = LogicalPipeline(
        transversal_rewrite=True, layout_heuristic=layout_heuristic
    ).emit(mt, no_raise=no_raise)
    return MoveToSquinLogical(
        arch_spec=physical.get_arch_spec(),
        noise_model=noise_model,
        add_noise=True,
        aggressive_unroll=False,
    ).emit(move_mt, no_raise=no_raise)


def compile_to_stim_program(
    mt: ir.Method,
    noise_model: LogicalNoiseModelABC | None = None,
    no_raise: bool = True,
    layout_heuristic=None,
) -> str:
    """Compile a logical squin kernel to a Stim program string with noise inserted."""
    noise_kernel = _to_physical_squin_noise_model(
        mt, noise_model, no_raise=no_raise, layout_heuristic=layout_heuristic
    )
    RemoveReturn().rewrite(noise_kernel.code)
    noise_kernel = squin_to_stim(noise_kernel)
    buf = io.StringIO()
    emit = EmitStimMain(dialects=noise_kernel.dialects, io=buf)
    emit.initialize()
    emit.run(node=noise_kernel)
    return buf.getvalue().strip()
```

- [ ] **Step 3: Create `gemini/compile/__init__.py`**

```python
from bloqade.gemini.compile.stim import (
    compile_to_stim_program as compile_to_stim_program,
)
from bloqade.gemini.compile.task import (
    append_measurements_and_annotations as append_measurements_and_annotations,
    compile_task as compile_task,
    run_squin_kernel_validation as run_squin_kernel_validation,
)
```

- [ ] **Step 4: Repoint the gemini device consumers**

- `python/bloqade/gemini/device/simulator.py:551`: `from bloqade.lanes.logical_mvp import compile_task` → `from bloqade.gemini.compile import compile_task`.
- `python/bloqade/gemini/device/physical_simulator.py:83`: `from bloqade.lanes.logical_mvp import (_find_qubit_ssas, _find_return_stmt, _insert_before)` → `from bloqade.gemini.compile.task import (_find_qubit_ssas, _find_return_stmt, _insert_before)`.

- [ ] **Step 5: Run gemini device tests + full suite**

Run: `uv run pytest python/tests/test_device.py python/tests/gemini -v`
Expected: PASS.
Run: `uv run coverage run -m pytest python/tests`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(gemini): add gemini.compile package (compile_task, stim) from logical_mvp

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: Delete `compile.py` + `logical_mvp.py` wrappers; migrate remaining consumers; retire split test

**Files:**
- Delete: `python/bloqade/lanes/compile.py`, `python/bloqade/lanes/logical_mvp.py`, `python/tests/test_compile_api_split.py`
- Modify consumers (tests): `python/tests/test_noise_init_generator.py:129`, `python/tests/test_integration.py:15,39`, `python/tests/analysis/atom/test_shot_remapping.py:181,193`, `python/tests/gemini/test_physical_simulator.py:21,173`, `python/tests/gemini/test_star_rz.py:18` (+call sites), `python/tests/test_device.py:645`, `python/tests/test_cudaq_integration.py:13`
- Modify consumers (benchmarks/demos): `python/benchmarks/harness/runner.py:15,27`, `demo/msd.py:8`, `demo/pipeline_demo.py:5`, `demo/pipeline_details.py:175`, `demo/community_call_jan_28_2026.py:99`, `demo/ghz_moves_demo.py:4`

**Interfaces:**
- Consumes: `bloqade.gemini.compile.{compile_task, compile_to_stim_program}`, `bloqade.gemini.compile.task._find_qubit_ssas`, `bloqade.lanes.transform.{LogicalPipeline, PhysicalPipeline, transversal_rewrites, MoveToSquin*}`, `bloqade.lanes.visualize`.

- [ ] **Step 1: Migrate `compile_squin_to_move` consumers to pipelines**

For each call to `compile_squin_to_move(mt, transversal_rewrite=..., ...)` (logical) replace with `LogicalPipeline(transversal_rewrite=..., ...).emit(mt, no_raise=...)`; for physical `compile_squin_to_move` replace with `PhysicalPipeline(...).emit(mt, ...)`. Concrete sites:
  - `python/tests/test_noise_init_generator.py:129,141`: `compile_squin_to_move(main, transversal_rewrite=True, no_raise=True)` → `from bloqade.lanes.transform import LogicalPipeline` then `LogicalPipeline(transversal_rewrite=True).emit(main, no_raise=True)`.
  - `python/tests/test_integration.py:15,39`: `compile_squin_to_move(main, no_raise=False)` → `LogicalPipeline().emit(main, no_raise=False)`.
  - `python/tests/analysis/atom/test_shot_remapping.py:181,193`: `compile_squin_to_move(main, transversal_rewrite=True)` → `LogicalPipeline(transversal_rewrite=True).emit(main)`.
  - `python/tests/gemini/test_star_rz.py:18` + call sites 123/165/194/222: replace each `compile_squin_to_move(...)` with `LogicalPipeline(...).emit(...)` preserving the `transversal_rewrite`/heuristic kwargs.

- [ ] **Step 2: Migrate `_find_qubit_ssas` + `compile_task` monkeypatch consumers**

  - `python/tests/gemini/test_physical_simulator.py:21`: `from bloqade.lanes.logical_mvp import _find_qubit_ssas` → `from bloqade.gemini.compile.task import _find_qubit_ssas`; line 173 `import bloqade.lanes.logical_mvp as logical_mvp` → `import bloqade.gemini.compile.task as logical_mvp` (keep the alias so downstream references resolve, or rename references).
  - `python/tests/test_device.py:645`: `monkeypatch.setattr("bloqade.lanes.logical_mvp.compile_task", compile_task)` → `monkeypatch.setattr("bloqade.gemini.compile.task.compile_task", compile_task)` AND, because `simulator.py` imports `compile_task` at call time from `bloqade.gemini.compile`, ensure the patch target matches the lookup site (`bloqade.gemini.compile.task.compile_task` is where `__init__` re-exports from — verify by checking `simulator.py:551` imports `from bloqade.gemini.compile import compile_task`; patch `bloqade.gemini.compile.task.compile_task`).
  - `python/tests/test_cudaq_integration.py:13`: `from bloqade.lanes.logical_mvp import (...)` → import the same names from `bloqade.gemini.compile` / `bloqade.gemini.compile.task` (e.g. `compile_task`, `append_measurements_and_annotations`).

- [ ] **Step 3: Migrate `compile_to_stim_program` consumers**

  - `demo/pipeline_demo.py:5-6,38`: `from bloqade.lanes.logical_mvp import (compile_to_stim_program, ...)` → `from bloqade.gemini.compile import compile_to_stim_program`.
  - `demo/msd.py:8-9,33`: same repoint to `bloqade.gemini.compile`.

- [ ] **Step 4: Migrate `transversal_rewrites` + `compile_squin_to_move_and_visualize` + noise-model consumers**

  - `demo/pipeline_details.py:175`, `demo/community_call_jan_28_2026.py:99`, `python/benchmarks/harness/runner.py:27`: `from bloqade.lanes.logical_mvp import transversal_rewrites` → `from bloqade.lanes.transform import transversal_rewrites`.
  - `demo/ghz_moves_demo.py:4,40-42` + `demo/msd.py:31-32`: `compile_squin_to_move_and_visualize(mt)` → inline:
    ```python
    from bloqade.lanes.transform import LogicalPipeline
    from bloqade.lanes import visualize
    from bloqade.lanes.arch.gemini import logical
    mt = LogicalPipeline().emit(log_depth_ghz)
    visualize.debugger(mt, logical.get_arch_spec(), interactive=True, atom_marker="s")
    ```
  - `demo/pipeline_demo.py:5`, `demo/pipeline_details.py`: replace any other `logical_mvp` imports (e.g. `compile_squin_to_move`) with the pipeline equivalents.
  - `python/benchmarks/harness/runner.py:15-16`: `from bloqade.lanes.compile import (compile_to_physical_squin_noise_model as compile_physical_noise_model)` → compose inline with `PhysicalPipeline` + `MoveToSquinPhysical`, or import the private `_to_physical_squin_noise_model` from `bloqade.gemini.compile.stim` if the benchmark is Gemini-scoped. Choose the composition that preserves the benchmark's current arch/noise arguments; run the benchmark test to confirm.

- [ ] **Step 5: Delete the wrapper modules and the split test**

```bash
git rm python/bloqade/lanes/compile.py python/bloqade/lanes/logical_mvp.py python/tests/test_compile_api_split.py
```

- [ ] **Step 6: Confirm no references remain**

Run: `grep -rn "lanes\.compile\b\|lanes import compile\|lanes\.logical_mvp\|lanes import logical_mvp\|compile_squin_to_move\|compile_to_physical_squin_noise_model\|compile_squin_to_move_best\|compile_squin_to_move_and_visualize" python/ demo/`
Expected: no matches (all wrappers gone; consumers on pipelines / `gemini.compile`).

- [ ] **Step 7: Full suite + demos + commit**

Run: `uv run coverage run -m pytest python/tests`
Expected: PASS.
Run: `just demo` (or run each `demo/*.py` that was touched)
Expected: demos execute without ImportError.

```bash
git add -A
git commit -m "refactor: delete compile.py/logical_mvp wrappers; consumers use pipelines + gemini.compile

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: Empty `lanes/__init__.py`; final verification

**Files:**
- Modify: `python/bloqade/lanes/__init__.py`
- Verify: whole repo

- [ ] **Step 1: Empty the top-level re-exports**

Replace the entire body of `python/bloqade/lanes/__init__.py` with a module docstring only (no `gemini.device`, `metrics`, `noise_model`, or `steane_defaults` re-exports):

```python
"""bloqade.lanes — machine-agnostic movement compilation.

Import submodules directly (e.g. ``from bloqade.lanes.transform import LogicalPipeline``).
Gemini-machine specifics live in ``bloqade.gemini``.
"""
```

- [ ] **Step 2: Find and fix any consumer relying on the removed top-level re-exports**

Run: `grep -rn "from bloqade.lanes import \(" python/ demo/ ; grep -rn "from bloqade\.lanes import [A-Za-z]" python/ demo/`
For each hit importing `Metrics`, `generate_logical_noise_model`, `generate_simple_noise_model`, `NoiseModelABC`, `DetectorResult`, `GeminiLogicalSimulator*`, `Result`, `steane7_*` directly from `bloqade.lanes`, repoint to the submodule (`bloqade.lanes.metrics`, `bloqade.lanes.noise_model`, `bloqade.lanes.rewrite.move2squin.noise`, `bloqade.gemini.device`, `bloqade.gemini.steane_defaults`).

- [ ] **Step 3: Assert the lanes→gemini edge count did not grow**

Run: `grep -rn "bloqade\.gemini" python/bloqade/lanes/ | wc -l`
Expected: strictly fewer than the pre-refactor baseline of 20 lines (the relocated `cudaq_integration`, `logical_mvp`, `steane_defaults`, and the `__init__` re-export are gone; the remaining Tier-2 edges in `dialects/`, `rewrite/circuit2place.py`, and `transform/native_to_place.py` are the tolerated, out-of-scope set). Record the new count in the commit message.

- [ ] **Step 4: Lint, type-check, full suite, demos**

Run:
```bash
uv run isort python
uv run black python
uv run ruff check python
uv run pyright python
uv run coverage run -m pytest python/tests
```
Expected: all clean / PASS. Fix any import-sorting or unused-import fallout from the moves.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor(lanes): empty __init__ re-exports; consumers import submodules directly

Closes part of #801 (Goal 1). Goal 2 (full lanes->gemini decoupling) tracked separately.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- §1 canonical transform sub-package → Tasks 2, 3, 7. ✅
- §2 gemini relocation (compile/, steane, cudaq) → Tasks 8, 9, 10. ✅
- §3 delete thin wrappers, keep orchestration → Tasks 10, 11. ✅
- §4 consumer migration map → Tasks 3, 5, 6, 10, 11. ✅
- §5 behavioral-gap rule (metrics, tracer) → Tasks 4, 5 (characterization + explicit equivalents). ✅
- §6 prelude consolidation → Task 1. ✅
- §7 `__init__` cleanup + metrics transversal repoint → Tasks 3 (metrics import), 12. ✅
- Testing strategy (retire `test_compile_api_split`, pin flagged behaviors, lint/type gates) → Tasks 4, 11, 12. ✅
- Success criteria 1–6 → covered across Tasks 2–12; criterion 6 (no new edges) explicitly checked in Task 12 Step 3. ✅

**Placeholder scan:** No "TBD/TODO/handle edge cases". The one deliberate branch (Task 4 decision gate) provides concrete both-outcome instructions rather than deferring. Benchmark noise-model composition (Task 11 Step 4) gives two concrete options with a selection criterion (preserve current args; run the benchmark test).

**Type consistency:** Stage classes are consistently named `NativeToPlaceBase`, `PhysicalNativeToPlace`, `LogicalNativeToPlace`, `PlaceToMove` from Task 3 onward (underscore-prefixed originals only referenced when describing the source). `LogicalPipeline`/`PhysicalPipeline`/`transversal_rewrites` names stable. `compile_task`/`compile_to_stim_program`/`append_measurements_and_annotations`/`run_squin_kernel_validation` stable across Tasks 10–11.

**Known follow-ups (out of scope, tracked):** Goal 2 full decoupling; `metrics.Metrics` → analysis pass; per-directory AGENT.md files (#821).
