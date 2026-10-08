# Code-Block-Aware Physical Placement — Implementation Plan

Spec: `docs/superpowers/specs/2026-10-08-code-block-placement-design.md`.

Each task ends with its tests passing and is committed on its own.

1. **Types.** `analysis/code_blocks.py`: `CodeBlockTag`, `CodeBlock`,
   `LocalCodeBlock`, `CodeBlockWarning`, `CodeBlockPlacementError`, and a
   `block_shape_error` helper (one word, contiguous, `site = offset + position`).
2. **Dialect.** `dialects/code_block/` with `Register` and the `register` wrapper,
   plus a no-op concrete interpreter impl so kernels still simulate. Re-export as
   `bloqade.lanes.code_block`.
3. **IR attribute.** `_NewQubitBase.code_block: CodeBlockTag | None`.
4. **Validation and lowering.** `validation/code_block.py`
   (`CodeBlockValidation`, every error at once) and `rewrite/resolve_code_blocks.py`
   (`resolve_code_blocks`, `strip_code_blocks`). Wire them into
   `PhysicalNativeToPlace` with `use_code_blocks`, and plumb the flag through
   `PhysicalPipeline`.
5. **Layout analysis.** Collect tags in `place.layout`, then group them into
   `CodeBlock`s. Add `CodeBlockLayoutHeuristicABC`, the dispatch in
   `process_results`, the non-aware warning and the shape post-condition.
6. **Placement analysis.** Add the `ConcreteState.code_blocks` field
   (`kw_only`, `compare=False`). `PlacementAnalysis` takes `code_blocks`, builds
   the local blocks per static circuit and re-attaches them at the five strategy
   call sites. `PlaceToMove` passes the blocks through.
7. **Layout algorithm.** Implement `compute_layout_with_blocks` on
   `PhysicalLayoutHeuristicGraphPartitionCenterOut`: capacity packing, ranking,
   greedy slots, block hill climbing, and anchored assignment of unblocked
   qubits.
8. **End-to-end test.** Two Steane blocks with a transversal CX through
   `PhysicalPipeline`.
9. **Benchmark baseline.** Add `kernels/medium/code422_physical_16.py` and its
   pin-shape test. Regenerate `latest_physical.csv`, confirm only new rows were
   added, and re-run to check determinism.

Throughout: black, isort, ruff and pyright stay clean, and the existing tests
keep passing.
