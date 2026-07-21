"""Pin that the canonical transform classes reproduce legacy upstream output
for the two behaviorally-divergent consumers migrated in the upstream removal
(metrics logical path, tracer physical stop-at-place path)."""

from __future__ import annotations

import bloqade.squin as squin
from kirin import ir

import bloqade.gemini as gemini
from bloqade.lanes.analysis.placement import PalindromePlacementStrategy
from bloqade.lanes.heuristics.logical.layout import LogicalLayoutHeuristic
from bloqade.lanes.heuristics.logical.placement import LogicalPlacementStrategyNoHome


def _bell_logical() -> ir.Method:
    """Bell kernel for the logical path: uses gemini.logical.kernel with terminal_measure.

    LogicalNativeToPlace requires GeminiLogicalValidation-compatible kernels (Clifford
    gates + terminal measurement), so we use the logical kernel decorator here.  This
    mirrors what metrics.py callers are expected to pass to squin_to_move.
    """

    @gemini.logical.kernel(aggressive_unroll=True)
    def kernel():
        reg = squin.qalloc(2)
        squin.h(reg[0])
        squin.cx(reg[0], reg[1])
        gemini.logical.terminal_measure(reg)

    return kernel


def _bell_physical() -> ir.Method:
    """Bell kernel for the physical path: plain @squin.kernel, no terminal_measure.

    NativeToPlace(logical_initialize=False) / PhysicalNativeToPlace accept physical
    kernels without the gemini.logical wrapper.  This mirrors what the tracer passes
    to upstream.NativeToPlace.
    """

    @squin.kernel
    def kernel():
        reg = squin.qalloc(2)
        squin.h(reg[0])
        squin.cx(reg[0], reg[1])

    return kernel


def _stmt_signature(mt: ir.Method) -> list[str]:
    return [type(s).__name__ for s in mt.callable_region.walk()]


def test_metrics_logical_path_matches_upstream() -> None:
    """metrics.py does squin_to_move(logical) then transversal_rewrites().

    The canonical LogicalNativeToPlace + SequentialPlacePass + PlaceToMove +
    transversal_rewrites must produce the same statement sequence.

    Legacy path (metrics._compile_to_noisy_physical_squin):
        squin_to_move(mt, layout_heuristic=..., placement_strategy=...)
        → NativeToPlace(logical_initialize=True).emit(mt)  (includes SequentialPlacePass)
        → upstream.PlaceToMove(logical_initialize=True).emit(place_mt)
        transversal_rewrites(move_mt)

    Canonical path (what metrics migrates to):
        LogicalNativeToPlace().emit(mt)
        SequentialPlacePass(dialects)(place_mt)
        PlaceToMove(insert_initialize=True, ...).emit(place_mt)
        transversal_rewrites(move_mt)
    """
    from bloqade.lanes.passes import SequentialPlacePass
    from bloqade.lanes.transform import (
        LogicalNativeToPlace,
        PlaceToMove,
        transversal_rewrites,
    )
    from bloqade.lanes.upstream import squin_to_move

    strategy = PalindromePlacementStrategy(inner=LogicalPlacementStrategyNoHome())
    heuristic = LogicalLayoutHeuristic()

    # Legacy path
    legacy = squin_to_move(
        _bell_logical(),
        layout_heuristic=heuristic,
        placement_strategy=strategy,
    )
    legacy = transversal_rewrites(legacy)
    legacy_sig = _stmt_signature(legacy)

    # Canonical path
    place = LogicalNativeToPlace().emit(_bell_logical())
    SequentialPlacePass(place.dialects)(place)
    canonical = PlaceToMove(
        layout_heuristic=heuristic,
        placement_strategy=strategy,
        insert_initialize=True,
    ).emit(place)
    canonical = transversal_rewrites(canonical)
    canonical_sig = _stmt_signature(canonical)

    assert legacy_sig == canonical_sig, (
        f"Statement sequences differ.\n"
        f"Legacy  ({len(legacy_sig)} stmts): {legacy_sig}\n"
        f"Canonical ({len(canonical_sig)} stmts): {canonical_sig}"
    )


def test_tracer_neutral_place_stage_matches_upstream() -> None:
    """tracer.py uses upstream.NativeToPlace(logical_initialize=False).emit for
    a raw physical place stage. Pin that the new generic NativeToPlace +
    SequentialPlacePass reproduces the same place-dialect statement sequence for
    a physical kernel.

    Legacy: NativeToPlace(logical_initialize=False).emit(kernel, no_raise=False)
        → lowers qubits with InitializeNewQubits (→ NewLogicalQubit)
        → runs SequentialPlacePass internally

    Canonical: NativeToPlace().emit(kernel, no_raise=False)
        → lowers qubits with InitializeNewQubits (→ NewLogicalQubit)
        → caller runs SequentialPlacePass(dialects, no_raise=False)(place)
    """
    from bloqade.lanes.passes import SequentialPlacePass
    from bloqade.lanes.transform import NativeToPlace
    from bloqade.lanes.upstream import NativeToPlace as LegacyNativeToPlace

    legacy = LegacyNativeToPlace(logical_initialize=False).emit(
        _bell_physical(), no_raise=False
    )

    canonical = NativeToPlace().emit(_bell_physical(), no_raise=False)
    SequentialPlacePass(canonical.dialects, no_raise=False)(canonical)

    assert _stmt_signature(legacy) == _stmt_signature(canonical), (
        f"Statement sequences differ.\n"
        f"Legacy  ({len(_stmt_signature(legacy))} stmts): {_stmt_signature(legacy)}\n"
        f"Canonical ({len(_stmt_signature(canonical))} stmts): {_stmt_signature(canonical)}"
    )
