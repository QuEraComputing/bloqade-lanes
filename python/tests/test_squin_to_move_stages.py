"""Tests exercising the canonical squin→move stage classes directly.

Replaces the legacy ``test_upstream.py`` which tested the removed
``bloqade.lanes.upstream.squin_to_move`` function.  These tests verify
the same behavioral contracts via the canonical stage composition:

    NativeToPlace / LogicalNativeToPlace  →  PlaceToMove

The critical behavioral assertion retained from the legacy test:
compiling with ``logical_initialize=False`` must NOT insert any
``move.LogicalInitialize`` statements into the resulting move IR.
"""

from __future__ import annotations

import math
import warnings

import pytest
from kirin.dialects import ilist
from tests._squin_to_move_helper import squin_to_move

from bloqade import squin
from bloqade.lanes.analysis.placement.exceptions import PlacementError
from bloqade.lanes.analysis.placement.strategy import PalindromePlacementStrategy
from bloqade.lanes.arch.gemini import physical
from bloqade.lanes.dialects import move, place
from bloqade.lanes.heuristics.physical.layout import (
    PhysicalLayoutHeuristicGraphPartitionCenterOut,
)
from bloqade.lanes.heuristics.physical.movement import RustPlacementTraversal
from bloqade.lanes.heuristics.physical.placement import PhysicalPlacementStrategy


def test_physical_compile_no_logical_initialize() -> None:
    """Compiling with logical_initialize=False must not insert LogicalInitialize."""

    @squin.kernel
    def main():
        reg = squin.qalloc(5)
        squin.broadcast.u3(0.3041 * math.pi, 0.25 * math.pi, 0.0, reg)

        squin.broadcast.sqrt_x(ilist.IList([reg[0], reg[1], reg[4]]))
        squin.broadcast.cz(ilist.IList([reg[0], reg[2]]), ilist.IList([reg[1], reg[3]]))
        squin.broadcast.sqrt_y(ilist.IList([reg[0], reg[3]]))
        squin.broadcast.cz(ilist.IList([reg[0], reg[3]]), ilist.IList([reg[2], reg[4]]))
        squin.sqrt_x_adj(reg[0])
        squin.broadcast.cz(ilist.IList([reg[0], reg[1]]), ilist.IList([reg[4], reg[3]]))
        squin.broadcast.sqrt_y_adj(reg)

    move_main = squin_to_move(
        main,
        layout_heuristic=PhysicalLayoutHeuristicGraphPartitionCenterOut(),
        placement_strategy=PhysicalPlacementStrategy(),
        logical_initialize=False,
    )

    # With logical_initialize=False we never insert LogicalInitialize.
    assert all(
        not isinstance(stmt, move.LogicalInitialize)
        for stmt in move_main.callable_region.walk()
    ), "logical_initialize=False must not produce any move.LogicalInitialize statements"


def test_physical_compile_produces_move_ir() -> None:
    """The canonical stage composition produces valid move-dialect IR."""

    @squin.kernel
    def kernel():
        reg = squin.qalloc(3)
        squin.h(reg[0])
        squin.cx(reg[0], reg[1])
        squin.cx(reg[0], reg[2])

    move_kernel = squin_to_move(
        kernel,
        layout_heuristic=PhysicalLayoutHeuristicGraphPartitionCenterOut(),
        placement_strategy=PhysicalPlacementStrategy(),
        logical_initialize=False,
    )

    # The output is a valid ir.Method and contains at least one move statement.
    stmts = list(move_kernel.callable_region.walk())
    assert any(
        isinstance(s, move.StatefulStatement) for s in stmts
    ), "Expected at least one move.StatefulStatement in compiled move IR"


def test_a_placement_out_of_budget_warns_with_the_solver_status() -> None:
    """A CZ stage whose routing search runs out of budget leaves its gates
    unlowered, exactly as before, but now says why: the placement error and the
    solver's ``BUDGET_EXCEEDED`` status reach the caller as a warning instead of
    being discarded with the exception."""

    @squin.kernel
    def ghz():
        q = squin.qalloc(4)
        squin.h(q[0])
        squin.cx(q[0], q[1])
        squin.cx(q[1], q[2])
        squin.cx(q[2], q[3])

    def compile_with(max_expansions: int):
        strategy = PalindromePlacementStrategy(
            inner=PhysicalPlacementStrategy(
                arch_spec=physical.get_arch_spec(),
                traversal=RustPlacementTraversal(
                    strategy="cascade-ids", max_expansions=max_expansions
                ),
            )
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = squin_to_move(
                ghz,
                layout_heuristic=PhysicalLayoutHeuristicGraphPartitionCenterOut(
                    arch_spec=physical.get_arch_spec()
                ),
                placement_strategy=strategy,
                logical_initialize=False,
            )
        placement_warnings = [
            str(w.message) for w in caught if "placement analysis" in str(w.message)
        ]
        remaining = sum(
            isinstance(stmt, place.CZ) for stmt in out.callable_region.walk()
        )
        return remaining, placement_warnings

    # Enough budget: every CZ lowers and nothing is reported.
    assert compile_with(300) == (0, [])

    remaining, reported = compile_with(1)
    assert remaining == 3, "the failed stage must still leave its gates unlowered"
    assert len(reported) == 1
    assert "routing solver failed" in reported[0]
    assert "BUDGET_EXCEEDED" in reported[0]


def test_a_failed_placement_is_still_raised_when_asked() -> None:
    """``no_raise=False`` keeps raising the placement error itself."""

    @squin.kernel
    def ghz():
        q = squin.qalloc(4)
        squin.h(q[0])
        squin.cx(q[0], q[1])
        squin.cx(q[1], q[2])
        squin.cx(q[2], q[3])

    strategy = PalindromePlacementStrategy(
        inner=PhysicalPlacementStrategy(
            arch_spec=physical.get_arch_spec(),
            traversal=RustPlacementTraversal(strategy="cascade-ids", max_expansions=1),
        )
    )
    with pytest.raises(PlacementError, match="BUDGET_EXCEEDED"):
        squin_to_move(
            ghz,
            layout_heuristic=PhysicalLayoutHeuristicGraphPartitionCenterOut(
                arch_spec=physical.get_arch_spec()
            ),
            placement_strategy=strategy,
            logical_initialize=False,
            no_raise=False,
        )
