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

from kirin.dialects import ilist
from tests._squin_to_move_helper import squin_to_move

from bloqade import squin
from bloqade.lanes.dialects import move
from bloqade.lanes.heuristics.physical.layout import (
    PhysicalLayoutHeuristicGraphPartitionCenterOut,
)
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
