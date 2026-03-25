import math

from kirin.dialects import ilist

from bloqade import squin
from bloqade.lanes.compile import squin_to_move
from bloqade.lanes.dialects import move
from bloqade.lanes.heuristics.physical_layout import (
    PhysicalLayoutHeuristicGraphPartitionCenterOut,
)
from bloqade.lanes.heuristics.physical_placement import PhysicalGreedyPlacementStrategy


def test_physical_compile():
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
        PhysicalLayoutHeuristicGraphPartitionCenterOut(),
        PhysicalGreedyPlacementStrategy(),
        logical_initialize=False,
    )
    # with no logical_initialization=False we never insert LogicalInitialize
    assert all(
        not isinstance(stmt, move.LogicalInitialize)
        for stmt in move_main.callable_region.walk()
    )
