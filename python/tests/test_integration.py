import math

import pytest
from kirin.dialects import ilist

from bloqade import qubit, squin
from bloqade.gemini import logical as gemini_logical
from bloqade.lanes.arch.gemini.logical import get_arch_spec
from bloqade.lanes.arch.gemini.physical import get_arch_spec as get_physical_arch_spec
from bloqade.lanes.heuristics.physical.layout import (
    PhysicalLayoutHeuristicGraphPartitionCenterOut,
)
from bloqade.lanes.heuristics.physical.placement import PhysicalPlacementStrategy
from bloqade.lanes.logical_mvp import compile_squin_to_move
from bloqade.lanes.transform import MoveToSquinPhysical
from bloqade.lanes.upstream import squin_to_move
from bloqade.lanes.utils import check_circuit


@pytest.mark.slow
def test_logical_compilation():
    from bloqade.rewrite.passes import AggressiveUnroll

    @gemini_logical.kernel(aggressive_unroll=True)
    def main():
        reg = qubit.qalloc(5)
        squin.broadcast.u3(0.3041 * math.pi, 0.25 * math.pi, 0.0, reg)

        squin.broadcast.sqrt_x(ilist.IList([reg[0], reg[1], reg[4]]))
        squin.broadcast.cz(ilist.IList([reg[0], reg[2]]), ilist.IList([reg[1], reg[3]]))
        squin.broadcast.sqrt_y(ilist.IList([reg[0], reg[3]]))
        squin.broadcast.cz(ilist.IList([reg[0], reg[3]]), ilist.IList([reg[2], reg[4]]))
        squin.sqrt_x_adj(reg[0])
        squin.broadcast.cz(ilist.IList([reg[0], reg[1]]), ilist.IList([reg[4], reg[3]]))
        squin.broadcast.sqrt_y_adj(reg)

    logical_move = compile_squin_to_move(main, no_raise=False)
    decompiled_squin = MoveToSquinPhysical(get_arch_spec()).emit(logical_move)

    AggressiveUnroll(main.dialects).fixpoint(main)

    assert check_circuit(main, decompiled_squin)


@pytest.mark.slow
def test_ghz_move_to_squin_roundtrip_state_vector():
    @squin.kernel(typeinfer=True, fold=True)
    def ghz():
        reg = squin.qalloc(7)
        squin.h(reg[0])
        for i in range(1, len(reg)):
            squin.cx(reg[0], reg[i])

    # GHZ is a state-prep circuit with no terminal measurement; use the
    # lower-level squin_to_move API which does not enforce that requirement.
    physical_move = squin_to_move(
        ghz,
        PhysicalLayoutHeuristicGraphPartitionCenterOut(),
        PhysicalPlacementStrategy(),
        logical_initialize=False,
        no_raise=False,
    )
    roundtrip_squin = MoveToSquinPhysical(get_physical_arch_spec()).emit(
        physical_move, no_raise=False
    )

    assert check_circuit(ghz, roundtrip_squin)
