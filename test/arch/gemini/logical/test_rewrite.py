from bloqade.test_utils import assert_nodes
from kirin import ir, rewrite
from kirin.dialects import ilist, py

from bloqade.lanes.arch.gemini.logical.rewrite import RewriteMoves
from bloqade.lanes.arch.gemini.logical.stmts import SiteBusMove
from bloqade.lanes.dialects import move
from bloqade.lanes.layout.encoding import (
    Direction,
    SiteLaneAddress,
)


def test_logical_architecture_rewrite_site():

    test_block = ir.Block()

    test_block.stmts.append(
        move.Move(
            lanes=(
                SiteLaneAddress(Direction.FORWARD, 0, 0, 0),
                SiteLaneAddress(Direction.FORWARD, 0, 2, 0),
                SiteLaneAddress(Direction.FORWARD, 0, 4, 0),
                SiteLaneAddress(Direction.FORWARD, 0, 6, 0),
            )
        )
    )

    rewrite_rule = rewrite.Walk(RewriteMoves())

    rewrite_rule.rewrite(test_block)

    expected_block = ir.Block()
    expected_block.stmts.append(
        const_list := py.Constant(ilist.IList([True, True, True, True, False]))
    )
    expected_block.stmts.append(
        SiteBusMove(
            y_mask=const_list.result,
            word=0,
            bus_id=0,
            direction=Direction.FORWARD,
        )
    )
    assert_nodes(test_block, expected_block)


def test_logical_architecture_rewrite_site_no_lanes():

    test_block = ir.Block()

    test_block.stmts.append(move.Move(lanes=()))

    expected_block = ir.Block()

    rewrite.Walk(RewriteMoves()).rewrite(test_block)

    assert_nodes(test_block, expected_block)
