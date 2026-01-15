from bloqade.test_utils import assert_nodes
from kirin import ir, rewrite, types
from kirin.dialects import ilist, py

from bloqade.lanes.arch.gemini.logical.rewrite import (
    RewriteFill,
    RewriteInitialize,
    RewriteMoves,
)
from bloqade.lanes.arch.gemini.logical.stmts import Fill, LogicalInitialize, SiteBusMove
from bloqade.lanes.dialects import move
from bloqade.lanes.layout.encoding import (
    Direction,
    LocationAddress,
    SiteLaneAddress,
)


def test_logical_architecture_rewrite_site():

    test_block = ir.Block()
    current_state = ir.TestValue()

    test_block.stmts.append(
        move.Move(
            current_state=current_state,
            lanes=(
                SiteLaneAddress(0, 0, 0, Direction.FORWARD),
                SiteLaneAddress(0, 1, 0, Direction.FORWARD),
                SiteLaneAddress(0, 2, 0, Direction.FORWARD),
                SiteLaneAddress(0, 3, 0, Direction.FORWARD),
            ),
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
            current_state=current_state,
            y_mask=const_list.result,
            word=0,
            bus_id=0,
            direction=Direction.FORWARD,
        )
    )
    assert_nodes(test_block, expected_block)


def test_logical_architecture_rewrite_site_no_lanes():

    test_block = ir.Block()

    test_block.stmts.append(move.Move(current_state=ir.TestValue(), lanes=()))

    expected_block = ir.Block()

    rewrite.Walk(RewriteMoves()).rewrite(test_block)

    assert_nodes(test_block, expected_block)


def test_logical_architecture_rewrite_fill():

    test_block = ir.Block()
    current_state = ir.TestValue()

    test_block.stmts.append(
        move.Fill(
            current_state=current_state,
            location_addresses=(
                LocationAddress(0, 0),
                LocationAddress(1, 2),
                LocationAddress(2, 4),
            ),
        )
    )

    expected_block = ir.Block()
    expected_block.stmts.append(
        const_list := py.Constant(
            ilist.IList(
                [(0, 0), (1, 2), (2, 4)], elem=types.Tuple[types.Int, types.Int]
            )
        )
    )
    expected_block.stmts.append(
        Fill(
            current_state=current_state,
            logical_addresses=const_list.result,
        )
    )

    rewrite.Walk(RewriteFill()).rewrite(test_block)
    assert_nodes(test_block, expected_block)


def test_logical_architecture_rewrite_logical_initialize():
    thetas = (ir.TestValue(), ir.TestValue(), ir.TestValue())
    phis = (ir.TestValue(), ir.TestValue(), ir.TestValue())
    lams = (ir.TestValue(), ir.TestValue(), ir.TestValue())
    location_addresses = (
        LocationAddress(0, 0),
        LocationAddress(1, 2),
        LocationAddress(2, 4),
    )

    test_block = ir.Block()
    current_state = ir.TestValue()

    test_block.stmts.append(
        move.LogicalInitialize(
            current_state=current_state,
            thetas=thetas,
            phis=phis,
            lams=lams,
            location_addresses=location_addresses,
        )
    )

    expected_block = ir.Block()
    expected_block.stmts.append(thetas_stmt := ilist.New(thetas))
    expected_block.stmts.append(phis_stmt := ilist.New(phis))
    expected_block.stmts.append(lams_stmt := ilist.New(lams))
    expected_block.stmts.append(
        const_list := py.Constant(
            ilist.IList(
                [ilist.IList([(0, 0)]), ilist.IList([(1, 2)]), ilist.IList([(2, 4)])]
            )
        )
    )
    expected_block.stmts.append(
        LogicalInitialize(
            current_state=current_state,
            thetas=thetas_stmt.result,
            phis=phis_stmt.result,
            lams=lams_stmt.result,
            logical_addresses=const_list.result,
        )
    )

    rewrite.Walk(RewriteInitialize()).rewrite(test_block)
    assert_nodes(test_block, expected_block)
