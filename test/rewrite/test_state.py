from bloqade.test_utils import assert_nodes
from kirin import ir, rewrite, types
from kirin.dialects import cf, func, py

from bloqade.lanes import layout
from bloqade.lanes.dialects import move
from bloqade.lanes.rewrite import state


def test_rewrite_branch():

    block0 = ir.Block()
    test_block = ir.Block([py.Constant(10), cf.Branch((), successor=block0)])

    rewrite.Walk(state.RewriteBranches()).rewrite(test_block)

    expected_block = ir.Block(
        [
            py.Constant(10),
            current_state := move.Load(),
            cf.Branch(successor=block0, arguments=(current_state.result,)),
        ]
    )

    assert_nodes(test_block, expected_block)


def test_rewrite_conditional_branch():
    cond = ir.TestValue()
    block0 = ir.Block()
    block1 = ir.Block()
    test_block = ir.Block(
        [
            py.Constant(10),
            cf.ConditionalBranch(
                cond, (), (), then_successor=block0, else_successor=block1
            ),
        ]
    )

    rewrite.Walk(state.RewriteBranches()).rewrite(test_block)

    expected_block = ir.Block(
        [
            py.Constant(10),
            current_state := move.Load(),
            cf.ConditionalBranch(
                cond,
                (current_state.result,),
                (current_state.result,),
                then_successor=block0,
                else_successor=block1,
            ),
        ]
    )

    assert_nodes(test_block, expected_block)


def test_insert_block_args():
    test_region = ir.Region([ir.Block(), ir.Block()])

    test_func = func.Function(
        sym_name="test_func",
        signature=func.Signature((), types.NoneType),
        slots=(),
        body=test_region,
    )

    rewrite.Walk(state.InsertBlockArgs()).rewrite(test_func)

    expected_region = ir.Region([ir.Block(), block1 := ir.Block()])
    block1.args.append_from(state.StateType, "current_state")

    expected_func = func.Function(
        sym_name="test_func",
        signature=func.Signature((), types.NoneType),
        slots=(),
        body=expected_region,
    )

    for test_block, expected_block in zip(
        test_func.body.blocks, expected_func.body.blocks
    ):
        assert_nodes(test_block, expected_block)


def test_insert_block_args_noop():
    result = state.InsertBlockArgs().rewrite(py.Constant(10))
    assert not result.has_done_something


def test_rewrite_load():
    test_block = ir.Block(
        [
            first_load := move.Load(),
            move.Load(),
            third_state := move.CZ(
                first_load.result, zone_address=layout.ZoneAddress(0)
            ),
            move.Store(third_state.result),
            move.Store(third_state.result),
            move.EndMeasure(
                third_state.result, zone_addresses=(layout.ZoneAddress(0),)
            ),
            move.Load(),
        ]
    )

    rewrite.Walk(state.RewriteLoadStore()).rewrite(test_block)

    expected_block = ir.Block(
        [
            first_load := move.Load(),
            third_state := move.CZ(
                first_load.result, zone_address=layout.ZoneAddress(0)
            ),
            move.Store(third_state.result),
            move.EndMeasure(
                third_state.result, zone_addresses=(layout.ZoneAddress(0),)
            ),
            move.Load(),
        ]
    )

    assert_nodes(test_block, expected_block)
