from kirin import ir
from kirin.dialects import func
from kirin.rewrite import Walk

from bloqade.lanes.bytecode.encoding import (
    Direction,
    LaneAddress,
    LocationAddress,
    MoveType,
)
from bloqade.lanes.dialects import move, stack_move
from bloqade.lanes.rewrite.move2stack_move import RewriteMoveToStackMove


def test_load_and_store_are_removed():
    load = move.Load()
    store = move.Store(current_state=load.result)
    none_stmt = func.ConstantNone()
    ret = func.Return(none_stmt.result)
    block = ir.Block()
    for s in [load, store, none_stmt, ret]:
        block.stmts.append(s)

    Walk(RewriteMoveToStackMove()).rewrite(block)

    assert not any(isinstance(s, move.Load) for s in block.stmts)
    assert not any(isinstance(s, move.Store) for s in block.stmts)


def test_first_fill_lowers_to_initial_fill():
    a0 = LocationAddress(0, 0, 0)
    load = move.Load()
    fill = move.Fill(current_state=load.result, location_addresses=(a0,))
    store = move.Store(current_state=fill.result)
    none_stmt = func.ConstantNone()
    block = ir.Block()
    for s in [load, fill, store, none_stmt, func.Return(none_stmt.result)]:
        block.stmts.append(s)

    Walk(RewriteMoveToStackMove()).rewrite(block)

    assert not any(isinstance(s, move.Fill) for s in block.stmts)
    assert any(isinstance(s, stack_move.InitialFill) for s in block.stmts)
    assert not any(isinstance(s, stack_move.Fill) for s in block.stmts)
    locs = [s for s in block.stmts if isinstance(s, stack_move.ConstLoc)]
    assert len(locs) == 1
    assert locs[0].value == a0


def test_second_fill_lowers_to_fill_not_initial_fill():
    a0 = LocationAddress(0, 0, 0)
    a1 = LocationAddress(0, 1, 0)
    load = move.Load()
    fill1 = move.Fill(current_state=load.result, location_addresses=(a0,))
    fill2 = move.Fill(current_state=fill1.result, location_addresses=(a1,))
    store = move.Store(current_state=fill2.result)
    none_stmt = func.ConstantNone()
    block = ir.Block()
    for s in [load, fill1, fill2, store, none_stmt, func.Return(none_stmt.result)]:
        block.stmts.append(s)

    Walk(RewriteMoveToStackMove()).rewrite(block)

    fills = [
        s
        for s in block.stmts
        if isinstance(s, (stack_move.InitialFill, stack_move.Fill))
    ]
    assert len(fills) == 2
    assert isinstance(fills[0], stack_move.InitialFill)
    assert isinstance(fills[1], stack_move.Fill)


def test_move_lowers_to_stack_move_move():
    lane0 = LaneAddress(MoveType.SITE, 0, 0, 0, Direction.FORWARD)
    lane1 = LaneAddress(MoveType.SITE, 1, 0, 0, Direction.FORWARD)
    load = move.Load()
    mv = move.Move(current_state=load.result, lanes=(lane0, lane1))
    store = move.Store(current_state=mv.result)
    none_stmt = func.ConstantNone()
    block = ir.Block()
    for s in [load, mv, store, none_stmt, func.Return(none_stmt.result)]:
        block.stmts.append(s)

    Walk(RewriteMoveToStackMove()).rewrite(block)

    assert not any(isinstance(s, move.Move) for s in block.stmts)
    sm_mv = next(s for s in block.stmts if isinstance(s, stack_move.Move))
    lane_consts = [s for s in block.stmts if isinstance(s, stack_move.ConstLane)]
    assert len(lane_consts) == 2
    assert {lc.value for lc in lane_consts} == {lane0, lane1}
    assert len(sm_mv.lanes) == 2
