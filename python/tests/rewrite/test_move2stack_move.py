from kirin import ir
from kirin.dialects import func, ilist, py as kirin_py
from kirin.rewrite import Walk

from bloqade.lanes.arch.gemini.logical import get_arch_spec
from bloqade.lanes.bytecode.encoding import (
    Direction,
    LaneAddress,
    LocationAddress,
    MoveType,
    ZoneAddress,
)
from bloqade.lanes.dialects import move, stack_move
from bloqade.lanes.rewrite.move2stack_move import RewriteMoveToStackMove

_ARCH = get_arch_spec()


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


def test_py_constant_float_converts_to_const_float():
    pc = kirin_py.Constant(value=ir.PyAttr(1.5))
    none_stmt = func.ConstantNone()
    block = ir.Block()
    for s in [pc, none_stmt, func.Return(none_stmt.result)]:
        block.stmts.append(s)

    Walk(RewriteMoveToStackMove()).rewrite(block)

    assert not any(isinstance(s, kirin_py.Constant) for s in block.stmts)
    cf = next(s for s in block.stmts if isinstance(s, stack_move.ConstFloat))
    assert cf.value == 1.5


def test_py_constant_int_converts_to_const_int():
    pc = kirin_py.Constant(value=ir.PyAttr(7))
    none_stmt = func.ConstantNone()
    block = ir.Block()
    for s in [pc, none_stmt, func.Return(none_stmt.result)]:
        block.stmts.append(s)

    Walk(RewriteMoveToStackMove()).rewrite(block)

    assert not any(isinstance(s, kirin_py.Constant) for s in block.stmts)
    ci = next(s for s in block.stmts if isinstance(s, stack_move.ConstInt))
    assert ci.value == 7


def test_local_r_lowers_with_const_loc_and_angles():
    axis_c = kirin_py.Constant(value=ir.PyAttr(0.1))
    rot_c = kirin_py.Constant(value=ir.PyAttr(0.2))
    addr = LocationAddress(0, 0, 0)
    load = move.Load()
    lr = move.LocalR(
        current_state=load.result,
        axis_angle=axis_c.result,
        rotation_angle=rot_c.result,
        location_addresses=(addr,),
    )
    store = move.Store(current_state=lr.result)
    none_stmt = func.ConstantNone()
    block = ir.Block()
    for s in [axis_c, rot_c, load, lr, store, none_stmt, func.Return(none_stmt.result)]:
        block.stmts.append(s)

    Walk(RewriteMoveToStackMove()).rewrite(block)

    assert not any(isinstance(s, move.LocalR) for s in block.stmts)
    sm_lr = next(s for s in block.stmts if isinstance(s, stack_move.LocalR))
    locs = [s for s in block.stmts if isinstance(s, stack_move.ConstLoc)]
    assert len(locs) == 1
    assert locs[0].value == addr
    assert isinstance(sm_lr.axis_angle.owner, stack_move.ConstFloat)
    assert isinstance(sm_lr.rotation_angle.owner, stack_move.ConstFloat)


def test_local_rz_lowers():
    rot_c = kirin_py.Constant(value=ir.PyAttr(0.5))
    addr = LocationAddress(0, 1, 0)
    load = move.Load()
    lrz = move.LocalRz(
        current_state=load.result,
        rotation_angle=rot_c.result,
        location_addresses=(addr,),
    )
    store = move.Store(current_state=lrz.result)
    none_stmt = func.ConstantNone()
    block = ir.Block()
    for s in [rot_c, load, lrz, store, none_stmt, func.Return(none_stmt.result)]:
        block.stmts.append(s)

    Walk(RewriteMoveToStackMove()).rewrite(block)

    sm_lrz = next(s for s in block.stmts if isinstance(s, stack_move.LocalRz))
    assert isinstance(sm_lrz.rotation_angle.owner, stack_move.ConstFloat)
    locs = [s for s in block.stmts if isinstance(s, stack_move.ConstLoc)]
    assert locs[0].value == addr


def test_global_r_lowers():
    axis_c = kirin_py.Constant(value=ir.PyAttr(0.3))
    rot_c = kirin_py.Constant(value=ir.PyAttr(0.4))
    load = move.Load()
    gr = move.GlobalR(
        current_state=load.result,
        axis_angle=axis_c.result,
        rotation_angle=rot_c.result,
    )
    store = move.Store(current_state=gr.result)
    none_stmt = func.ConstantNone()
    block = ir.Block()
    for s in [axis_c, rot_c, load, gr, store, none_stmt, func.Return(none_stmt.result)]:
        block.stmts.append(s)

    Walk(RewriteMoveToStackMove()).rewrite(block)

    sm_gr = next(s for s in block.stmts if isinstance(s, stack_move.GlobalR))
    assert isinstance(sm_gr.axis_angle.owner, stack_move.ConstFloat)
    assert isinstance(sm_gr.rotation_angle.owner, stack_move.ConstFloat)


def test_global_rz_lowers():
    rot_c = kirin_py.Constant(value=ir.PyAttr(0.6))
    load = move.Load()
    grz = move.GlobalRz(current_state=load.result, rotation_angle=rot_c.result)
    store = move.Store(current_state=grz.result)
    none_stmt = func.ConstantNone()
    block = ir.Block()
    for s in [rot_c, load, grz, store, none_stmt, func.Return(none_stmt.result)]:
        block.stmts.append(s)

    Walk(RewriteMoveToStackMove()).rewrite(block)

    sm_grz = next(s for s in block.stmts if isinstance(s, stack_move.GlobalRz))
    assert isinstance(sm_grz.rotation_angle.owner, stack_move.ConstFloat)


def test_cz_lowers_with_const_zone():
    load = move.Load()
    cz = move.CZ(current_state=load.result, zone_address=ZoneAddress(0))
    store = move.Store(current_state=cz.result)
    none_stmt = func.ConstantNone()
    block = ir.Block()
    for s in [load, cz, store, none_stmt, func.Return(none_stmt.result)]:
        block.stmts.append(s)

    Walk(RewriteMoveToStackMove()).rewrite(block)

    assert not any(isinstance(s, move.CZ) for s in block.stmts)
    sm_cz = next(s for s in block.stmts if isinstance(s, stack_move.CZ))
    zone_consts = [s for s in block.stmts if isinstance(s, stack_move.ConstZone)]
    assert len(zone_consts) == 1
    assert zone_consts[0].value == ZoneAddress(0)
    assert sm_cz.zone.owner is zone_consts[0]


# ---- measurement tests ----


def test_measure_lowers_to_stack_move_measure():
    load = move.Load()
    m = move.Measure(current_state=load.result, zone_addresses=(ZoneAddress(0),))
    store = move.Store(current_state=m.result)
    none_stmt = func.ConstantNone()
    block = ir.Block()
    for s in [load, m, store, none_stmt, func.Return(none_stmt.result)]:
        block.stmts.append(s)

    Walk(RewriteMoveToStackMove()).rewrite(block)

    assert not any(isinstance(s, move.Measure) for s in block.stmts)
    sm_m = next(s for s in block.stmts if isinstance(s, stack_move.Measure))
    zone_consts = [s for s in block.stmts if isinstance(s, stack_move.ConstZone)]
    assert len(zone_consts) == 1
    assert zone_consts[0].value == ZoneAddress(0)
    assert len(sm_m.zones) == 1


def test_getfutureresult_chain_lowers_to_await_measure():
    locs = list(_ARCH.yield_zone_locations(ZoneAddress(0)))
    load = move.Load()
    m = move.Measure(current_state=load.result, zone_addresses=(ZoneAddress(0),))
    gfrs = [
        move.GetFutureResult(
            measurement_future=m.future,
            zone_address=ZoneAddress(0),
            location_address=loc,
        )
        for loc in locs
    ]
    bundle = ilist.New(values=tuple(g.result for g in gfrs))
    store = move.Store(current_state=m.result)
    none_stmt = func.ConstantNone()
    block = ir.Block()
    for s in [load, m, *gfrs, bundle, store, none_stmt, func.Return(none_stmt.result)]:
        block.stmts.append(s)

    Walk(RewriteMoveToStackMove()).rewrite(block)

    assert not any(isinstance(s, move.GetFutureResult) for s in block.stmts)
    assert not any(isinstance(s, ilist.New) for s in block.stmts)
    sm_m = next(s for s in block.stmts if isinstance(s, stack_move.Measure))
    aw = next(s for s in block.stmts if isinstance(s, stack_move.AwaitMeasure))
    assert aw.future is sm_m.results[0]
