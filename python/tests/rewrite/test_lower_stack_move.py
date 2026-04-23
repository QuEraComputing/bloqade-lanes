from kirin import ir
from kirin.dialects import py
from kirin.rewrite import Walk

from bloqade.lanes.dialects import move, stack_move
from bloqade.lanes.layout.encoding import (
    LocationAddress,
    LocationAddress as EncodingLocationAddress,
    ZoneAddress as EncodingZoneAddress,
)
from bloqade.lanes.rewrite.lower_stack_move import LowerStackMove


def _build_stack_move_block(stmts: list[ir.Statement]) -> ir.Block:
    block = ir.Block()
    for stmt in stmts:
        block.stmts.append(stmt)
    return block


def test_empty_block_emits_load_and_func_return():
    block = _build_stack_move_block([stack_move.Return()])
    result = Walk(LowerStackMove()).rewrite(block)
    assert result.has_done_something
    # Expect a move.Load at block start and a func.Return; the stack_move
    # Return should have been deleted.
    assert any(isinstance(s, move.Load) for s in block.stmts)
    assert not any(isinstance(s, stack_move.Return) for s in block.stmts)


def test_const_float_emits_py_constant_and_tracks_value():
    cf = stack_move.ConstFloat(value=1.5)
    block = _build_stack_move_block([cf, stack_move.Return()])
    rule = LowerStackMove()
    Walk(rule).rewrite(block)
    # py.Constant statement emitted with value 1.5.
    py_const = next(s for s in block.stmts if isinstance(s, py.Constant))
    assert py_const.value.unwrap() == 1.5
    # Its result is tracked in ssa_to_attr for attribute lifting by
    # downstream stateful-op handlers (key is the new SSA, because
    # replace_by rewired all consumer operands to point there).
    assert rule.ssa_to_attr[py_const.result] == 1.5


def test_const_loc_tracks_attribute_value():
    addr = LocationAddress(0, 0, 0)
    cl = stack_move.ConstLoc(value=addr)
    block = _build_stack_move_block([cl, stack_move.Return()])
    rule = LowerStackMove()
    Walk(rule).rewrite(block)
    # The stack_move SSA is mapped to its raw attribute (for lifting into
    # downstream move.* attributes).
    assert rule.ssa_to_attr[cl.result] == addr


def test_pop_is_dropped():
    cf = stack_move.ConstFloat(value=1.0)
    pop = stack_move.Pop(value=cf.result)
    block = _build_stack_move_block([cf, pop, stack_move.Return()])
    Walk(LowerStackMove()).rewrite(block)
    # No target statement for Pop, and the original stack_move.Pop is gone.
    assert not any(isinstance(s, stack_move.Pop) for s in block.stmts)


def test_dup_redirects_uses_to_input():
    cf = stack_move.ConstFloat(value=1.0)
    dup = stack_move.Dup(value=cf.result)
    # Downstream consumer that references Dup's result.
    consumer = stack_move.Pop(value=dup.result)
    block = _build_stack_move_block([cf, dup, consumer, stack_move.Return()])
    Walk(LowerStackMove()).rewrite(block)
    # Dup is gone; Pop is also lowered away.
    assert not any(isinstance(s, stack_move.Dup) for s in block.stmts)
    assert not any(isinstance(s, stack_move.Pop) for s in block.stmts)


def test_swap_permutes_uses():
    a = stack_move.ConstInt(value=1)
    b = stack_move.ConstInt(value=2)
    sw = stack_move.Swap(in_top=b.result, in_bot=a.result)
    # Consumers that read Swap's outputs; pop them so the test has
    # something observable.
    p_top = stack_move.Pop(value=sw.out_top)
    p_bot = stack_move.Pop(value=sw.out_bot)
    block = _build_stack_move_block([a, b, sw, p_top, p_bot, stack_move.Return()])
    Walk(LowerStackMove()).rewrite(block)
    # Swap is gone.
    assert not any(isinstance(s, stack_move.Swap) for s in block.stmts)


def test_fill_lowers_to_move_fill_with_attribute_locations():
    a0 = LocationAddress(0, 0, 0)
    a1 = LocationAddress(0, 1, 0)
    cl0 = stack_move.ConstLoc(value=a0)
    cl1 = stack_move.ConstLoc(value=a1)
    fill = stack_move.Fill(locations=(cl0.result, cl1.result))
    block = _build_stack_move_block([cl0, cl1, fill, stack_move.Return()])
    Walk(LowerStackMove()).rewrite(block)
    mf = next(s for s in block.stmts if isinstance(s, move.Fill))
    # stack_move.ConstLoc now stores encoding-layer LocationAddress values
    # directly (matching the move dialect convention), so the rewrite just
    # forwards them to move.Fill.
    assert mf.location_addresses == (
        EncodingLocationAddress(0, 0, 0),
        EncodingLocationAddress(0, 1, 0),
    )


def test_local_r_lowers_with_attribute_lifting():
    cf_theta = stack_move.ConstFloat(value=0.1)
    cf_phi = stack_move.ConstFloat(value=0.2)
    cl = stack_move.ConstLoc(value=LocationAddress(0, 0, 0))
    lr = stack_move.LocalR(
        phi=cf_phi.result,
        theta=cf_theta.result,
        locations=(cl.result,),
    )
    block = _build_stack_move_block([cf_theta, cf_phi, cl, lr, stack_move.Return()])
    Walk(LowerStackMove()).rewrite(block)
    mr = next(s for s in block.stmts if isinstance(s, move.LocalR))
    # move.LocalR stores rotation angles as SSA values (axis_angle maps to
    # stack_move.LocalR.phi; rotation_angle maps to theta). The SSA values
    # trace back to py.Constant statements produced when lowering the
    # ConstFloat constants.
    axis_const = mr.axis_angle.owner
    rot_const = mr.rotation_angle.owner
    assert isinstance(axis_const, py.Constant)
    assert isinstance(rot_const, py.Constant)
    assert axis_const.value.unwrap() == 0.2
    assert rot_const.value.unwrap() == 0.1
    assert mr.location_addresses == (EncodingLocationAddress(0, 0, 0),)


def test_local_rz_lowers_with_attribute_lifting():
    cf_theta = stack_move.ConstFloat(value=0.3)
    cl = stack_move.ConstLoc(value=LocationAddress(0, 1, 0))
    lr = stack_move.LocalRz(theta=cf_theta.result, locations=(cl.result,))
    block = _build_stack_move_block([cf_theta, cl, lr, stack_move.Return()])
    Walk(LowerStackMove()).rewrite(block)
    mr = next(s for s in block.stmts if isinstance(s, move.LocalRz))
    rot_const = mr.rotation_angle.owner
    assert isinstance(rot_const, py.Constant)
    assert rot_const.value.unwrap() == 0.3
    # encoding LocationAddress(word_id=0, site_id=1, zone_id=0).
    assert mr.location_addresses == (EncodingLocationAddress(0, 1, 0),)


def test_global_r_lowers_with_attribute_lifting():
    cf_theta = stack_move.ConstFloat(value=0.4)
    cf_phi = stack_move.ConstFloat(value=0.5)
    gr = stack_move.GlobalR(phi=cf_phi.result, theta=cf_theta.result)
    block = _build_stack_move_block([cf_theta, cf_phi, gr, stack_move.Return()])
    Walk(LowerStackMove()).rewrite(block)
    mr = next(s for s in block.stmts if isinstance(s, move.GlobalR))
    axis_const = mr.axis_angle.owner
    rot_const = mr.rotation_angle.owner
    assert isinstance(axis_const, py.Constant)
    assert isinstance(rot_const, py.Constant)
    assert axis_const.value.unwrap() == 0.5
    assert rot_const.value.unwrap() == 0.4


def test_global_rz_lowers_with_attribute_lifting():
    cf_theta = stack_move.ConstFloat(value=0.6)
    gr = stack_move.GlobalRz(theta=cf_theta.result)
    block = _build_stack_move_block([cf_theta, gr, stack_move.Return()])
    Walk(LowerStackMove()).rewrite(block)
    mr = next(s for s in block.stmts if isinstance(s, move.GlobalRz))
    rot_const = mr.rotation_angle.owner
    assert isinstance(rot_const, py.Constant)
    assert rot_const.value.unwrap() == 0.6


def test_cz_lowers_with_attribute_zone():
    from bloqade.lanes.layout.encoding import ZoneAddress

    cz_zone = stack_move.ConstZone(value=ZoneAddress(0))
    cz = stack_move.CZ(zone=cz_zone.result)
    block = _build_stack_move_block([cz_zone, cz, stack_move.Return()])
    Walk(LowerStackMove()).rewrite(block)
    mcz = next(s for s in block.stmts if isinstance(s, move.CZ))
    # LowerStackMove wraps native ZoneAddress into the encoding wrapper.
    assert mcz.zone_address == EncodingZoneAddress(0)


def test_measure_single_zone_emits_single_zone_measure():
    cl0 = stack_move.ConstLoc(value=LocationAddress(0, 0, 0))
    cl1 = stack_move.ConstLoc(value=LocationAddress(0, 1, 0))
    m = stack_move.Measure(locations=(cl0.result, cl1.result))
    block = _build_stack_move_block([cl0, cl1, m, stack_move.Return()])
    Walk(LowerStackMove()).rewrite(block)
    mm = next(s for s in block.stmts if isinstance(s, move.Measure))
    # One zone (both locs are in zone 0).
    assert len(mm.zones) == 1


def test_measure_multi_zone_dedups():
    # Two locations in zone 0, one in zone 1. Expect 2 zone SSA values.
    cl0 = stack_move.ConstLoc(value=LocationAddress(0, 0, 0))
    cl1 = stack_move.ConstLoc(value=LocationAddress(0, 0, 1))
    cl2 = stack_move.ConstLoc(value=LocationAddress(0, 1, 0))
    m = stack_move.Measure(locations=(cl0.result, cl1.result, cl2.result))
    block = _build_stack_move_block([cl0, cl1, cl2, m, stack_move.Return()])
    Walk(LowerStackMove()).rewrite(block)
    mm = next(s for s in block.stmts if isinstance(s, move.Measure))
    assert len(mm.zones) == 2


def test_await_measure_lowers_without_error():
    # Smoke: await_measure after measure lowers cleanly. AwaitMeasure is
    # pure synchronisation in stack_move — no target-dialect emission.
    cl = stack_move.ConstLoc(value=LocationAddress(0, 0, 0))
    m = stack_move.Measure(locations=(cl.result,))
    aw = stack_move.AwaitMeasure(future=m.result)
    block = _build_stack_move_block([cl, m, aw, stack_move.Return()])
    Walk(LowerStackMove()).rewrite(block)  # should not raise


def test_new_array_lowers_to_ilist_new():
    from kirin.dialects import ilist

    na = stack_move.NewArray(type_tag=0, dim0=4, dim1=0)
    block = _build_stack_move_block([na, stack_move.Return()])
    Walk(LowerStackMove()).rewrite(block)
    assert any(isinstance(s, ilist.New) for s in block.stmts)


def test_set_detector_lowers_to_annotate():
    from bloqade.decoders.dialects import annotate

    na = stack_move.NewArray(type_tag=0, dim0=1, dim1=0)
    sd = stack_move.SetDetector(array=na.result)
    block = _build_stack_move_block([na, sd, stack_move.Return()])
    Walk(LowerStackMove()).rewrite(block)
    assert any(isinstance(s, annotate.stmts.SetDetector) for s in block.stmts)


def test_set_observable_lowers_to_annotate():
    from bloqade.decoders.dialects import annotate

    na = stack_move.NewArray(type_tag=0, dim0=1, dim1=0)
    so = stack_move.SetObservable(array=na.result)
    block = _build_stack_move_block([na, so, stack_move.Return()])
    Walk(LowerStackMove()).rewrite(block)
    assert any(isinstance(s, annotate.stmts.SetObservable) for s in block.stmts)
