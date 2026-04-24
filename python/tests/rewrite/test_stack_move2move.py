from kirin import ir
from kirin.dialects import py
from kirin.rewrite import Walk

from bloqade.lanes.arch.gemini.logical import get_arch_spec
from bloqade.lanes.dialects import move, stack_move
from bloqade.lanes.layout.encoding import (
    LocationAddress,
    LocationAddress as EncodingLocationAddress,
    ZoneAddress as EncodingZoneAddress,
)
from bloqade.lanes.rewrite.stack_move2move import RewriteStackMoveToMove

_ARCH = get_arch_spec()


def _build_stack_move_block(stmts: list[ir.Statement]) -> ir.Block:
    """Append ``stmts`` to a fresh block and terminate it with a dummy
    ``stack_move.ConstInt(0)`` + ``func.Return`` pair so every block has
    a valid terminator.

    Since the ``return`` bytecode opcode overlaps with ``func.Return``
    from the kirin.basic dialect group, the decoder emits ``func.Return``
    directly — and so does this helper.
    """
    from kirin.dialects import func

    block = ir.Block()
    for stmt in stmts:
        block.stmts.append(stmt)
    ret_sentinel = stack_move.ConstInt(value=0)
    block.stmts.append(ret_sentinel)
    block.stmts.append(func.Return(ret_sentinel.result))
    return block


def test_empty_block_emits_load_and_func_return():
    # _build_stack_move_block synthesises a trailing ConstInt(0) + Return.
    block = _build_stack_move_block([])
    result = Walk(RewriteStackMoveToMove(arch_spec=_ARCH)).rewrite(block)
    assert result.has_done_something
    # Expect a move.Load at block start and a func.Return; the stack_move
    # Return should have been deleted.
    assert any(isinstance(s, move.Load) for s in block.stmts)
    # The helper already terminates the block with func.Return — the
    # rewrite passes it through unchanged, so the block ends with
    # move.Store(state) + func.Return after rewrite.
    from kirin.dialects import func

    assert any(isinstance(s, func.Return) for s in block.stmts)


def test_const_float_emits_py_constant_and_tracks_value():
    cf = stack_move.ConstFloat(value=1.5)
    block = _build_stack_move_block([cf])
    rule = RewriteStackMoveToMove(arch_spec=_ARCH)
    Walk(rule).rewrite(block)
    # py.Constant statement emitted with value 1.5.
    py_const = next(s for s in block.stmts if isinstance(s, py.Constant))
    assert py_const.value.unwrap() == 1.5
    # Its result is tracked in ssa_to_attr for attribute lifting by
    # downstream stateful-op handlers (key is the new SSA, because
    # replace_by rewired all consumer operands to point there).
    assert rule.ssa_to_attr[py_const.result] == 1.5


def test_const_loc_is_left_untouched_for_dce():
    # Address constants are intentionally *not* rewritten — stack_move2move
    # leaves ConstLoc / ConstLane / ConstZone in place so a downstream DCE
    # pass can sweep them up once their attribute values have been lifted
    # into consumer move.* statements (via ``_lift_attrs``). The rewrite
    # rule's ``_try_lift`` reads ``.value`` off the defining stmt rather
    # than looking it up in ``ssa_to_attr``.
    addr = LocationAddress(0, 0, 0)
    cl = stack_move.ConstLoc(value=addr)
    block = _build_stack_move_block([cl])
    rule = RewriteStackMoveToMove(arch_spec=_ARCH)
    Walk(rule).rewrite(block)
    # ConstLoc was not deleted and does not appear in ssa_to_attr.
    assert any(isinstance(s, stack_move.ConstLoc) for s in block.stmts)
    assert cl.result not in rule.ssa_to_attr
    # _try_lift resolves the address off the defining stmt's ``.value``.
    assert rule._try_lift(cl.result, LocationAddress) == addr


def test_pop_is_dropped():
    cf = stack_move.ConstFloat(value=1.0)
    pop = stack_move.Pop(value=cf.result)
    block = _build_stack_move_block([cf, pop])
    Walk(RewriteStackMoveToMove(arch_spec=_ARCH)).rewrite(block)
    # No target statement for Pop, and the original stack_move.Pop is gone.
    assert not any(isinstance(s, stack_move.Pop) for s in block.stmts)


def test_dup_redirects_uses_to_input():
    cf = stack_move.ConstFloat(value=1.0)
    dup = stack_move.Dup(value=cf.result)
    # Downstream consumer that references Dup's result.
    consumer = stack_move.Pop(value=dup.result)
    block = _build_stack_move_block([cf, dup, consumer])
    Walk(RewriteStackMoveToMove(arch_spec=_ARCH)).rewrite(block)
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
    block = _build_stack_move_block([a, b, sw, p_top, p_bot])
    Walk(RewriteStackMoveToMove(arch_spec=_ARCH)).rewrite(block)
    # Swap is gone.
    assert not any(isinstance(s, stack_move.Swap) for s in block.stmts)


def test_fill_lowers_to_move_fill_with_attribute_locations():
    a0 = LocationAddress(0, 0, 0)
    a1 = LocationAddress(0, 1, 0)
    cl0 = stack_move.ConstLoc(value=a0)
    cl1 = stack_move.ConstLoc(value=a1)
    fill = stack_move.Fill(locations=(cl0.result, cl1.result))
    block = _build_stack_move_block([cl0, cl1, fill])
    Walk(RewriteStackMoveToMove(arch_spec=_ARCH)).rewrite(block)
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
        axis_angle=cf_phi.result,
        rotation_angle=cf_theta.result,
        locations=(cl.result,),
    )
    block = _build_stack_move_block([cf_theta, cf_phi, cl, lr])
    Walk(RewriteStackMoveToMove(arch_spec=_ARCH)).rewrite(block)
    mr = next(s for s in block.stmts if isinstance(s, move.LocalR))
    # move.LocalR stores rotation angles as SSA values; axis_angle and
    # rotation_angle pass through from the stack_move statement. The SSA
    # values trace back to py.Constant statements produced when lowering
    # the ConstFloat constants.
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
    lr = stack_move.LocalRz(rotation_angle=cf_theta.result, locations=(cl.result,))
    block = _build_stack_move_block([cf_theta, cl, lr])
    Walk(RewriteStackMoveToMove(arch_spec=_ARCH)).rewrite(block)
    mr = next(s for s in block.stmts if isinstance(s, move.LocalRz))
    rot_const = mr.rotation_angle.owner
    assert isinstance(rot_const, py.Constant)
    assert rot_const.value.unwrap() == 0.3
    # encoding LocationAddress(word_id=0, site_id=1, zone_id=0).
    assert mr.location_addresses == (EncodingLocationAddress(0, 1, 0),)


def test_global_r_lowers_with_attribute_lifting():
    cf_theta = stack_move.ConstFloat(value=0.4)
    cf_phi = stack_move.ConstFloat(value=0.5)
    gr = stack_move.GlobalR(axis_angle=cf_phi.result, rotation_angle=cf_theta.result)
    block = _build_stack_move_block([cf_theta, cf_phi, gr])
    Walk(RewriteStackMoveToMove(arch_spec=_ARCH)).rewrite(block)
    mr = next(s for s in block.stmts if isinstance(s, move.GlobalR))
    axis_const = mr.axis_angle.owner
    rot_const = mr.rotation_angle.owner
    assert isinstance(axis_const, py.Constant)
    assert isinstance(rot_const, py.Constant)
    assert axis_const.value.unwrap() == 0.5
    assert rot_const.value.unwrap() == 0.4


def test_global_rz_lowers_with_attribute_lifting():
    cf_theta = stack_move.ConstFloat(value=0.6)
    gr = stack_move.GlobalRz(rotation_angle=cf_theta.result)
    block = _build_stack_move_block([cf_theta, gr])
    Walk(RewriteStackMoveToMove(arch_spec=_ARCH)).rewrite(block)
    mr = next(s for s in block.stmts if isinstance(s, move.GlobalRz))
    rot_const = mr.rotation_angle.owner
    assert isinstance(rot_const, py.Constant)
    assert rot_const.value.unwrap() == 0.6


def test_cz_lowers_with_attribute_zone():
    from bloqade.lanes.layout.encoding import ZoneAddress

    cz_zone = stack_move.ConstZone(value=ZoneAddress(0))
    cz = stack_move.CZ(zone=cz_zone.result)
    block = _build_stack_move_block([cz_zone, cz])
    Walk(RewriteStackMoveToMove(arch_spec=_ARCH)).rewrite(block)
    mcz = next(s for s in block.stmts if isinstance(s, move.CZ))
    # RewriteStackMoveToMove wraps native ZoneAddress into the encoding wrapper.
    assert mcz.zone_address == EncodingZoneAddress(0)


def test_measure_single_zone_emits_single_zone_measure():
    # Two zone operands both with zone_id=0: dedup collapses to one.
    cz0 = stack_move.ConstZone(value=EncodingZoneAddress(0))
    cz1 = stack_move.ConstZone(value=EncodingZoneAddress(0))
    m = stack_move.Measure(zones=(cz0.result, cz1.result))
    block = _build_stack_move_block([cz0, cz1, m])
    Walk(RewriteStackMoveToMove(arch_spec=_ARCH)).rewrite(block)
    mm = next(s for s in block.stmts if isinstance(s, move.Measure))
    # One zone (both operands are zone 0) — zones now live on the
    # zone_addresses attribute, not SSA operands.
    assert mm.zone_addresses == (EncodingZoneAddress(0),)


def test_measure_multi_zone_dedups():
    # Three zone operands, two of which share zone_id=0. Expect 2
    # distinct zone addresses in the attribute tuple after dedup.
    cz0 = stack_move.ConstZone(value=EncodingZoneAddress(0))
    cz1 = stack_move.ConstZone(value=EncodingZoneAddress(0))
    cz2 = stack_move.ConstZone(value=EncodingZoneAddress(1))
    m = stack_move.Measure(zones=(cz0.result, cz1.result, cz2.result))
    block = _build_stack_move_block([cz0, cz1, cz2, m])
    Walk(RewriteStackMoveToMove(arch_spec=_ARCH)).rewrite(block)
    mm = next(s for s in block.stmts if isinstance(s, move.Measure))
    assert mm.zone_addresses == (EncodingZoneAddress(0), EncodingZoneAddress(1))


def test_await_measure_lowers_to_getfutureresult_chain():
    # AwaitMeasure expands to one move.GetFutureResult per location in
    # each measured zone (order defined by the ArchSpec), then an
    # ilist.New bundling the results.
    from kirin.dialects import ilist

    cz = stack_move.ConstZone(value=EncodingZoneAddress(0))
    m = stack_move.Measure(zones=(cz.result,))
    aw = stack_move.AwaitMeasure(future=m.results[0])
    block = _build_stack_move_block([cz, m, aw])
    Walk(RewriteStackMoveToMove(arch_spec=_ARCH)).rewrite(block)

    expected_locs = list(_ARCH.yield_zone_locations(EncodingZoneAddress(0)))
    gfrs = [s for s in block.stmts if isinstance(s, move.GetFutureResult)]
    assert len(gfrs) == len(expected_locs)
    assert [g.location_address for g in gfrs] == expected_locs
    assert all(g.zone_address == EncodingZoneAddress(0) for g in gfrs)
    # A single ilist.New bundles the GetFutureResult outputs in order.
    ilist_news = [s for s in block.stmts if isinstance(s, ilist.New)]
    # The measurement ilist.New is the one whose values are the GetFutureResult
    # results.
    bundle = next(
        n for n in ilist_news if tuple(n.values) == tuple(g.result for g in gfrs)
    )
    assert len(bundle.values) == len(expected_locs)


def test_new_array_lowers_to_ilist_new():
    from kirin.dialects import ilist

    na = stack_move.NewArray(values=(), type_tag=0, dim0=4, dim1=0)
    block = _build_stack_move_block([na])
    Walk(RewriteStackMoveToMove(arch_spec=_ARCH)).rewrite(block)
    assert any(isinstance(s, ilist.New) for s in block.stmts)


def test_new_array_2d_preserves_nested_structure():
    """ArrayType[ElemType, Dim0, Dim1] with Dim1>0 lowers to an outer
    ilist.New containing ``dim0`` inner ilist.News (each of length
    ``dim1``) — the 2-D shape is preserved as nested ilists."""
    from kirin.dialects import ilist

    # Six element values laid out row-major: row0=[c0,c1], row1=[c2,c3],
    # row2=[c4,c5].
    consts = [stack_move.ConstInt(value=i) for i in range(6)]
    na = stack_move.NewArray(
        values=tuple(c.result for c in consts),
        type_tag=0,
        dim0=3,
        dim1=2,
    )
    block = _build_stack_move_block([*consts, na])
    Walk(RewriteStackMoveToMove(arch_spec=_ARCH)).rewrite(block)

    ilist_news = [s for s in block.stmts if isinstance(s, ilist.New)]
    # Three inner ilists of length 2 (the rows) + one outer ilist of
    # length 3 (the rows wrapped). Empty block terminator synthesises
    # no ilist.News.
    assert len(ilist_news) == 4
    inners = [n for n in ilist_news if len(n.values) == 2]
    outers = [n for n in ilist_news if len(n.values) == 3]
    assert len(inners) == 3
    assert len(outers) == 1
    # Outer's values are exactly the inner ilist results, in row order.
    assert tuple(outers[0].values) == tuple(inner.result for inner in inners)


def test_set_detector_lowers_to_annotate():
    from bloqade.decoders.dialects import annotate

    na = stack_move.NewArray(values=(), type_tag=0, dim0=1, dim1=0)
    sd = stack_move.SetDetector(array=na.result)
    block = _build_stack_move_block([na, sd])
    Walk(RewriteStackMoveToMove(arch_spec=_ARCH)).rewrite(block)
    assert any(isinstance(s, annotate.stmts.SetDetector) for s in block.stmts)


def test_set_observable_lowers_to_annotate():
    from bloqade.decoders.dialects import annotate

    na = stack_move.NewArray(values=(), type_tag=0, dim0=1, dim1=0)
    so = stack_move.SetObservable(array=na.result)
    block = _build_stack_move_block([na, so])
    Walk(RewriteStackMoveToMove(arch_spec=_ARCH)).rewrite(block)
    assert any(isinstance(s, annotate.stmts.SetObservable) for s in block.stmts)
