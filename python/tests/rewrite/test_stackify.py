"""Tests for stackify stackification rewrite."""

from typing import cast

from kirin import ir, types
from kirin.dialects import func

from bloqade.lanes.bytecode.decode import load_program
from bloqade.lanes.bytecode.encode import dump_program
from bloqade.lanes.bytecode.encoding import LocationAddress, ZoneAddress
from bloqade.lanes.dialects import stack_move as sm
from bloqade.lanes.rewrite.stackify import stackify


def _make_method(*stmts) -> ir.Method:
    """Build a minimal stack_move ir.Method from an ordered list of statements."""
    block = ir.Block(argtypes=(types.MethodType,))
    for s in stmts:
        block.stmts.append(s)
    region = ir.Region(blocks=block)
    function = func.Function(
        sym_name="test",
        signature=func.Signature((), types.Any),
        slots=(),
        body=region,
    )
    from kirin.dialects import func as func_mod

    from bloqade.lanes.dialects import stack_move as sm_mod

    dialects = ir.DialectGroup([sm_mod.dialect, func_mod.dialect])
    return ir.Method(dialects=dialects, code=function, sym_name="test", arg_names=[])


def _stackify(method: ir.Method) -> list[ir.Statement]:
    """Apply stackify and return the resulting statement list."""
    stackify(method)
    return list(method.callable_region.blocks[0].stmts)


# ── CloneConstants: ordering fix ──────────────────────────────────────────────


def test_clone_constants_local_r_ordering():
    """ConstLoc (deepest arg of LocalR) must appear before ConstFloats.

    move2stack_move inserts ConstLoc immediately before LocalR but the
    ConstFloat angles come from earlier py.Constant rewrites, producing
    wrong block order: [ConstFloat, ConstFloat, ConstLoc, LocalR].
    CloneConstants must reorder to [ConstLoc, ConstFloat(rot), ConstFloat(ax), LocalR].
    """
    cf_rot = sm.ConstFloat(value=0.1)
    cf_ax = sm.ConstFloat(value=0.2)
    cl = sm.ConstLoc(value=LocationAddress(0, 0, 0))
    lr = sm.LocalR(
        axis_angle=cf_ax.result,
        rotation_angle=cf_rot.result,
        locations=(cl.result,),
    )
    ci = sm.ConstInt(value=0)
    ret = func.Return(ci.result)
    method = _make_method(cf_rot, cf_ax, cl, lr, ci, ret)

    stmts = _stackify(method)

    # Expected: ConstLoc (deepest), ConstFloat(rot), ConstFloat(ax), LocalR, ConstInt, Return
    assert [type(s) for s in stmts] == [
        sm.ConstLoc,
        sm.ConstFloat,
        sm.ConstFloat,
        sm.LocalR,
        sm.ConstInt,
        func.Return,
    ]
    local_r = cast(sm.LocalR, stmts[3])
    assert local_r.locations[0] is cast(sm.ConstLoc, stmts[0]).result
    assert local_r.rotation_angle is cast(sm.ConstFloat, stmts[1]).result
    assert local_r.axis_angle is cast(sm.ConstFloat, stmts[2]).result


def test_clone_constants_global_r_ordering():
    """GlobalR: rotation_angle (deepest, arg 1) clone before axis_angle (top, arg 0)."""
    cf_rot = sm.ConstFloat(value=0.1)
    cf_ax = sm.ConstFloat(value=0.2)
    gr = sm.GlobalR(axis_angle=cf_ax.result, rotation_angle=cf_rot.result)
    ci = sm.ConstInt(value=0)
    ret = func.Return(ci.result)
    method = _make_method(cf_rot, cf_ax, gr, ci, ret)

    stmts = _stackify(method)

    assert [type(s) for s in stmts] == [
        sm.ConstFloat,
        sm.ConstFloat,
        sm.GlobalR,
        sm.ConstInt,
        func.Return,
    ]
    gr_stmt = cast(sm.GlobalR, stmts[2])
    assert gr_stmt.rotation_angle is cast(sm.ConstFloat, stmts[0]).result
    assert gr_stmt.axis_angle is cast(sm.ConstFloat, stmts[1]).result


# ── CloneConstants: multi-use ─────────────────────────────────────────────────


def test_clone_constants_global_r_same_value():
    """GlobalR(%cf, %cf) → two distinct ConstFloat clones."""
    cf = sm.ConstFloat(value=1.5)
    gr = sm.GlobalR(axis_angle=cf.result, rotation_angle=cf.result)
    ci = sm.ConstInt(value=0)
    ret = func.Return(ci.result)
    method = _make_method(cf, gr, ci, ret)

    stmts = _stackify(method)

    assert [type(s) for s in stmts] == [
        sm.ConstFloat,
        sm.ConstFloat,
        sm.GlobalR,
        sm.ConstInt,
        func.Return,
    ]
    assert stmts[0] is not stmts[1]
    gr_stmt = cast(sm.GlobalR, stmts[2])
    assert gr_stmt.rotation_angle is cast(sm.ConstFloat, stmts[0]).result
    assert gr_stmt.axis_angle is cast(sm.ConstFloat, stmts[1]).result


def test_clone_constants_multi_consumer_same_value():
    """Same ConstFloat consumed by two separate statements → each gets its own clone."""
    cf = sm.ConstFloat(value=0.5)
    gr1 = sm.GlobalRz(rotation_angle=cf.result)
    gr2 = sm.GlobalRz(rotation_angle=cf.result)
    ci = sm.ConstInt(value=0)
    ret = func.Return(ci.result)
    method = _make_method(cf, gr1, gr2, ci, ret)

    stmts = _stackify(method)

    assert [type(s) for s in stmts] == [
        sm.ConstFloat,
        sm.GlobalRz,
        sm.ConstFloat,
        sm.GlobalRz,
        sm.ConstInt,
        func.Return,
    ]
    assert stmts[0] is not stmts[2]


# ── CloneConstants: non-Pure args left in place ───────────────────────────────


def test_clone_constants_skips_non_pure_arg():
    """Non-Pure args (e.g. AwaitMeasure result) are not cloned."""
    cz = sm.ConstZone(value=ZoneAddress(0))
    measure = sm.Measure(zones=(cz.result,))
    await_m = sm.AwaitMeasure(future=measure.results[0])
    ci = sm.ConstInt(value=0)
    gi = sm.GetItem(array=await_m.result, indices=(ci.result,))
    ret = func.Return(gi.result)
    method = _make_method(cz, measure, await_m, ci, gi, ret)

    stmts = _stackify(method)

    await_idx = next(i for i, s in enumerate(stmts) if isinstance(s, sm.AwaitMeasure))
    gi_idx = next(i for i, s in enumerate(stmts) if isinstance(s, sm.GetItem))

    # GetItem.array still references the AwaitMeasure result (not cloned)
    assert (
        cast(sm.GetItem, stmts[gi_idx]).array
        is cast(sm.AwaitMeasure, stmts[await_idx]).result
    )
    # ConstInt clone for the index sits immediately before GetItem
    assert gi_idx > 0 and isinstance(stmts[gi_idx - 1], sm.ConstInt)


# ── Round-trip integration ─────────────────────────────────────────────────────


def test_stackify_then_encode_local_r():
    """After stackify, LocalR IR encodes to correct bytecode."""
    from bloqade.lanes.bytecode import Instruction, Program

    expected = [
        Instruction.const_loc(0, 0, 0),
        Instruction.const_float(0.1),
        Instruction.const_float(0.2),
        Instruction.local_r(1),
        Instruction.const_int(0),
        Instruction.return_(),
    ]

    # Build IR with wrong ordering (as move2stack_move would produce)
    cf_rot = sm.ConstFloat(value=0.1)
    cf_ax = sm.ConstFloat(value=0.2)
    cl = sm.ConstLoc(value=LocationAddress(0, 0, 0))
    lr = sm.LocalR(
        axis_angle=cf_ax.result,
        rotation_angle=cf_rot.result,
        locations=(cl.result,),
    )
    ci = sm.ConstInt(value=0)
    ret = func.Return(ci.result)
    method = _make_method(cf_rot, cf_ax, cl, lr, ci, ret)

    stackify(method)

    encoded = dump_program(method)
    assert encoded.to_text() == Program(version=(1, 0), instructions=expected).to_text()


# ── CSE + stackify: real-pipeline simulation ──────────────────────────────────


def test_cse_then_stackify_shared_const_float():
    """CSE deduplicates two ConstFloat(0.5) → stackify re-clones one per consumer.

    The real pipeline runs DCE + CSE before stackify, which can collapse
    duplicate constants into a single SSA value with multiple uses.
    CloneConstants must then produce a fresh clone for each consumer so the
    encoder sees exactly one const_float per GlobalRz.
    """
    from kirin.rewrite import CommonSubexpressionElimination, Walk

    from bloqade.lanes.bytecode import Instruction, Program

    cf1 = sm.ConstFloat(value=0.5)
    cf2 = sm.ConstFloat(value=0.5)
    gr1 = sm.GlobalRz(rotation_angle=cf1.result)
    gr2 = sm.GlobalRz(rotation_angle=cf2.result)
    ci = sm.ConstInt(value=0)
    ret = func.Return(ci.result)
    method = _make_method(cf1, cf2, gr1, gr2, ci, ret)

    Walk(CommonSubexpressionElimination()).rewrite(method.code)
    assert len(cf1.result.uses) == 2  # both gr1 and gr2 now reference cf1

    stackify(method)

    expected = [
        Instruction.const_float(0.5),
        Instruction.global_rz(),
        Instruction.const_float(0.5),
        Instruction.global_rz(),
        Instruction.const_int(0),
        Instruction.return_(),
    ]
    encoded = dump_program(method)
    assert encoded.to_text() == Program(version=(1, 0), instructions=expected).to_text()


def test_cse_then_stackify_shared_const_loc():
    """CSE deduplicates identical ConstLoc + ConstFloat → stackify re-clones each.

    Two LocalRz on the same location with the same angle produce four constant
    stmts pre-CSE. CSE reduces them to two shared SSA values; stackify must
    re-clone both for each consumer and emit them in the correct stack order
    (location deepest, rotation_angle on top).
    """
    from kirin.rewrite import CommonSubexpressionElimination, Walk

    from bloqade.lanes.bytecode import Instruction, Program

    cl1 = sm.ConstLoc(value=LocationAddress(0, 0, 0))
    cl2 = sm.ConstLoc(value=LocationAddress(0, 0, 0))
    cf1 = sm.ConstFloat(value=0.3)
    cf2 = sm.ConstFloat(value=0.3)
    lrz1 = sm.LocalRz(rotation_angle=cf1.result, locations=(cl1.result,))
    lrz2 = sm.LocalRz(rotation_angle=cf2.result, locations=(cl2.result,))
    ci = sm.ConstInt(value=0)
    ret = func.Return(ci.result)
    method = _make_method(cl1, cl2, cf1, cf2, lrz1, lrz2, ci, ret)

    Walk(CommonSubexpressionElimination()).rewrite(method.code)
    assert len(cl1.result.uses) == 2  # both lrz1 and lrz2 now share cl1
    assert len(cf1.result.uses) == 2  # both lrz1 and lrz2 now share cf1

    stackify(method)

    expected = [
        Instruction.const_loc(0, 0, 0),
        Instruction.const_float(0.3),
        Instruction.local_rz(1),
        Instruction.const_loc(0, 0, 0),
        Instruction.const_float(0.3),
        Instruction.local_rz(1),
        Instruction.const_int(0),
        Instruction.return_(),
    ]
    encoded = dump_program(method)
    assert encoded.to_text() == Program(version=(1, 0), instructions=expected).to_text()


def test_stackify_then_encode_global_r_same_value():
    """GlobalR(%cf, %cf) encodes correctly after stackify."""
    cf = sm.ConstFloat(value=0.7)
    gr = sm.GlobalR(axis_angle=cf.result, rotation_angle=cf.result)
    ci = sm.ConstInt(value=0)
    ret = func.Return(ci.result)
    method = _make_method(cf, gr, ci, ret)

    stackify(method)

    encoded = dump_program(method)
    decoded = load_program(encoded)
    assert decoded is not None

    ops = [instr.op_name() for instr in encoded.instructions]
    assert ops.count("const_float") == 2
    assert "global_r" in ops


# ── Spilling to locals ────────────────────────────────────────────────────────


def _measure_await_chain() -> tuple:
    """Return (cz, measure, await_m) for use in the spill tests."""
    cz = sm.ConstZone(value=ZoneAddress(0))
    measure = sm.Measure(zones=(cz.result,))
    await_m = sm.AwaitMeasure(future=measure.results[0])
    return cz, measure, await_m


def _locals_ops(stmts: list[ir.Statement]) -> list[tuple[str, int]]:
    """The spill schedule: each StoreLocal/LoadLocal and the slot it names."""
    return [
        ("store" if isinstance(s, sm.StoreLocal) else "load", s.index)
        for s in stmts
        if isinstance(s, (sm.StoreLocal, sm.LoadLocal))
    ]


def _validates(method: ir.Method) -> None:
    """Encode ``method`` and run the Rust validator's stack simulation over it.

    The spill schedule is only right if every operand is where its consumer
    pops it, with the right tag — which is exactly what the simulation checks,
    locals included.
    """
    dump_program(method).validate(stack=True)


def test_stackify_spills_nothing_for_a_single_consumer():
    """A value consumed once stays on the stack: no locals are used."""
    cz, measure, await_m = _measure_await_chain()
    idx0 = sm.ConstInt(value=0)
    gi0 = sm.GetItem(array=await_m.result, indices=(idx0.result,))
    ci = sm.ConstInt(value=0)
    ret = func.Return(ci.result)
    method = _make_method(cz, measure, await_m, idx0, gi0, ci, ret)

    stmts = _stackify(method)

    assert _locals_ops(stmts) == []
    assert sm.Dup not in [type(s) for s in stmts]


def test_stackify_spills_a_value_with_two_consumers():
    """The array is stored once, right after it is produced, and reloaded
    below each consumer's constant index."""
    cz, measure, await_m = _measure_await_chain()
    idx0 = sm.ConstInt(value=0)
    gi0 = sm.GetItem(array=await_m.result, indices=(idx0.result,))
    idx1 = sm.ConstInt(value=1)
    gi1 = sm.GetItem(array=await_m.result, indices=(idx1.result,))
    na = sm.NewArray(values=(gi0.result, gi1.result), type_tag=1, dim0=2, dim1=0)
    ci = sm.ConstInt(value=0)
    ret = func.Return(ci.result)
    method = _make_method(cz, measure, await_m, idx0, gi0, idx1, gi1, na, ci, ret)

    stmts = _stackify(method)

    assert _locals_ops(stmts) == [("store", 0), ("load", 0), ("load", 0)]
    await_i = stmts.index(await_m)
    store = stmts[await_i + 1]
    assert isinstance(store, sm.StoreLocal) and store.value is await_m.result
    # Each GetItem reads its own reload, which sits just below its index.
    for gi in (gi0, gi1):
        gi_i = stmts.index(gi)
        load = stmts[gi_i - 2]
        assert isinstance(load, sm.LoadLocal) and gi.array is load.result
        assert isinstance(stmts[gi_i - 1], sm.ConstInt)
    # A placeholder at run time, so the local is typed `undef`.
    assert store.value_type == "undef"
    _validates(method)


def test_stackify_spills_a_value_with_three_consumers():
    """One store and a reload per consumer, all through one slot."""
    cz, measure, await_m = _measure_await_chain()
    idx0 = sm.ConstInt(value=0)
    gi0 = sm.GetItem(array=await_m.result, indices=(idx0.result,))
    idx1 = sm.ConstInt(value=1)
    gi1 = sm.GetItem(array=await_m.result, indices=(idx1.result,))
    idx2 = sm.ConstInt(value=2)
    gi2 = sm.GetItem(array=await_m.result, indices=(idx2.result,))
    na = sm.NewArray(
        values=(gi0.result, gi1.result, gi2.result), type_tag=1, dim0=3, dim1=0
    )
    ci = sm.ConstInt(value=0)
    ret = func.Return(ci.result)
    method = _make_method(
        cz, measure, await_m, idx0, gi0, idx1, gi1, idx2, gi2, na, ci, ret
    )

    stmts = _stackify(method)

    assert _locals_ops(stmts) == [("store", 0)] + [("load", 0)] * 3
    loads = [s for s in stmts if isinstance(s, sm.LoadLocal)]
    assert [gi.array for gi in (gi0, gi1, gi2)] == [load.result for load in loads]
    _validates(method)


def test_stackify_reuses_a_slot_once_its_value_is_dead():
    """Two arrays read one after the other share slot 0; read interleaved,
    they need two."""

    def two_arrays(interleaved: bool) -> ir.Method:
        cz_a, measure_a, await_a = _measure_await_chain()
        cz_b, measure_b, await_b = _measure_await_chain()
        reads = [await_a, await_a, await_b, await_b]
        if interleaved:
            reads = [await_a, await_b, await_a, await_b]
        stmts: list[ir.Statement] = [cz_a, measure_a, await_a]
        if interleaved:
            stmts += [cz_b, measure_b, await_b]
        items = []
        for n, source in enumerate(reads):
            if not interleaved and n == 2:
                stmts += [cz_b, measure_b, await_b]
            idx = sm.ConstInt(value=n)
            gi = sm.GetItem(array=source.result, indices=(idx.result,))
            stmts += [idx, gi]
            items.append(gi.result)
        na = sm.NewArray(values=tuple(items), type_tag=1, dim0=4, dim1=0)
        ci = sm.ConstInt(value=0)
        return _make_method(*stmts, na, ci, func.Return(ci.result))

    sequential = two_arrays(interleaved=False)
    assert {i for _, i in _locals_ops(_stackify(sequential))} == {0}
    _validates(sequential)

    interleaved = two_arrays(interleaved=True)
    assert {i for _, i in _locals_ops(_stackify(interleaved))} == {0, 1}
    _validates(interleaved)


def test_stackify_reloads_every_argument_of_a_consumer_that_takes_a_spilled_one():
    """A measurement shared by two detectors is spilled — and so is its
    neighbour in the first, or its reload would land on top of it and the
    elements would come out reversed."""
    cz, measure, await_m = _measure_await_chain()
    idx0 = sm.ConstInt(value=0)
    gi0 = sm.GetItem(array=await_m.result, indices=(idx0.result,))
    idx1 = sm.ConstInt(value=1)
    gi1 = sm.GetItem(array=await_m.result, indices=(idx1.result,))
    # gi1 appears in both detectors; gi0 only in the first, below gi1.
    na0 = sm.NewArray(values=(gi0.result, gi1.result), type_tag=1, dim0=2, dim1=0)
    na1 = sm.NewArray(values=(gi1.result,), type_tag=1, dim0=1, dim1=0)
    ci = sm.ConstInt(value=0)
    ret = func.Return(ci.result)
    method = _make_method(cz, measure, await_m, idx0, gi0, idx1, gi1, na0, na1, ci, ret)

    stmts = _stackify(method)

    na0_i = stmts.index(na0)
    below = stmts[na0_i - 2 : na0_i]
    assert all(isinstance(s, sm.LoadLocal) for s in below)
    # Deepest first: gi0's reload, then gi1's.
    assert [s.results[0] for s in below] == list(na0.values)
    stores = {s.value: s.index for s in stmts if isinstance(s, sm.StoreLocal)}
    assert [cast(sm.LoadLocal, s).index for s in below] == [
        stores[gi0.result],
        stores[gi1.result],
    ]
    _validates(method)


def test_stackify_takes_out_of_order_arguments_from_locals():
    """Two detector arrays built before either is set. Each is used once, but
    the first ``SetDetector`` would find the second array on top, so both are
    parked and each is reloaded for the detector that reads it."""
    cz, measure, await_m = _measure_await_chain()
    idx0 = sm.ConstInt(value=0)
    gi0 = sm.GetItem(array=await_m.result, indices=(idx0.result,))
    idx1 = sm.ConstInt(value=1)
    gi1 = sm.GetItem(array=await_m.result, indices=(idx1.result,))
    d0 = sm.NewArray(values=(gi0.result,), type_tag=1, dim0=1, dim1=0)
    d1 = sm.NewArray(values=(gi1.result,), type_tag=1, dim0=1, dim1=0)
    det0 = sm.SetDetector(array=d0.result)
    det1 = sm.SetDetector(array=d1.result)
    both = sm.NewArray(values=(det0.result, det1.result), type_tag=7, dim0=2, dim1=0)
    ci = sm.ConstInt(value=0)
    ret = func.Return(ci.result)
    method = _make_method(
        cz, measure, await_m, idx0, gi0, idx1, gi1, d0, d1, det0, det1, both, ci, ret
    )

    stmts = _stackify(method)

    for array, detector in ((d0, det0), (d1, det1)):
        load = stmts[stmts.index(detector) - 1]
        assert isinstance(load, sm.LoadLocal) and detector.array is load.result
        store = stmts[stmts.index(array) + 1]
        assert isinstance(store, sm.StoreLocal) and store.value is array.result
        assert store.index == load.index
    # The detector refs are consumed in order, so they stay on the stack.
    assert [a.owner for a in both.values] == [det0, det1]
    _validates(method)


def test_stackify_parks_the_results_above_a_spilled_one():
    """``measure 2`` leaves its first future on top. Spilling the second means
    taking the first off the stack on the way down and putting it back."""
    cz0 = sm.ConstZone(value=ZoneAddress(0))
    cz1 = sm.ConstZone(value=ZoneAddress(1))
    measure = sm.Measure(zones=(cz0.result, cz1.result))
    top, below = measure.results
    await_top = sm.AwaitMeasure(future=top)
    await_below_a = sm.AwaitMeasure(future=below)
    await_below_b = sm.AwaitMeasure(future=below)
    ci = sm.ConstInt(value=0)
    ret = func.Return(ci.result)
    method = _make_method(
        cz0, cz1, measure, await_top, await_below_a, await_below_b, ci, ret
    )

    stmts = _stackify(method)

    assert _locals_ops(stmts) == [
        ("store", 0),  # the top future, parked
        ("store", 1),  # the one that is spilled
        ("load", 0),  # the top future, put back
        ("load", 1),
        ("load", 1),
    ]
    assert (
        await_top.future is cast(sm.LoadLocal, stmts[stmts.index(await_top) - 1]).result
    )
    _validates(method)


# ── Integration: encode after stackify ────────────────────────────────────────


def test_stackify_encode_two_consumers():
    """After stackify, two-GetItem IR encodes to bytecode that parks the array
    in a local and reloads it for each read."""
    from bloqade.lanes.bytecode import Instruction, Program

    cz, measure, await_m = _measure_await_chain()
    idx0 = sm.ConstInt(value=0)
    gi0 = sm.GetItem(array=await_m.result, indices=(idx0.result,))
    idx1 = sm.ConstInt(value=1)
    gi1 = sm.GetItem(array=await_m.result, indices=(idx1.result,))
    na = sm.NewArray(values=(gi0.result, gi1.result), type_tag=1, dim0=2, dim1=0)
    ci = sm.ConstInt(value=0)
    ret = func.Return(ci.result)
    method = _make_method(cz, measure, await_m, idx0, gi0, idx1, gi1, na, ci, ret)

    stackify(method)

    expected = [
        Instruction.const_zone(0),
        Instruction.measure(1),
        Instruction.await_measure(),
        Instruction.store("undef", 0),
        Instruction.load("undef", 0),
        Instruction.const_int(0),
        Instruction.get_item(1),
        Instruction.load("undef", 0),
        Instruction.const_int(1),
        Instruction.get_item(1),
        Instruction.new_array(type_tag=1, dim0=2),
        Instruction.const_int(0),
        Instruction.return_(),
    ]

    encoded = dump_program(method)
    assert encoded.to_text() == Program(version=(1, 0), instructions=expected).to_text()
