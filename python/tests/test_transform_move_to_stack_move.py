"""Tests for the MoveToStackMove transformation.

Builds a ``move``-dialect kernel by running the inverse rewrite
(``RewriteStackMoveToMove``) on a hand-authored ``stack_move`` block, then
drives it through ``MoveToStackMove.emit`` / ``emit_bytecode`` and checks the
result is a canonicalized, bytecode-ready ``stack_move`` kernel.
"""

import pytest
from kirin import ir, types
from kirin.dialects import func, py
from kirin.rewrite import Walk

from bloqade.lanes.arch.gemini.logical import get_arch_spec
from bloqade.lanes.bytecode.decode import load_program
from bloqade.lanes.bytecode.encoding import LocationAddress, ZoneAddress
from bloqade.lanes.dialects import move, stack_move
from bloqade.lanes.rewrite.stack_move2move import RewriteStackMoveToMove
from bloqade.lanes.transform import MoveToStackMove

_ARCH = get_arch_spec()


def _move_kernel() -> ir.Method:
    """Build a small move-dialect ir.Method: InitialFill + LocalRz on one loc.

    Constructs the target ``stack_move`` IR, then runs the inverse rewrite so
    the resulting method holds equivalent ``move``-dialect statements (with
    Load/Store state threading) ready to feed into ``MoveToStackMove``.
    """
    a0 = LocationAddress(0, 0, 0)
    cf = stack_move.ConstFloat(value=0.5)
    cl = stack_move.ConstLoc(value=a0)
    initial_fill = stack_move.InitialFill(locations=(cl.result,))
    lrz = stack_move.LocalRz(rotation_angle=cf.result, locations=(cl.result,))
    none_stmt = func.ConstantNone()
    ret = func.Return(none_stmt.result)

    block = ir.Block(argtypes=(types.MethodType,))
    for s in [cf, cl, initial_fill, lrz, none_stmt, ret]:
        block.stmts.append(s)

    function = func.Function(
        sym_name="main",
        signature=func.Signature((), types.Any),
        slots=(),
        body=ir.Region(blocks=block),
    )
    dialects = ir.DialectGroup(
        [stack_move.dialect, move.dialect, func.dialect, py.dialect]
    )
    method = ir.Method(dialects=dialects, code=function, sym_name="main", arg_names=[])

    Walk(RewriteStackMoveToMove(arch_spec=_ARCH)).rewrite(method.code)
    return method


def _stmts(method: ir.Method) -> list[ir.Statement]:
    return list(method.callable_region.blocks[0].stmts)


def test_emit_lowers_move_to_stack_move():
    """emit() removes all move statements and produces stack_move equivalents."""
    method = _move_kernel()
    # sanity: the input is genuinely a move-dialect kernel
    assert any(isinstance(s, move.Fill) for s in _stmts(method))

    out = MoveToStackMove(arch_spec=_ARCH).emit(method)
    stmts = _stmts(out)

    assert not any(s.dialect is move.dialect for s in stmts)
    assert any(isinstance(s, stack_move.InitialFill) for s in stmts)
    assert any(isinstance(s, stack_move.LocalRz) for s in stmts)


def test_emit_is_stack_consistent():
    """Every SSA value in the emitted kernel is used at most once (stackified)."""
    out = MoveToStackMove(arch_spec=_ARCH).emit(_move_kernel())
    for stmt in _stmts(out):
        for result in stmt.results:
            assert len(result.uses) <= 1


def test_emit_does_not_mutate_input():
    """emit() operates on a copy; the input move kernel is untouched."""
    method = _move_kernel()
    before = [type(s).__name__ for s in _stmts(method)]
    MoveToStackMove(arch_spec=_ARCH).emit(method)
    after = [type(s).__name__ for s in _stmts(method)]
    assert before == after


def test_emit_no_raise_false_verifies():
    """emit(no_raise=False) runs verify() without raising on valid IR."""
    out = MoveToStackMove(arch_spec=_ARCH).emit(_move_kernel(), no_raise=False)
    assert any(isinstance(s, stack_move.InitialFill) for s in _stmts(out))


def test_emit_bytecode_round_trips():
    """emit_bytecode() produces a Program that load_program can decode."""
    xform = MoveToStackMove(arch_spec=_ARCH)
    prog = xform.emit_bytecode(_move_kernel())
    assert len(prog.instructions) > 0
    assert prog.version == (1, 0)

    decoded = load_program(prog)
    assert any(
        isinstance(s, stack_move.InitialFill)
        for s in decoded.callable_region.blocks[0].stmts
    )


def test_emit_bytecode_version_passthrough():
    """emit_bytecode() forwards the requested version onto the Program."""
    prog = MoveToStackMove(arch_spec=_ARCH).emit_bytecode(
        _move_kernel(), version=(2, 3)
    )
    assert prog.version == (2, 3)


def _move_kernel_with_unlowered_stmt() -> ir.Method:
    """A move kernel containing move.ConstZone, which RewriteMoveToStackMove
    does not lower (it passes through unchanged)."""
    cz = move.ConstZone(value=ZoneAddress(0))
    ret = func.Return(cz.result)
    block = ir.Block(argtypes=(types.MethodType,))
    for s in [cz, ret]:
        block.stmts.append(s)
    function = func.Function(
        sym_name="main",
        signature=func.Signature((), types.Any),
        slots=(),
        body=ir.Region(blocks=block),
    )
    dialects = ir.DialectGroup([stack_move.dialect, move.dialect, func.dialect])
    return ir.Method(dialects=dialects, code=function, sym_name="main", arg_names=[])


def test_emit_fails_fast_on_unlowered_move_stmt():
    """emit(no_raise=False) raises when a move statement is not lowered, naming
    the offending statement kind rather than failing later in dump_program."""
    method = _move_kernel_with_unlowered_stmt()
    with pytest.raises(ValueError, match="ConstZone"):
        MoveToStackMove(arch_spec=_ARCH).emit(method, no_raise=False)


def _shared_detector_kernel() -> ir.Method:
    """A move kernel shaped like real compiler output: one measurement, and
    two detectors that share a result, both built before either is set.

    The measurement array is read three times, the shared result feeds two
    detectors, and the first detector's array sits under the second's when it
    is set — every reason the compiler spills to a local, in one program.
    """
    from bloqade.decoders.dialects import annotate
    from kirin.dialects import ilist

    from bloqade.lanes.prelude import kernel

    zone = ZoneAddress(0)
    sites = list(_ARCH.yield_zone_locations(zone))[:3]
    load = move.Load()
    fill = move.Fill(load.result, location_addresses=tuple(sites))
    measure = move.Measure(fill.result, zone_addresses=(zone,))
    results = [
        move.GetFutureResult(measure.future, zone_address=zone, location_address=site)
        for site in sites
    ]
    d0 = ilist.New(values=(results[0].result, results[1].result))
    d1 = ilist.New(values=(results[1].result, results[2].result))
    coordinates = py.Constant(ilist.IList([0.0]))
    det0 = annotate.stmts.SetDetector(d0.result, coordinates.result)
    det1 = annotate.stmts.SetDetector(d1.result, coordinates.result)
    both = ilist.New(values=(det0.result, det1.result))
    store = move.Store(measure.result)
    ret = func.Return(both.result)

    block = ir.Block(argtypes=(types.MethodType,))
    for s in [load, fill, measure, *results, d0, d1, coordinates, det0, det1, both]:
        block.stmts.append(s)
    block.stmts.append(store)
    block.stmts.append(ret)
    function = func.Function(
        sym_name="main",
        signature=func.Signature((), types.Any),
        slots=(),
        body=ir.Region(blocks=block),
    )
    return ir.Method(
        dialects=kernel.union([annotate.dialect]),
        code=function,
        sym_name="main",
        arg_names=[],
    )


def test_emit_bytecode_spills_shared_measurements_and_validates():
    """The whole compile path, on the shape the spill pass exists for: the
    bytecode parks values in locals, and the Rust stack simulation — which
    checks every operand's tag and every typed `load`/`store` — accepts it."""
    prog = MoveToStackMove(arch_spec=_ARCH).emit_bytecode(
        _shared_detector_kernel(), no_raise=False
    )
    ops = [i.op_name() for i in prog.instructions]
    assert "store" in ops and "load" in ops
    assert "dup" not in ops
    prog.validate(stack=True)
