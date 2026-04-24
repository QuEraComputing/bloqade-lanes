from kirin import ir, types
from kirin.analysis.forward import ForwardFrame
from kirin.dialects import func
from kirin.rewrite import Walk

from bloqade.lanes._prelude import kernel
from bloqade.lanes.analysis.atom.lattice import MeasureFuture
from bloqade.lanes.arch.gemini.logical import get_arch_spec
from bloqade.lanes.dialects import move
from bloqade.lanes.layout.encoding import ZoneAddress
from bloqade.lanes.rewrite.measure_lower import MeasureLower


def _rule_with_future(stmt: move.Measure, future: MeasureFuture) -> MeasureLower:
    """Build a MeasureLower whose frame maps ``stmt.future`` to ``future``.

    Used by the isolated-IR tests that exercise the rewrite without
    going through a full AtomAnalysis run. The frame's code node is
    unused by the rewrite — we pass ``stmt`` just to satisfy the
    ForwardFrame(code: Statement) constructor.
    """
    frame: ForwardFrame = ForwardFrame(stmt)
    frame.entries[stmt.future] = future
    return MeasureLower(frame=frame)


def test_single_zone_measure_rewrites_to_endmeasure():
    state = ir.TestValue()
    zone = ZoneAddress(0)
    m = move.Measure(current_state=state, zone_addresses=(zone,))
    block = ir.Block([m])
    rule = _rule_with_future(m, MeasureFuture(results={zone: {}}, measurement_count=1))
    Walk(rule).rewrite(block)
    # m has been replaced by a move.EndMeasure in place.
    assert not any(isinstance(s, move.Measure) for s in block.stmts)
    assert any(isinstance(s, move.EndMeasure) for s in block.stmts)


def test_multi_zone_measure_is_skipped():
    """Rewrite gives up — no EndMeasure, Measure kept as-is."""
    state = ir.TestValue()
    zone0, zone1 = ZoneAddress(0), ZoneAddress(1)
    m = move.Measure(
        current_state=state,
        zone_addresses=(zone0, zone1),
    )
    block = ir.Block([m])
    rule = _rule_with_future(
        m,
        MeasureFuture(results={zone0: {}, zone1: {}}, measurement_count=1),
    )
    Walk(rule).rewrite(block)
    assert any(isinstance(s, move.Measure) for s in block.stmts)
    assert not any(isinstance(s, move.EndMeasure) for s in block.stmts)


def test_non_first_final_measurement_is_skipped():
    """Rewrite gives up — move.Measure left untouched when it isn't
    the first/only final measurement in the program."""
    state = ir.TestValue()
    zone = ZoneAddress(0)
    m = move.Measure(current_state=state, zone_addresses=(zone,))
    block = ir.Block([m])
    rule = _rule_with_future(m, MeasureFuture(results={zone: {}}, measurement_count=2))
    Walk(rule).rewrite(block)
    assert any(isinstance(s, move.Measure) for s in block.stmts)
    assert not any(isinstance(s, move.EndMeasure) for s in block.stmts)


def test_missing_future_is_skipped():
    """Rewrite gives up — move.Measure left untouched when the frame
    has no MeasureFuture for its future SSA."""
    state = ir.TestValue()
    m = move.Measure(current_state=state, zone_addresses=(ZoneAddress(0),))
    block = ir.Block([m])
    frame: ForwardFrame = ForwardFrame(m)
    rule = MeasureLower(frame=frame)
    Walk(rule).rewrite(block)
    assert any(isinstance(s, move.Measure) for s in block.stmts)
    assert not any(isinstance(s, move.EndMeasure) for s in block.stmts)


def _build_measure_method(zones: tuple[move.ZoneAddress, ...]) -> ir.Method:
    """Build a move IR method by hand exercising move.Measure.

    Mirrors the helper in ``tests/analysis/atom/test_atom_interpreter.py`` —
    there is no Python-level callable exposed for ``move.Measure`` since it is
    emitted by ``stack_move2move`` rather than written by users, so we
    assemble the IR directly.
    """
    block = ir.Block(argtypes=(types.MethodType,))
    load = move.Load()
    block.stmts.append(load)
    fill = move.Fill(
        load.result,
        location_addresses=(move.LocationAddress(0, 0),),
    )
    block.stmts.append(fill)
    measure = move.Measure(current_state=fill.result, zone_addresses=zones)
    block.stmts.append(measure)
    block.stmts.append(move.Store(measure.result))
    none_stmt = func.ConstantNone()
    block.stmts.append(none_stmt)
    block.stmts.append(func.Return(none_stmt.result))

    region = ir.Region(blocks=block)
    function = func.Function(
        sym_name="main",
        signature=func.Signature((), types.NoneType),
        slots=(),
        body=region,
    )
    return ir.Method(
        dialects=kernel,
        code=function,
        sym_name="main",
        arg_names=[],
    )


def test_measure_lower_runs_analysis_end_to_end():
    """End-to-end: build a move-dialect method with a single Measure,
    call MeasureLower.from_method, and assert the Measure was rewritten to
    EndMeasure."""
    method = _build_measure_method((move.ZoneAddress(0),))
    rule = MeasureLower.from_method(method, arch_spec=get_arch_spec())
    block = method.callable_region.blocks[0]
    Walk(rule).rewrite(block)
    assert any(isinstance(s, move.EndMeasure) for s in block.stmts)
    assert not any(isinstance(s, move.Measure) for s in block.stmts)
