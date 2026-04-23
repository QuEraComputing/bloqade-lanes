import pytest
from kirin import ir, types
from kirin.dialects import func
from kirin.rewrite import Walk

from bloqade.lanes._prelude import kernel
from bloqade.lanes.arch.gemini.logical import get_arch_spec
from bloqade.lanes.dialects import move
from bloqade.lanes.layout.encoding import ZoneAddress
from bloqade.lanes.rewrite.measure_lower import MeasureLower, MeasureLowerError


def test_single_zone_measure_rewrites_to_endmeasure():
    state = ir.TestValue()
    m = move.Measure(current_state=state, zone_addresses=(ZoneAddress(0),))
    block = ir.Block([m])
    rule = MeasureLower(final_measurement_count=1)
    Walk(rule).rewrite(block)
    # m has been replaced by a move.EndMeasure in place.
    assert not any(isinstance(s, move.Measure) for s in block.stmts)
    assert any(isinstance(s, move.EndMeasure) for s in block.stmts)


def test_multi_zone_measure_raises():
    state = ir.TestValue()
    m = move.Measure(
        current_state=state,
        zone_addresses=(ZoneAddress(0), ZoneAddress(1)),
    )
    block = ir.Block([m])
    rule = MeasureLower(final_measurement_count=1)
    with pytest.raises(MeasureLowerError):
        Walk(rule).rewrite(block)


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
