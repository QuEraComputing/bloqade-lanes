import pytest
from kirin import ir, types
from kirin.dialects import func
from kirin.rewrite import Walk

from bloqade.lanes._prelude import kernel
from bloqade.lanes.arch.gemini.logical import get_arch_spec
from bloqade.lanes.dialects import move
from bloqade.lanes.rewrite.measure_lower import MeasureLower, MeasureLowerError


def test_single_zone_measure_rewrites_to_endmeasure():
    state = ir.TestValue()
    zone = ir.TestValue()
    m = move.Measure(current_state=state, zones=(zone,))
    block = ir.Block([m])
    # For this test we mock the analysis result — in real usage MeasureLower
    # is constructed via MeasureLower.from_method which runs AtomAnalysis.
    zone_sets: dict[move.Measure, frozenset[int]] = {}
    zone_sets[m] = frozenset({0})
    rule = MeasureLower(zone_sets=zone_sets, final_measurement_count=1)
    Walk(rule).rewrite(block)
    # m has been replaced by a move.EndMeasure in place.
    assert not any(isinstance(s, move.Measure) for s in block.stmts)
    assert any(isinstance(s, move.EndMeasure) for s in block.stmts)


def test_multi_zone_measure_raises():
    state = ir.TestValue()
    z0, z1 = ir.TestValue(), ir.TestValue()
    m = move.Measure(current_state=state, zones=(z0, z1))
    block = ir.Block([m])
    zone_sets: dict[move.Measure, frozenset[int]] = {}
    zone_sets[m] = frozenset({0, 1})
    rule = MeasureLower(zone_sets=zone_sets, final_measurement_count=1)
    with pytest.raises(MeasureLowerError):
        Walk(rule).rewrite(block)


def _build_measure_method(zones: tuple[move.ZoneAddress, ...]) -> ir.Method:
    """Build a move IR method by hand exercising move.Measure.

    Mirrors the helper in ``tests/analysis/atom/test_atom_interpreter.py`` —
    there is no Python-level callable exposed for ``move.Measure`` since it is
    emitted by ``lower_stack_move`` rather than written by users, so we
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
    zone_ssa: list[ir.SSAValue] = []
    for zone in zones:
        cz = move.ConstZone(value=zone)
        block.stmts.append(cz)
        zone_ssa.append(cz.result)
    measure = move.Measure(current_state=fill.result, zones=tuple(zone_ssa))
    block.stmts.append(measure)
    block.stmts.append(move.Store(fill.result))
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
