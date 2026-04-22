import pytest
from kirin import ir
from kirin.rewrite import Walk

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
