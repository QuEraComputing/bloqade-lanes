from kirin import ir

from bloqade.lanes.dialects import move


def test_measure_fields():
    state = ir.TestValue()
    z0 = ir.TestValue()
    z1 = ir.TestValue()
    stmt = move.Measure(current_state=state, zones=(z0, z1))
    assert stmt.current_state is state
    assert stmt.zones == (z0, z1)
    assert stmt.result is not None


def test_measure_has_consumes_state_trait():
    state = ir.TestValue()
    z = ir.TestValue()
    stmt = move.Measure(current_state=state, zones=(z,))
    assert stmt.get_trait(move.ConsumesState) is not None
