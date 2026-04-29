from kirin import ir

from bloqade.lanes.bytecode.encoding import ZoneAddress
from bloqade.lanes.dialects import move


def test_measure_fields():
    state = ir.TestValue()
    stmt = move.Measure(
        current_state=state, zone_addresses=(ZoneAddress(0), ZoneAddress(1))
    )
    assert stmt.current_state is state
    assert stmt.zone_addresses == (ZoneAddress(0), ZoneAddress(1))
    # move.Measure inherits StatefulStatement, so it produces two
    # results: the threaded state plus the measurement future.
    assert stmt.result is not None
    assert stmt.future is not None


def test_measure_has_consumes_state_trait():
    state = ir.TestValue()
    stmt = move.Measure(current_state=state, zone_addresses=(ZoneAddress(0),))
    assert stmt.get_trait(move.ConsumesState) is not None
