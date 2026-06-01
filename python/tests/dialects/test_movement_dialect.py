from kirin import lowering

from bloqade.lanes.dialects import place
from bloqade.lanes.dialects.place import UserMoveTo


def test_user_move_to_in_place_dialect():
    assert UserMoveTo in place.dialect.stmts


def test_user_move_to_attributes():
    assert hasattr(UserMoveTo, "qubits")
    assert hasattr(UserMoveTo, "locations")
    assert hasattr(UserMoveTo, "multi_move_warning")


def test_user_move_to_from_python_call_trait():
    assert any(isinstance(t, lowering.FromPythonCall) for t in UserMoveTo.traits)


def test_user_move_to_no_result():
    assert not hasattr(UserMoveTo, "result")


def test_place_move_to_is_quantum_stmt():
    from bloqade.lanes.dialects.place import MoveTo as PlaceMoveTo, QuantumStmt

    assert issubclass(PlaceMoveTo, QuantumStmt)


def test_place_move_to_attributes():
    from bloqade.lanes.dialects.place import MoveTo as PlaceMoveTo

    assert hasattr(PlaceMoveTo, "qubits")
    assert hasattr(PlaceMoveTo, "locations")
    assert hasattr(PlaceMoveTo, "multi_move_warning")
