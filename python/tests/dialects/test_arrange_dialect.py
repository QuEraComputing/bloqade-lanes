from kirin import lowering

from bloqade.gemini.common.dialects import arrange
from bloqade.gemini.common.dialects.arrange.stmts import MoveTo


def test_move_to_in_arrange_dialect():
    assert MoveTo in arrange.dialect.stmts


def test_move_to_attributes():
    assert hasattr(MoveTo, "qubits")
    assert hasattr(MoveTo, "locations")
    assert hasattr(MoveTo, "multi_move_warning")


def test_move_to_from_python_call_trait():
    assert any(isinstance(t, lowering.FromPythonCall) for t in MoveTo.traits)


def test_move_to_no_result():
    assert not hasattr(MoveTo, "result")


def test_place_move_to_is_quantum_stmt():
    from bloqade.lanes.dialects.place import MoveTo as PlaceMoveTo, QuantumStmt

    assert issubclass(PlaceMoveTo, QuantumStmt)


def test_place_move_to_attributes():
    from bloqade.lanes.dialects.place import MoveTo as PlaceMoveTo

    assert hasattr(PlaceMoveTo, "qubits")
    assert hasattr(PlaceMoveTo, "locations")
    assert hasattr(PlaceMoveTo, "multi_move_warning")
