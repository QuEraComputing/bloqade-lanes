from kirin import lowering

from bloqade.lanes.dialects import movement
from bloqade.lanes.dialects.movement import MoveTo


def test_movement_dialect_has_move_to():
    assert MoveTo in movement.dialect.stmts


def test_move_to_attributes():
    # Kirin statement fields are class-level descriptors, not dataclass fields
    assert hasattr(MoveTo, "qubits")
    assert hasattr(MoveTo, "locations")
    assert hasattr(MoveTo, "multi_move_warning")


def test_move_to_from_python_call_trait():
    assert any(isinstance(t, lowering.FromPythonCall) for t in MoveTo.traits)


def test_move_to_no_result():
    # MoveTo is a directive with no SSA result
    assert not hasattr(MoveTo, "result")


def test_place_move_to_is_quantum_stmt():
    from bloqade.lanes.dialects.place import MoveTo as PlaceMoveTo, QuantumStmt

    assert issubclass(PlaceMoveTo, QuantumStmt)


def test_place_move_to_attributes():
    from bloqade.lanes.dialects.place import MoveTo as PlaceMoveTo

    assert hasattr(PlaceMoveTo, "qubits")
    assert hasattr(PlaceMoveTo, "locations")
    assert hasattr(PlaceMoveTo, "multi_move_warning")
