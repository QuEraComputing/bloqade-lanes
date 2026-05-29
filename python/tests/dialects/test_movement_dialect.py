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
