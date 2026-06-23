"""Tests for the movement-dialect permute(qubits, perm) feature."""

from kirin import ir, lowering

from bloqade.gemini.common.dialects import movement
from bloqade.gemini.common.dialects.movement.stmts import Permute


def test_permute_statement_shape():
    assert issubclass(Permute, ir.Statement)
    assert Permute.name == "permute"
    # Lowered from a Python call; NOT pure (it mutates placement state).
    assert any(isinstance(t, lowering.FromPythonCall) for t in Permute.traits)
    assert not any(isinstance(t, ir.Pure) for t in Permute.traits)


def test_permute_interface_exported():
    # `permute` is exported from the movement package and wraps Permute.
    assert movement.permute is not None
    assert callable(movement.permute)
