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
    # Registered on the movement dialect — catches wrong-dialect registration.
    assert Permute in movement.dialect.stmts


def test_permute_callable_lowering_wrapper():
    # movement.permute must be callable (it is the lowering wrapper for Permute).
    # The import itself exercises the re-export; this asserts the wrapper is usable.
    assert callable(movement.permute)
