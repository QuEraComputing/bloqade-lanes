"""Tests for the movement-dialect permute(qubits, perm) feature."""

from kirin import ir, lowering

from bloqade.gemini.common.dialects import movement
from bloqade.gemini.common.dialects.movement.stmts import Permute
from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.dialects import place
from bloqade.lanes.dialects.place import _resolve_permute_locations


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


def _loc(w, s):
    return LocationAddress(zone_id=0, word_id=w, site_id=s)


def test_place_permute_statement_shape():
    stmt = place.Permute(ir.TestValue(), qubits=(0, 1, 2), perm=(1, 2, 0))
    assert stmt.qubits == (0, 1, 2)
    assert stmt.perm == (1, 2, 0)


def test_resolve_permute_locations_cycle():
    layout = (_loc(0, 0), _loc(1, 0), _loc(2, 0))
    locations = _resolve_permute_locations(layout, qubits=(0, 1, 2), perm=(1, 2, 0))
    assert locations == (_loc(1, 0), _loc(2, 0), _loc(0, 0))


def test_resolve_permute_locations_identity():
    layout = (_loc(0, 0), _loc(1, 0))
    locations = _resolve_permute_locations(layout, qubits=(0, 1), perm=(0, 1))
    assert locations == (_loc(0, 0), _loc(1, 0))


def test_resolve_permute_locations_remapped_indices():
    # qubits are global layout indices (StaticPlacement merging remaps them);
    # perm indexes positions within the qubits tuple.
    layout = (_loc(9, 0), _loc(0, 0), _loc(1, 0), _loc(2, 0))
    locations = _resolve_permute_locations(layout, qubits=(1, 2, 3), perm=(2, 0, 1))
    assert locations == (_loc(2, 0), _loc(0, 0), _loc(1, 0))
