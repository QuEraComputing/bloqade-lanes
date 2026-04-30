"""Tests for the physical pipeline and NewPinnedQubit."""

from kirin import ir, rewrite

from bloqade import qubit
from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.dialects import place
from bloqade.lanes.rewrite.circuit2place import RewriteQubitsToPinnedQubits


def test_new_pinned_qubit_unpinned():
    """NewPinnedQubit with no address has location_address=None."""
    stmt = place.NewPinnedQubit()
    assert stmt.location_address is None


def test_new_pinned_qubit_pinned():
    """NewPinnedQubit with a LocationAddress stores it correctly."""
    addr = LocationAddress(word_id=4, site_id=2, zone_id=0)
    stmt = place.NewPinnedQubit(location_address=addr)
    assert stmt.location_address == addr


def test_rewrite_new_to_pinned():
    """qubit.stmts.New is lowered to NewPinnedQubit(location_address=None)."""
    block = ir.Block()
    plain_new = qubit.stmts.New()
    block.stmts.append(plain_new)

    rewrite.Walk(RewriteQubitsToPinnedQubits()).rewrite(block)

    pinned = [s for s in block.stmts if isinstance(s, place.NewPinnedQubit)]
    assert len(pinned) == 1
    assert pinned[0].location_address is None
