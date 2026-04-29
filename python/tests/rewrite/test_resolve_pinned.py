"""Tests for ResolvePinnedAddresses rewrite rule.

Verifies that every place.NewLogicalQubit ends up with a non-None
location_address after the rewrite, and that already-pinned addresses
are left untouched.
"""

from bloqade.analysis import address
from kirin import ir, rewrite, types
from kirin.rewrite import abc

from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.dialects import place
from bloqade.lanes.rewrite.resolve_pinned import ResolvePinnedAddresses

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_block(*stmts: ir.Statement) -> ir.Block:
    """Return a Block containing *stmts* (no terminator needed for rewrite tests)."""
    return ir.Block(list(stmts))


def _theta_phi_lam() -> tuple[ir.TestValue, ir.TestValue, ir.TestValue]:
    return (
        ir.TestValue(type=types.Float),
        ir.TestValue(type=types.Float),
        ir.TestValue(type=types.Float),
    )


def _run_rewrite(
    block: ir.Block,
    address_entries: dict[ir.SSAValue, address.Address],
    initial_layout: tuple[LocationAddress, ...],
) -> abc.RewriteResult:
    rule = rewrite.Walk(
        ResolvePinnedAddresses(
            address_entries=address_entries,
            initial_layout=initial_layout,
        )
    )
    return rule.rewrite(block)


# ---------------------------------------------------------------------------
# Test: mixed kernel (some pinned, some unpinned)
# ---------------------------------------------------------------------------


def test_mixed_kernel():
    """After rewrite, every NewLogicalQubit has a non-None location_address.

    Pinned attributes are unchanged; un-pinned ones get the heuristic address.
    """
    theta, phi, lam = _theta_phi_lam()

    pinned_addr = LocationAddress(1, 2, 0)
    q0 = place.NewLogicalQubit(theta, phi, lam, location_address=pinned_addr)
    q1 = place.NewLogicalQubit(theta, phi, lam)  # un-pinned
    q2 = place.NewLogicalQubit(theta, phi, lam)  # un-pinned

    block = _make_block(q0, q1, q2)

    layout = (
        LocationAddress(0, 0, 0),  # index 0 -> q0 (pinned, should be ignored)
        LocationAddress(3, 1, 0),  # index 1 -> q1
        LocationAddress(5, 2, 0),  # index 2 -> q2
    )
    addr_entries: dict[ir.SSAValue, address.Address] = {
        q0.result: address.AddressQubit(0),
        q1.result: address.AddressQubit(1),
        q2.result: address.AddressQubit(2),
    }

    result = _run_rewrite(block, addr_entries, layout)

    # Rewrite should have mutated something
    assert result.has_done_something

    # q0: pinned — must remain unchanged
    assert q0.location_address == pinned_addr

    # q1, q2: filled from layout
    assert q1.location_address == layout[1]
    assert q2.location_address == layout[2]

    # Post-condition: no None addresses
    for stmt in block.stmts:
        if isinstance(stmt, place.NewLogicalQubit):
            assert stmt.location_address is not None


# ---------------------------------------------------------------------------
# Test: all-pinned kernel (rewrite is a no-op)
# ---------------------------------------------------------------------------


def test_all_pinned_kernel():
    """When every NewLogicalQubit is already pinned the rewrite does nothing."""
    theta, phi, lam = _theta_phi_lam()

    addr_a = LocationAddress(1, 0, 0)
    addr_b = LocationAddress(2, 1, 0)
    q0 = place.NewLogicalQubit(theta, phi, lam, location_address=addr_a)
    q1 = place.NewLogicalQubit(theta, phi, lam, location_address=addr_b)

    block = _make_block(q0, q1)

    # Layout has different values — should never be written
    layout = (
        LocationAddress(99, 99, 0),
        LocationAddress(88, 88, 0),
    )
    addr_entries: dict[ir.SSAValue, address.Address] = {
        q0.result: address.AddressQubit(0),
        q1.result: address.AddressQubit(1),
    }

    result = _run_rewrite(block, addr_entries, layout)

    # Nothing was changed
    assert not result.has_done_something

    # Attributes still point to the original pinned values
    assert q0.location_address == addr_a
    assert q1.location_address == addr_b


# ---------------------------------------------------------------------------
# Test: all-unpinned kernel
# ---------------------------------------------------------------------------


def test_all_unpinned_kernel():
    """Every un-pinned NewLogicalQubit gets filled from initial_layout."""
    theta, phi, lam = _theta_phi_lam()

    q0 = place.NewLogicalQubit(theta, phi, lam)
    q1 = place.NewLogicalQubit(theta, phi, lam)
    q2 = place.NewLogicalQubit(theta, phi, lam)

    block = _make_block(q0, q1, q2)

    layout = (
        LocationAddress(0, 0, 0),
        LocationAddress(1, 1, 0),
        LocationAddress(2, 2, 0),
    )
    addr_entries: dict[ir.SSAValue, address.Address] = {
        q0.result: address.AddressQubit(0),
        q1.result: address.AddressQubit(1),
        q2.result: address.AddressQubit(2),
    }

    result = _run_rewrite(block, addr_entries, layout)

    assert result.has_done_something

    assert q0.location_address == layout[0]
    assert q1.location_address == layout[1]
    assert q2.location_address == layout[2]

    # Post-condition
    for stmt in block.stmts:
        if isinstance(stmt, place.NewLogicalQubit):
            assert stmt.location_address is not None
