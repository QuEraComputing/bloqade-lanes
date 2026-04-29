"""Tests for LayoutAnalysis.location_addresses collection via method table.

Verifies that the InitialLayoutMethods entry for place.NewLogicalQubit
populates LayoutAnalysis.location_addresses during the forward analysis run.
The heuristic invocation path is covered by Phase F integration tests.
"""

import pytest
from bloqade.analysis import address
from kirin import interp, ir, types
from kirin.dialects import func, ssacfg

from bloqade.lanes.analysis.layout import LayoutAnalysis
from bloqade.lanes.analysis.layout.analysis import LayoutHeuristicABC
from bloqade.lanes.dialects import place
from bloqade.lanes.layout.encoding import LocationAddress

# ---------------------------------------------------------------------------
# Minimal stub heuristic (never called by these unit tests)
# ---------------------------------------------------------------------------


class _StubHeuristic(LayoutHeuristicABC):
    def compute_layout(
        self,
        all_qubits: tuple[int, ...],
        stages: list[tuple[tuple[int, int], ...]],
        pinned: dict[int, LocationAddress] | None = None,
    ) -> tuple[LocationAddress, ...]:
        raise NotImplementedError("stub")  # pragma: no cover


# ---------------------------------------------------------------------------
# Helper: build a minimal ir.Method containing the given statements
# ---------------------------------------------------------------------------


def _build_method(stmts: list[ir.Statement]) -> ir.Method:
    """Wrap *stmts* in a callable ir.Method (void return)."""
    block = ir.Block(argtypes=(types.MethodType,))
    for stmt in stmts:
        block.stmts.append(stmt)
    none_stmt = func.ConstantNone()
    block.stmts.append(none_stmt)
    block.stmts.append(func.Return(none_stmt.result))

    region = ir.Region(blocks=block)
    fn = func.Function(
        sym_name="main",
        signature=func.Signature((), types.NoneType),
        slots=(),
        body=region,
    )
    dialects = ir.DialectGroup([ssacfg.dialect, func.dialect, place.dialect])
    return ir.Method(dialects=dialects, code=fn, sym_name="main", arg_names=[])


# ---------------------------------------------------------------------------
# Helper: build a minimal LayoutAnalysis with hand-crafted address_entries
# ---------------------------------------------------------------------------


def _make_analysis(
    address_entries: dict[ir.SSAValue, address.Address],
) -> LayoutAnalysis:
    """Return a LayoutAnalysis with the given address_entries pre-populated."""
    dialects = ir.DialectGroup([ssacfg.dialect, func.dialect, place.dialect])
    return LayoutAnalysis(
        dialects=dialects,
        heuristic=_StubHeuristic(),
        address_entries=address_entries,
        all_qubits=(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_location_addresses_no_pinned_qubits():
    """location_addresses is empty when no NewLogicalQubit has a location_address."""
    theta = ir.TestValue(type=types.Float)
    phi = ir.TestValue(type=types.Float)
    lam = ir.TestValue(type=types.Float)

    # location_address defaults to None
    q0 = place.NewLogicalQubit(theta, phi, lam)
    q1 = place.NewLogicalQubit(theta, phi, lam)

    method = _build_method([q0, q1])

    address_entries: dict[ir.SSAValue, address.Address] = {
        q0.result: address.AddressQubit(0),
        q1.result: address.AddressQubit(1),
    }
    analysis = _make_analysis(address_entries)
    analysis.run(method)

    assert analysis.location_addresses == {}


def test_location_addresses_mixed_pinned_and_unpinned():
    """location_addresses contains only pinned qubits, keyed by qubit ID."""
    theta = ir.TestValue(type=types.Float)
    phi = ir.TestValue(type=types.Float)
    lam = ir.TestValue(type=types.Float)

    pinned_addr = LocationAddress(1, 2, 0)

    # q0: pinned; q1: un-pinned; q2: pinned
    q0 = place.NewLogicalQubit(theta, phi, lam, location_address=pinned_addr)
    q1 = place.NewLogicalQubit(theta, phi, lam)  # location_address=None
    q2_addr = LocationAddress(3, 4, 0)
    q2 = place.NewLogicalQubit(theta, phi, lam, location_address=q2_addr)

    method = _build_method([q0, q1, q2])

    address_entries: dict[ir.SSAValue, address.Address] = {
        q0.result: address.AddressQubit(0),
        q1.result: address.AddressQubit(1),
        q2.result: address.AddressQubit(2),
    }
    analysis = _make_analysis(address_entries)
    analysis.run(method)

    assert set(analysis.location_addresses.keys()) == {0, 2}
    assert analysis.location_addresses[0] == pinned_addr
    assert analysis.location_addresses[2] == q2_addr
    # un-pinned qubit 1 must not appear
    assert 1 not in analysis.location_addresses


def test_location_addresses_duplicate_address_raises():
    """Two NewLogicalQubits pinned to the same LocationAddress raise ValueError."""
    theta = ir.TestValue(type=types.Float)
    phi = ir.TestValue(type=types.Float)
    lam = ir.TestValue(type=types.Float)

    shared_addr = LocationAddress(1, 2, 0)

    q0 = place.NewLogicalQubit(theta, phi, lam, location_address=shared_addr)
    q1 = place.NewLogicalQubit(theta, phi, lam, location_address=shared_addr)

    method = _build_method([q0, q1])

    address_entries: dict[ir.SSAValue, address.Address] = {
        q0.result: address.AddressQubit(0),
        q1.result: address.AddressQubit(1),
    }
    analysis = _make_analysis(address_entries)

    with pytest.raises((ValueError, interp.InterpreterError), match=str(shared_addr)):
        analysis.run(method)
