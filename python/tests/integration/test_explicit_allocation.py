"""End-to-end integration tests for explicit qubit allocation.

These tests exercise the full ``squin_to_move`` pipeline with kernels that mix
``squin.qalloc`` (un-pinned) and ``gemini.logical.dialects.operations.new_at``
(pinned) qubits, and verify the resulting move IR carries the requested pinned
addresses.
"""

import bloqade.squin as squin
from kirin.dialects import ilist

import bloqade.gemini as gemini
from bloqade.gemini.logical.dialects.operations import new_at
from bloqade.lanes.dialects import move
from bloqade.lanes.heuristics.logical.layout import LogicalLayoutHeuristic
from bloqade.lanes.heuristics.logical.placement import LogicalPlacementStrategyNoHome
from bloqade.lanes.layout.encoding import LocationAddress
from bloqade.lanes.upstream import squin_to_move
from bloqade.lanes.validation.address import Validation

_LAYOUT = LogicalLayoutHeuristic()
_PLACEMENT = LogicalPlacementStrategyNoHome()


def _compile(kernel):
    return squin_to_move(
        kernel,
        layout_heuristic=_LAYOUT,
        placement_strategy=_PLACEMENT,
    )


def _collect_fill_addresses(mt) -> tuple[LocationAddress, ...]:
    fills = [s for s in mt.callable_region.walk() if isinstance(s, move.Fill)]
    assert len(fills) == 1, f"expected exactly one move.Fill, got {len(fills)}"
    return fills[0].location_addresses


def test_e2e_mixed_pinning():
    """A kernel with mixed qalloc + new_at qubits compiles end-to-end and the
    pinned addresses appear in the resulting move.Fill, while unpinned qubits
    receive heuristic-chosen addresses that don't collide with the pins.
    """
    pin_a = LocationAddress(word_id=0, site_id=0, zone_id=0)
    pin_b = LocationAddress(word_id=4, site_id=0, zone_id=0)

    @gemini.logical.kernel(aggressive_unroll=True)
    def kernel():
        a = new_at(0, 0, 0)
        b = new_at(0, 4, 0)
        reg = squin.qalloc(2)
        squin.broadcast.cz(ilist.IList([a]), ilist.IList([reg[0]]))
        squin.broadcast.cz(ilist.IList([b]), ilist.IList([reg[1]]))
        gemini.logical.terminal_measure(ilist.IList([a, b, reg[0], reg[1]]))

    out = _compile(kernel)

    fill_addrs = _collect_fill_addresses(out)
    assert pin_a in fill_addrs
    assert pin_b in fill_addrs
    # Unpinned qubits must not collide with the pinned addresses.
    assert len(set(fill_addrs)) == len(fill_addrs)

    # Post-compile lanes validator should accept the result.
    arch_spec = _LAYOUT.arch_spec
    _, errors = Validation(arch_spec=arch_spec).run(out)
    assert errors == [], f"post-compile validator reported errors: {errors}"
