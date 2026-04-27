"""Unit tests for ``get_shot_remapping``.

Hand-builds a small ``ArchSpec`` with a known Zone-0 location order and
checks that the standalone shot-remapping function projects nested
``IListResult[IListResult[MeasureResult]]`` values onto the expected
indices into the architecture's Zone-0 bitstring.
"""

import pytest

from bloqade.lanes import layout
from bloqade.lanes.analysis.atom import (
    Bottom,
    IListResult,
    MeasureResult,
    Value,
    get_shot_remapping,
)
from bloqade.lanes.bytecode._native import (
    Grid as RustGrid,
    LocationAddress as RustLocAddr,
    Mode as RustMode,
    Zone as RustZone,
)
from bloqade.lanes.layout import word
from bloqade.lanes.layout.encoding import LocationAddress, ZoneAddress

# Small toy architecture: one zone with one word containing four sites
# at y-positions 0..3. ``yield_zone_locations(ZoneAddress(0))`` will
# emit those four ``LocationAddress``es in order, giving a Zone-0
# bitstring of length 4.
_word = word.Word(sites=((0, 0), (0, 1), (0, 2), (0, 3)))
_rust_grid = RustGrid.from_positions([0.0], [0.0, 1.0, 2.0, 3.0])
_rust_zone = RustZone(
    name="test",
    grid=_rust_grid,
    site_buses=[],
    word_buses=[],
    words_with_site_buses=[],
    sites_with_word_buses=[],
)
_rust_mode = RustMode(
    name="all",
    zones=[0],
    bitstring_order=[
        RustLocAddr(0, 0, 0),
        RustLocAddr(0, 0, 1),
        RustLocAddr(0, 0, 2),
        RustLocAddr(0, 0, 3),
    ],
)
_ARCH = layout.ArchSpec.from_components(
    words=(_word,),
    zones=(_rust_zone,),
    modes=[_rust_mode],
)


def _ll(*items):
    """Helper to wrap a sequence of lattice values as an ``IListResult``."""
    return IListResult(tuple(items))


def _mr(qubit_id: int, site_id: int) -> MeasureResult:
    """Helper to build a ``MeasureResult`` at a Zone-0 ``site_id``."""
    return MeasureResult(qubit_id, LocationAddress(0, site_id, 0))


def test_zone0_location_order_matches_arch_iteration():
    """Sanity: confirm the test fixture's Zone-0 bitstring layout
    matches arch_spec.yield_zone_locations iteration."""
    locs = list(_ARCH.yield_zone_locations(ZoneAddress(0)))
    assert locs == [
        LocationAddress(0, 0, 0),
        LocationAddress(0, 1, 0),
        LocationAddress(0, 2, 0),
        LocationAddress(0, 3, 0),
    ]


def test_single_logical_qubit():
    """One logical qubit at sites 0 and 1: remapping yields [[0, 1]]."""
    return_value = _ll(_ll(_mr(0, 0), _mr(0, 1)))
    assert get_shot_remapping(return_value, _ARCH) == [[0, 1]]


def test_two_logical_qubits_skipping_a_site():
    """Two logical qubits using sites 0/2 and 1/3: skipping one site
    in between exercises that the table reports the *Zone-0* index, not
    a packed enumeration."""
    return_value = _ll(
        _ll(_mr(0, 0), _mr(0, 2)),
        _ll(_mr(1, 1), _mr(1, 3)),
    )
    assert get_shot_remapping(return_value, _ARCH) == [[0, 2], [1, 3]]


def test_outer_not_ilist_returns_none():
    """A non-IListResult outer (e.g. ``Bottom``) means the analysis
    didn't refine the SSA value past the bottom of the lattice; the
    function gives up and returns ``None`` (callers handle the
    diagnostic)."""
    assert get_shot_remapping(Bottom(), _ARCH) is None


def test_inner_not_ilist_returns_none():
    """Each logical entry must itself be an IListResult; otherwise
    return ``None``."""
    return_value = _ll(_mr(0, 0))  # outer ilist of MeasureResult, no nesting
    assert get_shot_remapping(return_value, _ARCH) is None


def test_innermost_not_measureresult_returns_none():
    """Innermost element must be a ``MeasureResult``; anything else
    means the analysis didn't refine that operand. Return ``None``."""
    return_value = _ll(_ll(_mr(0, 0), Value(False)))
    assert get_shot_remapping(return_value, _ARCH) is None


def test_unknown_location_address_returns_none():
    """A ``MeasureResult`` whose ``location_address`` isn't in the
    architecture's Zone-0 iteration is a sign of analysis/arch
    disagreement; return ``None`` rather than raising."""
    out_of_arch = LocationAddress(99, 0, 0)
    return_value = _ll(_ll(MeasureResult(0, out_of_arch)))
    assert get_shot_remapping(return_value, _ARCH) is None


def test_empty_logical_blocks():
    """Empty inner lists (no physical qubits) are valid; an empty
    outer list is also valid (no logical qubits)."""
    assert get_shot_remapping(_ll(_ll(), _ll()), _ARCH) == [[], []]
    assert get_shot_remapping(_ll(), _ARCH) == []


# ── Integration: end-to-end via compile_squin_to_move ──────────────────


@pytest.mark.slow
def test_get_shot_remapping_end_to_end_via_compile_squin_to_move():
    """End-to-end: compile a Steane logical kernel that returns its
    ``terminal_measure`` value, then run ``AtomInterpreter.get_shot_remapping``
    on the lowered move kernel and assert the table has the expected
    Steane shape and no overlapping indices."""
    from bloqade import qubit, squin
    from bloqade.gemini import logical as gemini_logical
    from bloqade.lanes.analysis.atom import AtomInterpreter
    from bloqade.lanes.arch.gemini import physical
    from bloqade.lanes.logical_mvp import compile_squin_to_move

    num_logical = 2

    @gemini_logical.kernel(aggressive_unroll=True)
    def main():
        reg = qubit.qalloc(num_logical)
        squin.h(reg[0])
        squin.cx(reg[0], reg[1])
        return gemini_logical.terminal_measure(reg)

    arch_spec = physical.get_arch_spec()
    physical_move = compile_squin_to_move(main, transversal_rewrite=True)

    interp = AtomInterpreter(physical_move.dialects, arch_spec=arch_spec)
    remapping = interp.get_shot_remapping(physical_move)

    # The analysis must refine to a concrete remapping for this
    # well-formed Steane kernel; ``None`` here would indicate an
    # analysis or pipeline regression rather than legitimate
    # soft-fail behaviour.
    assert remapping is not None

    # Steane [[7,1,3]] encodes one logical qubit into seven physical
    # qubits; ``terminal_measure`` over ``num_logical`` logical qubits
    # yields an outer list of length ``num_logical`` with seven
    # ``MeasureResult``s each.
    assert len(remapping) == num_logical
    assert all(len(per_qubit) == 7 for per_qubit in remapping)

    # No two physical qubits map to the same Zone-0 index.
    flat = [idx for per_qubit in remapping for idx in per_qubit]
    assert len(set(flat)) == len(
        flat
    ), f"physical qubit indices overlap across logical blocks: {remapping}"

    # All indices fall inside the Zone-0 bitstring.
    zone0_size = sum(1 for _ in arch_spec.yield_zone_locations(ZoneAddress(0)))
    assert all(
        0 <= idx < zone0_size for idx in flat
    ), f"index out of Zone-0 range [0, {zone0_size}): {flat}"
