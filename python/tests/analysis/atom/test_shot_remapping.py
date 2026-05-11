"""Unit tests for ``get_shot_remapping``.

Hand-builds a small ``ArchSpec`` with a known Zone-0 location order and
checks that the standalone shot-remapping function projects nested
``IListResult[IListResult[MeasureResult]]`` values onto the expected
flat list of indices into the architecture's Zone-0 bitstring,
emitting structured diagnostics on the soft-fail paths.
"""

import pytest

from bloqade.lanes.analysis.atom import (
    Bottom,
    IListResult,
    MeasureResult,
    ShotRemappingDiagnostic,
    ShotRemappingErr,
    ShotRemappingOk,
    Value,
)
from bloqade.lanes.analysis.atom._shot_remapping import get_shot_remapping
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode import word
from bloqade.lanes.bytecode._native import (
    Grid as RustGrid,
    LocationAddress as RustLocAddr,
    Mode as RustMode,
    Zone as RustZone,
)
from bloqade.lanes.bytecode.encoding import LocationAddress, ZoneAddress

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
_ARCH = ArchSpec.from_components(
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
    """One logical qubit at sites 0 and 1: remapping yields [0, 1]."""
    return_value = _ll(_ll(_mr(0, 0), _mr(0, 1)))
    result = get_shot_remapping(return_value, _ARCH)
    assert isinstance(result, ShotRemappingOk)
    assert result.mapping == [0, 1]


def test_two_logical_qubits_skipping_a_site():
    """Two logical qubits using sites 0/2 and 1/3: the result is the
    flat row-major concatenation, and skipping a site in between
    exercises that the table reports the *Zone-0* index, not a packed
    enumeration."""
    return_value = _ll(
        _ll(_mr(0, 0), _mr(0, 2)),
        _ll(_mr(1, 1), _mr(1, 3)),
    )
    result = get_shot_remapping(return_value, _ARCH)
    assert isinstance(result, ShotRemappingOk)
    assert result.mapping == [0, 2, 1, 3]


def test_outer_not_ilist_returns_diagnostic():
    """A non-IListResult outer (e.g. ``Bottom``) means the analysis
    didn't refine the SSA value past the bottom of the lattice; the
    function gives up and returns ``ShotRemappingErr`` identifying
    the bad outer value."""
    bottom = Bottom()
    result = get_shot_remapping(bottom, _ARCH)
    assert isinstance(result, ShotRemappingErr)
    assert isinstance(result.diagnostic, ShotRemappingDiagnostic)
    assert "outer" in result.diagnostic.message
    assert result.diagnostic.offending_value is bottom


def test_inner_not_ilist_returns_diagnostic():
    """Each logical entry must itself be an IListResult; otherwise the
    diagnostic identifies which logical block went wrong."""
    bad_logical = _mr(0, 0)
    return_value = _ll(bad_logical)  # outer ilist of MeasureResult, no nesting
    result = get_shot_remapping(return_value, _ARCH)
    assert isinstance(result, ShotRemappingErr)
    assert "logical[0]" in result.diagnostic.message
    assert result.diagnostic.offending_value is bad_logical


def test_innermost_not_measureresult_returns_diagnostic():
    """Innermost element must be a ``MeasureResult``; the diagnostic
    points at the offending physical index."""
    bad_physical = Value(False)
    return_value = _ll(_ll(_mr(0, 0), bad_physical))
    result = get_shot_remapping(return_value, _ARCH)
    assert isinstance(result, ShotRemappingErr)
    assert "logical[0].physical[1]" in result.diagnostic.message
    assert result.diagnostic.offending_value is bad_physical


def test_unknown_location_address_returns_diagnostic():
    """A ``MeasureResult`` whose ``location_address`` isn't in the
    architecture's Zone-0 iteration is a sign of analysis/arch
    disagreement; return a diagnostic carrying the offending address."""
    out_of_arch = LocationAddress(99, 0, 0)
    return_value = _ll(_ll(MeasureResult(0, out_of_arch)))
    result = get_shot_remapping(return_value, _ARCH)
    assert isinstance(result, ShotRemappingErr)
    assert "logical[0].physical[0]" in result.diagnostic.message
    assert "Zone-0" in result.diagnostic.message
    assert result.diagnostic.offending_value == out_of_arch


def test_empty_logical_blocks():
    """Empty inner lists (no physical qubits) are valid; an empty
    outer list is also valid (no logical qubits). Both produce an
    empty mapping."""
    empty_inner = get_shot_remapping(_ll(_ll(), _ll()), _ARCH)
    assert isinstance(empty_inner, ShotRemappingOk)
    assert empty_inner.mapping == []
    empty_outer = get_shot_remapping(_ll(), _ARCH)
    assert isinstance(empty_outer, ShotRemappingOk)
    assert empty_outer.mapping == []


# ── Integration: end-to-end via compile_squin_to_move ──────────────────


@pytest.mark.slow
def test_get_shot_remapping_end_to_end_via_compile_squin_to_move():
    """End-to-end: compile a Steane logical kernel that returns its
    ``terminal_measure`` value, then run ``AtomInterpreter.get_shot_remapping``
    on the lowered move kernel and assert the flat index list matches
    the analysis output's measurement-leaf count and has no overlapping
    indices."""
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
    result = interp.get_shot_remapping(physical_move)

    # The analysis must refine to a concrete remapping for this
    # well-formed kernel; an Err here would indicate an analysis or
    # pipeline regression rather than legitimate soft-fail behaviour.
    assert isinstance(
        result, ShotRemappingOk
    ), f"unexpected diagnostic: {getattr(result, 'diagnostic', None)}"
    remapping = result.mapping

    # The remapping length should equal the number of MeasureResult
    # leaves in the analysis output. Re-run the analysis here to
    # derive the expected length from the output shape rather than
    # hard-coding the code's block size, so the assertion stays
    # honest if the encoder changes.
    _, output = interp.run(physical_move)
    assert isinstance(output, IListResult)
    expected_len = sum(
        len(logical.data) for logical in output.data if isinstance(logical, IListResult)
    )
    assert len(remapping) == expected_len

    # No two physical qubits map to the same Zone-0 index.
    assert len(set(remapping)) == len(
        remapping
    ), f"physical qubit indices overlap: {remapping}"

    # All indices fall inside the Zone-0 bitstring.
    zone0_size = sum(1 for _ in arch_spec.yield_zone_locations(ZoneAddress(0)))
    assert all(
        0 <= idx < zone0_size for idx in remapping
    ), f"index out of Zone-0 range [0, {zone0_size}): {remapping}"
