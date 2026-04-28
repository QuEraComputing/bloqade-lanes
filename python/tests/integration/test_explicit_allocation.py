"""End-to-end integration tests for explicit qubit allocation.

These tests exercise the full ``squin_to_move`` pipeline with kernels that mix
``squin.qalloc`` (un-pinned) and ``gemini.common.new_at`` (pinned) qubits, and
verify the resulting move IR carries the requested pinned addresses.
Failure-mode tests confirm const-prop, over-constraining and
semantic-illegality errors surface at compile time.
"""

import bloqade.squin as squin
import pytest
from kirin.dialects import ilist
from kirin.ir.exception import ValidationErrorGroup

import bloqade.gemini as gemini
from bloqade.gemini.common import new_at
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


# ---------------------------------------------------------------------------
# F2: regression gate — kernels with zero new_at compile unchanged.
# ---------------------------------------------------------------------------


def test_unannotated_kernel_unchanged():
    """A 2-qubit Bell-state kernel with no new_at calls produces a stable move
    IR shape. Assert characteristic counts + Fill addresses so any drift in
    the un-pinned compile path surfaces as a test failure.

    Captured against ``LogicalLayoutHeuristic`` + ``LogicalPlacementStrategyNoHome``
    on the gemini logical arch spec.
    """

    @gemini.logical.kernel(aggressive_unroll=True)
    def kernel():
        reg = squin.qalloc(2)
        squin.h(reg[0])
        squin.cx(reg[0], reg[1])
        gemini.logical.terminal_measure(reg)

    out = _compile(kernel)

    counts: dict[str, int] = {}
    for s in out.callable_region.walk():
        counts[type(s).__name__] = counts.get(type(s).__name__, 0) + 1

    expected_counts = {
        "CZ": 1,
        "Constant": 3,
        "ConstantNone": 1,
        "ConvertToPhysicalMeasurements": 1,
        "EndMeasure": 1,
        "Fill": 1,
        "GetFutureResult": 2,
        "Load": 2,
        "LocalR": 3,
        "LocalRz": 2,
        "LogicalInitialize": 1,
        "Move": 4,
        "Return": 1,
        "Store": 11,
    }
    assert counts == expected_counts, (
        f"un-pinned kernel statement counts drifted:\n"
        f"  expected: {expected_counts}\n"
        f"  actual:   {counts}"
    )

    fill_addrs = _collect_fill_addresses(out)
    expected_fill = (
        LocationAddress(word_id=0, site_id=0, zone_id=0),
        LocationAddress(word_id=2, site_id=0, zone_id=0),
    )
    assert fill_addrs == expected_fill


# ---------------------------------------------------------------------------
# F3: end-to-end failure modes.
# ---------------------------------------------------------------------------


def test_e2e_const_prop_failure_surfaces_at_compile_time():
    """new_at with a non-constant address arg makes squin_to_move raise with
    a 'compile-time constant' diagnostic."""

    @gemini.logical.kernel(verify=False)
    def kernel(z: int):
        q = new_at(z, 0, 0)  # noqa: F841

    with pytest.raises(ValidationErrorGroup) as exc_info:
        _compile(kernel)

    assert any("compile-time constant" in str(e) for e in exc_info.value.errors)


def test_e2e_overconstraining_pins_fail():
    """Asking for more total qubits than the arch supports — pins + un-pinned
    combined — should fail at LayoutAnalysis time before the move IR is
    produced.

    The logical arch has 10 home sites; pinning all 10 plus qalloc(1) makes
    11 total qubits, exceeding ``arch_spec.max_qubits`` (10). The heuristic's
    capacity check raises before the layout is computed.
    """
    arch_home_words = sorted({addr.word_id for addr in _LAYOUT.arch_spec.home_sites})

    @gemini.logical.kernel(verify=False)
    def kernel():
        # 10 pinned qubits at every home site.
        p0 = new_at(0, 0, 0)  # noqa: F841
        p1 = new_at(0, 2, 0)  # noqa: F841
        p2 = new_at(0, 4, 0)  # noqa: F841
        p3 = new_at(0, 6, 0)  # noqa: F841
        p4 = new_at(0, 8, 0)  # noqa: F841
        p5 = new_at(0, 10, 0)  # noqa: F841
        p6 = new_at(0, 12, 0)  # noqa: F841
        p7 = new_at(0, 14, 0)  # noqa: F841
        p8 = new_at(0, 16, 0)  # noqa: F841
        p9 = new_at(0, 18, 0)  # noqa: F841
        # One more qubit with no slot left.
        extra = squin.qalloc(1)  # noqa: F841

    # Sanity: the assumed home-word layout matches the arch.
    assert arch_home_words == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

    with pytest.raises(Exception, match="exceeds maximum supported"):
        _compile(kernel)


def test_e2e_pin_to_non_home_site_fails():
    """Pinning a qubit to an address that exists in the arch but is not a home
    site (e.g. word_id=1 — only even word_ids are home sites) should fail at
    layout time before the move IR is produced.
    """

    @gemini.logical.kernel(verify=False)
    def kernel():
        q = new_at(0, 1, 0)  # word=1 is not a home site  # noqa: F841

    with pytest.raises(
        (ValueError, ValidationErrorGroup),
        match="(?:home_positions|home_sites|Invalid location address)",
    ):
        _compile(kernel)
