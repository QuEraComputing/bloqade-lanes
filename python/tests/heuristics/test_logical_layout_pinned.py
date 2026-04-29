"""Tests for LogicalLayoutHeuristic and LogicalLayoutHeuristicRecencyWeighted
pinned-address handling in compute_layout."""

import pytest

from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.heuristics.logical.layout import (
    LogicalLayoutHeuristic,
    LogicalLayoutHeuristicRecencyWeighted,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_stages() -> list[tuple[tuple[int, int], ...]]:
    return []


def _heuristics() -> list:
    """Return one instance of each heuristic class."""
    return [
        LogicalLayoutHeuristic(),
        LogicalLayoutHeuristicRecencyWeighted(),
    ]


def _heuristic_ids() -> list[str]:
    return ["LogicalLayoutHeuristic", "LogicalLayoutHeuristicRecencyWeighted"]


def _home_sites_sorted(heuristic: LogicalLayoutHeuristic) -> list[LocationAddress]:
    """Return home sites sorted by (word_id, site_id) for deterministic indexing."""
    return sorted(heuristic.arch_spec.home_sites, key=lambda a: (a.word_id, a.site_id))


# ---------------------------------------------------------------------------
# Mixed-pinning: pinned IDs land at requested addresses; un-pinned avoid them
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("heuristic", _heuristics(), ids=_heuristic_ids())
def test_mixed_pinning_addresses_respected(
    heuristic: LogicalLayoutHeuristic,
) -> None:
    """Pinned qubits land at their requested addresses; un-pinned avoid those slots."""
    sites = _home_sites_sorted(heuristic)
    # Pin qubits 1 and 3 to the first two home sites.
    pin_1 = sites[0]
    pin_3 = sites[1]
    pinned = {1: pin_1, 3: pin_3}

    qubits = tuple(range(5))
    out = heuristic.compute_layout(qubits, _empty_stages(), pinned=pinned)

    assert len(out) == len(qubits), "output length must equal number of qubits"

    # Pinned qubits are at their exact addresses.
    assert out[1] == pin_1, f"qubit 1 expected {pin_1}, got {out[1]}"
    assert out[3] == pin_3, f"qubit 3 expected {pin_3}, got {out[3]}"

    # Un-pinned qubits do not collide with pinned addresses.
    pinned_addrs = {pin_1, pin_3}
    for i, addr in enumerate(out):
        if i not in (1, 3):
            assert (
                addr not in pinned_addrs
            ), f"qubit {i} landed at a pinned address {addr}"

    # All addresses are distinct.
    assert len(set(out)) == len(out), "duplicate addresses in layout output"


# ---------------------------------------------------------------------------
# Hard-failure: over-constrained → raises "no legal positions remain"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("heuristic", _heuristics(), ids=_heuristic_ids())
def test_over_constrained_raises(heuristic: LogicalLayoutHeuristic) -> None:
    """Pinning all-but-one site with two un-pinned qubits must raise InterpreterError.

    Since max_qubits == len(home_sites) for all supported archs, asking for
    len(home_sites) + 1 qubits will hit the max_qubits check first and raise
    InterpreterError rather than the pin-overflow ValueError.
    """
    sites = _home_sites_sorted(heuristic)
    total = len(sites)  # 10 for default arch

    # Pin qubits 0..total-2 to the first (total-1) home sites, leaving 1 slot free.
    # Then add two extra un-pinned qubits — only one slot remains.
    # However, this totals to (total-1) + 2 = total+1 qubits, which exceeds max_qubits.
    pinned = {i: sites[i] for i in range(total - 1)}
    # all_qubits = 0..total+1 (total-1 pinned + 2 un-pinned)
    qubits = tuple(range(total + 1))

    # Hits InterpreterError because len(all_qubits) > max_qubits
    with pytest.raises(Exception) as exc_info:
        heuristic.compute_layout(qubits, _empty_stages(), pinned=pinned)

    # Verify it's the max_qubits check, not a pin overflow.
    from kirin import interp

    assert isinstance(exc_info.value, interp.InterpreterError)
    assert "exceeds maximum supported" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Duplicate pinned addresses → raises "must be unique"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("heuristic", _heuristics(), ids=_heuristic_ids())
def test_duplicate_pinned_addresses_raises(
    heuristic: LogicalLayoutHeuristic,
) -> None:
    """Two qubits pinned to the same address must raise ValueError."""
    sites = _home_sites_sorted(heuristic)
    addr = sites[0]
    with pytest.raises(ValueError, match="must be unique"):
        heuristic.compute_layout(
            all_qubits=(0, 1, 2),
            stages=[],
            pinned={0: addr, 1: addr},
        )


# ---------------------------------------------------------------------------
# Extra keys in pinned (not in all_qubits) → raises "not in all_qubits"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("heuristic", _heuristics(), ids=_heuristic_ids())
def test_extra_keys_in_pinned_raises(heuristic: LogicalLayoutHeuristic) -> None:
    """pinned dict containing a qubit ID absent from all_qubits must raise."""
    sites = _home_sites_sorted(heuristic)
    with pytest.raises(ValueError, match="not in all_qubits"):
        heuristic.compute_layout(
            all_qubits=(0, 1, 2),
            stages=[],
            pinned={99: sites[0]},
        )


# ---------------------------------------------------------------------------
# Empty / None equivalence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("heuristic", _heuristics(), ids=_heuristic_ids())
def test_none_and_empty_pinned_produce_same_result(
    heuristic: LogicalLayoutHeuristic,
) -> None:
    """pinned=None, pinned={}, and the default (no arg) all produce identical layouts."""
    qubits = tuple(range(6))
    stages = _empty_stages()

    default_out = heuristic.compute_layout(qubits, stages)
    none_out = heuristic.compute_layout(qubits, stages, pinned=None)
    empty_out = heuristic.compute_layout(qubits, stages, pinned={})

    assert default_out == none_out, "pinned=None differs from default"
    assert default_out == empty_out, "pinned={} differs from default"


# ---------------------------------------------------------------------------
# Arch validation: pinned address outside arch home_sites → raises
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("heuristic", _heuristics(), ids=_heuristic_ids())
def test_pinned_out_of_arch_address_raises(
    heuristic: LogicalLayoutHeuristic,
) -> None:
    """Pinning a qubit to an address not in arch's home_sites raises ValueError."""
    # word_id=999, site_id=999 is far outside any valid arch address.
    bad_addr = LocationAddress(999, 999)
    with pytest.raises(ValueError, match="not valid home positions"):
        heuristic.compute_layout(
            all_qubits=(0, 1, 2),
            stages=[],
            pinned={0: bad_addr},
        )
