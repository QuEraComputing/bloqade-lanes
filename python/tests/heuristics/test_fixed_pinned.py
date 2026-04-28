"""Tests for PhysicalLayoutHeuristicFixed.compute_layout with pinned addresses."""

import pytest

from bloqade.lanes.arch import (
    ArchBlueprint,
    DeviceLayout,
    DiagonalWordTopology,
    HypercubeSiteTopology,
    ZoneSpec,
    build_arch,
)
from bloqade.lanes.heuristics.simple_layout import PhysicalLayoutHeuristicFixed
from bloqade.lanes.layout.encoding import LocationAddress


def _make_arch(num_rows: int = 5, sites_per_word: int = 16):
    bp = ArchBlueprint(
        zones={
            "gate": ZoneSpec(
                num_rows=num_rows,
                num_cols=2,
                entangling=True,
                word_topology=DiagonalWordTopology(),
                site_topology=HypercubeSiteTopology(),
            )
        },
        layout=DeviceLayout(sites_per_word=sites_per_word),
    )
    return build_arch(bp).arch


def _empty_stages() -> list[tuple[tuple[int, int], ...]]:
    return []


def test_fixed_pinned_mixed_qubits_land_at_requested_addresses():
    """Pinned qubits land at exactly their requested addresses; un-pinned avoid them."""
    strategy = PhysicalLayoutHeuristicFixed(arch_spec=_make_arch())
    # Pin qubit 2 to (word=0, site=5) and qubit 4 to (word=0, site=3)
    pin_2 = LocationAddress(0, 5)
    pin_4 = LocationAddress(0, 3)
    qubits = tuple(range(6))
    out = strategy.compute_layout(
        qubits,
        _empty_stages(),
        pinned={2: pin_2, 4: pin_4},
    )

    assert len(out) == len(qubits)

    # Pinned qubits are at their exact requested addresses
    assert out[2] == pin_2, f"qubit 2 expected {pin_2}, got {out[2]}"
    assert out[4] == pin_4, f"qubit 4 expected {pin_4}, got {out[4]}"

    # Un-pinned qubits do not collide with pinned addresses
    pinned_addrs = {pin_2, pin_4}
    for i, addr in enumerate(out):
        if i not in (2, 4):
            assert (
                addr not in pinned_addrs
            ), f"qubit {i} landed at a pinned address {addr}"

    # All addresses are distinct (no duplicates)
    assert len(set(out)) == len(out), "Duplicate addresses in layout output"


def test_fixed_pinned_none_preserves_baseline_behavior():
    """pinned=None and pinned={} both produce the same result as the default call."""
    strategy = PhysicalLayoutHeuristicFixed(arch_spec=_make_arch())
    qubits = tuple(range(10))
    stages = _empty_stages()

    baseline = strategy.compute_layout(qubits, stages)
    with_none = strategy.compute_layout(qubits, stages, pinned=None)
    with_empty = strategy.compute_layout(qubits, stages, pinned={})

    assert baseline == with_none
    assert baseline == with_empty


def test_fixed_pinned_over_constrained_raises():
    """All candidate sites pinned away → no room for extra qubit → ValueError."""
    # 1-row arch has exactly one home word with sites_per_word sites.
    strategy = PhysicalLayoutHeuristicFixed(arch_spec=_make_arch(num_rows=1))
    home_word = strategy.home_word_ids[0]
    sites_per_word = strategy.sites_per_home_word  # 16

    # Pin every available site to qubit ids 0..sites_per_word-1
    pinned: dict[int, LocationAddress] = {
        site_id: LocationAddress(home_word, site_id)
        for site_id in range(sites_per_word)
    }

    # One extra qubit (id = sites_per_word) has nowhere to go
    qubits = tuple(range(sites_per_word + 1))

    with pytest.raises(ValueError, match="no legal positions remain"):
        strategy.compute_layout(qubits, _empty_stages(), pinned=pinned)


def test_fixed_pinned_duplicate_addresses_raises():
    # Two qubits pinned to the same address must error.
    strategy = PhysicalLayoutHeuristicFixed(arch_spec=_make_arch())
    addr = LocationAddress(0, 0)
    with pytest.raises(ValueError, match="must be unique"):
        strategy.compute_layout(
            all_qubits=(0, 1, 2),
            stages=[],
            pinned={0: addr, 1: addr},
        )


def test_fixed_pinned_extra_keys_raises():
    # pinned contains a qubit ID not in all_qubits.
    strategy = PhysicalLayoutHeuristicFixed(arch_spec=_make_arch())
    with pytest.raises(ValueError, match="not in all_qubits"):
        strategy.compute_layout(
            all_qubits=(0, 1, 2),
            stages=[],
            pinned={99: LocationAddress(0, 0)},
        )
