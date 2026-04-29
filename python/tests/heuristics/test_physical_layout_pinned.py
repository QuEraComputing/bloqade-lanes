"""Tests for PhysicalLayoutHeuristicGraphPartitionCenterOut.compute_layout with pinned addresses."""

import pytest

from bloqade.lanes.arch import (
    ArchBlueprint,
    DeviceLayout,
    DiagonalWordTopology,
    HypercubeSiteTopology,
    ZoneSpec,
    build_arch,
)
from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.heuristics.physical.layout import (
    PhysicalLayoutHeuristicGraphPartitionCenterOut,
)


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


def _weighted_stages(
    edge_counts: dict[tuple[int, int], int],
) -> list[tuple[tuple[int, int], ...]]:
    stages: list[tuple[tuple[int, int], ...]] = []
    for (u, v), count in sorted(edge_counts.items()):
        for _ in range(count):
            stages.append(((u, v),))
    return stages


def test_pinned_mixed_qubits_land_at_requested_addresses():
    """Pinned qubits land at exactly their requested addresses; un-pinned avoid them."""
    strategy = PhysicalLayoutHeuristicGraphPartitionCenterOut(
        arch_spec=_make_arch(),
        max_words=1,
    )
    home_word = strategy.home_word_ids[0]
    zone_id = strategy.arch_spec.word_zone_map[home_word]

    pin_0 = LocationAddress(home_word, 7, zone_id)
    pin_2 = LocationAddress(home_word, 3, zone_id)

    qubits = tuple(range(6))
    stages = _weighted_stages(
        {
            (0, 1): 3,
            (1, 2): 3,
            (2, 3): 3,
            (3, 4): 3,
            (4, 5): 3,
        }
    )
    out = strategy.compute_layout(qubits, stages, pinned={0: pin_0, 2: pin_2})

    assert len(out) == len(qubits)

    # Pinned qubits are at their exact requested addresses
    assert out[0] == pin_0, f"qubit 0 expected {pin_0}, got {out[0]}"
    assert out[2] == pin_2, f"qubit 2 expected {pin_2}, got {out[2]}"

    # Un-pinned qubits do not collide with pinned addresses
    pinned_addrs = {pin_0, pin_2}
    for i, addr in enumerate(out):
        if i not in (0, 2):
            assert (
                addr not in pinned_addrs
            ), f"qubit {i} landed at a pinned address {addr}"

    # All addresses are distinct (no duplicates)
    assert len(set(out)) == len(out), "Duplicate addresses in layout output"


def test_pinned_none_and_empty_dict_behave_identically():
    """pinned=None and pinned={} both produce the same result."""
    strategy = PhysicalLayoutHeuristicGraphPartitionCenterOut(
        arch_spec=_make_arch(),
        max_words=1,
    )
    qubits = tuple(range(6))
    stages = _weighted_stages({(0, 1): 2, (2, 3): 2, (4, 5): 2})

    with_none = strategy.compute_layout(qubits, stages, pinned=None)
    with_empty = strategy.compute_layout(qubits, stages, pinned={})
    baseline = strategy.compute_layout(qubits, stages)

    assert with_none == baseline
    assert with_empty == baseline


def test_pinned_over_constrained_raises_no_legal_positions():
    """Pins fill all candidate slots → no room for the remaining qubit → ValueError."""
    # 1-row arch: exactly one home word.
    strategy = PhysicalLayoutHeuristicGraphPartitionCenterOut(
        arch_spec=_make_arch(num_rows=1, sites_per_word=4),
        max_words=1,
    )
    home_word = strategy.home_word_ids[0]
    zone_id = strategy.arch_spec.word_zone_map[home_word]
    sites_per_word = strategy.sites_per_partition  # 4

    # Pin every available site to qubit ids 0..sites_per_word-1
    pinned: dict[int, LocationAddress] = {
        site_id: LocationAddress(home_word, site_id, zone_id)
        for site_id in range(sites_per_word)
    }

    # One extra qubit (id = sites_per_word) has nowhere to go
    qubits = tuple(range(sites_per_word + 1))

    with pytest.raises(ValueError, match="no legal positions remain"):
        strategy.compute_layout(qubits, _empty_stages(), pinned=pinned)


def test_pinned_duplicate_addresses_raises():
    """Two qubits pinned to the same address must raise with 'must be unique'."""
    strategy = PhysicalLayoutHeuristicGraphPartitionCenterOut(
        arch_spec=_make_arch(),
        max_words=1,
    )
    home_word = strategy.home_word_ids[0]
    zone_id = strategy.arch_spec.word_zone_map[home_word]
    addr = LocationAddress(home_word, 0, zone_id)

    with pytest.raises(ValueError, match="must be unique"):
        strategy.compute_layout(
            all_qubits=(0, 1, 2),
            stages=[],
            pinned={0: addr, 1: addr},
        )


def test_pinned_extra_keys_raises():
    """pinned contains a qubit ID not in all_qubits → raises with 'not in all_qubits'."""
    strategy = PhysicalLayoutHeuristicGraphPartitionCenterOut(
        arch_spec=_make_arch(),
        max_words=1,
    )
    home_word = strategy.home_word_ids[0]
    zone_id = strategy.arch_spec.word_zone_map[home_word]

    with pytest.raises(ValueError, match="not in all_qubits"):
        strategy.compute_layout(
            all_qubits=(0, 1, 2),
            stages=[],
            pinned={99: LocationAddress(home_word, 0, zone_id)},
        )


def test_pinned_all_qubits_produces_exact_layout():
    """When every qubit is pinned the output is exactly those addresses, in qubit-ID order."""
    strategy = PhysicalLayoutHeuristicGraphPartitionCenterOut(
        arch_spec=_make_arch(),
        max_words=1,
    )
    home_word = strategy.home_word_ids[0]
    zone_id = strategy.arch_spec.word_zone_map[home_word]

    pinned = {
        0: LocationAddress(home_word, 2, zone_id),
        1: LocationAddress(home_word, 5, zone_id),
        2: LocationAddress(home_word, 9, zone_id),
    }
    out = strategy.compute_layout(
        all_qubits=(0, 1, 2),
        stages=[],
        pinned=pinned,
    )

    assert out == (pinned[0], pinned[1], pinned[2])


def test_pinned_out_of_arch_address_raises():
    """Pinning a qubit to an address not in arch's home_sites raises ValueError."""
    strategy = PhysicalLayoutHeuristicGraphPartitionCenterOut(
        arch_spec=_make_arch(),
        max_words=1,
    )
    # word_id=999, site_id=999 is far outside any valid arch address.
    bad_addr = LocationAddress(999, 999)
    with pytest.raises(ValueError, match="not valid home positions"):
        strategy.compute_layout(
            all_qubits=(0, 1, 2),
            stages=[],
            pinned={0: bad_addr},
        )
