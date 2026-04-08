import pytest

from bloqade.lanes.arch.gemini import logical, physical
from bloqade.lanes.layout.arch import ArchSpec
from bloqade.lanes.layout.encoding import (
    LocationAddress,
    ZoneAddress,
)


def test_logical_architecture():
    arch = logical.get_arch_spec()
    assert len(arch.words) == 10
    assert len(arch.words[0].site_indices) == 2
    # Entangling zone splits into 2 sub-zones; buses duplicated per sub-zone
    assert len(arch.site_buses) == 2  # 1 per sub-zone × 2
    assert len(arch.word_buses) == 18  # 9 per sub-zone × 2
    assert arch.max_qubits == 10
    assert sorted(arch._home_words) == [0, 2, 4, 6, 8]


def test_physical_architecture():
    arch = physical.get_arch_spec()
    assert len(arch.words) == 10
    assert len(arch.words[0].site_indices) == 16
    # Entangling zone splits into 2 sub-zones; buses duplicated per sub-zone
    assert len(arch.site_buses) == 86  # 43 per sub-zone × 2
    assert len(arch.word_buses) == 18  # 9 per sub-zone × 2
    assert arch.max_qubits == 80


def test_get_zone_index():
    arch = logical.get_arch_spec()

    loc_addr = LocationAddress(zone_id=0, word_id=0, site_id=0)
    zone_id = ZoneAddress(0)
    index = arch.get_zone_index(loc_addr, zone_id)
    assert index == 0

    loc_addr = LocationAddress(zone_id=0, word_id=0, site_id=1)
    zone_id = ZoneAddress(0)
    index = arch.get_zone_index(loc_addr, zone_id)
    assert index == 1

    # Word 1 is at index 2 in zone 0 (after word 0 sites 0 and 1)
    loc_addr = LocationAddress(zone_id=0, word_id=1, site_id=0)
    zone_id = ZoneAddress(0)
    index = arch.get_zone_index(loc_addr, zone_id)
    assert index == 2


def test_entangling_zone_pairs():
    arch = logical.get_arch_spec()
    # Entangling gate zone splits into sub-zone 0 (even cols) and sub-zone 1 (odd cols)
    assert len(arch.entangling_zone_pairs) == 1
    assert arch.entangling_zone_pairs[0] == (0, 1)


def test_cz_partner():
    """Test that get_cz_partner works for the logical arch spec."""
    arch = logical.get_arch_spec()
    # In a self-entangling zone, CZ partners are determined by zone pairs
    loc = LocationAddress(0, 0, 0)
    partner = arch.get_cz_partner(loc)
    assert partner is not None


def invalid_locations():
    arch_spec = logical.get_arch_spec()
    yield arch_spec, LocationAddress(0, 10, 0), {"invalid location zone_id=0, word_id=10, site_id=0"}
    yield arch_spec, LocationAddress(0, 0, 2), {"invalid location zone_id=0, word_id=0, site_id=2"}


@pytest.mark.parametrize("arch_spec, location_address, message", invalid_locations())
def test_location_validation(
    arch_spec: ArchSpec, location_address: LocationAddress, message: set[str]
):
    assert message == arch_spec.validate_location(location_address)


def test_negative_location_ids_rejected():
    """Negative IDs are rejected at construction time by the Rust-backed type."""
    with pytest.raises(ValueError, match="must be non-negative"):
        LocationAddress(0, -1, 0)
    with pytest.raises(ValueError, match="must be non-negative"):
        LocationAddress(0, 0, -1)
