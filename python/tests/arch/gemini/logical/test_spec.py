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
    assert len(arch.site_buses) == 1
    assert len(arch.word_buses) == 9
    assert arch.max_qubits == 10
    assert sorted(arch._home_words) == [0, 2, 4, 6, 8]


def test_physical_architecture():
    arch = physical.get_arch_spec()
    assert len(arch.words) == 10
    assert len(arch.words[0].site_indices) == 16
    assert len(arch.site_buses) == 120
    assert len(arch.word_buses) == 9
    assert arch.max_qubits == 80


def test_get_zone_index():
    arch = logical.get_arch_spec()

    loc_addr = LocationAddress(word_id=0, site_id=0)
    zone_id = ZoneAddress(0)
    index = arch.get_zone_index(loc_addr, zone_id)
    assert index == 0

    loc_addr = LocationAddress(word_id=0, site_id=1)
    zone_id = ZoneAddress(0)
    index = arch.get_zone_index(loc_addr, zone_id)
    assert index == 1

    loc_addr = LocationAddress(word_id=1, site_id=0)
    zone_id = ZoneAddress(0)
    index = arch.get_zone_index(loc_addr, zone_id)
    assert index == 2


def test_entangling_zones():
    arch = logical.get_arch_spec()
    assert len(arch.entangling_zones) == 1
    pairs = arch.entangling_zones[0]
    assert (0, 1) in pairs
    assert (2, 3) in pairs
    assert (4, 5) in pairs
    assert (6, 7) in pairs


def invalid_locations():
    arch_spec = logical.get_arch_spec()
    yield arch_spec, LocationAddress(10, 0), {"invalid location word_id=10, site_id=0"}
    yield arch_spec, LocationAddress(0, 2), {"invalid location word_id=0, site_id=2"}


@pytest.mark.parametrize("arch_spec, location_address, message", invalid_locations())
def test_location_validation(
    arch_spec: ArchSpec, location_address: LocationAddress, message: set[str]
):
    assert message == arch_spec.validate_location(location_address)


def test_negative_location_ids_rejected():
    """Negative IDs are rejected at construction time by the Rust-backed type."""
    with pytest.raises(ValueError, match="must be non-negative"):
        LocationAddress(-1, 0)
    with pytest.raises(ValueError, match="must be non-negative"):
        LocationAddress(0, -1)
