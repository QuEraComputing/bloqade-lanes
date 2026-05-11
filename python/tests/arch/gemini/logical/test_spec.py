import pytest

from bloqade.lanes.arch.gemini import logical, physical
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode.encoding import (
    LocationAddress,
)


def test_logical_architecture():
    arch = logical.get_arch_spec()
    assert len(arch.words) == 20
    assert len(arch.words[0].site_indices) == 1
    # Single zone with no site buses and 10 word buses
    assert len(arch.site_buses) == 0
    assert len(arch.word_buses) == 10
    assert arch.max_qubits == 10
    assert sorted(arch._home_words) == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]


def test_physical_architecture():
    arch = physical.get_arch_spec()
    assert len(arch.words) == 20
    assert len(arch.words[0].site_indices) == 8
    zone = arch._inner.zones[0]
    assert len(zone.site_buses) == 3  # 3D hypercube on 8 sites
    assert len(zone.word_buses) == 10  # 9 merged shifts + 1 cross-gap
    assert len(zone.entangling_pairs) == 10
    assert arch.max_qubits == 80  # 20 words × 8 sites // 2


def test_entangling_pairs():
    arch = logical.get_arch_spec()
    # Single zone with adjacent word pairs
    zones = arch._inner.zones
    expected_pairs = [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (8, 9),
        (10, 11),
        (12, 13),
        (14, 15),
        (16, 17),
        (18, 19),
    ]
    assert zones[0].entangling_pairs == expected_pairs


def test_cz_partner():
    """Test that get_cz_partner works for the logical arch spec."""
    arch = logical.get_arch_spec()
    # In a self-entangling zone, CZ partners are determined by zone pairs
    loc = LocationAddress(0, 0)
    partner = arch.get_cz_partner(loc)
    assert partner is not None


def invalid_locations():
    arch_spec = logical.get_arch_spec()
    yield arch_spec, LocationAddress(99, 0), {
        "invalid location zone_id=0, word_id=99, site_id=0"
    }
    yield arch_spec, LocationAddress(0, 2), {
        "invalid location zone_id=0, word_id=0, site_id=2"
    }


@pytest.mark.parametrize("arch_spec, location_address, message", invalid_locations())
def test_location_validation(
    arch_spec: ArchSpec, location_address: LocationAddress, message: set[str]
):
    errors = arch_spec.check_location_group([location_address])
    assert message == {str(e) for e in errors}


def test_negative_location_ids_rejected():
    """Negative IDs are rejected at construction time by the Rust-backed type."""
    with pytest.raises(ValueError, match="must be non-negative"):
        LocationAddress(-1, 0)
    with pytest.raises(ValueError, match="must be non-negative"):
        LocationAddress(0, -1)
