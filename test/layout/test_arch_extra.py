import pytest

# Minimal valid ArchSpec for testing
from bloqade.geometry.dialects.grid import Grid

from bloqade.lanes.layout.arch import ArchSpec, Bus
from bloqade.lanes.layout.encoding import (
    Direction,
    LocationAddress,
    SiteLaneAddress,
    WordLaneAddress,
    ZoneAddress,
)
from bloqade.lanes.layout.word import Word

word = Word(
    positions=Grid.from_positions([0.0, 1.0], [0.0]),
    site_indices=((0, 0), (1, 0)),
    has_cz=None,
)
arch_spec = ArchSpec(
    words=(word, word),
    zones=((0, 1),),
    measurement_mode_zones=(0,),
    entangling_zones=frozenset([0]),
    has_site_buses=frozenset([0]),
    has_word_buses=frozenset([0]),
    site_buses=(Bus(src=(0,), dst=(1,)),),
    word_buses=(Bus(src=(0,), dst=(1,)),),
)


def test_post_init_invalid_zone():
    with pytest.raises(ValueError):
        ArchSpec(
            words=(word, word),
            zones=((1,),),
            measurement_mode_zones=(0,),
            entangling_zones=frozenset([0]),
            has_site_buses=frozenset([0]),
            has_word_buses=frozenset([0]),
            site_buses=(Bus(src=(0,), dst=(1,)),),
            word_buses=(Bus(src=(0,), dst=(1,)),),
        )


def test_max_qubits():
    assert arch_spec.max_qubits == 2 * 2 // 2


def test_yield_zone_locations():
    locs = list(arch_spec.yield_zone_locations(ZoneAddress(0)))
    assert all(isinstance(loc, LocationAddress) for loc in locs)


def test_get_path_and_position():
    lane = SiteLaneAddress(word_id=0, site_id=0, bus_id=0, direction=Direction.FORWARD)
    path = arch_spec.get_path(lane)
    assert isinstance(path, tuple)
    src, dst = arch_spec.get_endpoints(lane)
    pos_src = arch_spec.get_position(src)
    assert isinstance(pos_src, tuple)


def test_get_zone_index():
    loc = LocationAddress(0, 0)
    zone = ZoneAddress(0)
    idx = arch_spec.get_zone_index(loc, zone)
    assert isinstance(idx, int)


def test_path_bounds_x_y_bounds():
    x_min, x_max, y_min, y_max = arch_spec.path_bounds()
    assert x_min <= x_max
    assert y_min <= y_max
    x_min2, x_max2 = arch_spec.x_bounds
    y_min2, y_max2 = arch_spec.y_bounds
    assert x_min2 <= x_max2
    assert y_min2 <= y_max2


def test_compatible_lane_error_and_lanes():
    lane1 = SiteLaneAddress(word_id=0, site_id=0, bus_id=0, direction=Direction.FORWARD)
    lane2 = SiteLaneAddress(word_id=0, site_id=1, bus_id=0, direction=Direction.FORWARD)
    errors = arch_spec.compatible_lane_error(lane1, lane2)
    assert isinstance(errors, set)
    assert arch_spec.compatible_lanes(lane1, lane2) in [True, False]


def test_validate_location():
    loc = LocationAddress(0, 0)
    errors = arch_spec.validate_location(loc)
    assert isinstance(errors, set)
    loc_invalid = LocationAddress(10, 0)
    errors_invalid = arch_spec.validate_location(loc_invalid)
    assert errors_invalid


def test_validate_lane():
    lane = SiteLaneAddress(word_id=0, site_id=0, bus_id=0, direction=Direction.FORWARD)
    errors = arch_spec.validate_lane(lane)
    assert isinstance(errors, set)


def test_get_endpoints_word_and_site():
    lane_site = SiteLaneAddress(
        word_id=0, site_id=0, bus_id=0, direction=Direction.FORWARD
    )
    src, dst = arch_spec.get_endpoints(lane_site)
    assert isinstance(src, LocationAddress)
    assert isinstance(dst, LocationAddress)
    lane_word = WordLaneAddress(
        word_id=0, site_id=0, bus_id=0, direction=Direction.FORWARD
    )
    src2, dst2 = arch_spec.get_endpoints(lane_word)
    assert isinstance(src2, LocationAddress)
    assert isinstance(dst2, LocationAddress)


def test_get_blockaded_location_none():
    loc = LocationAddress(0, 0)
    assert arch_spec.get_blockaded_location(loc) is None
