from unittest.mock import MagicMock, patch

from bloqade.lanes.bytecode._native import (
    Grid as RustGrid,
    LocationAddress as RustLocAddr,
    Mode as RustMode,
    SiteBus,
    WordBus,
    Zone as RustZone,
)
from bloqade.lanes.layout.arch import ArchSpec
from bloqade.lanes.layout.encoding import (
    Direction,
    LocationAddress,
    SiteLaneAddress,
    WordLaneAddress,
    ZoneAddress,
)
from bloqade.lanes.layout.word import Word

word = Word(sites=((0, 0), (1, 0)))

rust_grid = RustGrid.from_positions([0.0, 1.0], [0.0])
rust_zone = RustZone(
    grid=rust_grid,
    site_buses=[SiteBus(src=[0], dst=[1])],
    word_buses=[WordBus(src=[0], dst=[1])],
    words_with_site_buses=[0],
    sites_with_word_buses=[0],
    entangling_pairs=[(0, 1)],
)
rust_mode = RustMode(
    name="all",
    zones=[0],
    bitstring_order=[
        RustLocAddr(0, 0, 0),
        RustLocAddr(0, 0, 1),
        RustLocAddr(0, 1, 0),
        RustLocAddr(0, 1, 1),
    ],
)

arch_spec = ArchSpec.from_components(
    words=(word, word),
    zones=(rust_zone,),
    modes=[rust_mode],
)


def test__get_site_bus_paths():
    # Should yield at least one path for valid word and bus
    paths = list(arch_spec._get_site_bus_paths([0], [0]))
    assert paths, "No site bus paths yielded"
    for path in paths:
        assert isinstance(path, tuple)
        assert all(isinstance(coord, tuple) and len(coord) == 2 for coord in path)


def test__get_word_bus_paths():
    # Should yield at least one path for valid bus
    paths = list(arch_spec._get_word_bus_paths([0]))
    assert paths, "No word bus paths yielded"
    for path in paths:
        assert isinstance(path, tuple)
        assert all(isinstance(coord, tuple) and len(coord) == 2 for coord in path)


def test_show_with_mocked_pyplot():
    with (
        patch("matplotlib.pyplot.gca") as mock_gca,
        patch("matplotlib.pyplot.show") as mock_show,
        patch("matplotlib.pyplot.plot") as mock_plot,
    ):
        mock_ax = MagicMock()
        mock_gca.return_value = mock_ax
        arch_spec.show(ax=mock_ax, show_words=[0], show_intra=[0], show_inter=[0])
        # Check that plot was called (either on ax or pyplot)
        assert mock_ax.plot.called or mock_plot.called
        # Check that plt.show was called
        assert mock_show.called


def test_max_qubits():
    assert arch_spec.max_qubits == 2 * 2 // 2


def test_yield_zone_locations():
    locs = list(arch_spec.yield_zone_locations(ZoneAddress(0)))
    assert all(isinstance(loc, LocationAddress) for loc in locs)


def test_get_path_and_position():
    lane = SiteLaneAddress(
        zone_id=0, word_id=0, site_id=0, bus_id=0, direction=Direction.FORWARD
    )
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
    lane1 = SiteLaneAddress(
        zone_id=0, word_id=0, site_id=0, bus_id=0, direction=Direction.FORWARD
    )
    lane2 = SiteLaneAddress(
        zone_id=0, word_id=0, site_id=1, bus_id=0, direction=Direction.FORWARD
    )
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
    lane = SiteLaneAddress(
        zone_id=0, word_id=0, site_id=0, bus_id=0, direction=Direction.FORWARD
    )
    errors = arch_spec.validate_lane(lane)
    assert isinstance(errors, set)


def test_get_endpoints_word_and_site():
    lane_site = SiteLaneAddress(
        zone_id=0, word_id=0, site_id=0, bus_id=0, direction=Direction.FORWARD
    )
    src, dst = arch_spec.get_endpoints(lane_site)
    assert isinstance(src, LocationAddress)
    assert isinstance(dst, LocationAddress)
    lane_word = WordLaneAddress(
        zone_id=0, word_id=0, site_id=0, bus_id=0, direction=Direction.FORWARD
    )
    src2, dst2 = arch_spec.get_endpoints(lane_word)
    assert isinstance(src2, LocationAddress)
    assert isinstance(dst2, LocationAddress)


def test_get_blockaded_location_paired():
    loc = LocationAddress(0, 0)
    result = arch_spec.get_blockaded_location(loc)
    assert result is not None


def test_get_blockaded_location_none():
    word_no_cz = Word(sites=((0, 0), (1, 0)))
    rust_grid_no_cz = RustGrid.from_positions([0.0, 1.0], [0.0])
    rust_zone_no_cz = RustZone(
        grid=rust_grid_no_cz,
        site_buses=[],
        word_buses=[],
        words_with_site_buses=[],
        sites_with_word_buses=[],
    )
    rust_mode_no_cz = RustMode(
        name="all",
        zones=[0],
        bitstring_order=[RustLocAddr(0, 0, 0), RustLocAddr(0, 0, 1)],
    )
    spec_no_cz = ArchSpec.from_components(
        words=(word_no_cz,),
        zones=(rust_zone_no_cz,),
        modes=[rust_mode_no_cz],
    )
    loc = LocationAddress(0, 0)
    assert spec_no_cz.get_blockaded_location(loc) is None


def test_capability_flags_default():
    assert arch_spec.feed_forward is False
    assert arch_spec.atom_reloading is False


def test_capability_flags_from_components():
    spec = ArchSpec.from_components(
        words=(word, word),
        zones=(rust_zone,),
        modes=[rust_mode],
        feed_forward=True,
        atom_reloading=True,
    )
    assert spec.feed_forward is True
    assert spec.atom_reloading is True
