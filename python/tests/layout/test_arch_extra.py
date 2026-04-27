from unittest.mock import MagicMock, patch

import pytest

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
    LaneAddress,
    LocationAddress,
    MoveType,
    SiteLaneAddress,
    WordLaneAddress,
    ZoneAddress,
)
from bloqade.lanes.layout.word import Word
from bloqade.lanes.visualize.arch import ArchVisualizer

word = Word(sites=((0, 0), (1, 0)))

rust_grid = RustGrid.from_positions([0.0, 1.0], [0.0])
rust_zone = RustZone(
    name="test",
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


def test_show_with_mocked_pyplot():
    """``ArchSpec.show`` delegates to ``ArchVisualizer.show``; the existing
    API must still drive matplotlib and call ``plt.show``."""
    with (
        patch("matplotlib.pyplot.gca") as mock_gca,
        patch("matplotlib.pyplot.show") as mock_show,
        patch("matplotlib.pyplot.plot") as mock_plot,
    ):
        mock_ax = MagicMock()
        mock_gca.return_value = mock_ax
        ArchVisualizer(arch_spec).show(
            ax=mock_ax, show_words=[0], show_intra=[0], show_inter=[0]
        )
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


def test_path_bounds_x_y_bounds():
    viz = ArchVisualizer(arch_spec)
    x_min, x_max, y_min, y_max = viz.path_bounds()
    assert x_min <= x_max
    assert y_min <= y_max
    x_min2, x_max2 = viz.x_bounds
    y_min2, y_max2 = viz.y_bounds
    assert x_min2 <= x_max2
    assert y_min2 <= y_max2


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


def test_get_cz_partner_paired():
    loc = LocationAddress(0, 0)
    result = arch_spec.get_cz_partner(loc)
    assert result is not None


def test_get_cz_partner_none():
    word_no_cz = Word(sites=((0, 0), (1, 0)))
    rust_grid_no_cz = RustGrid.from_positions([0.0, 1.0], [0.0])
    rust_zone_no_cz = RustZone(
        name="test",
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
    assert spec_no_cz.get_cz_partner(loc) is None


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


def test_try_get_position_returns_none_for_invalid_address():
    """try_get_position should return None for invalid addresses."""
    bogus_word = len(arch_spec.words) + 99
    invalid_loc = LocationAddress(word_id=bogus_word, site_id=0, zone_id=0)
    assert arch_spec.try_get_position(invalid_loc) is None


def test_try_get_endpoints_returns_none_for_invalid_lane():
    """try_get_endpoints should return None for invalid lanes."""
    invalid_lane = LaneAddress(
        MoveType.SITE,
        word_id=999,
        site_id=0,
        bus_id=0,
        direction=Direction.FORWARD,
        zone_id=0,
    )
    assert arch_spec.try_get_endpoints(invalid_lane) is None


def test_get_position_raises_for_invalid_address():
    """get_position should raise ValueError for invalid addresses."""
    bogus_word = len(arch_spec.words) + 99
    invalid_loc = LocationAddress(word_id=bogus_word, site_id=0, zone_id=0)
    with pytest.raises(ValueError):
        arch_spec.get_position(invalid_loc)


def test_get_endpoints_raises_for_invalid_lane():
    """get_endpoints should raise ValueError for invalid lanes."""
    invalid_lane = LaneAddress(
        MoveType.SITE,
        word_id=999,
        site_id=0,
        bus_id=0,
        direction=Direction.FORWARD,
        zone_id=0,
    )
    with pytest.raises(ValueError):
        arch_spec.get_endpoints(invalid_lane)
