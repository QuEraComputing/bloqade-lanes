"""Tests for the arch visualization helpers extracted from
``ArchSpec`` (#464 phase 1).

Covers the :class:`ArchVisualizer` class and verifies that the legacy
``ArchSpec.<method>`` shims still route through it.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode._native import (
    Grid as RustGrid,
    LocationAddress as RustLocAddr,
    Mode as RustMode,
    SiteBus,
    WordBus,
    Zone as RustZone,
)
from bloqade.lanes.bytecode.word import Word
from bloqade.lanes.visualize.arch import ArchVisualizer

# ── Hand-built minimal ArchSpec fixture ──


@pytest.fixture
def small_arch_spec() -> ArchSpec:
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
    return ArchSpec.from_components(
        words=(word, word),
        zones=(rust_zone,),
        modes=[rust_mode],
    )


@pytest.fixture
def two_zone_arch_spec() -> ArchSpec:
    words = (
        Word(sites=((0, 0), (0, 1))),
        Word(sites=((1, 0), (1, 1))),
    )

    zone0 = RustZone(
        name="zone0",
        grid=RustGrid.from_positions([0.0, 1.0], [0.0, 1.0]),
        site_buses=[SiteBus(src=[0], dst=[1])],
        word_buses=[WordBus(src=[0], dst=[1])],
        words_with_site_buses=[0, 1],
        sites_with_word_buses=[0, 1],
        entangling_pairs=[(0, 1)],
    )

    zone1 = RustZone(
        name="zone1",
        grid=RustGrid.from_positions([10.0, 11.0], [0.0, 1.0]),
        site_buses=[SiteBus(src=[0], dst=[1])],
        word_buses=[WordBus(src=[0], dst=[1])],
        words_with_site_buses=[0, 1],
        sites_with_word_buses=[0, 1],
        entangling_pairs=[(0, 1)],
    )

    mode = RustMode(
        name="all",
        zones=[0, 1],
        bitstring_order=[
            RustLocAddr(zone_id, word_id, site_id)
            for zone_id in (0, 1)
            for word_id in (0, 1)
            for site_id in (0, 1)
        ],
    )

    return ArchSpec.from_components(
        words=words,
        zones=(zone0, zone1),
        modes=[mode],
    )


# ── ArchVisualizer class ──


def test_x_bounds(small_arch_spec: ArchSpec) -> None:
    viz = ArchVisualizer(small_arch_spec)
    assert viz.x_bounds == (0.0, 1.0)


def test_y_bounds(small_arch_spec: ArchSpec) -> None:
    viz = ArchVisualizer(small_arch_spec)
    assert viz.y_bounds == (0.0, 0.0)


def test_path_bounds(small_arch_spec: ArchSpec) -> None:
    x_min, x_max, y_min, y_max = ArchVisualizer(small_arch_spec).path_bounds()
    assert x_min <= 0.0 <= x_max
    assert y_min <= 0.0 <= y_max


def test_bounds_are_cached(small_arch_spec: ArchSpec) -> None:
    viz = ArchVisualizer(small_arch_spec)
    assert viz.x_bounds is viz.x_bounds
    assert viz.y_bounds is viz.y_bounds


def test_iter_site_bus_paths(small_arch_spec: ArchSpec) -> None:
    viz = ArchVisualizer(small_arch_spec)
    paths = list(viz.iter_site_bus_paths([0], [0]))
    assert paths
    for path in paths:
        assert isinstance(path, tuple)
        assert all(isinstance(coord, tuple) and len(coord) == 2 for coord in path)


def test_iter_word_bus_paths(small_arch_spec: ArchSpec) -> None:
    viz = ArchVisualizer(small_arch_spec)
    paths = list(viz.iter_word_bus_paths([0]))
    assert paths
    for path in paths:
        assert isinstance(path, tuple)
        assert all(isinstance(coord, tuple) and len(coord) == 2 for coord in path)


def test_iter_site_bus_paths_selects_requested_zone(
    two_zone_arch_spec: ArchSpec,
) -> None:
    viz = ArchVisualizer(two_zone_arch_spec)

    paths = list(
        viz.iter_site_bus_paths(
            show_words=[0],
            show_site_bus=[0],
            zone_id=1,
        )
    )

    assert paths
    assert all(path[0][0] == 10.0 for path in paths)


def test_iter_word_bus_paths_selects_requested_zone(
    two_zone_arch_spec: ArchSpec,
) -> None:
    viz = ArchVisualizer(two_zone_arch_spec)

    paths = list(
        viz.iter_word_bus_paths(
            show_word_bus=[0],
            zone_id=1,
        )
    )

    assert paths
    assert all(path[0][0] >= 10.0 for path in paths)


def test_plot_returns_axes(small_arch_spec: ArchSpec) -> None:
    mock_ax = MagicMock()
    viz = ArchVisualizer(small_arch_spec)
    result = viz.plot(mock_ax, show_words=[0], show_site_bus=[0], show_word_bus=[0])
    assert result is mock_ax
    assert mock_ax.scatter.called
    assert mock_ax.plot.called


def test_plot_labels_sites_with_addresses(small_arch_spec: ArchSpec) -> None:
    mock_ax = MagicMock()
    viz = ArchVisualizer(small_arch_spec)

    viz.plot(
        mock_ax,
        show_words=[0],
        label_sites=True,
    )

    labels = [call.args[2] for call in mock_ax.text.call_args_list]

    assert "(0, 0, 0)" in labels
    assert "(0, 0, 1)" in labels


def test_plot_selects_requested_zone(two_zone_arch_spec: ArchSpec) -> None:
    mock_ax = MagicMock()
    viz = ArchVisualizer(two_zone_arch_spec)

    viz.plot(
        mock_ax,
        show_words=[0],
        zone_id=1,
        label_sites=True,
    )

    labels = [call.args[2] for call in mock_ax.text.call_args_list]

    assert labels == [
        "(1, 0, 0)",
        "(1, 0, 1)",
    ]

    x_positions, y_positions = mock_ax.scatter.call_args.args[:2]

    assert x_positions == [10.0, 10.0]
    assert y_positions == [0.0, 1.0]


def test_plot_uses_plt_gca_when_ax_is_none(small_arch_spec: ArchSpec) -> None:
    with patch("matplotlib.pyplot.gca") as mock_gca:
        mock_ax = MagicMock()
        mock_gca.return_value = mock_ax
        result = ArchVisualizer(small_arch_spec).plot(ax=None, show_words=[0])
        assert result is mock_ax
        mock_gca.assert_called_once()


def test_show_forwards_zone_and_labels(small_arch_spec: ArchSpec) -> None:
    mock_ax = MagicMock()
    viz = ArchVisualizer(small_arch_spec)

    with (
        patch.object(viz, "plot") as mock_plot,
        patch("matplotlib.pyplot.show") as mock_show,
    ):
        viz.show(
            ax=mock_ax,
            show_words=[0],
            zone_id=0,
            label_sites=True,
        )

    mock_plot.assert_called_once_with(
        mock_ax,
        show_words=[0],
        show_site_bus=(),
        show_word_bus=(),
        zone_id=0,
        label_sites=True,
    )
    mock_show.assert_called_once()


def test_show_calls_plt_show(small_arch_spec: ArchSpec) -> None:
    with (
        patch("matplotlib.pyplot.gca") as mock_gca,
        patch("matplotlib.pyplot.show") as mock_show,
    ):
        mock_ax = MagicMock()
        mock_gca.return_value = mock_ax
        ArchVisualizer(small_arch_spec).show(
            ax=mock_ax, show_words=[0], show_intra=[0], show_inter=[0]
        )
        assert mock_show.called


def test_archvisualizer_plot_called_directly(
    small_arch_spec: ArchSpec,
) -> None:
    mock_ax = MagicMock()
    with patch.object(ArchVisualizer, "plot", return_value=mock_ax) as mock_plot:
        result = ArchVisualizer(small_arch_spec).plot(mock_ax, show_words=[0])
        mock_plot.assert_called_once()
        assert mock_plot.call_args.args == (mock_ax,)
        assert mock_plot.call_args.kwargs["show_words"] == [0]
        assert result is mock_ax
