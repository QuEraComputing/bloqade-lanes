"""Tests for the arch visualization helpers extracted from
``ArchSpec`` (#464 phase 1).

Covers the :class:`ArchVisualizer` class and verifies that the legacy
``ArchSpec.<method>`` shims still route through it.
"""

from __future__ import annotations

from typing import Any, cast
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
from bloqade.lanes.bytecode.encoding import Direction, SiteLaneAddress
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
    site_lane = SiteLaneAddress(
        word_id=0,
        site_id=0,
        bus_id=0,
        direction=Direction.FORWARD,
        zone_id=0,
    )
    return ArchSpec.from_components(
        words=(word, word),
        zones=(rust_zone,),
        modes=[rust_mode],
        paths={site_lane: ((0.0, 0.0), (0.5, 0.75), (1.0, 0.0))},
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


def test_plot_returns_axes(small_arch_spec: ArchSpec) -> None:
    mock_ax = MagicMock()
    viz = ArchVisualizer(small_arch_spec)
    result = viz.plot(mock_ax, show_words=[0], show_site_bus=[0], show_word_bus=[0])
    assert result is mock_ax
    assert mock_ax.scatter.called
    assert mock_ax.plot.called


def test_plot_interactive_starts_with_bus_paths_hidden(
    small_arch_spec: ArchSpec,
) -> None:
    figure = ArchVisualizer(small_arch_spec).plot_interactive()
    figure_data = cast(Any, figure.data)

    bus_indices = figure.layout.meta["archVisualizerBusTraceIndices"]
    bus_traces = [figure_data[index] for index in bus_indices]
    assert [trace.name for trace in bus_traces] == [
        "zone 0 · site bus 0",
        "zone 0 · word bus 0",
    ]
    assert all(trace.visible == "legendonly" for trace in bus_traces)
    assert all(trace.showlegend is False for trace in bus_traces)
    assert figure.layout.dragmode == "zoom"
    assert figure.layout.legend.groupclick == "toggleitem"
    assert figure.layout.paper_bgcolor == "#ffffff"
    assert figure.layout.plot_bgcolor == "#f8fafc"
    assert figure.layout.width == 1300


def test_plot_interactive_uses_architecture_paths(
    small_arch_spec: ArchSpec,
) -> None:
    visualizer = ArchVisualizer(small_arch_spec)
    figure = visualizer.plot_interactive(show_all_buses=True)
    figure_data = cast(Any, figure.data)
    site_trace = next(
        trace for trace in figure_data if trace.name == "zone 0 · site bus 0"
    )
    expected_path = next(visualizer.iter_site_bus_paths([0], [0]))

    assert len(expected_path) == 3
    assert list(zip(site_trace.x, site_trace.y))[: len(expected_path)] == list(
        expected_path
    )
    assert site_trace.visible is True


def test_plot_interactive_bus_hover_identifies_endpoints(
    small_arch_spec: ArchSpec,
) -> None:
    figure = ArchVisualizer(small_arch_spec).plot_interactive(show_all_buses=True)
    site_bus_trace = cast(Any, figure.data)[0]
    hover_data = next(value for value in site_bus_trace.customdata if value is not None)

    assert tuple(hover_data) == (
        "zone 0 · site bus 0",
        "(0, 0, 0)",
        "(0, 0, 1)",
        0,
    )
    assert "bus ID: %{customdata[3]}" in site_bus_trace.hovertemplate
    assert "source: %{customdata[1]}" in site_bus_trace.hovertemplate
    assert "destination: %{customdata[2]}" in site_bus_trace.hovertemplate
    assert "path point" not in site_bus_trace.hovertemplate


def test_plot_interactive_site_hover_keeps_text_compact_and_stores_lane_paths(
    small_arch_spec: ArchSpec,
) -> None:
    figure = ArchVisualizer(small_arch_spec).plot_interactive()
    assert [trace.name for trace in cast(Any, figure.data)].count("sites") == 1
    site_trace = next(
        trace for trace in cast(Any, figure.data) if trace.name == "sites"
    )
    site_data = next(
        value for value in site_trace.customdata if list(value[:3]) == [0, 0, 0]
    )

    assert list(site_data) == [0, 0, 0]
    assert "Touching lanes" not in site_trace.hovertemplate
    assert site_trace.hoverlabel.align == "left"

    meta = figure.layout.meta
    assert meta["archVisualizerSiteTraceIndex"] == 2
    assert meta["archVisualizerSiteLanePathRefs"]["0,0,0"] == [
        (0, False),
        (1, False),
    ]
    assert meta["archVisualizerSiteLanePathRefs"]["0,0,1"] == [(0, True)]
    first_path = meta["archVisualizerSiteLanePaths"][0]
    assert first_path["x"] == [0.0, 0.5, 1.0]
    assert first_path["y"] == [0.0, 0.75, 0.0]
    assert first_path["color"].startswith("hsl(")


def test_plot_interactive_html_highlights_hovered_bus(
    small_arch_spec: ArchSpec,
) -> None:
    figure = ArchVisualizer(small_arch_spec).plot_interactive()
    html = figure.to_html(full_html=False, include_plotlyjs=False)

    assert "plotly_hover" in html
    assert "plotly_unhover" in html
    assert "else if (busPoint)" in html
    assert "data-arch-visualizer-bus-highlight" in html
    assert "data-arch-visualizer-site-lane" in html
    assert "function siteCustomdataAt(x, y)" in html
    assert ".map((item) => siteCustomdataAt(item.x, item.y))" in html
    assert "function siteCustomdataNearPointer(event)" in html
    assert "plot.addEventListener('mousemove'" in html
    assert "activateSiteLaneOverlays(siteCustomdataNearPointer(event))" in html
    assert "element.isConnected" in html
    assert "plot.on('plotly_unhover'" in html
    assert "data-arch-visualizer-site-lane-arrow" not in html
    assert "data-arch-visualizer-bus-selectors" in html
    assert "data-arch-visualizer-bus-color" in html
    assert "checkbox.type = 'checkbox'" in html
    assert "text.textContent = `ID ${control.busId} · ${control.label}`" in html
    assert "archVisualizerSiteLanePathRefs" in html
    assert "stroke-opacity', '0.42'" in html
    assert "hoverLayer.insertBefore(path, hoverLayer.firstChild)" in html
    assert '"scrollZoom": true' in html
    assert '"responsive": true' in html
    assert list(figure.layout.meta["archVisualizerBusTraceIndices"]) == [0, 1]


def test_plot_interactive_notebook_representation_keeps_interactions(
    small_arch_spec: ArchSpec,
) -> None:
    figure = ArchVisualizer(small_arch_spec).plot_interactive()
    mimebundle = figure._repr_mimebundle_()

    assert set(mimebundle) == {"text/html"}
    assert "data-arch-visualizer-bus-selectors" in mimebundle["text/html"]
    assert "plotly_hover" in mimebundle["text/html"]


def test_plot_interactive_ipython_display_keeps_interactions(
    small_arch_spec: ArchSpec,
) -> None:
    figure = ArchVisualizer(small_arch_spec).plot_interactive()

    with patch("IPython.display.display") as display:
        figure._ipython_display_()

    html = display.call_args.args[0].data
    assert "data-arch-visualizer-bus-selectors" in html
    assert "data-arch-visualizer-site-lane" in html


def test_plot_interactive_show_keeps_interactions_in_jupyter(
    small_arch_spec: ArchSpec,
) -> None:
    figure = ArchVisualizer(small_arch_spec).plot_interactive()
    zmq_shell = type("ZMQInteractiveShell", (), {})()

    with (
        patch("IPython.core.getipython.get_ipython", return_value=zmq_shell),
        patch("IPython.display.display") as display,
    ):
        figure.show(config={"scrollZoom": False})

    html = display.call_args.args[0].data
    assert "data-arch-visualizer-bus-selectors" in html
    assert "data-arch-visualizer-site-lane" in html
    assert '"scrollZoom": false' in html


def test_plot_interactive_show_keeps_interactions_in_browser(
    small_arch_spec: ArchSpec,
) -> None:
    figure = ArchVisualizer(small_arch_spec).plot_interactive()

    with (
        patch("IPython.core.getipython.get_ipython", return_value=None),
        patch("plotly.io._base_renderers.open_html_in_browser") as open_browser,
    ):
        figure.show(renderer="browser", config={"scrollZoom": False})

    html = open_browser.call_args.args[0]
    assert "data-arch-visualizer-bus-selectors" in html
    assert "data-arch-visualizer-site-lane" in html
    assert '"scrollZoom": false' in html
    assert open_browser.call_args.kwargs == {}


def test_plot_interactive_html_preserves_caller_config(
    small_arch_spec: ArchSpec,
) -> None:
    figure = ArchVisualizer(small_arch_spec).plot_interactive()
    html = figure.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={"scrollZoom": False, "displayModeBar": False},
    )

    assert '"scrollZoom": false' in html
    assert '"responsive": true' in html
    assert '"displayModeBar": false' in html


def test_plot_interactive_site_identity_toggle(
    small_arch_spec: ArchSpec,
) -> None:
    figure = ArchVisualizer(small_arch_spec).plot_interactive(show_site_ids=True)
    label_trace = cast(Any, figure.data)[-1]

    assert set(label_trace.text) == {
        "(0,0,0)",
        "(0,0,1)",
        "(0,1,0)",
        "(0,1,1)",
    }
    assert list(label_trace.textposition) == [
        "top left",
        "top left",
        "bottom right",
        "bottom right",
    ]
    assert label_trace.name == "sites"
    assert label_trace.mode == "markers+text"
    assert label_trace.marker.size == 9
    assert label_trace.marker.opacity is None
    assert label_trace.textfont.size == 9
    assert label_trace.cliponaxis is False
    assert figure.layout.updatemenus[0].active == 1


def test_plot_interactive_controls_can_show_and_hide_all_buses(
    small_arch_spec: ArchSpec,
) -> None:
    figure = ArchVisualizer(small_arch_spec).plot_interactive()
    hide_button, show_button = figure.layout.updatemenus[1].buttons

    assert hide_button.args[0]["visible"] == "legendonly"
    assert show_button.args[0]["visible"] is True
    assert list(hide_button.args[1]) == [0, 1]


def test_plot_interactive_has_color_labelled_bus_multiselectors(
    small_arch_spec: ArchSpec,
) -> None:
    figure = ArchVisualizer(small_arch_spec).plot_interactive()
    controls = figure.layout.meta["archVisualizerBusControls"]

    assert all(menu.type != "dropdown" for menu in figure.layout.updatemenus)
    assert controls == [
        {
            "traceIndex": 0,
            "kind": "site",
            "busId": 0,
            "label": "zone 0 · site bus 0",
            "color": "hsl(0, 68%, 40%)",
        },
        {
            "traceIndex": 1,
            "kind": "word",
            "busId": 0,
            "label": "zone 0 · word bus 0",
            "color": "hsl(138, 68%, 40%)",
        },
    ]


def test_plot_interactive_can_restore_legacy_bus_legend(
    small_arch_spec: ArchSpec,
) -> None:
    figure = ArchVisualizer(small_arch_spec).plot_interactive(show_bus_legend=True)
    bus_traces = [
        trace for trace in cast(Any, figure.data) if "bus" in (trace.name or "")
    ]

    assert all(trace.showlegend is True for trace in bus_traces)
    assert figure.layout.legend.title.text == "Bus legend"
    assert figure.layout.legend.y == 0.62
    assert figure.layout.margin.r == 270


def test_plot_interactive_cartoon_paths_curve_site_buses(
    small_arch_spec: ArchSpec,
) -> None:
    figure = ArchVisualizer(small_arch_spec).plot_interactive(
        path_style="cartoon", show_all_buses=True
    )
    figure_data = cast(Any, figure.data)
    site_trace = next(
        trace for trace in figure_data if trace.name == "zone 0 · site bus 0"
    )
    site_path = [(x, y) for x, y in zip(site_trace.x, site_trace.y) if x is not None]

    assert len(site_path) == 21
    assert min(y for _, y in site_path) < 0.0
    assert figure.layout.updatemenus[2].active == 1


def test_plot_interactive_path_toggle_preserves_bus_selection(
    small_arch_spec: ArchSpec,
) -> None:
    figure = ArchVisualizer(small_arch_spec).plot_interactive()
    exact_button, cartoon_button = figure.layout.updatemenus[2].buttons

    assert exact_button.method == "update"
    assert cartoon_button.method == "update"
    assert "visible" not in cartoon_button.args[0]
    assert list(cartoon_button.args[2]) == [0, 1]


def test_plot_interactive_supports_dark_theme(
    small_arch_spec: ArchSpec,
) -> None:
    figure = ArchVisualizer(small_arch_spec).plot_interactive(theme="dark")

    assert figure.layout.paper_bgcolor == "#0f172a"
    assert figure.layout.plot_bgcolor == "#111827"


def test_plot_uses_plt_gca_when_ax_is_none(small_arch_spec: ArchSpec) -> None:
    with patch("matplotlib.pyplot.gca") as mock_gca:
        mock_ax = MagicMock()
        mock_gca.return_value = mock_ax
        result = ArchVisualizer(small_arch_spec).plot(ax=None, show_words=[0])
        assert result is mock_ax
        mock_gca.assert_called_once()


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
