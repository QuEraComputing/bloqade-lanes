from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from bloqade.lanes.analysis.atom import AtomState
from bloqade.lanes.analysis.atom.atom_state_data import AtomStateData
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode._native import (
    Grid as RustGrid,
    LocationAddress as RustLocAddr,
    Mode as RustMode,
    SiteBus,
    Zone as RustZone,
)
from bloqade.lanes.bytecode.encoding import Direction, SiteLaneAddress
from bloqade.lanes.bytecode.word import Word
from bloqade.lanes.dialects import move
from bloqade.lanes.visualize import plotly_debug
from bloqade.lanes.visualize.plotly_debug import DebugStep


@pytest.fixture
def small_arch_spec() -> ArchSpec:
    word = Word(sites=((0, 0), (1, 0)))
    rust_grid = RustGrid.from_positions([0.0, 1.0], [0.0])
    rust_zone = RustZone(
        name="test",
        grid=rust_grid,
        site_buses=[SiteBus(src=[0], dst=[1])],
        word_buses=[],
        words_with_site_buses=[0],
        sites_with_word_buses=[],
        entangling_pairs=[],
    )
    rust_mode = RustMode(
        name="all",
        zones=[0],
        bitstring_order=[RustLocAddr(0, 0, 0), RustLocAddr(0, 0, 1)],
    )
    lane = SiteLaneAddress(
        word_id=0,
        site_id=0,
        bus_id=0,
        direction=Direction.FORWARD,
        zone_id=0,
    )
    return ArchSpec.from_components(
        words=(word,),
        zones=(rust_zone,),
        modes=[rust_mode],
        paths={lane: ((0.0, 0.0), (0.5, 0.75), (1.0, 0.0))},
    )


def _state_at(arch_spec: ArchSpec, *, moved: bool) -> tuple[AtomState, Any]:
    lane = SiteLaneAddress(
        word_id=0,
        site_id=0,
        bus_id=0,
        direction=Direction.FORWARD,
        zone_id=0,
    )
    src, dst = arch_spec.get_endpoints(lane)
    location = dst if moved else src
    state = AtomState(
        AtomStateData.from_fields(
            locations_to_qubit={location: 0},
            qubit_to_locations={0: location},
            prev_lanes={0: lane} if moved else None,
        )
    )
    return state, lane


def test_build_debugger_figure_has_clickable_step_slider(
    monkeypatch, small_arch_spec: ArchSpec
) -> None:
    state, _ = _state_at(small_arch_spec, moved=False)
    step = DebugStep(move.Load(), state, "Step 1 / 1: Load()")
    monkeypatch.setattr(plotly_debug, "collect_debug_steps", lambda *_args: [step])

    figure = plotly_debug.build_plotly_debugger_figure(
        MagicMock(),
        small_arch_spec,
        interactive=True,
        pause_time=0.5,
        atom_marker="o",
        theme="light",
        height=600,
    )

    assert len(cast(Any, figure.data)) == 4
    assert len(figure.frames) == 1
    assert len(figure.layout.sliders) == 1
    assert figure.layout.sliders[0].steps[0].method == "animate"
    assert figure.layout.meta["bloqadePlotlyDebugger"] == {
        "stepCount": 1,
        "frameCount": 1,
        "routeTraceCount": 1,
    }
    site_trace = cast(Any, figure.data[0])
    atom_trace = cast(Any, figure.data[-1])
    assert site_trace.name == "SLM sites"
    assert site_trace.marker.symbol == "circle-open"
    assert site_trace.marker.color == "#475569"
    assert site_trace.marker.size == pytest.approx(
        plotly_debug._slm_marker_size(small_arch_spec)
    )
    assert list(atom_trace.text) == ["0"]
    assert "atom %{customdata[0]}" in atom_trace.hovertemplate


def test_move_path_hover_shows_source_and_destination(
    monkeypatch, small_arch_spec: ArchSpec
) -> None:
    state, _ = _state_at(small_arch_spec, moved=True)
    monkeypatch.setattr(
        plotly_debug,
        "collect_debug_steps",
        lambda *_args: [DebugStep(move.Load(), state, "Step 1 / 1: Load()")],
    )

    figure = plotly_debug.build_plotly_debugger_figure(
        MagicMock(),
        small_arch_spec,
        interactive=True,
        pause_time=1.0,
        atom_marker="s",
        theme="light",
        height=600,
    )

    route_traces = cast(Any, figure.data[1:3])
    route_trace = route_traces[0]
    atom_trace = cast(Any, figure.data[-1])
    assert len(cast(Any, figure.data)) == 5
    assert figure.layout.meta["bloqadePlotlyDebugger"]["routeTraceCount"] == 2
    assert route_trace.visible is True
    assert "source: %{customdata[1]}" in route_trace.hovertemplate
    assert "destination: %{customdata[2]}" in route_trace.hovertemplate
    assert route_trace.customdata[0] == [0, "(0, 0, 0)", "(0, 0, 1)"]
    assert list(route_trace.marker.symbol) == ["circle", "arrow", "circle"]
    assert route_trace.marker.size[1] == 11
    assert route_trace.marker.angleref == "up"
    assert route_trace.line.color != route_traces[1].line.color
    assert list(route_trace.x[:2]) == pytest.approx([0.0, 0.5])
    assert list(route_traces[1].x[:2]) == pytest.approx([0.5, 1.0])
    assert len(cast(Any, figure.frames[0].data)) == 4
    site_trace = cast(Any, figure.data[0])
    assert site_trace.marker.symbol == "square-open"
    assert atom_trace.marker.symbol == "square"


def test_shorter_route_frames_hide_preceding_arrows(
    monkeypatch, small_arch_spec: ArchSpec
) -> None:
    moved_state, _ = _state_at(small_arch_spec, moved=True)
    stationary_state, _ = _state_at(small_arch_spec, moved=False)
    monkeypatch.setattr(
        plotly_debug,
        "collect_debug_steps",
        lambda *_args: [
            DebugStep(move.Load(), moved_state, "Move"),
            DebugStep(move.Load(), stationary_state, "Stationary"),
        ],
    )

    figure = plotly_debug.build_plotly_debugger_figure(
        MagicMock(),
        small_arch_spec,
        interactive=True,
        pause_time=1.0,
        atom_marker="o",
        theme="light",
        height=600,
    )

    moved_route_traces = cast(Any, figure.frames[0].data[:2])
    stationary_route_traces = cast(Any, figure.frames[1].data[:2])
    assert all(trace.visible is True for trace in moved_route_traces)
    assert all(trace.visible is False for trace in stationary_route_traces)
    assert all(list(trace.x) == [] for trace in stationary_route_traces)


def test_noninteractive_figure_hides_controls(
    monkeypatch, small_arch_spec: ArchSpec
) -> None:
    state, _ = _state_at(small_arch_spec, moved=False)
    monkeypatch.setattr(
        plotly_debug,
        "collect_debug_steps",
        lambda *_args: [DebugStep(move.Load(), state, "Load()")],
    )

    figure = plotly_debug.build_plotly_debugger_figure(
        MagicMock(),
        small_arch_spec,
        interactive=False,
        pause_time=1.0,
        atom_marker="o",
        theme="light",
        height=600,
    )

    assert not figure.layout.sliders
    assert not figure.layout.updatemenus


def test_show_plotly_debugger_uses_browser_outside_jupyter(monkeypatch) -> None:
    figure = MagicMock()
    monkeypatch.setattr(plotly_debug, "_in_jupyter_kernel", lambda: False)

    plotly_debug._show_plotly_debugger(
        figure,
        renderer=None,
        auto_play=True,
        frame_duration_ms=25,
    )

    assert figure.show.call_args.kwargs["renderer"] == "browser"
    assert figure.show.call_args.kwargs["config"]["scrollZoom"] is True


def test_show_plotly_debugger_uses_default_renderer_in_jupyter(monkeypatch) -> None:
    figure = MagicMock()
    monkeypatch.setattr(plotly_debug, "_in_jupyter_kernel", lambda: True)

    plotly_debug._show_plotly_debugger(
        figure,
        renderer=None,
        auto_play=False,
        frame_duration_ms=100,
    )

    assert figure.show.call_args.kwargs["renderer"] is None


def test_plotly_debugger_displays_and_returns_none(
    monkeypatch, small_arch_spec: ArchSpec
) -> None:
    figure = MagicMock()
    show_figure = MagicMock()
    monkeypatch.setattr(
        plotly_debug, "build_plotly_debugger_figure", lambda *_args, **_kwargs: figure
    )
    monkeypatch.setattr(plotly_debug, "_show_plotly_debugger", show_figure)

    result = plotly_debug.plotly_debugger(MagicMock(), small_arch_spec)

    assert result is None
    show_figure.assert_called_once_with(
        figure,
        renderer=None,
        auto_play=False,
        frame_duration_ms=1000,
    )


def test_plotly_debugger_can_return_figure_without_displaying(
    monkeypatch, small_arch_spec: ArchSpec
) -> None:
    figure = MagicMock()
    show_figure = MagicMock()
    monkeypatch.setattr(
        plotly_debug, "build_plotly_debugger_figure", lambda *_args, **_kwargs: figure
    )
    monkeypatch.setattr(plotly_debug, "_show_plotly_debugger", show_figure)

    result = plotly_debug.plotly_debugger(MagicMock(), small_arch_spec, show=False)

    assert result is figure
    show_figure.assert_not_called()


def test_plotly_debugger_argument_validation(small_arch_spec: ArchSpec) -> None:
    with pytest.raises(ValueError, match="pause_time"):
        plotly_debug.build_plotly_debugger_figure(
            MagicMock(),
            small_arch_spec,
            interactive=True,
            pause_time=-1.0,
            atom_marker="o",
            theme="light",
            height=600,
        )
