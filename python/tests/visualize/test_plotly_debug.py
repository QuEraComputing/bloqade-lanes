from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from kirin import ir
from kirin.dialects import py

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
from bloqade.lanes.prelude import kernel
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

    # 2 architecture + 3 static circuit + 1 route + circuit highlight + gate + atoms
    assert len(cast(Any, figure.data)) == 9
    assert len(figure.frames) == 1
    assert len(figure.layout.sliders) == 1
    assert figure.layout.sliders[0].steps[0].method == "animate"
    assert figure.layout.meta["bloqadePlotlyDebugger"] == {
        "stepCount": 1,
        "frameCount": 1,
        "frameNames": ["step-0"],
        "routeTraceCount": 1,
        "architectureTraceCount": 2,
        "circuitTraceIndices": [2, 3, 4],
        "circuitColumnCount": 0,
        "circuitStepColumns": [],
    }
    html = figure.to_html(full_html=False, include_plotlyjs=False)
    assert "plotly_animatingframe" in html
    assert "'sliders[0].active': frameIndex" in html
    site_trace = cast(Any, figure.data)[
        figure.layout.meta["archVisualizerSiteTraceIndex"]
    ]
    atom_trace = cast(Any, figure.data[-1])
    assert site_trace.name == "sites"
    assert site_trace.marker.color == "#e2e8f0"
    assert site_trace.marker.size == 9
    assert figure.layout.meta["archVisualizerSiteLanePreviewMode"] == "click"
    assert "grid (x, y)" in site_trace.hovertemplate
    assert list(atom_trace.text) == ["0"]
    assert "atom %{customdata[0]}" in atom_trace.hovertemplate
    assert "(zone, word, site)" in atom_trace.hovertemplate
    assert "grid (x, y)" in atom_trace.hovertemplate
    assert list(atom_trace.customdata[0]) == [0, 0, 0, 0, 0, 0, ""]
    assert atom_trace.meta["bloqadeTraceKind"] == "atom"


def test_gate_parameters_are_shown_on_affected_atom_and_gate(
    monkeypatch, small_arch_spec: ArchSpec
) -> None:
    state, _ = _state_at(small_arch_spec, moved=False)
    location = state.data.qubit_to_locations[0]
    load = move.Load()
    axis_angle = py.Constant(value=ir.PyAttr(0.25))
    rotation_angle = py.Constant(value=ir.PyAttr(1.5))
    statement = move.LocalR(
        current_state=load.result,
        axis_angle=axis_angle.result,
        rotation_angle=rotation_angle.result,
        location_addresses=(location,),
    )
    step = DebugStep(
        statement,
        state,
        "Step 1 / 1: LocalR(0.25, 1.5)",
        parameter_values=(0.25, 1.5),
    )
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

    gate_trace = cast(Any, figure.data[-2])
    atom_trace = cast(Any, figure.data[-1])
    expected = "LocalR(axis_angle=0.25, rotation_angle=1.5)"
    assert expected in gate_trace.hovertemplate
    assert atom_trace.customdata[0][6] == f"<br><b>gate:</b> {expected}"
    assert "%{customdata[6]}" in atom_trace.hovertemplate


def test_global_gate_is_visible_and_hoverable_on_single_row_architecture(
    monkeypatch, small_arch_spec: ArchSpec
) -> None:
    state, _ = _state_at(small_arch_spec, moved=False)
    load = move.Load()
    axis_angle = py.Constant(value=ir.PyAttr(0.125))
    rotation_angle = py.Constant(value=ir.PyAttr(-0.5))
    statement = move.GlobalR(
        current_state=load.result,
        axis_angle=axis_angle.result,
        rotation_angle=rotation_angle.result,
    )
    step = DebugStep(
        statement,
        state,
        "Step 1 / 1: GlobalR(0.125, -0.5)",
        parameter_values=(0.125, -0.5),
    )
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

    gate_trace = cast(Any, figure.data[-2])
    atom_trace = cast(Any, figure.data[-1])
    expected = "GlobalR(axis_angle=0.125, rotation_angle=-0.5)"
    assert gate_trace.fill == "toself"
    assert gate_trace.hoveron == "fills"
    assert gate_trace.zorder == -10
    assert min(gate_trace.y) < max(gate_trace.y)
    assert expected in gate_trace.hovertemplate
    assert atom_trace.customdata[0][6] == f"<br><b>gate:</b> {expected}"


def test_collect_debug_steps_resolves_global_gate_parameters_from_frame(
    small_arch_spec: ArchSpec,
) -> None:
    @kernel
    def global_gate_kernel():
        state = move.load()
        state = move.fill(
            state,
            location_addresses=(move.LocationAddress(0, 0, 0),),
        )
        move.global_r(state, axis_angle=0.125, rotation_angle=-0.5)

    global_step = next(
        step
        for step in plotly_debug.collect_debug_steps(
            global_gate_kernel, small_arch_spec
        )
        if isinstance(step.statement, move.GlobalR)
    )

    assert global_step.parameter_values == (0.125, -0.5)
    assert global_step.title.endswith("GlobalR(0.125, -0.5)")


def test_end_measure_is_a_distinct_zone_wide_debug_step(
    small_arch_spec: ArchSpec,
) -> None:
    @kernel
    def measuring_kernel():
        state = move.load()
        state = move.fill(
            state,
            location_addresses=(move.LocationAddress(0, 0, 0),),
        )
        move.end_measure(state, zone_addresses=(move.ZoneAddress(0),))

    steps = plotly_debug.collect_debug_steps(measuring_kernel, small_arch_spec)

    assert isinstance(steps[-1].statement, move.EndMeasure)
    assert steps[-1].state.data.qubit_to_locations == {0: move.LocationAddress(0, 0, 0)}
    assert "EndMeasure()" in steps[-1].title

    figure = plotly_debug.build_plotly_debugger_figure(
        measuring_kernel,
        small_arch_spec,
        interactive=True,
        pause_time=0.5,
        atom_marker="o",
        theme="light",
        height=600,
    )
    gate_trace = cast(Any, figure.frames[-1].data[-2])
    atom_trace = cast(Any, figure.frames[-1].data[-1])
    assert gate_trace.name == "EndMeasure"
    assert gate_trace.fill == "toself"
    assert gate_trace.fillcolor == "#d97706"
    assert "EndMeasure(zones=(0,))" in gate_trace.hovertemplate
    assert "EndMeasure(zones=(0,))" in atom_trace.customdata[0][6]


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

    debugger_meta = figure.layout.meta["bloqadePlotlyDebugger"]
    dynamic_start = debugger_meta["architectureTraceCount"] + len(
        debugger_meta["circuitTraceIndices"]
    )
    route_traces = cast(Any, figure.data[dynamic_start : dynamic_start + 2])
    route_trace = route_traces[0]
    atom_trace = cast(Any, figure.data[-1])
    assert len(cast(Any, figure.data)) == 10
    assert figure.layout.meta["bloqadePlotlyDebugger"]["routeTraceCount"] == 2
    assert route_trace.visible is True
    assert route_trace.hoverinfo == "skip"
    assert route_trace.hovertemplate is None
    assert route_trace.meta["bloqadeTraceKind"] == "movePath"
    assert route_trace.meta["segments"] == [
        {
            "atomId": 0,
            "busLabel": "Zone ID 0, Site bus 0",
            "source": "(0, 0, 0)",
            "destination": "(0, 0, 1)",
            "color": "rgb(68, 1, 84)",
            "start": [0.0, 0.0],
            "end": [0.5, 0.75],
        }
    ]
    assert list(route_trace.marker.symbol) == ["circle", "arrow", "circle"]
    assert route_trace.marker.size[1] == 16
    assert route_trace.marker.angleref == "up"
    assert route_trace.marker.line.width == 1.5
    assert route_trace.line.width == 4
    assert route_trace.line.color != route_traces[1].line.color
    assert list(route_trace.x[:2]) == pytest.approx([0.0, 0.5])
    assert list(route_traces[1].x[:2]) == pytest.approx([0.5, 1.0])
    assert len(cast(Any, figure.frames[0].data)) == 5
    bus_trace = cast(Any, figure.data[0])
    assert bus_trace.line.dash == "dash"
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
    playback_menu = figure.layout.updatemenus[-1]
    assert [button.label for button in playback_menu.buttons] == [
        "Play steps",
        "Pause",
    ]
    assert [button.method for button in playback_menu.buttons] == [
        "animate",
        "animate",
    ]
    assert playback_menu.x == pytest.approx(0.0)
    assert playback_menu.xanchor == "left"
    assert playback_menu.y == pytest.approx(1.09)
    assert [menu.x for menu in figure.layout.updatemenus[:-1]] == pytest.approx(
        [0.18, 0.47, 0.75]
    )
    assert all(
        menu.y == pytest.approx(playback_menu.y)
        for menu in figure.layout.updatemenus[:-1]
    )


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
    assert len(figure.layout.updatemenus) == 3
    assert all(
        "Play steps" not in [button.label for button in menu.buttons]
        for menu in figure.layout.updatemenus
    )


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
