from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from kirin import ir

from bloqade.lanes.analysis.atom import AtomInterpreter, AtomState, Value
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.dialects import move
from bloqade.lanes.visualize.arch import ArchVisualizer

if TYPE_CHECKING:
    from plotly.graph_objects import Figure  # type: ignore[reportMissingImports]


Theme = Literal["light", "dark"]

_ATOM_SYMBOLS = {
    "o": "circle",
    ".": "circle",
    "s": "square",
    "+": "cross",
    "x": "x",
    "^": "triangle-up",
    "v": "triangle-down",
    "d": "diamond",
    "D": "diamond",
    "*": "star",
    "p": "pentagon",
    "h": "hexagon",
}

_OPEN_ATOM_SYMBOLS = {
    "o": "circle-open",
    ".": "circle-open",
    "s": "square-open",
    "+": "cross-open",
    "x": "x-open",
    "^": "triangle-up-open",
    "v": "triangle-down-open",
    "d": "diamond-open",
    "D": "diamond-open",
    "*": "star-open",
    "p": "pentagon-open",
    "h": "hexagon-open",
}


@dataclass(frozen=True)
class DebugStep:
    statement: ir.Statement
    state: AtomState
    title: str


@dataclass(frozen=True)
class RouteSegment:
    start: tuple[float, float]
    end: tuple[float, float]
    color: str
    hover_data: tuple[object, object, object]


def collect_debug_steps(mt: ir.Method, arch_spec: ArchSpec) -> list[DebugStep]:
    """Interpret ``mt`` and retain every statement that produces atom state."""
    frame, _ = AtomInterpreter(mt.dialects, arch_spec=arch_spec).run(mt)
    constants: dict[ir.SSAValue, float | int] = {}
    statements_and_states: list[tuple[ir.Statement, AtomState]] = []

    for statement in mt.callable_region.walk():
        results = frame.get_values(statement.results)
        match results:
            case (AtomState() as state,):
                statements_and_states.append((statement, state))
            case (Value(value),) if isinstance(value, (float, int)):
                constants[statement.results[0]] = value

    def statement_text(statement: ir.Statement) -> str:
        values = [constants[arg] for arg in statement.args if arg in constants]
        arguments = ", ".join(str(value) for value in values)
        return f"{type(statement).__name__}({arguments})"

    num_steps = len(statements_and_states)
    return [
        DebugStep(
            statement=statement,
            state=state,
            title=f"Step {index + 1} / {num_steps}: {statement_text(statement)}",
        )
        for index, (statement, state) in enumerate(statements_and_states)
    ]


def _plotly() -> Any:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "The Plotly move debugger requires the 'visualization' extra"
        ) from exc
    return go


def _theme_colors(theme: Theme) -> dict[str, str]:
    if theme == "light":
        return {
            "paper": "#ffffff",
            "plot": "#f8fafc",
            "text": "#1e293b",
            "site": "#ffffff",
            "site_edge": "#475569",
            "atom": "#6437ff",
            "route": "#7c3aed",
            "local_r": "#2563eb",
            "local_rz": "#16a34a",
            "cz": "#dc2626",
        }
    if theme == "dark":
        return {
            "paper": "#0f172a",
            "plot": "#111827",
            "text": "#f8fafc",
            "site": "#111827",
            "site_edge": "#cbd5e1",
            "atom": "#8f78ff",
            "route": "#c4b5fd",
            "local_r": "#60a5fa",
            "local_rz": "#4ade80",
            "cz": "#f87171",
        }
    raise ValueError("theme must be 'light' or 'dark'")


def _view_bounds(arch_spec: ArchSpec) -> tuple[float, float, float, float]:
    x_min, x_max, y_min, y_max = ArchVisualizer(arch_spec).path_bounds()
    x_width = x_max - x_min
    y_width = y_max - y_min
    x_padding = 0.08 * x_width if x_width else 1.0
    y_padding = 0.08 * y_width if y_width else 1.0
    return (
        x_min - x_padding,
        x_max + x_padding,
        y_min - y_padding,
        y_max + y_padding,
    )


def _slm_marker_size(arch_spec: ArchSpec) -> float:
    """Approximate the site-marker scaling used by ``PlotParameters``."""
    num_sites = sum(len(word.site_indices) for word in arch_spec.words)
    if num_sites == 0:
        return 12.0
    scale = 2.0 * math.sqrt(44.0 / num_sites)
    # Matplotlib's scatter size is an area in points squared. Convert its
    # diameter to pixels at the 100 dpi assumed by the existing debugger.
    return math.sqrt(scale * 80.0) * 100.0 / 72.0


def _site_trace(arch_spec: ArchSpec, colors: dict[str, str], atom_marker: str) -> Any:
    go = _plotly()
    locations = list(ArchVisualizer(arch_spec)._iter_locations())
    return go.Scatter(
        x=[position[0] for _, position in locations],
        y=[position[1] for _, position in locations],
        customdata=[
            [location.zone_id, location.word_id, location.site_id]
            for location, _ in locations
        ],
        mode="markers",
        marker={
            # Plotly's open symbols use ``color`` for the outline. This mirrors
            # the hollow, black-edged SLM sites drawn by ``StateArtist``.
            "color": colors["site_edge"],
            "line": {"color": colors["site_edge"], "width": 1},
            "size": _slm_marker_size(arch_spec),
            "symbol": _OPEN_ATOM_SYMBOLS.get(atom_marker, atom_marker),
        },
        name="SLM sites",
        showlegend=False,
        hovertemplate=(
            "zone %{customdata[0]}<br>word %{customdata[1]}<br>"
            "site %{customdata[2]}<br>(%{x:.3f}, %{y:.3f}) µm<extra></extra>"
        ),
    )


def _empty_trace(name: str) -> Any:
    go = _plotly()
    return go.Scatter(x=[], y=[], mode="lines", name=name, visible=False)


def _atom_trace(
    state: AtomState,
    arch_spec: ArchSpec,
    colors: dict[str, str],
    atom_marker: str,
) -> Any:
    go = _plotly()
    positions = {
        qubit_id: arch_spec.get_position(location)
        for qubit_id, location in state.data.qubit_to_locations.items()
    }
    qubit_ids = sorted(positions)
    return go.Scatter(
        x=[positions[qubit_id][0] for qubit_id in qubit_ids],
        y=[positions[qubit_id][1] for qubit_id in qubit_ids],
        customdata=[[qubit_id] for qubit_id in qubit_ids],
        text=[str(qubit_id) for qubit_id in qubit_ids],
        mode="markers+text",
        textposition="middle center",
        textfont={"color": "white", "size": 10},
        marker={
            "color": colors["atom"],
            "size": 18,
            "symbol": _ATOM_SYMBOLS.get(atom_marker, atom_marker),
            "line": {"color": colors["paper"], "width": 1},
        },
        name="Atoms",
        showlegend=False,
        hovertemplate=(
            "atom %{customdata[0]}<br>(%{x:.3f}, %{y:.3f}) µm<extra></extra>"
        ),
    )


def _viridis_colors(num_colors: int) -> list[str]:
    if num_colors == 0:
        return []
    from plotly.colors import sample_colorscale

    samples = (
        [0.0]
        if num_colors == 1
        else [index / (num_colors - 1) for index in range(num_colors)]
    )
    return cast(list[str], list(sample_colorscale("Viridis", samples)))


def _route_segments(state: AtomState, arch_spec: ArchSpec) -> list[RouteSegment]:
    """Split atom routes into the colored segments used by ``debugger``."""
    route_segments: list[RouteSegment] = []
    for qubit_id, lane in sorted(state.data.prev_lanes.items()):
        src, dst = arch_spec.get_endpoints(lane)
        hover_data: tuple[object, object, object] = (
            qubit_id,
            f"({src.zone_id}, {src.word_id}, {src.site_id})",
            f"({dst.zone_id}, {dst.word_id}, {dst.site_id})",
        )
        segments = list(itertools.pairwise(arch_spec.get_path(lane)))
        for color, (start, end) in zip(_viridis_colors(len(segments)), segments):
            route_segments.append(
                RouteSegment(
                    start=start,
                    end=end,
                    color=color,
                    hover_data=hover_data,
                )
            )
    return route_segments


def _group_route_segments(
    route_segments: Sequence[RouteSegment],
) -> list[list[RouteSegment]]:
    """Group equally colored segments so simultaneous moves share traces."""
    segments_by_color: dict[str, list[RouteSegment]] = {}
    for segment in route_segments:
        segments_by_color.setdefault(segment.color, []).append(segment)
    return list(segments_by_color.values())


def _route_trace(segments: Sequence[RouteSegment] | None) -> Any:
    """Render equally colored route segments with visible arrowheads."""
    go = _plotly()
    if not segments:
        return go.Scatter(
            x=[],
            y=[],
            mode="lines+markers",
            name="Move path",
            showlegend=False,
            visible=False,
            hoverinfo="skip",
        )

    x_values: list[float | None] = []
    y_values: list[float | None] = []
    hover_data: list[tuple[object, object, object] | None] = []
    marker_sizes: list[int] = []
    marker_symbols: list[str] = []
    marker_angles: list[float] = []
    for segment in segments:
        delta_x = segment.end[0] - segment.start[0]
        delta_y = segment.end[1] - segment.start[1]
        # Plotly marker angles are clockwise from screen-up. With equal axis
        # scaling, atan2(dx, dy) maps data-space segments to that convention.
        arrow_angle = math.degrees(math.atan2(delta_x, delta_y))
        x_values.extend([segment.start[0], segment.end[0], None])
        y_values.extend([segment.start[1], segment.end[1], None])
        hover_data.extend([segment.hover_data, segment.hover_data, None])
        marker_sizes.extend([0, 11, 0])
        marker_symbols.extend(["circle", "arrow", "circle"])
        marker_angles.extend([0, arrow_angle, 0])

    color = segments[0].color
    return go.Scatter(
        x=x_values,
        y=y_values,
        customdata=hover_data,
        mode="lines+markers",
        line={"color": color, "width": 2.25},
        marker={
            "color": color,
            "size": marker_sizes,
            "symbol": marker_symbols,
            "angle": marker_angles,
            "angleref": "up",
            "line": {"width": 0},
        },
        name="Move path",
        showlegend=False,
        visible=True,
        cliponaxis=False,
        hovertemplate=(
            "atom %{customdata[0]}<br>source: %{customdata[1]}<br>"
            "destination: %{customdata[2]}<extra></extra>"
        ),
    )


def _gate_trace(
    statement: ir.Statement,
    arch_spec: ArchSpec,
    bounds: tuple[float, float, float, float],
    colors: dict[str, str],
) -> Any:
    go = _plotly()
    local_gate_types = (move.LocalR, move.LocalRz, move.StarRz)
    if isinstance(statement, local_gate_types):
        positions = [
            arch_spec.get_position(location)
            for location in statement.location_addresses
        ]
        color = (
            colors["local_r"]
            if isinstance(statement, move.LocalR)
            else colors["local_rz"]
        )
        return go.Scatter(
            x=[position[0] for position in positions],
            y=[position[1] for position in positions],
            mode="markers",
            marker={"color": color, "size": 34, "opacity": 0.3},
            name=type(statement).__name__,
            showlegend=False,
            visible=True,
            hovertemplate=f"{type(statement).__name__}<extra></extra>",
        )

    region: tuple[float, float, float, float] | None = None
    color = colors["local_r"]
    if isinstance(statement, (move.GlobalR, move.GlobalRz)):
        region = bounds
        color = (
            colors["local_r"]
            if isinstance(statement, move.GlobalR)
            else colors["local_rz"]
        )
    elif isinstance(statement, move.CZ):
        zone_positions = [
            position
            for location, position in ArchVisualizer(arch_spec)._iter_locations()
            if location.zone_id == statement.zone_address.zone_id
        ]
        if zone_positions:
            x_values = [position[0] for position in zone_positions]
            y_values = [position[1] for position in zone_positions]
            y_span = max(y_values) - min(y_values)
            y_padding = 0.1 * y_span if y_span else 1.0
            region = (
                min(x_values) - 10.0,
                max(x_values) + 10.0,
                min(y_values) - y_padding,
                max(y_values) + y_padding,
            )
            color = colors["cz"]

    if region is None:
        return _empty_trace("Gate highlight")

    x_min, x_max, y_min, y_max = region
    return go.Scatter(
        x=[x_min, x_max, x_max, x_min, x_min],
        y=[y_min, y_min, y_max, y_max, y_min],
        mode="lines",
        line={"width": 0},
        fill="toself",
        fillcolor=color,
        opacity=0.25,
        name=type(statement).__name__,
        showlegend=False,
        visible=True,
        hovertemplate=f"{type(statement).__name__}<extra></extra>",
    )


def _dynamic_traces(
    step: DebugStep,
    route_groups: Sequence[Sequence[RouteSegment]],
    route_trace_count: int,
    arch_spec: ArchSpec,
    colors: dict[str, str],
    atom_marker: str,
    bounds: tuple[float, float, float, float],
) -> list[Any]:
    return [
        *(
            _route_trace(route_groups[index] if index < len(route_groups) else None)
            for index in range(route_trace_count)
        ),
        _gate_trace(step.statement, arch_spec, bounds, colors),
        _atom_trace(step.state, arch_spec, colors, atom_marker),
    ]


def _slider_step(frame_names: Sequence[str], label: str) -> dict[str, object]:
    return {
        "label": label,
        "method": "animate",
        "args": [
            list(frame_names),
            {
                "mode": "immediate",
                "frame": {"duration": 0, "redraw": True},
                "transition": {"duration": 0},
            },
        ],
    }


def build_plotly_debugger_figure(
    mt: ir.Method,
    arch_spec: ArchSpec,
    *,
    interactive: bool,
    pause_time: float,
    atom_marker: str,
    theme: Theme,
    height: int,
) -> Figure:
    """Build the Plotly figure without displaying it."""
    if pause_time < 0:
        raise ValueError("pause_time must be non-negative")
    go = _plotly()
    steps = collect_debug_steps(mt, arch_spec)
    colors = _theme_colors(theme)
    bounds = _view_bounds(arch_spec)
    frames: list[Any] = []
    route_groups_by_step = [
        _group_route_segments(_route_segments(step.state, arch_spec)) for step in steps
    ]
    route_trace_count = max(
        1,
        max((len(groups) for groups in route_groups_by_step), default=0),
    )
    dynamic_trace_indices = list(range(1, route_trace_count + 3))

    for step_index, step in enumerate(steps):
        frames.append(
            go.Frame(
                name=f"step-{step_index}",
                data=_dynamic_traces(
                    step,
                    route_groups_by_step[step_index],
                    route_trace_count,
                    arch_spec,
                    colors,
                    atom_marker,
                    bounds,
                ),
                traces=dynamic_trace_indices,
                layout={"title": {"text": step.title}},
            )
        )

    initial_data = (
        _dynamic_traces(
            steps[0],
            route_groups_by_step[0],
            route_trace_count,
            arch_spec,
            colors,
            atom_marker,
            bounds,
        )
        if steps
        else [
            *(_route_trace(None) for _ in range(route_trace_count)),
            _empty_trace("Gate highlight"),
            _empty_trace("Atoms"),
        ]
    )
    title = steps[0].title if steps else "Plotly move debugger: no atom-state steps"
    figure = go.Figure(
        data=[_site_trace(arch_spec, colors, atom_marker), *initial_data]
    )
    figure.frames = frames

    sliders = []
    update_menus = []
    if interactive and frames:
        sliders = [
            {
                "active": 0,
                "currentvalue": {"prefix": "Step: "},
                "pad": {"t": 45},
                "steps": [
                    _slider_step([frame.name], str(index + 1))
                    for index, frame in enumerate(frames)
                ],
            }
        ]
        if len(frames) > 1:
            update_menus = [
                {
                    "type": "buttons",
                    "direction": "left",
                    "x": 0.0,
                    "y": 1.10,
                    "showactive": False,
                    "buttons": [
                        {
                            "label": "Play steps",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "fromcurrent": True,
                                    "mode": "immediate",
                                    "frame": {
                                        "duration": max(1, round(pause_time * 1000)),
                                        "redraw": True,
                                    },
                                    "transition": {"duration": 0},
                                },
                            ],
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [
                                [None],
                                {
                                    "mode": "immediate",
                                    "frame": {"duration": 0, "redraw": True},
                                    "transition": {"duration": 0},
                                },
                            ],
                        },
                    ],
                }
            ]

    x_min, x_max, y_min, y_max = bounds
    figure.update_layout(
        template="plotly_white" if theme == "light" else "plotly_dark",
        title={"text": title, "x": 0.01, "xanchor": "left"},
        height=height,
        margin={"l": 55, "r": 25, "t": 95, "b": 65},
        paper_bgcolor=colors["paper"],
        plot_bgcolor=colors["plot"],
        font={"color": colors["text"]},
        hovermode="closest",
        dragmode="pan",
        showlegend=False,
        uirevision="bloqade-plotly-debugger",
        xaxis={
            "title": "x (µm)",
            "range": [x_min, x_max],
            "showgrid": False,
            "zeroline": False,
            "scaleanchor": "y",
            "scaleratio": 1,
        },
        yaxis={
            "title": "y (µm)",
            "range": [y_min, y_max],
            "showgrid": False,
            "zeroline": False,
        },
        sliders=sliders,
        updatemenus=update_menus,
        meta={
            "bloqadePlotlyDebugger": {
                "stepCount": len(steps),
                "frameCount": len(frames),
                "routeTraceCount": route_trace_count,
            }
        },
    )
    return figure


def _in_jupyter_kernel() -> bool:
    try:
        from IPython.core.getipython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and type(shell).__name__ == "ZMQInteractiveShell"


def _show_plotly_debugger(
    figure: Figure,
    *,
    renderer: str | None,
    auto_play: bool,
    frame_duration_ms: int,
) -> None:
    selected_renderer = renderer
    if selected_renderer is None and not _in_jupyter_kernel():
        selected_renderer = "browser"
    figure.show(
        renderer=selected_renderer,
        config={"responsive": True, "scrollZoom": True},
        auto_play=auto_play,
        animation_opts={
            "frame": {"duration": frame_duration_ms, "redraw": True},
            "transition": {"duration": 0},
            "fromcurrent": True,
            "mode": "immediate",
        },
    )


def plotly_debugger(
    mt: ir.Method,
    arch_spec: ArchSpec,
    interactive: bool = True,
    pause_time: float = 1.0,
    atom_marker: str = "o",
    *,
    show: bool = True,
    renderer: str | None = None,
    theme: Theme = "light",
    height: int = 720,
) -> Figure | None:
    """Display a browser-native, discrete-step move-program debugger.

    This is a Plotly alternative to :func:`debugger`; it does not replace the
    existing Matplotlib debugger or :func:`animated_debugger`. In a notebook,
    Plotly embeds the controls in the cell output. In a regular Python process,
    the default renderer opens the figure in a browser.

    Args:
        mt: Compiled physical move program to visualize.
        arch_spec: Architecture used to compile ``mt``.
        interactive: Include the step slider and play/pause controls. When
            false, displaying the figure automatically plays all steps.
        pause_time: Seconds between steps during playback.
        atom_marker: Matplotlib-style marker name for atoms (for example,
            ``"o"`` or ``"s"``), or a Plotly marker symbol.
        show: Display the figure immediately. Set false to customize it first.
        renderer: Explicit Plotly renderer. The default embeds in Jupyter and
            uses ``"browser"`` in a regular Python process.
        theme: Initial light or dark theme.
        height: Figure height in pixels.

    Returns:
        The constructed Plotly figure when ``show=False``; otherwise ``None``.
    """
    figure = build_plotly_debugger_figure(
        mt,
        arch_spec,
        interactive=interactive,
        pause_time=pause_time,
        atom_marker=atom_marker,
        theme=theme,
        height=height,
    )
    if show:
        _show_plotly_debugger(
            figure,
            renderer=renderer,
            auto_play=not interactive,
            frame_duration_ms=max(1, round(pause_time * 1000)),
        )
        return None
    return figure
