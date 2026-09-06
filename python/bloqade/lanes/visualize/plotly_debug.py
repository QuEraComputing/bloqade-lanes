from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from kirin import ir

from bloqade.lanes.analysis.atom import AtomState
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode.encoding import LaneAddress, LocationAddress, MoveType
from bloqade.lanes.dialects import move
from bloqade.lanes.visualize.arch import ArchVisualizer
from bloqade.lanes.visualize.artist import DebugStep, collect_debug_steps

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


@dataclass(frozen=True)
class _MovePathSegment:
    """Plotly data for one segment of an atom's latest move."""

    qubit_id: int
    source: LocationAddress
    destination: LocationAddress
    start: tuple[float, float]
    end: tuple[float, float]
    color_value: float
    bus_label: str


def _lane_bus_label(lane: LaneAddress) -> str:
    kind = lane.move_type.name.capitalize()
    if lane.move_type == MoveType.ZONE:
        return f"Zone bus {lane.bus_id}"
    return f"Zone ID {lane.zone_id}, {kind} bus {lane.bus_id}"


def _move_path_segments(
    state: AtomState, arch_spec: ArchSpec
) -> list[_MovePathSegment]:
    """Translate atom-analysis lanes into Plotly path segments."""
    segments: list[_MovePathSegment] = []
    for qubit_id, lane in sorted(state.data.prev_lanes.items()):
        source, destination = arch_spec.get_endpoints(lane)
        path = arch_spec.get_path(lane)
        path_segments = list(itertools.pairwise(path))
        denominator = max(1, len(path_segments) - 1)
        segments.extend(
            _MovePathSegment(
                qubit_id=qubit_id,
                source=source,
                destination=destination,
                start=start,
                end=end,
                color_value=index / denominator,
                bus_label=_lane_bus_label(lane),
            )
            for index, (start, end) in enumerate(path_segments)
        )
    return segments


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
            "measure": "#d97706",
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
            "measure": "#fbbf24",
        }
    raise ValueError("theme must be 'light' or 'dark'")


def _empty_trace(name: str) -> Any:
    go = _plotly()
    return go.Scatter(x=[], y=[], mode="lines", name=name, visible=False)


def _atom_trace(
    step: DebugStep,
    arch_spec: ArchSpec,
    colors: dict[str, str],
    atom_marker: str,
) -> Any:
    go = _plotly()
    state = step.state
    atoms = [
        (qubit_id, location, arch_spec.get_position(location))
        for qubit_id, location in sorted(state.data.qubit_to_locations.items())
    ]
    atom_customdata = [
        [
            qubit_id,
            location.zone_id,
            location.word_id,
            location.site_id,
            *arch_spec.words[location.word_id].sites[location.site_id],
            (
                f"<br><b>gate:</b> {_gate_description(step)}"
                if _gate_applies_to_location(step.statement, location)
                else ""
            ),
        ]
        for qubit_id, location, _ in atoms
    ]
    return go.Scatter(
        x=[position[0] for _, _, position in atoms],
        y=[position[1] for _, _, position in atoms],
        customdata=atom_customdata,
        text=[str(qubit_id) for qubit_id, _, _ in atoms],
        mode="markers+text",
        textposition="middle center",
        textfont={"color": "white", "size": 10},
        marker={
            "color": colors["atom"],
            "size": 18,
            "symbol": _ATOM_SYMBOLS.get(atom_marker, atom_marker),
            "line": {"color": colors["paper"], "width": 1},
        },
        meta={"bloqadeTraceKind": "atom"},
        name="Atoms",
        showlegend=False,
        hovertemplate=(
            "<b>atom %{customdata[0]}</b><br>"
            "(zone, word, site): (%{customdata[1]}, %{customdata[2]}, "
            "%{customdata[3]})<br>"
            "grid (x, y): (%{customdata[4]}, %{customdata[5]})<br>"
            "position (x, y): (%{x:.3f}, %{y:.3f}) µm"
            "%{customdata[6]}<extra></extra>"
        ),
    )


def _gate_parameter_names(statement: ir.Statement) -> tuple[str, ...]:
    if isinstance(statement, (move.LocalR, move.GlobalR)):
        return ("axis_angle", "rotation_angle")
    if isinstance(statement, (move.LocalRz, move.StarRz, move.GlobalRz)):
        return ("rotation_angle",)
    return ()


def _gate_description(step: DebugStep) -> str:
    """Format an interpreted gate with named, evaluated parameters."""
    if isinstance(step.statement, move.EndMeasure):
        zones = tuple(address.zone_id for address in step.statement.zone_addresses)
        return f"EndMeasure(zones={zones})"

    parameters = ", ".join(
        f"{name}={value!r}"
        for name, value in zip(
            _gate_parameter_names(step.statement),
            step.parameter_values,
            strict=False,
        )
    )
    return f"{type(step.statement).__name__}({parameters})"


def _gate_applies_to_location(
    statement: ir.Statement, location: LocationAddress
) -> bool:
    if isinstance(statement, (move.LocalR, move.LocalRz, move.StarRz)):
        return location in statement.location_addresses
    if isinstance(statement, (move.GlobalR, move.GlobalRz)):
        return True
    if isinstance(statement, move.CZ):
        return location.zone_id == statement.zone_address.zone_id
    if isinstance(statement, move.EndMeasure):
        return any(
            location.zone_id == address.zone_id for address in statement.zone_addresses
        )
    return False


def _viridis_color(color_value: float) -> str:
    from plotly.colors import sample_colorscale

    return cast(str, sample_colorscale("Viridis", [color_value])[0])


def _group_route_segments(
    route_segments: Sequence[_MovePathSegment],
) -> list[list[_MovePathSegment]]:
    """Group equally colored segments so simultaneous moves share traces."""
    segments_by_color: dict[float, list[_MovePathSegment]] = {}
    for segment in route_segments:
        segments_by_color.setdefault(segment.color_value, []).append(segment)
    return list(segments_by_color.values())


def _route_trace(segments: Sequence[_MovePathSegment] | None) -> Any:
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
            meta={"bloqadeTraceKind": "movePath", "segments": []},
        )

    x_values: list[float | None] = []
    y_values: list[float | None] = []
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
        marker_sizes.extend([0, 16, 0])
        marker_symbols.extend(["circle", "arrow", "circle"])
        marker_angles.extend([0, arrow_angle, 0])

    color = _viridis_color(segments[0].color_value)
    segment_metadata = [
        {
            "atomId": segment.qubit_id,
            "busLabel": segment.bus_label,
            "source": (
                f"({segment.source.zone_id}, {segment.source.word_id}, "
                f"{segment.source.site_id})"
            ),
            "destination": (
                f"({segment.destination.zone_id}, {segment.destination.word_id}, "
                f"{segment.destination.site_id})"
            ),
            "color": color,
            "start": list(segment.start),
            "end": list(segment.end),
        }
        for segment in segments
    ]
    return go.Scatter(
        x=x_values,
        y=y_values,
        mode="lines+markers",
        line={"color": color, "width": 4},
        marker={
            "color": color,
            "size": marker_sizes,
            "symbol": marker_symbols,
            "angle": marker_angles,
            "angleref": "up",
            "line": {"color": color, "width": 1.5},
        },
        name="Move path",
        meta={
            "bloqadeTraceKind": "movePath",
            "segments": segment_metadata,
        },
        showlegend=False,
        visible=True,
        cliponaxis=False,
        # A custom SVG hit target spans the entire segment and anchors its
        # single tooltip at the segment center. Native marker hover would add
        # a second tooltip specifically at each arrowhead.
        hoverinfo="skip",
    )


def _global_gate_bounds(arch_spec: ArchSpec) -> tuple[float, float, float, float]:
    visualizer = ArchVisualizer(arch_spec)
    x_min, x_max = visualizer.x_bounds
    y_min, y_max = visualizer.y_bounds
    x_span = x_max - x_min
    y_span = y_max - y_min
    x_padding = 0.5 * x_span if x_span else 1.0
    y_padding = 0.5 * y_span if y_span else 1.0
    return (
        x_min - x_padding,
        x_max + x_padding,
        y_min - y_padding,
        y_max + y_padding,
    )


def _zone_gate_bounds(
    zone_ids: set[int], arch_spec: ArchSpec
) -> tuple[float, float, float, float] | None:
    zone_positions = [
        position
        for location, position in ArchVisualizer(arch_spec)._iter_locations()
        if location.zone_id in zone_ids
    ]
    if not zone_positions:
        return None

    x_values = [position[0] for position in zone_positions]
    y_values = [position[1] for position in zone_positions]
    y_span = max(y_values) - min(y_values)
    y_padding = 0.1 * y_span if y_span else 1.0
    return (
        min(x_values) - 10.0,
        max(x_values) + 10.0,
        min(y_values) - y_padding,
        max(y_values) + y_padding,
    )


def _gate_trace(
    step: DebugStep,
    arch_spec: ArchSpec,
    colors: dict[str, str],
) -> Any:
    go = _plotly()
    statement = step.statement
    gate_description = _gate_description(step)
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
            hovertemplate=f"{gate_description}<extra></extra>",
        )

    region: tuple[float, float, float, float] | None = None
    color = colors["local_r"]
    if isinstance(statement, (move.GlobalR, move.GlobalRz)):
        region = _global_gate_bounds(arch_spec)
        color = (
            colors["local_r"]
            if isinstance(statement, move.GlobalR)
            else colors["local_rz"]
        )
    elif isinstance(statement, move.CZ):
        region = _zone_gate_bounds({statement.zone_address.zone_id}, arch_spec)
        color = colors["cz"]
    elif isinstance(statement, move.EndMeasure):
        region = _zone_gate_bounds(
            {address.zone_id for address in statement.zone_addresses},
            arch_spec,
        )
        color = colors["measure"]

    if region is None:
        return _empty_trace("Gate highlight")

    x_min, x_max, y_min, y_max = region
    return go.Scatter(
        x=[x_min, x_max, x_max, x_min, x_min],
        y=[y_min, y_min, y_max, y_max, y_min],
        mode="lines",
        line={"width": 0},
        fill="toself",
        hoveron="fills",
        fillcolor=color,
        opacity=0.25,
        name=type(statement).__name__,
        showlegend=False,
        visible=True,
        hovertemplate=f"{gate_description}<extra></extra>",
        zorder=-10,
    )


def _dynamic_traces(
    step: DebugStep,
    route_groups: Sequence[Sequence[_MovePathSegment]],
    route_trace_count: int,
    arch_spec: ArchSpec,
    colors: dict[str, str],
    atom_marker: str,
) -> list[Any]:
    return [
        *(
            _route_trace(route_groups[index] if index < len(route_groups) else None)
            for index in range(route_trace_count)
        ),
        _gate_trace(step, arch_spec, colors),
        _atom_trace(step, arch_spec, colors, atom_marker),
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
    route_groups_by_step = [
        _group_route_segments(_move_path_segments(step.state, arch_spec))
        for step in steps
    ]
    route_trace_count = max(
        1,
        max((len(groups) for groups in route_groups_by_step), default=0),
    )

    # Start with the complete interactive architecture rather than rebuilding
    # its sites and buses here. This preserves one source of truth for site
    # metadata, bus selectors, hover highlighting, and transport paths.
    figure = ArchVisualizer(arch_spec).plot_interactive(
        show_site_ids=False,
        show_all_buses=False,
        show_bus_legend=False,
        path_style="exact",
        site_lane_preview="click",
        bus_line_style="dashed",
        theme=theme,
        height=height,
    )
    architecture_trace_count = len(cast(Any, figure.data))
    dynamic_trace_indices = list(
        range(
            architecture_trace_count,
            architecture_trace_count + route_trace_count + 2,
        )
    )
    frames: list[Any] = []

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
        )
        if steps
        else [
            *(_route_trace(None) for _ in range(route_trace_count)),
            _empty_trace("Gate highlight"),
            _empty_trace("Atoms"),
        ]
    )
    title = steps[0].title if steps else "Plotly move debugger: no atom-state steps"
    figure.add_traces(initial_data)
    figure.frames = frames

    sliders = []
    debugger_update_menus = []
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
            debugger_update_menus = [
                {
                    "type": "buttons",
                    "direction": "left",
                    "x": 0.0,
                    "xanchor": "left",
                    # Keep playback on the same row as the architecture
                    # controls in the upper margin.
                    "y": 1.09,
                    "yanchor": "top",
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

    architecture_meta = dict(cast(Any, figure.layout.meta) or {})
    architecture_update_menus = list(figure.layout.updatemenus or ())
    # Make room for playback at the left, then center the complete four-group
    # control row over the plot. ``plot_interactive`` only has the latter
    # three groups, so this debugger-specific alignment belongs here.
    for menu, x_position in zip(
        architecture_update_menus,
        (0.18, 0.47, 0.75),
    ):
        menu.x = x_position
        menu.y = 1.09
    figure.layout.annotations = ()
    figure.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left"},
        height=height,
        margin={"l": 70, "r": 270, "t": 165, "b": 65},
        showlegend=False,
        uirevision="bloqade-plotly-debugger",
        sliders=sliders,
        updatemenus=[*architecture_update_menus, *debugger_update_menus],
        meta={
            **architecture_meta,
            "bloqadePlotlyDebugger": {
                "stepCount": len(steps),
                "frameCount": len(frames),
                "frameNames": [frame.name for frame in frames],
                "routeTraceCount": route_trace_count,
                "architectureTraceCount": architecture_trace_count,
            },
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
