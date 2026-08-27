"""Visualization helpers for :class:`~bloqade.lanes.arch.spec.ArchSpec`.

These used to live as methods on ``ArchSpec`` itself. They were extracted
as part of #464 phase 1 so the core ``ArchSpec`` Python wrapper stays
focused on architectural data and validation, keeping matplotlib out of
its import surface.

The primary entry point is the :class:`ArchVisualizer` class, which
caches bounds computations and provides ``plot`` / ``show`` methods.
The ``ArchSpec`` shims (``arch_spec.plot``, ``.show``, ``.x_bounds``,
``.y_bounds``, ``.path_bounds``) create an ``ArchVisualizer`` via a
``@cached_property`` so existing call sites keep working.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from functools import cached_property
from math import hypot
from types import MethodType
from typing import TYPE_CHECKING, Any, Literal, cast

from bloqade.lanes.bytecode.encoding import (
    Direction,
    LaneAddress,
    LocationAddress,
    MoveType,
    SiteLaneAddress,
    WordLaneAddress,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from matplotlib.axes import Axes
    from plotly.graph_objects import Figure

    from bloqade.lanes.arch.spec import ArchSpec


__all__ = [
    "ArchVisualizer",
]

_BusTraceData = tuple[
    list[float | None],
    list[float | None],
    list[tuple[str, str, str] | None],
]

_BUS_HOVER_POST_SCRIPT = """
(function () {
  const plot = document.getElementById('{plot_id}');
  if (!plot || plot.__archVisualizerHoverInstalled) return;
  plot.__archVisualizerHoverInstalled = true;

  const busIndices = (plot.layout.meta || {}).archVisualizerBusTraceIndices || [];
  const busIndexSet = new Set(busIndices);
  let highlightPath = null;

  function clearHighlight() {
    if (highlightPath !== null) highlightPath.remove();
    highlightPath = null;
  }

  function dashPattern(dash) {
    return {
      dot: '2,5',
      dash: '9,6',
      longdash: '14,7',
      dashdot: '9,5,2,5',
      longdashdot: '14,6,2,6'
    }[dash] || null;
  }

  function drawHighlight(index) {
    clearHighlight();

    const trace = plot.data[index];
    const xAxis = plot._fullLayout.xaxis;
    const yAxis = plot._fullLayout.yaxis;
    const hoverLayer = plot.querySelector('.hoverlayer');
    if (!trace || !xAxis || !yAxis || !hoverLayer) return;

    let pathData = '';
    let drawing = false;
    for (let pointIndex = 0; pointIndex < trace.x.length; pointIndex++) {
      const x = trace.x[pointIndex];
      const y = trace.y[pointIndex];
      if (!Number.isFinite(x) || !Number.isFinite(y)) {
        drawing = false;
        continue;
      }

      const pixelX = xAxis._offset + xAxis.l2p(x);
      const pixelY = yAxis._offset + yAxis.l2p(y);
      pathData += `${drawing ? 'L' : 'M'}${pixelX},${pixelY}`;
      drawing = true;
    }
    if (!pathData) return;

    const line = trace.line || {};
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', pathData);
    path.setAttribute('data-arch-visualizer-bus-highlight', '');
    path.setAttribute('fill', 'none');
    path.setAttribute('stroke', line.color || '#f59e0b');
    path.setAttribute(
      'stroke-width',
      String(Math.max(6, Number(line.width || 2.25) * 2.6))
    );
    path.setAttribute('stroke-opacity', '1');
    path.setAttribute('stroke-linecap', 'round');
    path.setAttribute('stroke-linejoin', 'round');
    path.setAttribute('pointer-events', 'none');
    const strokeDasharray = dashPattern(line.dash);
    if (strokeDasharray !== null) {
      path.setAttribute('stroke-dasharray', strokeDasharray);
    }

    // Keep the highlight above every WebGL trace, including exactly
    // overlapping buses, but behind Plotly's tooltip text.
    hoverLayer.insertBefore(path, hoverLayer.firstChild);
    highlightPath = path;
  }

  plot.on('plotly_hover', function (event) {
    const point = event.points.find((item) => busIndexSet.has(item.curveNumber));
    if (!point) return;
    drawHighlight(point.curveNumber);
  });

  plot.on('plotly_unhover', clearHighlight);
  plot.on('plotly_relayout', clearHighlight);
})();
"""


def _install_bus_hover_html(figure: Any) -> None:
    """Add bus highlighting to this figure's HTML representation.

    Plotly does not have a declarative per-trace hover-line-width property, so
    the behavior is installed with the ``plotly_hover`` and ``plotly_unhover``
    browser events supported by :meth:`plotly.graph_objects.Figure.to_html`.
    The instance remains an ordinary Plotly ``Figure``.
    """
    original_to_html = figure.to_html

    def to_html_with_bus_hover(
        _figure: Any,
        *args: Any,
        post_script: str | Sequence[str] | None = None,
        **kwargs: Any,
    ) -> str:
        scripts = [_BUS_HOVER_POST_SCRIPT]
        if isinstance(post_script, str):
            scripts.append(post_script)
        elif post_script is not None:
            scripts.extend(post_script)
        config = dict(kwargs.pop("config", None) or {})
        config.setdefault("scrollZoom", True)
        config.setdefault("responsive", True)
        return cast(
            str,
            original_to_html(
                *args,
                post_script=scripts,
                config=config,
                **kwargs,
            ),
        )

    figure.to_html = MethodType(to_html_with_bus_hover, figure)


def _location_position(
    arch_spec: ArchSpec, word_id: int, site_id: int, zone_id: int
) -> tuple[float, float] | None:
    """Return (x, y) for a site or ``None`` if the triple is invalid.

    Uses the optional-returning Rust lookup directly so callers can
    iterate all (zone, word, site) combinations without raising on
    architectures where not every word exists in every zone.
    """
    return arch_spec._inner.location_position(
        LocationAddress(word_id, site_id, zone_id)._inner
    )


class ArchVisualizer:
    """Visualization facade for an :class:`ArchSpec`.

    Construct once from an architecture spec; bounds are cached so
    repeated calls to ``plot`` or ``show`` don't recompute site
    positions.

    Example::

        viz = ArchVisualizer(arch_spec)
        viz.plot(ax, show_words=[0, 1], show_word_bus=[0])
        viz.plot_interactive()
        print(viz.x_bounds, viz.y_bounds)
    """

    def __init__(self, arch_spec: ArchSpec) -> None:
        self.arch_spec = arch_spec

    # ── Bounds (cached) ──────────────────────────────────────────

    @cached_property
    def x_bounds(self) -> tuple[float, float]:
        """``(x_min, x_max)`` across every site. Falls back to
        ``(-1.0, 1.0)`` when no sites are discoverable."""
        x_min = float("inf")
        x_max = float("-inf")
        arch = self.arch_spec
        for zone_id in range(len(arch.zones)):
            for word_id in range(len(arch.words)):
                for site_id in range(len(arch.words[word_id].site_indices)):
                    pos = _location_position(arch, word_id, site_id, zone_id)
                    if pos is not None:
                        x_min = min(x_min, pos[0])
                        x_max = max(x_max, pos[0])
        if x_min == float("inf"):
            x_min = -1.0
        if x_max == float("-inf"):
            x_max = 1.0
        return x_min, x_max

    @cached_property
    def y_bounds(self) -> tuple[float, float]:
        """``(y_min, y_max)`` across every site. Falls back to
        ``(-1.0, 1.0)`` when no sites are discoverable."""
        y_min = float("inf")
        y_max = float("-inf")
        arch = self.arch_spec
        for zone_id in range(len(arch.zones)):
            for word_id in range(len(arch.words)):
                for site_id in range(len(arch.words[word_id].site_indices)):
                    pos = _location_position(arch, word_id, site_id, zone_id)
                    if pos is not None:
                        y_min = min(y_min, pos[1])
                        y_max = max(y_max, pos[1])
        if y_min == float("inf"):
            y_min = -1.0
        if y_max == float("-inf"):
            y_max = 1.0
        return y_min, y_max

    def path_bounds(self) -> tuple[float, float, float, float]:
        """``(x_min, x_max, y_min, y_max)`` covering every site **and**
        every transport-path waypoint registered on the arch."""
        x_min, x_max = self.x_bounds
        y_min, y_max = self.y_bounds
        for path in self.arch_spec.paths.values():
            for x, y in path:
                x_min = min(x_min, x)
                x_max = max(x_max, x)
                y_min = min(y_min, y)
                y_max = max(y_max, y)
        return (x_min, x_max, y_min, y_max)

    # ── Bus-path iterators ───────────────────────────────────────

    def iter_word_bus_paths(
        self, show_word_bus: Sequence[int]
    ) -> Iterator[tuple[tuple[float, float], ...]]:
        arch = self.arch_spec
        for zone_id, zone in enumerate(arch.zones):
            for lane_id in show_word_bus:
                if lane_id >= len(zone.word_buses):
                    continue
                lane = zone.word_buses[lane_id]
                for site_id in zone.sites_with_word_buses:
                    for start_word_id in lane.src:
                        lane_addr = WordLaneAddress(
                            zone_id=zone_id,
                            word_id=start_word_id,
                            site_id=site_id,
                            bus_id=lane_id,
                            direction=Direction.FORWARD,
                        )
                        yield arch.get_path(lane_addr)

    def iter_site_bus_paths(
        self,
        show_words: Sequence[int],
        show_site_bus: Sequence[int],
    ) -> Iterator[tuple[tuple[float, float], ...]]:
        arch = self.arch_spec
        for zone_id, zone in enumerate(arch.zones):
            words_with_site_buses = set(zone.words_with_site_buses)
            for word_id in show_words:
                if word_id not in words_with_site_buses:
                    continue
                for lane_id in show_site_bus:
                    if lane_id >= len(zone.site_buses):
                        continue
                    lane = zone.site_buses[lane_id]
                    for i in range(len(lane.src)):
                        lane_addr = SiteLaneAddress(
                            zone_id=zone_id,
                            word_id=word_id,
                            site_id=lane.src[i],
                            bus_id=lane_id,
                            direction=Direction.FORWARD,
                        )
                        yield arch.get_path(lane_addr)

    def _iter_locations(
        self,
    ) -> Iterator[tuple[LocationAddress, tuple[float, float]]]:
        """Yield every valid location and its physical position."""
        arch = self.arch_spec
        for zone_id in range(len(arch.zones)):
            for word_id, word in enumerate(arch.words):
                for site_id in range(len(word.site_indices)):
                    location = LocationAddress(word_id, site_id, zone_id)
                    if (position := arch.try_get_position(location)) is not None:
                        yield location, position

    def _forward_lanes_by_bus(
        self,
    ) -> dict[tuple[MoveType, int | None, int], list[LaneAddress]]:
        """Group canonical forward lanes by their user-facing bus identity.

        Site- and word-bus IDs are local to a zone, while zone-bus IDs are
        architecture-wide. Consequently, zone buses deliberately use ``None``
        for the zone portion of the key.
        """
        buses: dict[tuple[MoveType, int | None, int], list[LaneAddress]] = defaultdict(
            list
        )
        for lane in self.arch_spec.iter_all_lanes():
            if lane.direction != Direction.FORWARD:
                continue
            zone_id = None if lane.move_type == MoveType.ZONE else lane.zone_id
            buses[(lane.move_type, zone_id, lane.bus_id)].append(lane)
        return dict(buses)

    def _cartoon_path(self, lane: LaneAddress) -> tuple[tuple[float, float], ...]:
        """Return a schematic path that emphasizes bus connectivity.

        Word buses preserve ``site_id`` and therefore represent motion within
        one site column; they are shown as direct lines. Site buses connect
        different site columns, so they use a curved arch like the diagrams in
        ``demo/physical_arch_customization.py``. Zone buses are also arched to
        distinguish inter-zone motion from within-column motion.
        """
        src, dst = self.arch_spec.get_endpoints(lane)
        start = self.arch_spec.get_position(src)
        end = self.arch_spec.get_position(dst)
        if lane.move_type == MoveType.WORD:
            return (start, end)

        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = hypot(dx, dy)
        if distance == 0.0:
            return (start, end)

        # A quadratic Bézier approximation of matplotlib's ``arc3`` path.
        # The perpendicular offset keeps the endpoint relationship legible
        # without pretending to be the device's exact transport trajectory.
        bend = -0.34 * distance
        control = (
            (start[0] + end[0]) / 2 - dy / distance * bend,
            (start[1] + end[1]) / 2 + dx / distance * bend,
        )
        path: list[tuple[float, float]] = []
        for step in range(21):
            t = step / 20
            one_minus_t = 1.0 - t
            path.append(
                (
                    one_minus_t**2 * start[0]
                    + 2 * one_minus_t * t * control[0]
                    + t**2 * end[0],
                    one_minus_t**2 * start[1]
                    + 2 * one_minus_t * t * control[1]
                    + t**2 * end[1],
                )
            )
        return tuple(path)

    # ── Rendering ────────────────────────────────────────────────

    def plot(
        self,
        ax: Axes | None = None,
        show_words: Sequence[int] = (),
        show_site_bus: Sequence[int] = (),
        show_word_bus: Sequence[int] = (),
        **scatter_kwargs,
    ) -> Axes:
        """Render the architecture onto a matplotlib axes.

        Returns the ``ax`` argument (or the auto-resolved current axes)
        so callers can chain or further customise the plot.
        """
        import matplotlib.pyplot as plt  # type: ignore[import-untyped]

        if ax is None:
            ax = plt.gca()

        arch = self.arch_spec
        for word_id in show_words:
            word = arch.words[word_id]
            positions: list[tuple[float, float]] = []
            for zone_id in range(len(arch.zones)):
                for site_id in range(len(word.site_indices)):
                    pos = _location_position(arch, word_id, site_id, zone_id)
                    if pos is not None:
                        positions.append(pos)
                if positions:
                    break
            if positions:
                x_positions = [p[0] for p in positions]
                y_positions = [p[1] for p in positions]
                ax.scatter(x_positions, y_positions, **scatter_kwargs)

        for path in self.iter_site_bus_paths(show_words, show_site_bus):
            x_vals, y_vals = zip(*path)
            ax.plot(x_vals, y_vals, linestyle="--")

        for path in self.iter_word_bus_paths(show_word_bus):
            x_vals, y_vals = zip(*path)
            ax.plot(x_vals, y_vals, linestyle="-")
        return ax

    def plot_interactive(
        self,
        *,
        show_site_ids: bool = False,
        show_all_buses: bool = False,
        path_style: Literal["exact", "cartoon"] = "exact",
        theme: Literal["light", "dark"] = "light",
        width: int = 1300,
        height: int = 800,
    ) -> Figure:
        """Build an interactive architecture and atom-path visualization.

        The returned Plotly figure supports pan, box zoom, and mode-bar zoom.
        Its HTML representation also enables wheel and trackpad zoom.
        Hovering a site always shows its ``(zone_id, word_id, site_id)``.
        Hovering a bus shows the source and destination site addresses and
        temporarily draws that bus as a thicker overlay above other buses.
        Site labels can be toggled with the controls above the plot.

        Every bus is a separate legend item, so clicking legend entries works
        as a multiselector and double-clicking isolates one bus. Bus traces are
        hidden by default to keep large architectures readable; pass
        ``show_all_buses=True`` or use the ``Show all`` control to display all
        of them. The path-view toggle switches between exact architecture paths
        from :meth:`ArchSpec.get_path` and a less cluttered schematic view.
        Schematic word-bus paths are straight because they preserve a site
        column; site- and zone-bus paths are curved to expose crossings.

        Args:
            show_site_ids: Show site identity labels initially. Site identity
                remains available on hover when labels are hidden.
            show_all_buses: Show all bus paths initially. By default, buses are
                available in the legend but hidden to avoid visual clutter.
            path_style: Initial bus-path representation. ``"exact"`` preserves
                architecture-defined transport waypoints; ``"cartoon"``
                emphasizes bus connectivity with straight and curved paths.
            theme: Initial color theme. The default is explicit light mode so
                the figure does not inherit a dark notebook or documentation
                background.
            width: Figure width in pixels. The roomy default helps separate the
                160 sites in the Gemini physical architecture.
            height: Figure height in pixels.

        Returns:
            A ``plotly.graph_objects.Figure``. In a notebook, return it from a
            cell or call ``figure.show(config={"scrollZoom": True})`` to also
            enable scroll-wheel zoom.

        Raises:
            ImportError: If the ``visualization`` optional dependency group is
                not installed.
            ValueError: If ``path_style`` or ``theme`` is unsupported.
        """
        try:
            import plotly.graph_objects as go
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "Interactive architecture plots require Plotly. Install "
                "bloqade-lanes with the 'visualization' extra."
            ) from exc

        if path_style not in ("exact", "cartoon"):
            raise ValueError("path_style must be 'exact' or 'cartoon'")
        if theme not in ("light", "dark"):
            raise ValueError("theme must be 'light' or 'dark'")

        locations = list(self._iter_locations())
        buses = self._forward_lanes_by_bus()
        theme_colors = {
            "light": {
                "paper": "#ffffff",
                "plot": "#f8fafc",
                "text": "#1e293b",
                "muted": "#64748b",
                "site": "#e2e8f0",
                "site_edge": "#475569",
                "grid": "rgba(100, 116, 139, 0.20)",
                "bus_lightness": "40%",
            },
            "dark": {
                "paper": "#0f172a",
                "plot": "#111827",
                "text": "#f8fafc",
                "muted": "#cbd5e1",
                "site": "#64748b",
                "site_edge": "#e2e8f0",
                "grid": "rgba(203, 213, 225, 0.16)",
                "bus_lightness": "62%",
            },
        }[theme]
        dash_by_type = {
            MoveType.SITE: "dot",
            MoveType.WORD: "solid",
            MoveType.ZONE: "dash",
        }

        figure = go.Figure()
        bus_trace_indices: list[int] = []
        exact_trace_data: list[_BusTraceData] = []
        cartoon_trace_data: list[_BusTraceData] = []
        for color_index, ((move_type, zone_id, bus_id), lanes) in enumerate(
            buses.items()
        ):
            # Golden-angle spacing avoids exact color repeats when an
            # architecture has more buses than a conventional fixed palette.
            bus_color = (
                f"hsl({round(color_index * 137.508) % 360}, 68%, "
                f"{theme_colors['bus_lightness']})"
            )
            kind = move_type.name.lower()
            bus_name = (
                f"zone bus {bus_id}"
                if zone_id is None
                else f"zone {zone_id} · {kind} bus {bus_id}"
            )
            exact_values: _BusTraceData = ([], [], [])
            cartoon_values: _BusTraceData = ([], [], [])
            for lane in lanes:
                src, dst = self.arch_spec.get_endpoints(lane)
                lane_hover_data = (
                    bus_name,
                    f"({src.zone_id}, {src.word_id}, {src.site_id})",
                    f"({dst.zone_id}, {dst.word_id}, {dst.site_id})",
                )
                for values, path in (
                    (exact_values, self.arch_spec.get_path(lane)),
                    (cartoon_values, self._cartoon_path(lane)),
                ):
                    values[0].extend(point[0] for point in path)
                    values[1].extend(point[1] for point in path)
                    values[2].extend(lane_hover_data for _ in path)
                    values[0].append(None)
                    values[1].append(None)
                    values[2].append(None)

            exact_trace_data.append(exact_values)
            cartoon_trace_data.append(cartoon_values)
            initial_values = exact_values if path_style == "exact" else cartoon_values

            bus_trace_indices.append(len(bus_trace_indices))
            figure.add_trace(
                go.Scattergl(
                    x=initial_values[0],
                    y=initial_values[1],
                    customdata=initial_values[2],
                    mode="lines+markers",
                    name=bus_name,
                    legendgroup=kind,
                    showlegend=True,
                    line={
                        "color": bus_color,
                        "dash": dash_by_type[move_type],
                        "width": 2.25,
                    },
                    marker={"size": 3},
                    opacity=0.82,
                    visible=True if show_all_buses else "legendonly",
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "source: %{customdata[1]}<br>"
                        "destination: %{customdata[2]}<br>"
                        "path point: (%{x:.3f}, %{y:.3f}) µm"
                        "<extra></extra>"
                    ),
                )
            )

        x_sites = [position[0] for _, position in locations]
        y_sites = [position[1] for _, position in locations]
        # Keep the tuple compact and put horizontally adjacent sites on opposite
        # sides of their row. Locations that share coordinates receive distinct
        # placements around the marker.
        site_labels = [
            f"({location.zone_id},{location.word_id},{location.site_id})"
            for location, _ in locations
        ]
        label_positions = ["top center"] * len(locations)
        indices_by_y: dict[float, list[int]] = defaultdict(list)
        indices_by_position: dict[tuple[float, float], list[int]] = defaultdict(list)
        for index, (_, (x, y)) in enumerate(locations):
            indices_by_y[y].append(index)
            indices_by_position[(x, y)].append(index)

        for row_indices in indices_by_y.values():
            for rank, index in enumerate(
                sorted(row_indices, key=lambda item: x_sites[item])
            ):
                label_positions[index] = (
                    "top center" if rank % 2 == 0 else "bottom center"
                )

        duplicate_positions = (
            "top left",
            "bottom right",
            "top right",
            "bottom left",
            "middle left",
            "middle right",
            "top center",
            "bottom center",
        )
        for position_indices in indices_by_position.values():
            if len(position_indices) > 1:
                for rank, index in enumerate(position_indices):
                    label_positions[index] = duplicate_positions[
                        rank % len(duplicate_positions)
                    ]
        site_customdata = [
            [location.zone_id, location.word_id, location.site_id]
            for location, _ in locations
        ]
        figure.add_trace(
            go.Scattergl(
                x=x_sites,
                y=y_sites,
                customdata=site_customdata,
                mode="markers",
                marker={
                    "color": theme_colors["site"],
                    "line": {"color": theme_colors["site_edge"], "width": 1},
                    "size": 9,
                },
                name="sites",
                showlegend=False,
                hovertemplate=(
                    "zone %{customdata[0]}<br>word %{customdata[1]}<br>"
                    "site %{customdata[2]}<br>(%{x:.3f}, %{y:.3f}) µm"
                    "<extra></extra>"
                ),
            )
        )
        label_trace_index = len(bus_trace_indices) + 1
        figure.add_trace(
            go.Scatter(
                x=x_sites,
                y=y_sites,
                text=site_labels if show_site_ids else [""] * len(site_labels),
                # A transparent marker gives Plotly room to offset the text
                # farther from the visible site marker.
                mode="markers+text",
                marker={"size": 18, "opacity": 0},
                textposition=label_positions,
                textfont={
                    "color": theme_colors["text"],
                    "family": "Menlo, Consolas, monospace",
                    "size": 9,
                },
                cliponaxis=False,
                hoverinfo="skip",
                showlegend=False,
            )
        )

        def view_ranges(
            trace_data: list[_BusTraceData],
        ) -> tuple[list[float], list[float]]:
            x_values = list(x_sites)
            y_values = list(y_sites)
            for trace_x, trace_y, _ in trace_data:
                x_values.extend(value for value in trace_x if value is not None)
                y_values.extend(value for value in trace_y if value is not None)

            def padded(values: list[float]) -> list[float]:
                if not values:
                    return [-1.0, 1.0]
                lower = min(values)
                upper = max(values)
                span = upper - lower
                padding = 0.035 * span if span else 1.0
                return [lower - padding, upper + padding]

            return padded(x_values), padded(y_values)

        exact_ranges = view_ranges(exact_trace_data)
        cartoon_ranges = view_ranges(cartoon_trace_data)
        initial_ranges = exact_ranges if path_style == "exact" else cartoon_ranges

        label_buttons = [
            {
                "label": "Labels off",
                "method": "restyle",
                "args": [{"text": [[""] * len(site_labels)]}, [label_trace_index]],
            },
            {
                "label": "Labels on",
                "method": "restyle",
                "args": [{"text": [site_labels]}, [label_trace_index]],
            },
        ]
        bus_buttons = [
            {
                "label": "Hide all buses",
                "method": "restyle",
                "args": [{"visible": "legendonly"}, bus_trace_indices],
            },
            {
                "label": "Show all buses",
                "method": "restyle",
                "args": [{"visible": True}, bus_trace_indices],
            },
        ]

        def path_button(
            label: str,
            trace_data: list[_BusTraceData],
            ranges: tuple[list[float], list[float]],
        ) -> dict[str, object]:
            return {
                "label": label,
                "method": "update",
                "args": [
                    {
                        "x": [values[0] for values in trace_data],
                        "y": [values[1] for values in trace_data],
                        "customdata": [values[2] for values in trace_data],
                    },
                    {"xaxis.range": ranges[0], "yaxis.range": ranges[1]},
                    bus_trace_indices,
                ],
            }

        path_buttons = [
            path_button("Exact paths", exact_trace_data, exact_ranges),
            path_button("Cartoon paths", cartoon_trace_data, cartoon_ranges),
        ]
        figure.update_layout(
            template="plotly_white" if theme == "light" else "plotly_dark",
            title={
                "text": "Architecture and atom transport paths",
                "x": 0.01,
                "xanchor": "left",
            },
            annotations=[
                {
                    "text": (
                        "Click bus names to select multiple paths; "
                        "double-click a name to isolate it."
                    ),
                    "showarrow": False,
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.0,
                    "y": 1.025,
                    "xanchor": "left",
                    "font": {"color": theme_colors["muted"], "size": 12},
                }
            ],
            updatemenus=[
                {
                    "type": "buttons",
                    "direction": "right",
                    "buttons": label_buttons,
                    "active": 1 if show_site_ids else 0,
                    "x": 0.0,
                    "xanchor": "left",
                    "y": 1.125,
                    "yanchor": "top",
                },
                {
                    "type": "buttons",
                    "direction": "right",
                    "buttons": bus_buttons,
                    # Legend selections can diverge from either global state,
                    # so keeping one button highlighted would be misleading.
                    "showactive": False,
                    "x": 0.42,
                    "xanchor": "left",
                    "y": 1.125,
                    "yanchor": "top",
                },
                {
                    "type": "buttons",
                    "direction": "right",
                    "buttons": path_buttons,
                    "active": 0 if path_style == "exact" else 1,
                    "x": 0.72,
                    "xanchor": "left",
                    "y": 1.125,
                    "yanchor": "top",
                },
            ],
            xaxis={
                "title": "x (µm)",
                "range": initial_ranges[0],
                "showgrid": True,
                "gridcolor": theme_colors["grid"],
                "zeroline": False,
            },
            yaxis={
                "title": "y (µm)",
                "range": initial_ranges[1],
                "scaleanchor": "x",
                "scaleratio": 1,
                "showgrid": True,
                "gridcolor": theme_colors["grid"],
                "zeroline": False,
            },
            legend={
                "title": {"text": "Buses"},
                "groupclick": "toggleitem",
                "itemclick": "toggle",
                "itemdoubleclick": "toggleothers",
                "x": 1.01,
                "xanchor": "left",
                "y": 1.0,
                "yanchor": "top",
                "font": {"size": 11},
            },
            dragmode="zoom",
            hovermode="closest",
            width=width,
            height=height,
            margin={"l": 70, "r": 255, "b": 65, "t": 145},
            font={"color": theme_colors["text"]},
            paper_bgcolor=theme_colors["paper"],
            plot_bgcolor=theme_colors["plot"],
            uirevision="arch-visualizer",
            meta={"archVisualizerBusTraceIndices": bus_trace_indices},
        )
        _install_bus_hover_html(figure)
        return figure

    def show(
        self,
        ax: Axes | None = None,
        show_words: Sequence[int] = (),
        show_intra: Sequence[int] = (),
        show_inter: Sequence[int] = (),
        **scatter_kwargs,
    ) -> None:
        """Render and immediately call ``plt.show()``.

        Convenience for interactive sessions; programmatic callers
        should prefer :meth:`plot`.
        """
        import matplotlib.pyplot as plt  # type: ignore[import-untyped]

        self.plot(
            ax,
            show_words=show_words,
            show_site_bus=show_intra,
            show_word_bus=show_inter,
            **scatter_kwargs,
        )
        plt.show()
