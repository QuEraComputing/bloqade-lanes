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
    from plotly.graph_objects import Figure  # type: ignore[reportMissingImports]

    from bloqade.lanes.arch.spec import ArchSpec


__all__ = [
    "ArchVisualizer",
]

_BusTraceData = tuple[
    list[float | None],
    list[float | None],
    list[tuple[str, str, str, int] | None],
]

_BUS_HOVER_POST_SCRIPT = """
(function () {
  const plot = document.getElementById('{plot_id}');
  if (!plot || plot.__archVisualizerHoverInstalled) return;
  plot.__archVisualizerHoverInstalled = true;

  const busIndices = (plot.layout.meta || {}).archVisualizerBusTraceIndices || [];
  const busIndexSet = new Set(busIndices);
  const siteTraceIndex = (plot.layout.meta || {}).archVisualizerSiteTraceIndex;
  const siteLanePaths = (plot.layout.meta || {}).archVisualizerSiteLanePaths || [];
  const siteLanePathRefs =
    (plot.layout.meta || {}).archVisualizerSiteLanePathRefs || {};
  const busControls = (plot.layout.meta || {}).archVisualizerBusControls || [];
  let highlightPath = null;
  let siteLaneOverlays = [];
  let activeSiteKey = null;

  function clearHighlight() {
    if (highlightPath !== null) highlightPath.remove();
    highlightPath = null;
  }

  function clearSiteLaneOverlays() {
    siteLaneOverlays.forEach((element) => element.remove());
    siteLaneOverlays = [];
  }

  function coordinatePath(xValues, yValues, reverse) {
    const xAxis = plot._fullLayout.xaxis;
    const yAxis = plot._fullLayout.yaxis;
    if (!xAxis || !yAxis) return {pathData: '', pixels: []};

    const pixels = [];
    const start = reverse ? xValues.length - 1 : 0;
    const stop = reverse ? -1 : xValues.length;
    const step = reverse ? -1 : 1;
    let pathData = '';
    for (let index = start; index !== stop; index += step) {
      const x = xValues[index];
      const y = yValues[index];
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      const pixel = [
        xAxis._offset + xAxis.l2p(x),
        yAxis._offset + yAxis.l2p(y)
      ];
      pathData += `${pixels.length ? 'L' : 'M'}${pixel[0]},${pixel[1]}`;
      pixels.push(pixel);
    }
    return {pathData, pixels};
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

  function drawSiteLaneOverlays(customdata) {
    clearSiteLaneOverlays();
    if (!customdata || customdata.length < 3) return;

    const key = `${customdata[0]},${customdata[1]},${customdata[2]}`;
    const refs = siteLanePathRefs[key] || [];
    const hoverLayer = plot.querySelector('.hoverlayer');
    if (!hoverLayer) return;

    refs.forEach(([pathIndex, reverse]) => {
      const lanePath = siteLanePaths[pathIndex];
      if (!lanePath) return;
      const {pathData, pixels} = coordinatePath(lanePath.x, lanePath.y, reverse);
      if (!pathData || pixels.length < 2) return;

      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      path.setAttribute('d', pathData);
      path.setAttribute('data-arch-visualizer-site-lane', '');
      path.setAttribute('fill', 'none');
      path.setAttribute('stroke', lanePath.color);
      path.setAttribute('stroke-width', '4');
      path.setAttribute('stroke-opacity', '0.42');
      path.setAttribute('stroke-linecap', 'round');
      path.setAttribute('stroke-linejoin', 'round');
      path.setAttribute('pointer-events', 'none');
      hoverLayer.insertBefore(path, hoverLayer.firstChild);
      siteLaneOverlays.push(path);
    });
  }

  function siteCustomdataAt(x, y) {
    const siteTrace = plot.data[siteTraceIndex];
    if (!siteTrace || !Number.isFinite(x) || !Number.isFinite(y)) return null;

    const coordinateTolerance = 1e-9;
    for (let pointIndex = 0; pointIndex < siteTrace.x.length; pointIndex++) {
      if (
        Math.abs(siteTrace.x[pointIndex] - x) <= coordinateTolerance &&
        Math.abs(siteTrace.y[pointIndex] - y) <= coordinateTolerance
      ) {
        return siteTrace.customdata[pointIndex];
      }
    }
    return null;
  }

  function siteCustomdataNearPointer(event) {
    const siteTrace = plot.data[siteTraceIndex];
    const xAxis = plot._fullLayout.xaxis;
    const yAxis = plot._fullLayout.yaxis;
    if (!siteTrace || !xAxis || !yAxis) return null;

    const plotRect = plot.getBoundingClientRect();
    const pointerX = event.clientX - plotRect.left;
    const pointerY = event.clientY - plotRect.top;
    const hoverRadiusSquared = 14 * 14;
    let nearestCustomdata = null;
    let nearestDistanceSquared = hoverRadiusSquared;

    for (let pointIndex = 0; pointIndex < siteTrace.x.length; pointIndex++) {
      const x = siteTrace.x[pointIndex];
      const y = siteTrace.y[pointIndex];
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;

      const siteX = xAxis._offset + xAxis.l2p(x);
      const siteY = yAxis._offset + yAxis.l2p(y);
      const dx = siteX - pointerX;
      const dy = siteY - pointerY;
      const distanceSquared = dx * dx + dy * dy;
      if (distanceSquared <= nearestDistanceSquared) {
        nearestDistanceSquared = distanceSquared;
        nearestCustomdata = siteTrace.customdata[pointIndex];
      }
    }
    return nearestCustomdata;
  }

  function activateSiteLaneOverlays(customdata) {
    const key = customdata
      ? `${customdata[0]},${customdata[1]},${customdata[2]}`
      : null;
    const overlaysAreConnected = siteLaneOverlays.some(
      (element) => element.isConnected
    );
    if (key === activeSiteKey && (key === null || overlaysAreConnected)) return;

    activeSiteKey = key;
    if (customdata) {
      drawSiteLaneOverlays(customdata);
    } else {
      clearSiteLaneOverlays();
    }
  }

  function installBusMultiselectors() {
    if (plot.querySelector('[data-arch-visualizer-bus-selectors]')) return;

    plot.style.position = 'relative';
    const panel = document.createElement('div');
    panel.setAttribute('data-arch-visualizer-bus-selectors', '');
    panel.style.position = 'absolute';
    panel.style.top = '92px';
    panel.style.right = '12px';
    panel.style.width = '236px';
    panel.style.zIndex = '1001';
    panel.style.color = plot.layout.font.color;
    panel.style.fontFamily = plot.layout.font.family || 'Arial, sans-serif';
    panel.style.fontSize = '12px';

    const groups = [
      ['site', 'Site Buses'],
      ['word', 'Word Buses'],
      ['zone', 'Zone Buses']
    ];
    const checkboxes = new Map();

    groups.forEach(([kind, heading]) => {
      const controls = busControls.filter((control) => control.kind === kind);
      const details = document.createElement('details');
      details.style.marginBottom = '8px';

      const summary = document.createElement('summary');
      summary.textContent = heading;
      summary.style.cursor = 'pointer';
      summary.style.padding = '7px 9px';
      summary.style.background = plot.layout.paper_bgcolor;
      summary.style.border = '1px solid rgba(100, 116, 139, 0.45)';
      summary.style.borderRadius = '4px';
      summary.style.userSelect = 'none';
      details.appendChild(summary);

      const options = document.createElement('div');
      options.style.maxHeight = '235px';
      options.style.overflowY = 'auto';
      options.style.padding = '6px 7px';
      options.style.background = plot.layout.paper_bgcolor;
      options.style.border = '1px solid rgba(100, 116, 139, 0.35)';
      options.style.borderTop = '0';

      if (!controls.length) {
        const empty = document.createElement('div');
        empty.textContent = `No ${heading.toLowerCase()}`;
        empty.style.opacity = '0.72';
        empty.style.padding = '4px';
        options.appendChild(empty);
      }

      controls.forEach((control) => {
        const label = document.createElement('label');
        label.style.display = 'flex';
        label.style.alignItems = 'center';
        label.style.gap = '7px';
        label.style.padding = '4px 2px';
        label.style.cursor = 'pointer';
        label.title = `${heading.slice(0, -2)} ID ${control.busId}`;
        label.addEventListener('mouseenter', () => drawHighlight(control.traceIndex));
        label.addEventListener('mouseleave', clearHighlight);

        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.checked = plot.data[control.traceIndex].visible === true;
        checkbox.setAttribute(
          'aria-label',
          `${heading.slice(0, -2)} ID ${control.busId}`
        );
        checkbox.addEventListener('change', () => {
          window.Plotly.restyle(
            plot,
            {visible: checkbox.checked ? true : 'legendonly'},
            [control.traceIndex]
          );
        });
        checkboxes.set(control.traceIndex, checkbox);

        const swatch = document.createElement('span');
        swatch.setAttribute('data-arch-visualizer-bus-color', control.color);
        swatch.style.display = 'inline-block';
        swatch.style.width = '12px';
        swatch.style.height = '12px';
        swatch.style.flex = '0 0 12px';
        swatch.style.borderRadius = '50%';
        swatch.style.background = control.color;

        const text = document.createElement('span');
        text.textContent = `ID ${control.busId} · ${control.label}`;

        label.appendChild(checkbox);
        label.appendChild(swatch);
        label.appendChild(text);
        options.appendChild(label);
      });

      details.appendChild(options);
      panel.appendChild(details);
    });

    plot.appendChild(panel);
    plot.on('plotly_restyle', function () {
      window.setTimeout(() => {
        checkboxes.forEach((checkbox, traceIndex) => {
          checkbox.checked = plot.data[traceIndex].visible === true;
        });
      }, 0);
    });
  }

  installBusMultiselectors();

  // Resolve site previews from pointer position instead of relying solely on
  // Plotly's hover winner. A visible or recently restyled WebGL bus can win
  // hover arbitration over a coincident SVG site marker, even after that bus
  // is hidden again. Direct proximity keeps site previews independent of bus
  // selection state and also lets them coexist with selected buses.
  plot.addEventListener('mousemove', function (event) {
    activateSiteLaneOverlays(siteCustomdataNearPointer(event));
  });
  plot.addEventListener('mouseleave', function () {
    activeSiteKey = null;
    clearSiteLaneOverlays();
  });

  plot.on('plotly_hover', function (event) {
    clearHighlight();
    const busPoint = event.points.find(
      (item) => busIndexSet.has(item.curveNumber)
    );
    const sitePoint = event.points.find(
      (item) => item.curveNumber === siteTraceIndex
    );
    // Once a bus is visible, Plotly may report its endpoint instead of the
    // coincident site marker. Treat an endpoint at a site coordinate as a site
    // hover so lane previews are independent of bus visibility.
    const siteCustomdata = sitePoint
      ? sitePoint.customdata
      : event.points
          .map((item) => siteCustomdataAt(item.x, item.y))
          .find((customdata) => customdata !== null);
    if (siteCustomdata) {
      activateSiteLaneOverlays(siteCustomdata);
    } else if (busPoint) {
      drawHighlight(busPoint.curveNumber);
    }
  });

  plot.on('plotly_unhover', function () {
    clearHighlight();
  });
  plot.on('plotly_relayout', function () {
    clearHighlight();
    activeSiteKey = null;
    clearSiteLaneOverlays();
  });
})();
"""


class _InteractiveArchFigureMixin:
    """Install architecture interactions in every HTML display route."""

    def to_html(
        self,
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
            cast(Any, super()).to_html(
                *args,
                post_script=scripts,
                config=config,
                **kwargs,
            ),
        )

    def _repr_mimebundle_(
        self,
        include: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
        validate: bool = True,
        **kwargs: Any,
    ) -> dict[str, str] | Any:
        if (include is not None and "text/html" not in include) or (
            exclude is not None and "text/html" in exclude
        ):
            return cast(Any, super())._repr_mimebundle_(
                include=include,
                exclude=exclude,
                validate=validate,
                **kwargs,
            )
        return {
            "text/html": self.to_html(
                full_html=False,
                include_plotlyjs="cdn",
                validate=validate,
            )
        }

    def _ipython_display_(self) -> None:
        """Render through HTML instead of Plotly's interaction-free MIME path."""
        try:
            from IPython.display import HTML, display
        except ImportError:  # pragma: no cover - only called by IPython
            cast(Any, super())._ipython_display_()
            return

        display(
            HTML(
                self.to_html(
                    full_html=False,
                    include_plotlyjs="cdn",
                )
            )
        )

    def show(self, *args: Any, **kwargs: Any) -> Any:
        """Preserve custom controls in notebooks and browser windows."""
        try:
            from IPython.core.getipython import get_ipython
        except ImportError:  # pragma: no cover - IPython is an optional dependency
            shell = None
        else:
            shell = get_ipython()

        renderer = kwargs.get("renderer")
        notebook_renderers = {
            None,
            "jupyterlab",
            "notebook",
            "notebook_connected",
            "plotly_mimetype",
        }
        if (
            shell is not None
            and shell.__class__.__name__ == "ZMQInteractiveShell"
            and not args
            and renderer in notebook_renderers
            and set(kwargs) <= {"config", "renderer", "validate"}
        ):
            from IPython.display import HTML, display

            display(
                HTML(
                    self.to_html(
                        full_html=False,
                        include_plotlyjs="cdn",
                        config=kwargs.get("config"),
                        validate=kwargs.get("validate", True),
                    )
                )
            )
            return None

        if (
            (shell is None or shell.__class__.__name__ != "ZMQInteractiveShell")
            and not args
            and renderer in {None, "browser"}
            and set(kwargs) <= {"config", "renderer", "validate"}
        ):
            # Plotly's browser renderer serializes the figure dictionary
            # directly, bypassing this class's ``to_html`` override and its
            # architecture-specific post-script. Open that HTML explicitly so
            # script usage retains the bus selectors and hover overlays.
            from plotly.io._base_renderers import open_html_in_browser

            open_html_in_browser(
                self.to_html(
                    full_html=True,
                    include_plotlyjs=True,
                    config=kwargs.get("config"),
                    validate=kwargs.get("validate", True),
                )
            )
            return None

        return cast(Any, super()).show(*args, **kwargs)


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
        show_bus_legend: bool = False,
        path_style: Literal["exact", "cartoon"] = "exact",
        theme: Literal["light", "dark"] = "light",
        width: int = 1300,
        height: int = 800,
    ) -> Figure:
        """Build an interactive architecture and atom-path visualization.

        The returned Plotly figure supports pan, box zoom, and mode-bar zoom.
        Its HTML representation also enables wheel and trackpad zoom.
        Hovering a site shows its ``(zone_id, word_id, site_id)`` and overlays
        every transport lane available from that site, using the lane's bus
        color and its exact architecture-defined path.
        Hovering a bus shows the source and destination site addresses and
        temporarily draws that bus as a thicker overlay above other buses.
        Site labels can be toggled with the controls above the plot.

        The site-, word-, and zone-bus multiselectors on the right contain a
        color-labelled checkbox for every bus. Selections accumulate without a
        long permanent bus legend. Bus traces are hidden by default to keep
        large architectures readable; pass ``show_all_buses=True`` or use the
        selectors to display them. The path-view toggle switches between exact
        architecture paths from :meth:`ArchSpec.get_path` and a less cluttered
        schematic view.
        Schematic word-bus paths are straight because they preserve a site
        column; site- and zone-bus paths are curved to expose crossings.

        Args:
            show_site_ids: Show site identity labels initially. Site identity
                remains available on hover when labels are hidden.
            show_all_buses: Show all bus paths initially. By default, buses are
                available from the right-side selectors but hidden to avoid
                visual clutter.
            show_bus_legend: Also show Plotly's legacy bus legend below the
                right-side selectors. The selectors are the default bus
                selection interface because the legend is unwieldy for large
                architectures.
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

        class InteractiveArchFigure(_InteractiveArchFigureMixin, go.Figure):
            pass

        figure = InteractiveArchFigure()
        bus_trace_indices: list[int] = []
        bus_controls: list[dict[str, object]] = []
        site_lane_paths: list[dict[str, object]] = []
        site_lane_path_refs: dict[LocationAddress, list[tuple[int, bool]]] = (
            defaultdict(list)
        )
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
                src_label = f"({src.zone_id}, {src.word_id}, {src.site_id})"
                dst_label = f"({dst.zone_id}, {dst.word_id}, {dst.site_id})"
                exact_path = self.arch_spec.get_path(lane)
                site_lane_path_index = len(site_lane_paths)
                site_lane_paths.append(
                    {
                        "x": [point[0] for point in exact_path],
                        "y": [point[1] for point in exact_path],
                        "color": bus_color,
                    }
                )
                site_lane_path_refs[src].append((site_lane_path_index, False))
                site_lane_path_refs[dst].append((site_lane_path_index, True))
                lane_hover_data = (
                    bus_name,
                    src_label,
                    dst_label,
                    bus_id,
                )
                for values, path in (
                    (exact_values, exact_path),
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

            trace_index = len(cast(Any, figure.data))
            bus_trace_indices.append(trace_index)
            bus_controls.append(
                {
                    "traceIndex": trace_index,
                    "kind": kind,
                    "busId": bus_id,
                    "label": bus_name,
                    "color": bus_color,
                }
            )
            figure.add_trace(
                go.Scattergl(
                    x=initial_values[0],
                    y=initial_values[1],
                    customdata=initial_values[2],
                    mode="lines+markers",
                    name=bus_name,
                    legendgroup=kind,
                    showlegend=show_bus_legend,
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
                        "bus ID: %{customdata[3]}<br>"
                        "source: %{customdata[1]}<br>"
                        "destination: %{customdata[2]}"
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
        site_trace_index = len(cast(Any, figure.data))
        figure.add_trace(
            go.Scatter(
                x=x_sites,
                y=y_sites,
                customdata=site_customdata,
                text=site_labels if show_site_ids else [""] * len(site_labels),
                mode="markers+text",
                marker={
                    "color": theme_colors["site"],
                    "line": {"color": theme_colors["site_edge"], "width": 1},
                    "size": 9,
                },
                textposition=label_positions,
                textfont={
                    "color": theme_colors["text"],
                    "family": "Menlo, Consolas, monospace",
                    "size": 9,
                },
                cliponaxis=False,
                name="sites",
                showlegend=False,
                hovertemplate=(
                    "<b>zone %{customdata[0]} · word %{customdata[1]} · "
                    "site %{customdata[2]}</b><br>(%{x:.3f}, %{y:.3f}) µm"
                    "<extra></extra>"
                ),
                hoverlabel={"align": "left"},
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
                "args": [{"text": [[""] * len(site_labels)]}, [site_trace_index]],
            },
            {
                "label": "Labels on",
                "method": "restyle",
                "args": [{"text": [site_labels]}, [site_trace_index]],
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
                        "Use the bus multiselectors at right to show or hide "
                        "individual paths; hover a site to preview every "
                        "available lane."
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
                "title": {"text": "Bus legend"},
                "groupclick": "toggleitem",
                "itemclick": "toggle",
                "itemdoubleclick": "toggleothers",
                "x": 1.01,
                "xanchor": "left",
                "y": 0.62,
                "yanchor": "top",
                "font": {"size": 11},
            },
            dragmode="zoom",
            hovermode="closest",
            width=width,
            height=height,
            margin={
                "l": 70,
                "r": 270,
                "b": 65,
                "t": 145,
            },
            font={"color": theme_colors["text"]},
            paper_bgcolor=theme_colors["paper"],
            plot_bgcolor=theme_colors["plot"],
            uirevision="arch-visualizer",
            meta={
                "archVisualizerBusTraceIndices": bus_trace_indices,
                "archVisualizerBusControls": bus_controls,
                "archVisualizerSiteTraceIndex": site_trace_index,
                "archVisualizerSiteLanePaths": site_lane_paths,
                "archVisualizerSiteLanePathRefs": {
                    f"{location.zone_id},{location.word_id},{location.site_id}": refs
                    for location, refs in site_lane_path_refs.items()
                },
            },
        )
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
