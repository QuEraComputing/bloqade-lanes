"""Visualization helpers for :class:`~bloqade.lanes.layout.ArchSpec`.

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

from functools import cached_property
from typing import TYPE_CHECKING, Sequence

from bloqade.lanes.bytecode.encoding import (
    Direction,
    LocationAddress,
    SiteLaneAddress,
    WordLaneAddress,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from matplotlib.axes import Axes

    from bloqade.lanes.arch.spec import ArchSpec


__all__ = [
    "ArchVisualizer",
]


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
