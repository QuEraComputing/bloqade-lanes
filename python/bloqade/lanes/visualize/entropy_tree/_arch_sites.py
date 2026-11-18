"""The architecture backdrop of the entropy-tree viewer: its sites and extent.

The full architecture visualizer (``ArchVisualizer``) lives in
``bloqade.gemini.visualize.arch``, in the ``bloqade`` distribution, which lanes
cannot import. The viewer only needs a scatter of the sites and the area the
architecture covers, so those two pieces live here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from bloqade.lanes.bytecode.encoding import LocationAddress

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from bloqade.lanes.arch.spec import ArchSpec


def _site_positions(
    arch_spec: ArchSpec, word_ids: Sequence[int] | None = None
) -> list[tuple[float, float]]:
    """Positions of the sites of ``word_ids`` (every word when ``None``).

    A word's sites are taken from the first zone that holds it, so a word
    template shared by several zones is drawn once.
    """
    words = range(len(arch_spec.words)) if word_ids is None else word_ids
    positions: list[tuple[float, float]] = []
    for word_id in words:
        for zone_id in range(len(arch_spec.zones)):
            in_zone = [
                position
                for site_id in range(len(arch_spec.words[word_id].site_indices))
                if (
                    position := arch_spec._inner.location_position(
                        LocationAddress(word_id, site_id, zone_id)._inner
                    )
                )
                is not None
            ]
            if in_zone:
                positions.extend(in_zone)
                break
    return positions


class ArchSites:
    """Draw the sites of an architecture and report its extent."""

    def __init__(self, arch_spec: ArchSpec) -> None:
        self.arch_spec = arch_spec

    def path_bounds(self) -> tuple[float, float, float, float]:
        """``(x_min, x_max, y_min, y_max)`` covering every site and every
        transport-path waypoint.

        Without any site, the site extent defaults to ``(-1, 1)`` on each axis
        before the waypoints are added.
        """
        sites = self._all_site_positions()
        xs = [x for x, _ in sites]
        ys = [y for _, y in sites]
        x_min, x_max = (min(xs), max(xs)) if sites else (-1.0, 1.0)
        y_min, y_max = (min(ys), max(ys)) if sites else (-1.0, 1.0)
        for path in self.arch_spec.paths.values():
            for x, y in path:
                x_min = min(x_min, x)
                x_max = max(x_max, x)
                y_min = min(y_min, y)
                y_max = max(y_max, y)
        return (x_min, x_max, y_min, y_max)

    def plot(self, ax: Axes, show_words: Sequence[int] = (), **scatter_kwargs) -> Axes:
        """Scatter the sites of ``show_words`` onto ``ax``, one call per word.

        One call per word, as ``ArchVisualizer.plot`` does, keeps matplotlib's
        colour cycle where later default-coloured artists expect it.
        """
        for word_id in show_words:
            positions = _site_positions(self.arch_spec, [word_id])
            if positions:
                ax.scatter(
                    [x for x, _ in positions],
                    [y for _, y in positions],
                    **scatter_kwargs,
                )
        return ax

    def _all_site_positions(self) -> list[tuple[float, float]]:
        arch_spec = self.arch_spec
        return [
            position
            for zone_id in range(len(arch_spec.zones))
            for word_id in range(len(arch_spec.words))
            for site_id in range(len(arch_spec.words[word_id].site_indices))
            if (
                position := arch_spec._inner.location_position(
                    LocationAddress(word_id, site_id, zone_id)._inner
                )
            )
            is not None
        ]
