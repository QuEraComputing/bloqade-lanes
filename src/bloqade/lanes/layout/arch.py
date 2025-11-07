from dataclasses import dataclass, field
from typing import Generic, Sequence

import numpy as np

from bloqade.lanes.layout.encoding import EncodingType

from .word import SiteType, Word


@dataclass(frozen=True)
class Bus:
    """A group of inter-lanes that can be executed in parallel.

    For inter-lanes, src and dst are the word indices involved in the inter-lane.
    For intra-lanes, src are the source site indices and dst are the destination site indices.

    """

    src: tuple[int, ...]
    dst: tuple[int, ...]


@dataclass(frozen=True)
class ArchSpec(Generic[SiteType]):
    words: tuple[Word[SiteType], ...]
    """tuple of all words in the architecture. words[i] gives the word at word address i."""
    has_site_buses: frozenset[int]
    """Set of words that have site-lane moves."""
    has_word_buses: frozenset[int]  # set of sites in word that have inter-lanes moves
    """Set of sites (by index) that have word-lane moves. These sites are the same across all words."""
    site_buses: tuple[Bus, ...]
    """List of all site buses in the architecture by site address."""
    word_buses: tuple[Bus, ...]
    """List of all word buses in the architecture by word address."""
    site_bus_compatibility: tuple[frozenset[int], ...]
    """Mapping from word id indicating which other word ids can execute site-buses in parallel."""
    encoding: EncodingType = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "encoding", EncodingType.infer(self))  # type: ignore

    def plot(
        self,
        ax=None,
        show_words: Sequence[int] = (),
        show_intra: Sequence[int] = (),
        show_inter: Sequence[int] = (),
        **scatter_kwargs,
    ):
        import matplotlib.pyplot as plt  # type: ignore
        from scipy import interpolate as interp  # type: ignore

        if ax is None:
            ax = plt.gca()

        for word_id in show_words:
            word = self.words[word_id]
            word.plot(ax)

        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        bow_y = (y_max - y_min) * 0.025
        bow_x = (x_max - x_min) * 0.025

        colors = {}
        for word_id in show_words:
            word = self.words[word_id]
            for lane_id in show_intra:
                lane = self.site_buses[lane_id]

                for start, end in zip(lane.src, lane.dst):
                    start = word[start]
                    end = word[end]

                    for (x_start, y_start), (x_end, y_end) in zip(
                        start.positions(), end.positions()
                    ):
                        mid_x = (x_start + x_end) / 2
                        mid_y = (y_start + y_end) / 2

                        if x_start == x_end:
                            mid_x += bow_y
                        elif y_start == y_end:
                            mid_y += bow_x

                        f = interp.interp1d(
                            [x_start, mid_x, x_end],
                            [y_start, mid_y, y_end],
                            kind="quadratic",
                        )
                        x_vals = np.linspace(x_start, x_end, num=10)
                        y_vals = f(x_vals)
                        (ln,) = ax.plot(
                            x_vals, y_vals, color=colors.get(lane), linestyle="--"
                        )
                        if lane not in colors:
                            colors[lane] = ln.get_color()

        for lane in show_inter:
            lane = self.word_buses[lane]
            for start_word_id, end_word_id in zip(lane.src, lane.dst):
                start_word = self.words[start_word_id]
                end_word = self.words[end_word_id]

                for site in self.has_word_buses:
                    start = start_word[site]
                    end = end_word[site]
                    for (x_start, y_start), (x_end, y_end) in zip(
                        start.positions(), end.positions()
                    ):
                        mid_x = (x_start + x_end) / 2
                        mid_y = (y_start + y_end) / 2

                        if x_start == x_end:
                            mid_x += bow_y
                        elif y_start == y_end:
                            mid_y += bow_x

                        f = interp.interp1d(
                            [x_start, mid_x, x_end],
                            [y_start, mid_y, y_end],
                            kind="quadratic",
                        )
                        x_vals = np.linspace(x_start, x_end, num=10)
                        y_vals = f(x_vals)
                        (ln,) = ax.plot(
                            x_vals, y_vals, color=colors.get(lane), linestyle="-"
                        )
                        if lane not in colors:
                            colors[lane] = ln.get_color()

        return ax

    def show(
        self,
        ax=None,
        show_words: Sequence[int] = (),
        show_intra: Sequence[int] = (),
        show_inter: Sequence[int] = (),
        **scatter_kwargs,
    ):
        import matplotlib.pyplot as plt  # type: ignore

        self.plot(
            ax,
            show_words=show_words,
            show_intra=show_intra,
            show_inter=show_inter,
            **scatter_kwargs,
        )
        plt.show()
