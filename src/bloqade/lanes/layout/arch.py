from dataclasses import dataclass, field
from functools import cached_property
from typing import Generic, Sequence

import numpy as np

from bloqade.lanes.layout.encoding import (
    Direction,
    EncodingType,
    LaneAddress,
    LocationAddress,
    MoveType,
)

from .word import SiteType, Word


@dataclass(frozen=True)
class Bus:
    """A group of word-buses that can be executed in parallel.

    For word-buses, src and dst are the word indices involved in the word-bus.
    For site-buses, src are the source site indices and dst are the destination site indices.

    """

    src: tuple[int, ...]
    dst: tuple[int, ...]


@dataclass(frozen=True)
class ArchSpec(Generic[SiteType]):
    words: tuple[Word[SiteType], ...]
    """tuple of all words in the architecture. words[i] gives the word at word address i."""
    zones: tuple[tuple[int, ...], ...]
    """A tuple of zones where a zone is a tuple of word addresses and zone[i] gives the ith zone."""
    has_site_buses: frozenset[int]
    """Set of words that have site-bus moves."""
    has_word_buses: frozenset[int]
    """Set of sites (by index) that have word-bus moves. These sites are the same across all words."""
    site_buses: tuple[Bus, ...]
    """List of all site buses in the architecture by site address."""
    word_buses: tuple[Bus, ...]
    """List of all word buses in the architecture by word address."""
    site_bus_compatibility: tuple[frozenset[int], ...]
    """Mapping from word id indicating which other word ids can execute site-buses in parallel."""
    encoding: EncodingType = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "encoding", EncodingType.infer(self))  # type: ignore

    @property
    def max_qubits(self) -> int:
        """Get the maximum number of qubits supported by this architecture."""
        num_sites_per_word = len(self.words[0].sites)
        return len(self.words) * num_sites_per_word // 2

    @cached_property
    def x_bounds(self) -> tuple[float, float]:
        x_min = float("inf")
        x_max = float("-inf")
        for word in self.words:
            for site_id in range(len(word.sites)):
                for x_pos, _ in word.site_positions(site_id):
                    x_min = min(x_min, x_pos)
                    x_max = max(x_max, x_pos)

        if x_min == float("inf"):
            x_min = -1.0

        if x_max == float("-inf"):
            x_max = 1.0

        return x_min, x_max

    @cached_property
    def y_bounds(self) -> tuple[float, float]:
        y_min = float("inf")
        y_max = float("-inf")
        for word in self.words:
            for site_id in range(len(word.sites)):
                for _, y_pos in word.site_positions(site_id):
                    y_min = min(y_min, y_pos)
                    y_max = max(y_max, y_pos)

        if y_min == float("inf"):
            y_min = -1.0

        if y_max == float("-inf"):
            y_max = 1.0

        return y_min, y_max

    def plot(
        self,
        ax=None,
        show_words: Sequence[int] = (),
        show_site_bus: Sequence[int] = (),
        show_word_bus: Sequence[int] = (),
        **scatter_kwargs,
    ):
        import matplotlib.pyplot as plt  # type: ignore
        from scipy import interpolate as interp  # type: ignore

        if ax is None:
            ax = plt.gca()

        for word_id in show_words:
            word = self.words[word_id]
            word.plot(ax, **scatter_kwargs)

        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        bow_y = (y_max - y_min) * 0.025
        bow_x = (x_max - x_min) * 0.025

        colors = {}
        for word_id in show_words:
            word = self.words[word_id]
            for lane_id in show_site_bus:
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

        for lane in show_word_bus:
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
            show_site_bus=show_intra,
            show_word_bus=show_inter,
            **scatter_kwargs,
        )
        plt.show()

    def compatible_lane_error(
        self, lane1: LaneAddress, lane2: LaneAddress
    ) -> str | None:
        """Get the error message if two lanes are not compatible, or None if they are.

        Args:
            lane1: The first lane address.
            lane2: The second lane address.
        Returns:
            An error message if the lanes are not compatible, or None if they are.

        NOTE: this function assumes that both lanes are valid.

        """
        if lane1.direction != lane2.direction:
            return "Lanes have different directions"

        if lane1.move_type == MoveType.SITE and lane2.move_type == MoveType.SITE:
            if lane2.word_id not in self.site_bus_compatibility[lane1.word_id]:
                return "Lanes are on incompatible words for parallel site-bus moves"
            if lane1.bus_id != lane2.bus_id:
                return "Lanes are on different site-buses"
            if lane1.word_id == lane2.word_id and lane1.site_id == lane2.site_id:
                return "Lanes are the same"
        elif lane1.move_type == MoveType.WORD and lane2.move_type == MoveType.WORD:
            if lane2.bus_id != lane1.bus_id:
                return "Lanes are on different word-buses"
            if lane1.word_id == lane2.word_id and lane1.site_id == lane2.site_id:
                return "Lanes are the same"
        else:
            return "Lanes have different move types"

    def compatible_lanes(self, lane1: LaneAddress, lane2: LaneAddress) -> bool:
        """Check if two lanes are compatible (can be executed in parallel)."""
        return self.compatible_lane_error(lane1, lane2) is None

    def validate_location(self, location_address: LocationAddress) -> str | None:
        """Check if a location address is valid in this architecture."""
        num_words = len(self.words)
        if location_address.word_id < 0 or location_address.word_id >= num_words:
            return f"Word id {location_address.word_id} out of range of {num_words}"

        word = self.words[location_address.word_id]

        num_sites = len(word.sites)
        if location_address.site_id < 0 or location_address.site_id >= num_sites:
            return f"Site id {location_address.site_id} out of range of {num_sites}"

    def validate_lane(self, lane_address: LaneAddress) -> str | None:
        """Check if a lane address is valid in this architecture."""

        if (
            source_error := self.validate_location(lane_address.src_site())
        ) is not None:
            return source_error

        if lane_address.move_type is MoveType.WORD:
            if lane_address.site_id not in self.has_word_buses:
                return f"Site {lane_address.site_id} does not support word-bus moves"

            num_word_buses = len(self.word_buses)
            if lane_address.bus_id < 0 or lane_address.bus_id >= num_word_buses:
                return f"Bus id {lane_address.bus_id} out of range of {num_word_buses}"

            bus = self.word_buses[lane_address.bus_id]
            if lane_address.word_id not in bus.src:
                return f"Word {lane_address.word_id} not in bus source {bus.src}"

        elif lane_address.move_type is MoveType.SITE:
            if lane_address.word_id not in self.has_site_buses:
                return f"Word {lane_address.word_id} does not support site-bus moves"

            num_site_buses = len(self.site_buses)
            if lane_address.bus_id < 0 or lane_address.bus_id >= num_site_buses:
                return f"Bus id {lane_address.bus_id} out of range of {num_site_buses}"

            bus = self.site_buses[lane_address.bus_id]
            if lane_address.site_id not in bus.src:
                return f"Site {lane_address.site_id} not in bus source {bus.src}"
        else:
            return f"Unsupported move type {lane_address.move_type} for lane address"

    def get_endpoints(self, lane_address: LaneAddress):
        src = lane_address.src_site()
        if lane_address.move_type is MoveType.WORD:
            bus = self.word_buses[lane_address.bus_id]
            dst_word = bus.dst[bus.src.index(src.word_id)]
            dst = LocationAddress(dst_word, src.site_id)
        elif lane_address.move_type is MoveType.SITE:
            bus = self.site_buses[lane_address.bus_id]
            dst_site = bus.dst[bus.src.index(src.site_id)]
            dst = LocationAddress(src.word_id, dst_site)
        else:
            raise ValueError("Unsupported lane address type")

        if lane_address.direction is Direction.FORWARD:
            return src, dst
        else:
            return dst, src
