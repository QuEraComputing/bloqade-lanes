from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from typing import Sequence

import numpy as np

from bloqade.lanes.layout.encoding import (
    Direction,
    EncodingType,
    LaneAddress,
    LocationAddress,
    MoveType,
    SiteLaneAddress,
    WordLaneAddress,
    ZoneAddress,
)

from .word import Word


@dataclass(frozen=True)
class Bus:
    """A group of word-buses that can be executed in parallel.

    For word-buses, src and dst are the word indices involved in the word-bus.
    For site-buses, src are the source site indices and dst are the destination site indices.

    """

    src: tuple[int, ...]
    dst: tuple[int, ...]


@dataclass(frozen=True)
class ArchSpec:
    words: tuple[Word, ...]
    """tuple of all words in the architecture. words[i] gives the word at word address i."""
    zones: tuple[tuple[int, ...], ...]
    """A tuple of zones where a zone is a tuple of word addresses and zone[i] gives the ith zone."""
    measurement_mode_zones: tuple[int, ...]
    """Map from from contiguous mode value to zone id for measurement mode operations."""
    entangling_zones: frozenset[int]
    """Set of zone ids that support CZ gates."""
    has_site_buses: frozenset[int]
    """Set of words that have site-bus moves."""
    has_word_buses: frozenset[int]
    """Set of sites (by index) that have word-bus moves. These sites are the same across all words."""
    site_buses: tuple[Bus, ...]
    """List of all site buses in the architecture by site address."""
    word_buses: tuple[Bus, ...]
    """List of all word buses in the architecture by word address."""
    encoding: EncodingType = field(init=False)
    """Mapping from location addresses to zone addresses and indices within the zone."""
    zone_address_map: dict[LocationAddress, dict[ZoneAddress, int]] = field(
        init=False, default_factory=dict
    )
    paths: dict[LaneAddress, tuple[tuple[float, float], ...]] = field(
        default_factory=dict, hash=False, compare=False
    )
    """Optional precomputed paths for lanes in the architecture."""
    _lane_map: dict[tuple[LocationAddress, LocationAddress], LaneAddress] = field(
        init=False, default_factory=dict, compare=False, hash=False
    )
    """Map of site-site tuples to the lane that addresses the move between that pair of sites (None if no lane exists). Note that direction is factored in."""

    def __post_init__(self):
        if self.zones[0] != tuple(range(len(self.words))):
            raise ValueError("Zone 0 must include all words in the architecture")

        if len(self.measurement_mode_zones) == 0:
            raise ValueError("There must be at least one measurement mode zone")

        if self.measurement_mode_zones[0] != 0:
            raise ValueError("Measurement mode zone 0 must be zone 0")

        if any(
            zone_id < 0 or zone_id >= len(self.zones)
            for zone_id in self.entangling_zones
        ):
            raise ValueError("Entangling zone ids must be valid zone ids")

        if any(
            zone_id < 0 or zone_id >= len(self.zones)
            for zone_id in self.measurement_mode_zones
        ):
            raise ValueError("Measurement mode zone ids must be valid zone ids")

        zone_address_map = defaultdict(dict)
        for zone_id, zone in enumerate(self.zones):
            index = 0
            for word_id in zone:
                word = self.words[word_id]
                for site_id, _ in enumerate(word.site_indices):
                    loc_addr = LocationAddress(word_id, site_id)
                    zone_address = ZoneAddress(zone_id)
                    zone_address_map[loc_addr][zone_address] = index
                    index += 1
        object.__setattr__(self, "zone_address_map", dict(zone_address_map))
        object.__setattr__(self, "encoding", EncodingType.infer(self))  # type: ignore

        lane_map: dict[tuple[LocationAddress, LocationAddress], LaneAddress] = {}
        for word_id in self.has_site_buses:
            for bus_id, bus in enumerate(self.site_buses):
                for i in range(len(bus.src)):
                    for direction in (Direction.FORWARD, Direction.BACKWARD):
                        lane_addr = SiteLaneAddress(
                            word_id=word_id,
                            site_id=bus.src[i],
                            bus_id=bus_id,
                            direction=direction,
                        )
                        src, dst = self.get_endpoints(lane_addr)
                        lane_map[(src, dst)] = lane_addr
        for bus_id, bus in enumerate(self.word_buses):
            for site_id in self.has_word_buses:
                for word_id in bus.src:
                    for direction in (Direction.FORWARD, Direction.BACKWARD):
                        lane_addr = WordLaneAddress(
                            word_id=word_id,
                            site_id=site_id,
                            bus_id=bus_id,
                            direction=direction,
                        )
                        src, dst = self.get_endpoints(lane_addr)
                        lane_map[(src, dst)] = lane_addr
        object.__setattr__(self, "_lane_map", lane_map)

    @property
    def max_qubits(self) -> int:
        """Get the maximum number of qubits supported by this architecture."""
        num_sites_per_word = len(self.words[0].site_indices)
        return len(self.words) * num_sites_per_word // 2

    def yield_zone_locations(self, zone_address: ZoneAddress):
        """Yield all location addresses in a given zone address."""
        zone_id = zone_address.zone_id
        zone = self.zones[zone_id]
        for word_id in zone:
            word = self.words[word_id]
            for site_id, _ in enumerate(word.site_indices):
                yield LocationAddress(word_id, site_id)

    def get_path(
        self,
        lane_address: LaneAddress,
    ) -> tuple[tuple[float, float], ...]:
        if (path := self.paths.get(lane_address)) is None:
            src, dst = self.get_endpoints(lane_address)
            return (self.get_position(src), self.get_position(dst))
        return path

    def get_zone_index(
        self,
        loc_addr: LocationAddress,
        zone_id: ZoneAddress,
    ) -> int | None:
        """Get the index of a location address within a zone address."""
        return self.zone_address_map[loc_addr].get(zone_id)

    def path_bounds(self) -> tuple[float, float, float, float]:
        x_min, x_max = self.x_bounds
        y_min, y_max = self.y_bounds

        x_values = set(x for path in self.paths.values() for x, _ in path)
        y_values = set(y for path in self.paths.values() for _, y in path)

        y_min = min(y_min, min(y_values, default=y_min))
        y_max = max(y_max, max(y_values, default=y_max))

        x_min = min(x_min, min(x_values, default=x_min))
        x_max = max(x_max, max(x_values, default=x_max))
        return (x_min, x_max, y_min, y_max)

    @cached_property
    def x_bounds(self) -> tuple[float, float]:
        x_min = float("inf")
        x_max = float("-inf")
        for word in self.words:
            for x_pos, _ in word.all_positions():
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
            for _, y_pos in word.all_positions():
                y_min = min(y_min, y_pos)
                y_max = max(y_max, y_pos)

        if y_min == float("inf"):
            y_min = -1.0

        if y_max == float("-inf"):
            y_max = 1.0

        return y_min, y_max

    def get_position(self, location: LocationAddress) -> tuple[float, float]:
        return self.words[location.word_id].site_position(location.site_id)

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

                    x_start, y_start = start.position()
                    x_end, y_end = end.position()

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
                    (x_start, y_start), (x_end, y_end) = (
                        start.position(),
                        end.position(),
                    )
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

    def compatible_lane_error(self, lane1: LaneAddress, lane2: LaneAddress) -> set[str]:
        """Get the error message if two lanes are not compatible, or None if they are.

        Args:
            lane1: The first lane address.
            lane2: The second lane address.
        Returns:
            set[str]: A set of error messages indicating why the lanes are not compatible.

        NOTE: this function assumes that both lanes are valid.

        """
        errors = set()
        if lane1.direction != lane2.direction:
            errors.add("Lanes have different directions")

        if lane1.move_type == MoveType.SITE and lane2.move_type == MoveType.SITE:
            if lane1.bus_id != lane2.bus_id:
                errors.add("Lanes are on different site-buses")
            if lane1.word_id == lane2.word_id and lane1.site_id == lane2.site_id:
                errors.add("Lanes are the same")
        elif lane1.move_type == MoveType.WORD and lane2.move_type == MoveType.WORD:
            if lane2.bus_id != lane1.bus_id:
                errors.add("Lanes are on different word-buses")
            if lane1.word_id == lane2.word_id and lane1.site_id == lane2.site_id:
                errors.add("Lanes are the same")
        else:
            errors.add("Lanes have different move types")

        return errors

    def compatible_lanes(self, lane1: LaneAddress, lane2: LaneAddress) -> bool:
        """Check if two lanes are compatible (can be executed in parallel)."""
        return len(self.compatible_lane_error(lane1, lane2)) == 0

    def validate_location(self, location_address: LocationAddress) -> set[str]:
        """Check if a location address is valid in this architecture."""
        errors = set()

        num_words = len(self.words)
        if location_address.word_id < 0 or location_address.word_id >= num_words:
            errors.add(
                f"Word id {location_address.word_id} out of range of {num_words}"
            )
            return errors

        word = self.words[location_address.word_id]

        num_sites = len(word.site_indices)
        if location_address.site_id < 0 or location_address.site_id >= num_sites:
            errors.add(
                f"Site id {location_address.site_id} out of range of {num_sites}"
            )

        return errors

    def get_lane_address(
        self, src: LocationAddress, dst: LocationAddress
    ) -> LaneAddress | None:
        return self._lane_map.get((src, dst))

    def validate_lane(self, lane_address: LaneAddress) -> set[str]:
        """Check if a lane address is valid in this architecture."""
        errors = self.validate_location(lane_address.src_site())

        if lane_address.move_type is MoveType.WORD:
            if lane_address.site_id not in self.has_word_buses:
                errors.add(
                    f"Site {lane_address.site_id} does not support word-bus moves"
                )
            num_word_buses = len(self.word_buses)
            if lane_address.bus_id < 0 or lane_address.bus_id >= num_word_buses:
                errors.add(
                    f"Bus id {lane_address.bus_id} out of range of {num_word_buses}"
                )
                return errors

            bus = self.word_buses[lane_address.bus_id]
            if lane_address.word_id not in bus.src:
                errors.add(f"Word {lane_address.word_id} not in bus source {bus.src}")

        elif lane_address.move_type is MoveType.SITE:
            if lane_address.word_id not in self.has_site_buses:
                errors.add(
                    f"Word {lane_address.word_id} does not support site-bus moves"
                )

            num_site_buses = len(self.site_buses)
            if lane_address.bus_id < 0 or lane_address.bus_id >= num_site_buses:
                errors.add(
                    f"Bus id {lane_address.bus_id} out of range of {num_site_buses}"
                )
                return errors

            bus = self.site_buses[lane_address.bus_id]
            if lane_address.site_id not in bus.src:
                errors.add(f"Site {lane_address.site_id} not in bus source {bus.src}")
        else:
            errors.add(
                f"Unsupported move type {lane_address.move_type} for lane address"
            )

        return errors

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

    def get_blockaded_location(
        self, location: LocationAddress
    ) -> LocationAddress | None:
        """Get the blockaded location (CZ pair) for a given location.

        Args:
            location: The location address to find the blockaded location for.

        Returns:
            The LocationAddress of the blockaded location if one exists, None otherwise.
        """
        return self.words[location.word_id][location.site_id].cz_pair
