import math
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from typing import ClassVar, Sequence

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
    _lane_duration_cache_us: dict[tuple[LaneAddress, float], float] = field(
        init=False, default_factory=dict, compare=False, hash=False
    )
    _max_lane_duration_cache_us: dict[float, float] = field(
        init=False, default_factory=dict, compare=False, hash=False
    )

    _FLAIR_MAX_RAMP_US: ClassVar[float] = 0.2
    _FLAIR_MAX_JERK_UM_PER_US3: ClassVar[float] = 0.0004
    _FLAIR_MAX_ACCEL_UM_PER_US2: ClassVar[float] = 0.0015

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
        super().__setattr__("_lane_map", lane_map)

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

    def _path_segment_distances_um(
        self, path: tuple[tuple[float, float], ...]
    ) -> tuple[float, ...]:
        if len(path) <= 1:
            return ()
        return tuple(
            max(abs(x1 - x0), abs(y1 - y0))
            for (x0, y0), (x1, y1) in zip(path, path[1:])
        )

    def _const_jerk_min_duration_us(self, max_dist_um: float) -> float:
        max_dist_um = abs(max_dist_um)
        if max_dist_um < 1e-8:
            return 0.0

        t1 = self._FLAIR_MAX_ACCEL_UM_PER_US2 / self._FLAIR_MAX_JERK_UM_PER_US3
        a = self._FLAIR_MAX_JERK_UM_PER_US3 * t1
        b = 3 * self._FLAIR_MAX_JERK_UM_PER_US3 * t1**2
        c = 2 * self._FLAIR_MAX_JERK_UM_PER_US3 * t1**3 - max_dist_um
        if c >= 0:
            t1_jerk = (max_dist_um / (2 * self._FLAIR_MAX_JERK_UM_PER_US3)) ** (1 / 3)
            return 4 * t1_jerk

        discriminant = b**2 - 4 * a * c
        t2 = (-b + math.sqrt(discriminant)) / (2 * a)
        return 4 * t1 + 2 * t2

    def get_lane_duration_us(
        self, lane_address: LaneAddress, *, amplitude_delta: float = 1.0
    ) -> float:
        """Return lane execution duration in microseconds."""
        normalized_amp = abs(float(amplitude_delta))
        cache_key = (lane_address, normalized_amp)
        if (duration_us := self._lane_duration_cache_us.get(cache_key)) is not None:
            return duration_us

        segment_distances = self._path_segment_distances_um(self.get_path(lane_address))
        ramp_time_us = normalized_amp / self._FLAIR_MAX_RAMP_US
        duration_us = (
            ramp_time_us
            + sum(self._const_jerk_min_duration_us(dist) for dist in segment_distances)
            + ramp_time_us
        )
        self._lane_duration_cache_us[cache_key] = duration_us
        return duration_us

    def _iter_lane_addresses(self) -> tuple[LaneAddress, ...]:
        return tuple(self._lane_map.values())

    def _max_lane_duration_us(self, *, amplitude_delta: float = 1.0) -> float:
        normalized_amp = abs(float(amplitude_delta))
        if (
            max_duration_us := self._max_lane_duration_cache_us.get(normalized_amp)
        ) is not None:
            return max_duration_us

        lane_addresses = self._iter_lane_addresses()
        if len(lane_addresses) == 0:
            max_duration_us = 0.0
        else:
            max_duration_us = max(
                self.get_lane_duration_us(lane, amplitude_delta=normalized_amp)
                for lane in lane_addresses
            )
        self._max_lane_duration_cache_us[normalized_amp] = max_duration_us
        return max_duration_us

    def get_lane_duration_cost(
        self, lane_address: LaneAddress, *, amplitude_delta: float = 1.0
    ) -> float:
        """Return normalized lane duration cost in [0, 1].

        This API standardizes lane costs by scaling each lane's duration to the
        architecture-local maximum duration using:
            cost = lane_duration_us / max_lane_duration_us
        """
        max_duration_us = self._max_lane_duration_us(amplitude_delta=amplitude_delta)
        if max_duration_us <= 0.0:
            return 0.0
        lane_duration_us = self.get_lane_duration_us(
            lane_address, amplitude_delta=amplitude_delta
        )
        return min(1.0, max(0.0, lane_duration_us / max_duration_us))

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

    def _get_word_bus_paths(self, show_word_bus):
        for lane_id in show_word_bus:
            lane = self.word_buses[lane_id]
            for site_id in self.has_word_buses:
                for start_word_id, end_word_id in zip(lane.src, lane.dst):
                    lane_addr = WordLaneAddress(
                        word_id=start_word_id,
                        site_id=site_id,
                        bus_id=lane_id,
                        direction=Direction.FORWARD,
                    )
                    yield self.get_path(lane_addr)

    def _get_site_bus_paths(self, show_words, show_site_bus):
        for word_id in show_words:
            if word_id not in self.has_site_buses:
                continue  # Only show lanes for words in has_site_buses
            for lane_id in show_site_bus:
                lane = self.site_buses[lane_id]
                for i in range(len(lane.src)):
                    lane_addr = SiteLaneAddress(
                        word_id=word_id,
                        site_id=lane.src[i],
                        bus_id=lane_id,
                        direction=Direction.FORWARD,
                    )
                    yield self.get_path(lane_addr)

    def plot(
        self,
        ax=None,
        show_words: Sequence[int] = (),
        show_site_bus: Sequence[int] = (),
        show_word_bus: Sequence[int] = (),
        **scatter_kwargs,
    ):
        import matplotlib.pyplot as plt  # type: ignore

        if ax is None:
            ax = plt.gca()

        for word_id in show_words:
            word = self.words[word_id]
            word.plot(ax, **scatter_kwargs)

        site_paths = self._get_site_bus_paths(show_words, show_site_bus)
        for path in site_paths:
            x_vals, y_vals = zip(*path)
            ax.plot(x_vals, y_vals, linestyle="--")

        word_paths = self._get_word_bus_paths(show_word_bus)
        for path in word_paths:
            x_vals, y_vals = zip(*path)
            ax.plot(x_vals, y_vals, linestyle="-")
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
        """Given an input tuple of locations, gets the lane (w/direction)."""
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
