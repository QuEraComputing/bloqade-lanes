from __future__ import annotations

from collections import defaultdict
from functools import cached_property
from types import MappingProxyType
from typing import TYPE_CHECKING, Sequence

from bloqade.lanes.bytecode._native import (
    ArchSpec as _RustArchSpec,
    Bus as Bus,
    Buses as _RustBuses,
    Geometry as _RustGeometry,
    LaneAddress as _RustLaneAddress,
    TransportPath as _RustTransportPath,
    Zone as _RustZone,
)
from bloqade.lanes.layout.encoding import (
    Direction,
    LaneAddress,
    LocationAddress,
    MoveType,
    SiteLaneAddress,
    WordLaneAddress,
    ZoneAddress,
)

from .word import Word

if TYPE_CHECKING:
    from collections.abc import Iterator

    from bloqade.lanes.bytecode.exceptions import LaneGroupError, LocationGroupError


class ArchSpec:
    """Architecture specification for a quantum device."""

    _inner: _RustArchSpec
    words: tuple[Word, ...]
    paths: MappingProxyType[LaneAddress, tuple[tuple[float, float], ...]]

    def __init__(
        self,
        inner: _RustArchSpec,
        words: tuple[Word, ...],
        paths: dict[LaneAddress, tuple[tuple[float, float], ...]] | None = None,
    ):
        self._inner = inner
        self.words = words
        self.paths = MappingProxyType(paths if paths is not None else {})

        self._inner.validate()

    @cached_property
    def zone_address_map(self) -> dict[LocationAddress, dict[ZoneAddress, int]]:
        result: dict[LocationAddress, dict[ZoneAddress, int]] = defaultdict(dict)
        for zone_id, zone in enumerate(self.zones):
            index = 0
            for word_id in zone:
                word = self.words[word_id]
                for site_id, _ in enumerate(word.site_indices):
                    loc_addr = LocationAddress(word_id, site_id)
                    zone_address = ZoneAddress(zone_id)
                    result[loc_addr][zone_address] = index
                    index += 1
        return dict(result)

    @cached_property
    def _lane_map(self) -> dict[tuple[LocationAddress, LocationAddress], LaneAddress]:
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
        return lane_map

    # ── Properties derived from Rust inner ──

    @property
    def zones(self) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(z.words) for z in self._inner.zones)

    @cached_property
    def measurement_mode_zones(self) -> tuple[int, ...]:
        return tuple(self._inner.measurement_mode_zones)

    @cached_property
    def entangling_zones(self) -> frozenset[int]:
        return frozenset(self._inner.entangling_zones)

    @property
    def feed_forward(self) -> bool:
        """Whether the device supports mid-circuit measurement with classical feedback."""
        return self._inner.feed_forward

    @property
    def atom_reloading(self) -> bool:
        """Whether the device supports reloading atoms after initial fill."""
        return self._inner.atom_reloading

    @cached_property
    def has_site_buses(self) -> frozenset[int]:
        return frozenset(self._inner.words_with_site_buses)

    @cached_property
    def has_word_buses(self) -> frozenset[int]:
        return frozenset(self._inner.sites_with_word_buses)

    @cached_property
    def site_buses(self) -> tuple[Bus, ...]:
        return tuple(self._inner.buses.site_buses)

    @cached_property
    def word_buses(self) -> tuple[Bus, ...]:
        return tuple(self._inner.buses.word_buses)

    @cached_property
    def _site_bus_dst_by_src(self) -> tuple[dict[int, int], ...]:
        return tuple(
            {src: dst for src, dst in zip(bus.src, bus.dst, strict=True)}
            for bus in self.site_buses
        )

    @cached_property
    def _word_bus_dst_by_src(self) -> tuple[dict[int, int], ...]:
        return tuple(
            {src: dst for src, dst in zip(bus.src, bus.dst, strict=True)}
            for bus in self.word_buses
        )

    # ── Constructor classmethod ──

    @classmethod
    def from_components(
        cls,
        words: tuple[Word, ...],
        zones: tuple[tuple[int, ...], ...],
        measurement_mode_zones: tuple[int, ...],
        entangling_zones: frozenset[int],
        has_site_buses: frozenset[int],
        has_word_buses: frozenset[int],
        site_buses: tuple[Bus, ...],
        word_buses: tuple[Bus, ...],
        paths: dict[LaneAddress, tuple[tuple[float, float], ...]] | None = None,
        feed_forward: bool = False,
        atom_reloading: bool = False,
    ) -> ArchSpec:
        """Construct an ArchSpec from Python component types."""
        sites_per_word = len(words[0].site_indices) if words else 0
        rust_geometry = _RustGeometry(
            sites_per_word=sites_per_word,
            words=[w._inner for w in words],
        )
        rust_buses = _RustBuses(
            site_buses=list(site_buses),
            word_buses=list(word_buses),
        )
        rust_zones = [_RustZone(words=list(z)) for z in zones]

        rust_paths = None
        if paths:
            rust_paths = [
                _RustTransportPath(
                    lane=_RustLaneAddress(
                        lane.move_type,
                        lane.word_id,
                        lane.site_id,
                        lane.bus_id,
                        lane.direction,
                    ),
                    waypoints=list(waypoints),
                )
                for lane, waypoints in paths.items()
            ]

        inner = _RustArchSpec(
            version=(1, 0),
            geometry=rust_geometry,
            buses=rust_buses,
            words_with_site_buses=sorted(has_site_buses),
            sites_with_word_buses=sorted(has_word_buses),
            zones=rust_zones,
            entangling_zones=sorted(entangling_zones),
            measurement_mode_zones=list(measurement_mode_zones),
            paths=rust_paths,
            feed_forward=feed_forward,
            atom_reloading=atom_reloading,
        )
        return cls(inner, words, paths)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ArchSpec):
            return NotImplemented
        return self._inner == other._inner and self.words == other.words

    def __hash__(self) -> int:
        return hash(self._inner)

    @property
    def max_qubits(self) -> int:
        """Get the maximum number of qubits supported by this architecture."""
        num_sites_per_word = len(self.words[0].site_indices)
        return len(self.words) * num_sites_per_word // 2

    def yield_zone_locations(
        self, zone_address: ZoneAddress
    ) -> Iterator[LocationAddress]:
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

    def _get_word_bus_paths(
        self, show_word_bus: Sequence[int]
    ) -> Iterator[tuple[tuple[float, float], ...]]:
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

    def _get_site_bus_paths(
        self, show_words: Sequence[int], show_site_bus: Sequence[int]
    ) -> Iterator[tuple[tuple[float, float], ...]]:
        for word_id in show_words:
            if word_id not in self.has_site_buses:
                continue
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
        ax=None,  # type: ignore[no-untyped-def]
        show_words: Sequence[int] = (),
        show_site_bus: Sequence[int] = (),
        show_word_bus: Sequence[int] = (),
        **scatter_kwargs,  # type: ignore[no-untyped-def]
    ):  # type: ignore[no-untyped-def]
        import matplotlib.pyplot as plt  # type: ignore[import-untyped]

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
        ax=None,  # type: ignore[no-untyped-def]
        show_words: Sequence[int] = (),
        show_intra: Sequence[int] = (),
        show_inter: Sequence[int] = (),
        **scatter_kwargs,  # type: ignore[no-untyped-def]
    ):  # type: ignore[no-untyped-def]
        import matplotlib.pyplot as plt  # type: ignore[import-untyped]

        self.plot(
            ax,
            show_words=show_words,
            show_site_bus=show_intra,
            show_word_bus=show_inter,
            **scatter_kwargs,
        )
        plt.show()

    def check_location_group(
        self, locations: Sequence[LocationAddress]
    ) -> Sequence[LocationGroupError]:
        """Validate a group of location addresses via Rust.

        Returns a list of LocationGroupError exceptions (empty if all valid).
        """
        rust_addrs = [loc._inner for loc in locations]
        return self._inner.check_locations(rust_addrs)

    def check_lane_group(
        self, lanes: Sequence[LaneAddress]
    ) -> Sequence[LaneGroupError]:
        """Validate a group of lane addresses via Rust.

        Checks individual lane validity, group consistency (direction, bus_id,
        move_type), bus membership, and AOD geometry constraints.
        Returns a list of LaneGroupError exceptions (empty if all valid).
        """
        rust_addrs = [lane._inner for lane in lanes]
        return self._inner.check_lanes(rust_addrs)

    def compatible_lane_error(self, lane1: LaneAddress, lane2: LaneAddress) -> set[str]:
        """Get error messages if two lanes are not compatible.

        Delegates to Rust group validation.
        """
        errors = self.check_lane_group([lane1, lane2])
        return {str(e) for e in errors}

    def compatible_lanes(self, lane1: LaneAddress, lane2: LaneAddress) -> bool:
        """Check if two lanes are compatible (can be executed in parallel)."""
        return len(self.check_lane_group([lane1, lane2])) == 0

    def validate_location(self, location_address: LocationAddress) -> set[str]:
        """Check if a location address is valid in this architecture.

        Delegates to Rust validation.
        """
        errors = self.check_location_group([location_address])
        return {str(e) for e in errors}

    def validate_lane(self, lane_address: LaneAddress) -> set[str]:
        """Check if a lane address is valid in this architecture.

        Delegates to Rust validation.
        """
        errors = self.check_lane_group([lane_address])
        return {str(e) for e in errors}

    def get_lane_address(
        self, src: LocationAddress, dst: LocationAddress
    ) -> LaneAddress | None:
        """Given an input tuple of locations, gets the lane (w/direction)."""
        return self._lane_map.get((src, dst))

    def get_endpoints(
        self, lane_address: LaneAddress
    ) -> tuple[LocationAddress, LocationAddress]:
        src = lane_address.src_site()
        if lane_address.move_type == MoveType.WORD:
            dst_word = self._word_bus_dst_by_src[lane_address.bus_id][src.word_id]
            dst = LocationAddress(dst_word, src.site_id)
        elif lane_address.move_type == MoveType.SITE:
            dst_site = self._site_bus_dst_by_src[lane_address.bus_id][src.site_id]
            dst = LocationAddress(src.word_id, dst_site)
        else:
            raise ValueError("Unsupported lane address type")

        if lane_address.direction == Direction.FORWARD:
            return src, dst
        else:
            return dst, src

    def get_blockaded_location(
        self, location: LocationAddress
    ) -> LocationAddress | None:
        """Get the blockaded location (CZ pair) for a given location."""
        return self.words[location.word_id][location.site_id].cz_pair
