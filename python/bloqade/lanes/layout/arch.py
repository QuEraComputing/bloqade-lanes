from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from types import MappingProxyType
from typing import TYPE_CHECKING, Sequence

from bloqade.lanes.bytecode._native import (
    ArchSpec as _RustArchSpec,
    LaneAddress as _RustLaneAddress,
    Mode as _RustMode,
    SiteBus,
    TransportPath as _RustTransportPath,
    WordBus,
    Zone as _RustZone,
    ZoneBus,
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

    from bloqade.geometry.dialects.grid import Grid as GeoGrid

    from bloqade.lanes.bytecode.exceptions import LaneGroupError, LocationGroupError


@dataclass(frozen=True)
class BusDescriptor:
    """Descriptor for a bus within a zone."""

    bus_id: int
    move_type: MoveType
    direction: Direction
    num_lanes: int


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
        for zone_id, zone in enumerate(self._inner.zones):
            index = 0
            for word_id in range(len(self.words)):
                word = self.words[word_id]
                for site_id in range(len(word.site_indices)):
                    loc_addr = LocationAddress(word_id, site_id, zone_id)
                    zone_address = ZoneAddress(zone_id)
                    result[loc_addr][zone_address] = index
                    index += 1
        return dict(result)

    @cached_property
    def _lane_map(self) -> dict[tuple[LocationAddress, LocationAddress], LaneAddress]:
        lane_map: dict[tuple[LocationAddress, LocationAddress], LaneAddress] = {}
        for zone_id, zone in enumerate(self._inner.zones):
            for bus_id, bus in enumerate(zone.site_buses):
                bus_word_ids = zone.words_with_site_buses
                for word_id in bus_word_ids:
                    for i in range(len(bus.src)):
                        for direction in (Direction.FORWARD, Direction.BACKWARD):
                            lane_addr = SiteLaneAddress(
                                zone_id=zone_id,
                                word_id=word_id,
                                site_id=bus.src[i],
                                bus_id=bus_id,
                                direction=direction,
                            )
                            src, dst = self.get_endpoints(lane_addr)
                            lane_map[(src, dst)] = lane_addr
            for bus_id, bus in enumerate(zone.word_buses):
                for site_id in zone.sites_with_word_buses:
                    for word_id in bus.src:
                        for direction in (Direction.FORWARD, Direction.BACKWARD):
                            lane_addr = WordLaneAddress(
                                zone_id=zone_id,
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
    def zones(self) -> tuple[_RustZone, ...]:
        return tuple(self._inner.zones)

    @cached_property
    def modes(self) -> tuple[_RustMode, ...]:
        return tuple(self._inner.modes)

    @cached_property
    def _home_words(self) -> frozenset[int]:
        """Words that are 'home' (not CZ-staging) -- lower word_id in each pair."""
        home: set[int] = set()
        paired: set[int] = set()
        for w_a, w_b in self._word_partner_map.items():
            paired.add(w_a)
            paired.add(w_b)
            home.add(min(w_a, w_b))
        # Unpaired words are also home
        all_words = set(range(len(self.words)))
        home |= all_words - paired
        return frozenset(home)

    def is_home_position(self, addr: LocationAddress) -> bool:
        """True if this address is at a home (non-CZ-staging) word."""
        return addr.word_id in self._home_words

    @cached_property
    def word_zone_map(self) -> dict[int, int]:
        """Map each word_id to the zone_id it belongs to.

        Derived from each zone's entangling_pairs, word_buses, and
        words_with_site_buses. A word may only appear in one zone; if
        multiple zones claim the same word the first match wins.
        """
        mapping: dict[int, int] = {}
        for zone_id, zone in enumerate(self._inner.zones):
            for w_a, w_b in zone.entangling_pairs:
                mapping.setdefault(w_a, zone_id)
                mapping.setdefault(w_b, zone_id)
            for bus in zone.word_buses:
                for w in bus.src:
                    mapping.setdefault(w, zone_id)
                for w in bus.dst:
                    mapping.setdefault(w, zone_id)
            for w in zone.words_with_site_buses:
                mapping.setdefault(w, zone_id)
        # Any unreferenced word is (conservatively) assigned to zone 0.
        for word_id in range(len(self.words)):
            mapping.setdefault(word_id, 0)
        return mapping

    @cached_property
    def home_sites(self) -> frozenset[LocationAddress]:
        """All home LocationAddresses with correct zone_id per word.

        A home site is ``(zone_id, word_id, site_id)`` where ``word_id`` is
        a home word (lower word_id in each entangling pair, or unpaired)
        and ``zone_id`` is the zone that word belongs to.
        """
        sites: set[LocationAddress] = set()
        word_zone = self.word_zone_map
        for word_id in self._home_words:
            zone_id = word_zone[word_id]
            for site_id in range(len(self.words[word_id].site_indices)):
                sites.add(LocationAddress(word_id, site_id, zone_id))
        return frozenset(sites)

    @cached_property
    def cz_zone_addresses(self) -> frozenset[ZoneAddress]:
        """Zones that host CZ entangling operations (have entangling_pairs)."""
        return frozenset(
            ZoneAddress(zone_id)
            for zone_id, zone in enumerate(self._inner.zones)
            if zone.entangling_pairs
        )

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
        """Word IDs that have site-bus transport capability."""
        result: set[int] = set()
        for zone in self._inner.zones:
            result.update(zone.words_with_site_buses)
        return frozenset(result)

    @cached_property
    def has_word_buses(self) -> frozenset[int]:
        """Site indices that serve as word bus landing positions."""
        result: set[int] = set()
        for zone in self._inner.zones:
            result.update(zone.sites_with_word_buses)
        return frozenset(result)

    @cached_property
    def site_buses(self) -> tuple[SiteBus, ...]:
        """Aggregate all site buses across all zones.

        Note: indices in this flat list do NOT correspond to per-zone
        bus_id values in LaneAddress. Prefer iterating zones directly
        via ``self.zones[i].site_buses``.
        """
        result: list[SiteBus] = []
        for zone in self._inner.zones:
            result.extend(zone.site_buses)
        return tuple(result)

    @cached_property
    def word_buses(self) -> tuple[WordBus, ...]:
        """Aggregate all word buses across all zones.

        Note: indices in this flat list do NOT correspond to per-zone
        bus_id values in LaneAddress. Prefer iterating zones directly
        via ``self.zones[i].word_buses``.
        """
        result: list[WordBus] = []
        for zone in self._inner.zones:
            result.extend(zone.word_buses)
        return tuple(result)

    @cached_property
    def zone_buses(self) -> tuple[ZoneBus, ...]:
        return tuple(self._inner.zone_buses)

    # ── Constructor classmethod ──

    @classmethod
    def from_components(
        cls,
        words: tuple[Word, ...],
        zones: tuple[_RustZone, ...],
        modes: Sequence[_RustMode],
        zone_buses: Sequence[ZoneBus] = (),
        paths: dict[LaneAddress, tuple[tuple[float, float], ...]] | None = None,
        feed_forward: bool = False,
        atom_reloading: bool = False,
    ) -> ArchSpec:
        """Construct an ArchSpec from Python component types."""

        rust_paths = None
        if paths:
            rust_paths = [
                _RustTransportPath(
                    lane=_RustLaneAddress(
                        lane.move_type,
                        lane.zone_id,
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
            version=(2, 0),
            words=[w._inner for w in words],
            zones=list(zones),
            zone_buses=list(zone_buses),
            modes=list(modes),
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
    def sites_per_word(self) -> int:
        """Get the number of sites per word."""
        return self._inner.sites_per_word

    @property
    def max_qubits(self) -> int:
        """Get the maximum number of qubits supported by this architecture."""
        num_sites_per_word = self.sites_per_word
        return len(self.words) * num_sites_per_word // 2

    def yield_zone_locations(
        self, zone_address: ZoneAddress
    ) -> Iterator[LocationAddress]:
        """Yield all location addresses in a given zone address.

        Yields all words for the given zone_id. The Python heuristic layer
        addresses qubits with a single zone_id (typically 0) for all words,
        so this must iterate over every word to find all qubits.
        """
        zone_id = zone_address.zone_id
        for word_id in range(len(self.words)):
            word = self.words[word_id]
            for site_id in range(len(word.site_indices)):
                yield LocationAddress(word_id, site_id, zone_id)

    def _zone_word_ids(self, zone_id: int) -> list[int]:
        """Get the word IDs that belong to a specific Rust zone.

        Derives this from the zone's words_with_site_buses and
        sites_with_word_buses. If both are empty, falls back to
        looking at the modes bitstring_order.
        """
        zone = self._inner.zones[zone_id]
        # Use words_with_site_buses as the canonical word list for the zone
        word_ids = list(zone.words_with_site_buses)
        if word_ids:
            return word_ids
        # Fallback: derive from sites_with_word_buses via word_buses
        # If the zone has word buses, collect src words
        for bus in zone.word_buses:
            word_ids.extend(bus.src)
            word_ids.extend(bus.dst)
        if word_ids:
            return sorted(set(word_ids))
        # Final fallback: all words (for zones with no buses at all)
        return list(range(len(self.words)))

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
        for zone_id in range(len(self.zones)):
            for word_id in range(len(self.words)):
                for site_id in range(len(self.words[word_id].site_indices)):
                    pos = self.get_position(LocationAddress(word_id, site_id, zone_id))
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
        y_min = float("inf")
        y_max = float("-inf")
        for zone_id in range(len(self.zones)):
            for word_id in range(len(self.words)):
                for site_id in range(len(self.words[word_id].site_indices)):
                    pos = self.get_position(LocationAddress(word_id, site_id, zone_id))
                    if pos is not None:
                        y_min = min(y_min, pos[1])
                        y_max = max(y_max, pos[1])

        if y_min == float("inf"):
            y_min = -1.0

        if y_max == float("-inf"):
            y_max = 1.0

        return y_min, y_max

    def get_position(self, location: LocationAddress) -> tuple[float, float]:
        pos = self._inner.location_position(location._inner)
        if pos is None:
            raise ValueError(f"Invalid location address: {location!r}")
        return pos

    # ── Zone-addressed APIs (#419/#420) ──

    def get_zone_grid(self, zone_id: int) -> GeoGrid:
        """Get the coordinate grid for a zone as a ``bloqade.geometry.Grid``.

        Args:
            zone_id: Zone index.

        Returns:
            A ``bloqade.geometry.dialects.grid.Grid`` with the zone's
            x and y positions.

        Raises:
            ValueError: If zone_id is out of range.
        """
        from bloqade.geometry.dialects.grid import Grid as GeoGrid

        if zone_id < 0 or zone_id >= len(self._inner.zones):
            raise ValueError(
                f"zone_id {zone_id} out of range [0, {len(self._inner.zones)})"
            )
        zone = self._inner.zones[zone_id]
        return GeoGrid.from_positions(
            tuple(zone.grid.x_positions), tuple(zone.grid.y_positions)
        )

    def get_all_sites(self) -> list[tuple[float, float]]:
        """Get all site positions across all zones in canonical order.

        Returns positions in zone-major order, with each zone flattened
        in column-major grid order (``x`` outer, ``y`` inner).
        Each position is an ``(x, y)`` tuple.
        """
        sites: list[tuple[float, float]] = []
        for zone in self._inner.zones:
            for x in zone.grid.x_positions:
                for y in zone.grid.y_positions:
                    sites.append((x, y))
        return sites

    def get_available_buses(self, zone_id: int) -> list[BusDescriptor]:
        """Enumerate all valid bus descriptors for a zone.

        Args:
            zone_id: Zone index.

        Returns:
            List of ``BusDescriptor`` for each (bus_id, move_type, direction)
            combination that has at least one lane in this zone.

        Raises:
            ValueError: If zone_id is out of range.
        """
        if zone_id < 0 or zone_id >= len(self._inner.zones):
            raise ValueError(
                f"zone_id {zone_id} out of range [0, {len(self._inner.zones)})"
            )
        zone = self._inner.zones[zone_id]
        result: list[BusDescriptor] = []

        for bus_id, bus in enumerate(zone.site_buses):
            n = len(bus.src) * len(zone.words_with_site_buses)
            for direction in (Direction.FORWARD, Direction.BACKWARD):
                result.append(
                    BusDescriptor(
                        bus_id=bus_id,
                        move_type=MoveType.SITE,
                        direction=direction,
                        num_lanes=n,
                    )
                )

        for bus_id, bus in enumerate(zone.word_buses):
            n = len(bus.src) * len(zone.sites_with_word_buses)
            for direction in (Direction.FORWARD, Direction.BACKWARD):
                result.append(
                    BusDescriptor(
                        bus_id=bus_id,
                        move_type=MoveType.WORD,
                        direction=direction,
                        num_lanes=n,
                    )
                )

        return result

    def get_grid_endpoints(
        self,
        zone_id: int,
        bus_id: int,
        move_type: MoveType,
        direction: Direction,
    ) -> tuple[GeoGrid, GeoGrid]:
        """Get start and end grids for a bus move at full occupancy.

        Returns two ``bloqade.geometry.Grid`` objects representing the
        source and destination positions for all lanes in the specified
        bus group.

        Args:
            zone_id: Zone index.
            bus_id: Bus index within the zone.
            move_type: SITE or WORD.
            direction: FORWARD or BACKWARD.

        Returns:
            ``(src_grid, dst_grid)`` where each grid contains the physical
            positions of all source/destination sites for this bus.

        Raises:
            ValueError: If zone_id or bus_id is out of range, or
                move_type is not SITE or WORD.
        """
        from bloqade.geometry.dialects.grid import Grid as GeoGrid

        if zone_id < 0 or zone_id >= len(self._inner.zones):
            raise ValueError(
                f"zone_id {zone_id} out of range [0, {len(self._inner.zones)})"
            )
        zone = self._inner.zones[zone_id]

        src_positions: list[tuple[float, float]] = []
        dst_positions: list[tuple[float, float]] = []

        if move_type == MoveType.SITE:
            if bus_id < 0 or bus_id >= len(zone.site_buses):
                raise ValueError(
                    f"site bus_id {bus_id} out of range [0, {len(zone.site_buses)})"
                )
            bus = zone.site_buses[bus_id]
            for word_id in zone.words_with_site_buses:
                for src_site, dst_site in zip(bus.src, bus.dst):
                    lane = LaneAddress(
                        move_type, word_id, src_site, bus_id, direction, zone_id
                    )
                    endpoints = self._inner.lane_endpoints(lane._inner)
                    if endpoints is not None:
                        src_loc, dst_loc = endpoints
                        src_pos = self._inner.location_position(src_loc)
                        dst_pos = self._inner.location_position(dst_loc)
                        if src_pos is not None and dst_pos is not None:
                            src_positions.append(src_pos)
                            dst_positions.append(dst_pos)
        elif move_type == MoveType.WORD:
            if bus_id < 0 or bus_id >= len(zone.word_buses):
                raise ValueError(
                    f"word bus_id {bus_id} out of range [0, {len(zone.word_buses)})"
                )
            bus = zone.word_buses[bus_id]
            for src_word, dst_word in zip(bus.src, bus.dst):
                for site_id in zone.sites_with_word_buses:
                    lane = LaneAddress(
                        move_type, src_word, site_id, bus_id, direction, zone_id
                    )
                    endpoints = self._inner.lane_endpoints(lane._inner)
                    if endpoints is not None:
                        src_loc, dst_loc = endpoints
                        src_pos = self._inner.location_position(src_loc)
                        dst_pos = self._inner.location_position(dst_loc)
                        if src_pos is not None and dst_pos is not None:
                            src_positions.append(src_pos)
                            dst_positions.append(dst_pos)
        else:
            raise ValueError(f"Unsupported move_type: {move_type}")

        src_xs = sorted(set(p[0] for p in src_positions))
        src_ys = sorted(set(p[1] for p in src_positions))
        dst_xs = sorted(set(p[0] for p in dst_positions))
        dst_ys = sorted(set(p[1] for p in dst_positions))

        return (
            GeoGrid.from_positions(tuple(src_xs), tuple(src_ys)),
            GeoGrid.from_positions(tuple(dst_xs), tuple(dst_ys)),
        )

    def _get_word_bus_paths(
        self, show_word_bus: Sequence[int]
    ) -> Iterator[tuple[tuple[float, float], ...]]:
        for zone_id, zone in enumerate(self._inner.zones):
            for lane_id in show_word_bus:
                if lane_id >= len(zone.word_buses):
                    continue
                lane = zone.word_buses[lane_id]
                for site_id in zone.sites_with_word_buses:
                    for start_word_id, end_word_id in zip(lane.src, lane.dst):
                        lane_addr = WordLaneAddress(
                            zone_id=zone_id,
                            word_id=start_word_id,
                            site_id=site_id,
                            bus_id=lane_id,
                            direction=Direction.FORWARD,
                        )
                        yield self.get_path(lane_addr)

    def _get_site_bus_paths(
        self, show_words: Sequence[int], show_site_bus: Sequence[int]
    ) -> Iterator[tuple[tuple[float, float], ...]]:
        for zone_id, zone in enumerate(self._inner.zones):
            for word_id in show_words:
                if word_id not in set(zone.words_with_site_buses):
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
            # Plot sites using their positions from the arch spec.
            # Try each zone to find valid positions for this word.
            positions = []
            for zone_id in range(len(self.zones)):
                for site_id in range(len(word.site_indices)):
                    pos = self.get_position(LocationAddress(word_id, site_id, zone_id))
                    if pos is not None:
                        positions.append(pos)
                if positions:
                    break
            if positions:
                x_positions = [p[0] for p in positions]
                y_positions = [p[1] for p in positions]
                ax.scatter(x_positions, y_positions, **scatter_kwargs)

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
        result = self._inner.lane_endpoints(lane_address._inner)
        if result is None:
            raise ValueError(f"Invalid lane address: {lane_address!r}")
        rust_src, rust_dst = result
        src = LocationAddress(rust_src.word_id, rust_src.site_id, rust_src.zone_id)
        dst = LocationAddress(rust_dst.word_id, rust_dst.site_id, rust_dst.zone_id)
        return src, dst

    def get_cz_partner(self, location: LocationAddress) -> LocationAddress | None:
        """Get the CZ partner for a given location.

        Uses Rust-side get_cz_partner which resolves via the zone's
        entangling_pairs.
        """
        result = self._inner.get_cz_partner(location._inner)
        if result is None:
            return None
        return LocationAddress(result.word_id, result.site_id, result.zone_id)

    def get_blockaded_location(
        self, location: LocationAddress
    ) -> LocationAddress | None:
        """Get the CZ partner for a given location using word-level pairing.

        Maps to the partner word, preserving the input zone_id. This is used
        by the Python heuristics layer where CZ partners are resolved within
        the same zone coordinate frame (word buses connect words within a zone).
        """
        partner_word = self._word_partner_map.get(location.word_id)
        if partner_word is None:
            return None
        return LocationAddress(partner_word, location.site_id, location.zone_id)

    @cached_property
    def _word_partner_map(self) -> dict[int, int]:
        """Map word_id -> partner_word_id from each zone's entangling_pairs.

        Iterates entangling_pairs on each zone and builds a bidirectional
        word partner mapping.
        """
        partner_map: dict[int, int] = {}
        for zone in self._inner.zones:
            for w_a, w_b in zone.entangling_pairs:
                partner_map[w_a] = w_b
                partner_map[w_b] = w_a
        return partner_map
