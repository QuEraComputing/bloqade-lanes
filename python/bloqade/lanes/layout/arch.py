from __future__ import annotations

from functools import cached_property
from types import MappingProxyType
from typing import TYPE_CHECKING, Sequence

from bloqade.lanes._wrapper import RustWrapper
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
    ZoneAddress,
)

from .word import Word

if TYPE_CHECKING:
    from collections.abc import Iterator

    from bloqade.lanes.bytecode.exceptions import LaneGroupError, LocationGroupError
    from bloqade.lanes.visualize.arch import ArchVisualizer


class ArchSpec(RustWrapper[_RustArchSpec]):
    """Architecture specification for a quantum device."""

    def __init__(self, inner: _RustArchSpec):
        self._inner = inner
        self._inner.validate()

    @cached_property
    def words(self) -> tuple[Word, ...]:
        """Python Word wrappers, derived from the Rust ArchSpec."""
        return tuple(Word.from_inner(w) for w in self._inner.words)

    @cached_property
    def paths(self) -> MappingProxyType[LaneAddress, tuple[tuple[float, float], ...]]:
        """Transport path waypoints keyed by LaneAddress.

        Derived from the Rust ``ArchSpec.paths`` on first access.
        """
        raw = self._inner.paths
        if raw is None:
            return MappingProxyType({})
        return MappingProxyType(
            {LaneAddress.from_inner(p.lane): tuple(p.waypoints) for p in raw}
        )

    def iter_all_lanes(self) -> Iterator[LaneAddress]:
        """Yield every valid lane address in the architecture.

        Enumerates site-bus, word-bus, and zone-bus lanes in both forward
        and backward directions. Used by
        :class:`~bloqade.lanes.layout.MoveMetricCalculator` to compute
        max-duration bounds. Prefer ``get_lane_address(src, dst)`` for
        single-pair lookups.
        """
        sites_per_word = self.sites_per_word

        # Intra-zone: site buses and word buses.
        for zone_id, zone in enumerate(self._inner.zones):
            for bus_id, bus in enumerate(zone.site_buses):
                for word_id in zone.words_with_site_buses:
                    for i in range(len(bus.src)):
                        for direction in (Direction.FORWARD, Direction.BACKWARD):
                            yield LaneAddress(
                                MoveType.SITE,
                                word_id,
                                bus.src[i],
                                bus_id,
                                direction,
                                zone_id,
                            )
            for bus_id, bus in enumerate(zone.word_buses):
                for site_id in zone.sites_with_word_buses:
                    for word_id in bus.src:
                        for direction in (Direction.FORWARD, Direction.BACKWARD):
                            yield LaneAddress(
                                MoveType.WORD,
                                word_id,
                                site_id,
                                bus_id,
                                direction,
                                zone_id,
                            )

        # Inter-zone: zone buses.
        for bus_id, zb in enumerate(self._inner.zone_buses):
            for (src_zone, src_word), (_dst_zone, _dst_word) in zip(zb.src, zb.dst):
                for site_id in range(sites_per_word):
                    for direction in (Direction.FORWARD, Direction.BACKWARD):
                        yield LaneAddress(
                            MoveType.ZONE,
                            src_word,
                            site_id,
                            bus_id,
                            direction,
                            src_zone,
                        )

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
        return frozenset(self._inner.left_cz_word_ids())

    def is_home_position(self, addr: LocationAddress) -> bool:
        """True if this address is at a home (non-CZ-staging) word."""
        return addr.word_id in self._home_words

    @cached_property
    def word_zone_map(self) -> dict[int, int]:
        """Map each word_id to the zone_id it belongs to.

        Delegates to Rust ``ArchSpec.word_zone_map()``.
        """
        return self._inner.word_zone_map()

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

    @property
    def blockade_radius(self) -> float | None:
        """Rydberg blockade radius (µm), or ``None`` if not provided.

        This is metadata — when present, it indicates the radius
        associated with the architecture and is typically used to
        interpret the entangling pairs.  It is **not** independently
        verified at the ArchSpec level; use
        :meth:`ZoneBuilder.set_blockade_radius` /
        :meth:`ArchBuilder.set_blockade_radius` if you want the pair
        list to be derived from and checked against a radius.
        """
        return self._inner.blockade_radius

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
        blockade_radius: float | None = None,
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
            blockade_radius=blockade_radius,
        )
        return cls(inner)

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

    def get_zone_index(
        self,
        loc_addr: LocationAddress,
        zone_id: ZoneAddress,
    ) -> int | None:
        """O(1) flat index of a location within a zone.

        Delegates to Rust ``ArchSpec.zone_location_index()``.
        """
        return self._inner.zone_location_index(loc_addr._inner, zone_id.zone_id)

    def get_path(
        self,
        lane_address: LaneAddress,
    ) -> tuple[tuple[float, float], ...]:
        if (path := self.paths.get(lane_address)) is None:
            src, dst = self.get_endpoints(lane_address)
            return (self.get_position(src), self.get_position(dst))
        return path

    # ── Visualization shims ────────────────────────────────────────
    # The real implementations live in ``bloqade.lanes.visualize.arch``
    # via :class:`ArchVisualizer`. These shims preserve the historical
    # ``arch_spec.<method>()`` call sites.  A single deferred-import
    # helper builds the visualizer and caches bounds on it, so repeated
    # access to ``x_bounds``/``y_bounds`` avoids recomputation.

    @cached_property
    def _visualizer(self) -> ArchVisualizer:
        from bloqade.lanes.visualize.arch import ArchVisualizer

        return ArchVisualizer(self)

    def path_bounds(self) -> tuple[float, float, float, float]:
        return self._visualizer.path_bounds()

    @property
    def x_bounds(self) -> tuple[float, float]:
        return self._visualizer.x_bounds

    @property
    def y_bounds(self) -> tuple[float, float]:
        return self._visualizer.y_bounds

    def get_position(self, location: LocationAddress) -> tuple[float, float]:
        pos = self._inner.location_position(location._inner)
        if pos is None:
            raise ValueError(f"Invalid location address: {location!r}")
        return pos

    def plot(
        self,
        ax=None,  # type: ignore[no-untyped-def]
        show_words: Sequence[int] = (),
        show_site_bus: Sequence[int] = (),
        show_word_bus: Sequence[int] = (),
        **scatter_kwargs,  # type: ignore[no-untyped-def]
    ):  # type: ignore[no-untyped-def]
        return self._visualizer.plot(
            ax,
            show_words=show_words,
            show_site_bus=show_site_bus,
            show_word_bus=show_word_bus,
            **scatter_kwargs,
        )

    def show(
        self,
        ax=None,  # type: ignore[no-untyped-def]
        show_words: Sequence[int] = (),
        show_intra: Sequence[int] = (),
        show_inter: Sequence[int] = (),
        **scatter_kwargs,  # type: ignore[no-untyped-def]
    ):  # type: ignore[no-untyped-def]
        self._visualizer.show(
            ax,
            show_words=show_words,
            show_intra=show_intra,
            show_inter=show_inter,
            **scatter_kwargs,
        )

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

    def get_lane_address(
        self, src: LocationAddress, dst: LocationAddress
    ) -> LaneAddress | None:
        """Given an input tuple of locations, gets the lane (w/direction).

        Delegates to Rust ``ArchSpec.lane_for_endpoints()``.
        """
        result = self._inner.lane_for_endpoints(src._inner, dst._inner)
        if result is None:
            return None
        return LaneAddress.from_inner(result)

    def get_endpoints(
        self, lane_address: LaneAddress
    ) -> tuple[LocationAddress, LocationAddress]:
        result = self._inner.lane_endpoints(lane_address._inner)
        if result is None:
            raise ValueError(f"Invalid lane address: {lane_address!r}")
        rust_src, rust_dst = result
        return (
            LocationAddress.from_inner(rust_src),
            LocationAddress.from_inner(rust_dst),
        )

    def get_cz_partner(self, location: LocationAddress) -> LocationAddress | None:
        """Get the CZ partner for a given location.

        Uses Rust-side get_cz_partner which resolves via the zone's
        entangling_pairs.
        """
        result = self._inner.get_cz_partner(location._inner)
        if result is None:
            return None
        return LocationAddress.from_inner(result)
