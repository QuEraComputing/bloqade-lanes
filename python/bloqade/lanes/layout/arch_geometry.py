"""Geometry-level query helpers for ArchSpec.

This module provides :class:`ArchSpecGeometry`, a helper class for downstream
consumers who want geometry-level queries (grids, flat site lists, bus
descriptors) without walking Rust zone objects directly.

The class accepts either a Python :class:`~bloqade.lanes.layout.arch.ArchSpec`
wrapper or a raw Rust ``_RustArchSpec`` object, so it can be used at either
layer of the stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from bloqade.lanes.bytecode._native import ArchSpec as _RustArchSpec
from bloqade.lanes.layout.encoding import Direction, LaneAddress, MoveType

if TYPE_CHECKING:
    from bloqade.geometry.dialects.grid import Grid as GeoGrid

    from bloqade.lanes.layout.arch import ArchSpec


@dataclass(frozen=True)
class BusDescriptor:
    """Descriptor for a bus within a zone."""

    bus_id: int
    move_type: MoveType
    direction: Direction
    num_lanes: int


class ArchSpecGeometry:
    """Geometry-level query helper wrapping an ArchSpec.

    Provides methods for retrieving coordinate grids, flat site lists, and bus
    descriptors from an architecture specification. Intended for downstream
    consumers who need geometry-level data without walking Rust zone objects
    directly.

    Accepts either a Python :class:`~bloqade.lanes.layout.arch.ArchSpec` or a
    raw Rust ``_RustArchSpec`` object, so it can be constructed from either
    layer of the stack.
    """

    def __init__(self, arch_spec: "ArchSpec | _RustArchSpec") -> None:
        if isinstance(arch_spec, _RustArchSpec):
            self._inner = arch_spec
        else:
            self._inner = arch_spec._inner

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
                for src_site, _dst_site in zip(bus.src, bus.dst):
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
            for src_word, _dst_word in zip(bus.src, bus.dst):
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
