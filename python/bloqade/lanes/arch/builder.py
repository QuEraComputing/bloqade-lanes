"""Architecture builder: assembles an ArchSpec from an ArchBlueprint.

Takes a high-level blueprint (zones + layout) and inter-zone connections,
generates all words, buses, and zones, and produces a validated ArchSpec.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from bloqade.lanes.layout.arch import ArchSpec

if TYPE_CHECKING:
    from bloqade.lanes.layout.arch import Bus

from .topology import InterZoneTopology
from .word_factory import WordGrid, create_zone_words
from .zone import ArchBlueprint


@dataclass(frozen=True)
class ArchResult:
    """Result of build_arch(), containing the ArchSpec and metadata."""

    arch: ArchSpec
    zone_grids: dict[str, WordGrid]
    zone_indices: dict[str, int]


def build_arch(
    blueprint: ArchBlueprint,
    connections: dict[tuple[str, str], InterZoneTopology] | None = None,
) -> ArchResult:
    """Build an ArchSpec from a blueprint and inter-zone connections.

    Args:
        blueprint: Architecture blueprint with zones and layout.
        connections: Inter-zone connectivity. Keys are (zone_a, zone_b) name
            pairs, values are InterZoneTopology instances.

    Returns:
        ArchResult with the validated ArchSpec and metadata.
    """
    connections = connections or {}
    layout = blueprint.layout

    # 1. Create word grids for each zone, stacked vertically
    zone_grids: dict[str, WordGrid] = {}
    word_id_offset = 0
    zone_y = 0.0
    word_height = (layout.sites_per_word - 1) * layout.site_spacing
    row_step = word_height + layout.row_spacing

    for zone_name, zone_spec in blueprint.zones.items():
        grid = create_zone_words(
            zone_spec, layout,
            y_offset=zone_y,
            word_id_offset=word_id_offset,
        )
        zone_grids[zone_name] = grid
        word_id_offset += zone_spec.num_words
        zone_height = zone_spec.num_rows * row_step - layout.row_spacing
        zone_y += zone_height + layout.zone_gap

    all_words = tuple(
        word for grid in zone_grids.values() for word in grid.words
    )
    total_words = len(all_words)

    # 2. Generate site buses (union approach: first non-None topology)
    site_buses: tuple[Bus, ...] = ()
    site_bus_word_ids: set[int] = set()
    first_site_topology = None

    for zone_name, zone_spec in blueprint.zones.items():
        if zone_spec.site_topology is not None:
            if first_site_topology is None:
                first_site_topology = zone_spec.site_topology
            grid = zone_grids[zone_name]
            site_bus_word_ids.update(
                grid.word_id_at(r, c)
                for r in range(grid.num_rows)
                for c in range(grid.num_cols)
            )

    if first_site_topology is not None:
        site_buses = first_site_topology.generate_site_buses(layout.sites_per_word)

    # 3. Generate word buses
    all_word_buses: list[Bus] = []

    for zone_name, zone_spec in blueprint.zones.items():
        if zone_spec.word_topology is not None:
            buses = zone_spec.word_topology.generate_word_buses(zone_grids[zone_name])
            all_word_buses.extend(buses)

    for (zone_a, zone_b), topology in connections.items():
        if zone_a not in zone_grids:
            raise ValueError(f"Unknown zone '{zone_a}' in connection")
        if zone_b not in zone_grids:
            raise ValueError(f"Unknown zone '{zone_b}' in connection")
        buses = topology.generate_word_buses(zone_grids[zone_a], zone_grids[zone_b])
        all_word_buses.extend(buses)

    # 4. Build zone lists (zone 0 = all words, Rust requirement)
    all_word_ids = tuple(range(total_words))
    arch_zones: list[tuple[int, ...]] = [all_word_ids]
    zone_indices: dict[str, int] = {}

    for i, zone_name in enumerate(blueprint.zones):
        grid = zone_grids[zone_name]
        zone_word_ids = tuple(
            grid.word_id_at(r, c)
            for r in range(grid.num_rows)
            for c in range(grid.num_cols)
        )
        arch_zones.append(zone_word_ids)
        zone_indices[zone_name] = i + 1

    # 5. Entangling and measurement zones
    entangling_zones = frozenset(
        zone_indices[name]
        for name, spec in blueprint.zones.items()
        if spec.entangling
    )

    measurement_mode_zones = (0,) + tuple(
        zone_indices[name]
        for name, spec in blueprint.zones.items()
        if spec.measurement
    )

    # 6. Build ArchSpec
    arch = ArchSpec.from_components(
        words=all_words,
        zones=tuple(arch_zones),
        measurement_mode_zones=measurement_mode_zones,
        entangling_zones=entangling_zones,
        has_site_buses=frozenset(site_bus_word_ids),
        has_word_buses=frozenset(range(layout.sites_per_word)),
        site_buses=site_buses,
        word_buses=tuple(all_word_buses),
    )

    return ArchResult(
        arch=arch,
        zone_grids=zone_grids,
        zone_indices=zone_indices,
    )
