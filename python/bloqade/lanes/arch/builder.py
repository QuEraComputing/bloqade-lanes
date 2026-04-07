"""Architecture builder: assembles an ArchSpec from an ArchBlueprint.

Takes a high-level blueprint (zones + layout) and inter-zone connections,
generates all words, buses, and zones, and produces a validated ArchSpec.
"""

from __future__ import annotations

from dataclasses import dataclass

from bloqade.lanes.bytecode._native import Bus as _NativeBus
from bloqade.lanes.layout.arch import ArchSpec

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
    for zone_name, zone_spec in blueprint.zones.items():
        grid = create_zone_words(
            zone_spec,
            layout,
            y_offset=zone_y,
            word_id_offset=word_id_offset,
        )
        zone_grids[zone_name] = grid
        word_id_offset += zone_spec.num_words
        zone_height = (zone_spec.num_rows - 1) * layout.row_spacing
        zone_y += zone_height + layout.zone_gap

    all_words = tuple(word for grid in zone_grids.values() for word in grid.words)
    total_words = len(all_words)

    # 2. Generate per-zone site buses with bus.words set
    all_site_buses: list[_NativeBus] = []
    site_bus_word_ids: set[int] = set()

    for zone_name, zone_spec in blueprint.zones.items():
        if zone_spec.site_topology is not None:
            grid = zone_grids[zone_name]
            zone_word_ids = list(grid.all_word_ids)
            site_bus_word_ids.update(zone_word_ids)
            buses = zone_spec.site_topology.generate_site_buses(layout.sites_per_word)
            for bus in buses:
                all_site_buses.append(
                    _NativeBus(src=bus.src, dst=bus.dst, words=zone_word_ids)
                )

    site_buses = tuple(all_site_buses)

    # 3. Generate word buses
    all_word_buses: list[_NativeBus] = []

    for zone_name, zone_spec in blueprint.zones.items():
        if zone_spec.word_topology is not None:
            buses = zone_spec.word_topology.generate_word_buses(zone_grids[zone_name])
            all_word_buses.extend(buses)

    for (zone_a, zone_b), topology in connections.items():
        if zone_a == zone_b:
            raise ValueError(
                f"Self-connection not allowed: '{zone_a}'. "
                "Use word_topology on ZoneSpec for intra-zone connectivity."
            )
        if zone_a not in zone_grids:
            raise ValueError(f"Unknown zone '{zone_a}' in connection")
        if zone_b not in zone_grids:
            raise ValueError(f"Unknown zone '{zone_b}' in connection")
        buses = topology.generate_word_buses(zone_grids[zone_a], zone_grids[zone_b])
        all_word_buses.extend(buses)

    # 4. Build zone lists.
    # Zone 0 is always "all words" (Rust validation requirement).
    # User-defined zones are appended starting at index 1.
    all_word_ids = tuple(range(total_words))
    arch_zones: list[tuple[int, ...]] = [all_word_ids]
    zone_indices: dict[str, int] = {}

    for i, zone_name in enumerate(blueprint.zones):
        grid = zone_grids[zone_name]
        arch_zones.append(tuple(grid.all_word_ids))
        zone_indices[zone_name] = i + 1

    # 5. Entangling zones as word pairs + measurement zones
    entangling_zones: list[list[tuple[int, int]]] = [
        list(zone_grids[name].cz_pairs())
        for name, spec in blueprint.zones.items()
        if spec.entangling
    ]

    measurement_mode_zones = (0,) + tuple(
        zone_indices[name] for name, spec in blueprint.zones.items() if spec.measurement
    )

    # 6. Build ArchSpec
    arch = ArchSpec.from_components(
        words=all_words,
        zones=tuple(arch_zones),
        measurement_mode_zones=measurement_mode_zones,
        entangling_zones=entangling_zones,
        has_site_buses=frozenset(site_bus_word_ids),
        # Word buses move entire words, so all site indices are valid landing positions.
        has_word_buses=frozenset(range(layout.sites_per_word)),
        site_buses=site_buses,
        word_buses=tuple(all_word_buses),
        blockade_radius=layout.site_spacing,
    )

    return ArchResult(
        arch=arch,
        zone_grids=zone_grids,
        zone_indices=zone_indices,
    )
