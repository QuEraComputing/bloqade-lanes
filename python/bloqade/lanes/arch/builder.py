"""Architecture builder: assembles an ArchSpec from an ArchBlueprint.

Takes a high-level blueprint (zones + layout) and inter-zone connections,
generates all words, buses, and zones, and produces a validated ArchSpec.

Internally uses ``ZoneBuilder`` + ``ArchBuilder`` for correct-by-construction
zone assembly. One blueprint zone = one Rust zone (no splitting).
"""

from __future__ import annotations

from dataclasses import dataclass

from bloqade.lanes.bytecode._native import Grid as _RustGrid
from bloqade.lanes.layout.arch import ArchSpec

from .arch_builder import ArchBuilder, ZoneBuilder
from .topology import InterZoneTopology
from .word_factory import WordGrid, create_zone_words
from .zone import ArchBlueprint, DeviceLayout, ZoneSpec


@dataclass(frozen=True)
class ArchResult:
    """Result of build_arch(), containing the ArchSpec and metadata."""

    arch: ArchSpec
    zone_grids: dict[str, WordGrid]
    zone_indices: dict[str, int]


def _build_zone_grid(
    zone_spec: ZoneSpec,
    layout: DeviceLayout,
    n: int,
    s: float,
) -> _RustGrid:
    """Build a Rust Grid covering all columns and rows in a zone.

    The grid must have enough x-positions to cover all interleaved CZ pairs
    and enough y-positions for all rows.
    """
    num_rows = zone_spec.num_rows
    num_cols = zone_spec.num_cols
    pair_width = (2 * n - 1) * s

    x_positions: list[float] = []
    num_pairs = num_cols // 2
    for pair_idx in range(num_pairs):
        pair_x = pair_idx * (pair_width + layout.pair_spacing)
        for i in range(n):
            x_positions.append(pair_x + 2.0 * s * i)
            x_positions.append(pair_x + s + 2.0 * s * i)

    x_pos_sorted = sorted(set(x_positions))
    y_positions = [row * layout.row_spacing for row in range(num_rows)]
    if not y_positions:
        y_positions = [0.0]

    return _RustGrid.from_positions(x_pos_sorted, y_positions)


def build_arch(
    blueprint: ArchBlueprint,
    connections: dict[tuple[str, str], InterZoneTopology] | None = None,
) -> ArchResult:
    """Build an ArchSpec from a blueprint and inter-zone connections.

    One blueprint zone maps to one Rust zone. Entangling pairs are
    metadata on the zone, not a reason to split into sub-zones.

    Args:
        blueprint: Architecture blueprint with zones and layout.
        connections: Inter-zone connectivity. Keys are (zone_a, zone_b) name
            pairs, values are InterZoneTopology instances.

    Returns:
        ArchResult with the validated ArchSpec and metadata.
    """
    connections = connections or {}
    layout = blueprint.layout
    n = layout.sites_per_word
    s = layout.site_spacing

    # Validate connections reference valid zones.
    for zone_a, zone_b in connections:
        if zone_a == zone_b:
            raise ValueError(
                f"Self-connection not allowed: '{zone_a}'. "
                "Use word_topology on ZoneSpec for intra-zone connectivity."
            )
        if zone_a not in blueprint.zones:
            raise ValueError(f"Unknown zone '{zone_a}' in connection")
        if zone_b not in blueprint.zones:
            raise ValueError(f"Unknown zone '{zone_b}' in connection")

    # 1. Create word grids (preserves row/col structure for topology generators).
    zone_grids: dict[str, WordGrid] = {}
    word_id_offset = 0
    for zone_name, zone_spec in blueprint.zones.items():
        grid = create_zone_words(
            zone_spec,
            layout,
            word_id_offset=word_id_offset,
        )
        zone_grids[zone_name] = grid
        word_id_offset += zone_spec.num_words

    # 2. Build ZoneBuilders from blueprint zones.
    zone_builders: dict[str, ZoneBuilder] = {}

    for zone_name, zone_spec in blueprint.zones.items():
        word_grid = zone_grids[zone_name]
        rust_grid = _build_zone_grid(zone_spec, layout, n, s)
        word_shape = _word_shape_from_layout(zone_spec, layout)

        zone = ZoneBuilder(zone_name, rust_grid, word_shape)

        # Place words on the grid using the same index pattern as create_zone_words.
        for row in range(zone_spec.num_rows):
            for col in range(zone_spec.num_cols):
                word = word_grid.word_at(row, col)
                # Extract x and y indices from the word's site positions.
                x_indices = sorted({site[0] for site in word.site_indices})
                y_indices = sorted({site[1] for site in word.site_indices})
                zone.add_word(x_indices, y_indices)

        # Site buses from topology.
        if zone_spec.site_topology is not None:
            for bus in zone_spec.site_topology.generate_site_buses(
                layout.sites_per_word
            ):
                zone.add_site_bus(list(bus.src), list(bus.dst))

        # Intra-zone word buses from topology.
        if zone_spec.word_topology is not None:
            for bus in zone_spec.word_topology.generate_word_buses(word_grid):
                # Topology generators use global word IDs; convert to zone-local.
                offset = word_grid.word_id_offset
                zone.add_word_bus(
                    src=[w - offset for w in bus.src],
                    dst=[w - offset for w in bus.dst],
                )

        # Entangling pairs.
        if zone_spec.entangling:
            offset = word_grid.word_id_offset
            pairs = list(word_grid.cz_pairs())
            zone.add_entangling_pairs(
                [a - offset for a, _ in pairs],
                [b - offset for _, b in pairs],
            )

        zone_builders[zone_name] = zone

    # 3. Compose zones into ArchBuilder.
    arch_builder = ArchBuilder()
    zone_indices: dict[str, int] = {}

    for zone_name, zone in zone_builders.items():
        zid = arch_builder.add_zone(zone)
        zone_indices[zone_name] = zid

    # 4. Inter-zone connections → zone_buses.
    for (zone_a_name, zone_b_name), topology in connections.items():
        grid_a = zone_grids[zone_a_name]
        grid_b = zone_grids[zone_b_name]
        offset_a = grid_a.word_id_offset
        offset_b = grid_b.word_id_offset

        for bus in topology.generate_word_buses(grid_a, grid_b):
            # Convert global word IDs to zone-local for connect().
            src_local = [w - offset_a for w in bus.src]
            dst_local = [w - offset_b for w in bus.dst]
            arch_builder.connect(
                src=(zone_a_name, src_local),
                dst=(zone_b_name, dst_local),
            )

    # 5. Modes.
    all_zone_names = list(blueprint.zones.keys())
    arch_builder.add_mode("all", all_zone_names)

    for name, spec in blueprint.zones.items():
        if spec.measurement:
            arch_builder.add_mode(name, [name])

    # 6. Build and return.
    arch = arch_builder.build()

    return ArchResult(
        arch=arch,
        zone_grids=zone_grids,
        zone_indices=zone_indices,
    )


def _word_shape_from_layout(
    zone_spec: ZoneSpec, layout: DeviceLayout
) -> tuple[int, int]:
    """Derive word_shape from zone spec and layout.

    For interleaved CZ pairs, each word occupies ``sites_per_word`` x-positions
    and 1 y-position (row). This assumes 1D words; 2D word shapes would require
    extending ``ZoneSpec`` with a word shape parameter.
    """
    _ = zone_spec  # reserved for future 2D word shapes
    n = layout.sites_per_word
    return (n, 1)
