"""Architecture builder: assembles an ArchSpec from an ArchBlueprint.

Takes a high-level blueprint (zones + layout) and inter-zone connections,
generates all words, buses, and zones, and produces a validated ArchSpec.
"""

from __future__ import annotations

from dataclasses import dataclass

from bloqade.lanes.bytecode._native import (
    Grid as _RustGrid,
    LocationAddress as _RustLocAddr,
    Mode as _RustMode,
    Zone as _RustZone,
    ZoneBus as _RustZoneBus,
)
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


def _build_zone_grid(
    zone_spec,
    layout,
    n,
    s,
):
    """Build a Rust Grid covering all columns and rows in a zone.

    The grid must have enough x-positions to cover all interleaved CZ pairs
    and enough y-positions for all rows.
    """
    num_rows = zone_spec.num_rows
    num_cols = zone_spec.num_cols
    pair_width = (2 * n - 1) * s

    # Build x positions for ALL columns (interleaved pattern)
    x_positions = []
    num_pairs = num_cols // 2
    for pair_idx in range(num_pairs):
        pair_x = pair_idx * (pair_width + layout.pair_spacing)
        for i in range(n):
            # Even col site
            x_positions.append(pair_x + 2.0 * s * i)
            # Odd col site
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

    For entangling ZoneSpecs (with paired word columns), the builder splits
    the ZoneSpec into two Rust zones (even columns and odd columns) and adds
    an entangling_zone_pair between them.

    Args:
        blueprint: Architecture blueprint with zones and layout.
        connections: Inter-zone connectivity. Keys are (zone_a, zone_b) name
            pairs, values are InterZoneTopology instances.

    Returns:
        ArchResult with the validated ArchSpec and metadata.
    """
    connections = connections or {}
    layout = blueprint.layout

    # 1. Create word grids for each zone
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

    all_words = tuple(word for grid in zone_grids.values() for word in grid.words)
    n = layout.sites_per_word
    s = layout.site_spacing

    # 2. Build Rust Zone objects.
    # For entangling zones: split into even-col and odd-col sub-zones.
    zone_indices: dict[str, int] = {}
    rust_zones: list[_RustZone] = []

    # Track which Rust zone indices correspond to even/odd columns
    # for building word buses between them
    zone_even_odd_map: dict[str, tuple[int, int]] = {}

    for zone_name, zone_spec in blueprint.zones.items():
        grid = zone_grids[zone_name]

        # Build per-zone site buses
        site_buses = []
        if zone_spec.site_topology is not None:
            site_buses = list(
                zone_spec.site_topology.generate_site_buses(layout.sites_per_word)
            )

        # Build per-zone word buses (intra-zone only)
        word_buses = []
        if zone_spec.word_topology is not None:
            word_buses = list(zone_spec.word_topology.generate_word_buses(grid))

        # Also add inter-zone word buses from connections
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
            if zone_name == zone_a or zone_name == zone_b:
                buses = topology.generate_word_buses(
                    zone_grids[zone_a], zone_grids[zone_b]
                )
                word_buses.extend(buses)

        all_zone_word_ids = list(grid.all_word_ids)

        if zone_spec.entangling:
            # Split into even-col and odd-col sub-zones
            even_word_ids = [
                grid.word_id_at(r, c)
                for r in range(zone_spec.num_rows)
                for c in range(0, zone_spec.num_cols, 2)
            ]
            odd_word_ids = [
                grid.word_id_at(r, c)
                for r in range(zone_spec.num_rows)
                for c in range(1, zone_spec.num_cols, 2)
            ]

            # Build entangling pairs: pair even-col words with odd-col words
            # at the same row position.
            entangling_pairs = list(zip(even_word_ids, odd_word_ids))

            # Both sub-zones share the same full grid (all x/y positions)
            rust_grid = _build_zone_grid(zone_spec, layout, n, s)
            rust_grid_even = rust_grid
            rust_grid_odd = rust_grid

            even_zone_id = len(rust_zones)
            rust_zones.append(
                _RustZone(
                    grid=rust_grid_even,
                    site_buses=site_buses,
                    word_buses=word_buses,
                    words_with_site_buses=even_word_ids if site_buses else [],
                    sites_with_word_buses=(
                        list(range(layout.sites_per_word)) if word_buses else []
                    ),
                    entangling_pairs=entangling_pairs,
                )
            )

            odd_zone_id = len(rust_zones)
            rust_zones.append(
                _RustZone(
                    grid=rust_grid_odd,
                    site_buses=site_buses,
                    word_buses=word_buses,
                    words_with_site_buses=odd_word_ids if site_buses else [],
                    sites_with_word_buses=(
                        list(range(layout.sites_per_word)) if word_buses else []
                    ),
                    entangling_pairs=entangling_pairs,
                )
            )

            zone_even_odd_map[zone_name] = (even_zone_id, odd_zone_id)
            zone_indices[zone_name] = even_zone_id
        else:
            # Non-entangling zone: single Rust zone
            rust_grid = _build_zone_grid(zone_spec, layout, n, s)
            zone_id = len(rust_zones)
            rust_zones.append(
                _RustZone(
                    grid=rust_grid,
                    site_buses=site_buses,
                    word_buses=word_buses,
                    words_with_site_buses=all_zone_word_ids if site_buses else [],
                    sites_with_word_buses=(
                        list(range(layout.sites_per_word)) if word_buses else []
                    ),
                )
            )
            zone_indices[zone_name] = zone_id

    # 3. Build a mapping from Rust zone index → valid word IDs.
    # For entangling zones, even sub-zone gets even-col words and odd sub-zone
    # gets odd-col words.  Non-entangling zones get all their words.
    rust_zone_word_ids: dict[int, list[int]] = {}
    for zone_name, zone_spec in blueprint.zones.items():
        grid = zone_grids[zone_name]
        if zone_name in zone_even_odd_map:
            even_id, odd_id = zone_even_odd_map[zone_name]
            rust_zone_word_ids[even_id] = [
                grid.word_id_at(r, c)
                for r in range(zone_spec.num_rows)
                for c in range(0, zone_spec.num_cols, 2)
            ]
            rust_zone_word_ids[odd_id] = [
                grid.word_id_at(r, c)
                for r in range(zone_spec.num_rows)
                for c in range(1, zone_spec.num_cols, 2)
            ]
        else:
            rust_zone_word_ids[zone_indices[zone_name]] = list(grid.all_word_ids)

    # Modes (measurement modes)
    all_zone_ids = list(range(len(rust_zones)))
    all_bitstring_order: list[_RustLocAddr] = []
    for zone_id in all_zone_ids:
        for word_id in rust_zone_word_ids[zone_id]:
            for site_id in range(layout.sites_per_word):
                all_bitstring_order.append(_RustLocAddr(zone_id, word_id, site_id))
    modes: list[_RustMode] = [
        _RustMode(name="all", zones=all_zone_ids, bitstring_order=all_bitstring_order)
    ]

    # Per-zone measurement modes
    for name, spec in blueprint.zones.items():
        if spec.measurement:
            if name in zone_even_odd_map:
                even_id, odd_id = zone_even_odd_map[name]
                zone_ids = [even_id, odd_id]
            else:
                zone_ids = [zone_indices[name]]
            bitstring_order: list[_RustLocAddr] = []
            for z_id in zone_ids:
                for word_id in rust_zone_word_ids[z_id]:
                    for site_id in range(layout.sites_per_word):
                        bitstring_order.append(_RustLocAddr(z_id, word_id, site_id))
            modes.append(
                _RustMode(name=name, zones=zone_ids, bitstring_order=bitstring_order)
            )

    # 4. Zone buses (inter-zone)
    zone_buses: list[_RustZoneBus] = []
    for (zone_a_name, zone_b_name), topology in connections.items():
        z_a = zone_indices[zone_a_name]
        z_b = zone_indices[zone_b_name]
        grid_a = zone_grids[zone_a_name]
        grid_b = zone_grids[zone_b_name]
        buses = topology.generate_word_buses(grid_a, grid_b)
        for bus in buses:
            src_pairs = [(z_a, w) for w in bus.src]
            dst_pairs = [(z_b, w) for w in bus.dst]
            zone_buses.append(_RustZoneBus(src=src_pairs, dst=dst_pairs))

    # 5. Build ArchSpec
    arch = ArchSpec.from_components(
        words=all_words,
        zones=tuple(rust_zones),
        modes=modes,
        zone_buses=zone_buses,
    )

    return ArchResult(
        arch=arch,
        zone_grids=zone_grids,
        zone_indices=zone_indices,
    )
