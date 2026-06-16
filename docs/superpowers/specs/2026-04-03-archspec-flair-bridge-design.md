# ArchSpec Bridge to Flair Standard Library

**Date:** 2026-04-03
**Builds on:** PR #398 (zone-based architecture builder)

## Problem

### Flair integration gap

Flair is a pulse-level language that generates AOD waveforms using a waypoint abstraction. AOD moves are expressed as sequences of grids (Cartesian products of x-positions and y-positions). The ArchSpec in bloqade-lanes defines the physical architecture but does not expose move information in a form natural to flair. This design adds zone-level APIs to ArchSpec that flair can consume directly for generating AOD grid moves.

### Unaligned grids and zone composition

The current ArchSpec data model has a deeper structural problem: it does not support unaligned grids well. The architecture is constructed by laying out all words in a flat global space and then grouping those words into zones after the fact. This means zones are an afterthought rather than a compositional primitive — you cannot naturally express zones with different grid alignments, spacings, or orientations.

PR #398 begins to address this by introducing a zone-based builder API (`ZoneSpec`, `ArchBlueprint`, `build_arch()`) where zones are composed together rather than carved out of a monolithic word layout. However, the underlying data model still stores words globally and zones as word-ID groupings. The new grid-based APIs proposed here (particularly `get_zone_grid` and `get_grid_endpoints`) make this limitation more acute: if zones have unaligned grids, the per-zone Grid returned must faithfully represent each zone's geometry independently, which requires the data model to treat zones as first-class compositional units with their own coordinate systems.

This also motivates extending `LaneAddress` and `LocationAddress` to include a zone index in the address itself. Currently, a `LaneAddress` encodes `(word_id, site_id, bus_id, move_type, direction)` and a `LocationAddress` encodes `(word_id, site_id)` — neither carries zone context. When zones have unaligned grids, the same `word_id` in different zones refers to fundamentally different physical geometries. Embedding the zone index in the address allows the system to meaningfully distinguish between different move topologies (e.g., a site bus move within a tightly-spaced entangling zone vs. the same bus type within a wider storage zone) and correctly dispatch to the right grid geometry without ambient zone context. This change is scoped to step 2 (#421).

For this iteration we constrain all zones to have the same number of x/y grid points to avoid the cross-zone mapping problem. The schema restructuring (#422) and the zone-based builder from PR #398 lay the groundwork for lifting this constraint in the future.

## Background: Flair AOD Moves

A **waypoint** is a grid configuration: `(x_positions, y_positions)` where the AOD traps form the Cartesian product of these lists. An AOD move is a sequence of waypoints.

On the Gemini architecture, moves never change inter-trap spacing along X or Y, so every move can be represented as an initial grid plus a sequence of `(shift_x, shift_y)` offsets. This means flair only needs the **start and end grid** from the ArchSpec — flair owns the trajectory between them.

## Scope

- Add four new zone-addressed methods to `ArchSpec`
- Restructure ArchSpec internals to zone-indexed bus lookups
- All new APIs assume Zone 0 default for now; the Kirin compiler is not refactored in this step
- Flair refactoring to consume these APIs is a separate follow-up

### Migration strategy

1. **Step 1 (this work):** Restructure ArchSpec internals to zone-indexed data models. Add new zone-addressed APIs. Existing word/bus-level APIs remain unchanged. Current compiler continues using Zone 0 implicitly.
2. **Step 2 (follow-up):** Refactor Kirin-based compiler components to use zone-addressed APIs. Expand `LaneAddress` and `LocationAddress` to include zone index.

## API Design

### Type aliases

```python
from bloqade.geometry import Grid

BusDescriptor = tuple[int, MoveType, Direction]  # (bus_id, move_type, direction)
```

Return types use `bloqade.geometry.Grid` directly since it is a dependency of both bloqade-lanes and bloqade-flair. The `Grid.positions` property provides column-major flattening (`product(x_positions, y_positions)`).

### New methods on ArchSpec

#### `get_zone_grid(zone_id: int) -> Grid`

Returns the grid of all site positions in a zone. Collects unique sorted x-values and y-values across all words in `zones[zone_id]`, constructs `Grid.from_positions(x_positions, y_positions)`.

Raises `ValueError` if `zone_id` is out of range.

#### `get_all_sites() -> list[tuple[float, float]]`

Returns all site positions across all zones. Iterates zones in order (0, 1, ...), within each zone flattens using `bloqade-geometry` column-major convention (`Grid.positions`). The result is the canonical site ordering for the entire architecture.

#### `get_available_buses(zone_id: int) -> list[BusDescriptor]`

Returns all valid `(bus_id, move_type, direction)` combinations for buses operating within a zone. Enumerates both site and word buses, each in both forward and backward directions.

Raises `ValueError` if `zone_id` is out of range.

#### `get_grid_endpoints(zone_id: int, bus_id: int, move_type: MoveType, direction: Direction) -> tuple[Grid, Grid]`

Returns `(start_grid, end_grid)` for a bus move assuming full occupancy. Collects all source positions and all destination positions for the given bus across the zone, constructs a `bloqade.geometry.Grid` for each.

Both grids must form valid Cartesian products (asserted as a construction invariant).

Raises `ValueError` if the `(zone_id, bus_id, move_type, direction)` combination is invalid (caller can check via `get_available_buses`).

## Internal restructuring

### Zone-indexed bus lookups

PR #398 introduces zone-based architecture with `ZoneSpec` and `ArchBlueprint`. Building on this, ArchSpec internals are restructured to maintain zone-indexed bus data:

```python
_zone_site_buses: dict[int, list[tuple[int, Bus]]]  # zone_id -> [(bus_id, Bus), ...]
_zone_word_buses: dict[int, list[tuple[int, Bus]]]  # zone_id -> [(bus_id, Bus), ...]
```

Built during `__post_init__` by cross-referencing which words belong to which zone and which words/sites have buses. Existing flat `site_buses`/`word_buses` remain for backwards compatibility with the current compiler.

Note: The exact internal data model will be finalized during implementation planning. The above is illustrative.

## Flattening convention

All position enumeration follows `bloqade-geometry`'s `Grid.positions` convention: `itertools.product(x_positions, y_positions)` — column-major, x varies slowest, y varies fastest:

```
(x0, y0), (x0, y1), ..., (x1, y0), (x1, y1), ...
```

## Validation

- **Invalid `zone_id`:** `ValueError` if `zone_id >= len(self.zones)`
- **Invalid bus descriptor:** `ValueError` if `(bus_id, move_type, direction)` is not in `get_available_buses(zone_id)`
- **Grid Cartesian product invariant:** Source and destination positions must form valid Cartesian products. This is a construction-time invariant of the ArchSpec, asserted rather than raised as a user error.

## Constraints and assumptions

- All zones have the same number of x/y grid points (avoids cross-zone mapping complexity)
- Flair owns the trajectory between start and end grids — ArchSpec only provides endpoints
- The zone index is not yet added to `LaneAddress` or `LocationAddress` — that is step 2

## Testing

- Unit tests for each API method with a small hand-built ArchSpec (2 zones, 4 words, 1 site bus, 1 word bus)
- `get_zone_grid`: verify Grid positions match expected x/y values, verify column-major flattening order
- `get_all_sites`: verify concatenation order across zones
- `get_available_buses`: verify correct descriptors per zone, no cross-zone leakage
- `get_grid_endpoints`: verify start/end grids are valid Cartesian products, verify positions match known bus source/destination locations
- Integration test with Gemini arch via `build_arch()` (from PR #398): verify grids are consistent with existing path data
