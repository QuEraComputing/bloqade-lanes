# Arch Builder Redesign

**Date**: 2026-04-09
**Status**: Draft
**Issue**: Builder produces invalid ArchSpecs (even/odd zone splitting, cross-zone bus references)

## Problem

The current `build_arch()` splits entangling zones into even/odd column sub-zones,
creating 2 Rust zones per blueprint zone. This causes:

1. Duplicate buses — both sub-zones get identical copies of all site and word buses
2. Cross-zone entangling pairs — word 0 (zone 0) paired with word 1 (zone 1)
3. Global word IDs in zone-local bus fields (WordRef)
4. Inter-zone word buses appended to intra-zone `word_buses` instead of `zone_buses`

The Rust validator passes because it doesn't enforce zone-local scoping.

## Design

Two layers: `ZoneBuilder` for constructing individual zones with validation,
and `ArchBuilder` for composing zones into a complete ArchSpec. The existing
`build_arch()` function becomes a convenience wrapper that uses both internally.

### ZoneBuilder

Builds one zone in zone-local coordinates. All indices are local to this zone.

```python
class ZoneBuilder:
    def __init__(self, name: str, grid: Grid, word_shape: tuple[int, int]):
        """Initialize a zone.

        Args:
            name: Human-readable zone name (stored in Rust Zone).
            grid: Coordinate grid for this zone.
            word_shape: (num_x_sites, num_y_sites) — uniform shape for
                all words in this zone. sites_per_word = product of shape.
        """

    def add_word(self, x_sites: slice | list[int], y_sites: slice | list[int]) -> int:
        """Add a word occupying the given grid positions.

        The number of x-indices and y-indices must match word_shape.
        Grid positions must not overlap with any existing word.

        Returns:
            Zone-local word index.

        Raises:
            ValueError: Shape mismatch or grid position overlap
                (message includes which word owns the conflicting position).
            IndexError: Indices out of range for this zone's grid.
        """

    def add_site_bus(self, src: list[int], dst: list[int]):
        """Add a site bus (intra-word movement).

        src/dst are site indices within word_shape (0..sites_per_word).
        Validates that src/dst positions form a valid AOD Cartesian product
        on the word grid.
        """

    def add_word_bus(self, src: list[int], dst: list[int]):
        """Add a word bus (intra-zone movement).

        src/dst are zone-local word indices.
        Validates that src/dst positions form a valid AOD Cartesian product
        on the zone grid.
        """

    def add_entangling_pair(self, word_a: int, word_b: int):
        """Mark two zone-local words as a CZ pair."""

    @property
    def words(self) -> _GridQuery:
        """Query word indices by grid region.

        Returns (zone_name, list[int]) — the zone name and zone-local
        word indices whose sites intersect the queried region.

        Usage:
            zone.words[slice(0, 2), :]       → ("gate", [0, 2, 4])
            zone.words[:, slice(0, 5)]        → ("gate", [0, 1])
            zone.words[0, 0]                  → ("gate", [0])

        The returned tuple can be passed directly to ArchBuilder.connect().
        """

    @property
    def sites(self) -> _GridQuery:
        """Query site indices within the word shape.

        Usage:
            zone.sites[:, 0]                  → site indices in first y-row
            zone.sites[slice(0, 5), 1]        → x=[0,5) at y=1
        """
```

#### Grid querying

`zone.words[x, y]` and `zone.sites[x, y]` both accept `(slice | int | list, slice | int | list)`
and return `list[int]`. The `words` query returns zone-local word indices whose sites intersect
the specified grid region. The `sites` query returns word-local site indices within the word shape.

#### Validation on add_word

Each grid position `(x, y)` can belong to at most one word. The zone maintains a
`dict[tuple[int, int], int]` mapping grid positions to word indices. On `add_word`,
every position in the word's footprint is checked against this map.

#### AOD Cartesian product validation on buses

For `add_site_bus(src, dst)`: resolve each site index to its `(x, y)` position
within the word shape. The set of src x-positions × src y-positions must equal the
src position set (and likewise for dst). This ensures the bus defines a complete
rectangular AOD grid.

Same validation for `add_word_bus` using word positions on the zone grid,
and for `ArchBuilder.connect` using word positions across the two zone grids
(resolved from the `(zone_name, word_indices)` tuples).

### ArchBuilder

Composes zones into a complete ArchSpec.

```python
class ArchBuilder:
    def add_zone(self, zone: ZoneBuilder) -> int:
        """Add a zone. Returns zone_id. Assigns global word IDs.

        Validates:
            - sites_per_word matches across all zones.
        """

    def connect(
        self,
        src: tuple[str, list[int]],
        dst: tuple[str, list[int]],
    ):
        """Add an inter-zone bus (zone_buses).

        Args:
            src: (zone_name, zone_local_word_indices) — typically from
                zone.words[...] which returns this tuple directly.
            dst: (zone_name, zone_local_word_indices) — same format.

        Validates AOD Cartesian product across the two zone grids.

        Example:
            arch.connect(src=proc.words[:, :], dst=mem.words[:, :])
        """

    def add_mode(self, name: str, zones: list[str]):
        """Add an operational mode.

        Args:
            name: Mode name (e.g. "all", "gate", "measure").
            zones: Zone names to include in this mode.
        """

    def build(self) -> ArchSpec:
        """Assemble the ArchSpec and validate via Rust.

        Steps:
            1. Collect all words (with global IDs).
            2. Build Rust Zone objects (with name field).
            3. Build zone_buses from connect() calls.
            4. Build modes.
            5. Call ArchSpec.from_components() → Rust validate().

        Raises:
            ValueError: If Rust validation fails.
        """
```

#### Global word ID assignment

When `add_zone()` is called, the builder assigns global word IDs starting from
the current offset. Zone-local word index `i` in zone `z` maps to global
`word_id_offsets[z] + i`. This mapping is used when building Rust Zone objects
and zone_buses.

### build_arch() wrapper

The public `build_arch(blueprint, connections)` function is unchanged in signature.
Internally it:

1. For each blueprint zone: creates a `ZoneBuilder`, places words using the
   zone spec's grid layout, generates site/word buses from topologies,
   adds entangling pairs if `entangling=True`.
2. Feeds all `ZoneBuilder`s into `ArchBuilder.add_zone()`.
3. For each connection: calls `ArchBuilder.connect(src=zone_a.words[...], dst=zone_b.words[...])`.
4. Adds default "all" mode (`arch.add_mode("all", list(blueprint.zones.keys()))`)
   and per-zone measurement modes by zone name.
5. Returns `ArchResult` with the built ArchSpec and metadata.

The critical difference from the current implementation: no even/odd zone splitting.
One blueprint zone = one Rust zone. Entangling pairs are metadata on the zone,
not a reason to create sub-zones.

### Rust Zone name field

Add `name: String` to the Rust `Zone` struct:

```rust
pub struct Zone {
    pub name: String,           // NEW
    pub grid: Grid,
    pub site_buses: Vec<Bus<SiteRef>>,
    pub word_buses: Vec<Bus<WordRef>>,
    pub words_with_site_buses: Vec<u32>,
    pub sites_with_word_buses: Vec<u32>,
    pub entangling_pairs: Vec<[u32; 2]>,
}
```

The name is serialized to/from JSON. It has no semantic effect on validation
or lane resolution — it's metadata for debugging and documentation.

## Testing

- **ZoneBuilder unit tests**: word placement, overlap rejection, shape validation,
  grid querying (words/sites), AOD validation on buses
- **ArchBuilder unit tests**: multi-zone composition, sites_per_word enforcement,
  global word ID assignment, zone bus validation
- **build_arch integration tests**: verify single-zone and multi-zone blueprints
  produce valid ArchSpecs (Rust validation passes), correct bus/zone counts,
  correct entangling pairs
- **Regression**: verify the Gemini logical/physical specs still produce the
  same ArchSpec (same positions, same connectivity)

## Migration

- `build_arch()` signature unchanged — no downstream breakage
- `ArchResult` unchanged
- Topology interfaces unchanged
- The even/odd splitting code and `zone_even_odd_map` are deleted
- Rust Zone gains a `name` field (schema addition, backward compatible with
  `#[serde(default)]`)
