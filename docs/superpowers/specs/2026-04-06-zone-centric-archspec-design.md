# Zone-Centric ArchSpec Redesign

**Date:** 2026-04-06
**Issue:** #422
**Builds on:** PR #398 (zone-based architecture builder)
**Related:** #419 (ArchSpec-flair bridge), #420 (zone-addressed APIs), #421 (address encoding)
**Supersedes:** `plans/2026-04-03-archspec-schema-update-design.md`

## Problem

The ArchSpec data model stores buses as global flat arrays, zones as word-ID groupings with no structural ownership, and words as independent geometric entities each carrying their own grid. This design has several deficiencies:

1. **No bus hierarchy.** There is no distinction between intra-word (site bus), intra-zone (word bus), and inter-zone (zone bus) movement. All buses live in a single global list.
2. **Zones are not structural.** A zone is just a list of word IDs with no coordinate system, no bus ownership, and no geometric identity. Zone grids must be reconstructed by aggregating across words.
3. **No inter-zone bus type.** There is no concept of a bus that moves atoms between zones. The `word_buses` list conflates intra-zone and inter-zone movement.
4. **Addresses lack zone context.** `LocationAddr` (32-bit) and `LaneAddr` (64-bit) encode `(word_id, site_id)` but carry no zone information.
5. **Entangling model is word-level.** CZ gate capability is expressed via `has_cz` word pairs and a flat `entangling_zones` list, rather than as a zone-pair relationship.
6. **Measurement configuration is implicit.** `measurement_mode_zones` lists zones but doesn't define bitstring ordering or support multiple measurement configurations.

For flair integration, the ArchSpec needs zone-level geometric identity and bus ownership so that flair can import an ArchSpec and derive zone-level AOD grid moves directly.

## Design

### Core principle: zones as the primary structural unit

Each zone owns a `Grid` (its physical coordinate system) and its internal bus connectivity. Words are a global slicing template — the same logical structure repeated across every zone with potentially different physical spacing. A site's physical position is resolved by indexing into its zone's grid.

### Structural constraint: uniform zone dimensions

All zones must have the same grid dimensions (n_x, n_y), the same number of words, and the same sites_per_word. What differs per zone is the actual grid spacing (physical coordinates) and the internal bus connectivity (site buses, word buses). This constraint is required for the entangling model (zone pairs with 1:1 site correspondence) and simplifies cross-zone operations.

### Rust data model

#### Address newtypes

```rust
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct SiteRef(pub u16);

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct WordRef(pub u16);

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ZonedWordRef {
    pub zone_id: u8,
    pub word_id: u16,
}
```

Widths match the address encoding: `SiteRef` = 16-bit, `WordRef` = 16-bit, `ZonedWordRef` = 8-bit zone + 16-bit word.

#### Generic bus

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Bus<T> {
    pub src: Vec<T>,
    pub dst: Vec<T>,
}
```

One type parameterized by address type. Monomorphized for PyO3 bindings as `SiteBus = Bus<SiteRef>`, `WordBus = Bus<WordRef>`, `ZoneBus = Bus<ZonedWordRef>`.

#### Zone

```rust
pub struct Zone {
    pub grid: Grid,
    pub site_buses: Vec<Bus<SiteRef>>,
    pub word_buses: Vec<Bus<WordRef>>,
    pub words_with_site_buses: Vec<u32>,
    pub sites_with_word_buses: Vec<u32>,
    pub entangling_pairs: Vec<[u32; 2]>,  // word pairs within this zone for CZ gates
}
```

Each zone is a self-contained unit for internal connectivity and entangling capability. The `grid` is the zone's coordinate system — a site's physical position is `(grid.x[word.sites[s][0]], grid.y[word.sites[s][1]])`. The `entangling_pairs` field lists word pairs within this zone that are within blockade radius for CZ gates. A zone with empty `entangling_pairs` is a storage/low-connectivity zone.

#### Word

```rust
pub struct Word {
    pub sites: Vec<[u32; 2]>,  // (x_index, y_index) into parent zone's grid
}
```

Words are a global slicing template. Every zone shares the same word definitions. A word is just a grouping of grid index pairs — it has no independent geometry.

#### Mode

```rust
pub struct Mode {
    pub name: String,
    pub zones: Vec<u32>,
    pub bitstring_order: Vec<LocationAddr>,
}
```

A measurement configuration specifying which zones are imaged and the explicit bit-position-to-site mapping.

#### ArchSpec

```rust
pub struct ArchSpec {
    pub version: Version,
    pub words: Vec<Word>,
    pub zones: Vec<Zone>,
    pub zone_buses: Vec<Bus<ZonedWordRef>>,
    pub modes: Vec<Mode>,
    pub paths: Option<Vec<TransportPath>>,
    pub feed_forward: bool,
    pub atom_reloading: bool,
}
```

Removed fields: `geometry` (struct), `buses` (struct), `words_with_site_buses`, `sites_with_word_buses`, `entangling_zones`, `measurement_mode_zones`, `has_cz` (on Word), `entangling_zone_pairs` (moved into Zone).

#### Entangling pairs (intra-zone)

Entangling pairs are a **zone-level** concept. Each zone declares its own `entangling_pairs: Vec<[u32; 2]>` — word pairs within that zone that are at blockade radius for CZ gates. A gate zone has entangling pairs; a storage zone does not.

For example, a single gate zone containing 10 words in a 2-column interleaved grid might have `entangling_pairs: [[0,5], [1,6], [2,7], [3,8], [4,9]]` — pairing words in column 0 with words in column 1.

**Replacement API for `has_cz` / `get_blockaded_location`:** Given a `LocationAddr(zone=z, word=w, site=s)`, its CZ partner is `LocationAddr(zone=z, word=w', site=s)` where `(w, w')` or `(w', w)` appears in `zones[z].entangling_pairs`. The partner is always in the **same zone**. If no pair exists, the site has no CZ partner. The Python `ArchSpec` wrapper provides a `get_cz_partner(loc: LocationAddr) -> Optional[LocationAddr]` method implementing this lookup. The old `Word.cz_pair`, `Word.has_cz`, and `ArchSpec.get_blockaded_location()` are removed.

**Why intra-zone, not inter-zone:** CZ gate execution targets a specific zone. The zone must contain both words in the pair. This means all compilation algorithms operate within a single zone_id — no cross-zone addressing is needed for gate execution. Zones with entangling pairs are high-connectivity gate zones; zones without are low-connectivity storage zones. Zone buses handle transport between them.

#### sites_per_word

`sites_per_word` is no longer stored as an explicit field. It is derived as `words[0].sites.len()`. The uniform zone dimension constraint guarantees all words have the same number of sites. Validation checks that `words[i].sites.len() == words[0].sites.len()` for all `i`.

### Address encoding

#### LocationAddr — widened to 64-bit

```
[zone_id:8][word_id:16][site_id:16][pad:24]
```

- 256 zones, 65536 words, 65536 sites
- `ZonedWordRef` maps to the upper 24 bits (zone_id + word_id)
- Encoded/decoded as `u64`. All existing code using `u32` encoding must be updated.

#### LaneAddr — stays 64-bit, reorganized

```
data0: [word_id:16][site_id:16]         (unchanged from current layout)
data1: [dir:1][mt:2][zone_id:8][pad:5][bus_id:16]
```

`data0` retains its current 32-bit `[word_id:16][site_id:16]` layout. Zone context in `LaneAddr` comes exclusively from `data1`. This differs from `LocationAddr` where `zone_id` is in the high bits — the two types have independent encoding layouts optimized for their respective use cases.

#### MoveType — expanded to 2 bits

```rust
#[repr(u8)]
pub enum MoveType {
    SiteBus = 0,
    WordBus = 1,
    ZoneBus = 2,
    // value 3 is reserved / invalid
}
```

### JSON schema

Schema version bumped to reflect the breaking change. No backwards compatibility — clean break. The `Version` type remains two-component (`major.minor`).

The `Grid` serialization format is unchanged — it uses the existing `x_start`/`y_start`/`x_spacing`/`y_spacing` representation (cumulative spacing from a start point), which already supports non-uniform spacing across zones.

```json
{
  "version": "2.0",
  "words": [
    { "sites": [[0, 0], [0, 1], [1, 0], [1, 1]] },
    { "sites": [[2, 0], [2, 1], [0, 2], [0, 3]] }
  ],
  "zones": [
    {
      "grid": { "x_start": 0.0, "y_start": 0.0, "x_spacing": [5.0, 5.0], "y_spacing": [3.0, 3.0, 3.0] },
      "site_buses": [{ "src": [0, 1], "dst": [2, 3] }],
      "word_buses": [{ "src": [0], "dst": [1] }],
      "words_with_site_buses": [0, 1],
      "sites_with_word_buses": [0, 1, 2],
      "entangling_pairs": [[0, 1]]
    },
    {
      "grid": { "x_start": 0.0, "y_start": 0.0, "x_spacing": [7.5, 7.5], "y_spacing": [4.0, 4.0, 4.0] },
      "site_buses": [],
      "word_buses": [],
      "words_with_site_buses": [],
      "sites_with_word_buses": [],
      "entangling_pairs": []
    }
  ],
  "zone_buses": [
    {
      "src": [{ "zone_id": 0, "word_id": 0 }],
      "dst": [{ "zone_id": 1, "word_id": 0 }]
    }
  ],
  "modes": [
    {
      "name": "full",
      "zones": [0, 1],
      "bitstring_order": [...]
    }
  ]
}
```

### Validation

#### Per-zone bus validation
- `words_with_site_buses` entries must be valid word indices (< number of words)
- `sites_with_word_buses` entries must be valid site indices within the zone's words
- `SiteRef` values in site buses must be < sites_per_word
- `WordRef` values in word buses must be valid word indices
- No intra-zone bus references words outside the valid range

#### Zone bus validation
- All `ZonedWordRef` entries must reference valid `(zone_id, word_id)` pairs
- Every `(src[i], dst[i])` pair must cross a zone boundary — i.e., `src[i].zone_id != dst[i].zone_id` for all `i`

#### Grid invariant (all bus types)
- For every bus, the src positions and dst positions (when resolved to physical coordinates) must each form a complete rectangular grid (Cartesian product of unique x-values and y-values)
- The src grid and dst grid must have the same number of rows and columns

#### Structural invariants
- All zones must have the same grid dimensions (same number of x-positions and y-positions)
- All words must have the same number of sites (sites_per_word)
- Word site indices must be within the zone grid dimensions

#### Entangling pairs (per-zone)
- Word indices in `entangling_pairs` must be valid (< number of words)
- Both words in a pair must be distinct
- No duplicate pairs (order-independent)

#### Modes
- Zone indices in `modes[*].zones` must be valid
- `bitstring_order` entries must be valid `LocationAddr` values referencing sites in the mode's zones

### PyO3 bindings

Monomorphized bus types for the Python boundary:

```rust
#[pyclass(name = "SiteBus")]
pub struct PySiteBus(Bus<SiteRef>);

#[pyclass(name = "WordBus")]
pub struct PyWordBus(Bus<WordRef>);

#[pyclass(name = "ZoneBus")]
pub struct PyZoneBus(Bus<ZonedWordRef>);
```

Updated wrappers:
- `PyZone`: expose `grid`, `site_buses`, `word_buses`, membership lists
- `PyArchSpec`: expose `zones`, `zone_buses`, `modes`, `words`
- `PyLocationAddr`: widen to 64-bit, add `zone_id` property
- `PyLaneAddr`: add `zone_id` property, `MoveType` gains `ZoneBus` variant
- `PyMode`: new wrapper for measurement configurations

### Python layer

#### ArchSpec wrapper (`layout/arch.py`)
- `__post_init__` builds zone-indexed bus lookups from per-zone data
- `get_zone_grid(zone_id)` returns the zone's grid directly
- `get_available_buses(zone_id)` iterates the zone's bus lists
- `get_grid_endpoints(zone_id, bus_id, move_type, direction)` resolves src/dst through the zone's grid
- Old global bus accessors removed

#### Encoding (`layout/encoding.py`)
- `LocationAddress` widens to 64-bit, gains `zone_id` property
- `LaneAddress` gains `zone_id` property
- `MoveType` gains `ZoneBus` variant
- `ZonedWordRef` exposed as a Python class

#### Word wrapper (`layout/word.py`)
- The Python `Word` class currently stores `positions: grid.Grid` and implements `site_position()`, `all_positions()`, and `plot()`. Since words no longer own geometry, these methods move to the zone or are removed. A word becomes a thin wrapper over `sites: list[tuple[int, int]]`. Position resolution requires the parent zone's grid.

#### PathFinder (`layout/path.py`)
- Graph construction becomes zone-aware — builds edges from per-zone buses
- Zone bus edges added for inter-zone connectivity
- `LaneAddress` carries zone context so paths can cross zone boundaries

### Builder changes

The zone-based builder from PR #398 (`ZoneSpec`, `ArchBlueprint`, `build_arch()`) is updated. PR #398 must be merged before this work begins, as it introduces the builder API that this spec modifies.

#### `ZoneSpec`
- May carry per-zone layout overrides (grid spacing) instead of relying solely on global `DeviceLayout`
- Builder validates that all `ZoneSpec` entries produce the same grid dimensions and sites_per_word

#### `DeviceLayout`
- Per-zone spacing parameters or zone-specific overrides

#### `build_arch()`
- Produces global word template (shared across zones)
- Each zone gets its own `Grid` with zone-specific spacing
- Site buses and word buses written per-zone
- `InterZoneTopology` produces `Bus<ZonedWordRef>` entries for `zone_buses`
- Generates per-zone `entangling_pairs` from zone specs with `entangling=True`
- Generates default `Mode` entries from zone specs with `measurement=True`

#### Zone 0 convention removed
- Zone 0 = "all words" no longer exists
- Zones are indexed 0, 1, 2, ... and each is a real zone with its own grid
- If the Rust bytecode runtime requires a "default zone", use Mode or derive it

### Schema and documentation

- `docs/src/arch/archspec-schema.json`: updated to zone-centric model
- `examples/arch/`: updated example JSON files
- `docs/src/arch/`: mdBook pages updated for zone-centric model, bus hierarchy, address encoding
- Rust API docs: updated doc comments on all changed types
- Python type stubs (`bytecode/_native.pyi`): updated for new PyO3 bindings

### Testing

- **Rust**: serialize/deserialize round-trip with zone-owned grids and per-zone buses
- **Rust**: validation tests — grid dimension consistency, per-zone bus membership, zone bus cross-zone requirement, grid completeness invariant, entangling pair validity, mode validity
- **Rust**: address encoding round-trip with zone_id in LocationAddr (64-bit) and LaneAddr
- **Python**: `build_arch()` produces correct zone-centric data with per-zone grids
- **Python**: flair bridge APIs return correct results
- **Python**: PathFinder builds correct graph with zone-aware edges
- **CLI smoke test**: bytecode validation with updated schema

## Migration

This is a breaking change across all API surfaces (Rust, Python, C FFI). No backwards compatibility shim — clean break to schema v2.

### Breaking changes by surface

**Rust:**
- `ArchSpec` struct: removed `geometry`, `buses`, `words_with_site_buses`, `sites_with_word_buses`, `entangling_zones`, `measurement_mode_zones`. Added `words`, `zone_buses`, `modes`.
- `Zone` struct: added `grid`, `site_buses`, `word_buses`, membership lists, `entangling_pairs`.
- `Word` struct: simplified to `sites: Vec<[u32; 2]>`. Removed `positions`, `site_indices`, `has_cz`.
- `Bus` struct: now generic `Bus<T>`. Old `Bus { src: Vec<u32>, dst: Vec<u32> }` replaced.
- `Buses` struct: removed.
- `Geometry` struct: removed.
- `LocationAddr`: widened from 32-bit to 64-bit, gains `zone_id`.
- `LaneAddr`: `MoveType` gains `ZoneBus` variant (2-bit), `zone_id` added to data1.
- New types: `SiteRef`, `WordRef`, `ZonedWordRef`, `Mode`.

**Python:**
- `ArchSpec` wrapper: API changes mirror Rust.
- `LocationAddress`: widens to 64-bit, gains `zone_id`.
- `LaneAddress`: gains `zone_id`, `MoveType.ZoneBus`.
- Builder: `build_arch()` output structure changes.

**C FFI / JSON:**
- Schema version bump.
- All JSON files must be regenerated for new schema.
- C header (`bloqade_lanes_bytecode.h`) regenerated from updated Rust types via cbindgen. The CLI crate's `ffi/arch.rs` functions that operate on the current data model will need updating to reflect the new struct layout.
