# ArchSpec Schema Update: Zone-Indexed Bus Data Models

**Date:** 2026-04-03
**Builds on:** PR #398 (zone-based architecture builder)
**Related:** #419 (ArchSpec-flair bridge), #420 (step 1: zone-addressed APIs)

## Problem

The ArchSpec JSON schema and Rust data models store buses as global flat arrays (`site_buses`, `word_buses`) with separate membership lists (`words_with_site_buses`, `sites_with_word_buses`). There is no per-zone bus configuration. For flair integration, we need the ArchSpec schema to express zone-scoped bus information so that flair can import an ArchSpec JSON file and derive zone-level AOD grid moves directly.

## Current Rust data model

```rust
pub struct ArchSpec {
    pub version: Version,
    pub geometry: Geometry,                    // Words + grids
    pub buses: Buses,                          // Global: { site_buses, word_buses }
    pub words_with_site_buses: Vec<u32>,       // Flat membership list
    pub sites_with_word_buses: Vec<u32>,       // Flat membership list
    pub zones: Vec<Zone>,                      // Zone = { words: Vec<u32> }
    pub entangling_zones: Vec<u32>,
    pub measurement_mode_zones: Vec<u32>,
    pub paths: Option<Vec<TransportPath>>,
    pub feed_forward: bool,
    pub atom_reloading: bool,
}

pub struct Buses {
    pub site_buses: Vec<Bus>,
    pub word_buses: Vec<Bus>,
}

pub struct Bus {
    pub src: Vec<u32>,
    pub dst: Vec<u32>,
}

pub struct Zone {
    pub words: Vec<u32>,
}
```

Key files:
- Rust types: `crates/bloqade-lanes-bytecode-core/src/arch/types.rs`
- Rust queries: `crates/bloqade-lanes-bytecode-core/src/arch/query.rs`
- Rust validation: `crates/bloqade-lanes-bytecode-core/src/arch/validate.rs`
- PyO3 bindings: `crates/bloqade-lanes-bytecode-python/src/arch_python.rs`
- JSON schema: `docs/src/arch/archspec-schema.json`
- Python wrapper: `python/bloqade/lanes/layout/arch.py`

## Proposed changes

### 1. Zone-scoped bus configuration in Rust

Extend `Zone` to carry its own bus configuration:

```rust
pub struct Zone {
    pub words: Vec<u32>,
    pub site_buses: Vec<Bus>,
    pub word_buses: Vec<Bus>,
    pub words_with_site_buses: Vec<u32>,  // Scoped to this zone's words
    pub sites_with_word_buses: Vec<u32>,
}
```

The top-level `buses`, `words_with_site_buses`, and `sites_with_word_buses` fields on `ArchSpec` become derived/deprecated. For backwards compatibility during migration, the top-level fields can be retained as the union across all zones.

### 2. JSON schema update

```json
{
  "zones": [
    {
      "words": [0, 1, 2, 3],
      "site_buses": [{"src": [0, 1], "dst": [3, 4]}],
      "word_buses": [{"src": [0], "dst": [1]}],
      "words_with_site_buses": [0, 1, 2, 3],
      "sites_with_word_buses": [0, 1, 2]
    }
  ]
}
```

Bump schema version to reflect the breaking change.

### 3. Validation updates

In `validate.rs`:
- Validate per-zone bus membership: `words_with_site_buses` entries must be in the zone's `words`
- Validate per-zone bus indices: site bus indices < `sites_per_word`, word bus indices refer to words within the zone
- Validate cross-zone consistency: ensure no bus references words outside its zone
- Keep existing global validations as cross-checks

### 4. PyO3 binding updates

In `arch_python.rs`:
- `PyZone` exposes `site_buses`, `word_buses`, `words_with_site_buses`, `sites_with_word_buses`
- Add zone-scoped query methods (or expose via the Python `ArchSpec` wrapper in `arch.py`)

### 5. Python ArchSpec wrapper

In `layout/arch.py`:
- Build zone-indexed bus lookups from the new per-zone data during `__post_init__`
- Existing APIs continue to work by aggregating across zones (or defaulting to Zone 0)
- New zone-addressed APIs (#420) consume the zone-indexed data directly

### 6. Transport paths

The `paths` field currently stores `TransportPath` entries keyed by `LaneAddr` (which has no zone index). For now, paths remain global. Adding zone to `LaneAddr` is step 2 (#421).

## Migration strategy

1. Update Rust `Zone` struct with bus fields
2. Update JSON schema and bump version
3. Update serde: support new format, optionally support old format for migration
4. Update validation
5. Update PyO3 bindings
6. Update Python `ArchSpec.__post_init__` to build zone-indexed lookups
7. Update `build_arch()` (from PR #398) to populate per-zone bus fields
8. Update example JSON files

The Python builder (`build_arch()`) already has zone-level information via `ZoneSpec` and `ArchBlueprint` from PR #398. It currently flattens bus info into the global fields. The change is to write per-zone bus data into the new `Zone` fields instead.

## Testing

- Rust unit tests: serialize/deserialize round-trip with zone-scoped buses
- Rust validation tests: per-zone bus membership, cross-zone isolation
- Python integration tests: `build_arch()` produces correct zone-indexed bus data
- Backwards compat test: ensure old JSON files can still be loaded (if migration support is needed)
- CLI smoke test: bytecode validation still works with updated schema
