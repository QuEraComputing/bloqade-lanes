# Zone-Centric ArchSpec Redesign — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the ArchSpec data model so zones are the primary structural unit, each owning a Grid and bus connectivity, with a three-tier bus hierarchy (site/word/zone) and zone-aware address encoding.

**Architecture:** Bottom-up implementation starting from Rust core types, through validation/query, PyO3 bindings, Python wrappers, and finally consumers + docs. Each layer is tested before the next begins. PR #398 (zone-based builder) must be merged first.

**Tech Stack:** Rust (serde, PyO3, cbindgen), Python (kirin-toolchain, bloqade-geometry, rustworkx, dataclasses), pytest, cargo test

**Spec:** `docs/superpowers/specs/2026-04-06-zone-centric-archspec-design.md`

**Prerequisite:** PR #398 merged (introduces `ZoneSpec`, `ArchBlueprint`, `build_arch()`)

---

## File Structure

### Rust — modified files

| File | Responsibility | Change |
|------|---------------|--------|
| `crates/bloqade-lanes-bytecode-core/src/arch/addr.rs` | Bit-packed address types | Widen LocationAddr to 64-bit, add zone_id to LaneAddr, expand MoveType to 2-bit, add SiteRef/WordRef/ZonedWordRef newtypes |
| `crates/bloqade-lanes-bytecode-core/src/arch/types.rs` | Core data model structs | Generic Bus\<T\>, restructured Zone (owns Grid+buses), simplified Word, new Mode, restructured ArchSpec |
| `crates/bloqade-lanes-bytecode-core/src/arch/validate.rs` | Validation logic | Zone-centric validation rules, grid invariant checks, entangling pair validation, mode validation |
| `crates/bloqade-lanes-bytecode-core/src/arch/query.rs` | Query methods + JSON loading | Zone-aware position lookup, lane resolution, get_cz_partner, updated error types |
| `crates/bloqade-lanes-bytecode-core/src/arch/mod.rs` | Module exports | Export new types (SiteRef, WordRef, ZonedWordRef, Mode) |
| `crates/bloqade-lanes-bytecode-core/src/bytecode/instruction.rs` | Instruction encoding | ConstLoc now u64, MoveType 2-bit in ConstLane |
| `crates/bloqade-lanes-bytecode-core/src/bytecode/value.rs` | Device values | DeviceValue::Location becomes u64 |
| `crates/bloqade-lanes-bytecode-core/src/bytecode/text.rs` | Text assembly | Updated address parsing/printing |
| `crates/bloqade-lanes-bytecode-core/src/bytecode/encode.rs` | Binary encoding | LocationAddr encoding width change |
| `crates/bloqade-lanes-bytecode-core/src/bytecode/validate.rs` | Bytecode validation | Updated address validation |
| `crates/bloqade-lanes-bytecode-core/src/atom_state.rs` | Atom state tracking | Zone-aware location resolution |
| `crates/bloqade-lanes-bytecode-python/src/arch_python.rs` | PyO3 arch bindings | Monomorphized bus types, updated Zone/Word/ArchSpec/Mode wrappers |
| `crates/bloqade-lanes-bytecode-python/src/lib.rs` | PyO3 module init | Register new types |
| `crates/bloqade-lanes-bytecode-python/src/instruction_python.rs` | PyO3 instruction bindings | Updated address widths |
| `crates/bloqade-lanes-bytecode-python/src/validation.rs` | Error conversion | New zone error types |
| `crates/bloqade-lanes-bytecode-python/src/atom_state_python.rs` | PyO3 atom state | Zone-aware wrappers |
| `crates/bloqade-lanes-bytecode-cli/src/ffi/arch.rs` | C FFI | Updated struct layout |
| `crates/bloqade-lanes-bytecode-cli/src/ffi/validate.rs` | C FFI validation | Updated error types |

### Python — modified files

| File | Responsibility | Change |
|------|---------------|--------|
| `python/bloqade/lanes/bytecode/_native.pyi` | Type stubs | Updated stubs for all new/changed Rust types |
| `python/bloqade/lanes/layout/encoding.py` | Address wrappers | LocationAddress 64-bit + zone_id, LaneAddress zone_id, MoveType.ZoneBus |
| `python/bloqade/lanes/layout/arch.py` | ArchSpec wrapper | Zone-indexed bus lookups, get_cz_partner(), get_zone_grid(), zone-centric API |
| `python/bloqade/lanes/layout/word.py` | Word wrapper | Simplified to grid index pairs, remove geometry ownership |
| `python/bloqade/lanes/layout/path.py` | PathFinder | Zone-aware graph construction, zone bus edges |
| `python/bloqade/lanes/arch/zone.py` | ZoneSpec/Blueprint | Per-zone layout overrides, dimension validation |
| `python/bloqade/lanes/arch/builder.py` | build_arch() | Produce zone-centric ArchSpec, zone buses, modes |
| `python/bloqade/lanes/arch/word_factory.py` | Word creation | Global word template, grid index pairs |
| `python/bloqade/lanes/arch/topology.py` | Bus topology | InterZoneTopology produces Bus\<ZonedWordRef\> |
| `python/bloqade/lanes/arch/gemini/` | Gemini specs | Updated to zone-centric builder |
| `python/bloqade/lanes/analysis/placement/lattice.py` | CZ placement | Use get_cz_partner() instead of has_cz |
| `python/bloqade/lanes/heuristics/logical_placement.py` | Placement heuristics | Zone-pair CZ model |
| `python/bloqade/lanes/heuristics/move_synthesis.py` | Move synthesis | Zone-aware bus access |
| `python/bloqade/lanes/rewrite/place2move.py` | Place→Move rewrite | Zone-pair CZ gates |
| `python/bloqade/lanes/rewrite/move2squin/noise.py` | Noise model | Zone-aware noise |
| `python/bloqade/lanes/dialects/move.py` | Move dialect | Zone bus lane addresses |

### Docs & Schema — modified files

| File | Change |
|------|--------|
| `docs/src/arch/archspec-schema.json` | Version 2.0 zone-centric schema |
| `docs/src/arch/archspec.md` | Restructured documentation |
| `docs/src/bytecode/inst-spec.md` | Updated address encoding docs |
| `examples/arch/simple.json` | Version 2.0 format |
| `examples/arch/full.json` | Version 2.0 format |
| `examples/arch/gemini-logical.json` | Version 2.0 format |

---

## Chunk 1: Rust Core Types and Address Encoding

### Task 1: Address newtypes (SiteRef, WordRef, ZonedWordRef)

**Files:**
- Modify: `crates/bloqade-lanes-bytecode-core/src/arch/addr.rs`

- [ ] **Step 1: Write tests for new newtypes**

Add to the `tests` module in `addr.rs`:

```rust
#[test]
fn test_site_ref_newtype() {
    let s = SiteRef(42);
    assert_eq!(s.0, 42);
    // Serialize round-trip
    let json = serde_json::to_string(&s).unwrap();
    let deserialized: SiteRef = serde_json::from_str(&json).unwrap();
    assert_eq!(s, deserialized);
}

#[test]
fn test_word_ref_newtype() {
    let w = WordRef(100);
    assert_eq!(w.0, 100);
    let json = serde_json::to_string(&w).unwrap();
    let deserialized: WordRef = serde_json::from_str(&json).unwrap();
    assert_eq!(w, deserialized);
}

#[test]
fn test_zoned_word_ref() {
    let zwr = ZonedWordRef { zone_id: 3, word_id: 42 };
    assert_eq!(zwr.zone_id, 3);
    assert_eq!(zwr.word_id, 42);
    let json = serde_json::to_string(&zwr).unwrap();
    let deserialized: ZonedWordRef = serde_json::from_str(&json).unwrap();
    assert_eq!(zwr, deserialized);
}
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cargo test -p bloqade-lanes-bytecode-core test_site_ref_newtype test_word_ref_newtype test_zoned_word_ref`
Expected: FAIL — types not defined

- [ ] **Step 3: Implement newtypes**

Add to `addr.rs` before the existing types:

```rust
/// Site index within a word. Matches the 16-bit site_id field in LocationAddr.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SiteRef(pub u16);

/// Word index within a zone. Matches the 16-bit word_id field in LocationAddr.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WordRef(pub u16);

/// Zone-qualified word reference for inter-zone bus entries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ZonedWordRef {
    pub zone_id: u8,
    pub word_id: u16,
}
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cargo test -p bloqade-lanes-bytecode-core test_site_ref_newtype test_word_ref_newtype test_zoned_word_ref`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/bloqade-lanes-bytecode-core/src/arch/addr.rs
git commit -m "feat(core): add SiteRef, WordRef, ZonedWordRef address newtypes"
```

### Task 2: Widen LocationAddr to 64-bit, expand MoveType, add zone_id to LaneAddr

**Files:**
- Modify: `crates/bloqade-lanes-bytecode-core/src/arch/addr.rs`

- [ ] **Step 1: Write tests for MoveType expansion and new encoding layouts**

```rust
#[test]
fn test_move_type_zone_bus() {
    assert_eq!(MoveType::SiteBus as u8, 0);
    assert_eq!(MoveType::WordBus as u8, 1);
    assert_eq!(MoveType::ZoneBus as u8, 2);
}

#[test]
fn test_location_addr_64bit_round_trip() {
    let addr = LocationAddr {
        zone_id: 5,
        word_id: 0x1234,
        site_id: 0x5678,
    };
    let bits = addr.encode();
    assert_eq!(LocationAddr::decode(bits), addr);
    // Check bit positions: [zone_id:8][word_id:16][site_id:16][pad:24]
    assert_eq!((bits >> 56) & 0xFF, 5);       // zone_id in top 8 bits
    assert_eq!((bits >> 40) & 0xFFFF, 0x1234); // word_id next 16
    assert_eq!((bits >> 24) & 0xFFFF, 0x5678); // site_id next 16
    assert_eq!(bits & 0xFFFFFF, 0);             // padding
}

#[test]
fn test_location_addr_zero() {
    let addr = LocationAddr { zone_id: 0, word_id: 0, site_id: 0 };
    assert_eq!(addr.encode(), 0u64);
    assert_eq!(LocationAddr::decode(0), addr);
}

#[test]
fn test_lane_addr_with_zone_id() {
    let addr = LaneAddr {
        direction: Direction::Backward,
        move_type: MoveType::ZoneBus,
        zone_id: 7,
        word_id: 0x1234,
        site_id: 0x5678,
        bus_id: 0x9ABC,
    };
    let (data0, data1) = addr.encode();
    let decoded = LaneAddr::decode(data0, data1);
    assert_eq!(decoded, addr);
    // data0 unchanged: [word_id:16][site_id:16]
    assert_eq!((data0 >> 16) & 0xFFFF, 0x1234);
    assert_eq!(data0 & 0xFFFF, 0x5678);
    // data1: [dir:1][mt:2][zone_id:8][pad:5][bus_id:16]
    assert_eq!((data1 >> 31) & 1, 1);          // dir = Backward
    assert_eq!((data1 >> 29) & 0x3, 2);        // mt = ZoneBus
    assert_eq!((data1 >> 21) & 0xFF, 7);       // zone_id
    assert_eq!(data1 & 0xFFFF, 0x9ABC);        // bus_id
}
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cargo test -p bloqade-lanes-bytecode-core test_move_type_zone_bus test_location_addr_64bit test_lane_addr_with_zone`
Expected: FAIL — ZoneBus not defined, zone_id field doesn't exist, encode returns u32

- [ ] **Step 3: Add ZoneBus variant to MoveType**

```rust
pub enum MoveType {
    SiteBus = 0,
    WordBus = 1,
    ZoneBus = 2,
}
```

- [ ] **Step 4: Implement LocationAddr 64-bit encoding**

Rewrite `LocationAddr`:

```rust
/// Bit-packed atom location address (zone + word + site).
///
/// Layout: `[zone_id:8][word_id:16][site_id:16][pad:24]`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LocationAddr {
    pub zone_id: u32,
    pub word_id: u32,
    pub site_id: u32,
}

impl LocationAddr {
    pub fn encode(&self) -> u64 {
        ((self.zone_id as u8 as u64) << 56)
            | ((self.word_id as u16 as u64) << 40)
            | ((self.site_id as u16 as u64) << 24)
    }

    pub fn decode(bits: u64) -> Self {
        Self {
            zone_id: ((bits >> 56) & 0xFF) as u32,
            word_id: ((bits >> 40) & 0xFFFF) as u32,
            site_id: ((bits >> 24) & 0xFFFF) as u32,
        }
    }
}
```

- [ ] **Step 5: Implement LaneAddr with zone_id**

Rewrite `LaneAddr`:

```rust
/// Bit-packed lane address for atom move operations.
///
/// Layout:
/// - data0: `[word_id:16][site_id:16]` (unchanged)
/// - data1: `[dir:1][mt:2][zone_id:8][pad:5][bus_id:16]`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LaneAddr {
    pub direction: Direction,
    pub move_type: MoveType,
    pub zone_id: u32,
    pub word_id: u32,
    pub site_id: u32,
    pub bus_id: u32,
}

impl LaneAddr {
    pub fn encode(&self) -> (u32, u32) {
        let data0 = ((self.word_id as u16 as u32) << 16)
            | (self.site_id as u16 as u32);
        let data1 = ((self.direction as u32) << 31)
            | ((self.move_type as u32) << 29)
            | ((self.zone_id as u8 as u32) << 21)
            | (self.bus_id as u16 as u32);
        (data0, data1)
    }

    pub fn decode(data0: u32, data1: u32) -> Self {
        let direction = if (data1 >> 31) & 1 == 0 {
            Direction::Forward
        } else {
            Direction::Backward
        };
        let mt_bits = (data1 >> 29) & 0x3;
        let move_type = match mt_bits {
            0 => MoveType::SiteBus,
            1 => MoveType::WordBus,
            2 => MoveType::ZoneBus,
            _ => panic!("invalid move type bits: {}", mt_bits),
        };
        Self {
            direction,
            move_type,
            zone_id: ((data1 >> 21) & 0xFF) as u32,
            word_id: (data0 >> 16) & 0xFFFF,
            site_id: data0 & 0xFFFF,
            bus_id: data1 & 0xFFFF,
        }
    }

    pub fn encode_u64(&self) -> u64 {
        let (d0, d1) = self.encode();
        (d0 as u64) | ((d1 as u64) << 32)
    }

    pub fn decode_u64(bits: u64) -> Self {
        Self::decode(bits as u32, (bits >> 32) as u32)
    }
}
```

- [ ] **Step 6: Add Serialize/Deserialize impls for LocationAddr**

`LocationAddr` is used in `Mode.bitstring_order: Vec<LocationAddr>` which derives Serialize/Deserialize. Implement custom serde for LocationAddr as 64-bit hex string (e.g., `"0x0500123456780000"`):

```rust
impl Serialize for LocationAddr {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u64(self.encode())
    }
}

impl<'de> Deserialize<'de> for LocationAddr {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let bits = u64::deserialize(deserializer)?;
        Ok(Self::decode(bits))
    }
}
```

- [ ] **Step 7: Remove or repurpose ZoneAddr**

The existing `ZoneAddr` type (32-bit `[pad:16][zone_id:16]`) is now superseded — `LocationAddr` carries zone_id directly, and `ZonedWordRef` is used in bus topology. Remove `ZoneAddr` from `addr.rs` if no bytecode instruction still references it. If `ConstZone` bytecode instruction still needs a standalone zone address, keep `ZoneAddr` but update it to use the same 8-bit zone_id width. Check `bytecode/instruction.rs` for `ConstZone` usage.

- [ ] **Step 8: Fix all existing tests in addr.rs for new layouts**

Update existing round-trip tests to include `zone_id` field. Remove the old `test_location_addr_round_trip` and `test_location_addr_zero` tests (replaced by new 64-bit versions). Update `test_lane_addr_round_trip` and `test_lane_addr_forward_sitebus` to include `zone_id: 0`.

- [ ] **Step 9: Run all addr.rs tests**

Run: `cargo test -p bloqade-lanes-bytecode-core -- addr`
Expected: PASS

- [ ] **Step 10: Commit**

```bash
git add crates/bloqade-lanes-bytecode-core/src/arch/addr.rs
git commit -m "feat(core)!: widen LocationAddr to 64-bit, expand MoveType, add zone_id to LaneAddr"
```

### Task 3: Generic Bus\<T\> and restructured Zone/Word/Mode/ArchSpec

**Files:**
- Modify: `crates/bloqade-lanes-bytecode-core/src/arch/types.rs`

- [ ] **Step 1: Write tests for generic Bus\<T\> serde**

Add tests at bottom of `types.rs`:

```rust
#[test]
fn test_site_bus_serde() {
    let bus: Bus<SiteRef> = Bus {
        src: vec![SiteRef(0), SiteRef(1)],
        dst: vec![SiteRef(3), SiteRef(4)],
    };
    let json = serde_json::to_string(&bus).unwrap();
    let deserialized: Bus<SiteRef> = serde_json::from_str(&json).unwrap();
    assert_eq!(bus.src, deserialized.src);
    assert_eq!(bus.dst, deserialized.dst);
}

#[test]
fn test_zone_bus_serde() {
    let bus: Bus<ZonedWordRef> = Bus {
        src: vec![ZonedWordRef { zone_id: 0, word_id: 1 }],
        dst: vec![ZonedWordRef { zone_id: 1, word_id: 1 }],
    };
    let json = serde_json::to_string(&bus).unwrap();
    let deserialized: Bus<ZonedWordRef> = serde_json::from_str(&json).unwrap();
    assert_eq!(bus.src, deserialized.src);
}
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cargo test -p bloqade-lanes-bytecode-core test_site_bus_serde test_zone_bus_serde`
Expected: FAIL — Bus is not generic

- [ ] **Step 3: Implement generic Bus\<T\>**

Replace the existing `Bus` and `Buses` structs in `types.rs`:

```rust
/// Generic transport bus mapping source addresses to destination addresses.
///
/// Parameterized by address type:
/// - `Bus<SiteRef>`: intra-word site movement
/// - `Bus<WordRef>`: intra-zone word movement
/// - `Bus<ZonedWordRef>`: inter-zone movement
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Bus<T> {
    pub src: Vec<T>,
    pub dst: Vec<T>,
}
```

Remove `Buses` struct entirely.

- [ ] **Step 4: Implement restructured Word**

Replace existing `Word`:

```rust
/// A logical grouping of sites within the zone grid.
///
/// Words are a global slicing template shared across all zones.
/// Each entry is `[x_index, y_index]` into the parent zone's Grid.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Word {
    pub sites: Vec<[u32; 2]>,
}
```

- [ ] **Step 5: Implement restructured Zone**

Replace existing `Zone`:

```rust
/// A self-contained architectural unit with its own coordinate system
/// and internal bus connectivity.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Zone {
    pub grid: Grid,
    pub site_buses: Vec<Bus<SiteRef>>,
    pub word_buses: Vec<Bus<WordRef>>,
    pub words_with_site_buses: Vec<u32>,
    pub sites_with_word_buses: Vec<u32>,
}
```

- [ ] **Step 6: Implement Mode struct**

Add new struct:

```rust
/// Measurement configuration specifying which zones are imaged
/// and the bit-position-to-site mapping.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Mode {
    pub name: String,
    pub zones: Vec<u32>,
    pub bitstring_order: Vec<LocationAddr>,
}
```

Note: `LocationAddr` needs `Serialize`/`Deserialize` derives — implement these as the 64-bit hex encoding.

- [ ] **Step 7: Implement restructured ArchSpec**

Replace existing `ArchSpec`:

```rust
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArchSpec {
    pub version: Version,
    pub words: Vec<Word>,
    pub zones: Vec<Zone>,
    pub zone_buses: Vec<Bus<ZonedWordRef>>,
    pub entangling_zone_pairs: Vec<[u32; 2]>,
    pub modes: Vec<Mode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub paths: Option<Vec<TransportPath>>,
    #[serde(default)]
    pub feed_forward: bool,
    #[serde(default)]
    pub atom_reloading: bool,
}

impl ArchSpec {
    /// Derived sites_per_word from the global word template.
    pub fn sites_per_word(&self) -> usize {
        self.words.first().map_or(0, |w| w.sites.len())
    }
}
```

- [ ] **Step 8: Run bus serde tests**

Run: `cargo test -p bloqade-lanes-bytecode-core test_site_bus_serde test_zone_bus_serde`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add crates/bloqade-lanes-bytecode-core/src/arch/types.rs
git commit -m "feat(core)!: generic Bus<T>, zone-centric Zone/Word/ArchSpec, Mode struct"
```

### Task 4: Update module exports

**Files:**
- Modify: `crates/bloqade-lanes-bytecode-core/src/arch/mod.rs`

- [ ] **Step 1: Update pub use statements**

Add new exports, remove old ones:

```rust
pub use addr::{SiteRef, WordRef, ZonedWordRef};
pub use types::Mode;
// Remove: pub use types::{Buses, Geometry};
```

- [ ] **Step 2: Run `cargo check -p bloqade-lanes-bytecode-core`**

Expected: Compilation errors in `validate.rs`, `query.rs`, and downstream crates — these are expected and will be fixed in Chunk 2.

- [ ] **Step 3: Commit**

```bash
git add crates/bloqade-lanes-bytecode-core/src/arch/mod.rs
git commit -m "refactor(core): update arch module exports for zone-centric model"
```

---

## Chunk 2: Rust Validation and Query

### Task 5: Rewrite validation for zone-centric model

**Files:**
- Modify: `crates/bloqade-lanes-bytecode-core/src/arch/validate.rs`

- [ ] **Step 1: Write `make_valid_two_zone_spec()` test helper**

This helper is used by all subsequent validation tests. Add it first so tests compile:

```rust
#[cfg(test)]
fn make_valid_two_zone_spec() -> ArchSpec {
    let grid0 = Grid::from_positions(&[0.0, 5.0, 10.0], &[0.0, 3.0]);
    let grid1 = Grid::from_positions(&[0.0, 7.5, 15.0], &[0.0, 4.0]);

    ArchSpec {
        version: Version { major: 2, minor: 0 },
        words: vec![
            Word { sites: vec![[0, 0], [0, 1]] },
            Word { sites: vec![[1, 0], [1, 1]] },
        ],
        zones: vec![
            Zone {
                grid: grid0,
                site_buses: vec![Bus { src: vec![SiteRef(0)], dst: vec![SiteRef(1)] }],
                word_buses: vec![Bus { src: vec![WordRef(0)], dst: vec![WordRef(1)] }],
                words_with_site_buses: vec![0, 1],
                sites_with_word_buses: vec![0],
            },
            Zone {
                grid: grid1,
                site_buses: vec![],
                word_buses: vec![],
                words_with_site_buses: vec![],
                sites_with_word_buses: vec![],
            },
        ],
        zone_buses: vec![Bus {
            src: vec![ZonedWordRef { zone_id: 0, word_id: 0 }],
            dst: vec![ZonedWordRef { zone_id: 1, word_id: 0 }],
        }],
        entangling_zone_pairs: vec![[0, 1]],
        modes: vec![Mode {
            name: "full".to_string(),
            zones: vec![0, 1],
            bitstring_order: vec![],
        }],
        paths: None,
        feed_forward: false,
        atom_reloading: false,
    }
}
```

- [ ] **Step 2: Update ArchSpecError variants**

Replace zone/geometry/bus error variants to match new model:

```rust
pub enum ArchSpecError {
    /// Structural invariant violation (uniform dimensions, word consistency).
    Structure(String),
    /// Per-zone bus validation failure.
    ZoneBus(String),
    /// Inter-zone bus validation failure.
    InterZoneBus(String),
    /// Grid invariant violation (bus src/dst not rectangular).
    GridInvariant(String),
    /// Entangling zone pair validation failure.
    EntanglingPair(String),
    /// Mode validation failure.
    Mode(String),
    /// Transport path validation failure.
    Path(String),
}
```

- [ ] **Step 2: Write test for uniform zone dimension validation**

```rust
#[test]
fn test_validate_zones_must_have_same_grid_dimensions() {
    let mut spec = make_valid_two_zone_spec();
    // Make zone 1 grid have different x count
    spec.zones[1].grid = Grid::from_positions(
        vec![0.0, 1.0, 2.0, 3.0],  // 4 x-points vs zone 0's 3
        vec![0.0, 1.0],
    );
    let result = spec.validate();
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), ArchSpecError::Structure(_)));
}
```

- [ ] **Step 3: Write test for per-zone site bus validation**

```rust
#[test]
fn test_validate_site_bus_ref_out_of_range() {
    let mut spec = make_valid_two_zone_spec();
    spec.zones[0].site_buses = vec![Bus {
        src: vec![SiteRef(0)],
        dst: vec![SiteRef(999)],  // out of range
    }];
    let result = spec.validate();
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), ArchSpecError::ZoneBus(_)));
}
```

- [ ] **Step 4: Write test for zone bus cross-zone requirement**

```rust
#[test]
fn test_validate_zone_bus_must_cross_zones() {
    let mut spec = make_valid_two_zone_spec();
    spec.zone_buses = vec![Bus {
        src: vec![ZonedWordRef { zone_id: 0, word_id: 0 }],
        dst: vec![ZonedWordRef { zone_id: 0, word_id: 1 }],  // same zone!
    }];
    let result = spec.validate();
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), ArchSpecError::InterZoneBus(_)));
}
```

- [ ] **Step 5: Write test for entangling zone pair validation**

```rust
#[test]
fn test_validate_entangling_zone_pair_invalid_zone() {
    let mut spec = make_valid_two_zone_spec();
    spec.entangling_zone_pairs = vec![[0, 99]];  // zone 99 doesn't exist
    let result = spec.validate();
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), ArchSpecError::EntanglingPair(_)));
}
```

- [ ] **Step 6: Write test for mode validation**

```rust
#[test]
fn test_validate_mode_invalid_zone() {
    let mut spec = make_valid_two_zone_spec();
    spec.modes = vec![Mode {
        name: "bad".to_string(),
        zones: vec![99],  // doesn't exist
        bitstring_order: vec![],
    }];
    let result = spec.validate();
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), ArchSpecError::Mode(_)));
}
```

- [ ] **Step 8: Implement all validation checks**

Rewrite `validate.rs` implementing:
1. `check_uniform_zone_dimensions()` — all zones same grid nx/ny
2. `check_uniform_sites_per_word()` — all words same site count
3. `check_word_site_indices()` — word indices within grid bounds
4. `check_zone_site_buses()` — per-zone SiteRef < sites_per_word
5. `check_zone_word_buses()` — per-zone WordRef < num_words
6. `check_zone_bus_membership()` — words_with_site_buses valid
7. `check_zone_buses()` — ZonedWordRef valid, cross-zone requirement
8. `check_entangling_zone_pairs()` — valid zone indices
9. `check_modes()` — valid zone indices and LocationAddr values
10. `check_path_lanes()` — updated for new LaneAddr

- [ ] **Step 9: Run all validation tests**

Run: `cargo test -p bloqade-lanes-bytecode-core -- validate`
Expected: PASS

- [ ] **Step 10: Commit**

```bash
git add crates/bloqade-lanes-bytecode-core/src/arch/validate.rs
git commit -m "feat(core)!: zone-centric validation rules"
```

### Task 6: Rewrite query methods

**Files:**
- Modify: `crates/bloqade-lanes-bytecode-core/src/arch/query.rs`

- [ ] **Step 1: Write test for JSON round-trip with zone-centric model**

```rust
#[test]
fn test_archspec_json_round_trip() {
    let spec = make_valid_two_zone_spec();
    let json = serde_json::to_string_pretty(&spec).unwrap();
    let deserialized: ArchSpec = serde_json::from_str(&json).unwrap();
    assert_eq!(spec, deserialized);
}
```

- [ ] **Step 2: Write test for zone-aware position lookup**

```rust
#[test]
fn test_location_position_with_zone() {
    let spec = make_valid_two_zone_spec();
    // Zone 0, Word 0, Site 0 → grid0.x[0], grid0.y[0] = (0.0, 0.0)
    let pos = spec.location_position(&LocationAddr { zone_id: 0, word_id: 0, site_id: 0 });
    assert!(pos.is_some());
    let (x, y) = pos.unwrap();
    assert!((x - 0.0).abs() < 1e-10);
    assert!((y - 0.0).abs() < 1e-10);

    // Zone 1, Word 0, Site 0 → grid1.x[0], grid1.y[0] = (0.0, 0.0)
    // Zone 1, Word 1, Site 1 → grid1.x[1], grid1.y[1] = (7.5, 4.0)
    let pos2 = spec.location_position(&LocationAddr { zone_id: 1, word_id: 1, site_id: 1 });
    assert!(pos2.is_some());
    let (x2, y2) = pos2.unwrap();
    assert!((x2 - 7.5).abs() < 1e-10);
    assert!((y2 - 4.0).abs() < 1e-10);
}
```

- [ ] **Step 3: Update query methods**

Key methods to rewrite:
- `location_position()` — resolve via `zones[zone_id].grid` + `words[word_id].sites[site_id]`
- `word_by_id()` — now indexes into global `self.words`
- `zone_by_id()` — indexes into `self.zones`
- `lane_endpoints()` — zone-aware resolution using LaneAddr.zone_id
- `from_json()` / `from_json_validated()` — updated for new schema
- `check_location()` — validate zone_id, word_id, site_id ranges
- `check_lane()` — validate zone_id, bus_id ranges, move_type

Remove methods that no longer apply:
- `site_bus_by_id()`, `word_bus_by_id()` — these become zone-scoped
- `get_blockaded_location()` — replaced by Python-side `get_cz_partner()`

- [ ] **Step 4: Run query tests**

Run: `cargo test -p bloqade-lanes-bytecode-core -- query`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/bloqade-lanes-bytecode-core/src/arch/query.rs
git commit -m "feat(core)!: zone-aware query methods and JSON round-trip"
```

### Task 7: Fix compilation across entire core crate

**Files:**
- Modify: `crates/bloqade-lanes-bytecode-core/src/bytecode/instruction.rs`
- Modify: `crates/bloqade-lanes-bytecode-core/src/bytecode/value.rs`
- Modify: `crates/bloqade-lanes-bytecode-core/src/bytecode/text.rs`
- Modify: `crates/bloqade-lanes-bytecode-core/src/bytecode/encode.rs`
- Modify: `crates/bloqade-lanes-bytecode-core/src/bytecode/validate.rs`
- Modify: `crates/bloqade-lanes-bytecode-core/src/atom_state.rs`

- [ ] **Step 1: Update instruction.rs**

`LaneConstInstruction::ConstLoc` changes from `u32` to `u64`:

```rust
ConstLoc(u64),  // was u32
```

- [ ] **Step 2: Update value.rs**

`DeviceValue::Location` changes from `u32` to `u64`:

```rust
Location(u64),  // was u32
```

- [ ] **Step 3: Update text.rs**

Update address parsing/printing to handle 64-bit LocationAddr and 2-bit MoveType. The zone_id must be parsed from the hex representation.

- [ ] **Step 4: Update encode.rs**

Update binary instruction encoding for the wider LocationAddr. The 16-byte instruction format may need adjustment for the 64-bit location — check if the `data` fields have room.

- [ ] **Step 5: Update bytecode validate.rs**

Update address validation to check zone_id bounds against the ArchSpec's zone count.

- [ ] **Step 6: Update atom_state.rs**

Update `AtomStateData` to use the new LocationAddr (with zone_id) for qubit-to-location mappings. Update `apply_moves()` to pass zone context to lane resolution.

- [ ] **Step 7: Run full crate tests**

Run: `cargo test -p bloqade-lanes-bytecode-core`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add crates/bloqade-lanes-bytecode-core/
git commit -m "refactor(core): update bytecode and atom_state for zone-centric addresses"
```

---

## Chunk 3: PyO3 Bindings and C FFI

### Task 8: Update PyO3 arch bindings

**Files:**
- Modify: `crates/bloqade-lanes-bytecode-python/src/arch_python.rs`
- Modify: `crates/bloqade-lanes-bytecode-python/src/lib.rs`

- [ ] **Step 1: Add monomorphized bus wrappers**

```rust
/// Type aliases for monomorphized bus types.
pub type SiteBus = Bus<SiteRef>;
pub type WordBus = Bus<WordRef>;
pub type ZoneBus = Bus<ZonedWordRef>;

#[pyclass(name = "SiteBus")]
#[derive(Clone)]
pub struct PySiteBus(pub SiteBus);

#[pymethods]
impl PySiteBus {
    #[getter]
    fn src(&self) -> Vec<u16> { self.0.src.iter().map(|s| s.0).collect() }
    #[getter]
    fn dst(&self) -> Vec<u16> { self.0.dst.iter().map(|s| s.0).collect() }
}

#[pyclass(name = "WordBus")]
#[derive(Clone)]
pub struct PyWordBus(pub WordBus);

#[pymethods]
impl PyWordBus {
    #[getter]
    fn src(&self) -> Vec<u16> { self.0.src.iter().map(|w| w.0).collect() }
    #[getter]
    fn dst(&self) -> Vec<u16> { self.0.dst.iter().map(|w| w.0).collect() }
}

#[pyclass(name = "ZoneBus")]
#[derive(Clone)]
pub struct PyZoneBus(pub ZoneBus);

#[pymethods]
impl PyZoneBus {
    #[getter]
    fn src(&self) -> Vec<(u8, u16)> {
        self.0.src.iter().map(|z| (z.zone_id, z.word_id)).collect()
    }
    #[getter]
    fn dst(&self) -> Vec<(u8, u16)> {
        self.0.dst.iter().map(|z| (z.zone_id, z.word_id)).collect()
    }
}
```

- [ ] **Step 2: Update PyZone to include grid and buses**

```rust
#[pymethods]
impl PyZone {
    #[getter]
    fn grid(&self) -> PyGrid { PyGrid(self.0.grid.clone()) }
    #[getter]
    fn site_buses(&self) -> Vec<PySiteBus> {
        self.0.site_buses.iter().map(|b| PySiteBus(b.clone())).collect()
    }
    #[getter]
    fn word_buses(&self) -> Vec<PyWordBus> {
        self.0.word_buses.iter().map(|b| PyWordBus(b.clone())).collect()
    }
    #[getter]
    fn words_with_site_buses(&self) -> Vec<u32> { self.0.words_with_site_buses.clone() }
    #[getter]
    fn sites_with_word_buses(&self) -> Vec<u32> { self.0.sites_with_word_buses.clone() }
}
```

- [ ] **Step 3: Add PyMode wrapper**

```rust
#[pyclass(name = "Mode")]
#[derive(Clone)]
pub struct PyMode(pub Mode);

#[pymethods]
impl PyMode {
    #[getter]
    fn name(&self) -> &str { &self.0.name }
    #[getter]
    fn zones(&self) -> Vec<u32> { self.0.zones.clone() }
    #[getter]
    fn bitstring_order(&self) -> Vec<PyLocationAddr> {
        self.0.bitstring_order.iter().map(|a| PyLocationAddr(*a)).collect()
    }
}
```

- [ ] **Step 4: Update PyLocationAddr for 64-bit**

Add `zone_id` getter, change `encode()` return to `u64`.

- [ ] **Step 5: Update PyLaneAddr with zone_id**

Add `zone_id` getter, constructor parameter.

- [ ] **Step 6: Update PyArchSpec**

Remove old properties (`geometry`, `buses`, `words_with_site_buses`, `sites_with_word_buses`, `entangling_zones`, `measurement_mode_zones`). Add new properties:

```rust
#[getter]
fn words(&self) -> Vec<PyWord> { ... }
#[getter]
fn zones(&self) -> Vec<PyZone> { ... }
#[getter]
fn zone_buses(&self) -> Vec<PyZoneBus> { ... }
#[getter]
fn entangling_zone_pairs(&self) -> Vec<[u32; 2]> { ... }
#[getter]
fn modes(&self) -> Vec<PyMode> { ... }
fn sites_per_word(&self) -> usize { self.0.sites_per_word() }
```

- [ ] **Step 7: Update PyWord to simplified model**

Remove `positions`, `site_indices`, `has_cz`. Add `sites` getter returning `Vec<[u32; 2]>`.

- [ ] **Step 8: Register new types in lib.rs**

Add `PySiteBus`, `PyWordBus`, `PyZoneBus`, `PyMode`, `PyZonedWordRef` to the PyO3 module.

- [ ] **Step 9: Run `cargo check -p bloqade-lanes-bytecode-python`**

Expected: PASS (compilation)

- [ ] **Step 10: Commit**

```bash
git add crates/bloqade-lanes-bytecode-python/
git commit -m "feat(python)!: PyO3 bindings for zone-centric model"
```

### Task 9: Update remaining PyO3 files and Python exceptions

**Files:**
- Modify: `crates/bloqade-lanes-bytecode-python/src/instruction_python.rs`
- Modify: `crates/bloqade-lanes-bytecode-python/src/validation.rs`
- Modify: `crates/bloqade-lanes-bytecode-python/src/atom_state_python.rs`
- Modify: `crates/bloqade-lanes-bytecode-python/src/errors.rs`
- Modify: `python/bloqade/lanes/bytecode/exceptions.py`

- [ ] **Step 1: Update instruction_python.rs** — `ConstLoc` parameter changes from `u32` to `u64`
- [ ] **Step 2: Update validation.rs** — Map new `ArchSpecError` variants (`Structure`, `ZoneBus`, `InterZoneBus`, `GridInvariant`, `EntanglingPair`, `Mode`) to Python exception classes
- [ ] **Step 3: Update atom_state_python.rs** — `LocationAddr` is now 64-bit with zone_id
- [ ] **Step 4: Update errors.rs** — Add conversion functions for new error variants
- [ ] **Step 5: Update exceptions.py** — Add new Python exception classes that the Rust error conversion targets:

```python
class ArchSpecStructureError(ArchSpecError): ...
class ArchSpecZoneBusError(ArchSpecError): ...
class ArchSpecInterZoneBusError(ArchSpecError): ...
class ArchSpecGridInvariantError(ArchSpecError): ...
class ArchSpecEntanglingPairError(ArchSpecError): ...
class ArchSpecModeError(ArchSpecError): ...
```

- [ ] **Step 6: Run `cargo check -p bloqade-lanes-bytecode-python`**

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add crates/bloqade-lanes-bytecode-python/
git commit -m "refactor(python): update remaining PyO3 wrappers for zone-centric model"
```

### Task 10: Update C FFI and CLI

**Files:**
- Modify: `crates/bloqade-lanes-bytecode-cli/src/ffi/arch.rs`
- Modify: `crates/bloqade-lanes-bytecode-cli/src/ffi/validate.rs`
- Modify: `crates/bloqade-lanes-bytecode-cli/src/ffi/handles.rs`

- [ ] **Step 1: Update FFI arch functions** for new ArchSpec layout
- [ ] **Step 2: Update FFI validation** for new error types
- [ ] **Step 3: Regenerate C header**

Run: `just check-header` (or `cbindgen` directly)

- [ ] **Step 4: Run `cargo test -p bloqade-lanes-bytecode-cli`**

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/bloqade-lanes-bytecode-cli/
git commit -m "refactor(cli): update C FFI for zone-centric ArchSpec"
```

### Task 11: Full Rust test suite

- [ ] **Step 1: Run all Rust tests**

Run: `just test-rust`
Expected: PASS

- [ ] **Step 2: Run clippy**

Run: `just lint`
Expected: No warnings

- [ ] **Step 3: Run format check**

Run: `just format-check`
Expected: PASS

- [ ] **Step 4: Commit any fixups**

---

## Chunk 4: Python Layer

### Task 12: Update Python type stubs

**Files:**
- Modify: `python/bloqade/lanes/bytecode/_native.pyi`

- [ ] **Step 1: Update stubs for all changed/new Rust types**

Add stubs for `SiteBus`, `WordBus`, `ZoneBus`, `Mode`, `ZonedWordRef`. Update `LocationAddress` (add `zone_id`), `LaneAddress` (add `zone_id`), `MoveType` (add `ZONE`), `Zone` (add `grid`, `site_buses`, etc.), `Word` (simplified), `ArchSpec` (new properties).

- [ ] **Step 2: Verify stubs are correct**

Run: `uv run pyright python/bloqade/lanes/bytecode/_native.pyi`
Expected: PASS (no type errors in stub file itself)

- [ ] **Step 3: Commit**

```bash
git add python/bloqade/lanes/bytecode/_native.pyi
git commit -m "docs(python): update type stubs for zone-centric model"
```

### Task 13: Update encoding.py

**Files:**
- Modify: `python/bloqade/lanes/layout/encoding.py`

- [ ] **Step 1: Write test for LocationAddress with zone_id**

File: `python/tests/layout/test_encoding.py`

```python
def test_location_address_zone_id():
    addr = LocationAddress(zone_id=2, word_id=3, site_id=4)
    assert addr.zone_id == 2
    assert addr.word_id == 3
    assert addr.site_id == 4
    encoded = addr.encode()
    assert isinstance(encoded, int)
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run coverage run -m pytest python/tests/layout/test_encoding.py::test_location_address_zone_id -v`
Expected: FAIL

- [ ] **Step 3: Update LocationAddress class**

Add `zone_id` parameter to `__init__`, update `_inner` construction to pass zone_id to Rust. Update `__lt__` to compare `(zone_id, word_id, site_id)`.

- [ ] **Step 4: Update LaneAddress class**

Add `zone_id` property and parameter.

- [ ] **Step 5: Add MoveType.ZONE variant handling**

Verify the Rust `MoveType::ZoneBus` is exposed as `MoveType.ZONE` in Python.

- [ ] **Step 6: Add ZonedWordRef Python class**

The spec requires `ZonedWordRef` exposed as a Python class. Add to `encoding.py`:

```python
class ZonedWordRef(Encoder):
    """Zone-qualified word reference for inter-zone bus entries."""
    _inner: _RustZonedWordRef

    def __init__(self, zone_id: int, word_id: int):
        self._inner = _RustZonedWordRef(zone_id, word_id)
        self.__post_init__()

    @property
    def zone_id(self) -> int:
        return self._inner.zone_id

    @property
    def word_id(self) -> int:
        return self._inner.word_id
```

- [ ] **Step 7: Run tests**

Run: `uv run coverage run -m pytest python/tests/layout/test_encoding.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add python/bloqade/lanes/layout/encoding.py python/tests/layout/test_encoding.py
git commit -m "feat(python)!: zone-aware LocationAddress, LaneAddress, ZonedWordRef"
```

### Task 14: Update Word wrapper

**Files:**
- Modify: `python/bloqade/lanes/layout/word.py`
- Create: `python/tests/layout/test_word.py` (does not exist yet)

- [ ] **Step 1: Write test for simplified Word**

Create `python/tests/layout/test_word.py`:

```python
from bloqade.lanes.layout.word import Word

def test_word_sites():
    word = Word(sites=((0, 0), (0, 1), (1, 0), (1, 1)))
    assert len(word.sites) == 4
    assert word.sites[0] == (0, 0)

def test_word_sites_per_word():
    word = Word(sites=((0, 0), (0, 1)))
    assert word.sites_per_word == 2
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run coverage run -m pytest python/tests/layout/test_word.py -v`
Expected: FAIL — Word constructor/API doesn't match

- [ ] **Step 3: Simplify Word class**

Rewrite `word.py`: Remove `positions: grid.Grid`, `site_position()`, `all_positions()`, `plot()`, `cz_pair`, any geometry ownership. Replace with:

```python
@dataclass(frozen=True)
class Word:
    """A logical grouping of sites as grid index pairs (x_idx, y_idx)."""
    sites: tuple[tuple[int, int], ...]

    @property
    def sites_per_word(self) -> int:
        return len(self.sites)
```

Position resolution now requires the parent zone's grid and lives on ArchSpec.

- [ ] **Step 4: Run tests**

Run: `uv run coverage run -m pytest python/tests/layout/test_word.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/layout/word.py python/tests/layout/test_word.py
git commit -m "refactor(python)!: simplify Word to grid index pairs"
```

### Task 15: Update ArchSpec wrapper

**Files:**
- Modify: `python/bloqade/lanes/layout/arch.py`
- Test: `python/tests/layout/test_arch.py`

- [ ] **Step 1: Create `make_two_zone_arch()` test helper**

Create a helper in `python/tests/layout/conftest.py` (or `test_arch.py`) that builds a minimal 2-zone ArchSpec via the Rust bindings — 2 zones, 2 words, 2 sites per word, matching the Rust test helper structure.

- [ ] **Step 2: Write test for get_cz_partner**

```python
def test_get_cz_partner():
    arch = make_two_zone_arch()
    loc = LocationAddress(zone_id=0, word_id=0, site_id=0)
    partner = arch.get_cz_partner(loc)
    assert partner is not None
    assert partner.zone_id == 1
    assert partner.word_id == 0
    assert partner.site_id == 0
```

- [ ] **Step 3: Write test for get_zone_grid**

```python
def test_get_zone_grid():
    arch = make_two_zone_arch()
    grid = arch.get_zone_grid(0)
    assert grid is not None
```

- [ ] **Step 4: Rewrite ArchSpec wrapper**

Key changes:
- Remove `site_buses`, `word_buses`, `has_site_buses`, `has_word_buses` properties (now per-zone)
- Remove `entangling_zones`, `measurement_mode_zones` properties
- Remove `get_blockaded_location()` method
- Add `get_cz_partner(loc) -> Optional[LocationAddress]`
- Add `get_zone_grid(zone_id) -> Grid`
- Add `get_available_buses(zone_id) -> list[BusDescriptor]`
- Add `get_grid_endpoints(zone_id, bus_id, move_type, direction) -> tuple[Grid, Grid]`
- Update `__post_init__` to build zone-indexed lookups
- Update `_lane_map` construction to iterate per-zone buses + zone buses

- [ ] **Step 5: Run tests**

Run: `uv run coverage run -m pytest python/tests/layout/test_arch.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add python/bloqade/lanes/layout/arch.py python/tests/layout/test_arch.py python/tests/layout/conftest.py
git commit -m "feat(python)!: zone-centric ArchSpec wrapper with get_cz_partner, get_zone_grid"
```

### Task 16: Update PathFinder

**Files:**
- Modify: `python/bloqade/lanes/layout/path.py`

- [ ] **Step 1: Update graph construction** to iterate per-zone buses + zone buses
- [ ] **Step 2: Add zone bus edges** — for each zone bus, create edges between the ZonedWordRef endpoints
- [ ] **Step 3: Update LaneAddress construction** to include zone_id
- [ ] **Step 4: Run PathFinder tests**
- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/layout/path.py python/tests/layout/
git commit -m "feat(python): zone-aware PathFinder with inter-zone edges"
```

---

## Chunk 5: Python Builder, Consumers, Schema, and Docs

### Task 17: Update builder (build_arch)

**Files:**
- Modify: `python/bloqade/lanes/arch/zone.py`
- Modify: `python/bloqade/lanes/arch/builder.py`
- Modify: `python/bloqade/lanes/arch/word_factory.py`
- Modify: `python/bloqade/lanes/arch/topology.py`
- Test: `python/tests/arch/`

- [ ] **Step 1: Update ZoneSpec** for per-zone layout overrides
- [ ] **Step 2: Update DeviceLayout** for per-zone spacing
- [ ] **Step 3: Add dimension validation** — all ZoneSpecs must produce same grid dimensions
- [ ] **Step 4: Update word_factory** — `create_zone_words()` returns global word template with grid index pairs
- [ ] **Step 5: Update topology** — `InterZoneTopology` produces `Bus<ZonedWordRef>` entries
- [ ] **Step 6: Rewrite build_arch()** to produce zone-centric ArchSpec:
  - Global word template
  - Per-zone Grid with zone-specific spacing
  - Per-zone site_buses and word_buses
  - zone_buses from InterZoneTopology
  - entangling_zone_pairs from zones with `entangling=True`
  - Mode entries from zones with `measurement=True`
  - Remove Zone 0 = "all words" convention
- [ ] **Step 7: Run builder tests**

Run: `uv run coverage run -m pytest python/tests/arch/ -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add python/bloqade/lanes/arch/
git commit -m "feat(python)!: zone-centric build_arch with per-zone grids and buses"
```

### Task 18: Update Gemini architecture specs

**Files:**
- Modify: `python/bloqade/lanes/arch/gemini/impls.py`
- Modify: `python/bloqade/lanes/arch/gemini/logical/spec.py`
- Modify: `python/bloqade/lanes/arch/gemini/physical/spec.py`

- [ ] **Step 1: Update Gemini specs** to use zone-centric builder output
- [ ] **Step 2: Remove `get_cz_pair()`** if it exists — replaced by entangling_zone_pairs
- [ ] **Step 3: Run Gemini tests**

Run: `uv run coverage run -m pytest python/tests/arch/gemini/ -v`

- [ ] **Step 4: Commit**

```bash
git add python/bloqade/lanes/arch/gemini/
git commit -m "refactor(python): update Gemini architecture for zone-centric model"
```

### Task 19: Update analysis and heuristic consumers

**Files:**
- Modify: `python/bloqade/lanes/analysis/placement/lattice.py`
- Modify: `python/bloqade/lanes/heuristics/logical_placement.py`
- Modify: `python/bloqade/lanes/heuristics/move_synthesis.py`
- Audit: `python/bloqade/lanes/heuristics/physical_layout.py` (may not need changes — audit first)
- Audit: `python/bloqade/lanes/heuristics/physical_movement.py` (may not need changes — audit first)

Update pattern: replace `has_cz` / `get_blockaded_location()` with `arch.get_cz_partner()`, access buses through `arch.zones[zone_id].site_buses`, use `arch.entangling_zone_pairs` instead of `arch.entangling_zones`.

- [ ] **Step 1: Update lattice.py** — `ExecuteCZ` uses `get_cz_partner()` instead of `get_blockaded_location()`
- [ ] **Step 2: Update logical_placement.py** — CZ layout uses zone pairs instead of word-level `has_cz`
- [ ] **Step 3: Update move_synthesis.py** — zone-aware bus access via `arch.zones[zone_id]`
- [ ] **Step 4: Audit physical_layout.py and physical_movement.py** — grep for `has_cz`, `site_buses`, `word_buses`, `entangling_zones`, `measurement_mode_zones`. Only modify if they reference removed APIs.
- [ ] **Step 5: Run analysis/heuristics tests**

Run: `uv run coverage run -m pytest python/tests/analysis/ python/tests/heuristics/ -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add python/bloqade/lanes/analysis/ python/bloqade/lanes/heuristics/
git commit -m "refactor(python)!: update analysis and heuristics for zone-centric ArchSpec"
```

### Task 20: Update rewrite passes and dialects

**Files:**
- Modify: `python/bloqade/lanes/rewrite/place2move.py`
- Modify: `python/bloqade/lanes/rewrite/move2squin/noise.py`
- Modify: `python/bloqade/lanes/dialects/move.py`

- [ ] **Step 1: Update place2move.py** — CZ gate rewriting uses zone-pair entangling model
- [ ] **Step 2: Update noise.py** — zone-aware noise model, access buses through zone
- [ ] **Step 3: Update move.py dialect** — zone bus lane addresses, MoveType.ZoneBus handling
- [ ] **Step 4: Run rewrite/dialect tests**

Run: `uv run coverage run -m pytest python/tests/rewrite/ python/tests/dialects/ -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/rewrite/ python/bloqade/lanes/dialects/
git commit -m "refactor(python)!: update rewrite passes and dialects for zone-centric ArchSpec"
```

### Task 21: Update JSON schema and examples

**Files:**
- Modify: `docs/src/arch/archspec-schema.json`
- Modify: `examples/arch/simple.json`
- Modify: `examples/arch/full.json`
- Modify: `examples/arch/gemini-logical.json`

- [ ] **Step 1: Rewrite archspec-schema.json** for version 2.0 zone-centric format
- [ ] **Step 2: Rewrite simple.json** with single zone, grid, per-zone buses
- [ ] **Step 3: Rewrite full.json** with zones, entangling_zone_pairs, modes
- [ ] **Step 4: Rewrite gemini-logical.json**
- [ ] **Step 5: Run CLI smoke test**

Run: `just cli-smoke-test`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add docs/src/arch/archspec-schema.json examples/arch/
git commit -m "docs(arch)!: update JSON schema and examples to version 2.0"
```

### Task 22: Update documentation

**Files:**
- Modify: `docs/src/arch/archspec.md`
- Modify: `docs/src/bytecode/inst-spec.md`

- [ ] **Step 1: Rewrite archspec.md** — zone-centric model, bus hierarchy, address encoding, validation rules
- [ ] **Step 2: Update inst-spec.md** — 64-bit LocationAddr, 2-bit MoveType, zone_id in LaneAddr
- [ ] **Step 3: Build docs**

Run: `just doc-book`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add docs/src/
git commit -m "docs: update ArchSpec and instruction spec for zone-centric model"
```

### Task 23: Full integration test

- [ ] **Step 1: Run all Rust tests**

Run: `just test-rust`
Expected: PASS

- [ ] **Step 2: Build Python extension**

Run: `just develop-python`
Expected: PASS

- [ ] **Step 3: Run all Python tests**

Run: `just test-python`
Expected: PASS

- [ ] **Step 4: Run full test suite**

Run: `just test`
Expected: PASS

- [ ] **Step 5: Run linters**

Run: `just lint && uv run black python && uv run isort python && uv run ruff check python && uv run pyright python`
Expected: PASS

- [ ] **Step 6: Run demos**

Run: `just demo`
Expected: PASS (or update demos if needed)

- [ ] **Step 7: Final commit if any fixups needed**

### Task 24: Create PR

- [ ] **Step 1: Create PR with `breaking` label**

Per project conventions, this PR carries the `breaking` label since it changes all API surfaces:

```bash
gh pr create --title "feat(arch)!: zone-centric ArchSpec redesign" --label "category: breaking change" --label "category: feature" --label "area: Lane" --body "$(cat <<'EOF'
## Summary

- Restructure ArchSpec so zones are the primary structural unit, each owning a Grid and bus connectivity
- Introduce three-tier bus hierarchy: Bus<SiteRef> (intra-word), Bus<WordRef> (intra-zone), Bus<ZonedWordRef> (inter-zone)
- Widen LocationAddr to 64-bit with zone_id, expand MoveType to 2-bit with ZoneBus variant
- Replace entangling_zones/has_cz with entangling_zone_pairs (zone-pair CZ model)
- Replace measurement_mode_zones with explicit Mode structs
- Update builder, schema, docs, and all downstream consumers

Closes #422

## Test plan
- [ ] `just test-rust` passes
- [ ] `just test-python` passes
- [ ] `just cli-smoke-test` passes
- [ ] `just demo` passes
- [ ] `just lint` passes

BREAKING CHANGE: Complete restructuring of ArchSpec data model across Rust, Python, and C FFI surfaces. See spec at docs/superpowers/specs/2026-04-06-zone-centric-archspec-design.md.
EOF
)"
```
