# ArchSpec — Architecture Specification

The `ArchSpec` defines the physical topology and transport capabilities of a Bloqade quantum device. It is the input that the bytecode compiler and validator use to determine which instructions are legal for a given hardware configuration.

The formal JSON Schema is available at [`archspec-schema.json`](./archspec-schema.json).

## Top-Level Structure

```jsonc
{
  "version": "2.0",
  "words": [...],
  "zones": [...],
  "zone_buses": [...],
  "modes": [...],
  "paths": [...],                 // optional
  "feed_forward": false,          // optional, default false
  "atom_reloading": false,        // optional, default false
  "blockade_radius": 2.0          // optional
}
```

| Field | Type | Description |
|---|---|---|
| `version` | string | Format version as `"major.minor"` (e.g. `"2.0"`). |
| `words` | Word[] | Word definitions. A word's ID is its index in this array. |
| `zones` | Zone[] | Logical zones, each owning a coordinate grid and intra-zone buses. |
| `zone_buses` | InterZoneBus[] | Inter-zone word buses. |
| `modes` | Mode[] | Named operational modes (zone subsets + measurement bitstring ordering). |
| `paths` | TransportPath[] | *(optional)* AOD transport paths for lanes. |
| `feed_forward` | bool | *(optional, default `false`)* Whether the device supports mid-circuit measurement with classical feedback. |
| `atom_reloading` | bool | *(optional, default `false`)* Whether the device supports reloading atoms after initial fill. |
| `blockade_radius` | float | *(optional)* Rydberg blockade radius in micrometers — metadata for interpreting entangling pairs. |

---

## Words

A **word** is an independent register of atom trapping sites. It is the fundamental unit of the device topology. A word's ID is its index in the top-level `words` array (e.g., the first word is word 0).

```jsonc
"words": [
  { "sites": [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]] }
]
```

| Field | Type | Description |
|---|---|---|
| `sites` | [x_idx, y_idx][] | Site positions as index pairs into the owning zone grid's x and y coordinate arrays. |

All words must have the same number of sites (`sites_per_word` is derived as the site count of the first word), and every site's `[x, y]` indices must lie within the zone grid.

---

## Zones

A **zone** is a logical region owning a coordinate grid and the transport buses that operate within it. A zone's ID is its index in the `zones` array (e.g., the first zone is zone 0).

```jsonc
"zones": [
  {
    "name": "entangling",
    "grid": {
      "x_start": 1.0,
      "y_start": 2.5,
      "x_spacing": [2.0, 2.0, 2.0, 2.0],
      "y_spacing": [2.5]
    },
    "site_buses": [
      { "src": [0, 1], "dst": [3, 4] }
    ],
    "word_buses": [
      { "src": [0], "dst": [1] }
    ],
    "words_with_site_buses": [0, 1],
    "sites_with_word_buses": [0],
    "entangling_pairs": [[0, 1]]
  }
]
```

| Field | Type | Description |
|---|---|---|
| `name` | string | *(optional, default `""`)* Human-readable zone name. |
| `grid` | Grid | Coordinate grid for all words in this zone. |
| `site_buses` | Bus[] | Site buses moving atoms between sites within words of this zone. |
| `word_buses` | Bus[] | Word buses moving atoms between words within this zone. |
| `words_with_site_buses` | integer[] | Word IDs with site-bus transport capability in this zone. |
| `sites_with_word_buses` | integer[] | Site indices serving as landing pads for word-bus moves. |
| `entangling_pairs` | [w_a, w_b][] | *(optional, default `[]`)* Word pairs at blockade radius for CZ gates. |

### Grid

A **grid** defines the physical coordinate axes for a zone using a start position and spacing values. Positions are typically in micrometers (µm).

| Field | Type | Description |
|---|---|---|
| `x_start` | float | X-coordinate of the first grid point. |
| `y_start` | float | Y-coordinate of the first grid point. |
| `x_spacing` | float[] | Spacing between consecutive x-coordinates. The number of x grid points is `len(x_spacing) + 1`. |
| `y_spacing` | float[] | Spacing between consecutive y-coordinates. The number of y grid points is `len(y_spacing) + 1`. |

The x-coordinates are computed as `[x_start, x_start + x_spacing[0], x_start + x_spacing[0] + x_spacing[1], ...]` (cumulative sum of spacings from the start). Same for y. Sites reference grid positions by index: site `[2, 1]` is located at the 3rd x-coordinate and 2nd y-coordinate. Spacings must be non-negative.

All zones must have the same grid dimensions — i.e., the same number of x and y grid points (same `x_spacing` and `y_spacing` lengths). The actual coordinate values differ (zones are at different physical locations), and zone bounding boxes must not overlap in physical space.

### Entangling Pairs

Each zone's `entangling_pairs` lists which word pairs within it can perform CZ (entangling) gates. Within a pair, sites at matching indices in `w_a` and `w_b` are within blockade radius. A zone with no entangling pairs is a storage/low-connectivity zone.

---

## Buses

Buses are the physical transport channels that move atoms. Each bus defines a paired mapping via parallel arrays: the atom at `src[i]` moves to `dst[i]`, and all pairs of one bus execute **simultaneously as one AOD operation**. There are three kinds:

### Site Bus

A **site bus** moves atoms between sites *within the same word*. Entries are site indices. A site bus's ID is its index in the owning zone's `site_buses` array.

```jsonc
{ "src": [0, 1, 2, 3, 4], "dst": [5, 6, 7, 8, 9] }
```

This means the atom at site 0 moves to site 5, the atom at site 1 moves to site 6, and so on — all in a single transport operation. Only words listed in the zone's `words_with_site_buses` can execute site-bus moves.

### Word Bus

A **word bus** moves atoms between *different words within a zone*. The `src` and `dst` arrays contain word IDs (not site indices). A word bus's ID is its index in the owning zone's `word_buses` array.

```jsonc
{ "src": [0], "dst": [1] }
```

The specific sites involved in inter-word transport are those listed in the zone's `sites_with_word_buses` — the "landing pad" positions within each word.

### Zone Bus

A **zone bus** (top-level `zone_buses`) moves words *across zone boundaries*. Entries are zone-qualified word references, and every `(src[i], dst[i])` pair must have different `zone_id`s.

```jsonc
{
  "src": [{ "zone_id": 0, "word_id": 0 }],
  "dst": [{ "zone_id": 1, "word_id": 0 }]
}
```

### Bus Well-Formedness

For every bus kind, the `src`→`dst` relation must be well-formed: `src` entries unique, `dst` entries unique, and **acyclic** (no rotations, including self-loops) — a bus is a set of explicit transports, never a permutation. Overlapping-but-acyclic relations (conveyor chains such as `0→1, 1→2`) are legal. See [Validation Rules](#validation-rules).

---

## Modes

A **mode** is a named operational configuration: a subset of zones plus the bitstring ordering used for measurement results.

```jsonc
"modes": [
  { "name": "full", "zones": [0, 1], "bitstring_order": [] }
]
```

| Field | Type | Description |
|---|---|---|
| `name` | string | Human-readable mode name. |
| `zones` | integer[] | Zone IDs active in this mode. |
| `bitstring_order` | integer[] | Bit-to-location mapping for measurement results. Each entry is a `LocationAddr` encoded as a packed integer (layout `[zone_id:8][word_id:16][site_id:16][pad:24]`, most-significant first). |

---

## Paths (Optional)

AOD (Acousto-Optic Deflector) transport paths. Each path identifies a transport lane and provides a sequence of `[x, y]` waypoints defining the physical trajectory atoms follow during transport.

The lane is identified by its encoded `LaneAddr`, serialized as a hex string. See [Address Encoding](#address-encoding) for the `LaneAddr` bit layout.

```jsonc
"paths": [
  {
    "lane": "0x2000000000000000",                       // encoded LaneAddr (hex, 16-digit)
    "waypoints": [[0.0, 0.0], [0.0, 5.0], [2.0, 5.0]]   // physical trajectory
  }
]
```

Each `TransportPath` entry has:

| Field | Type | Description |
|---|---|---|
| `lane` | string | Encoded `LaneAddr` as a `"0x..."` hex string. |
| `waypoints` | [x, y][] | Sequence of physical coordinate waypoints (at least 2, all finite). |

To decode the lane hex string, parse it as a 64-bit unsigned integer. The low 32 bits (data0) contain `[word_id:16][site_id:16]` and the high 32 bits (data1) contain `[dir:1][mt:2][zone_id:8][pad:5][bus_id:16]`. For example, `"0x2000000000000000"` has data1 = `0x20000000`: direction=Forward (bit 31 clear), move_type=WordBus (bits 30–29 = `01`), zone=0, word=0, site=0, bus=0. In the lane address convention, `word_id` always encodes the forward-direction source word for that lane, so a `Backward` lane with the same address fields moves the atom from the bus destination back to that source.

This field is omitted from the JSON when not needed.

---

## Capability Flags (Optional)

Two boolean flags describe device capabilities that affect bytecode validation:

| Field | Default | Description |
|---|---|---|
| `feed_forward` | `false` | Mid-circuit measurement with classical feedback. When `false`, at most one `measure` instruction is allowed per program. |
| `atom_reloading` | `false` | Atom reloading after initial fill. When `false`, no `fill` instruction is allowed (only `initial_fill`). |

Both fields are optional in the JSON — existing arch spec files that omit them default to `false`, which is the most restrictive setting.

```jsonc
{
  "feed_forward": true,
  "atom_reloading": false
}
```

---

## Validation Rules

The `ArchSpec::validate()` method checks all structural rules in a single pass, collecting every error rather than failing fast. Errors are grouped into coarse `ArchSpecError` categories, each carrying a descriptive message; the **Error** column below names the category (Rust enum variant). Through the Python bindings the categories map onto exception classes:

| `ArchSpecError` variant | Python exception |
|---|---|
| `Structure` | `ArchSpecGeometryError` |
| `ZoneBus`, `InterZoneBus`, `CyclicBus` | `ArchSpecBusError` |
| `EntanglingPair`, `Mode` | `ArchSpecZoneError` |
| `Path` | `ArchSpecPathError` |

### Structural Rules

| Rule | Error |
|---|---|
| At least one zone and at least one word must exist | `Structure` |
| All grid spacings must be non-negative | `Structure` |
| All zones must have the same grid dimensions (same number of x and y positions) | `Structure` |
| All words must have the same number of sites | `Structure` |
| Every word site's `[x, y]` indices must lie within the zone grid | `Structure` |
| No two zones may have overlapping bounding boxes in physical (x, y) space | `Structure` |

### Per-Zone Bus Rules

| Rule | Error |
|---|---|
| Every ID in `words_with_site_buses` must be a valid word ID | `ZoneBus` |
| Every index in `sites_with_word_buses` must be < `sites_per_word` | `ZoneBus` |
| Site bus `src` and `dst` must have equal length | `ZoneBus` |
| All site bus indices in `src` and `dst` must be < `sites_per_word` | `ZoneBus` |
| Word bus `src` and `dst` must have equal length | `ZoneBus` |
| All word bus IDs in `src` and `dst` must be valid word IDs | `ZoneBus` |

### Inter-Zone Bus Rules

| Rule | Error |
|---|---|
| Zone bus `src` and `dst` must have equal length | `InterZoneBus` |
| All zone bus `zone_id` / `word_id` entries must be in range | `InterZoneBus` |
| Every `(src[i], dst[i])` pair must cross a zone boundary | `InterZoneBus` |

### Bus Well-Formedness Rules (all bus kinds)

A bus is a set of explicit edge transports executed simultaneously as one AOD
operation — never a permutation. These rules apply to site buses, word buses,
and zone buses alike:

| Rule | Error |
|---|---|
| `src` entries must be unique (endpoint resolution is positional first-match, so a duplicated source silently shadows later pairs) | `ZoneBus` / `InterZoneBus` |
| `dst` entries must be unique (two simultaneous transports into one site cannot both complete) | `ZoneBus` / `InterZoneBus` |
| The `src`→`dst` relation must be acyclic, including self-loops — a cycle would rotate a fully-occupied set of atoms with no empty site, which AOD hardware cannot do. Overlapping-but-acyclic relations (conveyor chains such as `0→1, 1→2`) are legal. | `CyclicBus` |

### Entangling Pair Rules

| Rule | Error |
|---|---|
| Both word IDs in every `entangling_pairs` entry must be valid word IDs | `EntanglingPair` |
| A word must not be paired with itself | `EntanglingPair` |
| No duplicate pairs (order-insensitive: `[a, b]` duplicates `[b, a]`) | `EntanglingPair` |

### Mode Rules

| Rule | Error |
|---|---|
| Every zone ID in a mode must reference a defined zone | `Mode` |
| Every `bitstring_order` entry's `zone_id`, `word_id`, and `site_id` must be in range | `Mode` |

### Path Rules

| Rule | Error |
|---|---|
| Waypoint coordinates must be finite (no NaN or Inf) | `Path` |
| Every path's lane must have a valid `zone_id` | `Path` |
| Every path must have at least 2 waypoints | `Path` |

### Capability Rules (Bytecode Validation)

These rules are checked during bytecode validation (`ValidationError`, not `ArchSpecError`) when an `ArchSpec` is provided:

| Rule | Error |
|---|---|
| If `feed_forward = false`, control flow and multiple `measure` instructions are rejected | `ControlFlowRequiresFeedForward`, `MultipleMeasuresRequireFeedForward` |
| If `atom_reloading = false`, no `fill` instruction is allowed (`initial_fill` is a separate instruction and is always permitted) | `FillRequiresAtomReloading` |

---

## Address Encoding

At the bytecode level, locations and lanes are encoded as bit-packed integers with 16-bit address fields. Each address type is packed into instruction data words (u32):

| Type | Width | Layout | Description |
|---|---|---|---|
| `LocationAddr` | 64 bits | `[zone_id:8][word_id:16][site_id:16][pad:24]` (most-significant first) | Identifies a specific site within a word of a zone. |
| `LaneAddr` | 64 bits (2 × u32) | data0 (low): `[word_id:16][site_id:16]`, data1 (high): `[dir:1][mt:2][zone_id:8][pad:5][bus_id:16]` | Identifies a transport lane (direction + move type + zone + word/site + bus). |
| `ZoneAddr` | 32 bits (1 × u32) | `[pad:24][zone_id:8]` | Identifies a zone. |

These packed addresses are used in 16-byte bytecode instructions (opcode + 3 data words) and are validated against the arch spec during program validation. In JSON, `LaneAddr` is represented as a 16-digit hex string (`data0 | data1 << 32`) and `LocationAddr` as a plain integer; in Python both are `u64` values.

---

## Examples

Minimal spec with one word, one zone, and one site bus (this is [`examples/arch/simple.json`](../../examples/arch/simple.json)):

```json
{
  "version": "2.0",
  "words": [
    { "sites": [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]] }
  ],
  "zones": [
    {
      "grid": {
        "x_start": 1.0,
        "y_start": 2.0,
        "x_spacing": [2.0, 2.0, 2.0, 2.0],
        "y_spacing": []
      },
      "site_buses": [
        { "src": [0, 1], "dst": [3, 4] }
      ],
      "word_buses": [],
      "words_with_site_buses": [0],
      "sites_with_word_buses": []
    }
  ],
  "zone_buses": [],
  "modes": [
    { "name": "default", "zones": [0], "bitstring_order": [] }
  ]
}
```

A fuller example with multiple words, CZ pairs, and word buses is available at [`examples/arch/full.json`](../../examples/arch/full.json).
