# ArchSpec — Architecture Specification

The `ArchSpec` defines the physical topology and transport capabilities of a Bloqade quantum device. It is the input that the bytecode compiler and validator use to determine which instructions are legal for a given hardware configuration.

The formal JSON Schema is available at [`archspec-schema.json`](./archspec-schema.json).

## Top-Level Structure

```jsonc
{
  "version": "2.0",
  "geometry": { ... },
  "buses": { ... },
  "words_with_site_buses": [...],
  "sites_with_word_buses": [...],
  "zones": [...],
  "entangling_zones": [...],
  "measurement_mode_zones": [...],
  "blockade_radius": 2.0,         // optional, default 2.0
  "paths": [...],                 // optional
  "feed_forward": false,          // optional, default false
  "atom_reloading": false          // optional, default false
}
```

| Field | Type | Description |
|---|---|---|
| `version` | string | Format version as `"major.minor"` (e.g. `"2.0"`). |
| `geometry` | object | Physical geometry — words, grids, and site positions. |
| `buses` | object | Transport bus definitions (site buses and word buses). |
| `words_with_site_buses` | integer[] | Word IDs with intra-word site transport capability. |
| `sites_with_word_buses` | integer[] | Site indices that serve as landing pads for inter-word transport. |
| `zones` | Zone[] | Logical groupings of words for execution phases. |
| `entangling_zones` | [word_a, word_b][][] | Entanglement zones, each containing word-ID pairs whose matching sites are within blockade radius. |
| `measurement_mode_zones` | integer[] | Zone IDs that support measurement. |
| `blockade_radius` | float | *(optional, default `2.0`)* Rydberg blockade radius in micrometers. |
| `paths` | TransportPath[] | *(optional)* AOD transport paths between locations. |
| `feed_forward` | bool | *(optional, default `false`)* Whether the device supports mid-circuit measurement with classical feedback. |
| `atom_reloading` | bool | *(optional, default `false`)* Whether the device supports reloading atoms after initial fill. |

---

## Geometry

The geometry describes the physical layout of the device: how many words exist, how many sites each word contains, and where those sites are in 2D space.

```jsonc
"geometry": {
  "sites_per_word": 10,
  "words": [
    {
      "positions": {
        "x_start": 1.0,
        "y_start": 2.5,
        "x_spacing": [2.0, 2.0, 2.0, 2.0],
        "y_spacing": [2.5]
      },
      "site_indices": [[0, 0], [1, 0], [2, 0], ...]
    }
  ]
}
```

### Word

A **word** is an independent register of atom trapping sites arranged on a 2D grid. It is the fundamental unit of the device topology.

A word's ID is its index in the `geometry.words` array (e.g., the first word is word 0).

| Field | Type | Description |
|---|---|---|
| `positions` | Grid | The 2D coordinate system for this word. |
| `site_indices` | [x_idx, y_idx][] | Site positions as index pairs into the grid's x and y coordinate arrays. |

### Grid

A **grid** defines the physical coordinate axes for a word using a start position and spacing values. Positions are typically in micrometers (µm).

| Field | Type | Description |
|---|---|---|
| `x_start` | float | X-coordinate of the first grid point. |
| `y_start` | float | Y-coordinate of the first grid point. |
| `x_spacing` | float[] | Spacing between consecutive x-coordinates. The number of x grid points is `len(x_spacing) + 1`. |
| `y_spacing` | float[] | Spacing between consecutive y-coordinates. The number of y grid points is `len(y_spacing) + 1`. |

The x-coordinates are computed as `[x_start, x_start + x_spacing[0], x_start + x_spacing[0] + x_spacing[1], ...]` (cumulative sum of spacings from the start). Same for y. Sites reference grid positions by index: site `[2, 1]` is located at the 3rd x-coordinate and 2nd y-coordinate.

All words must have the same grid shape — i.e., the same number of x and y grid points (same `x_spacing` and `y_spacing` lengths). The actual coordinate values may differ (words can be at different physical locations), but the grid dimensions must be consistent.

---

## Buses

Buses are the physical transport channels that move atoms. There are two kinds:

### Site Bus

A **site bus** moves atoms between sites *within the same word*. It defines a paired mapping where `src[i]` moves to `dst[i]`.

```jsonc
{ "src": [0, 1, 2, 3, 4], "dst": [5, 6, 7, 8, 9] }
```

This means atom at site 0 moves to site 5, atom at site 1 moves to site 6, and so on — all in a single transport operation. The `src` and `dst` arrays must be the same length and must not overlap (no site can be both a source and a destination in the same bus). A site bus's ID is its index in the `buses.site_buses` array.

### Word Bus

A **word bus** moves atoms between sites across *different words*. The `src` and `dst` arrays contain word IDs (not site indices). A word bus's ID is its index in the `buses.word_buses` array.

```jsonc
{ "src": [0], "dst": [1] }
```

The specific sites involved in inter-word transport are those listed in `sites_with_word_buses` — these are the "landing pad" positions within each word.

### Supporting Fields

| Field | Description |
|---|---|
| `words_with_site_buses` | Which words have site bus hardware. Only these words can execute intra-word site moves. |
| `sites_with_word_buses` | Which site indices serve as landing pads for word-bus moves. These positions within each word are where atoms arrive and depart during inter-word transport. |

---

## Zones

A **zone** groups words into logical regions for different execution phases (entangling, measurement, etc.).

```jsonc
"zones": [
  { "words": [0, 1, 2] }
]
```

A zone's ID is its index in the `zones` array (e.g., the first zone is zone 0).

Zone 0 is special — it must contain every word in the geometry. This ensures there is always a "global" zone that covers the entire device.

### Entangling Zones

`entangling_zones` defines which word pairs can perform CZ (entangling) gates. Each entangling zone is a list of `[word_a, word_b]` pairs. Within a pair, sites at matching indices in `word_a` and `word_b` are within blockade radius.

```jsonc
"entangling_zones": [
  [[0, 1], [2, 3]]   // zone 0: words 0↔1 and 2↔3 are CZ pairs
]
```

### Measurement Mode Zones

`measurement_mode_zones` lists zone IDs that support measurement operations. If non-empty, the first entry must be zone 0.

---

## Paths (Optional)

AOD (Acousto-Optic Deflector) transport paths. Each path identifies a transport lane and provides a sequence of `[x, y]` waypoints defining the physical trajectory atoms follow during transport.

The lane is identified by its encoded `LaneAddr`, serialized as a hex string. See [Address Encoding](#address-encoding) for the `LaneAddr` bit layout.

```jsonc
"paths": [
  {
    "lane": "0xC000000000000000",                        // encoded LaneAddr (hex, 16-digit)
    "waypoints": [[1.0, 12.5], [1.0, 7.5], [1.0, 2.5]]  // physical trajectory
  }
]
```

Each `TransportPath` entry has:

| Field | Type | Description |
|---|---|---|
| `lane` | string | Encoded `LaneAddr` as a `"0x..."` hex string. |
| `waypoints` | [x, y][] | Sequence of physical coordinate waypoints. |

To decode the lane hex string, parse it as a 64-bit unsigned integer. The low 32 bits (data0) contain `[word_id:16][site_id:16]` and the high 32 bits (data1) contain `[dir:1][mt:1][pad:14][bus_id:16]`. For example, `"0xC000000000000000"` decodes to direction=Backward, move_type=WordBus, word=0, site=0, bus=0. In the lane address convention, `word_id` always encodes the forward-direction source word for that lane; in this specific example the bus mapping is src=[0], dst=[1], so `direction=Backward` means the move goes from word 1 back to word 0.

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

The `ArchSpec::validate()` method checks all structural rules in a single pass, collecting every error rather than failing fast. The following rules are enforced:

### Zone Rules

| Rule | Error |
|---|---|
| Zone 0 must include every word ID in the geometry | `Zone0MissingWords` |
| `measurement_mode_zones` must not be empty | `MeasurementModeZonesEmpty` |
| `measurement_mode_zones[0]` must be zone 0 | `MeasurementModeFirstNotZone0` |
| Every ID in `entangling_zones` must reference a defined zone | `InvalidEntanglingZone` |
| Every ID in `measurement_mode_zones` must reference a defined zone | `InvalidMeasurementModeZone` |

### Word / Site Rules

| Rule | Error |
|---|---|
| Every word must have exactly `sites_per_word` site_indices | `WrongSiteCount` |
| All words must have the same grid shape (same number of x and y positions) | `InconsistentGridShape` |
| Grid coordinates must be finite (no NaN or Inf) | `NonFiniteGridValue` |
| Site `x_idx` must be < number of x grid points (`len(x_spacing) + 1`) | `SiteXIndexOutOfRange` |
| Site `y_idx` must be < number of y grid points (`len(y_spacing) + 1`) | `SiteYIndexOutOfRange` |

### Site Bus Rules

| Rule | Error |
|---|---|
| `src.length` must equal `dst.length` | `SiteBusLengthMismatch` |
| `src` and `dst` must be disjoint (no shared site indices) | `SiteBusSrcDstOverlap` |
| All site indices in `src` and `dst` must be < `sites_per_word` | `SiteBusIndexOutOfRange` |

### Word Bus Rules

| Rule | Error |
|---|---|
| `src.length` must equal `dst.length` | `WordBusLengthMismatch` |
| All word IDs in `src` and `dst` must exist in `geometry.words` | `WordBusInvalidWordId` |

### Cross-Reference Rules

| Rule | Error |
|---|---|
| Every ID in `words_with_site_buses` must be a valid word ID | `InvalidWordWithSiteBus` |
| Every index in `sites_with_word_buses` must be < `sites_per_word` | `InvalidSiteWithWordBus` |

### Path Rules

| Rule | Error |
|---|---|
| Every path's `lane` must decode to a valid `LaneAddr` (valid bus ID, word ID, site ID) | `InvalidPathLane` |
| Every path must have at least 2 waypoints | `PathTooFewWaypoints` |
| The first waypoint must match the source position of the lane, and the last waypoint must match the destination position | `PathEndpointMismatch` |
| Waypoint coordinates must be finite (no NaN or Inf) | `NonFiniteWaypoint` |

### Capability Rules (Bytecode Validation)

These rules are checked during bytecode validation when an `ArchSpec` is provided:

| Rule | Error |
|---|---|
| If `feed_forward = false`, at most one `measure` instruction is allowed | `FeedForwardNotSupported` |
| If `atom_reloading = false`, no `fill` instruction is allowed | `AtomReloadingNotSupported` |

---

## Address Encoding

At the bytecode level, locations and lanes are encoded as bit-packed integers with 16-bit address fields. Each address type is packed into instruction data words (u32):

| Type | Width | Layout | Description |
|---|---|---|---|
| `LocationAddr` | 32 bits (1 × u32) | data0: `[word_id:16][site_id:16]` | Identifies a specific site within a word. |
| `LaneAddr` | 64 bits (2 × u32) | data0: `[word_id:16][site_id:16]`, data1: `[dir:1][mt:1][pad:14][bus_id:16]` | Identifies a transport lane (direction + move type + site + bus). |
| `ZoneAddr` | 32 bits (1 × u32) | data0: `[pad:16][zone_id:16]` | Identifies a zone. |

These packed addresses are used in 16-byte bytecode instructions (opcode + 3 data words) and are validated against the arch spec during program validation. In JSON and Python, `LaneAddr` is represented as a 64-bit integer (16-digit hex string in JSON, `u64` in Python).

---

## Examples

Minimal spec with one word and one site bus:

```json
{
  "version": "2.0",
  "geometry": {
    "sites_per_word": 5,
    "words": [
      {
        "positions": {
          "x_start": 1.0,
          "y_start": 2.0,
          "x_spacing": [2.0, 2.0, 2.0, 2.0],
          "y_spacing": []
        },
        "site_indices": [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]]
      }
    ]
  },
  "buses": {
    "site_buses": [
      { "src": [0, 1], "dst": [3, 4] }
    ],
    "word_buses": []
  },
  "words_with_site_buses": [0],
  "sites_with_word_buses": [],
  "zones": [
    { "words": [0] }
  ],
  "entangling_zones": [],
  "measurement_mode_zones": [0]
}
```

A fuller example with multiple words, CZ pairs, and word buses is available at [`examples/arch/full.json`](../../examples/arch/full.json).
