# Zone-Centric ArchSpec — Architecture Guide

**Date:** 2026-04-06
**Audience:** Developers implementing, maintaining, or integrating with the ArchSpec data model
**Companion spec:** `2026-04-06-zone-centric-archspec-design.md`

## 1. The physical picture

A neutral atom quantum processor holds individual atoms in optical trap sites arranged in a 2D grid. Atoms are moved between sites using Acousto-Optic Deflector (AOD) beams that shift entire rows or columns of traps simultaneously. Two-qubit entangling gates (CZ) require pairs of atoms to be physically close — within the **blockade radius**.

The hardware is naturally divided into **zones** — physically distinct regions of the trap array, each with its own grid of trap sites. Different zones serve different purposes (gate execution, atom storage) and can have different physical spacing between sites. Atoms are transported between zones using AOD moves that cross zone boundaries.

## 2. Why zones are the primary unit

### The old model

In the original ArchSpec, the data model looked like this:

```
ArchSpec
├── geometry
│   └── words: [Word { positions: Grid, site_indices, has_cz }, ...]
├── buses: { site_buses: [...], word_buses: [...] }   ← global flat lists
├── zones: [{ words: [0, 1, 2, 3] }, ...]             ← just word ID groupings
├── entangling_zones: [...]
└── measurement_mode_zones: [...]
```

Zones were an afterthought — a list of word IDs with no structural ownership. Each word carried its own grid, buses were global, and the zone's geometry had to be reconstructed by aggregating across its member words.

### The problem

This model breaks down when zones have different physical spacing. Consider a processor with a **gate region** (tight spacing for entangling) and a **memory region** (wider spacing for storage). The gate zone needs 10-unit spacing between columns; the memory zone uses 8-unit spacing. In the old model, there was no natural place to express this — zones didn't own a coordinate system.

The flair pulse-level language needs to generate AOD waveforms per zone, which requires knowing each zone's grid geometry directly. Reconstructing it from scattered word definitions is fragile and unnatural.

### The new model

Zones become the primary structural unit. Each zone owns a `Grid` (its physical coordinate system) and its internal bus connectivity:

```
ArchSpec
├── version: 2.0
├── words: [Word { sites }, ...]                      ← global slicing template
├── zones:
│   ├── Zone 0: { grid, site_buses, word_buses,
│   │             words_with_site_buses, sites_with_word_buses }
│   ├── Zone 1: { grid, site_buses, word_buses, ... }
│   └── ...
├── zone_buses: [Bus<ZonedWordRef>, ...]              ← inter-zone movement
├── entangling_zone_pairs: [(0, 1)]
├── modes: [Mode { name, zones, bitstring_order }]
├── paths, feed_forward, atom_reloading
└── ...
```

Each zone also carries **bus membership lists**: `words_with_site_buses` (which words participate in site bus operations) and `sites_with_word_buses` (which sites participate in word bus operations). These constrain which parts of the zone are moveable by each bus type.

### Uniform zone dimension constraint

All zones must have the same grid dimensions (same number of x-positions and y-positions), the same number of words, and the same sites_per_word. Only the physical grid coordinates and internal bus connectivity differ per zone. This constraint is foundational — it enables the 1:1 site correspondence needed for entangling zone pairs and simplifies all cross-zone operations.

Words are a global slicing template — the same logical structure repeated in every zone, just mapped onto different physical coordinates.

## 3. Running example: gate + memory architecture

Throughout this document we use a four-zone architecture with two region types.

### Zone grids

| Zone | Region | x_positions | y_positions | Role |
|------|--------|-------------|-------------|------|
| 0 | Gate | [0, 10, 20, 30] | [0, 10] | Entangling (primary) |
| 1 | Gate | [2, 12, 22, 32] | [0, 10] | Entangling (partner, +2.0 shift) |
| 2 | Memory | [0, 8, 16, 24] | [20, 24] | Storage (primary) |
| 3 | Memory | [4, 12, 20, 28] | [20, 24] | Storage (partner, +4.0 shift) |

All four zones share the same grid dimensions (4 x-positions, 2 y-positions) and the same word slicing (4 words, 2 sites each). The physical coordinates differ.

### Word slicing (global template)

Words are horizontal pairs of adjacent grid columns within the same row:

```
  x-index:  0    1    2    3
         ┌────────┐ ┌────────┐
y=0      │ W0     │ │ W1     │     W0 = [(0,0), (1,0)]
         │ s0  s1 │ │ s0  s1 │     W1 = [(2,0), (3,0)]
         └────────┘ └────────┘     W2 = [(0,1), (1,1)]
         ┌────────┐ ┌────────┐     W3 = [(2,1), (3,1)]
y=1      │ W2     │ │ W3     │
         │ s0  s1 │ │ s0  s1 │     4 words x 2 sites = 8 sites per zone
         └────────┘ └────────┘
```

This template is the same in every zone. A site's physical position is resolved by looking up its grid indices in the parent zone's grid: site (x_idx, y_idx) in zone Z is at position `(zones[Z].grid.x[x_idx], zones[Z].grid.y[y_idx])`.

### Physical layout

The gate region shows interleaved zone pairs with alternating tight/wide spacing:

```
Gate Region (zones 0, 1)                    entangling_zone_pairs: [(0, 1)]

x-axis:  0  2      10 12      20 22      30 32
         ●  ○       ●  ○       ●  ○       ●  ○     y = 0
         │  │       │  │       │  │       │  │
         ●  ○       ●  ○       ●  ○       ●  ○     y = 10
         ├──┤       ├──┤       ├──┤       ├──┤
          2    gap    2    gap    2    gap    2
               8          8          8

● = Zone 0 site    ○ = Zone 1 site    2.0 shift = within blockade radius
```

The memory region shows uniform spacing — the 4.0 shift places zone 3 sites exactly halfway between zone 2 sites:

```
Memory Region (zones 2, 3)                  no entangling pair

x-axis:  0    4    8   12   16   20   24   28
         ●    ○    ●    ○    ●    ○    ●    ○     y = 20
         │    │    │    │    │    │    │    │
         ●    ○    ●    ○    ●    ○    ●    ○     y = 24
         ├────┤────┤────┤────┤────┤────┤────┤
          4    4    4    4    4    4    4
                    uniform spacing

● = Zone 2 site    ○ = Zone 3 site    4.0 shift = too far for CZ
```

**The spacing pattern reveals the physics:** alternating tight/wide gaps in the gate region show where entangling pairs sit. Uniform gaps in the memory region show storage without entangling capability. Same logical structure, different physical geometry — this is why zones must own their own Grid.

## 4. The three-tier bus hierarchy

Atoms move between trap sites via AOD transport buses. The zone-centric model introduces a three-tier bus hierarchy, each level describing movement at a different scope:

### Site buses (intra-word)

Move atoms between sites **within the same word**. Owned by the zone.

A site bus definition applies to **every word listed in `words_with_site_buses`** for that zone. In our example, a site bus in zone 0 moves atoms from site 0 to site 1 — in Word 0, that's physically from x=0 to x=10 along the y=0 row. The same bus applies to all participating words simultaneously (this is how AOD beams work — they move entire grids of atoms at once).

```rust
// Zone 0's site buses — applies to all words in words_with_site_buses
site_buses: [Bus<SiteRef> { src: [SiteRef(0)], dst: [SiteRef(1)] }]
words_with_site_buses: [0, 1, 2, 3]  // all words participate
```

### Word buses (intra-zone)

Move atoms between **different words within the same zone**. Owned by the zone.

A word bus in zone 0 might move atoms from Word 0 to Word 2 — physically from the top row to the bottom row.

```rust
// Zone 0's word buses
word_buses: [Bus<WordRef> { src: [WordRef(0)], dst: [WordRef(2)] }]
```

### Zone buses (inter-zone)

Move atoms between **words in different zones**. Owned by the ArchSpec.

A zone bus might move atoms from zone 0 (gate) to zone 2 (memory) — physically transporting atoms from the gate region to the memory region.

```rust
// ArchSpec-level zone buses
zone_buses: [Bus<ZonedWordRef> {
    src: [ZonedWordRef { zone_id: 0, word_id: 0 }],
    dst: [ZonedWordRef { zone_id: 2, word_id: 0 }],
}]
```

Every zone bus entry must cross a zone boundary — `src[i].zone_id != dst[i].zone_id` for all i.

### The generic Bus\<T\> type

All three bus levels share the same structure, parameterized by address type:

```rust
struct Bus<T> {
    src: Vec<T>,
    dst: Vec<T>,
}
```

| Level | Type parameter | Scope | Owned by |
|-------|---------------|-------|----------|
| Site bus | `Bus<SiteRef>` | Within a word | Zone |
| Word bus | `Bus<WordRef>` | Within a zone | Zone |
| Zone bus | `Bus<ZonedWordRef>` | Across zones | ArchSpec |

### Grid invariant

For every bus at every level: the source positions and destination positions, when resolved to physical coordinates, must each form a **complete rectangular grid** (Cartesian product of x-values and y-values), and both grids must have the same number of rows and columns. This reflects the AOD hardware constraint — AOD beams move entire rows or columns simultaneously, so every move must operate on a rectangular grid of atoms.

## 5. Entangling zone pairs

Two-qubit CZ gates require atoms to be within the blockade radius. In the zone-centric model, this is expressed as a **zone-pair relationship**: two zones whose grids are physically interleaved such that corresponding sites are close enough for entangling.

### How it works

`entangling_zone_pairs: [(0, 1)]` means: for every `(word_id, site_id)` in zone 0, the CZ partner is the same `(word_id, site_id)` in zone 1. This 1:1 correspondence is guaranteed by the **uniform zone dimension constraint** — all zones have the same grid dimensions and word slicing.

In our running example:
- Zone 0, Word 0, Site 0 is at position (0, 0)
- Zone 1, Word 0, Site 0 is at position (2, 0)
- Distance = 2.0 — within blockade radius

Every site pair across the two zones has this same 2.0 separation because zone 1's grid is zone 0's grid shifted by 2.0 in x.

### Why this replaced has_cz

The old model stored CZ capability per-word (`Word.has_cz`) as pairs of site indices within a word. This was fundamentally a word-level concept — CZ happened between sites in the same word.

The new architecture model uses inter-word CZ pairing: atoms in one zone entangle with atoms in another zone. Expressing this as word-level `has_cz` fields would require cross-referencing words across zones, which is awkward and error-prone. The zone-pair model captures the physics directly: two interleaved grids, 1:1 site correspondence, one declaration.

### Not all paired zones entangle

Zones 2 and 3 in our example are also interleaved (4.0 shift), but 4.0 is beyond the blockade radius — they are **not** in `entangling_zone_pairs`. Their interleaved structure serves a different purpose (e.g., load balancing, parallel operations). The data model distinguishes these cases explicitly.

## 6. Measurement modes

The old model used `measurement_mode_zones: Vec<u32>` — a flat list of zone indices that support measurement — combined with a special "Zone 0 = all words" convention. This had two problems: it didn't define bitstring ordering, and the Zone 0 convention conflated a measurement configuration with zone identity.

### Modes replace both

A `Mode` explicitly defines a measurement configuration:

```rust
struct Mode {
    name: String,                       // e.g. "full", "gate_only"
    zones: Vec<u32>,                    // which zones are imaged
    bitstring_order: Vec<LocationAddr>, // bit position → site mapping
}
```

Example modes for our four-zone architecture:

```json
[
  {
    "name": "full",
    "zones": [0, 1, 2, 3],
    "bitstring_order": ["<all 32 sites in canonical order>"]
  },
  {
    "name": "gate_only",
    "zones": [0, 1],
    "bitstring_order": ["<16 gate-region sites>"]
  }
]
```

The `bitstring_order` array maps each bit position in the measurement result to a specific `LocationAddr`. This makes the mapping explicit — no convention needed, no special Zone 0.

## 7. Address encoding

Every site in the architecture is identified by a `LocationAddr` encoding `(zone_id, word_id, site_id)`. Every transport lane is identified by a `LaneAddr` encoding the bus address plus zone context.

### LocationAddr — 64-bit

```
[zone_id : 8][word_id : 16][site_id : 16][padding : 24]
 └─ 256 zones  └─ 65536 words  └─ 65536 sites
```

Widened from 32-bit to accommodate the zone_id field. The `ZonedWordRef` type used in zone bus entries maps to the upper 24 bits (zone_id + word_id).

### LaneAddr — 64-bit

```
data0: [word_id : 16][site_id : 16]         (unchanged from previous layout)
data1: [dir : 1][mt : 2][zone_id : 8][pad : 5][bus_id : 16]
```

The `MoveType` field expands from 1 bit to 2 bits:

| Value | MoveType | Scope |
|-------|----------|-------|
| 0 | SiteBus | Intra-word |
| 1 | WordBus | Intra-zone |
| 2 | ZoneBus | Inter-zone |
| 3 | Reserved | — |

Zone context in `LaneAddr` comes from `data1`, not `data0`. The two address types have independent encoding layouts: `LocationAddr` puts zone_id in the high bits for natural sorting; `LaneAddr` puts it in the metadata word alongside direction and move type.

### Address newtypes for bus entries

The `Bus<T>` generic uses address newtypes that match the encoding widths:

| Type | Width | Usage |
|------|-------|-------|
| `SiteRef(u16)` | 16-bit | Site bus src/dst entries |
| `WordRef(u16)` | 16-bit | Word bus src/dst entries |
| `ZonedWordRef { zone_id: u8, word_id: u16 }` | 24-bit | Zone bus src/dst entries |
