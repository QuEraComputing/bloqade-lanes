# Zone-Centric Architecture — Visual Guide

This guide explains the zone-centric ArchSpec model using diagrams and visual
examples. For the formal specification, see [Architecture Specification](archspec.md).

---

## The Big Picture

A neutral atom quantum processor traps individual atoms in a 2D grid of optical
trap sites. Two-qubit entangling gates (CZ) require pairs of atoms to be
physically close — within the **blockade radius**. Atoms are shuttled between
sites using **AOD beams** (Acousto-Optic Deflectors) that move entire rows or
columns of traps simultaneously.

The architecture is organized into **zones** — physically distinct regions of the
trap array. Each zone has its own grid of trap sites. Different zones serve
different purposes: some perform entangling gates, others store atoms.

```
┌─────────────────────────────────────────────────────────────┐
│                     QUANTUM PROCESSOR                       │
│                                                             │
│   ┌───────────────────────┐   ┌───────────────────────┐    │
│   │     GATE REGION       │   │    MEMORY REGION       │    │
│   │                       │   │                        │    │
│   │  Zone 0    Zone 1     │   │  Zone 2    Zone 3      │    │
│   │  (primary) (partner)  │   │  (primary) (partner)   │    │
│   │                       │   │                        │    │
│   │   CZ gates happen     │   │   Atom storage only    │    │
│   │   between zone pairs  │   │   no entangling        │    │
│   └───────────────────────┘   └───────────────────────┘    │
│                                                             │
│             ◄── zone buses transport atoms ──►              │
└─────────────────────────────────────────────────────────────┘
```

---

## From Atoms to Zones: Building Up

### Step 1 — Sites on a Grid

Each zone contains trap sites arranged on a 2D grid. A site's position is
determined by indexing into the zone's x and y coordinate arrays:

```
Zone 0 Grid
                                                     x_positions: [0, 10, 20, 30]
  x-index:    0       1       2       3              y_positions: [0, 10]
            ┌───────────────────────────┐
  y=0       │  ●       ●       ●       ●  │  ◄─ y-index 0
            │(0,0)  (10,0)  (20,0)  (30,0) │
            │                               │
  y=10      │  ●       ●       ●       ●  │  ◄─ y-index 1
            │(0,10) (10,10) (20,10) (30,10) │
            └───────────────────────────────┘
                  8 sites total
```

### Step 2 — Words Slice the Grid

A **word** groups sites into a logical register. Words are a **global template**
— the same slicing pattern is used in every zone. Only the physical coordinates
change.

In our example, words are horizontal pairs — two adjacent x-indices in the same
row:

```
  x-index:   0     1     2     3
           ┌─────────┐ ┌─────────┐
  y=0      │  Word 0  │ │  Word 1  │      Word 0 = sites (0,0), (1,0)
           │  s0   s1 │ │  s0   s1 │      Word 1 = sites (2,0), (3,0)
           └─────────┘ └─────────┘      Word 2 = sites (0,1), (1,1)
           ┌─────────┐ ┌─────────┐      Word 3 = sites (2,1), (3,1)
  y=1      │  Word 2  │ │  Word 3  │
           │  s0   s1 │ │  s0   s1 │      4 words × 2 sites = 8 sites
           └─────────┘ └─────────┘
```

### Step 3 — Zones Own the Geometry

Each zone applies the same word template onto its own physical grid. The
**uniform zone dimension constraint** requires all zones to have the same
number of x-positions, y-positions, words, and sites per word.

```
 Zone 0 (Gate)                 Zone 1 (Gate)
 x: [0, 10, 20, 30]           x: [2, 12, 22, 32]
 y: [0, 10]                   y: [0, 10]

   0    10    20    30           2    12    22    32
   ●─────●─────●─────●          ○─────○─────○─────○       y = 0
   │     │     │     │          │     │     │     │
   ●─────●─────●─────●          ○─────○─────○─────○       y = 10


 Zone 2 (Memory)               Zone 3 (Memory)
 x: [0, 8, 16, 24]            x: [4, 12, 20, 28]
 y: [20, 24]                  y: [20, 24]

   0     8    16    24           4    12    20    28
   ●─────●─────●─────●          ○─────○─────○─────○       y = 20
   │     │     │     │          │     │     │     │
   ●─────●─────●─────●          ○─────○─────○─────○       y = 24
```

Same logical structure (4×2 grid, 4 words, 2 sites each), different physical
coordinates.

---

## How Zones Interleave

When two zones are paired, their grids interleave — zone 1's sites sit between
zone 0's sites. The spacing between interleaved sites reveals whether
entangling is possible:

### Gate Region — Tight Interleaving (CZ Capable)

```
 Zone 0 (●) and Zone 1 (○) interleaved on the x-axis:

                    2.0         2.0         2.0         2.0
                  ◄─────►     ◄─────►     ◄─────►     ◄─────►
  x:  0   2      10  12      20  22      30  32
      ●   ○       ●   ○       ●   ○       ●   ○       y = 0
      │   │       │   │       │   │       │   │
      ●   ○       ●   ○       ●   ○       ●   ○       y = 10
      ├───┤       ├───┤       ├───┤       ├───┤
       2.0         2.0         2.0         2.0
    within       within      within      within
    blockade     blockade    blockade    blockade
    radius ✓     radius ✓    radius ✓    radius ✓
```

The 2.0 unit gap between ● and ○ is within the blockade radius → CZ gates are
possible between corresponding sites.

### Memory Region — Wide Interleaving (No CZ)

```
 Zone 2 (●) and Zone 3 (○) interleaved on the x-axis:

  x:  0    4    8   12   16   20   24   28
      ●    ○    ●    ○    ●    ○    ●    ○       y = 20
      │    │    │    │    │    │    │    │
      ●    ○    ●    ○    ●    ○    ●    ○       y = 24
      ├────┤────┤────┤────┤────┤────┤────┤
       4.0  4.0  4.0  4.0  4.0  4.0  4.0
              uniform spacing
            too far for CZ ✗
```

The 4.0 unit gap is beyond the blockade radius → no entangling here. The
interleaving serves other purposes (load balancing, parallel operations).

### Entangling Zone Pairs

The ArchSpec declares which zone pairs can entangle:

```
  entangling_zone_pairs: [(0, 1)]

  Zone 0 ←──── CZ ────→ Zone 1       ✓  gap = 2.0 ≤ blockade_radius
  Zone 2 ←── no CZ ───→ Zone 3       ✗  gap = 4.0 > blockade_radius
```

For every `(word_id, site_id)` in zone 0, the CZ partner is the same
`(word_id, site_id)` in zone 1 — a 1:1 correspondence guaranteed by the
uniform zone dimension constraint.

---

## The Three-Tier Bus Hierarchy

Atoms move via AOD transport buses organized in three tiers. Each tier describes
movement at a different scope:

```
┌─────────────────────────────────────────────────────────────────┐
│                         ArchSpec                                │
│                                                                 │
│   zone_buses: [Bus<ZonedWordRef>]     ◄── TIER 3: across zones │
│                                                                 │
│   ┌─────────────── Zone 0 ──────────────┐  ┌──── Zone 2 ────┐  │
│   │                                     │  │                 │  │
│   │  word_buses: [Bus<WordRef>]  ◄── TIER 2: across words   │  │
│   │                                     │  │                 │  │
│   │  ┌── Word 0 ──┐  ┌── Word 1 ──┐    │  │  ┌── Word ──┐  │  │
│   │  │            │  │            │    │  │  │          │  │  │
│   │  │ site_buses │  │ site_buses │    │  │  │          │  │  │
│   │  │ [Bus       │  │ [Bus       │    │  │  │          │  │  │
│   │  │ <SiteRef>] │  │ <SiteRef>] │    │  │  │          │  │  │
│   │  │     ▲      │  │            │    │  │  │          │  │  │
│   │  │     │      │  │            │    │  │  │          │  │  │
│   │  │  TIER 1:   │  │            │    │  │  │          │  │  │
│   │  │  within    │  │            │    │  │  │          │  │  │
│   │  │  a word    │  │            │    │  │  │          │  │  │
│   │  └────────────┘  └────────────┘    │  │  └──────────┘  │  │
│   └─────────────────────────────────────┘  └────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Tier 1 — Site Buses (Intra-Word)

Move atoms between sites **within the same word**. Owned by the zone.

```
  Word 0 in Zone 0:
                                   site_buses: [{ src: [0], dst: [1] }]
     site 0         site 1
       ●  ──────────► ●           Atom moves from site 0 → site 1
    (0, 0)          (10, 0)       Physical: x=0 → x=10

  A site bus applies to every word in words_with_site_buses simultaneously
  (AOD beams move entire grids at once):

     Word 0:   s0 ──► s1
     Word 1:   s0 ──► s1      All four words move in lockstep
     Word 2:   s0 ──► s1
     Word 3:   s0 ──► s1
```

### Tier 2 — Word Buses (Intra-Zone)

Move atoms between **different words within the same zone**. Owned by the zone.

```
  Zone 0:                          word_buses: [{ src: [0], dst: [2] }]

     Word 0   ●───●
              │        ╲
              │         ╲         Atom moves from Word 0 → Word 2
              │          ╲        (top row → bottom row)
              │           ▼
     Word 2   ●───●

  Only sites listed in sites_with_word_buses participate as landing pads.
```

### Tier 3 — Zone Buses (Inter-Zone)

Move atoms between **words in different zones**. Owned by the ArchSpec.

```
  ArchSpec zone_buses: [{ src: [(zone:0, word:0)], dst: [(zone:2, word:0)] }]

  ┌──── Zone 0 (Gate) ────┐         ┌──── Zone 2 (Memory) ────┐
  │                        │         │                          │
  │  Word 0  ● ● ─────────│────────►│──► ● ●  Word 0           │
  │                        │  AOD    │                          │
  │  Word 1  ● ●           │ move    │        ● ●  Word 1       │
  │                        │         │                          │
  └────────────────────────┘         └──────────────────────────┘

  Every zone bus entry MUST cross a zone boundary.
```

### The Generic Bus\<T\> Type

All three tiers share the same generic structure — only the address type changes:

```
  Bus<T> { src: [T, ...], dst: [T, ...] }

  ┌────────────┬──────────────────┬──────────────┬───────────┐
  │ Tier       │ Type Parameter   │ Scope        │ Owned By  │
  ├────────────┼──────────────────┼──────────────┼───────────┤
  │ Site bus   │ Bus<SiteRef>     │ Within word  │ Zone      │
  │ Word bus   │ Bus<WordRef>     │ Within zone  │ Zone      │
  │ Zone bus   │ Bus<ZonedWordRef>│ Across zones │ ArchSpec  │
  └────────────┴──────────────────┴──────────────┴───────────┘
```

### The Rectangle Constraint

For every bus at every level, the source and destination positions must each
form a **complete rectangular grid** (Cartesian product of x and y values), and
both rectangles must have the same dimensions. This reflects the AOD hardware
— beams move entire rows or columns, so every move operates on a rectangle:

```
  Valid site bus (2×1 rectangle → 2×1 rectangle):

     src              dst
    ┌─────┐         ┌─────┐
    │ ● ● │  ────►  │ ● ● │       src and dst are both 2×1 grids  ✓
    └─────┘         └─────┘

  Invalid (L-shape — not a rectangle):

    ● ●
    ●          This is NOT a valid bus configuration  ✗
```

---

## Old Model vs. New Model

The zone-centric redesign shifts structural ownership from words to zones:

```
  OLD MODEL                              NEW MODEL
  ─────────                              ─────────

  ArchSpec                               ArchSpec
  ├── geometry                           ├── version: 2.0
  │   └── words[]                        ├── words[]  ◄── global template
  │       ├── positions: Grid  ◄─ each   │   └── sites: [(x_idx, y_idx)]
  │       │   word owns a grid           │
  │       ├── site_indices               ├── zones[]  ◄── primary unit
  │       └── has_cz  ◄─ CZ per word     │   ├── grid  ◄── zone owns grid
  │                                      │   ├── site_buses
  ├── buses  ◄── global flat lists       │   ├── word_buses
  │   ├── site_buses[]                   │   ├── words_with_site_buses
  │   └── word_buses[]                   │   └── sites_with_word_buses
  │                                      │
  ├── zones[]  ◄── afterthought          ├── zone_buses[]  ◄── inter-zone
  │   └── words: [0, 1, 2, 3]           │
  │                                      ├── entangling_zone_pairs
  ├── entangling_zones                   ├── modes[]  ◄── replaces
  └── measurement_mode_zones             │   ├── name       measurement_mode
                                         │   ├── zones      _zones
                                         │   └── bitstring_order
                                         │
                                         └── blockade_radius
```

Key shifts:

| Aspect | Old | New |
|--------|-----|-----|
| Grid ownership | Each word owns a Grid | Each zone owns a Grid |
| Bus ownership | Global flat lists | Per-zone (site + word) + ArchSpec (zone) |
| CZ declaration | `Word.has_cz` (intra-word pairs) | `entangling_zone_pairs` (inter-zone) |
| Measurement | `measurement_mode_zones: [int]` | `modes: [{ name, zones, bitstring_order }]` |
| Zone role | Grouping of word IDs | Primary structural unit |

---

## Address Encoding

Every site and transport lane has a compact bit-packed address used in bytecode
instructions.

### LocationAddr — Identifying a Site

```
  64-bit LocationAddr:

  ┌──────────┬────────────────┬────────────────┬────────────────────────┐
  │ zone_id  │    word_id     │    site_id     │        padding         │
  │  8 bits  │    16 bits     │    16 bits     │        24 bits         │
  └──────────┴────────────────┴────────────────┴────────────────────────┘
   ◄── 256    ◄── 65,536       ◄── 65,536
       zones       words            sites

  Example: Zone 0, Word 2, Site 1
  ┌──────────┬────────────────┬────────────────┬────────────────────────┐
  │ 00000000 │ 0000000000000010│ 0000000000000001│ 000000000000000000000000│
  └──────────┴────────────────┴────────────────┴────────────────────────┘
```

### LaneAddr — Identifying a Transport Lane

```
  64-bit LaneAddr (two 32-bit data words):

  data0 (low 32 bits):
  ┌────────────────┬────────────────┐
  │    word_id     │    site_id     │
  │    16 bits     │    16 bits     │
  └────────────────┴────────────────┘

  data1 (high 32 bits):
  ┌─────┬──────┬──────────┬─────────┬────────────────┐
  │ dir │  mt  │ zone_id  │ padding │    bus_id       │
  │ 1b  │  2b  │  8 bits  │  5 bits │   16 bits      │
  └─────┴──────┴──────────┴─────────┴────────────────┘

  dir: 0 = Forward, 1 = Backward
  mt (MoveType):
    ┌───────┬──────────┬──────────────────┐
    │ Value │   Name   │      Scope       │
    ├───────┼──────────┼──────────────────┤
    │   0   │ SiteBus  │ Within a word    │
    │   1   │ WordBus  │ Within a zone    │
    │   2   │ ZoneBus  │ Across zones     │
    │   3   │ Reserved │       —          │
    └───────┴──────────┴──────────────────┘
```

### Address Type Summary

```
  Bus<T> address newtypes — each matches an encoding width:

  ┌─────────────────────────────────────────────────────────────┐
  │  SiteRef(u16)              16-bit    Site bus src/dst       │
  │  ┌──────────────────┐                                      │
  │  │    site_id: u16   │                                      │
  │  └──────────────────┘                                      │
  │                                                             │
  │  WordRef(u16)              16-bit    Word bus src/dst       │
  │  ┌──────────────────┐                                      │
  │  │    word_id: u16   │                                      │
  │  └──────────────────┘                                      │
  │                                                             │
  │  ZonedWordRef              24-bit    Zone bus src/dst       │
  │  ┌──────────┬──────────────────┐                            │
  │  │zone_id:u8│   word_id: u16   │                            │
  │  └──────────┴──────────────────┘                            │
  └─────────────────────────────────────────────────────────────┘
```

---

## Measurement Modes

The old model used a flat list of zone IDs for measurement with a special
"Zone 0 = all words" convention. The new model replaces this with explicit
**Modes**:

```
  Old: measurement_mode_zones: [0, 1]     What's the bitstring order? 🤷

  New: modes:
  ┌──────────────────────────────────────────────────────┐
  │  Mode "full"                                         │
  │  zones: [0, 1, 2, 3]                                │
  │  bitstring_order: [                                  │
  │    (z0,w0,s0), (z0,w0,s1), (z0,w1,s0), ...         │
  │    ... all 32 sites in explicit order                │
  │  ]                                                   │
  │                                                      │
  │  Bit 0 in the measurement result = site (z0,w0,s0)  │
  │  Bit 1 in the measurement result = site (z0,w0,s1)  │
  │  ...                                                 │
  └──────────────────────────────────────────────────────┘
  ┌──────────────────────────────────────────────────────┐
  │  Mode "gate_only"                                    │
  │  zones: [0, 1]                                       │
  │  bitstring_order: [                                  │
  │    ... 16 gate-region sites only                     │
  │  ]                                                   │
  └──────────────────────────────────────────────────────┘
```

Each mode explicitly maps bit positions to physical sites — no conventions or
special zone IDs needed.

---

## Putting It All Together

Here is the complete data model for our four-zone example:

```
ArchSpec
│
├── version: "2.0"
├── blockade_radius: 2.0
│
├── words (global template):
│   ├── Word 0: sites [(0,0), (1,0)]
│   ├── Word 1: sites [(2,0), (3,0)]
│   ├── Word 2: sites [(0,1), (1,1)]
│   └── Word 3: sites [(2,1), (3,1)]
│
├── zones:
│   ├── Zone 0 (Gate Primary)
│   │   ├── grid: x=[0,10,20,30]  y=[0,10]
│   │   ├── site_buses: [{src:[0], dst:[1]}]
│   │   ├── word_buses: [{src:[0], dst:[2]}]
│   │   ├── words_with_site_buses: [0,1,2,3]
│   │   └── sites_with_word_buses: [0]
│   │
│   ├── Zone 1 (Gate Partner)
│   │   ├── grid: x=[2,12,22,32]  y=[0,10]
│   │   └── ... (same bus structure)
│   │
│   ├── Zone 2 (Memory Primary)
│   │   ├── grid: x=[0,8,16,24]   y=[20,24]
│   │   └── ... (may have different buses)
│   │
│   └── Zone 3 (Memory Partner)
│       ├── grid: x=[4,12,20,28]  y=[20,24]
│       └── ...
│
├── zone_buses:
│   └── [{src: (z:0,w:0), dst: (z:2,w:0)}]   gate ↔ memory transport
│
├── entangling_zone_pairs: [(0, 1)]
│
└── modes:
    ├── "full":      zones=[0,1,2,3]  bitstring_order=[...]
    └── "gate_only": zones=[0,1]      bitstring_order=[...]
```

### Physical Layout — All Four Zones

```
                          GATE REGION
   Zone 0 (●)  +  Zone 1 (○)              entangling_zone_pairs: [(0, 1)]

   x: 0  2     10 12     20 22     30 32
      ●  ○      ●  ○      ●  ○      ●  ○     y = 0       ─┐
      │  │      │  │      │  │      │  │                    │ 10 units
      ●  ○      ●  ○      ●  ○      ●  ○     y = 10      ─┘
      ╠══╣      ╠══╣      ╠══╣      ╠══╣
      2.0       2.0       2.0       2.0  ← within blockade radius


  ~~~~~~~~~~~~~~~ zone buses cross here ~~~~~~~~~~~~~~~


                         MEMORY REGION
   Zone 2 (●)  +  Zone 3 (○)              not in entangling_zone_pairs

   x: 0    4    8   12   16   20   24   28
      ●    ○    ●    ○    ●    ○    ●    ○     y = 20      ─┐
      │    │    │    │    │    │    │    │                    │ 4 units
      ●    ○    ●    ○    ●    ○    ●    ○     y = 24      ─┘
      ╠════╣════╣════╣════╣════╣════╣════╣
       4.0  4.0  4.0  4.0  4.0  4.0  4.0  ← too far for CZ
```

---

## Key Invariants (Quick Reference)

```
  ┌─────────────────────────────────────────────────────────────────┐
  │  1. UNIFORM DIMENSIONS                                         │
  │     All zones have the same grid dimensions, word count,       │
  │     and sites_per_word.                                        │
  │                                                                 │
  │  2. RECTANGLE CONSTRAINT                                       │
  │     Every bus's src and dst positions form complete             │
  │     rectangular grids with matching dimensions.                │
  │                                                                 │
  │  3. ZONE BUS CROSSING                                          │
  │     Every zone bus entry must cross a zone boundary.           │
  │                                                                 │
  │  4. ENTANGLING = ZONE PAIR                                     │
  │     CZ capability is declared between zone pairs, not words.   │
  │     1:1 site correspondence via matching (word_id, site_id).   │
  │                                                                 │
  │  5. WORDS ARE TEMPLATES                                        │
  │     Words define a slicing pattern, not physical positions.    │
  │     Physical positions come from the parent zone's grid.       │
  └─────────────────────────────────────────────────────────────────┘
```
