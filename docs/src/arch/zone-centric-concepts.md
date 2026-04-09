# Zone-Centric Architecture — Visual Guide

This guide explains the zone-centric ArchSpec model using diagrams and
examples. The audience is hardware engineers and architects validating that the
spec correctly models the physical device. For the formal specification, see
[Architecture Specification](archspec.md).

---

## Zones — The Primary Unit

The processor is organized into **zones** — physically distinct regions of the
trap array. Each zone owns its own coordinate grid of trap sites. Zones differ
in purpose and connectivity:

- **Gate zones** — contain entangling word pairs; high internal connectivity
  (many site and word buses)
- **Memory zones** — atom storage; no entangling pairs; lower connectivity

```
  Zone 0 (Gate)                          Zone 1 (Memory)
  ┌──────────────────────┐               ┌──────────────────────┐
  │  ●──●    ●──●        │               │  ●   ●   ●   ●       │
  │  ╠══╣    ╠══╣  dense │               │                loose │
  │  ●──●    ●──●        │               │  ●   ●   ●   ●       │
  │                      │               │                      │
  │  entangling_pairs:   │               │  entangling_pairs:   │
  │    word 0 ↔ word 1   │               │    (none)            │
  │    word 2 ↔ word 3   │               │                      │
  └──────────────────────┘               └──────────────────────┘
        ◄──── zone buses ────►
```

All zones share the same global word definitions and have the same grid
dimensions (**uniform zone dimension constraint**). What differs is the
physical spacing and the connectivity. This uniformity is what makes uniform
addressing work — `(zone, word, site)` addresses the same logical position in
any zone.

---

## Words — The Shared Template

A **word** groups grid sites into a logical register. Words are a **global
template** — the same slicing pattern applied identically across every zone. A
word references index pairs into the parent zone's grid; the physical
coordinates come from the zone.

```
  Word template (global):

    Word 0 = sites (0,0), (1,0)      ┌──────────┐
    Word 1 = sites (2,0), (3,0)      │  w0   w1 │  ← row 0
    Word 2 = sites (0,1), (1,1)      │  w2   w3 │  ← row 1
    Word 3 = sites (2,1), (3,1)      └──────────┘

  Applied to each zone:

    Zone 0 (Gate)              Zone 1 (Memory)
    ┌─────────┐                ┌─────────┐
    │ w0 │ w1 │                │ w0 │ w1 │        same word IDs
    │────┼────│                │────┼────│        same site IDs
    │ w2 │ w3 │                │ w2 │ w3 │        different physical positions
    └─────────┘                └─────────┘
```

A zone bus moving an atom from `(zone:0, word:0)` to `(zone:1, word:0)` moves
it from the gate zone's Word 0 to the memory zone's Word 0. The uniform
template means the same word ID refers to the same logical slot across zones —
this is what makes cross-zone transport well-defined.

---

## Entangling Word Pairs

Each zone declares its own `entangling_pairs` — pairs of word IDs within that
zone whose sites are physically close enough for CZ gates. A gate zone has
entangling pairs; a memory zone has none.

```
  Zone 0 (Gate) — entangling pairs: [0,1] and [2,3]

    Word 0         Word 1
    ┌────┐  CZ ↔  ┌────┐
    │ s0 │────────│ s0 │     site 0 ↔ site 0
    │ s1 │────────│ s1 │     site 1 ↔ site 1
    └────┘        └────┘

    Word 2         Word 3
    ┌────┐  CZ ↔  ┌────┐
    │ s0 │────────│ s0 │
    │ s1 │────────│ s1 │
    └────┘        └────┘

  CZ partner of (zone=0, word=0, site=1)
              is (zone=0, word=1, site=1)
              — always same zone, same site ID
```

Zone 1 (Memory) has `entangling_pairs: []` — no CZ capability. The presence
of pairs is the signal; there is no separate flag. A zone with entangling
pairs is a gate zone; a zone without is a storage zone.

---

## Transport — How Atoms Move

Atoms move via AOD transport buses at three scopes. Gate zones have more buses
(high connectivity); memory zones have fewer (lower connectivity).

### Within a word — Site Buses

Zone-owned buses that move atoms between sites within words. A site bus
applies simultaneously to all words listed in `words_with_site_buses`
(lockstep, reflecting AOD beam physics).

```
  Site bus: move atoms between sites within a word

    Word 0:   s0 ──► s1
    Word 1:   s0 ──► s1       all words in words_with_site_buses
    Word 2:   s0 ──► s1       move in lockstep (AOD beam)
    Word 3:   s0 ──► s1
```

### Within a zone — Word Buses

Zone-owned buses that move atoms between different words in the same zone.
Only sites listed in `sites_with_word_buses` participate as landing pads.

```
  Word bus: move atoms between words within a zone

    Zone 0:
      Word 0  ●───●
              │     ╲
              │      ╲        atom moves Word 0 → Word 2
              │       ▼
      Word 2  ●───●
```

### Across zones — Zone Buses

ArchSpec-owned buses that move atoms between words in different zones. Every
entry must cross a zone boundary.

```
  Zone bus: move atoms between zones

    Zone 0 (Gate)              Zone 1 (Memory)
    ┌────────────┐             ┌──────────────┐
    │ Word 0 ●●──│────────────►│──►●● Word 0  │
    │ Word 1 ●●  │             │   ●● Word 1  │
    └────────────┘             └──────────────┘
```

### Rectangle Constraint

For all bus types, the source and destination positions must each form a
**complete rectangular grid** (Cartesian product of x and y values), and both
rectangles must have the same dimensions. This reflects the AOD hardware —
beams move entire rows or columns, so every move operates on a rectangle.

### Summary

```
  ┌────────────┬──────────────┬───────────┐
  │ Bus Type   │ Scope        │ Owned By  │
  ├────────────┼──────────────┼───────────┤
  │ Site bus   │ Within word  │ Zone      │
  │ Word bus   │ Within zone  │ Zone      │
  │ Zone bus   │ Across zones │ ArchSpec  │
  └────────────┴──────────────┴───────────┘
```

---

## Measurement Modes

A **Mode** names a subset of zones and provides an explicit bit-position-to-site
mapping for measurement results.

```
  Mode "all_zones":  zones = [0, 1]     all sites, explicit ordering
  Mode "subset":     zones = [0]        gate zone sites only
```

Each mode's `bitstring_order` maps bit positions to `(zone, word, site)`
addresses — no implicit conventions or special zone IDs. The mode names and
zone subsets are user-defined; the examples above are illustrative, not
canonical.

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
  │   └── word_buses[]                   │   ├── sites_with_word_buses
  │                                      │   └── entangling_pairs ◄── per-zone
  ├── zones[]  ◄── afterthought          │
  │   └── words: [0, 1, 2, 3]           ├── zone_buses[]  ◄── inter-zone
  │                                      │
  ├── entangling_zones                   ├── modes[]  ◄── replaces
  └── measurement_mode_zones             │   ├── name       measurement_mode
                                         │   ├── zones      _zones
                                         │   └── bitstring_order
                                         │
                                         ├── feed_forward
                                         └── atom_reloading
```

| Aspect | Old | New |
|--------|-----|-----|
| Grid ownership | Each word owns a Grid | Each zone owns a Grid |
| Bus ownership | Global flat lists | Per-zone (site + word) + ArchSpec (zone) |
| CZ declaration | `Word.has_cz` (intra-word) | `Zone.entangling_pairs` (intra-zone word pairs) |
| Measurement | `measurement_mode_zones: [int]` | `modes: [{ name, zones, bitstring_order }]` |
| Zone role | Grouping of word IDs | Primary structural unit |

---

## Key Invariants (Quick Reference)

```
  ┌─────────────────────────────────────────────────────────────────┐
  │  1. UNIFORM DIMENSIONS                                         │
  │     All zones share the same global word definitions and       │
  │     have the same grid dimensions.                             │
  │                                                                 │
  │  2. RECTANGLE CONSTRAINT                                       │
  │     Every bus's src and dst positions form complete             │
  │     rectangular grids with matching dimensions.                │
  │                                                                 │
  │  3. ZONE BUS CROSSING                                          │
  │     Every zone bus entry must cross a zone boundary.           │
  │                                                                 │
  │  4. ENTANGLING = INTRA-ZONE WORD PAIRS                        │
  │     Each zone declares its own entangling_pairs (word pairs).  │
  │     CZ partner is always in the same zone, same site_id.      │
  │                                                                 │
  │  5. WORDS ARE TEMPLATES                                        │
  │     Words define a slicing pattern, not physical positions.    │
  │     Physical positions come from the parent zone's grid.       │
  └─────────────────────────────────────────────────────────────────┘
```

---

## Full Data Model (Optional Reading)

Complete ArchSpec field tree for reference. Sites are addressed as
`(zone_id, word_id, site_id)`, encoded into 64-bit addresses for bytecode —
see [Architecture Specification](archspec.md) for encoding details.

```
ArchSpec
│
├── version: Version
│
├── words: [Word]                          global template
│   └── sites: [(x_idx, y_idx)]           index pairs into parent zone's grid
│
├── zones: [Zone]                          primary structural unit
│   ├── grid: Grid                         zone's coordinate system
│   │   ├── x_start, y_start              origin
│   │   ├── x_spacing: [f64]              cumulative spacing
│   │   └── y_spacing: [f64]
│   ├── site_buses: [Bus<SiteRef>]         intra-word transport
│   ├── word_buses: [Bus<WordRef>]         intra-zone transport
│   ├── words_with_site_buses: [u32]       which words participate in site buses
│   ├── sites_with_word_buses: [u32]       which sites are word-bus landing pads
│   └── entangling_pairs: [[u32; 2]]       word pairs for CZ gates (empty = storage zone)
│
├── zone_buses: [Bus<ZonedWordRef>]        inter-zone transport
│   └── Bus { src: [ZonedWordRef], dst: [ZonedWordRef] }
│       └── ZonedWordRef { zone_id: u8, word_id: u16 }
│
├── modes: [Mode]                          measurement configurations
│   ├── name: String
│   ├── zones: [u32]                       which zones are imaged
│   └── bitstring_order: [LocationAddr]    bit-to-site mapping
│
├── paths: Option<[TransportPath]>         optional AOD transport paths
│   └── TransportPath { lane: u64, waypoints: [[f64; 2]] }
│
├── feed_forward: bool                     mid-circuit measurement + classical feedback
└── atom_reloading: bool                   atom reload after initial fill
```
