# ArchSpec Redesign Proposal: Decoupled Physical Layout and Address Space

## Motivation

The current `ArchSpec` couples bus definitions to the word/site address space. Buses are defined as index-level src→dst pairs within a zone (`SiteBus.src/dst` are site indices; `WordBus.src/dst` are word indices). This means:

- Changing zone word/site addressing invalidates all bus definitions
- Adding new bus types (e.g. right-direction AOD moves) requires extending a hardcoded per-zone list
- The `bus_id` in `LaneAddress` is zone-local, not globally unique
- `MoveType` (SITE / WORD / ZONE) is an artifact of how buses were structured, not a fundamental property of a move

## Proposed Design

Split the architecture description into three layers:

### Layer 1: `PhysicalSpec`

Pure physical description — no logical addressing.

```python
class Mode:
    name: str               # human-readable mode name
    grid_keys: list[str]    # ordered slm_grid keys; order = bitstring order across grids


class PhysicalSpec:
    slm_grids: dict[str, Grid]        # grid_key → Grid; human-readable stable key
    buses: dict[str, PhysicalBus]     # bus_key → PhysicalBus; human-readable stable key
    modes: dict[int, Mode]            # mode_index → Mode
    blockade_radius: float             # Rydberg blockade radius (µm)
    feed_forward: bool                # supports mid-circuit measurement with classical feedback
    atom_reloading: bool              # supports reloading atoms after initial fill
```

```python
class PhysicalBus:
    src: Grid
    dst: Grid
    waypoints: list[Grid] | None = None   # intermediate stops, excluding src and dst
```

`Grid` is the same `Grid` already defined in the Rust library. All `Grid`s in a `PhysicalBus` have the same number of atom positions. `src` and `dst` are the start and end configurations; `waypoints` carries any intermediate physical stops for multi-hop moves (e.g. zone_1 → zone_2 → zone_3). `PhysicalSpec` makes no reference to zones, words, or sites — it describes only what atoms exist and how they move in physical (x, y) space.

`slm_grids` and `buses` use human-readable string keys — stable, meaningful, and safe to reference across the `PhysicalSpec`/`AddressSpace` boundary without leaking geometry. `PhysicalSpec` overlap is key intersection: `spec_a.slm_grids.keys() & spec_b.slm_grids.keys()` (with value equality check for shared keys). `modes` uses the mode name as key directly.

### Layer 2: `AddressSpace`

Logical addressing layer that sits on top of `PhysicalSpec`.

```python
class Word:
    x_indices: list[int]   # indices into the parent Grid's x positions
    y_indices: list[int]   # indices into the parent Grid's y positions


class AddressMapping:
    words: list[Word]  # word_id = list index; partial partition of the referenced Grid
```

`Word` is a slice of a `Grid` by physical position. Partial grid coverage is allowed — not every grid position needs to be addressed.

**Validation rules for `AddressMapping`** (validated against `PhysicalSpec` using the zone name from `AddressSpace.zone_labels`):
1. The zone name must exist as a key in `PhysicalSpec.slm_grids` — zone name = grid name.
2. Every word must conform to `AddressSpace.word_shape`: `len(word.x_indices) == word_shape.n_x` and `len(word.y_indices) == word_shape.n_y`.
3. No two words share a grid position (indices do not overlap across words).
4. All `x_indices` and `y_indices` in every word are within the bounds of `PhysicalSpec.slm_grids[zone_name]`.

```python
class SiteSlice:
    x_indices: list[int]   # offsets within the word's abstract n_x × n_y grid
    y_indices: list[int]   # offsets within the word's abstract n_x × n_y grid


class WordShape:
    n_x: int                    # width of the abstract word grid
    n_y: int                    # height of the abstract word grid
    sites: list[SiteSlice]      # site_id = list index; must fully cover n_x × n_y


class AddressSpace:
    word_shape: WordShape              # single declared shape for all words across all zones
    zones: dict[int, AddressMapping]   # zone_id → AddressMapping
```

`WordShape` defines the internal structure of every word: an abstract `n_x × n_y` grid partitioned into sites via `SiteSlice`. Unlike `AddressMapping` (partial coverage OK), the `sites` in `WordShape` **must fully cover** all `n_x × n_y` positions — no position may be left unaddressed. `x_indices` and `y_indices` in each `SiteSlice` are offsets into `range(n_x)` and `range(n_y)` respectively.

`site_id` is the list index into `sites`. The list pattern is safe here because `WordShape` is unique to an architecture — it is never independently reordered or shared across contexts the way buses or zones are.

A `LocationAddress` is therefore `(zone_id, word_id, site_id)` where `site_id` keys into `WordShape.sites`, and the physical position resolves as `(Word.x_indices[SiteSlice.x_indices[i]], Word.y_indices[SiteSlice.y_indices[i]])` for each position `i` within the site.

**Validation rules for `WordShape`:**
1. All `x_indices` values in every `SiteSlice` are in `range(n_x)`; all `y_indices` in `range(n_y)`.
2. No two sites share an abstract grid position (no overlap across `SiteSlice`s).
3. The union of all sites covers all `n_x * n_y` positions (full coverage required).

**CZ pairs are inferred, not stored.** CZ pairs are computed globally by finding all `LocationAddress` pairs whose physical distance is within `PhysicalSpec.blockade_radius`.

`zones` is keyed by integer `zone_id`, which maps directly into the `LaneAddress` encoding. `zone_labels` provides the human-readable names. `zone_id` is NOT derivable from `word_id` — different zones can reference grids of different sizes, and `word_id`s are local to each zone's `AddressMapping`.

### Layer 3: `DerivedSpec`

All items inferred from `PhysicalSpec` + `AddressSpace` together live in a single explicit object:

```python
class Bus:
    src: list[LocationAddress]
    dst: list[LocationAddress]


class CZPair:
    a: LocationAddress
    b: LocationAddress


class DerivedSpec:
    bus_graph: dict[int, Bus]   # bus_id → Bus; keyed by PhysicalSpec.buses key
    cz_pairs: list[CZPair]      # global list of entangling pairs within blockade_radius
```

`bus_graph` is produced by resolving the physical grid positions in each `PhysicalSpec.PhysicalBus` through the zone `AddressMapping`s. `Bus` captures only the logical src/dst `LocationAddress` pairs; intermediate waypoints remain in `PhysicalSpec`.

`cz_pairs` is a flat global list of `LocationAddress` pairs whose physical distance is within `PhysicalSpec.blockade_radius`. There is no assumption that pairs are intra-zone or between whole words — any two addressable sites anywhere in the layout can form a CZ pair.

`DerivedSpec` is constructed by a factory that takes `PhysicalSpec` and `AddressSpace` as inputs after both have been individually validated.

**Address space / bus compatibility check.** Because all words have the same shape (uniform sites-per-word), bus compatibility is structural: each `PhysicalBus` in `PhysicalSpec.buses` is compatible with the `AddressSpace` if the number of positions in `src` (and `dst`) is a multiple of the sites-per-word. The derivation does not need to verify that every grid position is covered by a word — partial `AddressMapping` coverage is fine as long as the shape invariant holds.

### Combined: `ArchSpec`

```python
ArchSpec(
    physical_spec: PhysicalSpec,        # slm_grids, buses, modes, blockade_radius, feed_forward, atom_reloading
    address_space: AddressSpace,        # zones (AddressMapping dict), word_shape
    zone_labels: dict[str, int],        # zone name → zone_id; shared context for AddressSpace comparison
    derived: DerivedSpec,               # bus_graph, cz_pairs — inferred from the above
)
```

## New `LaneAddress` Encoding

### Current (64-bit)

```
data0: [ word_id : 16 ][ site_id : 16 ]
data1: [ dir : 1 ][ mt : 2 ][ zone_id : 8 ][ pad : 5 ][ bus_id : 16 ]
```

- `mt` (MoveType): 2 bits encoding SITE / WORD / ZONE — **removed**
- `bus_id`: 16-bit zone-local index — **promoted to key into `PhysicalSpec.buses`**

### Proposed (64-bit)

```
data0: [ word_id : 16 ][ site_id : 16 ]
data1: [ dir : 1 ][ pad : 7 ][ zone_id : 8 ][ bus_id : 16 ]
```

The 2 bits freed by removing `mt` are absorbed into `pad` (5 → 7 bits). All other field widths and positions are unchanged.

**Field semantics:**

| Field | Bits | Range | Notes |
|---|---|---|---|
| `word_id` | 16 | 0..65535 | List index into the zone's `AddressMapping.words` |
| `site_id` | 16 | 0..65535 | Index into the word's `x_indices`/`y_indices` lists |
| `dir` | 1 | 0=FORWARD, 1=BACKWARD | Transport direction along the bus |
| `pad` | 7 | — | Reserved, must be zero |
| `zone_id` | 8 | 0..255 | Key into `AddressSpace.zones`; use `AddressSpace.zone_labels` for human-readable name |
| `bus_id` | 16 | 0..65535 | Index into `DerivedSpec.bus_graph`; assigned canonically during derivation |

`PhysicalSpec.buses` is a set with no integer labels — `bus_id` is assigned by `DerivedSpec` construction using a deterministic canonical ordering of the set. `zone_id` remains a separate field from `word_id` because zones can reference differently-sized grids and `word_id`s are not unique across zones.

## Impact on Downstream Consumers

- **`gemini.logical` dialect**: `SiteBusMove` and `WordBusMove` collapse into a single `BusMove` statement; `bus_id` references the `PhysicalSpec.buses` key. `MoveType` attribute is removed.
- **`lanes2flair` rewrites** (bloqade-flair): the hardcoded `{bus_id: lib_fn}` dict in `RewriteSiteBusMove` is replaced by a lookup driven by `DerivedSpec.bus_graph`. New bus types (right-direction, etc.) are handled without modifying the rewrite rule.
- **`iter_all_lanes` / `check_lane_group`**: rewritten in terms of `DerivedSpec.bus_graph` rather than per-zone bus enumeration.
- **Validation**: bus membership checks operate on `LocationAddress` lists directly, no `MoveType` dispatch needed.

## Ownership and Responsibility

| Layer | Owner | Can change independently? |
|---|---|---|
| `PhysicalSpec` (slm_grids, buses, blockade_radius, feed_forward, atom_reloading) | **bloqade-flair** | Yes — flair defines and evolves the physical hardware description |
| `AddressSpace` (AddressMapping per zone) | **bloqade-lanes** | Yes — the lanes compiler owns the logical address space |
| `DerivedSpec` (bus_graph, cz_pairs) | Neither — computed | Regenerated whenever either input changes |

This boundary means:
- The flair team can add new bus types (e.g. right-direction AOD moves), resize grids, or adjust the blockade radius without touching anything in bloqade-lanes.
- The lanes compiler can redefine how words slice a grid, add zones, or change site numbering without touching `PhysicalSpec`.
- `DerivedSpec` is always a pure function of the two — no state is shared across the boundary.

## Subset Checking: `arch_spec_1 ⊆ arch_spec_2`

A key goal of this design is to support checking whether one `ArchSpec` is a subset of another — i.e. any circuit that compiles and runs correctly under `arch_spec_1` will also compile and run correctly under `arch_spec_2`.

The dict-keyed structure makes this check compositional: each layer is checked independently.

### `PhysicalSpec` subsetting

`physical_spec_1 ⊆ physical_spec_2` iff:
- `physical_spec_1.slm_grids.keys() ⊆ physical_spec_2.slm_grids.keys()` and all shared keys map to equal `Grid` values.
- Every `bus_id` in `physical_spec_1.buses` exists in `physical_spec_2.buses` with compatible `PhysicalBus`.
- `physical_spec_1.blockade_radius` ≥ `physical_spec_2.blockade_radius` (a stricter radius is a subset — fewer CZ pairs are permitted).

### `AddressSpace` subsetting

`address_space_1 ⊆ address_space_2` iff:
- Every `zone_id` in `address_space_1.zones` exists in `address_space_2.zones`.
- For each zone, every `word_id` in `address_space_1.zones[zone_id].words` exists in `address_space_2.zones[zone_id].words` with identical `x_indices`/`y_indices`.

### `DerivedSpec` subsetting

`derived_1 ⊆ derived_2` is a consequence of the above — it does not need to be checked directly. If both `PhysicalSpec` and `AddressSpace` satisfy the subset relation, `DerivedSpec` is guaranteed to follow:
- Every `bus_id` in `derived_1.bus_graph` will exist in `derived_2.bus_graph` with the same `src`/`dst`.
- Every `CZPair` in `derived_1.cz_pairs` will exist in `derived_2.cz_pairs`.

### Why the dict structure is essential here

With list-indexed structures, subset checking would require positional alignment — any reordering of buses or zones across the two specs would break the comparison even when the specs are semantically identical. With stable dict keys, the check is purely key-and-value equality, independent of insertion order.

## What This Fixes

- Changing zone word/site addressing does not touch `PhysicalSpec` or bus definitions
- Right-direction and any future AOD bus types are first-class entries in `PhysicalSpec.buses`, not hardcoded extensions
- `bus_id` is a stable dict key, not a list position — reordering never silently invalidates encoded addresses
- `MoveType` disappears from the encoding; the bus graph is the single source of truth for move semantics
