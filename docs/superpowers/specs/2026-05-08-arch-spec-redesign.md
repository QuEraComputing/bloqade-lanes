# ArchSpec Redesign Proposal: Decoupled Physical Layout and Address Space

## Executive Summary

The current `ArchSpec` entangles physical hardware geometry with the virtual addressing scheme, making it impossible to evolve either independently. This proposal splits the description into two stored layers — [`PhysicalSpec`](#layer-1-physicalspec) (flair-owned hardware geometry) and [`AddressSpace`](#layer-2-addressspace) (lanes-owned virtual addressing) — a derived artifact [`MachineModel`](#layer-3-machinemodel) (computed on demand, never stored), and a [`NameBridge`](#combined-archspec) index that pins the stable integer identifiers used in compiled programs. Bus identifiers are promoted from zone-local integers to stable human-readable string keys.

**What this fixes:**

- Zone word/site addressing can change without touching [`PhysicalSpec`](#layer-1-physicalspec) or any bus definition.
- New bus types are first-class entries in `PhysicalSpec.buses` and require no changes to the rewrite pipeline — string keys drive downstream kernel dispatch directly (see [Impact on Downstream Consumers](#impact-on-downstream-consumers)).
- `bus_id` is globally stable — it is the list index into `NameBridge.bus_names`, not a zone-local position — so reordering or extending `PhysicalSpec.buses` never silently invalidates compiled addresses (see [New `LaneAddress` Encoding](#new-laneaddress-encoding)).
- `MoveType` (SITE / WORD / ZONE) disappears from the encoding; `MachineModel.bus_graph` is the single source of truth for move semantics (see [Impact on Downstream Consumers](#impact-on-downstream-consumers)).
- Ownership is explicit: flair evolves `PhysicalSpec`, lanes evolves `AddressSpace`, and `MachineModel` is always a pure function of the two (see [Ownership and Responsibility](#ownership-and-responsibility)).
- Architectural [subset checking](#subset-checking-arch_spec_1--arch_spec_2) enables backward-compatibility validation at the spec level — if `arch_spec_1 ⊆ arch_spec_2`, any compiled move program that is valid under `arch_spec_1` is also valid under `arch_spec_2` without recompilation. This requires `PhysicalSpec`, `AddressSpace`, and `NameBridge` all to satisfy the subset relation — `NameBridge` stability is what ensures encoded integer IDs remain valid across specs.

## Motivation

The current `ArchSpec` couples bus definitions to the word/site address space. Buses are defined as index-level src→dst pairs within a zone (`SiteBus.src/dst` are site indices; `WordBus.src/dst` are word indices). This means:

- Changing zone word/site addressing invalidates all bus definitions
- Adding new bus types (e.g. right-direction AOD moves) requires extending a hardcoded per-zone list
- The `bus_id` in `LaneAddress` is zone-local, not globally unique
- `MoveType` (SITE / WORD / ZONE) is an artifact of how buses were structured, not a fundamental property of a move

## Proposed Design

Split the architecture description into three layers:

- **[`PhysicalSpec`](#layer-1-physicalspec)** — flair-owned, hardware-only. Describes SLM grids, AOD buses, Rydberg top-hat beams, and device capabilities (`feed_forward`, `atom_reloading`). No zones, words, or sites. Uses human-readable string keys throughout so downstream consumers (e.g. `lanes2flair`) can dispatch on bus name without hardcoded integer IDs.
- **[`AddressSpace`](#layer-2-addressspace)** — lanes-owned, virtual addressing. Defines how grid positions are grouped into words and how words are partitioned into sites. A single `word_shape: tuple[int, int]` applies uniformly across all zones; `site_slices` partitions that abstract grid into named sites; `zones` is a list of `AddressMapping`s (one per zone_id).
- **[`MachineModel`](#layer-3-machinemodel)** — derived artifact, computed on demand by calling `ArchSpec.derive()`, never stored. Resolves the two stored layers into a `BusGraph` of virtual `LocationAddress` src/dst pairs, a `GateInfo` with CZ pairs per top-hat and local gate site addresses, and a `CapacityInfo` with word/site counts. An optional `NoiseModel` is included when `derive(physical_noise_spec=...)` is called.

These three layers combine into a single **[`ArchSpec`](#combined-archspec)** object, plus a **[`NameBridge`](#combined-archspec)** that pins the integer indices for zone names, top-hat keys, and bus keys — ensuring compiled programs remain valid across spec revisions.

### Layer 1: `PhysicalSpec`

Pure physical description — no virtual addressing.

```python
class Mode:
    name: str               # human-readable mode name
    grid_keys: list[str]    # ordered slm_grid keys; order = bitstring order across grids


class TopHat:
    y_min: float            # lower edge of the top-hat beam (µm)
    y_max: float            # upper edge of the top-hat beam (µm)
    y_min_keepout: float    # keepout boundary below the beam
    y_max_keepout: float    # keepout boundary above the beam


class PhysicalSpec:
    slm_grids: dict[str, Grid]           # grid_key → Grid; human-readable stable key
    buses: dict[str, PhysicalBus]        # bus_key → PhysicalBus; human-readable stable key
    modes: dict[int, Mode]               # mode_index → Mode
    local_gate_grids: set[str]           # grid_keys where local single-qubit gates are supported
    rydberg_tophats: dict[str, TopHat]   # tophat_key → TopHat; named Rydberg top-hat beam regions
    blockade_radius: float               # Rydberg blockade radius (µm)
    feed_forward: bool                   # supports mid-circuit measurement with classical feedback
    atom_reloading: bool                 # supports reloading atoms after initial fill
```

```python
class Grid:
    x_start: float                # x coordinate of the first column (µm)
    y_start: float                # y coordinate of the first row (µm)
    x_spacing: tuple[float, ...]  # per-column x spacings; len = number of columns - 1
    y_spacing: tuple[float, ...]  # per-row y spacings; len = number of rows - 1


class PhysicalBus:
    src: Grid
    dst: Grid
    waypoints: list[Grid] | None = None   # intermediate stops, excluding src and dst
```

`Grid` defines a 2D array of atom positions via a start coordinate and per-gap spacings; `len(x_spacing) + 1` gives the number of columns and `len(y_spacing) + 1` gives the number of rows. All `Grid`s in a `PhysicalBus` have the same number of atom positions. `src` and `dst` are the start and end configurations; `waypoints` carries any intermediate physical stops for multi-hop moves (e.g. zone_1 → zone_2 → zone_3). `PhysicalSpec` makes no reference to zones, words, or sites — it describes only what atoms exist and how they move in physical (x, y) space.

**`Grid` canonical unrolling.** Every `Grid` has a canonical unrolling to `list[tuple[float, float]]`: positions are enumerated x-index fastest, y-index slowest (i.e. `[(x, y) for y in y_positions for x in x_positions]`). This convention is the single source of truth wherever a `Grid` must be flattened to an ordered position list — e.g. when matching atom indices across `PhysicalBus.src` and `PhysicalBus.dst`, or when resolving `AddressMapping.Word` indices against a grid.

`slm_grids` and `buses` use human-readable string keys — stable, meaningful, and safe to reference across the `PhysicalSpec`/`AddressSpace` boundary without leaking geometry. `modes` is keyed by integer mode index and passed through opaquely — the lanes compiler does not inspect mode definitions; they are used by downstream consumers (e.g. for bitstring ordering in `lanes2flair`).

The string bus keys also serve as the stable handle for downstream kernel dispatch: a consumer (e.g. `lanes2flair`) can maintain a `dict[str, KernelFn]` keyed by bus name and look up the correct library function directly from `PhysicalSpec.buses.keys()`, without any hardcoded integer bus IDs. This is the primary fix for the hardcoded `{0: ..., 8: ...}` dispatch that motivated this redesign.

> **Feedback requested:** Should `PhysicalSpec` group related fields into sub-objects for clarity? For example, gate-specific fields (`local_gate_grids`, `rydberg_tophats`, `blockade_radius`) could be collected into a `GateSpec` or `EntanglingSpec` object, and transport fields (`buses`, `slm_grids`) could form a `TransportSpec`. This would make the ownership boundary more explicit and allow each sub-object to be versioned or replaced independently. Alternatively, keeping everything flat in `PhysicalSpec` avoids indirection and keeps construction simple. Feedback welcome before this is implemented.

### Layer 2: `AddressSpace`

Virtual addressing layer that sits on top of `PhysicalSpec`.

```python
class Word:
    x_indices: list[int]   # indices into the parent Grid's x positions
    y_indices: list[int]   # indices into the parent Grid's y positions


class AddressMapping:
    words: list[Word]  # word_id = list index; partial partition of the referenced Grid
```

`Word` is a slice of a `Grid` by physical position. Partial grid coverage is allowed — not every grid position needs to be addressed.

**Validation rules for `AddressMapping`** (self-contained; no reference to `PhysicalSpec`):
1. Must contain at least one word — empty `AddressMapping`s are not allowed.
2. No two words share a grid position — Cartesian products of `x_indices` × `y_indices` do not overlap across words.

Bounds checking of `x_indices`/`y_indices` against the physical grid is a cross-layer rule covered in `ArchSpec` validation.

```python
class SiteSlice:
    x_indices: list[int]   # offsets within the word's abstract n_x × n_y grid
    y_indices: list[int]   # offsets within the word's abstract n_x × n_y grid


class AddressSpace:
    word_shape: tuple[int, int]        # (n_x, n_y) — abstract word grid dimensions
    site_slices: list[SiteSlice]       # site_id = list index; must fully cover n_x × n_y
    zones: list[AddressMapping]        # zone_id = list index
```

```python
class LocationAddress:
    zone_id: int   # list index into AddressSpace.zones
    word_id: int   # list index into AddressMapping.words
    site_id: int   # list index into AddressSpace.site_slices
```

Throughout this section, `n_x = word_shape[0]` and `n_y = word_shape[1]`.

**Validation rules for `AddressSpace`** (self-contained):
1. Every word in every `AddressMapping` must conform to `word_shape`: `len(word.x_indices) == n_x` and `len(word.y_indices) == n_y`.
2. All `x_indices` values in every `SiteSlice` are in `range(n_x)`; all `y_indices` in `range(n_y)`.
3. No two sites share an abstract grid position (no overlap across `SiteSlice`s).
4. The union of all sites covers all `n_x * n_y` positions (full coverage required).

`word_shape` declares the abstract grid dimensions shared by every word across all zones. `site_slices` partitions that grid into named sites — the site_id is the list index. Separating `word_shape` from `site_slices` means two `AddressSpace`s can share the same `(n_x, n_y)` while using different site partitionings, enabling transformations between architectures by updating only the slicing.

`x_indices` and `y_indices` in each `SiteSlice` are offsets into `range(n_x)` and `range(n_y)` respectively. Unlike `AddressMapping` (partial coverage OK), `site_slices` **must fully cover** all `n_x × n_y` positions.

`site_id` is the list index into `site_slices`. Since `site_id` appears directly in the `LaneAddress` encoding, the list order must be stable — it is never independently reordered.

A `LocationAddress` is therefore `(zone_id, word_id, site_id)` where `site_id` keys into `site_slices`. The physical positions covered by a site are the Cartesian product: `{(Word.x_indices[xi], Word.y_indices[yi]) for xi in SiteSlice.x_indices for yi in SiteSlice.y_indices}` — where `SiteSlice.x_indices` are offsets into `range(n_x)` and `Word.x_indices[xi]` is the actual grid column index.

`zones` is a list where the list index is the `zone_id`, mapping directly into the `LaneAddress` encoding. `ArchSpec.name_bridge.zone_names[zone_id]` gives the human-readable name. `zone_id` is NOT derivable from `word_id` — different zones can reference grids of different sizes, and `word_id`s are local to each zone's `AddressMapping`.

### Derived Artifact: `MachineModel`

`MachineModel` is emitted by calling `ArchSpec.derive()` — it is not stored inside `ArchSpec`. All fields are pure functions of `PhysicalSpec`, `AddressSpace`, and `NameBridge`.

```python
class Bus:
    src: list[LocationAddress]
    dst: list[LocationAddress]


class CZPair:  # must be frozen/hashable — used as a dict key in NoiseModel.cz_noise
    a: LocationAddress
    b: LocationAddress


class CapacityInfo:
    num_sites_per_word: int       # number of sites in every word (uniform across the address space)
    words_per_zone: list[int]     # zone_id → number of words in that zone


class TopHatCZ:
    cz_pairs: list[CZPair]   # entangling pairs within this top-hat beam


class GateInfo:
    top_hats: list[TopHatCZ]                     # ordered by NameBridge.top_hat_names; top-hat index is NOT in the LaneAddress encoding
    local_gate_addresses: list[LocationAddress]  # sites addressable by local single-qubit gates


class BusGraph:
    buses: list[Bus]   # bus_id = list index; matches NameBridge.bus_names and the LaneAddress encoding


class MachineModel:
    arch_spec: ArchSpec              # back-reference to the parent ArchSpec (e.g. for visualization)
    capacity: CapacityInfo
    gate_info: GateInfo
    bus_graph: BusGraph
    noise: NoiseModel | None         # None when derived without a PhysicalNoiseSpec
```

**CZ pairs are not stored in `ArchSpec` — they are computed by `derive()`.** For each name in `NameBridge.top_hat_names` (in list order), all pairs of addressed sites whose physical positions both fall within `[y_min, y_max]` and whose mutual distance is at or below `PhysicalSpec.blockade_radius` are collected into a `TopHatCZ` and emitted as `MachineModel.gate_info.top_hats`. There is no intra-zone word-word pairing constraint — any two addressable sites that co-occur within a top-hat beam and are within blockade range form a CZ pair.

`bus_graph` is produced by resolving the physical grid positions in each `PhysicalBus` through the zone `AddressMapping`s. `Bus` captures only the virtual src/dst `LocationAddress` pairs; intermediate waypoints remain in `PhysicalSpec`. The integer `bus_id` for each bus is its index in `NameBridge.bus_names`.

> **Compiler gap:** In the current bytecode, the top-hat index is tied to the zone index — there is no independent top-hat identifier in the `LaneAddress` encoding. The `GateInfo.top_hats` list (indexed by `NameBridge.top_hat_names`) therefore represents a logical concept that the compiler currently resolves implicitly through zone membership. A new bytecode statement will likely be needed to surface the top-hat index as a first-class encoded field, decoupling it from zone identity.

`gate_info.local_gate_addresses` is derived by collecting all `LocationAddress`es whose zone maps (via `ArchSpec.name_bridge.zone_names[zone_id]`, which equals the grid key by `ArchSpec` validation rule 1) to a grid key present in `PhysicalSpec.local_gate_grids`.

`capacity.words_per_zone` is indexed by `zone_id` and gives the number of words in `AddressSpace.zones[zone_id].words`.

**Address space / bus compatibility check.** Because all words have the same shape (uniform sites-per-word), bus compatibility is structural: each `PhysicalBus` in `PhysicalSpec.buses` is compatible with the `AddressSpace` if the number of positions in `src` (and `dst`) is a multiple of `capacity.num_sites_per_word`. The derivation does not need to verify that every grid position is covered by a word — partial `AddressMapping` coverage is fine as long as the shape invariant holds.

### `NameBridge`

`NameBridge` is the index that pins every integer identifier used in compiled programs to a human-readable string key in `PhysicalSpec`. It is the bridge between the string-keyed world of `PhysicalSpec` (where bus names, grid names, and top-hat names are stable identifiers) and the integer-keyed world of the `LaneAddress` encoding (where `zone_id`, `bus_id`, and `site_id` are compact bit fields). Without `NameBridge`, adding or reordering entries in `PhysicalSpec.buses` or `PhysicalSpec.rydberg_tophats` could silently shift integer IDs and corrupt compiled programs. With it, the mapping is explicit and stable: entries may only be appended, never reordered or inserted.

`NameBridge` is owned jointly — flair contributes the string keys (by defining entries in `PhysicalSpec`) and lanes assigns the integer positions (by appending to the lists). In practice, a new bus added by flair requires a coordinated append to `NameBridge.bus_names` before any compiled program can reference it.

```python
class NameBridge:
    zone_names: list[str]     # zone_id = list index; value is the zone name AND the PhysicalSpec.slm_grids key
    top_hat_names: list[str]  # top-hat index = list index; value is the top-hat name AND the PhysicalSpec.rydberg_tophats key
    bus_names: list[str]      # bus_id = list index; value is the bus name AND the PhysicalSpec.buses key


class ArchSpec:
    physical_spec: PhysicalSpec   # slm_grids, buses, modes, local_gate_grids, rydberg_tophats, blockade_radius, feed_forward, atom_reloading
    address_space: AddressSpace   # word_shape (n_x, n_y), site_slices, zones
    name_bridge: NameBridge       # zone_names, top_hat_names, bus_names — list-indexed string↔int mappings across the PhysicalSpec/AddressSpace boundary

    def derive(self) -> MachineModel: ...                                          # derive without noise
    def derive(self, physical_noise_spec: PhysicalNoiseSpec) -> MachineModel: ...  # derive with noise
```

**Validation rules for `ArchSpec`** (cross-layer; requires both `PhysicalSpec` and `AddressSpace`):
1. Every name in `name_bridge.zone_names` must exist as a key in `PhysicalSpec.slm_grids` — zone name = grid name.
2. `len(address_space.zones)` must equal `len(name_bridge.zone_names)` — every zone_id (list index) must have a corresponding entry in `name_bridge.zone_names`. Together, rules 1 and 2 establish a bijection: each `zone_id` maps to exactly one `zone_names` entry, which maps to exactly one `slm_grids` grid.
3. For each zone, all `x_indices` and `y_indices` in every `Word` must be within the bounds of `PhysicalSpec.slm_grids[name_bridge.zone_names[zone_id]]`.
4. `name_bridge.top_hat_names` must contain exactly the elements of `PhysicalSpec.rydberg_tophats.keys()`, in a stable order — entries may only be appended when the spec is extended, never reordered or inserted.
5. `name_bridge.bus_names` must contain exactly the elements of `PhysicalSpec.buses.keys()`, in a stable order — entries may only be appended when the spec is extended, never reordered or inserted.
6. For each `PhysicalBus` in `PhysicalSpec.buses`, the number of atom positions in `src` (and `dst`) must be a multiple of `n_x * n_y * len(address_space.site_slices)` (i.e. a whole number of words).

## New `LaneAddress` Encoding

### Current (64-bit)

```
data0: [ word_id : 16 ][ site_id : 16 ]
data1: [ dir : 1 ][ mt : 2 ][ zone_id : 8 ][ pad : 5 ][ bus_id : 16 ]
```

- `mt` (MoveType): 2 bits encoding SITE / WORD / ZONE — **removed**
- `bus_id`: 16-bit zone-local index — **promoted to a globally-scoped list index into `NameBridge.bus_names` and `MachineModel.bus_graph.buses`**

### Proposed (64-bit)

```
data0: [ word_id : 16 ][ site_id : 16 ]
data1: [ dir : 1 ][ pad : 7 ][ zone_id : 8 ][ bus_id : 16 ]
```

The 2 bits freed by removing `mt` are absorbed into `pad` (5 → 7 bits). `pad` moves before `zone_id` to keep `zone_id` and `bus_id` contiguous in the lower 24 bits. `word_id`, `site_id`, and `dir` field widths and positions are unchanged.

**Field semantics:**

| Field | Bits | Range | Notes |
|---|---|---|---|
| `word_id` | 16 | 0..65535 | List index into the zone's `AddressMapping.words` |
| `site_id` | 16 | 0..65535 | List index into `AddressSpace.site_slices` |
| `dir` | 1 | 0=FORWARD, 1=BACKWARD | Transport direction along the bus |
| `pad` | 7 | — | Reserved, must be zero |
| `zone_id` | 8 | 0..255 | List index into `AddressSpace.zones`; `ArchSpec.name_bridge.zone_names[zone_id]` gives the human-readable name |
| `bus_id` | 16 | 0..65535 | List index into `ArchSpec.name_bridge.bus_names` and `MachineModel.bus_graph.buses` |

`PhysicalSpec.buses` uses human-readable string keys; the integer `bus_id` is the list index into `ArchSpec.name_bridge.bus_names` — the same way `zone_id` is the list index into `zone_names` and the top-hat index is the list index into `top_hat_names`. This ensures that compiled programs remain valid across spec revisions: adding a new bus to `PhysicalSpec.buses` does not shift the integer IDs of existing buses. `zone_id` remains a separate field from `word_id` because zones can reference differently-sized grids and `word_id`s are not unique across zones.

## Impact on Downstream Consumers

- **`gemini.logical` dialect**: `SiteBusMove` and `WordBusMove` collapse into a single `BusMove` statement; `bus_id` is the list index into `MachineModel.bus_graph.buses`. `MoveType` attribute is removed.
- **`lanes2flair` rewrites** (bloqade-flair): the hardcoded `{bus_id: lib_fn}` dict in `RewriteSiteBusMove` is replaced by a lookup driven by `MachineModel.bus_graph`. New bus types (right-direction, etc.) are handled without modifying the rewrite rule.
- **`iter_all_lanes` / `check_lane_group`**: rewritten in terms of `MachineModel.bus_graph` rather than per-zone bus enumeration.
- **Validation**: bus membership checks operate on `LocationAddress` lists directly, no `MoveType` dispatch needed.

## Ownership and Responsibility

| Layer | Owner | Can change independently? |
|---|---|---|
| `PhysicalSpec` (slm_grids, buses, modes, local_gate_grids, rydberg_tophats, blockade_radius, feed_forward, atom_reloading) | **bloqade-flair** | Yes — flair defines and evolves the physical hardware description |
| `AddressSpace` (word_shape, site_slices, zones) | **bloqade-lanes** | Yes — the lanes compiler owns the virtual address space |
| `NameBridge` (zone_names, top_hat_names, bus_names) | **Jointly** — flair contributes string keys, lanes assigns integer positions | No — changes require coordination; new entries may only be appended |
| `MachineModel` (bus_graph, gate_info, capacity) | Neither — computed | Regenerated whenever either input changes |

This boundary means:
- The flair team can add new bus types (e.g. right-direction AOD moves), resize grids, or adjust the blockade radius without touching anything in bloqade-lanes.
- The lanes compiler can redefine how words slice a grid, add zones, or change site numbering without touching `PhysicalSpec`.
- `MachineModel` is always a pure function of the two — no state is shared across the boundary.

## Subset Checking: `arch_spec_1 ⊆ arch_spec_2`

A key goal of this design is to support checking whether one `ArchSpec` is a subset of another — i.e. any circuit that compiles and runs correctly under `arch_spec_1` will also compile and run correctly under `arch_spec_2`.

The dict-keyed structure makes this check compositional: each layer is checked independently.

### `PhysicalSpec` subsetting

`physical_spec_1 ⊆ physical_spec_2` iff:
- `physical_spec_1.slm_grids.keys() ⊆ physical_spec_2.slm_grids.keys()` and all shared keys map to equal `Grid` values.
- Every bus key in `physical_spec_1.buses` exists in `physical_spec_2.buses` with an equal `PhysicalBus` value.
- `physical_spec_1.local_gate_grids ⊆ physical_spec_2.local_gate_grids` (every local-gate-capable grid in spec_1 must also support local gates in spec_2).
- `physical_spec_1.rydberg_tophats.keys() ⊆ physical_spec_2.rydberg_tophats.keys()` and all shared keys map to equal `TopHat` values.
- `physical_spec_1.blockade_radius` ≥ `physical_spec_2.blockade_radius` (a stricter radius is a subset — fewer CZ pairs are permitted).
- If `physical_spec_1.feed_forward` is `True`, then `physical_spec_2.feed_forward` must also be `True` — a circuit compiled for a feed-forward architecture cannot run on a non-feed-forward one, but the reverse is fine.
- If `physical_spec_1.atom_reloading` is `True`, then `physical_spec_2.atom_reloading` must also be `True` — same logic as feed-forward.

```python
def is_subset(self: PhysicalSpec, other: PhysicalSpec) -> bool:
    # slm_grids: every grid in self must exist in other with identical geometry
    if not self.slm_grids.keys() <= other.slm_grids.keys():
        return False
    if any(self.slm_grids[k] != other.slm_grids[k] for k in self.slm_grids):
        return False

    # buses: every bus in self must exist in other with identical src/dst/waypoints
    if not self.buses.keys() <= other.buses.keys():
        return False
    if any(self.buses[k] != other.buses[k] for k in self.buses):
        return False

    # local_gate_grids: self must not require local gates on a grid other doesn't support
    if not self.local_gate_grids <= other.local_gate_grids:
        return False

    # rydberg_tophats: every top-hat in self must exist in other with identical geometry
    if not self.rydberg_tophats.keys() <= other.rydberg_tophats.keys():
        return False
    if any(self.rydberg_tophats[k] != other.rydberg_tophats[k] for k in self.rydberg_tophats):
        return False

    # blockade_radius: self must be at least as strict (stricter radius = fewer CZ pairs = subset)
    if self.blockade_radius < other.blockade_radius:
        return False

    # feed_forward: a feed-forward circuit cannot run on a non-feed-forward architecture
    if self.feed_forward and not other.feed_forward:
        return False

    # atom_reloading: same logic as feed_forward
    if self.atom_reloading and not other.atom_reloading:
        return False

    return True
```

### `AddressSpace` subsetting

`address_space_1 ⊆ address_space_2` iff:
- `address_space_1.word_shape == address_space_2.word_shape` — the `(n_x, n_y)` grid dimensions must be identical.
- `address_space_1.site_slices == address_space_2.site_slices` — the site partitioning must be identical (same number of sites, same slicing); `site_id` appears in the `LaneAddress` encoding, so any change to the slicing would corrupt compiled programs.
- `len(address_space_1.zones)` ≤ `len(address_space_2.zones)` — every zone_id (list index) in spec_1 exists in spec_2.
- For each zone_id, every `word_id` in `address_space_1.zones[zone_id].words` exists at the same index in `address_space_2.zones[zone_id].words` with identical `x_indices`/`y_indices`.

### `NameBridge` subsetting

`name_bridge_1 ⊆ name_bridge_2` iff:
- `name_bridge_1.zone_names[i] == name_bridge_2.zone_names[i]` for all `i < len(name_bridge_1.zone_names)`, and `len(name_bridge_1.zone_names) ≤ len(name_bridge_2.zone_names)` — the list index is the zone_id in the `LaneAddress` encoding, so position must be preserved.
- `name_bridge_1.top_hat_names[i] == name_bridge_2.top_hat_names[i]` for all `i < len(name_bridge_1.top_hat_names)`, and `len(name_bridge_1.top_hat_names) ≤ len(name_bridge_2.top_hat_names)` — the list index positions `TopHatCZ` entries in `MachineModel.gate_info.top_hats` and must be stable.
- `name_bridge_1.bus_names[i] == name_bridge_2.bus_names[i]` for all `i < len(name_bridge_1.bus_names)`, and `len(name_bridge_1.bus_names) ≤ len(name_bridge_2.bus_names)` — the list index is the bus_id in the `LaneAddress` encoding; a position shift would silently corrupt compiled move programs.

In practice this means new entries may only be appended to the end of each list — inserting or removing entries in the middle breaks all compiled programs that reference subsequent indices.

### `MachineModel` subsetting

`machine_model_1 ⊆ machine_model_2` is a consequence of the above — it does not need to be checked directly. If both `PhysicalSpec` and `AddressSpace` satisfy the subset relation, `MachineModel` is guaranteed to follow:
- Every `bus_id` in `range(len(machine_model_1.bus_graph.buses))` will have a corresponding entry in `machine_model_2.bus_graph.buses` at the same index with the same `src`/`dst`.
- For each `TopHatCZ` in `machine_model_1.gate_info.top_hats`, the corresponding `TopHatCZ` in `machine_model_2.gate_info.top_hats` will contain a superset of its `cz_pairs` (as a set — list order within `cz_pairs` is not guaranteed to be preserved).
- Every address in `machine_model_1.gate_info.local_gate_addresses` will exist in `machine_model_2.gate_info.local_gate_addresses`.

### `NoiseModel` subsetting

`NoiseModel` subsetting is not defined. Noise compatibility is a separate concern from architectural compatibility — whether a circuit's noise budget holds on a given device is a simulation/analysis question, not an `ArchSpec` subset check.

### Why `NameBridge` uses lists

All integer identifiers that appear in the `LaneAddress` encoding — `zone_id`, `bus_id`, and the top-hat index — are pinned by list position in `NameBridge`. The list index IS the integer: `zone_names[zone_id]`, `bus_names[bus_id]`, `top_hat_names[top_hat_index]`. This is the same pattern used throughout the design (`zones: list[AddressMapping]`, `site_slices: list[SiteSlice]`): stable integer identifiers are list indices, never inferred from dict key ordering. `PhysicalSpec.buses` and `PhysicalSpec.rydberg_tophats` remain dicts with human-readable string keys for their own equality and subset checks, but it is `NameBridge` that controls which integer is assigned to each key.

## Open Question: Noise Model

> **Status: open — sketch only, not yet designed.**

The noise model is a natural third derived artifact alongside `MachineModel`. The rough shape:

- `NoiseModel` is derived from `PhysicalSpec` — gate fidelities, coherence times, SPAM errors, and crosstalk are all physical hardware properties that live alongside `PhysicalSpec`.
- Once derived, the noise model can be attached to `MachineModel` to produce an enriched model suitable for noisy simulation or error-aware compilation.

The proposed API shape:

```python
class BusNoiseParams:
    # TBD — transport error rates, timing jitter, etc.
    ...


class SPAMNoiseParams:
    # TBD — state preparation and measurement error rates per grid
    ...


class TopHatNoiseParams:
    fidelity: float   # CZ entangling gate fidelity for this top-hat beam
    # expandable: crosstalk, timing errors, leakage, etc.


class LocalGateNoiseParams:
    fidelity: float   # local single-qubit gate fidelity for this grid
    # expandable: per-axis error rates, pulse distortion, etc.


class GlobalGateNoiseParams:
    fidelity: float   # global single-qubit gate fidelity (e.g. global Rydberg pulse)
    # expandable: inhomogeneity, pulse area error, etc.


class PhysicalNoiseSpec:
    bus_noise: dict[str, BusNoiseParams]              # per-bus transport noise, keyed by bus_key
    cz_noise: dict[str, TopHatNoiseParams]            # per-top-hat CZ noise, keyed by top_hat_key
    local_gate_noise: dict[str, LocalGateNoiseParams] # per-grid local gate noise, keyed by grid_key
    global_gate_noise: GlobalGateNoiseParams          # global gate noise (applies to all atoms)
    spam_noise: dict[str, SPAMNoiseParams]            # per-grid SPAM noise, keyed by grid_key
    T1: float                                         # longitudinal coherence time (µs)
    T2: float                                         # transverse coherence time (µs)


class CZNoiseParams:
    fidelity: float   # CZ gate fidelity for this pair
    # Intentionally a separate type from TopHatNoiseParams — per-pair parameters (e.g. crosstalk,
    # leakage asymmetry) are expected to diverge from per-beam parameters as the noise model matures.


class NoiseModel:
    global_gate_noise: GlobalGateNoiseParams
    cz_noise: dict[CZPair, CZNoiseParams]                        # per CZ pair; derived by expanding TopHatNoiseParams across all pairs in each top-hat
    local_gate_noise: dict[LocationAddress, LocalGateNoiseParams] # per site; derived by expanding per-grid noise across all LocationAddresses in that grid
    bus_noise: list[BusNoiseParams]                               # bus_id = list index, matching NameBridge.bus_names
    spam_noise: dict[LocationAddress, SPAMNoiseParams]            # per site; derived by expanding per-grid SPAM noise
    T1: float
    T2: float


# PhysicalNoiseSpec is a peer of PhysicalSpec, not nested inside it.
# See ArchSpec.derive() for the full API.
```

`PhysicalNoiseSpec` shares string keys with `PhysicalSpec` — every key in `cz_noise` must exist in `PhysicalSpec.rydberg_tophats`, every key in `local_gate_noise` and `spam_noise` must exist in `PhysicalSpec.slm_grids`, and every key in `bus_noise` must exist in `PhysicalSpec.buses`. Using structured objects (`TopHatNoiseParams`, `LocalGateNoiseParams`, `GlobalGateNoiseParams`) rather than bare floats allows finer-grained noise parameters to be added later without breaking the interface.

**Open questions:**
- Should `T1`/`T2` be global scalars or per-grid (atoms in different grids may have different coherence)?
