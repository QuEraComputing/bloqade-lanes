# ArchSpec Redesign Proposal: Decoupled Physical Layout and Address Space

## Executive Summary

The current `ArchSpec` entangles physical hardware geometry with the virtual addressing scheme, making it impossible to evolve either independently. This proposal splits the description into three clean layers — [`PhysicalSpec`](#layer-1-physicalspec) (flair-owned hardware geometry), [`AddressSpace`](#layer-2-addressspace) (lanes-owned virtual addressing), and [`MachineModel`](#layer-3-machinemodel) (derived on demand) — and promotes bus identifiers from zone-local integers to stable human-readable string keys.

**What this fixes:**

- Zone word/site addressing can change without touching [`PhysicalSpec`](#layer-1-physicalspec) or any bus definition.
- Right-direction moves and any future AOD bus types are first-class entries in `PhysicalSpec.buses` — no hardcoded extensions needed. The string keys also drive downstream kernel dispatch directly, eliminating the hardcoded `{0: ..., 8: ...}` lookup in `lanes2flair` (see [Impact on Downstream Consumers](#impact-on-downstream-consumers)).
- `bus_id` is a stable dict key, not a list position — reordering never silently invalidates encoded addresses (see [New `LaneAddress` Encoding](#new-laneaddress-encoding)).
- `MoveType` (SITE / WORD / ZONE) disappears from the encoding; `MachineModel.bus_graph` is the single source of truth for move semantics (see [Impact on Downstream Consumers](#impact-on-downstream-consumers)).
- Architectural subset checking becomes compositional: [`PhysicalSpec`](#physicalspec-subsetting), [`AddressSpace`](#addressspace-subsetting), and [`MachineModel`](#machinemodel-subsetting) are each checked independently.
- Ownership is explicit: flair evolves `PhysicalSpec`, lanes evolves `AddressSpace`, and `MachineModel` is always a pure function of the two (see [Ownership and Responsibility](#ownership-and-responsibility)).

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
- **[`MachineModel`](#layer-3-machinemodel)** — computed on demand by calling `ArchSpec.derive()`, never stored. Resolves the two layers above into a `BusGraph` of virtual `LocationAddress` src/dst pairs, a `GateInfo` with CZ pairs per top-hat and local gate site addresses, and a `CapacityInfo` with word/site counts. An optional `NoiseModel` is included when `derive(physical_noise_spec=...)` is called.

These three layers combine into a single **[`ArchSpec`](#combined-archspec)** object, plus a **[`NameBridge`](#combined-archspec)** that maps human-readable zone/top-hat names to their integer indices.

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
class PhysicalBus:
    src: Grid
    dst: Grid
    waypoints: list[Grid] | None = None   # intermediate stops, excluding src and dst
```

`Grid` defines a regular 2D array of atom positions via start coordinates and per-axis spacings. All `Grid`s in a `PhysicalBus` have the same number of atom positions. `src` and `dst` are the start and end configurations; `waypoints` carries any intermediate physical stops for multi-hop moves (e.g. zone_1 → zone_2 → zone_3). `PhysicalSpec` makes no reference to zones, words, or sites — it describes only what atoms exist and how they move in physical (x, y) space.

**`Grid` canonical unrolling.** Every `Grid` has a canonical unrolling to `list[tuple[float, float]]`: positions are enumerated in row-major order — x-index varies fastest, y-index varies slowest (i.e. `[(x, y) for y in y_positions for x in x_positions]`). This convention is the single source of truth wherever a `Grid` must be flattened to an ordered position list — e.g. when matching atom indices across `PhysicalBus.src` and `PhysicalBus.dst`, or when resolving `AddressMapping.Word` indices against a grid.

`slm_grids` and `buses` use human-readable string keys — stable, meaningful, and safe to reference across the `PhysicalSpec`/`AddressSpace` boundary without leaking geometry. `PhysicalSpec` overlap is key intersection: `spec_a.slm_grids.keys() & spec_b.slm_grids.keys()` (with value equality check for shared keys). `modes` is keyed by integer mode index.

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

```python
class SiteSlice:
    x_indices: list[int]   # offsets within the word's abstract n_x × n_y grid
    y_indices: list[int]   # offsets within the word's abstract n_x × n_y grid


class AddressSpace:
    word_shape: tuple[int, int]        # (n_x, n_y) — abstract word grid dimensions
    site_slices: list[SiteSlice]       # site_id = list index; must fully cover n_x × n_y
    zones: list[AddressMapping]        # zone_id = list index
```

**Validation rules for `AddressSpace`** (self-contained):
1. Every word in every `AddressMapping` must conform to `word_shape`: `len(word.x_indices) == word_shape[0]` (n_x) and `len(word.y_indices) == word_shape[1]` (n_y).

`word_shape` declares the abstract grid dimensions shared by every word across all zones. `site_slices` partitions that grid into named sites — the site_id is the list index. Separating `word_shape` from `site_slices` means two `AddressSpace`s can share the same `(n_x, n_y)` while using different site partitionings, enabling transformations between architectures by updating only the slicing.

`x_indices` and `y_indices` in each `SiteSlice` are offsets into `range(n_x)` and `range(n_y)` respectively. Unlike `AddressMapping` (partial coverage OK), `site_slices` **must fully cover** all `n_x × n_y` positions.

`site_id` is the list index into `site_slices`. The list pattern is safe here because the slicing is unique to an architecture and is never independently reordered.

A `LocationAddress` is therefore `(zone_id, word_id, site_id)` where `site_id` keys into `site_slices`. The physical positions covered by a site are the Cartesian product: `{(Word.x_indices[xi], Word.y_indices[yi]) for xi in SiteSlice.x_indices for yi in SiteSlice.y_indices}`.

**Validation rules for `AddressSpace.site_slices`:**
1. All `x_indices` values in every `SiteSlice` are in `range(n_x)`; all `y_indices` in `range(n_y)`.
2. No two sites share an abstract grid position (no overlap across `SiteSlice`s).
3. The union of all sites covers all `n_x * n_y` positions (full coverage required).

`zones` is a list where the list index is the `zone_id`, mapping directly into the `LaneAddress` encoding. `ArchSpec.name_bridge.zone_labels` provides the human-readable names. `zone_id` is NOT derivable from `word_id` — different zones can reference grids of different sizes, and `word_id`s are local to each zone's `AddressMapping`.

### Layer 3: `MachineModel`

`MachineModel` is emitted by calling `ArchSpec.derive()` — it is not stored inside `ArchSpec`. All fields are pure functions of `PhysicalSpec`, `AddressSpace`, and `NameBridge`.

```python
class Bus:
    src: list[LocationAddress]
    dst: list[LocationAddress]


class CZPair:
    a: LocationAddress
    b: LocationAddress


class CapacityInfo:
    num_sites_per_word: int       # number of sites in every word (uniform across the address space)
    words_per_zone: list[int]     # zone_id → number of words in that zone


class TopHatCZ:
    cz_pairs: list[CZPair]   # entangling pairs within this top-hat beam


class GateInfo:
    top_hats: list[TopHatCZ]                     # one entry per key in PhysicalSpec.rydberg_tophats (canonical order)
    local_gate_addresses: list[LocationAddress]  # sites addressable by local single-qubit gates


class BusGraph:
    buses: dict[int, Bus]   # bus_id → Bus; bus_id assigned canonically during derivation


class MachineModel:
    arch_spec: ArchSpec              # back-reference to the parent ArchSpec (e.g. for visualization)
    capacity: CapacityInfo
    gate_info: GateInfo
    bus_graph: BusGraph
    noise: NoiseModel | None         # None when derived without a PhysicalNoiseSpec
```

**CZ pairs are not stored in `ArchSpec` — they are computed by `derive()`.** For each entry in `PhysicalSpec.rydberg_tophats` (in canonical key order), all pairs of addressed sites that both fall within `[y_min, y_max]` and are at or below `PhysicalSpec.blockade_radius` apart are collected into a `TopHatCZ` and emitted as `MachineModel.gate_info.top_hats`.

`bus_graph` is produced by resolving the physical grid positions in each `PhysicalBus` through the zone `AddressMapping`s. `Bus` captures only the virtual src/dst `LocationAddress` pairs; intermediate waypoints remain in `PhysicalSpec`.

`gate_info.top_hats` is derived by iterating over `PhysicalSpec.rydberg_tophats` in canonical key order. For each top-hat, a `TopHatCZ` is produced by finding all pairs of addressed sites whose physical positions both fall within `[y_min, y_max]` and whose mutual distance is at or below `PhysicalSpec.blockade_radius`. There is no assumption that pairs are intra-zone or between whole words — any two addressable sites that co-occur within a top-hat beam and are within blockade range form a CZ pair.

`gate_info.local_gate_addresses` is derived by collecting all `LocationAddress`es whose zone maps (via `ArchSpec.name_bridge.zone_labels`) to a grid key present in `PhysicalSpec.local_gate_grids`.

`capacity.words_per_zone` is indexed by `zone_id` and gives the number of words in `AddressSpace.zones[zone_id].words`.

**Address space / bus compatibility check.** Because all words have the same shape (uniform sites-per-word), bus compatibility is structural: each `PhysicalBus` in `PhysicalSpec.buses` is compatible with the `AddressSpace` if the number of positions in `src` (and `dst`) is a multiple of `capacity.num_sites_per_word`. The derivation does not need to verify that every grid position is covered by a word — partial `AddressMapping` coverage is fine as long as the shape invariant holds.

### Combined: `ArchSpec`

```python
class NameBridge:
    zone_labels: dict[str, int]    # zone name → zone_id; maps PhysicalSpec grid keys to AddressSpace zone indices
    top_hat_index: dict[str, int]  # top-hat key → index into MachineModel.gate_info.top_hats


ArchSpec(
    physical_spec: PhysicalSpec,        # slm_grids, buses, modes, local_gate_grids, rydberg_tophats, blockade_radius, feed_forward, atom_reloading
    address_space: AddressSpace,        # word_shape (n_x, n_y), site_slices, zones
    name_bridge: NameBridge,            # zone_labels, top_hat_index — string↔int mappings across the PhysicalSpec/AddressSpace boundary
)

# MachineModel is not stored — call derive() to emit it on demand:
arch_spec.derive() -> MachineModel
```

**Validation rules for `ArchSpec`** (cross-layer; requires both `PhysicalSpec` and `AddressSpace`):
1. Every zone name in `name_bridge.zone_labels` must exist as a key in `PhysicalSpec.slm_grids` — zone name = grid name.
2. `len(address_space.zones)` must equal `len(name_bridge.zone_labels)` — every zone_id (list index) must have a corresponding entry in `name_bridge.zone_labels`.
3. For each zone, all `x_indices` and `y_indices` in every `Word` must be within the bounds of `PhysicalSpec.slm_grids[zone_name]`.
4. Every key in `name_bridge.top_hat_index` must exist in `PhysicalSpec.rydberg_tophats`, and the indices must be a valid permutation of `range(len(rydberg_tophats))`.

## New `LaneAddress` Encoding

### Current (64-bit)

```
data0: [ word_id : 16 ][ site_id : 16 ]
data1: [ dir : 1 ][ mt : 2 ][ zone_id : 8 ][ pad : 5 ][ bus_id : 16 ]
```

- `mt` (MoveType): 2 bits encoding SITE / WORD / ZONE — **removed**
- `bus_id`: 16-bit zone-local index — **promoted to key into `MachineModel.bus_graph.buses`**

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
| `site_id` | 16 | 0..65535 | List index into `AddressSpace.site_slices` |
| `dir` | 1 | 0=FORWARD, 1=BACKWARD | Transport direction along the bus |
| `pad` | 7 | — | Reserved, must be zero |
| `zone_id` | 8 | 0..255 | Key into `AddressSpace.zones`; use `ArchSpec.name_bridge.zone_labels` for human-readable name |
| `bus_id` | 16 | 0..65535 | Index into `MachineModel.bus_graph.buses`; assigned canonically during derivation |

`PhysicalSpec.buses` uses human-readable string keys; `bus_id` (integer) is assigned by `MachineModel` construction using a deterministic canonical ordering of those keys. `zone_id` remains a separate field from `word_id` because zones can reference differently-sized grids and `word_id`s are not unique across zones.

## Impact on Downstream Consumers

- **`gemini.logical` dialect**: `SiteBusMove` and `WordBusMove` collapse into a single `BusMove` statement; `bus_id` is the integer key into `MachineModel.bus_graph.buses`. `MoveType` attribute is removed.
- **`lanes2flair` rewrites** (bloqade-flair): the hardcoded `{bus_id: lib_fn}` dict in `RewriteSiteBusMove` is replaced by a lookup driven by `MachineModel.bus_graph`. New bus types (right-direction, etc.) are handled without modifying the rewrite rule.
- **`iter_all_lanes` / `check_lane_group`**: rewritten in terms of `MachineModel.bus_graph` rather than per-zone bus enumeration.
- **Validation**: bus membership checks operate on `LocationAddress` lists directly, no `MoveType` dispatch needed.

## Ownership and Responsibility

| Layer | Owner | Can change independently? |
|---|---|---|
| `PhysicalSpec` (slm_grids, buses, modes, local_gate_grids, rydberg_tophats, blockade_radius, feed_forward, atom_reloading) | **bloqade-flair** | Yes — flair defines and evolves the physical hardware description |
| `AddressSpace` (word_shape, site_slices, zones) | **bloqade-lanes** | Yes — the lanes compiler owns the virtual address space |
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
- `address_space_1.site_slices == address_space_2.site_slices` — the site partitioning must be identical (same number of sites, same slicing).
- `len(address_space_1.zones)` ≤ `len(address_space_2.zones)` — every zone_id (list index) in spec_1 exists in spec_2.
- For each zone_id, every `word_id` in `address_space_1.zones[zone_id].words` exists at the same index in `address_space_2.zones[zone_id].words` with identical `x_indices`/`y_indices`.

### `MachineModel` subsetting

`machine_model_1 ⊆ machine_model_2` is a consequence of the above — it does not need to be checked directly. If both `PhysicalSpec` and `AddressSpace` satisfy the subset relation, `MachineModel` is guaranteed to follow:
- Every `bus_id` in `machine_model_1.bus_graph.buses` will exist in `machine_model_2.bus_graph.buses` with the same `src`/`dst`.
- For each `TopHatCZ` in `machine_model_1.gate_info.top_hats`, the corresponding `TopHatCZ` in `machine_model_2.gate_info.top_hats` will contain a superset of its `cz_pairs`.
- Every address in `machine_model_1.gate_info.local_gate_addresses` will exist in `machine_model_2.gate_info.local_gate_addresses`.

### Why the dict structure is essential here

For buses, string dict keys mean the subset check is purely key-and-value equality, independent of insertion order — reordering never silently breaks comparison. Zones and words use list indices intentionally: zone_id and word_id are architecture-stable identifiers that map directly into the `LaneAddress` encoding and are never independently reordered.

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
    cz_noise: dict[str, TopHatNoiseParams]            # per-top-hat CZ noise, keyed by tophat_key
    local_gate_noise: dict[str, LocalGateNoiseParams] # per-grid local gate noise, keyed by grid_key
    global_gate_noise: GlobalGateNoiseParams          # global gate noise (applies to all atoms)
    spam_noise: dict[str, SPAMNoiseParams]            # per-grid SPAM noise, keyed by grid_key
    T1: float                                         # longitudinal coherence time (µs)
    T2: float                                         # transverse coherence time (µs)


class CZNoiseParams:
    fidelity: float   # CZ gate fidelity for this pair
    # expandable: crosstalk, leakage, etc. — same fields as TopHatNoiseParams, resolved per-pair


class NoiseModel:
    global_gate_noise: GlobalGateNoiseParams
    cz_noise: dict[CZPair, CZNoiseParams]                        # per CZ pair; derived by expanding TopHatNoiseParams across all pairs in each top-hat
    local_gate_noise: dict[LocationAddress, LocalGateNoiseParams] # per site; derived by expanding per-grid noise across all LocationAddresses in that grid
    bus_noise: dict[int, BusNoiseParams]                          # per bus_id; resolved from PhysicalNoiseSpec bus string keys
    spam_noise: dict[LocationAddress, SPAMNoiseParams]            # per site; derived by expanding per-grid SPAM noise
    T1: float
    T2: float


# derive() without noise:
arch_spec.derive() -> MachineModel

# derive() with noise — PhysicalNoiseSpec is a peer of PhysicalSpec, not nested inside it:
arch_spec.derive(physical_noise_spec: PhysicalNoiseSpec) -> MachineModel
```

`PhysicalNoiseSpec` shares string keys with `PhysicalSpec` — every key in `cz_noise` must exist in `PhysicalSpec.rydberg_tophats`, every key in `local_gate_noise` and `spam_noise` must exist in `PhysicalSpec.slm_grids`, and every key in `bus_noise` must exist in `PhysicalSpec.buses`. Using structured objects (`TopHatNoiseParams`, `LocalGateNoiseParams`, `GlobalGateNoiseParams`) rather than bare floats allows finer-grained noise parameters to be added later without breaking the interface.

**Resolved:** `NoiseModel` subsetting is not defined. Noise compatibility is a separate concern from architectural compatibility — whether a circuit's noise budget holds on a given device is a simulation/analysis question, not an ArchSpec subset check.

**Open questions:**
1. Should `T1`/`T2` be global scalars or per-grid (atoms in different grids may have different coherence)?
