# Bloqade Lanes Bytecode Instruction Specification

This document specifies the bytecode instruction set used by Bloqade Lanes to describe atom shuttling programs for neutral atom quantum processors. A bytecode program is a sequence of fixed-width instructions that drive the full lifecycle of a computation: loading atoms into an optical lattice, shuttling them between sites using AOD (Acousto-Optic Deflector) transport, applying quantum gates, and reading out measurement results.

The instruction set is organized around the physical structure of the hardware. Atoms occupy **sites** within **words** (rows of trapping positions in the lattice). **Buses** define the AOD transport channels that move atoms between sites (site buses) or between words (word buses). A **lane** is a single atom trajectory along a bus — one source site to one destination site. A **zone** groups words that share a global entangling interaction (e.g. a Rydberg pulse) or define locations where atoms are measured. These concepts map directly to the address types used in the bytecode: `LocationAddr` (word, site), `LaneAddr` (word, site, bus, direction), and `ZoneAddr` (zone).

Programs execute on a stack machine. Address constants and numeric parameters are pushed onto the stack, then consumed by operation instructions (fills, moves, gates, measurements). The bytecode is designed to be validated offline against an architecture specification (`ArchSpec`) that captures the geometry, bus topology, and zone layout of a specific device.

## Instruction Format

Every instruction is a fixed **16 bytes**: a 32-bit opcode word followed by three 32-bit data words, all little-endian.

```
┌──────────────┬──────────────┬──────────────┬──────────────┐
│ opcode (u32) │ data0 (u32)  │ data1 (u32)  │ data2 (u32)  │
├──────────────┼──────────────┼──────────────┼──────────────┤
│  bytes 0–3   │  bytes 4–7   │  bytes 8–11  │ bytes 12–15  │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

Instructions that take no operands ignore the data words (should be zero). Instructions with operands encode them in the data words as described per-instruction below.

## Opcode Packing

The opcode word is packed as a 1-byte **instruction code** and a 1-byte **device code** in the low 16 bits of the u32. The upper 16 bits are unused (must be zero). The device code occupies the least significant byte.

```
┌──────────────┬──────────────────┬──────────────────┐
│   unused     │ instruction code │   device code    │
│  (16 bits)   │    (8 bits)      │    (8 bits)      │
└──────────────┴──────────────────┴──────────────────┘
  bits 31–16       bits 15–8          bits 7–0
```

Full opcode = `(instruction_code << 8) | device_code`.

In little-endian memory layout:

```
byte[0] = device_code        (bits 7–0)
byte[1] = instruction_code   (bits 15–8)
byte[2] = 0x00               (unused)
byte[3] = 0x00               (unused)
```

Instruction codes **can overlap** across different devices — the device code byte disambiguates.

## Device Codes

| Device Code | Name | Description |
|---|---|---|
| `0x00` | `Cpu` | Stack manipulation, constants, control flow (FLAIR-aligned) |
| `0x0F` | `LaneConstants` | Lane-specific constant instructions |
| `0x10` | `AtomArrangement` | Atom filling and movement |
| `0x11` | `QuantumGate` | Single- and multi-qubit gate operations |
| `0x12` | `Measurement` | Qubit measurement |
| `0x13` | `Array` | Array construction and indexing |
| `0x14` | `DetectorObservable` | Detector and observable setup |

Device codes `0x01`–`0x0E` are reserved for future FLAIR device types.

## Address Encoding

All address field components are 16-bit, packed into the data words.

### `LocationAddr`

Packed in a single data word (data0):

```
data0: [word_id:16][site_id:16]
        bits 31–16   bits 15–0
```

Total: 32 bits (u32).

### `LaneAddr`

Packed across two data words (data0 + data1):

```
data0: [word_id:16][site_id:16]
        bits 31–16   bits 15–0

data1: [dir:1][mt:1][pad:14][bus_id:16]
       bit 31  bit 30  29–16  bits 15–0
```

- `dir` — direction: 0 = Forward, 1 = Backward
- `mt` — move type: 0 = SiteBus, 1 = WordBus

Total: 64 bits across two u32 words. Note that data0 shares the same layout as `LocationAddr`.

#### Lane address convention

The `word_id` and `site_id` fields in a `LaneAddr` always encode the **forward-direction source** — the position where the atom starts in a forward move. The `direction` field does **not** change which position is encoded; it only controls which endpoint is treated as source vs destination when the lane is resolved.

**Endpoint resolution** always starts by resolving the forward direction:

1. Look up the bus (site bus or word bus, selected by `move_type` and `bus_id`)
2. Find the index `i` where `bus.src[i]` matches the encoded `site_id` (for site buses) or `word_id` (for word buses)
3. The forward source is `(word_id, site_id)` as encoded; the forward destination is `(word_id, bus.dst[i])` for site buses or `(bus.dst[i], site_id)` for word buses
4. If `direction = Forward`: return `(fwd_source, fwd_destination)`
5. If `direction = Backward`: return `(fwd_destination, fwd_source)` — the endpoints are swapped

**Example:** Given a site bus with `src=[0,1,2,3,4] dst=[5,6,7,8,9]`:

| Lane | Encoded | Resolved src → dst |
|------|---------|-------------------|
| `site_id=0, dir=Forward` | Forward source is site 0 | Site 0 → Site 5 |
| `site_id=0, dir=Backward` | Forward source is still site 0 | Site 5 → Site 0 |
| `site_id=2, dir=Backward` | Forward source is site 2 | Site 7 → Site 2 |

Note that a backward lane with `site_id=0` means the atom moves **from** site 5 **to** site 0 — not that site 0 is the destination of a forward move.

#### Lane validation rules

The validator (`check_lane`) checks the following for each `LaneAddr`:

| Rule | Error condition |
|------|----------------|
| Bus must exist | `bus_id` out of range for the given `move_type` |
| `word_id` in range | `word_id >= num_words` |
| `site_id` in range | `site_id >= sites_per_word` |
| Bus membership | For site buses: `word_id` must be in `words_with_site_buses`. For word buses: `site_id` must be in `sites_with_word_buses`. |
| Valid forward source | For site buses: `bus.resolve_forward(site_id)` must succeed (i.e. `site_id` is in `bus.src`). For word buses: `bus.resolve_forward(word_id)` must succeed (i.e. `word_id` is in `bus.src`). |

Validation is always performed against the forward-direction source, regardless of the `direction` field.

### `ZoneAddr`

Packed in a single data word (data0):

```
data0: [pad:16][zone_id:16]
       bits 31–16  bits 15–0
```

Total: 32 bits (u32).

## Instructions

### Cpu (`0x00`) — FLAIR-aligned shared opcodes

These instruction codes are shared with the FLAIR VM/IR spec and use identical values.

#### `const_int` — Push integer constant

| Field | Value |
|---|---|
| Device Code | `0x00` |
| Instruction Code | `0x02` |
| Full Opcode | `0x0200` |
| data0 | `i64` LE low 32 bits |
| data1 | `i64` LE high 32 bits |
| data2 | unused |
| Stack | `( -- int)` |

Pushes a signed 64-bit integer onto the stack. The value is stored as a little-endian i64 across data0 (low) and data1 (high).

#### `const_float` — Push float constant

| Field | Value |
|---|---|
| Device Code | `0x00` |
| Instruction Code | `0x03` |
| Full Opcode | `0x0300` |
| data0 | `f64` LE low 32 bits |
| data1 | `f64` LE high 32 bits |
| data2 | unused |
| Stack | `( -- float)` |

Pushes a 64-bit float onto the stack. The value is stored as a little-endian f64 across data0 (low) and data1 (high).

#### `dup` — Duplicate top of stack

| Field | Value |
|---|---|
| Device Code | `0x00` |
| Instruction Code | `0x04` |
| Full Opcode | `0x0400` |
| data0–2 | unused |
| Stack | `(a -- a a)` |

#### `pop` — Discard top of stack

| Field | Value |
|---|---|
| Device Code | `0x00` |
| Instruction Code | `0x05` |
| Full Opcode | `0x0500` |
| data0–2 | unused |
| Stack | `(a -- )` |

#### `swap` — Swap top two stack elements

| Field | Value |
|---|---|
| Device Code | `0x00` |
| Instruction Code | `0x06` |
| Full Opcode | `0x0600` |
| data0–2 | unused |
| Stack | `(a b -- b a)` |

#### `return` — Return from program

| Field | Value |
|---|---|
| Device Code | `0x00` |
| Instruction Code | `0x64` |
| Full Opcode | `0x6400` |
| data0–2 | unused |
| Stack | `( -- )` |

#### `halt` — Halt execution

| Field | Value |
|---|---|
| Device Code | `0x00` |
| Instruction Code | `0xFF` |
| Full Opcode | `0xFF00` |
| data0–2 | unused |
| Stack | `( -- )` |

### LaneConstants (`0x0F`)

#### `const_loc` — Push location address

| Field | Value |
|---|---|
| Device Code | `0x0F` |
| Instruction Code | `0x00` |
| Full Opcode | `0x000F` |
| data0 | `LocationAddr` — `[word_id:16][site_id:16]` |
| data1 | unused |
| data2 | unused |
| Stack | `( -- loc)` |

#### `const_lane` — Push lane address

| Field | Value |
|---|---|
| Device Code | `0x0F` |
| Instruction Code | `0x01` |
| Full Opcode | `0x010F` |
| data0 | `[word_id:16][site_id:16]` |
| data1 | `[dir:1][mt:1][pad:14][bus_id:16]` |
| data2 | unused |
| Stack | `( -- lane)` |

#### `const_zone` — Push zone address

| Field | Value |
|---|---|
| Device Code | `0x0F` |
| Instruction Code | `0x02` |
| Full Opcode | `0x020F` |
| data0 | `ZoneAddr` — `[pad:16][zone_id:16]` |
| data1 | unused |
| data2 | unused |
| Stack | `( -- zone)` |

### AtomArrangement (`0x10`)

#### `initial_fill` — Initial atom loading

| Field | Value |
|---|---|
| Device Code | `0x10` |
| Instruction Code | `0x00` |
| Full Opcode | `0x0010` |
| data0 | `u32` LE arity |
| data1 | unused |
| data2 | unused |
| Stack | `(loc₁ loc₂ … locₙ -- )` |

Pops `n` location addresses and performs the initial atom fill at those sites.

#### `fill` — Atom refill

| Field | Value |
|---|---|
| Device Code | `0x10` |
| Instruction Code | `0x01` |
| Full Opcode | `0x0110` |
| data0 | `u32` LE arity |
| data1 | unused |
| data2 | unused |
| Stack | `(loc₁ loc₂ … locₙ -- )` |

Pops `n` location addresses and refills atoms at those sites.

#### `move` — Atom transport

| Field | Value |
|---|---|
| Device Code | `0x10` |
| Instruction Code | `0x02` |
| Full Opcode | `0x0210` |
| data0 | `u32` LE arity |
| data1 | unused |
| data2 | unused |
| Stack | `(lane₁ lane₂ … laneₙ -- )` |

Pops `n` lane addresses and performs atom moves along those lanes. All lanes in a single `move` instruction are executed simultaneously as one AOD transport operation.

##### Lane group validation

When an `ArchSpec` is provided, the validator checks the group of lanes as a whole — not just each lane individually. These constraints reflect the physical limitations of a single AOD (Acousto-Optic Deflector). Each `move` instruction corresponds to one AOD operation:

**Consistency** — all lanes in the group must share the same `move_type`, `bus_id`, and `direction`. A single AOD operation cannot mix site-bus and word-bus moves, use different buses, or move atoms in different directions simultaneously.

**Bus membership** — for site-bus moves, every lane's `word_id` must be in `words_with_site_buses`. For word-bus moves, every lane's `site_id` must be in `sites_with_word_buses`.

**Grid constraint** — the physical positions of the lane sources must form a complete grid (Cartesian product of unique X and Y coordinates). An AOD addresses rows and columns independently, so it cannot select an arbitrary subset of positions — it must address every intersection of the selected rows and columns.

For example, if a move group contains lanes at positions `(0,0)`, `(0,1)`, `(1,0)`, and `(1,1)`, this is a valid 2x2 grid. But `(0,0)`, `(0,1)`, `(1,0)` alone is invalid — the AOD would also address `(1,1)`, so the group must include it.

| Check | Error |
|-------|-------|
| All lanes share `move_type`, `bus_id`, `direction` | `Inconsistent` |
| Site-bus lane `word_id` in `words_with_site_buses` | `WordNotInSiteBusList` |
| Word-bus lane `site_id` in `sites_with_word_buses` | `SiteNotInWordBusList` |
| Lane positions form a complete grid | `AODConstraintViolation` |

### QuantumGate (`0x11`)

#### `local_r` — Local R rotation

| Field | Value |
|---|---|
| Device Code | `0x11` |
| Instruction Code | `0x00` |
| Full Opcode | `0x0011` |
| data0 | `u32` LE arity |
| data1 | unused |
| data2 | unused |
| Stack | `(loc₁ loc₂ … locₙ θ φ -- )` |

Pops 2 float parameters (φ = axis angle, θ = rotation angle) then `n` location addresses, and applies a local R rotation. The call convention matches the SSA IR: `local_r(%φ, %θ, %loc₁, …)` — first argument (φ) is pushed last and popped first.

#### `local_rz` — Local Rz rotation

| Field | Value |
|---|---|
| Device Code | `0x11` |
| Instruction Code | `0x01` |
| Full Opcode | `0x0111` |
| data0 | `u32` LE arity |
| data1 | unused |
| data2 | unused |
| Stack | `(loc₁ loc₂ … locₙ θ -- )` |

Pops 1 float parameter (θ = rotation angle) then `n` location addresses, and applies a local Rz rotation. The call convention matches the SSA IR: `local_rz(%θ, %loc₁, …)`.

#### `global_r` — Global R rotation

| Field | Value |
|---|---|
| Device Code | `0x11` |
| Instruction Code | `0x02` |
| Full Opcode | `0x0211` |
| data0–2 | unused |
| Stack | `(θ φ -- )` |

Pops 2 float parameters (φ = axis angle, θ = rotation angle), applies a global R rotation. The call convention matches the SSA IR: `global_r(%φ, %θ)`.

#### `global_rz` — Global Rz rotation

| Field | Value |
|---|---|
| Device Code | `0x11` |
| Instruction Code | `0x03` |
| Full Opcode | `0x0311` |
| data0–2 | unused |
| Stack | `(θ -- )` |

Pops 1 float parameter (θ = rotation angle), applies a global Rz rotation. Since there is only one parameter, it is both pushed last and popped first.

#### `cz` — Controlled-Z gate

| Field | Value |
|---|---|
| Device Code | `0x11` |
| Instruction Code | `0x04` |
| Full Opcode | `0x0411` |
| data0–2 | unused |
| Stack | `(zone -- )` |

Pops a zone address and applies a CZ gate across the zone.

### Measurement (`0x12`)

#### `measure` — Initiate measurement

| Field | Value |
|---|---|
| Device Code | `0x12` |
| Instruction Code | `0x00` |
| Full Opcode | `0x0012` |
| data0 | `u32` LE arity |
| data1 | unused |
| data2 | unused |
| Stack | `(zone₁ zone₂ … zoneₙ -- future₁ future₂ … futureₙ)` |

Pops `n` zone addresses and pushes `n` measure futures.

#### `await_measure` — Wait for measurement result

| Field | Value |
|---|---|
| Device Code | `0x12` |
| Instruction Code | `0x01` |
| Full Opcode | `0x0112` |
| data0–2 | unused |
| Stack | `(future -- array_ref)` |

Pops a measure future and pushes an array reference containing the measurement results.

### Array (`0x13`)

#### `new_array` — Construct array from stack

| Field | Value |
|---|---|
| Device Code | `0x13` |
| Instruction Code | `0x00` |
| Full Opcode | `0x0013` |
| data0 | `[type_tag:8][pad:8][dim0:16]` |
| data1 | `[pad:16][dim1:16]` |
| data2 | unused |
| Stack | `(elem₁ elem₂ … elemₙ -- array_ref)` |

Constructs an array of `dim0 × dim1` elements with element type `type_tag`. If `dim1` is 0, the array is 1-dimensional with `dim0` elements.

#### `get_item` — Index into array

| Field | Value |
|---|---|
| Device Code | `0x13` |
| Instruction Code | `0x01` |
| Full Opcode | `0x0113` |
| data0 | `u16` LE ndims (upper 16 bits unused) |
| data1 | unused |
| data2 | unused |
| Stack | `(array_ref idx₁ … idxₙ -- value)` |

Pops `ndims` index values then the array reference, and pushes the indexed element.

### DetectorObservable (`0x14`)

#### `set_detector` — Set detector

| Field | Value |
|---|---|
| Device Code | `0x14` |
| Instruction Code | `0x00` |
| Full Opcode | `0x0014` |
| data0–2 | unused |
| Stack | `(array_ref -- detector_ref)` |

Pops an array reference and pushes a detector reference.

#### `set_observable` — Set observable

| Field | Value |
|---|---|
| Device Code | `0x14` |
| Instruction Code | `0x01` |
| Full Opcode | `0x0114` |
| data0–2 | unused |
| Stack | `(array_ref -- observable_ref)` |

Pops an array reference and pushes an observable reference.

## Reserved Opcode Ranges

| Range | Owner |
|---|---|
| Device `0x00`, inst codes `0x00`–`0x8F` | Reserved for FLAIR. This project uses only `0x02`–`0x06`, `0x64`, `0xFF`. |
| Device codes `0x01`–`0x0E` | Reserved for future FLAIR device types |
| Device codes `0x0F`–`0xFF` | Project-specific (currently `0x0F`–`0x14` allocated) |

### Known FLAIR allocations (device `0x00`)

| Instruction Code | Purpose |
|---|---|
| `0x01` | `const.bool` |
| `0x10`–`0x17` | Arithmetic (`arith.add_int`, `arith.add_float`, etc.) |
| `0x20`–`0x23` | Comparison (`cmp.gt_int`, `cmp.eq_float`, etc.) |
| `0x28`–`0x2A` | Boolean (`bool.not`, `bool.and`, `bool.or`) |
| `0x30`–`0x33` | Waveform (`waveform.poly4`, `waveform.delay`, etc.) |
| `0x40`–`0x43` | Channel (`channel.emit`, `channel.play`, etc.) |
| `0x50`–`0x52` | Peer messaging |
| `0x60`–`0x63` | Control flow (`cf.jump`, `cf.branch`, `cf.call`) |
| `0x80` | `debug.trace` |
