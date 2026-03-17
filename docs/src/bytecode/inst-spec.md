# Bloqade Lanes Bytecode Instruction Specification

## Instruction Format

Every instruction is a fixed **16 bytes**: a 32-bit opcode word followed by three 32-bit data words, all little-endian.

```
┌──────────────┬──────────────┬──────────────┬──────────────┐
│ opcode (u32) │ data0 (u32)  │ data1 (u32)  │ data2 (u32)  │
├──────────────┼──────────────┼──────────────┼──────────────┤
│  bytes 0–3   │  bytes 4–7   │  bytes 8–11  │ bytes 12–15  │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

Instructions that take no operand ignore the data words (should be zero). Instructions with operands encode them in the data words as described per-instruction below.

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
| Stack | `( -- value)` |

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
| Stack | `( -- value)` |

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

Pops `n` lane addresses and performs atom moves along those lanes.

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
| Stack | `(rotation_angle axis_angle loc₁ loc₂ … locₙ -- )` |

Pops `n` location addresses and 2 float parameters (rotation_angle, axis_angle), applies a local R rotation.

#### `local_rz` — Local Rz rotation

| Field | Value |
|---|---|
| Device Code | `0x11` |
| Instruction Code | `0x01` |
| Full Opcode | `0x0111` |
| data0 | `u32` LE arity |
| data1 | unused |
| data2 | unused |
| Stack | `(rotation_angle loc₁ loc₂ … locₙ -- )` |

Pops `n` location addresses and 1 float parameter (rotation_angle), applies a local Rz rotation.

#### `global_r` — Global R rotation

| Field | Value |
|---|---|
| Device Code | `0x11` |
| Instruction Code | `0x02` |
| Full Opcode | `0x0211` |
| data0–2 | unused |
| Stack | `(rotation_angle axis_angle -- )` |

Pops 2 float parameters (rotation_angle, axis_angle), applies a global R rotation.

#### `global_rz` — Global Rz rotation

| Field | Value |
|---|---|
| Device Code | `0x11` |
| Instruction Code | `0x03` |
| Full Opcode | `0x0311` |
| data0–2 | unused |
| Stack | `(rotation_angle -- )` |

Pops 1 float parameter (rotation_angle), applies a global Rz rotation.

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
| Stack | `(measure_future -- array_ref)` |

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
