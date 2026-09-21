# Bloqade Lanes Bytecode Instruction Specification

This document specifies the bytecode instruction set used by Bloqade Lanes to describe atom shuttling programs for neutral atom quantum processors. A bytecode program is a sequence of fixed-width instructions that drive the full lifecycle of a computation: loading atoms into an optical lattice, shuttling them between sites using AOD (Acousto-Optic Deflector) transport, applying quantum gates, and reading out measurement results.

The instruction set is organized around the physical structure of the hardware. Atoms occupy **sites** within **words** (rows of trapping positions in the lattice). **Buses** define the AOD transport channels that move atoms between sites (site buses) or between words (word buses). A **lane** is a single atom trajectory along a bus — one source site to one destination site. A **zone** groups words that share a global entangling interaction (e.g. a Rydberg pulse) or define locations where atoms are measured. These concepts map directly to the address types used in the bytecode: `LocationAddr` (word, site), `LaneAddr` (word, site, bus, direction), and `ZoneAddr` (zone).

Programs execute on a stack machine. Address constants and numeric parameters are pushed onto the stack, then consumed by operation instructions (fills, moves, gates, measurements). The bytecode is designed to be validated offline against an architecture specification (`ArchSpec`) that captures the geometry, bus topology, and zone layout of a specific device.

## Instruction Format

Every instruction is a fixed **13 bytes**: a 1-byte opcode followed by the
operands in declaration order, each little-endian, zero-padded out to the full
width.

```
┌────────────┬────────────────────────────────────────────┐
│ opcode(u8) │ operands (LE, in order) ‖ zero padding      │
├────────────┼────────────────────────────────────────────┤
│   byte 0   │                bytes 1–12                  │
└────────────┴────────────────────────────────────────────┘
```

The width is not a chosen constant: vihaco derives it as the widest variant,
which today is `new_array` (1 + 3×u32 = 13). It will change if a wider operand
set is added. Instructions with no operands are the opcode byte followed by 12
zero bytes.

Because every word is the same width, a program is just N concatenated words and
decodes without desync.

Examples:

```
cpu.halt              04 00 00 00 00 00 00 00 00 00 00 00 00
cpu.const_int 42      06 2a 00 00 00 00 00 00 00 00 00 00 00   (i64 LE, 4B pad)
lanes.move 1          0c 01 00 00 00 00 00 00 00 00 00 00 00   (u32 LE, 8B pad)
lanes.new_array 2 10 20
                      14 02 00 00 00 0a 00 00 00 14 00 00 00   (3 × u32 LE)
```

## Opcodes

vihaco assigns the opcode byte from the variant's **position** in the
`Instruction` enum: the first variant is `0x00`, the second `0x01`, and so on.
There is no device-code packing and no structure inside the byte.

The practical consequence is that opcodes are **not stable across revisions** —
inserting an instruction anywhere but the end renumbers every instruction after
it, which changes the binary encoding of existing programs. The values in this
document are correct for this revision; `Instruction::opcode()` is the authority.

Bloqade Lanes programs are not binary-compatible with the pre-`LANES` container
format, nor with the FLAIR-aligned device/instruction-code scheme this
specification previously described.

## Text Format (`.sst`)

The text form is a version directive followed by a single `@main` function:

```
version 1.0;
fn @main() {
  lanes.const_loc 0x0000000000000000
  lanes.const_loc 0x0000000001000000
  lanes.initial_fill 2
  cpu.halt
}
```

- Comments are `//` to end of line.
- Every instruction carries a **dialect head**: `cpu.` for the stack ops,
  `lanes.` for the device ops. The head is required — a bare `move 2` does not
  parse, and neither does a mnemonic under the wrong head (`cpu.move`).
- Address operands are `0x`-prefixed hexadecimal; arities and array dimensions
  are decimal.
- Exactly one function is allowed and it must be named `@main`.

`to_text` emits this form and `parse_text` accepts it, round-tripping losslessly.

## Address Encoding

Addresses are bit-packed into a single integer operand, written little-endian.

### `LocationAddr`

Packed into one `u64`:

```
[zone_id:8][word_id:16][site_id:16][pad:24]
 bits 63–56  bits 55–40  bits 39–24  bits 23–0
```

### `LaneAddr`

Packed into one `u64`:

```
[dir:1][mt:2][zone_id:8][pad:5][bus_id:16][word_id:16][site_id:16]
 bit 63  62–61  60–53     52–48   47–32      31–16       15–0
```

- `dir` — direction: 0 = Forward, 1 = Backward
- `mt` — move type: 0 = SiteBus, 1 = WordBus, 2 = ZoneBus

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

Packed into one `u32`:

```
[pad:24][zone_id:8]
 bits 31–8  bits 7–0
```

## Instructions

### Stack ops

Spelled under the `cpu.` dialect head. These mirror vihaco-cpu's stack ops, but
are declared natively in our own `Instruction` enum: as of vihaco 0.4,
vihaco-cpu is a runtime *component* whose instructions carry no binary codec, so
they cannot be nested in an encodable ISA.

#### `cpu.const_int` — Push integer constant

| Field | Value |
|---|---|
| Opcode | `0x06` |
| Operands | `i64` LE (8 bytes) |
| Stack | `( -- int)` |

Pushes a signed 64-bit integer onto the stack.

#### `cpu.const_float` — Push float constant

| Field | Value |
|---|---|
| Opcode | `0x05` |
| Operands | `f64` LE (8 bytes) |
| Stack | `( -- float)` |

Pushes a 64-bit float onto the stack.

#### `cpu.dup` — Duplicate top of stack

| Field | Value |
|---|---|
| Opcode | `0x03` |
| Operands | none |
| Stack | `(a -- a a)` |

#### `cpu.pop` — Discard top of stack

| Field | Value |
|---|---|
| Opcode | `0x00` |
| Operands | none |
| Stack | `(a -- )` |

#### `cpu.swap` — Swap top two stack elements

| Field | Value |
|---|---|
| Opcode | `0x01` |
| Operands | none |
| Stack | `(a b -- b a)` |

#### `cpu.return` — Return from program

| Field | Value |
|---|---|
| Opcode | `0x02` |
| Operands | none |
| Stack | `( -- )` |

#### `cpu.halt` — Halt execution

| Field | Value |
|---|---|
| Opcode | `0x04` |
| Operands | none |
| Stack | `( -- )` |

### Address constants

#### `lanes.const_loc` — Push location address

| Field | Value |
|---|---|
| Opcode | `0x07` |
| Operands | `LocationAddr` as `u64` LE — `[zone_id:8][word_id:16][site_id:16][pad:24]` |
| Stack | `( -- loc)` |

#### `lanes.const_lane` — Push lane address

| Field | Value |
|---|---|
| Opcode | `0x08` |
| Operands | `LaneAddr` as `u64` LE — `[dir:1][mt:2][zone_id:8][pad:5][bus_id:16][word_id:16][site_id:16]` |
| Stack | `( -- lane)` |

#### `lanes.const_zone` — Push zone address

| Field | Value |
|---|---|
| Opcode | `0x09` |
| Operands | `ZoneAddr` as `u32` LE — `[pad:24][zone_id:8]` |
| Stack | `( -- zone)` |

### Atom arrangement

#### `lanes.initial_fill` — Initial atom loading

| Field | Value |
|---|---|
| Opcode | `0x0A` |
| Operands | `u32` LE arity |
| Stack | `(loc₁ loc₂ … locₙ -- )` |

Pops `n` location addresses and performs the initial atom fill at those sites.

#### `lanes.fill` — Atom refill

| Field | Value |
|---|---|
| Opcode | `0x0B` |
| Operands | `u32` LE arity |
| Stack | `(loc₁ loc₂ … locₙ -- )` |

Pops `n` location addresses and refills atoms at those sites.

#### `lanes.move` — Atom transport

| Field | Value |
|---|---|
| Opcode | `0x0C` |
| Operands | `u32` LE arity |
| Stack | `(lane₁ lane₂ … laneₙ -- )` |

Pops `n` lane addresses and performs atom moves along those lanes. All lanes in a single `move` instruction are executed simultaneously as one AOD transport operation: every endpoint is resolved against the pre-move atom state, so the result is independent of lane order, and a multi-hop route (`x→y` then `y→z` for the *same* atom) must be split across separate `move` instructions.

A lane whose source holds no atom is a no-op (AOD rectangle filler), but the trap site still arrives at its destination — so an occupied destination is only legal when its occupant is itself moved by another lane in the same instruction (conveyor chains such as `x→y, y→z` executed as one shot). A destination occupied by an atom that does not move in the group makes the group **not executable**.

This executability rule is state-dependent, so it is *not* checked by the static program validator below (which has no atom-occupancy state). It is enforced by `AtomStateData::validate_moves`, which reports it as `MoveValidationError::DestinationOccupiedByStationaryAtom`. An implementation that applies an unvalidated group (`AtomStateData::apply_moves`) instead models a mover landing on a stationary atom as a collision: both atoms are removed from the location maps and recorded in the state's `collision` field.

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

### Quantum gates

#### `lanes.local_r` — Local R rotation

| Field | Value |
|---|---|
| Opcode | `0x0E` |
| Operands | `u32` LE arity |
| Stack | `(loc₁ loc₂ … locₙ θ φ -- )` |

Pops 2 float parameters (φ = axis angle, θ = rotation angle) then `n` location addresses, and applies a local R rotation. The call convention matches the SSA IR: `local_r(%φ, %θ, %loc₁, …)` — first argument (φ) is pushed last and popped first.

#### `lanes.local_rz` — Local Rz rotation

| Field | Value |
|---|---|
| Opcode | `0x0D` |
| Operands | `u32` LE arity |
| Stack | `(loc₁ loc₂ … locₙ θ -- )` |

Pops 1 float parameter (θ = rotation angle) then `n` location addresses, and applies a local Rz rotation. The call convention matches the SSA IR: `local_rz(%θ, %loc₁, …)`.

#### `lanes.global_r` — Global R rotation

| Field | Value |
|---|---|
| Opcode | `0x10` |
| Operands | none |
| Stack | `(θ φ -- )` |

Pops 2 float parameters (φ = axis angle, θ = rotation angle), applies a global R rotation. The call convention matches the SSA IR: `global_r(%φ, %θ)`.

#### `lanes.global_rz` — Global Rz rotation

| Field | Value |
|---|---|
| Opcode | `0x0F` |
| Operands | none |
| Stack | `(θ -- )` |

Pops 1 float parameter (θ = rotation angle), applies a global Rz rotation. Since there is only one parameter, it is both pushed last and popped first.

#### `lanes.cz` — Controlled-Z gate

| Field | Value |
|---|---|
| Opcode | `0x11` |
| Operands | none |
| Stack | `(zone -- )` |

Pops a zone address and applies a CZ gate across the zone.

### Measurement

#### `lanes.measure` — Initiate measurement

| Field | Value |
|---|---|
| Opcode | `0x12` |
| Operands | `u32` LE arity |
| Stack | `(zone₁ zone₂ … zoneₙ -- future₁ future₂ … futureₙ)` |

Pops `n` zone addresses and pushes `n` measure futures.

#### `lanes.await_measure` — Wait for measurement result

| Field | Value |
|---|---|
| Opcode | `0x13` |
| Operands | none |
| Stack | `(future -- array_ref)` |

Pops a measure future and pushes an array reference containing the measurement results.

### Arrays

#### `lanes.new_array` — Construct array from stack

| Field | Value |
|---|---|
| Opcode | `0x14` |
| Operands | three `u32` LE: `type_tag`, `dim0`, `dim1` (`dim1 = 0` for 1-D) |
| Stack | `(elem₁ elem₂ … elemₙ -- array_ref)` |

Constructs an array of `dim0 × dim1` elements with element type `type_tag`. If `dim1` is 0, the array is 1-dimensional with `dim0` elements.

#### `lanes.get_item` — Index into array

| Field | Value |
|---|---|
| Opcode | `0x15` |
| Operands | `u32` LE ndims |
| Stack | `(array_ref idx₁ … idxₙ -- value)` |

Pops `ndims` index values then the array reference, and pushes the indexed element.

### Detectors and observables

#### `lanes.set_detector` — Set detector

| Field | Value |
|---|---|
| Opcode | `0x16` |
| Operands | none |
| Stack | `(array_ref -- detector_ref)` |

Pops an array reference and pushes a detector reference.

#### `lanes.set_observable` — Set observable

| Field | Value |
|---|---|
| Opcode | `0x17` |
| Operands | none |
| Stack | `(array_ref -- observable_ref)` |

Pops an array reference and pushes an observable reference.
