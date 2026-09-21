# Bloqade Lanes Bytecode Instruction Specification

This document specifies the bytecode instruction set used by Bloqade Lanes to describe atom shuttling programs for neutral atom quantum processors. A bytecode program is a sequence of fixed-width instructions that drive the full lifecycle of a computation: loading atoms into an optical lattice, shuttling them between sites using AOD (Acousto-Optic Deflector) transport, applying quantum gates, and reading out measurement results.

The instruction set is organized around the physical structure of the hardware. Atoms occupy **sites** within **words** (rows of trapping positions in the lattice). **Buses** define the AOD transport channels that move atoms between sites (site buses) or between words (word buses). A **lane** is a single atom trajectory along a bus — one source site to one destination site. A **zone** groups words that share a global entangling interaction (e.g. a Rydberg pulse) or define locations where atoms are measured. These concepts map directly to the address types used in the bytecode: `LocationAddr` (word, site), `LaneAddr` (word, site, bus, direction), and `ZoneAddr` (zone).

Programs execute on a stack machine. Address constants and numeric parameters are pushed onto the stack, then consumed by operation instructions (fills, moves, gates, measurements). The bytecode is designed to be validated offline against an architecture specification (`ArchSpec`) that captures the geometry, bus topology, and zone layout of a specific device.

## The machine

A lanes program runs on a **composite machine** of two vihaco devices:

| Device | Code | Supplies |
|---|---|---|
| `cpu` | `0x00` | vihaco-cpu's `CPU` component: the stack, constants, arithmetic, comparisons, control flow, the heap allocator |
| `lanes` | `0x01` | atom movement, gates, measurement, arrays, and the `pop`/`swap` the CPU lacks |

Instructions are spelled `<device>::<dialect>.<mnemonic>`. The first half is the
device; the second is that device's own dialect head.

Only the atom-arrangement ops (`initial_fill`, `fill`, `move`) are executed by
the machine — they advance the atom state and fault on an illegal move. The
quantum ops are emitted as effects rather than simulated, and the array and
measurement ops likewise, pending
[#776](https://github.com/QuEraComputing/bloqade-lanes/issues/776).

## Instruction Format

Every instruction is a fixed **14-byte** word: a 1-byte device opcode, a 1-byte
instruction opcode, then the operands in declaration order, each little-endian,
zero-padded out to the full width.

```
┌────────────┬────────────┬──────────────────────────────────┐
│ device(u8) │ opcode(u8) │ operands (LE) ‖ zero padding      │
├────────────┼────────────┼──────────────────────────────────┤
│   byte 0   │   byte 1   │            bytes 2–13            │
└────────────┴────────────┴──────────────────────────────────┘
```

The width is not a chosen constant — vihaco derives it as the widest variant
across both devices, and it changes if either gains a wider operand.

Because every word is the same width, the code region is just N concatenated
words and decodes without desync.

## Opcodes

vihaco assigns opcodes by **declaration position**. The Python API reports them
packed as `(device_code << 8) | instruction_code`.

Both halves are positional, so adding an instruction to either device anywhere
but the end renumbers everything after it — which changes the binary encoding of
existing programs. The values in this document are correct for this revision;
`Instruction.op_name()` is the stable identity.

Bloqade Lanes programs are not binary-compatible with either earlier container
(`BLQD`, `LANES`), nor with the FLAIR-aligned device/instruction-code scheme this
specification previously described.

## Binary Container (`VHBC`)

The instruction words sit inside vihaco's `VHBC` section container, which
mirrors the `sst v1` text structure: one root section, whose header is the
version and whose bytecode is the code region.

```text
magic                : 4 bytes = b"VHBC"
version              : u16 LE  = 1
flags                : u16 LE  = 0
context_len          : u64 LE  = 0        (empty global context)
── root section ──
section_len          : u64 LE             (total, including this frame)
composite_header_len : u64 LE  = 4
composite header     : u32 LE  = (major << 16) | minor
bytecode_len         : u64 LE
bytecode             : N × instruction words
child_count          : u32 LE  = 0        (no child sections)
```

vihaco ships readers for this container but no writers, so Bloqade Lanes owns
the emitters (`isa::container`); the round-trip tests read everything back
through vihaco's own parser to keep the two in step.

Neither device's instruction enum carries a binary codec — vihaco-cpu's has none
and `#[composite]` derives none — so encoding goes through a parallel mirror ISA
(`isa::bytecode`) that does. PPVM solves this the same way.

## Text Format (`.sst`)

The text form is vihaco's `sst v1` section container. A lanes program is one
root section: a header carrying the version, and a text body holding a single
`@main` function.

```
sst v1

.section(root):
.header(root):
version 1.0
.header(root).
.text(root):
fn @main() {
  lanes::lanes.const_loc 0x0000000000000000
  lanes::lanes.const_loc 0x0000000001000000
  lanes::lanes.initial_fill 2
  cpu::cpu.halt
}
.text(root).
.section(root).
```

Container rules:

- `sst v1` must be the first significant line.
- The root section must be named `root`; a lanes program declares no child
  sections.
- `.name(x):` opens a block and `.name(x).` closes it — note the trailing dot
  versus colon.
- The global context block (`.global:` … `.global.`) may be omitted, and must be
  empty if present: a lanes program has no child-section names to resolve.
- Only `.global:` or the root section may appear between `sst v1` and the first
  section, so file-level comments belong **inside** `.text(root):`.

Instruction rules:

- Comments are `//` to end of line.
- Every instruction carries a **device prefix and dialect head**:
  `lanes::lanes.move 2`, `cpu::cpu.halt`. Both halves are required — a bare
  `move 2` does not parse, and neither does a mnemonic under the wrong device.
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

#### `cpu::cpu.const_int` — Push integer constant

| Field | Value |
|---|---|
| Opcode | `0x0011` |
| Operands | `i64` LE (8 bytes) |
| Stack | `( -- int)` |

Pushes a signed 64-bit integer onto the stack.

#### `cpu::cpu.const_float` — Push float constant

| Field | Value |
|---|---|
| Opcode | `0x0011` |
| Operands | `f64` LE (8 bytes) |
| Stack | `( -- float)` |

Pushes a 64-bit float onto the stack.

#### `cpu::cpu.dup` — Duplicate top of stack

| Field | Value |
|---|---|
| Opcode | `0x000D` |
| Operands | none |
| Stack | `(a -- a a)` |

#### `lanes::lanes.pop` — Discard top of stack

| Field | Value |
|---|---|
| Opcode | `0x0100` |
| Operands | none |
| Stack | `(a -- )` |

#### `lanes::lanes.swap` — Swap top two stack elements

| Field | Value |
|---|---|
| Opcode | `0x0101` |
| Operands | none |
| Stack | `(a b -- b a)` |

#### `cpu::cpu.return` — Return from program

| Field | Value |
|---|---|
| Opcode | `0x0006` |
| Operands | none |
| Stack | `( -- )` |

#### `cpu::cpu.halt` — Halt execution

| Field | Value |
|---|---|
| Opcode | `0x0009` |
| Operands | none |
| Stack | `( -- )` |

### Address constants

#### `lanes::lanes.const_loc` — Push location address

| Field | Value |
|---|---|
| Opcode | `0x0102` |
| Operands | `LocationAddr` as `u64` LE — `[zone_id:8][word_id:16][site_id:16][pad:24]` |
| Stack | `( -- loc)` |

#### `lanes::lanes.const_lane` — Push lane address

| Field | Value |
|---|---|
| Opcode | `0x0103` |
| Operands | `LaneAddr` as `u64` LE — `[dir:1][mt:2][zone_id:8][pad:5][bus_id:16][word_id:16][site_id:16]` |
| Stack | `( -- lane)` |

#### `lanes::lanes.const_zone` — Push zone address

| Field | Value |
|---|---|
| Opcode | `0x0104` |
| Operands | `ZoneAddr` as `u32` LE — `[pad:24][zone_id:8]` |
| Stack | `( -- zone)` |

### Atom arrangement

#### `lanes::lanes.initial_fill` — Initial atom loading

| Field | Value |
|---|---|
| Opcode | `0x0105` |
| Operands | `u32` LE arity |
| Stack | `(loc₁ loc₂ … locₙ -- )` |

Pops `n` location addresses and performs the initial atom fill at those sites.

#### `lanes::lanes.fill` — Atom refill

| Field | Value |
|---|---|
| Opcode | `0x0106` |
| Operands | `u32` LE arity |
| Stack | `(loc₁ loc₂ … locₙ -- )` |

Pops `n` location addresses and refills atoms at those sites.

#### `lanes::lanes.move` — Atom transport

| Field | Value |
|---|---|
| Opcode | `0x0107` |
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

#### `lanes::lanes.local_r` — Local R rotation

| Field | Value |
|---|---|
| Opcode | `0x0109` |
| Operands | `u32` LE arity |
| Stack | `(loc₁ loc₂ … locₙ θ φ -- )` |

Pops 2 float parameters (φ = axis angle, θ = rotation angle) then `n` location addresses, and applies a local R rotation. The call convention matches the SSA IR: `local_r(%φ, %θ, %loc₁, …)` — first argument (φ) is pushed last and popped first.

#### `lanes::lanes.local_rz` — Local Rz rotation

| Field | Value |
|---|---|
| Opcode | `0x0108` |
| Operands | `u32` LE arity |
| Stack | `(loc₁ loc₂ … locₙ θ -- )` |

Pops 1 float parameter (θ = rotation angle) then `n` location addresses, and applies a local Rz rotation. The call convention matches the SSA IR: `local_rz(%θ, %loc₁, …)`.

#### `lanes::lanes.global_r` — Global R rotation

| Field | Value |
|---|---|
| Opcode | `0x010B` |
| Operands | none |
| Stack | `(θ φ -- )` |

Pops 2 float parameters (φ = axis angle, θ = rotation angle), applies a global R rotation. The call convention matches the SSA IR: `global_r(%φ, %θ)`.

#### `lanes::lanes.global_rz` — Global Rz rotation

| Field | Value |
|---|---|
| Opcode | `0x010A` |
| Operands | none |
| Stack | `(θ -- )` |

Pops 1 float parameter (θ = rotation angle), applies a global Rz rotation. Since there is only one parameter, it is both pushed last and popped first.

#### `lanes::lanes.cz` — Controlled-Z gate

| Field | Value |
|---|---|
| Opcode | `0x010C` |
| Operands | none |
| Stack | `(zone -- )` |

Pops a zone address and applies a CZ gate across the zone.

### Measurement

#### `lanes::lanes.measure` — Initiate measurement

| Field | Value |
|---|---|
| Opcode | `0x010D` |
| Operands | `u32` LE arity |
| Stack | `(zone₁ zone₂ … zoneₙ -- future₁ future₂ … futureₙ)` |

Pops `n` zone addresses and pushes `n` measure futures.

#### `lanes::lanes.await_measure` — Wait for measurement result

| Field | Value |
|---|---|
| Opcode | `0x010E` |
| Operands | none |
| Stack | `(future -- array_ref)` |

Pops a measure future and pushes an array reference containing the measurement results.

### Arrays

#### `lanes::lanes.new_array` — Construct array from stack

| Field | Value |
|---|---|
| Opcode | `0x010F` |
| Operands | three `u32` LE: `type_tag`, `dim0`, `dim1` (`dim1 = 0` for 1-D) |
| Stack | `(elem₁ elem₂ … elemₙ -- array_ref)` |

Constructs an array of `dim0 × dim1` elements with element type `type_tag`. If `dim1` is 0, the array is 1-dimensional with `dim0` elements.

#### `lanes::lanes.get_item` — Index into array

| Field | Value |
|---|---|
| Opcode | `0x0110` |
| Operands | `u32` LE ndims |
| Stack | `(array_ref idx₁ … idxₙ -- value)` |

Pops `ndims` index values then the array reference, and pushes the indexed element.

### Detectors and observables

#### `lanes::lanes.set_detector` — Set detector

| Field | Value |
|---|---|
| Opcode | `0x0111` |
| Operands | none |
| Stack | `(array_ref -- detector_ref)` |

Pops an array reference and pushes a detector reference.

#### `lanes::lanes.set_observable` — Set observable

| Field | Value |
|---|---|
| Opcode | `0x0112` |
| Operands | none |
| Stack | `(array_ref -- observable_ref)` |

Pops an array reference and pushes an observable reference.
