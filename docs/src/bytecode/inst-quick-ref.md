# Instruction Quick Reference

A compact summary of the Bloqade Lanes instruction set. See the [Instruction Set](inst-spec.md) for full encoding details.

A lanes program runs on a **composite machine** of two devices: vihaco-cpu's
`CPU`, which supplies the stack and arithmetic ops, and the lanes device, which
supplies atom movement, gates, measurement and arrays. Each instruction is
spelled `<device>::<dialect>.<mnemonic>` — `lanes::lanes.move 2`, `cpu::cpu.halt`.

> **Opcodes shift.** The packed opcode is `(device_code << 8) | instruction_code`,
> and both halves are assigned by declaration order — so they renumber whenever
> either instruction set gains a variant. The values below are correct for this
> revision; compare identity with `op_name()`, never a literal.

## `lanes` device (`0x01`) — stack ops vihaco-cpu lacks

`pop` and `swap` live here because vihaco-cpu's CPU has neither. The machine
does the stack work on the device's behalf.

| Instruction | Opcode | Stack Effect | Description |
|-------------|--------|--------------|-------------|
| `lanes::lanes.pop` | `0x0100` | `(a -- )` | Discard top of stack |
| `lanes::lanes.swap` | `0x0101` | `(a b -- b a)` | Swap top two elements |

## `lanes` device (`0x01`) — address constants

Kept as lanes instructions, rather than plain `cpu::cpu.const u64`, so that
validation can tell a location from a lane from a zone.

| Instruction | Opcode | Stack Effect | Description |
|-------------|--------|--------------|-------------|
| `lanes::lanes.const_loc` | `0x0102` | `( -- loc)` | Push location address |
| `lanes::lanes.const_lane` | `0x0103` | `( -- lane)` | Push lane address |
| `lanes::lanes.const_zone` | `0x0104` | `( -- zone)` | Push zone address |

## `lanes` device (`0x01`) — atom arrangement

The only ops the machine actually **executes**: they drive the atom state and
fault on an illegal move.

| Instruction | Opcode | Stack Effect | Description |
|-------------|--------|--------------|-------------|
| `lanes::lanes.initial_fill` | `0x0105` | `(loc₁..locₙ -- )` | Initial atom loading |
| `lanes::lanes.fill` | `0x0106` | `(loc₁..locₙ -- )` | Atom refill |
| `lanes::lanes.move` | `0x0107` | `(lane₁..laneₙ -- )` | Atom transport along lanes |

## `lanes` device (`0x01`) — quantum gates

Not simulated — emitted as effects for hardware or a simulator downstream.

| Instruction | Opcode | Stack Effect | Description |
|-------------|--------|--------------|-------------|
| `lanes::lanes.local_rz` | `0x0108` | `(loc₁..locₙ θ -- )` | Local Rz rotation |
| `lanes::lanes.local_r` | `0x0109` | `(loc₁..locₙ θ φ -- )` | Local R rotation |
| `lanes::lanes.global_rz` | `0x010A` | `(θ -- )` | Global Rz rotation |
| `lanes::lanes.global_r` | `0x010B` | `(θ φ -- )` | Global R rotation |
| `lanes::lanes.cz` | `0x010C` | `(zone -- )` | Controlled-Z gate on zone |

## `lanes` device (`0x01`) — measurement

| Instruction | Opcode | Stack Effect | Description |
|-------------|--------|--------------|-------------|
| `lanes::lanes.measure` | `0x010D` | `(zone₁..zoneₙ -- future₁..futureₙ)` | Initiate measurement |
| `lanes::lanes.await_measure` | `0x010E` | `(future -- array_ref)` | Wait for measurement result |

## `lanes` device (`0x01`) — arrays

| Instruction | Opcode | Stack Effect | Description |
|-------------|--------|--------------|-------------|
| `lanes::lanes.new_array` | `0x010F` | `(elem₁..elemₙ -- array_ref)` | Construct array from stack |
| `lanes::lanes.get_item` | `0x0110` | `(array_ref idx₁..idxₙ -- value)` | Index into array |

## `lanes` device (`0x01`) — detectors and observables

| Instruction | Opcode | Stack Effect | Description |
|-------------|--------|--------------|-------------|
| `lanes::lanes.set_detector` | `0x0111` | `(array_ref -- detector_ref)` | Set detector |
| `lanes::lanes.set_observable` | `0x0112` | `(array_ref -- observable_ref)` | Set observable |

## `cpu` device (`0x00`) — vihaco-cpu

The full vihaco-cpu instruction set is available (42 ops: arithmetic,
comparison, bitwise, control flow, the heap allocator). The lanes compiler emits
only these four; the rest decode and validate but are never generated.

| Instruction | Opcode | Stack Effect | Description |
|-------------|--------|--------------|-------------|
| `cpu::cpu.const f64, <v>` | `0x0011` | `( -- float)` | Push 64-bit float constant |
| `cpu::cpu.const i64, <v>` | `0x0011` | `( -- int)` | Push 64-bit integer constant |
| `cpu::cpu.dup` | `0x000D` | `(a -- a a)` | Duplicate top of stack |
| `cpu::cpu.halt` | `0x0009` | `( -- )` | Halt execution |
| `cpu::cpu.ret <n>` | `0x0006` | `( -- )` | Return from program |

Note that `const` is one typed instruction, so `const f64` and `const i64` share
an opcode and differ in their operand. The Python `op_name()` still reports
`const_float` / `const_int`, because the decoder needs the distinction.
