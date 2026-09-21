# Instruction Quick Reference

A compact summary of all 24 bytecode instructions. See the [Instruction Set](inst-spec.md) for full encoding details.

Instructions are spelled in `.sst` text under a **dialect head**: `cpu.` for the
stack ops, `lanes.` for the device ops. The head is part of the syntax —
`lanes.move 2` parses, a bare `move 2` does not.

> **Opcodes shift.** vihaco assigns the opcode byte from the variant's position
> in the `Instruction` enum, so adding an instruction anywhere but the end
> renumbers everything after it. The values below are correct for this revision;
> `Instruction::opcode()` is the authority.

## `cpu.*` — stack ops

Stack manipulation, constants, and termination. These mirror vihaco-cpu's stack
ops but are declared natively (see [Instruction Set](inst-spec.md#stack-ops)).

| Instruction | Opcode | Stack Effect | Description |
|-------------|--------|--------------|-------------|
| `cpu.pop` | `0x00` | `(a -- )` | Discard top of stack |
| `cpu.swap` | `0x01` | `(a b -- b a)` | Swap top two elements |
| `cpu.return` | `0x02` | `( -- )` | Return from program |
| `cpu.dup` | `0x03` | `(a -- a a)` | Duplicate top of stack |
| `cpu.halt` | `0x04` | `( -- )` | Halt execution |
| `cpu.const_float` | `0x05` | `( -- float)` | Push 64-bit float constant |
| `cpu.const_int` | `0x06` | `( -- int)` | Push 64-bit integer constant |

## `lanes.*` — address constants

| Instruction | Opcode | Stack Effect | Description |
|-------------|--------|--------------|-------------|
| `lanes.const_loc` | `0x07` | `( -- loc)` | Push location address |
| `lanes.const_lane` | `0x08` | `( -- lane)` | Push lane address |
| `lanes.const_zone` | `0x09` | `( -- zone)` | Push zone address |

## `lanes.*` — atom arrangement

Atom filling and transport.

| Instruction | Opcode | Stack Effect | Description |
|-------------|--------|--------------|-------------|
| `lanes.initial_fill` | `0x0A` | `(loc₁..locₙ -- )` | Initial atom loading |
| `lanes.fill` | `0x0B` | `(loc₁..locₙ -- )` | Atom refill |
| `lanes.move` | `0x0C` | `(lane₁..laneₙ -- )` | Atom transport along lanes |

## `lanes.*` — quantum gates

Single- and multi-qubit gate operations.

| Instruction | Opcode | Stack Effect | Description |
|-------------|--------|--------------|-------------|
| `lanes.local_rz` | `0x0D` | `(loc₁..locₙ θ -- )` | Local Rz rotation |
| `lanes.local_r` | `0x0E` | `(loc₁..locₙ θ φ -- )` | Local R rotation |
| `lanes.global_rz` | `0x0F` | `(θ -- )` | Global Rz rotation |
| `lanes.global_r` | `0x10` | `(θ φ -- )` | Global R rotation |
| `lanes.cz` | `0x11` | `(zone -- )` | Controlled-Z gate on zone |

## `lanes.*` — measurement

| Instruction | Opcode | Stack Effect | Description |
|-------------|--------|--------------|-------------|
| `lanes.measure` | `0x12` | `(zone₁..zoneₙ -- future₁..futureₙ)` | Initiate measurement |
| `lanes.await_measure` | `0x13` | `(future -- array_ref)` | Wait for measurement result |

## `lanes.*` — arrays

Array construction and indexing.

| Instruction | Opcode | Stack Effect | Description |
|-------------|--------|--------------|-------------|
| `lanes.new_array` | `0x14` | `(elem₁..elemₙ -- array_ref)` | Construct array from stack |
| `lanes.get_item` | `0x15` | `(array_ref idx₁..idxₙ -- value)` | Index into array |

## `lanes.*` — detectors and observables

| Instruction | Opcode | Stack Effect | Description |
|-------------|--------|--------------|-------------|
| `lanes.set_detector` | `0x16` | `(array_ref -- detector_ref)` | Set detector |
| `lanes.set_observable` | `0x17` | `(array_ref -- observable_ref)` | Set observable |
