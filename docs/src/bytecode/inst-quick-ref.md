# Instruction Quick Reference

A compact summary of all 24 bytecode instructions. See the [Instruction Set](inst-spec.md) for full encoding details.

## Cpu (`0x00`)

Stack manipulation, constants, and control flow (FLAIR-aligned).

| Instruction | Opcode | Stack Effect | Description |
|-------------|--------|--------------|-------------|
| `const_int` | `0x0200` | `( -- int)` | Push 64-bit integer constant |
| `const_float` | `0x0300` | `( -- float)` | Push 64-bit float constant |
| `dup` | `0x0400` | `(a -- a a)` | Duplicate top of stack |
| `pop` | `0x0500` | `(a -- )` | Discard top of stack |
| `swap` | `0x0600` | `(a b -- b a)` | Swap top two elements |
| `return` | `0x6400` | `( -- )` | Return from program |
| `halt` | `0xFF00` | `( -- )` | Halt execution |

## LaneConstants (`0x0F`)

Address constant instructions.

| Instruction | Opcode | Stack Effect | Description |
|-------------|--------|--------------|-------------|
| `const_loc` | `0x000F` | `( -- loc)` | Push location address |
| `const_lane` | `0x010F` | `( -- lane)` | Push lane address |
| `const_zone` | `0x020F` | `( -- zone)` | Push zone address |

## AtomArrangement (`0x10`)

Atom filling and transport.

| Instruction | Opcode | Stack Effect | Description |
|-------------|--------|--------------|-------------|
| `initial_fill` | `0x0010` | `(loc₁..locₙ -- )` | Initial atom loading |
| `fill` | `0x0110` | `(loc₁..locₙ -- )` | Atom refill |
| `move` | `0x0210` | `(lane₁..laneₙ -- )` | Atom transport along lanes |

## QuantumGate (`0x11`)

Single- and multi-qubit gate operations.

| Instruction | Opcode | Stack Effect | Description |
|-------------|--------|--------------|-------------|
| `local_r` | `0x0011` | `(θ φ loc₁..locₙ -- )` | Local R rotation |
| `local_rz` | `0x0111` | `(θ loc₁..locₙ -- )` | Local Rz rotation |
| `global_r` | `0x0211` | `(θ φ -- )` | Global R rotation |
| `global_rz` | `0x0311` | `(θ -- )` | Global Rz rotation |
| `cz` | `0x0411` | `(zone -- )` | Controlled-Z gate on zone |

## Measurement (`0x12`)

Qubit measurement.

| Instruction | Opcode | Stack Effect | Description |
|-------------|--------|--------------|-------------|
| `measure` | `0x0012` | `(zone₁..zoneₙ -- future₁..futureₙ)` | Initiate measurement |
| `await_measure` | `0x0112` | `(future -- array_ref)` | Wait for measurement result |

## Array (`0x13`)

Array construction and indexing.

| Instruction | Opcode | Stack Effect | Description |
|-------------|--------|--------------|-------------|
| `new_array` | `0x0013` | `(elem₁..elemₙ -- array_ref)` | Construct array from stack |
| `get_item` | `0x0113` | `(array_ref idx₁..idxₙ -- value)` | Index into array |

## DetectorObservable (`0x14`)

Detector and observable setup.

| Instruction | Opcode | Stack Effect | Description |
|-------------|--------|--------------|-------------|
| `set_detector` | `0x0014` | `(array_ref -- detector_ref)` | Set detector |
| `set_observable` | `0x0114` | `(array_ref -- observable_ref)` | Set observable |
