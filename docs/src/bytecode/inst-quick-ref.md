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

## `lanes` device (`0x01`) — address constants

Kept as lanes instructions, rather than plain `cpu::cpu.const u64`, so that
validation can tell a location from a lane from a zone.

| Instruction | Opcode | Stack Effect | Description |
|-------------|--------|--------------|-------------|
| `lanes::lanes.const_loc` | `0x0100` | `( -- loc)` | Push location address |
| `lanes::lanes.const_lane` | `0x0101` | `( -- lane)` | Push lane address |
| `lanes::lanes.const_zone` | `0x0102` | `( -- zone)` | Push zone address |

## `lanes` device (`0x01`) — atom arrangement

The only ops the machine actually **executes**: they drive the atom state and
fault on an illegal move.

| Instruction | Opcode | Stack Effect | Description |
|-------------|--------|--------------|-------------|
| `lanes::lanes.initial_fill` | `0x0103` | `(loc₁..locₙ -- )` | Initial atom loading |
| `lanes::lanes.fill` | `0x0104` | `(loc₁..locₙ -- )` | Atom refill |
| `lanes::lanes.move` | `0x0105` | `(lane₁..laneₙ -- )` | Atom transport along lanes |

## `lanes` device (`0x01`) — quantum gates

Not simulated — emitted as effects for hardware or a simulator downstream.

| Instruction | Opcode | Stack Effect | Description |
|-------------|--------|--------------|-------------|
| `lanes::lanes.local_rz` | `0x0106` | `(loc₁..locₙ θ -- )` | Local Rz rotation |
| `lanes::lanes.local_r` | `0x0107` | `(loc₁..locₙ θ φ -- )` | Local R rotation |
| `lanes::lanes.global_rz` | `0x0108` | `(θ -- )` | Global Rz rotation |
| `lanes::lanes.global_r` | `0x0109` | `(θ φ -- )` | Global R rotation |
| `lanes::lanes.cz` | `0x010A` | `(zone -- )` | Controlled-Z gate on zone |

## `lanes` device (`0x01`) — measurement

| Instruction | Opcode | Stack Effect | Description |
|-------------|--------|--------------|-------------|
| `lanes::lanes.measure` | `0x010B` | `(zone₁..zoneₙ -- future₁..futureₙ)` | Initiate measurement |
| `lanes::lanes.await_measure` | `0x010C` | `(future -- array_ref)` | Wait for measurement result |

## `lanes` device (`0x01`) — arrays

| Instruction | Opcode | Stack Effect | Description |
|-------------|--------|--------------|-------------|
| `lanes::lanes.new_array` | `0x010D` | `(elem₁..elemₙ -- array_ref)` | Construct array from stack |
| `lanes::lanes.get_item` | `0x010E` | `(array_ref idx₁..idxₙ -- value)` | Index into array |

## `lanes` device (`0x01`) — detectors and observables

| Instruction | Opcode | Stack Effect | Description |
|-------------|--------|--------------|-------------|
| `lanes::lanes.set_detector` | `0x010F` | `(array_ref -- detector_ref)` | Set detector |
| `lanes::lanes.set_observable` | `0x0110` | `(array_ref -- observable_ref)` | Set observable |

## `cpu` device (`0x00`) — vihaco-cpu

The full vihaco-cpu instruction set is available (42 ops: arithmetic,
comparison, bitwise, control flow, the heap allocator). The ones below are the
ones a lanes program actually contains; the rest decode and validate but are
never generated.

### Values, locals and termination

| Instruction | Opcode | Stack Effect | Description |
|-------------|--------|--------------|-------------|
| `cpu::cpu.const f64, <v>` | `0x0011` | `( -- float)` | Push 64-bit float constant |
| `cpu::cpu.const i64, <v>` | `0x0011` | `( -- int)` | Push 64-bit integer constant |
| `cpu::cpu.dup` | `0x000D` | `(a -- a a)` | Duplicate top of stack |
| `cpu::cpu.load <ty>, <n>` | `0x000B` | `( -- a)` | Push a copy of local `n` |
| `cpu::cpu.store <ty>, <n>` | `0x000C` | `(a -- )` | Pop the top into local `n` |
| `cpu::cpu.halt` | `0x0009` | `( -- )` | Halt execution |
| `cpu::cpu.ret <n>` | `0x0006` | `( -- )` | Return the top `n` values from the current function |

Note that `const` is one typed instruction, so `const f64` and `const i64` share
an opcode and differ in their operand. The Python `op_name()` still reports
`const_float` / `const_int`, because the decoder needs the distinction.

There is no `pop` or `swap`. A function's locals are slots of their own below
its operands, so parking a value in one takes it out of the operands' way:
`store <ty>, 0` discards the top, and `store 0; store 1; load 0; load 1` swaps
the top two. The compiler spills a value it needs more than once and loads it
back for each use.

### Structure

Emitted for **every** program, not only ones that declare several functions: the
assembler wraps each `fn @name()` body in a marker pair, so even a lone `@main`
is delimited. They are no-ops at runtime — they carry the layout, not an effect.

| Instruction | Opcode | Stack Effect | Description |
|-------------|--------|--------------|-------------|
| `cpu::cpu.func_start` | `0x0001` | `( -- )` | Opens a function body; its address is the function's entry |
| `cpu::cpu.func_end` | `0x0002` | `( -- )` | Closes a function body |

A `label` is not in this table because it is not an instruction: it names an
address and occupies none. It is written syntactically and recorded in the
`labels` table.

### Control flow

Emitted when the source declares them. All three require the architecture's
`feed_forward` capability — without mid-circuit classical feedback the hardware
runs straight-line code only, and validation rejects them.

| Instruction | Opcode | Stack Effect | Description |
|-------------|--------|--------------|-------------|
| `cpu::cpu.br @<label>` | `0x0004` | `( -- )` | Unconditional branch within the current function |
| `cpu::cpu.cond_br @<t>, @<f>` | `0x0005` | `(bool -- )` | Branch on the top of stack |
| `cpu::cpu.call <arity>, <fn>` | `0x0008` | `( -- )` | Call, making the top `arity` operands the callee's locals |

See [Functions, labels and control flow](inst-spec.md#functions-labels-and-control-flow)
for the calling convention — in particular that `call` moves a frame boundary
rather than copying arguments: the top `arity` operands *become* the callee's
first locals, and the rest of its locals are reserved above them.
