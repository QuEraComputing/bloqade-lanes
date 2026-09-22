# Migrating to v0.12

This release upgrades the bytecode's underlying framework from vihaco 0.1 to
0.4 and restructures the instruction set onto vihaco's composite-machine model.
Both bytecode formats change. Nothing you previously assembled will load, and
`.sst` files need rewriting.

The Python **factory API is unchanged** — `Instruction.const_float(...)`,
`Instruction.move_(...)`, `op_name()` and the decoder all behave as before. If
you only build programs through `bloqade.lanes`, there is nothing to do. Two
additions and one behaviour change are noted under
[Python](#python) below.

## The machine

A lanes program now runs on a composite of two vihaco devices rather than a
single flat instruction set:

| Device | Supplies |
|---|---|
| `cpu` | vihaco-cpu's `CPU` component — stack, constants, arithmetic, control flow, the heap allocator |
| `lanes` | atom movement, gates, measurement, arrays, and the `pop`/`swap` the CPU lacks |

This is the idiomatic vihaco composition (PPVM is built the same way), and it
means a lanes program can now be **executed**, via the new `bloqade-bytecode
run` subcommand:

```bash
bloqade-bytecode run prog.sst --arch gemini-logical.json
# halted after 50 instruction(s); 4 atom(s) placed
```

What runs is the *atom movement*: `initial_fill`, `fill` and `move` advance
the atom state and fail on an illegal move. That is a check validation cannot
make — the validator does not track which sites are occupied — and it found a
bug in this repository's own `stack_full_pipeline.sst` example, which
validated clean while refilling the two sites its own `move` had just filled.

The quantum, array and measurement ops are reported as effects rather than
simulated. Each carries the operands it consumed, so an observer knows which
array a `set_detector` referenced and not merely that one happened; see
[#1022](https://github.com/QuEraComputing/bloqade-lanes/issues/1022). They
still push placeholder values, so the stack stays at the depth validation
predicts and a program that validates does not underflow when it runs.

## Text format (`.sst`)

Two changes stack: instructions gained a dialect head, and the whole program is
now wrapped in vihaco's `sst v1` section container.

Before:

```
version 1.0;
fn @main() {
  const_loc 0x0000000000000000
  initial_fill 1
  const.f64 1.5708
  global_rz
  halt
}
```

After:

```
sst v1

.section(root):
.header(root):
version 1.0
.header(root).
.text(root):
fn @main() {
  lanes::lanes.const_loc 0x0000000000000000
  lanes::lanes.initial_fill 1
  cpu::cpu.const f64, 1.5708
  lanes::lanes.global_rz
  cpu::cpu.halt
}
.text(root).
.section(root).
```

Point by point:

- **Every instruction needs a device prefix and a dialect head**, written
  `<device>::<dialect>.<mnemonic>`. The device ops (addresses, fills, moves,
  gates, measurement, arrays, detectors) are `lanes::lanes.*`; the CPU ops are
  `cpu::cpu.*`. A bare `move 2` no longer parses, and neither does a mnemonic
  under the wrong device.
- **`pop` and `swap` are lanes ops**, not CPU ops: `lanes::lanes.pop`,
  `lanes::lanes.swap`. vihaco-cpu's CPU has neither.
- **The constants use vihaco-cpu's single typed `const`.** `const.i64 42` →
  `cpu::cpu.const i64, 42`, and `const.f64 1.5` → `cpu::cpu.const f64, 1.5` —
  note the comma. The Python `op_name()` still reports `const_float` /
  `const_int`, because the decoder needs that distinction.
- **`return` is now `ret` with a keep-count**: `cpu::cpu.ret 0`. `op_name()`
  still reports `"return"`.
- **The version directive moves into a header section** and loses its trailing
  semicolon: `version 1.0;` → a `.header(root):` block containing `version 1.0`.
- **The program body moves into `.text(root):`**, and the root section must be
  named `root`.
- **Comments must be inside `.text(root):`.** vihaco's container grammar allows
  only `.global:` or the root section between `sst v1` and the first section, so
  a file-level comment above the section is a parse error. Comments are still
  `//` to end of line.
- **The global context block is optional** and must be empty if present — a
  lanes program declares no child sections.

**There is no conversion path, and no toolchain combination that provides
one.** The old CLI writes the `LANES` container, which the new one rejects at
the magic bytes; the new parser rejects the old text syntax. So neither
direction of assemble/disassemble can bridge the two formats — an existing
`.sst` or `.bin` has to be replaced, not converted.

Two ways to do that:

- **Regenerate from source.** Anything produced by the `bloqade.lanes`
  compilation pipeline is reproducible: re-run the pipeline and it emits the
  new format. This is the right answer for every `.bin` and for any `.sst` that
  was generated rather than written.
- **Rewrite a hand-written `.sst`.** Apply the point-by-point list above:
  prefix each instruction, fold the constants into `cpu::cpu.const <type>,
  <value>`, rename `return` to `ret <n>`, and wrap the whole thing in the
  `sst v1` container. Then `bloqade-bytecode validate <file> --simulate-stack`
  to confirm it took.

## Binary format

The container changes from the Bloqade-specific `LANES` header to vihaco's
`VHBC` section container, and the instruction word narrows from 17 bytes to 14.

- **Magic is now `VHBC`**, not `LANES`. `BadMagicError`'s message changed to
  match.
- **Instruction words are 14 bytes**, not 17: a device byte, an instruction
  byte, and up to 12 bytes of operand. The width is derived as the widest
  variant across both devices rather than chosen, so it moves again if either
  device gains a wider operand.
- **Opcodes are renumbered and repacked.** `Instruction.opcode` is now
  `(device_code << 8) | instruction_code` — `0x00` for the CPU, `0x01` for the
  lanes device. Both halves are assigned by declaration order, so they shift
  whenever either instruction set gains a variant. Compare instruction identity
  with `Instruction.op_name()`, never with a literal opcode value.
- **Structural container faults report differently.** A file too short to hold
  the file header still raises `TruncatedError` with real byte counts; vihaco's
  other structural complaints (a section running past the end of the file, an
  out-of-bounds header) raise `DecodeErrorInProgram` carrying vihaco's own
  message, because they carry no byte counts to report.

Re-assemble any persisted `.bin` from source.

## Python

- **New:** `Instruction.device()` returns `"cpu"` or `"lanes"`.
- **Changed:** `Instruction.opcode` is repacked (see above).
- **Behaviour:** a decoded program can now contain any of vihaco-cpu's 42 ops,
  including arithmetic and control flow the lanes compiler never emits. Those
  load and validate fine, but the `stack_move` dialect has no statement for
  them, so `BytecodeDecoder.decode` raises `DecodingError` naming the
  instruction and its index:

  ```
  DecodingError at instruction 2 (add): `cpu::add` has no stack_move
  representation [stack depth=2]
  ```

  Previously every decodable instruction had a handler, so this path was
  unreachable from a valid program.

## Why the CPU instructions changed

In vihaco 0.1 the instruction set nested vihaco-cpu's opcodes wholesale, which
is where `const.i64` / `const.f64` came from. As of vihaco 0.4, vihaco-cpu is a
runtime *component* rather than an opcode library: its instruction enums carry
no binary codec, so they can no longer be nested in an encodable instruction
set.

Rather than reimplement those ops, the machine now **composes vihaco-cpu's CPU
as a device**, which is how vihaco intends a VM to be built. The binary format
still needs a codec neither device provides, so encoding goes through a parallel
mirror instruction set — the same approach PPVM takes.

`pop` and `swap` are the exception: vihaco-cpu has neither, and the lanes
compiler emits both, so they live on the lanes device.
