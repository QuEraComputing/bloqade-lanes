# Migrating to v0.12

This release upgrades the bytecode's underlying framework from vihaco 0.1 to
0.4. Both bytecode formats change. Nothing you previously assembled will load,
and `.sst` files need rewriting.

The Python API is **unchanged** — `Instruction` factories, `op_name()`, and the
decoder's `_visit_*` dispatch all behave as before. If you only build programs
through `bloqade.lanes`, there is nothing to do.

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
  lanes.const_loc 0x0000000000000000
  lanes.initial_fill 1
  cpu.const_float 1.5708
  lanes.global_rz
  cpu.halt
}
.text(root).
.section(root).
```

Point by point:

- **Every instruction needs a dialect head.** `lanes.` for the device ops
  (addresses, fills, moves, gates, measurement, arrays, detectors) and `cpu.`
  for the stack ops (`pop`, `swap`, `return`, `dup`, `halt`, `const_int`,
  `const_float`). A bare `move 2` no longer parses, and neither does a mnemonic
  under the wrong head.
- **The stack constants are renamed.** `const.i64 42` → `cpu.const_int 42`, and
  `const.f64 1.5` → `cpu.const_float 1.5`. The vihaco-cpu spellings are gone
  (see below).
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

The quickest way to migrate a file is to let the CLI do it: assemble with the
old toolchain, then disassemble with the new one.

## Binary format

The container changes from the Bloqade-specific `LANES` header to vihaco's
`VHBC` section container, and the instruction word narrows from 17 bytes to 13.

- **Magic is now `VHBC`**, not `LANES`. `BadMagicError`'s message changed to
  match.
- **Instruction words are 13 bytes**, not 17. The width was pinned at 17 to fit
  a nested vihaco-cpu instruction; with that nesting gone it follows our own
  operands, and is derived rather than chosen — it will move again if a wider
  operand is added.
- **Opcodes are renumbered.** They are assigned by variant declaration order, so
  they shift whenever the instruction set gains a variant. Compare instruction
  identity with `Instruction.op_name()`, never with a literal opcode value.
- **Structural container faults report differently.** A file too short to hold
  the file header still raises `TruncatedError` with real byte counts; vihaco's
  other structural complaints (a section running past the end of the file, an
  out-of-bounds header) raise `DecodeErrorInProgram` carrying vihaco's own
  message, because they carry no byte counts to report.

Re-assemble any persisted `.bin` from source.

## Why the CPU instructions changed

In vihaco 0.1 the instruction set nested vihaco-cpu's opcodes wholesale, which
is where `const.i64` / `const.f64` came from. As of vihaco 0.4, vihaco-cpu is a
runtime *component* rather than an opcode library: its instruction enums carry
no binary codec, so they can no longer be nested in an encodable instruction
set.

The four stack ops Bloqade Lanes actually uses — `const_float`, `const_int`,
`dup`, `halt` — are therefore declared natively, alongside `pop`, `swap` and
`return`, which already were. They keep the `cpu.` namespace to signal that
their semantics still mirror vihaco-cpu's.

One consequence worth noting: the instruction set no longer has any branch or
call instructions, because those arrived only via the nesting. Programs could
not use them meaningfully before, and `ValidationError::ControlFlowRequiresFeedForward`
is now unreachable — it is retained because `feed_forward` remains a real
capability and control flow is expected to return.
