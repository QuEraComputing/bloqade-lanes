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
| `lanes` | atom movement, gates, measurement, arrays |

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
- **There is no `pop` or `swap`.** vihaco-cpu's CPU has neither, and a
  function's locals now spell both — see
  [Functions have locals of their own](#functions-have-locals-of-their-own).
  `pop` becomes `cpu::cpu.store <ty>, <n>` into a scratch local; `swap`
  becomes two `store`s and two `load`s.
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
- **Removed:** `Instruction.pop()` and `Instruction.swap()`, and the
  `stack_move.Pop` / `stack_move.Swap` statements. **New:**
  `Instruction.load(value_type, index)` and `Instruction.store(value_type,
  index)`, with `local_index()` / `value_type()` accessors, and the
  `stack_move.LoadLocal` / `stack_move.StoreLocal` statements they decode to.
  `value_type` is spelled as in the text format (`"u64"`, `"undef"`, …).
  `stack_move.Dup` stays: `dup` is vihaco-cpu's own.
- **Changed:** `stackify` no longer emits `Dup`/`Swap` chains for a value with
  several consumers. It stores the value in a local right after it is produced
  and loads a copy before each consumer, reusing a slot once its value is dead.
  Compiled programs therefore contain `cpu::cpu.store undef, 0` /
  `cpu::cpu.load undef, 0` where they used to contain `dup` and `swap`:
  `undef`, because the value is the placeholder a lanes op pushes in place of a
  result it does not simulate. Slots scale with how many values are live at
  once, and `stackify` raises `ValueError` rather than need more than the 1024
  a frame may hold. It also raises on two shapes only decoded bytecode has — a
  `stack_move.Dup`, and a constant operand below a non-constant one — which it
  would otherwise reorder
  ([#1050](https://github.com/QuEraComputing/bloqade-lanes/issues/1050)).
- **New:** `Program.entry_parameters` lists the entry point's declared
  parameter types. `BytecodeDecoder.decode` raises `DecodingError` for an entry
  point that declares any: the kernel it builds takes no arguments, and a
  parameter would otherwise read as zero.
- **Behaviour:** `RewriteStackMoveToMove` lowers a `load` to the value last
  stored only when it reads that value back as itself: `load undef` of a lanes
  op's placeholder, or `load <ty>` of a constant of that type. A typed load of
  a placeholder, which the machine reads as that type's zero, and a mistyped
  `load`/`store` of a constant, which it refuses, raise `ValueError`.
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

## Functions and control flow

Programs are no longer restricted to a single flat `@main`. Any number of
functions may be declared, and branch and call targets are written as symbols:

```
fn @main() {
  cpu::cpu.call 0, helper
  cpu::cpu.br @done
  lanes::lanes.cz
  cpu::cpu.label @done
  cpu::cpu.halt
}

fn @helper() {
  cpu::cpu.ret 0
}
```

Nothing the lanes compiler emits uses this yet — it still produces a single flat
`@main` — but the format and loader support it now rather than being retrofitted
later.

- **Each function body is wrapped in `cpu::cpu.func_start` / `cpu::cpu.func_end`.**
  These are emitted by the assembler and executed as no-ops; you do not write
  them. They make a function's extent part of the code stream rather than a
  span recorded beside it, so an empty function still occupies an address and
  a function table can never disagree with the code it indexes. They are real
  instructions, so they appear in `Program.instructions` and count towards
  `len(program)` — a three-instruction `@main` is five words.
- `br` / `cond_br` name a **label** with `@`; `call` names a **function**
  without one (`call <arity>, <name>`).
- Labels are module-global; duplicates are an error.
- A label occupies no address and is not stored in the code stream. It is
  recorded in the label table, and re-emitted when the program is written back
  out.
- The binary container gained three child sections — `functions`, `labels`,
  `strings` — so these survive a round-trip. A file without them does not load:
  it predates the function markers too, so there are no extents to name. This
  is the same "replace, do not convert" rule as the text format above.

### Functions declare their signatures

A function may declare parameters and a return type, and both survive the
binary round trip:

```
fn @measure_zone(z: u32) -> heap_ref {
  cpu::cpu.load u32, 0
  lanes::lanes.measure 1
  lanes::lanes.await_measure
  cpu::cpu.ret 1
}
```

Types are vihaco's: `undef`, `str`, `bool`, `i64`, `u32`, `u64`, `f64`,
`fn_ref`, `heap_ref`. `fn @name()` with no parameters and no return type parses
exactly as before, so nothing hand-written needs updating unless it uses a
`call` with a nonzero arity or a `ret` that keeps a value.

The declaration is checked, not decorative. `call <arity>` is not a hint — it
sets the callee's frame base to `stack.len() - arity`, so an unchecked operand
would silently redefine the callee's shape at each call site:

- **`CallArityMismatchError`** — a `call` passes a different number of operands
  than the callee declares.
- **`ReturnCountMismatchError`** — a `ret` keeps a different number of values
  than its function declares returning. Two `ret`s that disagree make every
  caller's post-call stack depth path-dependent; each is checked against the
  declaration, which names the offender rather than reporting a pair that
  happens to differ.

So a function that returns something has to say so: `ret 1` in a function
declaring no return type is now an error.

`@main` may declare parameters too. Its caller is the host, which passes the
arguments with `LanesMachine::run_with_args`. They are checked against the
declaration's count and types, and they sit at the bottom of the entry frame
as locals `0..n`, where a `call` would put them. `run` enters with no
arguments, so it refuses a `@main` that declares any. The CLI's `run`
subcommand has no way to pass arguments yet.

### Functions have locals of their own

A frame is its locals — the parameters, then scratch slots — and above them
its operands. This is the frame model of vihaco#110, which vihaco has merged
but not yet released; `LanesMachine` runs it on vihaco 0.4.1 until the
dependency moves past it.

- **A function reserves `max(arity, every load/store index + 1)` locals**,
  counted from its body. There is nothing to declare, and the function table's
  `local_count` — always 0 before — now carries that count. It is recomputed
  from the code whenever a program is loaded, so a table cannot disagree with
  its body.
- **A local holding the `Undefined` placeholder reads as zero** of whatever
  type loads it: an unwritten one, so a counter needs no initialising `store`,
  and one holding the placeholder a lanes op pushes in place of a result it
  does not simulate. `load undef` reads that placeholder back as itself. A
  local holding a concrete value of another type is still a type error.
- **Only `load` and `store` reach a local.** No operand op consumes one, a
  parameter included: a callee that used to consume its argument directly now
  has to `load` it first, or it underflows. `store` no longer grows the stack
  to reach its index.

So a program that relied on 0.4.1's aliasing — local `n` being whatever sat
`n` slots above the frame base — behaves differently: a `store` into a scratch
slot used to overwrite a working value, and now does not.

### Stack validation follows control flow

`validate(stack=True)` (the CLI's `--simulate-stack`, the C API's
`lanes_simulate_stack`) used to walk each function in a straight line from an
empty stack. It stopped at the first `br`, `cond_br` or `call`, and it skipped
any function that a `call` passed operands to. It now walks every function's
control-flow graph, starting from a frame that holds its declared parameters:

- **Code after a branch or call is checked.** A program with control flow no
  longer gets less validation than one without.
- **A callee that takes operands is checked.** Its frame starts with its
  locals — declared parameters, then scratch — so the `load` that reads an
  argument is no longer an underflow. A `call` is modelled from the callee's
  signature alone: pop its parameters, push its results.
- **`StackDepthMismatchError`**: two paths reach the same instruction with
  different stack depths. Examples are a `cond_br` whose arms leave different
  depths at their join, or a loop whose body changes the depth.
- **`PopBelowFrameBaseError`**: a function other than `@main` pops with no
  operands left, reaching into its own locals and below them the values its
  caller owns. This is an error even when the machine's stack is not empty. It
  subclasses `StackUnderflowError`, so existing `except` clauses still catch it.
  In `@main`, which has no caller, the same condition is still reported as a
  plain `StackUnderflowError`.
- **`LocalTypeMismatchError`**: a typed `load`/`store` names a type the value
  does not have, so the machine refuses it. A `store` checks the value it pops;
  a `load` checks what the local holds. The placeholder a lanes op pushes for a
  result it does not simulate passes under any type, except that `load undef`
  refuses every concrete value. A call's result is checked by what the callee's
  `ret` really keeps, not its declared return type, which nothing enforces.
- **`TooManyParametersError`**: a function declares more parameters than the
  1024 locals a frame may hold, so every call to it would fail.

Everything after a `call_indirect` goes unchecked, because its target and
arity are only known at run time. That includes any path that merges with it
afterwards: an error on the arm without the call is not reported. A branch
condition's type is not checked either, only that one is present.

### Lowering to kirin is single-function only

`BytecodeDecoder.decode` (and `load_program`) lower the instruction stream into
one kirin block, so they accept a program declaring exactly one function and
refuse anything else:

```
DecodingError at instruction 3 (func_start): program declares 2 functions;
only a single-function program can be lowered to kirin [stack depth=0]
```

The format is ahead of the compiler here on purpose. Nothing in the pipeline
emits multi-function bytecode or lowers it further, so there is no correct
lowering for the decoder to fall back to — only a silently wrong one. It used
to take that one: the bodies concatenated into a single block, and because the
marker handlers skip `func_start` / `func_end` the seam left no trace. A
`@helper` declared before `@main` produced a kernel whose *first* statement was
the helper's `func.return`, carrying two terminators — which `method.verify()`
accepted.

Validation, execution, disassembly and the binary round-trip are unaffected;
this restriction applies only to the kirin lowering.

**Stack validation stops at the first branch or call.** The type simulator walks
straight through, so its state is only correct while control flow is linear;
past a branch it would report underflows and mismatches derived from a state it
cannot know. The linear prefix is still checked. Full CFG-aware simulation is
tracked in [#1042](https://github.com/QuEraComputing/bloqade-lanes/issues/1042).

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
