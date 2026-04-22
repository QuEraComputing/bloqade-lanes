# `stack_move` Dialect — Design

**Date**: 2026-04-21
**Status**: Draft
**Tracking issue**: #248 (scope revised — see below)
**Companion doc**: [2026-04-21-bytecode-to-ssa-lowering.md](./2026-04-21-bytecode-to-ssa-lowering.md)

## Goal

Accept Rust-backed `bloqade.lanes.bytecode.Program` objects as a first-class input to the Bloqade Lanes compiler pipeline. A user loads a bytecode program (e.g. via `Program.from_binary`) and the decoder turns it into a Kirin `ir.Method` that the existing downstream passes (state, transversal, move2squin, validation, visualization, metrics) consume unchanged.

This is an additive change. No existing dialect, pass, or public API is broken.

## Design stance

The `stack_move` dialect has **one statement per bytecode instruction** and every statement uses **explicit SSA operands and results**. The decoder's job is the inverse of the usual SSA-to-stack compilation: going *from* stack registers *to* explicit SSA values. Each bytecode stack position becomes a named SSA value; each bytecode instruction becomes a `stack_move` statement that consumes and/or produces those named values.

Stack-manipulation instructions (`Pop` / `Dup` / `Swap`) are preserved as **first-class SSA statements** rather than being consumed invisibly at decode time:

```text
%loc0 = stack_move.ConstLoc [value=LocationAddress(0,0,0)]
%loc1 = stack_move.ConstLoc [value=LocationAddress(0,0,1)]
%loc2 = stack_move.Dup %loc1                # identity — %loc2 ≡ %loc1
%a, %b = stack_move.Swap %loc0, %loc1       # reorder — %a ≡ %loc1, %b ≡ %loc0
        stack_move.Pop %loc2                # discard
        stack_move.Fill %a, %b              # arity=2 implicit in operand count
```

This is the "linear IR" style — categorical / explicit-structure rather than the mainstream "stack ops vanish" Cranelift/V8 style. The motivation is twofold:

1. **Atom non-cloning as a first-class invariant.** The Bloqade Lanes machine cannot duplicate atoms. Keeping `Dup` visible as an SSA op gives downstream passes (e.g. a linear-type checker) a hook to enforce that invariant — any `Dup` whose operand has an atom-bearing type is a physical error.
2. **Faithful round-tripping to bytecode.** Because every bytecode instruction has a corresponding `stack_move` statement, lowering from a Kirin frontend to bytecode later is a structural inverse of decoding.

This work is the **first instance of a bytecode decoder framework** for Bloqade Lanes. The v1 implementation ships as one monolithic dialect + one rewrite to get something small and end-to-end running. Once the boundaries are validated in practice, the dialect decomposes into sub-dialects, the decoder decomposes into registered per-sub-dialect handlers, and the rewrite decomposes into composable per-sub-dialect rewrite chunks (see §"Planned follow-up: sub-dialect decomposition" below).

## Scope

**In scope (v1):**

- A new Kirin dialect `stack_move` with one SSA-using statement per bytecode instruction, including `Pop`, `Dup`, `Swap` as explicit SSA ops.
- A decoder that translates a `Program` instruction-by-instruction into a `stack_move`-dialect `ir.Method`, simulating a virtual stack of SSA values during decoding.
- A rewrite pass `lower_stack_move` that does per-statement mechanical translation from `stack_move` into target dialects (`move`, `ilist`, `py.constant`, `py.indexing`, `annotate`, `func`), eliminating `Pop`/`Dup`/`Swap` along the way.
- A new multi-zone `move.Measure` statement (target of the rewrite).
- Extensions to `AtomAnalysis` that track measurement zone sets and whether the program performs a single final measurement.
- A new `move → move` rewrite pass `measure_lower` that uses the extended analysis to validate preconditions and lower `move.Measure` into the existing `move.EndMeasure`.

**Out of scope (follow-ups):**

- Exposing per-instruction operand accessors on the `Instruction` PyO3 binding. Prerequisite — tracked separately.
- File/bytes entry points. `Program.from_binary` and `Program.from_text` already exist.
- Changes to `move2squin` or any other downstream pass.
- The linear-type checker that would enforce atom non-cloning by catching disallowed `Dup` uses. Separate pass, separate spec.

## Architecture

```text
┌─────────────┐  load_program   ┌──────────────┐  lower_stack_move    ┌──────────┐  measure_lower     ┌──────────┐
│  Program    │────────────────▶│  stack_move  │─────────────────────▶│  multi-  │───────────────────▶│  move    │
│  (Rust)     │ (virtual SSA    │  ir.Method   │  (per-statement      │ dialect  │  (analysis +       │  (old,   │
└─────────────┘  stack, emits   │  (SSA with   │   mechanical         │ IR       │   rewrite pass)    │  EndMea-)│
                 explicit SSA)  │   Pop/Dup/   │   translation,        │ (move,   │                    └──────────┘
                                │   Swap)      │   eliminates stack    │ ilist,   │                         │
                                └──────────────┘   ops)                │ etc.)    │                         ▼
                                                                       └──────────┘               existing downstream pipeline
                                                                                                   (transversal, move2squin, …)
```

## New code

### 1. `python/bloqade/lanes/dialects/stack_move.py`

New Kirin dialect. **Every statement uses SSA operands and results.** No state threading — state plumbing is inserted when lowering into the `move` dialect.

**Types:**

- `LocationAddressType`, `LaneAddressType`, `ZoneAddressType`, `MeasurementFutureType`, `BitstringType`, `ArrayType`
- Existing kirin types are reused for `FloatType`, `IntType`.

**Constants** (attribute + SSA result):

```python
class ConstFloat(ir.Statement):
    value: float = info.attribute()
    result: ir.ResultValue = info.result(FloatType)

class ConstInt(ir.Statement):
    value: int = info.attribute()
    result: ir.ResultValue = info.result(IntType)

class ConstLoc(ir.Statement):
    value: LocationAddress = info.attribute()
    result: ir.ResultValue = info.result(LocationAddressType)

class ConstLane(ir.Statement):
    value: LaneAddress = info.attribute()
    result: ir.ResultValue = info.result(LaneAddressType)

class ConstZone(ir.Statement):
    value: ZoneAddress = info.attribute()
    result: ir.ResultValue = info.result(ZoneAddressType)
```

**Stack manipulation** (explicit SSA inputs/outputs):

```python
class Pop(ir.Statement):
    # consumes 1 SSA value, produces none
    value: ir.SSAValue = info.argument()

class Dup(ir.Statement):
    # semantic identity — result ≡ value — but preserved as explicit op
    value: ir.SSAValue = info.argument()
    result: ir.ResultValue = info.result()

class Swap(ir.Statement):
    # permutation — out_top ≡ in_bot, out_bot ≡ in_top
    in_top: ir.SSAValue = info.argument()
    in_bot: ir.SSAValue = info.argument()
    out_top: ir.ResultValue = info.result()
    out_bot: ir.ResultValue = info.result()
```

`Dup` and `Swap` are semantically identity / permutation, respectively, but preserved as explicit SSA ops for the reasons in §"Design stance" (linear-type hook, round-trippability).

**Atom operations** (SSA locations/lanes; arity is implicit in operand count; no state threading):

```python
class InitialFill(ir.Statement):
    locations: tuple[ir.SSAValue, ...] = info.argument(type=LocationAddressType)

class Fill(ir.Statement):
    locations: tuple[ir.SSAValue, ...] = info.argument(type=LocationAddressType)

class Move(ir.Statement):
    lanes: tuple[ir.SSAValue, ...] = info.argument(type=LaneAddressType)
```

**Gates:**

```python
class LocalR(ir.Statement):
    phi: ir.SSAValue = info.argument(type=FloatType)
    theta: ir.SSAValue = info.argument(type=FloatType)
    locations: tuple[ir.SSAValue, ...] = info.argument(type=LocationAddressType)

class LocalRz(ir.Statement):
    theta: ir.SSAValue = info.argument(type=FloatType)
    locations: tuple[ir.SSAValue, ...] = info.argument(type=LocationAddressType)

class GlobalR(ir.Statement):
    phi: ir.SSAValue = info.argument(type=FloatType)
    theta: ir.SSAValue = info.argument(type=FloatType)

class GlobalRz(ir.Statement):
    theta: ir.SSAValue = info.argument(type=FloatType)

class CZ(ir.Statement):
    zone: ir.SSAValue = info.argument(type=ZoneAddressType)
```

**Measurement:**

```python
class Measure(ir.Statement):
    locations: tuple[ir.SSAValue, ...] = info.argument(type=LocationAddressType)
    result: ir.ResultValue = info.result(MeasurementFutureType)

class AwaitMeasure(ir.Statement):
    future: ir.SSAValue = info.argument(type=MeasurementFutureType)
    result: ir.ResultValue = info.result(BitstringType)
```

`Measure` takes **location** operands here (matching the bytecode's pop shape). Zone grouping happens in `lower_stack_move`.

**Arrays and data:**

```python
class NewArray(ir.Statement):
    type_tag: int = info.attribute()
    dim0: int = info.attribute()
    dim1: int = info.attribute()  # 0 when 1-D
    result: ir.ResultValue = info.result(ArrayType)

class GetItem(ir.Statement):
    array: ir.SSAValue = info.argument(type=ArrayType)
    indices: tuple[ir.SSAValue, ...] = info.argument(type=IntType)
    result: ir.ResultValue = info.result()  # element type

class SetDetector(ir.Statement):
    array: ir.SSAValue = info.argument(type=ArrayType)

class SetObservable(ir.Statement):
    array: ir.SSAValue = info.argument(type=ArrayType)
```

**Control flow:**

```python
class Return(ir.Statement):
    # optional return operand (None for Halt)
    value: ir.SSAValue | None = info.argument(default=None)

class Halt(ir.Statement):
    pass  # lowered to Return(None)
```

### 2. `python/bloqade/lanes/bytecode/lowering.py`

The decoder maintains a **virtual stack of SSA values** while walking the bytecode. Each bytecode instruction is dispatched to a per-opcode handler that reads/mutates the virtual stack and emits a `stack_move` statement with the right SSA operands.

```python
class BytecodeDecoder:
    stack: list[ir.SSAValue]
    block: ir.Block

    def decode(self, program: Program, kernel_name: str) -> ir.Method:
        for idx, instruction in enumerate(program.instructions):
            try:
                self._visit(instruction)
            except _StackError as e:
                raise DecodeError.from_context(idx, instruction, self.stack, e)
        return self._finalize(kernel_name)

    def _visit(self, instr: Instruction) -> None:
        # dispatch by opcode
        ...

    def _visit_const_loc(self, instr):
        stmt = stack_move.ConstLoc(value=instr.location_address())
        self.block.stmts.append(stmt)
        self.stack.append(stmt.result)

    def _visit_dup(self, instr):
        top = self.stack[-1]
        stmt = stack_move.Dup(value=top)
        self.block.stmts.append(stmt)
        self.stack.append(stmt.result)

    def _visit_swap(self, instr):
        in_top, in_bot = self.stack.pop(), self.stack.pop()
        stmt = stack_move.Swap(in_top=in_top, in_bot=in_bot)
        self.block.stmts.append(stmt)
        # new top first — matches "top-of-stack last argument" convention
        self.stack.append(stmt.out_bot)
        self.stack.append(stmt.out_top)

    def _visit_pop(self, instr):
        top = self.stack.pop()
        self.block.stmts.append(stack_move.Pop(value=top))

    def _visit_fill(self, instr):
        locs = [self.stack.pop() for _ in range(instr.arity)]
        locs.reverse()
        self._check_type(locs, LocationAddressType)
        self.block.stmts.append(stack_move.Fill(locations=tuple(locs)))

    # … one handler per opcode …

def load_program(
    program: Program,
    kernel_name: str = "main",
) -> ir.Method:
    """Decode a bytecode Program into a stack_move ir.Method."""
```

`load_program` returns a `stack_move` method. Callers run the `lower_stack_move` rewrite and the rest of the pipeline explicitly.

A `DecodeError` exception carries the offending instruction index, opcode, and virtual-stack snapshot.

### 3. New statement in old `move` dialect

A multi-zone stateful measurement, alongside the existing `EndMeasure`:

```python
class Measure(StatefulStatement):
    traits = frozenset({lowering.FromPythonCall(), ConsumesState(True)})
    current_state: ir.SSAValue = info.argument(StateType)
    zones: tuple[ir.SSAValue, ...] = info.argument(type=ZoneAddressType)
    result: ir.ResultValue = info.result(MeasurementFutureType)
```

This is produced by `lower_stack_move` and consumed by `measure_lower`. `EndMeasure` stays unchanged.

### 4. `python/bloqade/lanes/rewrite/lower_stack_move.py`

**Mechanical per-statement rewrite.** Because `stack_move` is already SSA, the rewrite is a straightforward statement-by-statement translation into target dialects — no virtual stack to simulate, no SSA construction.

```python
class LowerStackMove:
    state: ir.SSAValue  # threaded StateType for stateful move ops
    # SSA-value map: stack_move SSA value → target-dialect SSA value
    mapping: dict[ir.SSAValue, ir.SSAValue]
    target_block: ir.Block

    def run(self, source_block: ir.Block) -> ir.Block:
        self.state = self._emit_initial_state()
        for stmt in source_block.stmts:
            self._rewrite(stmt)
        return self.target_block

    def _rewrite(self, stmt: ir.Statement) -> None:
        # dispatch by statement type
        ...
```

**Per-statement behavior (abridged):**

| `stack_move` statement | Target-dialect emission |
|---|---|
| `ConstFloat` / `ConstInt` | `py.constant.Constant` |
| `ConstLoc` / `ConstLane` / `ConstZone` | `move.Const*` with matching result |
| `Pop` | nothing (dropped; operand becomes dead or is DCE'd) |
| `Dup` | map result to same target SSA value as input (identity in target IR) |
| `Swap` | map results (reordering is purely a decoder concept; no target statement) |
| `InitialFill` / `Fill` / `Move` | `move.InitialFill` / `move.Fill` / `move.Move` with state threading |
| `LocalR` / `LocalRz` / `GlobalR` / `GlobalRz` / `CZ` | `move.LocalR` / `move.LocalRz` / `move.GlobalR` / `move.GlobalRz` / `move.CZ` with state threading |
| `Measure(*locs)` | dedup zones from `locs` via `ConstLoc` chase; synthesize `move.ConstZone` per distinct zone; emit `move.Measure(state, *zones)` |
| `AwaitMeasure(future)` | `move.AwaitMeasure(future)` |
| `NewArray` | `ilist.New` (1-D) or nested `ilist.New` (2-D) |
| `GetItem` | chained `py.indexing.GetItem` |
| `SetDetector(arr)` | `annotate.SetDetector(arr, empty_coords)` |
| `SetObservable(arr)` | `annotate.SetObservable(arr)` |
| `Return(val?)` | `func.Return(val?)` |
| `Halt` | `func.Return(None)` |

`Pop`/`Dup`/`Swap` all collapse during rewrite: `Pop` produces no target statement (its operand may become dead); `Dup` and `Swap` map their result SSA values back to the same target-dialect SSA values as their inputs. The linear-IR shape lives in `stack_move` only.

### 5. `AtomAnalysis` extensions

Abstract interpretation methods for `move.Measure`. The analysis tracks:

- The zone set measured at each `move.Measure` site.
- Whether the program contains **exactly one** `move.Measure` (program-wide, across all reachable control flow).

Consumed by `measure_lower`.

### 6. `python/bloqade/lanes/rewrite/measure_lower.py`

New `move → move` rewrite. Runs `AtomAnalysis`, then:

- Asserts every `move.Measure` covers exactly one zone — else descriptive error.
- Asserts the program contains exactly one final measurement — else descriptive error.
- Rewrites `move.Measure(state, zone)` → `move.EndMeasure(state)` with the zone promoted back into an attribute.

## Measure semantics

- **Bytecode `measure(arity)`** — pops `arity` location addresses.
- **Decoder** — pops `arity` SSA location values and emits `stack_move.Measure(*locs)` with those SSA operands.
- **`lower_stack_move`** — reads each location operand's defining `ConstLoc`, extracts the zone id, deduplicates, synthesizes one `move.ConstZone` per distinct zone, and emits a single `move.Measure(state, *zones)`.
- **`measure_lower`** — enforces single-zone-per-measure + single-final-measurement invariants and rewrites to `move.EndMeasure`.

## Error handling

- **Decoder (`DecodeError`)**: stack underflow, operand type mismatch (e.g. `fill` consumed a non-`LocationAddressType` value), non-empty virtual stack at `Return`/`Halt`, unknown opcode. Carries the offending instruction index, opcode, and a snapshot of the virtual stack at failure.
- **`lower_stack_move`**: should be infallible on well-typed `stack_move` IR; any failure indicates a decoder bug.
- **`measure_lower`**: descriptive errors for multi-zone measurements or multiple final measurements.

## Testing strategy

- **Decoder unit tests**, one per opcode. Build a minimal `Program`, decode, assert the resulting `stack_move` IR shape — operand SSA bindings, result types, stack ops in place.
- **Decoder error tests** for each `DecodeError` case.
- **`lower_stack_move` unit tests**, one per statement family. Small `stack_move` inputs; verify target-dialect output IR, state threading, `Pop`/`Dup`/`Swap` collapsing, zone grouping for `Measure`.
- **Analysis unit tests** for the new `AtomAnalysis` methods.
- **`measure_lower` unit tests** — valid and invalid cases.
- **End-to-end test** — a small `Program` → `load_program` → `lower_stack_move` → `measure_lower` → an existing downstream pass.

## File layout summary

```
python/bloqade/lanes/
├── bytecode/
│   └── lowering.py                      # NEW — BytecodeDecoder + load_program
├── dialects/
│   ├── move.py                          # EDIT — add Measure(*zones) stmt
│   └── stack_move.py                    # NEW — new dialect (all bytecode opcodes, SSA)
├── analysis/atom/
│   └── impl.py                          # EDIT — abstract interpretation for move.Measure
└── rewrite/
    ├── lower_stack_move.py              # NEW — per-statement mechanical rewrite
    └── measure_lower.py                 # NEW — move→move rewrite with analysis gate
```

## Prerequisites

- PyO3 accessors for `Instruction` operands (arity, address values, type tags, dimensions). Blocking follow-up.

## Planned follow-up: sub-dialect decomposition (the bytecode decoder framework)

The initial draft above treats `stack_move` as one monolithic dialect for prototyping speed. Once the mechanics are validated, the natural next step is to **group the instructions into sub-dialects** along the boundaries they already have, and then **decompose the decoder and `lower_stack_move` into composable per-sub-dialect chunks**. This is the point at which the implementation stops being a one-off and becomes the **bytecode decoder framework** the v1 prototype is paving the way for.

A provisional grouping:

| Sub-dialect | Instructions |
|---|---|
| `stack_move.stack` | `Pop`, `Dup`, `Swap` |
| `stack_move.constants` | `ConstFloat`, `ConstInt`, `ConstLoc`, `ConstLane`, `ConstZone` |
| `stack_move.atom` | `InitialFill`, `Fill`, `Move` |
| `stack_move.gates` | `LocalR`, `LocalRz`, `GlobalR`, `GlobalRz`, `CZ` |
| `stack_move.measure` | `Measure`, `AwaitMeasure` |
| `stack_move.array` | `NewArray`, `GetItem` |
| `stack_move.annotate` | `SetDetector`, `SetObservable` |
| `stack_move.control` | `Return`, `Halt` |

Framework goals:

- **Composable decoding.** Each sub-dialect owns a decoding handler for its opcodes; the top-level `BytecodeDecoder` is a dispatch that consults registered handlers. Adding a new instruction family (or swapping one out for a different target dialect) is a localized change.
- **Composable rewrites.** `lower_stack_move` decomposes into a chain of per-sub-dialect rewrite chunks, each of which knows how to consume its own statements and emit into a specific target dialect.
- **Independent testability.** Each sub-dialect + its handler + its rewrite chunk is a unit that can be tested in isolation.
- **Reuse beyond bytecode.** Sub-dialects like `stack_move.stack` or `stack_move.constants` are shape-agnostic; other stack-oriented frontends that share their abstractions can reuse them directly.

This is explicitly a **post-initial-draft** milestone.

## Known limitations

- **Straight-line programs only.** The decoder's virtual-SSA-stack is a single-basic-block algorithm and does not handle branching control flow. This matches today's Bloqade Lanes bytecode (no branches). When branching is introduced, the decoder will need a dedicated redesign. The recommended future direction (Wasm-style structured control flow + block-argument SSA per Approach 3 in the companion walkthrough) is a clean fit for Variant 2 as written: each basic block's SSA block-argument list is the "stack at entry", and the explicit `stack_move.Dup`/`Swap`/`Pop` statements operate on those arguments and on locally produced SSA values uniformly. See the companion walkthrough doc for the detailed reasoning.
