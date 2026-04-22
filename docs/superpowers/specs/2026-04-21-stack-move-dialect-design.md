# `stack_move` Dialect — Design

**Date**: 2026-04-21
**Status**: Draft
**Tracking issue**: #248 (scope revised — see below)
**Companion doc**: [2026-04-21-bytecode-to-ssa-lowering.md](./2026-04-21-bytecode-to-ssa-lowering.md)

## Goal

Accept Rust-backed `bloqade.lanes.bytecode.Program` objects as a first-class input to the Bloqade Lanes compiler pipeline. A user loads a bytecode program (e.g. via `Program.from_binary`) and the decoder turns it into a Kirin `ir.Method` that the existing downstream passes (state, transversal, move2squin, validation, visualization, metrics) consume unchanged.

This is an additive change. No existing dialect, pass, or public API is broken.

## Design stance

For this first pass, the `stack_move` dialect is a **faithful 1:1 image of the bytecode**: every bytecode instruction becomes a `stack_move` statement, and statements carry their operand data as attributes (no SSA operands, no SSA results). The IR is a linear sequence of stack operations that mirrors the bytecode exactly.

This pushes all interesting lowering work into a single **rewrite pass** (`lower_stack_move`) that walks the `stack_move` sequence, maintains a virtual SSA stack, and emits well-typed SSA into the target dialects (`move`, `ilist`, `py.constant`, `py.indexing`, `annotate`, `func`). The decoder is trivially mechanical; the rewrite is where the real work lives.

The goal is **prototyping speed and clarity of responsibility**: the decoder becomes a pure syntactic mapping (byte-in → stack_move IR out) that can be verified by inspection; all stack-to-SSA reconstruction is concentrated in one reviewable place.

## Scope

**In scope (v1):**

- A new Kirin dialect `stack_move` with one statement per bytecode instruction, including the pure stack-manipulation ops (`Pop`/`Dup`/`Swap`). Statements have no SSA operands — operand data lives in attributes.
- A bespoke decoder that translates a `Program` instruction-by-instruction into a `stack_move`-dialect `ir.Method`. Purely syntactic; no stack simulation.
- A rewrite pass `lower_stack_move` that walks the `stack_move` IR with a virtual SSA stack and emits into `move`, `ilist`, `py.constant`, `py.indexing`, `annotate`, and `func`.
- A new multi-zone `move.Measure` statement (target of the rewrite).
- Extensions to `AtomAnalysis` that track measurement zone sets and whether the program performs a single final measurement.
- A new `move → move` rewrite pass `measure_lower` that uses the extended analysis to validate preconditions and lower `move.Measure` into the existing `move.EndMeasure`.

**Out of scope (follow-ups):**

- Exposing per-instruction operand accessors on the `Instruction` PyO3 binding. Prerequisite — tracked separately.
- File/bytes entry points. `Program.from_binary` and `Program.from_text` already exist.
- Changes to `move2squin` or any other downstream pass.

## Architecture

```text
┌─────────────┐  load_program  ┌─────────────┐  lower_stack_move    ┌──────────┐  measure_lower     ┌──────────┐
│  Program    │───────────────▶│  stack_move │─────────────────────▶│ multi-   │───────────────────▶│  move    │
│  (Rust)     │ (syntactic,    │  ir.Method  │  (virtual SSA stack, │ dialect  │  (analysis +       │  (old,   │
└─────────────┘  no stack sim) │  (linear    │   emits into target  │ IR (move,│   rewrite pass)    │  EndMea-) │
                                │   sequence) │   dialects)          │ ilist,   │                    └──────────┘
                                └─────────────┘                      │ etc.)    │                         │
                                                                     └──────────┘                         ▼
                                                                                                existing downstream pipeline
                                                                                                (transversal, move2squin, …)
```

## New code

### 1. `python/bloqade/lanes/dialects/stack_move.py`

New Kirin dialect. **One statement per bytecode instruction.** Statements have **no SSA operands and no SSA results** — they carry operand data as attributes only. The IR is a linear sequence of statements that mirrors the bytecode.

**Constants** — attribute-only:

```python
class ConstFloat(ir.Statement):
    value: float = info.attribute()

class ConstInt(ir.Statement):
    value: int = info.attribute()

class ConstLoc(ir.Statement):
    value: LocationAddress = info.attribute()  # Rust-backed

class ConstLane(ir.Statement):
    value: LaneAddress = info.attribute()  # Rust-backed

class ConstZone(ir.Statement):
    value: ZoneAddress = info.attribute()  # Rust-backed
```

**Stack manipulation** — pure bytecode markers:

```python
class Pop(ir.Statement):
    pass

class Dup(ir.Statement):
    pass

class Swap(ir.Statement):
    pass
```

**Atom operations** — arity as attribute:

```python
class InitialFill(ir.Statement):
    arity: int = info.attribute()

class Fill(ir.Statement):
    arity: int = info.attribute()

class Move(ir.Statement):
    arity: int = info.attribute()
```

**Gates:**

```python
class LocalR(ir.Statement):
    arity: int = info.attribute()

class LocalRz(ir.Statement):
    arity: int = info.attribute()

class GlobalR(ir.Statement):
    pass

class GlobalRz(ir.Statement):
    pass

class CZ(ir.Statement):
    pass
```

**Measurement:**

```python
class Measure(ir.Statement):
    arity: int = info.attribute()

class AwaitMeasure(ir.Statement):
    pass
```

**Arrays and data:**

```python
class NewArray(ir.Statement):
    type_tag: int = info.attribute()
    dim0: int = info.attribute()
    dim1: int = info.attribute()  # 0 when 1-D

class GetItem(ir.Statement):
    ndims: int = info.attribute()

class SetDetector(ir.Statement):
    pass

class SetObservable(ir.Statement):
    pass
```

**Control flow:**

```python
class Return(ir.Statement):
    pass

class Halt(ir.Statement):
    pass
```

That's the whole dialect. Every bytecode instruction in the [`Instruction` PyO3 stub](../../../python/bloqade/lanes/bytecode/_native.pyi) has exactly one corresponding `stack_move` statement.

### 2. `python/bloqade/lanes/bytecode/lowering.py`

**Trivially syntactic decoder.** No stack simulation, no SSA construction. One case per opcode that constructs the matching `stack_move` statement with its attributes populated from the instruction, then appends it to the method's single block.

```python
class BytecodeDecoder:
    """Syntactic decoder: bytecode Program → stack_move ir.Method.

    One-pass, one-statement-per-instruction. No virtual stack, no SSA
    operand resolution — all of that is the lower_stack_move rewrite's job.
    """

    block: ir.Block

    def decode(self, program: Program, kernel_name: str) -> ir.Method:
        for instruction in program.instructions:
            self.block.stmts.append(self._statement_for(instruction))
        return self._finalize(kernel_name)

    def _statement_for(self, instruction: Instruction) -> ir.Statement:
        # one case per opcode — each reads the instruction's attributes and
        # constructs the matching stack_move statement
        ...

def load_program(
    program: Program,
    kernel_name: str = "main",
) -> ir.Method:
    """Decode a bytecode Program into a stack_move ir.Method."""
```

`load_program` returns a `stack_move` method that is a linear sequence of statements. Callers run the `lower_stack_move` rewrite and the rest of the pipeline explicitly.

Error handling at the decoder level is minimal: the only failure mode is an unknown opcode, which shouldn't occur given the `Instruction` type is a closed enum from Rust. A `DecodeError` exception is raised if it does.

### 3. New statement in old `move` dialect

A multi-zone stateful measurement, alongside the existing `EndMeasure`:

```python
class Measure(StatefulStatement):
    traits = frozenset({lowering.FromPythonCall(), ConsumesState(True)})
    current_state: ir.SSAValue = info.argument(StateType)
    zones: tuple[ir.SSAValue, ...] = info.argument(type=ZoneAddressType)
    result: ir.ResultValue = info.result(MeasurementFutureType)
```

This is produced by the `lower_stack_move` rewrite and consumed by `measure_lower`. `EndMeasure` stays unchanged.

### 4. `python/bloqade/lanes/rewrite/lower_stack_move.py`

**This is where the real work happens.** The rewrite walks the `stack_move` IR in order, maintains a virtual stack of SSA values, and emits target-dialect statements (`move`, `ilist`, `py.constant`, `py.indexing`, `annotate`, `func`) with proper SSA operands.

```python
class LowerStackMove:
    stack: list[ir.SSAValue]
    target_block: ir.Block
    state: ir.SSAValue  # threaded StateType value for stateful ops

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

| `stack_move` statement | Virtual-stack action | Emit into target dialect |
|---|---|---|
| `ConstFloat(v)` | push result | `py.constant.Constant(v)` |
| `ConstInt(v)` | push result | `py.constant.Constant(v)` |
| `ConstLoc(v)` / `ConstLane(v)` / `ConstZone(v)` | push result | `move.Const*(value=v)` with SSA result |
| `Pop` | pop 1 | nothing |
| `Dup` | peek, push again | nothing |
| `Swap` | swap top two | nothing |
| `InitialFill(n)` | pop n LocAddr SSA values | `move.InitialFill(state, *locs)`; update `state` |
| `Fill(n)` | pop n LocAddr SSA values | `move.Fill(state, *locs)`; update `state` |
| `Move(n)` | pop n LaneAddr SSA values | `move.Move(state, *lanes)`; update `state` |
| `LocalR(n)` | pop phi, theta, n LocAddr | `move.LocalR(state, phi, theta, *locs)` |
| `LocalRz(n)` | pop theta, n LocAddr | `move.LocalRz(state, theta, *locs)` |
| `GlobalR` | pop phi, theta | `move.GlobalR(state, phi, theta)` |
| `GlobalRz` | pop theta | `move.GlobalRz(state, theta)` |
| `CZ` | pop ZoneAddr | `move.CZ(state, zone)` |
| `Measure(n)` | pop n LocAddr, dedup zones, synth ConstZones, push future | `move.Measure(state, *zones)` |
| `AwaitMeasure` | pop future, push bitstring | `move.AwaitMeasure(future)` |
| `NewArray(type_tag, d0, d1)` | push result | `ilist.New(...)` — 1-D or nested for 2-D |
| `GetItem(ndims)` | pop ndims indices + array, push element | chained `py.indexing.GetItem` |
| `SetDetector` | pop array | `annotate.SetDetector(array, empty_coords)` |
| `SetObservable` | pop array | `annotate.SetObservable(array)` |
| `Return` | — | `func.Return(state)` |
| `Halt` | — | `func.Return(None)` |

State threading (the `StateType` SSA chain through stateful `move` statements) happens inline during this rewrite — no separate infrastructure call. The virtual stack plus state-threading logic are colocated in one pass.

The rewrite is also where all the stack-discipline errors surface (stack underflow, type mismatches, non-empty stack at `Return`/`Halt`). See §"Error handling" below.

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

- **Bytecode `measure(arity)`** — pops `arity` location addresses. In the `stack_move` IR this is just `stack_move.Measure(arity=n)` — no SSA, no zone grouping yet.
- **`lower_stack_move`** pops `arity` location SSA values from the virtual stack, reads each one's zone via its defining `move.ConstLoc`, dedupes, synthesizes a `move.ConstZone` per distinct zone, and emits a single `move.Measure(state, *zones)`.
- **`measure_lower`** enforces that each `move.Measure` covers exactly one zone and that the program contains exactly one final measurement, then rewrites to `move.EndMeasure(state)`.

The zone-uniqueness invariant lives in `measure_lower`, not in `lower_stack_move` or the decoder — diagnostic-quality error messages concentrate in one pass.

## Error handling

- **Decoder (`DecodeError`)**: essentially never fires — the only case is an unknown opcode, which shouldn't be possible given the Rust-backed closed-enum `Instruction` type.
- **`lower_stack_move` (`LowerError`)**: stack underflow, operand type mismatch, non-empty virtual stack at `Return`/`Halt`. Carries the offending `stack_move` statement, its index within the block, and a snapshot of the virtual stack at failure.
- **`measure_lower`**: descriptive errors for multi-zone measurements or multiple final measurements.

## Testing strategy

- **Decoder unit tests**, one per opcode. Build a minimal `Program` containing the target opcode, decode, assert the resulting `stack_move` IR is a single-statement sequence with the expected attributes.
- **`lower_stack_move` unit tests**, one per statement family. Input `stack_move` IR sequences, output target-dialect IR; verify SSA shape, state threading, and attribute recovery.
- **`lower_stack_move` error tests** for each `LowerError` case (stack underflow, type mismatch, dangling stack at `Return`).
- **Analysis unit tests** for the new `AtomAnalysis` methods.
- **`measure_lower` unit tests** — valid (single zone, single final measurement) and invalid (multi-zone, multiple measurements) cases.
- **End-to-end test** — a small `Program` → `load_program` → `lower_stack_move` → `measure_lower` → an existing downstream pass (at minimum `transversal`, ideally `move2squin`).

## File layout summary

```
python/bloqade/lanes/
├── bytecode/
│   └── lowering.py                      # NEW — BytecodeDecoder + load_program
├── dialects/
│   ├── move.py                          # EDIT — add Measure(*zones) stmt
│   └── stack_move.py                    # NEW — new dialect (all bytecode opcodes)
├── analysis/atom/
│   └── impl.py                          # EDIT — abstract interpretation for move.Measure
└── rewrite/
    ├── lower_stack_move.py              # NEW — virtual-SSA-stack lowering to multi-dialect IR
    └── measure_lower.py                 # NEW — move→move rewrite with analysis gate
```

## Prerequisites

- PyO3 accessors for `Instruction` operands (arity, address values, type tags, dimensions). Blocking follow-up.

## Planned follow-up: sub-dialect decomposition

The initial draft above treats `stack_move` as one monolithic dialect for prototyping speed. Once the mechanics are validated, the natural next step is to **group the instructions into sub-dialects** along the boundaries they already have, and then **decompose `lower_stack_move` into composable per-sub-dialect rewrite chunks**. A provisional grouping:

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

Decomposition goals:

- **Composable decoding.** Each sub-dialect owns a decoding handler for its opcodes; the top-level `BytecodeDecoder` is a dispatch that consults registered handlers. Adding a new instruction family (or swapping one out for a different target dialect) is a localized change.
- **Composable rewrites.** `lower_stack_move` decomposes into a chain of per-sub-dialect rewrite passes that each know how to consume their own statements and emit target-dialect IR. The virtual-SSA-stack state is the shared context threaded through the chain.
- **Independent testability.** Each sub-dialect + its rewrite chunk can be unit-tested in isolation.
- **Reuse beyond bytecode.** Sub-dialects like `stack_move.constants` or `stack_move.stack` could be reused by other stack-oriented frontends if they arise.

This is explicitly a **post-initial-draft** plan. The v1 implementation lands as one dialect and one rewrite to keep the prototype small and reviewable; the decomposition lands as a follow-up once the boundaries have been validated in practice.

## Known limitations

- **Straight-line programs only.** The virtual-SSA-stack technique in `lower_stack_move` is a single-basic-block algorithm and does not handle branching control flow. This matches today's Bloqade Lanes bytecode (no branches). When branching is introduced, the rewrite will need a dedicated redesign — reconciling stack state across control-flow joins requires per-branch snapshots and block-argument insertion. See the companion walkthrough doc for the detailed reasoning and recommended future direction (Wasm-style structured control flow + block-argument SSA).
