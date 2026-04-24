# `stack_move` Dialect — Design

**Date**: 2026-04-21 (updated 2026-04-24 to reflect implementation)
**Status**: Draft — reflects shipped implementation on `feat/stack-move-impl`
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
- A rewrite pass `stack_move2move` (`RewriteStackMoveToMove`) that does per-statement mechanical translation from `stack_move` into target dialects (`move`, `ilist`, `py.constant`, `py.indexing`, `annotate`, `func`), eliminating `Pop`/`Dup`/`Swap` along the way. Requires an `ArchSpec` (consumed by the `AwaitMeasure` lowering).
- A new multi-zone stateful `move.Measure` statement (target of the rewrite).
- Extensions to `AtomAnalysis`: the `MeasureFuture` lattice element carries a `measurement_count` ordinal populated from the running final-measurement count.
- A new `move → move` rewrite pass `measure_lower` that reads each `move.Measure`'s `future` SSA from an analysis `ForwardFrame` and lowers it into `move.EndMeasure` when the preconditions hold. Precondition failures cause the rewrite to give up silently (no exception) — validation is a dedicated-pass concern, not a rewrite-rule concern.

**Out of scope (follow-ups):**

- Exposing per-instruction operand accessors on the `Instruction` PyO3 binding. Prerequisite — tracked separately.
- File/bytes entry points. `Program.from_binary` and `Program.from_text` already exist.
- Changes to `move2squin` or any other downstream pass.
- The linear-type checker that would enforce atom non-cloning by catching disallowed `Dup` uses. Separate pass, separate spec.

## Architecture

```text
┌─────────────┐  load_program   ┌──────────────┐  stack_move2move    ┌──────────┐  measure_lower     ┌──────────┐
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

- `LocationAddressType`, `LaneAddressType`, `ZoneAddressType` — address wrappers in `bloqade.lanes.layout.encoding`.
- `MeasurementFutureType`, `MeasurementResultType`, `DetectorType`, `ObservableType` — from `bloqade.decoders.dialects.annotate.types`.
- `ArrayType` — parameterised as `ArrayType[ElemType, Dim0Type, Dim1Type]` where `Dim1Type == Literal(0)` means 1-D (by the bytecode convention).
- Existing kirin types are reused for `FloatType`, `IntType`.

**Constants** (attribute + SSA result; all marked `ir.Pure()` so CSE/DCE passes can collapse duplicates and drop dead constants after the rewrite):

```python
class ConstFloat(ir.Statement):
    traits = frozenset({lowering.FromPythonCall(), ir.Pure()})
    value: float = info.attribute()
    result: ir.ResultValue = info.result(FloatType)

class ConstInt(ir.Statement):
    traits = frozenset({lowering.FromPythonCall(), ir.Pure()})
    value: int = info.attribute()
    result: ir.ResultValue = info.result(IntType)

class ConstLoc(ir.Statement):
    traits = frozenset({lowering.FromPythonCall(), ir.Pure()})
    value: LocationAddress = info.attribute()
    result: ir.ResultValue = info.result(LocationAddressType)

class ConstLane(ir.Statement):
    traits = frozenset({lowering.FromPythonCall(), ir.Pure()})
    value: LaneAddress = info.attribute()
    result: ir.ResultValue = info.result(LaneAddressType)

class ConstZone(ir.Statement):
    traits = frozenset({lowering.FromPythonCall(), ir.Pure()})
    value: ZoneAddress = info.attribute()
    result: ir.ResultValue = info.result(ZoneAddressType)
```

**Stack manipulation** (explicit SSA inputs/outputs; also `ir.Pure()`):

```python
class Pop(ir.Statement):
    # consumes 1 SSA value, produces none
    traits = frozenset({lowering.FromPythonCall(), ir.Pure()})
    value: ir.SSAValue = info.argument()

class Dup(ir.Statement):
    # semantic identity — result ≡ value — but preserved as explicit op
    traits = frozenset({lowering.FromPythonCall(), ir.Pure()})
    value: ir.SSAValue = info.argument()
    result: ir.ResultValue = info.result()

class Swap(ir.Statement):
    # permutation — out_top ≡ in_bot, out_bot ≡ in_top
    traits = frozenset({lowering.FromPythonCall(), ir.Pure()})
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

**Gates** (argument names match the `move` dialect — `axis_angle` / `rotation_angle` rather than `phi` / `theta`):

```python
class LocalR(ir.Statement):
    axis_angle: ir.SSAValue = info.argument(type=FloatType)
    rotation_angle: ir.SSAValue = info.argument(type=FloatType)
    locations: tuple[ir.SSAValue, ...] = info.argument(type=LocationAddressType)

class LocalRz(ir.Statement):
    rotation_angle: ir.SSAValue = info.argument(type=FloatType)
    locations: tuple[ir.SSAValue, ...] = info.argument(type=LocationAddressType)

class GlobalR(ir.Statement):
    axis_angle: ir.SSAValue = info.argument(type=FloatType)
    rotation_angle: ir.SSAValue = info.argument(type=FloatType)

class GlobalRz(ir.Statement):
    rotation_angle: ir.SSAValue = info.argument(type=FloatType)

class CZ(ir.Statement):
    zone: ir.SSAValue = info.argument(type=ZoneAddressType)
```

**Measurement** (matches the Rust validator's `sim_measure`: pops `arity` zones and pushes `arity` futures):

```python
class Measure(ir.Statement):
    # arity zones popped; arity MeasurementFutureType results pushed —
    # one per zone. The per-zone futures are fanned to the single
    # move.Measure future in stack_move2move via replace_by.
    zones: tuple[ir.SSAValue, ...] = info.argument(type=ZoneAddressType)
    # results declared dynamically in __init__:
    #   result_types=(MeasurementFutureType,) * arity

class AwaitMeasure(ir.Statement):
    # Consumes the future (linear) and pushes a 1-D array of
    # measurement results. Element order is defined by the ArchSpec
    # (flattened zones × yield_zone_locations), documented on the
    # stack_move2move AwaitMeasure handler.
    traits = frozenset({lowering.FromPythonCall()})
    future: ir.SSAValue = info.argument(type=MeasurementFutureType)
    result: ir.ResultValue = info.result(
        ArrayType[MeasurementResultType, types.Any, types.Literal(0)]
    )
```

`Measure` takes **zones** directly as SSA operands (not locations — the bytecode `measure(arity)` opcode pops zone addresses). Zone deduplication happens in `stack_move2move` when lowering to the single-future `move.Measure`.

**Arrays and data:**

```python
class NewArray(ir.Statement):
    # Result type is the fully-parameterised ArrayType[ElemType,
    # Literal(dim0), Literal(dim1)] built from the type_tag byte (via
    # the TYPE_TAG table) and the two dimension attributes.
    type_tag: int = info.attribute()
    dim0: int = info.attribute()
    dim1: int = info.attribute()  # 0 when 1-D
    values: tuple[ir.SSAValue, ...]  # dim0 * max(dim1, 1) elements
    result: ir.ResultValue = info.result(ArrayType[ElemType, Dim0Type, Dim1Type])

class GetItem(ir.Statement):
    traits = frozenset({lowering.FromPythonCall(), ir.Pure()})
    array: ir.SSAValue = info.argument(type=ArrayType)
    indices: tuple[ir.SSAValue, ...] = info.argument(type=IntType)
    result: ir.ResultValue = info.result()  # element type

class SetDetector(ir.Statement):
    # 1-D array input (dim1=0 per the bytecode convention); validation
    # of the shape/element type is a type-inference concern, not a
    # decoder concern.
    array: ir.SSAValue = info.argument(
        type=ArrayType[MeasurementResultType, types.Any, types.Literal(0)]
    )
    result: ir.ResultValue = info.result(DetectorType)

class SetObservable(ir.Statement):
    array: ir.SSAValue = info.argument(
        type=ArrayType[MeasurementResultType, types.Any, types.Literal(0)]
    )
    result: ir.ResultValue = info.result(ObservableType)
```

**Control flow:**

```python
class Return(ir.Statement):
    # Required non-optional return operand — Return always pops one
    # stack value as the return value (matches the bytecode's
    # `return_` opcode). Halt is the no-return counterpart.
    traits = frozenset({lowering.FromPythonCall(), ir.IsTerminator()})
    value: ir.SSAValue = info.argument()

class Halt(ir.Statement):
    # No operand — lowers to a func.ConstantNone + func.Return pair.
    traits = frozenset({lowering.FromPythonCall(), ir.IsTerminator()})
```

### 2. `python/bloqade/lanes/bytecode/decode.py`

The decoder maintains a **virtual stack of SSA values** (a `StackMachineFrame`, mirroring Kirin's `lowering.Frame` idiom) while walking the bytecode. Each bytecode instruction is dispatched to a per-opcode handler that reads/mutates the virtual stack and emits a `stack_move` statement with the right SSA operands.

```python
@dataclass
class StackMachineFrame:
    current_region: ir.Region
    current_block: ir.Block
    stack: list[ir.SSAValue]

    def push(self, stmt: T) -> T:
        """Append a statement to the current block and push its
        results onto the virtual stack in reverse declaration order."""
        ...

    def pop_value(self) -> ir.SSAValue: ...
    def pop_n(self, n: int) -> list[ir.SSAValue]: ...


class BytecodeDecoder:
    frame: StackMachineFrame

    def decode(self, program: Program, kernel_name: str) -> ir.Method:
        for idx, instruction in enumerate(program.instructions):
            try:
                self._visit(instruction)
            except StackUnderflowError as e:
                raise DecodingError.from_context(idx, instruction, e)
        return self._finalize(kernel_name)

    def _visit_const_loc(self, idx, instr):
        self.frame.push(stack_move.ConstLoc(value=instr.location_address()))

    def _visit_fill(self, idx, instr):
        locs = self.frame.pop_n(instr.arity())
        self.frame.push(stack_move.Fill(locations=tuple(locs)))

    # … one handler per opcode …

def load_program(
    program: Program,
    kernel_name: str = "main",
) -> ir.Method:
    """Decode a bytecode Program into a stack_move ir.Method and run a
    TypeInfer pass to populate SSA types."""
```

`load_program` returns a `stack_move` method with concrete SSA types (the decoder runs `kirin.passes.typeinfer.TypeInfer` before returning). Callers run the `stack_move2move` rewrite and the rest of the pipeline explicitly.

A `DecodingError` exception wraps stack-underflow failures with the offending instruction index, opcode, and virtual-stack snapshot.

### 3. New statement in old `move` dialect

A multi-zone stateful measurement, alongside the existing `EndMeasure`:

```python
class Measure(StatefulStatement):
    # StatefulStatement inherits current_state: StateType input and a
    # result: StateType output, giving Measure two results overall:
    # the threaded state and the measurement future. zone_addresses
    # is a compile-time attribute tuple of plain Python ZoneAddress
    # values (not an SSA operand) — matching the shape of other move
    # statements such as CZ / EndMeasure.
    zone_addresses: tuple[ZoneAddress, ...] = info.attribute()
    future: ir.ResultValue = info.result(MeasurementFutureType)
```

This is produced by `stack_move2move` and consumed by `measure_lower`. `EndMeasure` stays unchanged.

### 4. `python/bloqade/lanes/rewrite/stack_move2move.py`

**Mechanical per-statement rewrite** (`RewriteStackMoveToMove`). Because `stack_move` is already SSA, the rewrite is a straightforward in-place statement-by-statement translation into target dialects — no separate target block, no full SSA reconstruction. The rewrite uses Kirin's `old_ssa.replace_by(new_ssa)` idiom to redirect SSA uses and collects source statements into a deletion list walked at the end.

```python
@dataclass
class RewriteStackMoveToMove(RewriteRule):
    # Required. Consumed by the AwaitMeasure lowering to iterate
    # arch_spec.yield_zone_locations(zone) for each zone on the
    # originating move.Measure.
    arch_spec: ArchSpec
    # stack_move SSA → raw Python attribute value (float, int,
    # LocationAddress, LaneAddress, ZoneAddress) — used to lift
    # SSA operands back to target-dialect attributes (addresses,
    # rotation angles).
    ssa_to_attr: dict[ir.SSAValue, Any] = field(default_factory=dict)
    # Current StateType SSA value threaded through move.* stateful ops.
    state: ir.SSAValue | None = None

    def rewrite_Block(self, node: ir.Block) -> RewriteResult:
        # 1. Insert move.Load at block start to source StateType.
        # 2. Walk stmts, dispatch to per-type handlers that emit target
        #    statements via insert_before and accumulate source stmts
        #    into a deletion list.
        # 3. Reverse-order deletion so consumers are removed before
        #    their producers.
        # 4. Insert the closing move.Store(state) just before the
        #    block's terminator — the state-boundary Store lives here
        #    (not in the Return/Halt handlers), so it's emitted
        #    regardless of which terminator (if any) is present.
        ...
```

**Per-statement behavior:**

| `stack_move` statement | Target emission |
|---|---|
| `ConstFloat` / `ConstInt` | `py.Constant` (value forwarded into `ssa_to_attr` for downstream attribute lifting) |
| `ConstLoc` / `ConstLane` / `ConstZone` | no target stmt — raw value recorded in `ssa_to_attr`; downstream `move.*` ops lift it back as an attribute |
| `Pop` | nothing (dropped; operand becomes dead or is DCE'd) |
| `Dup` | `stmt.result.replace_by(stmt.value)` — identity, no target stmt |
| `Swap` | `out_top.replace_by(in_bot)`, `out_bot.replace_by(in_top)` — permutation, no target stmt |
| `InitialFill` / `Fill` | `move.Fill` (both `stack_move.InitialFill` and `stack_move.Fill` lower to `move.Fill`; the distinction exists only at the bytecode/stack_move layer for validation) |
| `Move` | `move.Move` with lifted `lanes` attribute |
| `LocalR` / `LocalRz` / `GlobalR` / `GlobalRz` / `CZ` | `move.LocalR` / `move.LocalRz` / `move.GlobalR` / `move.GlobalRz` / `move.CZ` with state threading |
| `Measure(*zones)` | Lift each zone SSA to its `ZoneAddress` attribute, dedup by zone_id (preserving first-seen order); emit `move.Measure(state, zone_addresses=distinct_zones)`; fan each per-zone future result to the single `move.Measure.future` via `replace_by` |
| `AwaitMeasure(future)` | Chase `future.owner` to the originating `move.Measure` (whose `zone_addresses` attribute tells us which zones were measured); emit one `move.GetFutureResult(future, zone, loc)` per `(zone, loc)` from `arch_spec.yield_zone_locations(zone)`; bundle the results in an `ilist.New` and redirect the AwaitMeasure result |
| `NewArray` | `ilist.New` (1-D when `dim1 == 0`) or `dim0` inner `ilist.New`s wrapped in an outer `ilist.New` (2-D, matching `IList[IList[ElemType, Dim1], Dim0]`) |
| `GetItem` | chained `py.indexing.GetItem` (one per index; last result replaces the stack_move.GetItem result) |
| `SetDetector(arr)` | `annotate.SetDetector(arr, empty_coords)` |
| `SetObservable(arr)` | `annotate.SetObservable(arr)` |
| `Return(value)` | `func.Return(value)` — the closing `move.Store(state)` is emitted by `rewrite_Block`, not here |
| `Halt` | `func.ConstantNone` + `func.Return(none.result)` — the closing `move.Store(state)` is emitted by `rewrite_Block`, not here |

`Pop`/`Dup`/`Swap` all collapse during rewrite: `Pop` produces no target statement (its operand may become dead); `Dup` and `Swap` redirect their result SSA values back to their inputs via `replace_by`. The linear-IR shape lives in `stack_move` only.

### 5. `AtomAnalysis` extensions

Abstract interpretation methods for `move.Measure` and `move.EndMeasure`. The analysis tracks:

- The zone set and per-location qubit occupancy measured at each site, stored on the `MeasureFuture` lattice element's `results: dict[ZoneAddress, dict[LocationAddress, int]]` field (dict iteration order preserves zone insertion order).
- A `measurement_count: int` ordinal populated on every `MeasureFuture` the analysis produces, taken from the running `final_measurement_count` counter on the interpreter after it is incremented. For programs with a single final measurement, the one `MeasureFuture` has `measurement_count == 1`; programs with more than one measurement produce futures with counts 2, 3, …

`MeasureLower` consumes the `ForwardFrame` returned by the analysis — it does not read pass-object fields.

### 6. `python/bloqade/lanes/rewrite/measure_lower.py`

New `move → move` rewrite. Runs `AtomAnalysis`, stashes the returned `ForwardFrame`, then for each `move.Measure` statement looks up its `future` SSA in the frame, reads the associated `MeasureFuture` lattice element, and decides whether to rewrite:

- Rewrite proceeds iff the future exists in the frame, `measurement_count == 1`, **and** the future's `results` dict has exactly one key (exactly one zone observed).
- On rewrite: emit `move.EndMeasure(current_state, zone_addresses=tuple(results.keys()))` just before `move.Measure`; rewire state uses via `node.result.replace_by(node.current_state)` (forward the input state through any residual state consumers); rewire future uses via `node.future.replace_by(replacement.result)`; delete the original `move.Measure`.
- If any precondition fails the rewrite **gives up silently** — returns `RewriteResult(has_done_something=False)` without touching the IR. Validation is a dedicated-pass concern, not a rewrite-rule concern; `MeasureLower` does not raise.

## Measure semantics

- **Bytecode `measure(arity)`** — pops `arity` zone addresses and pushes `arity` measurement futures (one per zone), matching the Rust validator's `sim_measure`.
- **Decoder** — pops `arity` SSA zone values and emits `stack_move.Measure(zones=tuple(zones))` with dynamic `result_types=(MeasurementFutureType,) * arity`.
- **`stack_move2move`** — lifts each zone SSA operand to its `ZoneAddress` attribute via `ssa_to_attr`, dedups by `zone_id` (preserving first-seen order), and emits a single `move.Measure(state, zone_addresses=distinct_zones)`. Every per-zone future from `stack_move.Measure` is fanned to the single `move.Measure.future` result via `replace_by`.
- **`measure_lower`** — rewrites `move.Measure` → `move.EndMeasure` when the analysis frame confirms single-zone + single-final-measurement; otherwise silently defers.

## Error handling

- **Decoder (`DecodingError`)**: wraps stack underflow failures with the offending instruction index, opcode, and a snapshot of the virtual stack at failure.
- **`stack_move2move`**: should be infallible on well-typed `stack_move` IR; any failure indicates a decoder bug. Per-statement handlers that can't produce a valid lowering (e.g. an AwaitMeasure whose future doesn't trace back to a `move.Measure`) return without modifying the IR.
- **`measure_lower`**: never raises. Validation of "exactly one final measurement" and "exactly one zone per measurement" is the job of a dedicated validation pass that runs between dialect transformations; the rewrite just defers when preconditions fail.

## Testing strategy

- **Decoder unit tests**, one per opcode. Build a minimal `Program`, decode, assert the resulting `stack_move` IR shape — operand SSA bindings, result types, stack ops in place.
- **Decoder error tests** for each `DecodeError` case.
- **`stack_move2move` unit tests**, one per statement family. Small `stack_move` inputs; verify target-dialect output IR, state threading, `Pop`/`Dup`/`Swap` collapsing, zone grouping for `Measure`.
- **Analysis unit tests** for the new `AtomAnalysis` methods.
- **`measure_lower` unit tests** — valid and invalid cases.
- **End-to-end test** — a small `Program` → `load_program` → `stack_move2move` → `measure_lower` → an existing downstream pass.

## File layout summary

```
python/bloqade/lanes/
├── bytecode/
│   └── decode.py                        # NEW — BytecodeDecoder + StackMachineFrame + load_program
├── dialects/
│   ├── move.py                          # EDIT — add multi-zone Measure stmt (StatefulStatement)
│   └── stack_move.py                    # NEW — new dialect (all bytecode opcodes, SSA)
├── analysis/atom/
│   ├── lattice.py                       # EDIT — MeasureFuture gains measurement_count ordinal
│   └── impl.py                          # EDIT — abstract interpretation for move.Measure / EndMeasure
└── rewrite/
    ├── stack_move2move.py               # NEW — RewriteStackMoveToMove (per-statement in-place rewrite; requires ArchSpec)
    └── measure_lower.py                 # NEW — move→move rewrite reading an analysis ForwardFrame
```

## Prerequisites

- PyO3 accessors for `Instruction` operands (arity, address values, type tags, dimensions). Blocking follow-up.

## Planned follow-up: sub-dialect decomposition (the bytecode decoder framework)

The initial draft above treats `stack_move` as one monolithic dialect for prototyping speed. Once the mechanics are validated, the natural next step is to **group the instructions into sub-dialects** along the boundaries they already have, and then **decompose the decoder and `stack_move2move` into composable per-sub-dialect chunks**. This is the point at which the implementation stops being a one-off and becomes the **bytecode decoder framework** the v1 prototype is paving the way for.

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
- **Composable rewrites.** `stack_move2move` decomposes into a chain of per-sub-dialect rewrite chunks, each of which knows how to consume its own statements and emit into a specific target dialect.
- **Independent testability.** Each sub-dialect + its handler + its rewrite chunk is a unit that can be tested in isolation.
- **Reuse beyond bytecode.** Sub-dialects like `stack_move.stack` or `stack_move.constants` are shape-agnostic; other stack-oriented frontends that share their abstractions can reuse them directly.

This is explicitly a **post-initial-draft** milestone.

## Known limitations

- **Straight-line programs only.** The decoder's virtual-SSA-stack is a single-basic-block algorithm and does not handle branching control flow. This matches today's Bloqade Lanes bytecode (no branches). When branching is introduced, the decoder will need a dedicated redesign. The recommended future direction (Wasm-style structured control flow + block-argument SSA per Approach 3 in the companion walkthrough) is a clean fit for Variant 2 as written: each basic block's SSA block-argument list is the "stack at entry", and the explicit `stack_move.Dup`/`Swap`/`Pop` statements operate on those arguments and on locally produced SSA values uniformly. See the companion walkthrough doc for the detailed reasoning.
