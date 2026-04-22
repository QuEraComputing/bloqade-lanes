# `stack_move` Dialect — Design

**Date**: 2026-04-21
**Status**: Draft
**Tracking issue**: #248 (scope revised — see below)
**Companion doc**: [2026-04-21-bytecode-to-ssa-lowering.md](./2026-04-21-bytecode-to-ssa-lowering.md)

## Goal

Accept Rust-backed `bloqade.lanes.bytecode.Program` objects as a first-class input to the Bloqade Lanes compiler pipeline. A user loads a bytecode program (e.g. via `Program.from_binary`) and the decoder turns it into a Kirin `ir.Method` that the existing downstream passes (state, transversal, move2squin, validation, visualization, metrics) consume unchanged.

This is an additive change. No existing dialect, pass, or public API is broken.

## Scope

**In scope (v1):**

- A new Kirin dialect `stack_move` whose statements mirror the execution semantics of the bytecode (addresses are SSA values, no state threading).
- A Kirin `LoweringABC[Program]` implementation that decodes a `Program` instruction-by-instruction into a `stack_move`-dialect `ir.Method`.
- A rewrite pass `stack_move → move` that mechanically translates into the old `move` dialect and adds state threading via existing infrastructure.
- A new multi-zone `move.Measure` statement.
- Extensions to `AtomAnalysis` that track measurement zone sets and whether the program performs a single final measurement.
- A new `move → move` rewrite pass that uses the extended analysis to validate preconditions and lower `move.Measure` into the existing `move.EndMeasure`.

**Out of scope (follow-ups):**

- Exposing per-instruction operand accessors on the `Instruction` PyO3 binding. This is a prerequisite (tracked separately) — the decoder needs to read instruction operands, which the current bindings do not expose.
- File/bytes entry points. `Program.from_binary` and `Program.from_text` already exist; callers can use them.
- Changes to `move2squin` or any other downstream pass. The new dialect reaches them through the `measure_lower` rewrite as ordinary old-dialect IR.

## Architecture

```text
┌─────────────┐  load_program      ┌─────────────┐  stack_move2move    ┌──────────┐  measure_lower    ┌──────────┐
│  Program    │───────────────────▶│  stack_move │────────────────────▶│  move    │──────────────────▶│  move    │
│  (Rust)     │  Kirin Lowering    │  ir.Method  │  rewrite pass       │  (new    │  analysis +        │  (old,   │
└─────────────┘                    └─────────────┘                     │ .Measure)│  rewrite pass      │  EndMea-) │
                                                                       └──────────┘                    └──────────┘
                                                                                                            │
                                                                                                            ▼
                                                                                              existing downstream pipeline
                                                                                              (transversal, move2squin, …)
```

## New code

### 1. `python/bloqade/lanes/dialects/stack_move.py`

New Kirin dialect. **No state threading.** Addresses are SSA values produced by explicit `Const*` statements.

**Types:**

- `LocationAddressType` — SSA type for `LocationAddress` values
- `LaneAddressType` — SSA type for `LaneAddress` values
- `ZoneAddressType` — SSA type for `ZoneAddress` values
- `MeasurementFutureType` — SSA type for pending measurement results

**Constant statements** (wrap Rust address objects as attributes, produce typed SSA):

```python
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

**Atom operations** (no state — threading is added during rewrite):

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

**Measurement** (takes zones, not locations — see "measure semantics" below):

```python
class Measure(ir.Statement):
    zones: tuple[ir.SSAValue, ...] = info.argument(type=ZoneAddressType)
    result: ir.ResultValue = info.result(MeasurementFutureType)

class AwaitMeasure(ir.Statement):
    future: ir.SSAValue = info.argument(type=MeasurementFutureType)
    result: ir.ResultValue = info.result(BitstringType)
```

**Control flow:** reuse `func.Return`. Both `return_` and `halt` lower to `func.Return`; `halt` returns `None`.

**Arrays / detectors / observables:** these are *not* `stack_move` statements. The decoder emits directly into existing dialects:

- `new_array` → `kirin.dialects.ilist.New` (1D) or nested `ilist.New` (2D)
- `get_item(ndims)` → chained `kirin.dialects.py.indexing.GetItem`
- `set_observable()` → `bloqade.decoders.dialects.annotate.SetObservable(measurements)`
- `set_detector()` → `bloqade.decoders.dialects.annotate.SetDetector(measurements, empty_coords)` — coordinates default to empty (they're visualization-only metadata, out of scope for decoded bytecode)

### 2. `python/bloqade/lanes/bytecode/lowering.py`

Kirin `LoweringABC[Program]` specialization. See the companion doc [2026-04-21-bytecode-to-ssa-lowering.md](./2026-04-21-bytecode-to-ssa-lowering.md) for the full walkthrough.

```python
class StackMoveLowering(lowering.LoweringABC[Program]):
    def run(self, stmt: Program, state: lowering.State) -> ir.Method: ...
    def visit(self, state: lowering.State, instruction: Instruction) -> lowering.Result: ...
    # per-opcode visitors: visit_const_loc, visit_fill, visit_move, visit_dup, …

def load_program(
    program: Program,
    kernel_name: str = "main",
    dialects: ir.DialectGroup = ...,  # stack_move + ilist + py.indexing + py.constant + annotate + func
) -> ir.Method:
    """Decode a bytecode Program into a stack_move ir.Method."""
```

`load_program` returns a `stack_move` method. Callers run the `stack_move → move` rewrite and the rest of the pipeline explicitly.

### 3. New statement in old `move` dialect

A multi-zone stateful measurement, alongside the existing `EndMeasure`:

```python
class Measure(StatefulStatement):
    traits = frozenset({lowering.FromPythonCall(), ConsumesState(True)})
    current_state: ir.SSAValue = info.argument(StateType)
    zones: tuple[ir.SSAValue, ...] = info.argument(type=ZoneAddressType)
    result: ir.ResultValue = info.result(MeasurementFutureType)
```

This is produced by the `stack_move → move` rewrite and consumed by the `measure_lower` rewrite. `EndMeasure` stays unchanged.

### 4. `python/bloqade/lanes/rewrite/stack_move2move.py`

Mechanical rewrite. Operates on a `stack_move` method and produces a `move` method.

1. For each `stack_move` statement that consumes address SSA operands, walk back to the defining `ConstLoc`/`ConstLane`/`ConstZone` and extract the attribute value. Emit the corresponding old-dialect statement with those values as attributes.
2. `stack_move.Measure(*zones) → move.Measure(state, *zones)` — 1:1, gains state-threading.
3. Insert state threading (`Load`/`Store` and the `StateType` chain through stateful statements) using existing infrastructure.
4. Drop dead `Const*` statements.
5. Pass through `ilist`, `py.indexing`, `py.constant`, `annotate`, and `func` statements unchanged.

### 5. `AtomAnalysis` extensions

Add abstract interpretation methods for `move.Measure`. The analysis tracks, at each program point:

- The zone set measured at each `move.Measure` site.
- Whether the program contains **exactly one** `move.Measure` (program-wide, across all reachable control flow).

These facts are consumed by `measure_lower`.

### 6. `python/bloqade/lanes/rewrite/measure_lower.py`

New `move → move` rewrite. Runs `AtomAnalysis` and:

- Asserts every `move.Measure` covers exactly one zone. Otherwise raises with a descriptive error listing the offending sites.
- Asserts the program contains exactly one final measurement. Otherwise raises similarly.
- Rewrites `move.Measure(state, zone)` → `move.EndMeasure(state)` with the zone promoted back into an attribute.

## Measure semantics

Two different measure shapes exist:

- **Bytecode `measure(arity)`** — pops `arity` location addresses.
- **`stack_move.Measure(*zones)` (this design)** — takes one or more zone SSA values.
- **`move.Measure(state, *zones)` (this design)** — same as `stack_move.Measure`, plus state threading.
- **`move.EndMeasure(state)` (existing, unchanged)** — implicit single-zone attribute form.

For each bytecode `measure(arity)` instruction the decoder pops `arity` location SSA values, walks each back to its defining `ConstLoc` to read the `zone_id` field of the underlying `LocationAddress`, deduplicates, synthesizes a `ConstZone` per distinct zone, and emits a single `stack_move.Measure` that takes every distinct zone as an argument. Multi-zone `stack_move.Measure` is therefore legal IR out of the decoder.

The zone-uniqueness and single-final-measurement invariants required by the existing `EndMeasure` pipeline are enforced downstream by `measure_lower`, not by the decoder or the `stack_move → move` rewrite. This keeps the decoder and the mechanical rewrite free of semantic decomposition logic and concentrates the diagnostic-quality invariant checks in one place.

## Error handling

- **Decoder (`BuildError`)**: unsupported opcodes (none in v1 scope — all opcodes mapped), stack underflow, operand type mismatch (e.g. `fill` consumed a non-`LocationAddressType` value), non-empty virtual stack at `return`/`halt`.
- **`stack_move → move` rewrite**: should be infallible on well-typed `stack_move` IR; any failure indicates a bug in the decoder.
- **`measure_lower`**: descriptive errors for multi-zone measurements or multiple final measurements.

## Testing strategy

- **Decoder unit tests**, one per opcode. Build a minimal `Program` containing the target opcode plus whatever setup it needs, decode, assert the resulting `stack_move` IR shape.
- **Decoder error tests** for each `BuildError` case (stack underflow, type mismatch, unsupported opcode, dangling stack at return).
- **Rewrite unit tests** for `stack_move2move` — one per statement family, verifying attribute values recovered from `Const*` defining ops and state threading inserted correctly.
- **Analysis unit tests** for the new `AtomAnalysis` methods — verifying zone-set computation and single-measurement detection.
- **`measure_lower` unit tests** — valid (single zone, single final measurement) and invalid (multi-zone, multiple measurements) cases.
- **End-to-end test** — a small but non-trivial `Program` → `load_program` → `stack_move2move` → `measure_lower` → an existing downstream pass (at minimum `transversal`, ideally `move2squin` to exercise the full pipeline).

## File layout summary

```
python/bloqade/lanes/
├── bytecode/
│   └── lowering.py                      # NEW — StackMoveLowering + load_program
├── dialects/
│   ├── move.py                          # EDIT — add Measure(*zones) stmt
│   └── stack_move.py                    # NEW — new dialect
├── analysis/atom/
│   └── impl.py                          # EDIT — abstract interpretation for move.Measure
└── rewrite/
    ├── stack_move2move.py               # NEW — mechanical rewrite
    └── measure_lower.py                 # NEW — move→move rewrite with analysis gate
```

## Prerequisites

- PyO3 accessors for `Instruction` operands (arity, address values, type tags, dimensions). This is the blocking follow-up the decoder needs. Tracked separately.
