# `stack_move` Dialect Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Rust-backed `Program` bytecode as a first-class input to the Bloqade Lanes compiler pipeline via a new `stack_move` dialect, a bespoke decoder, and a rewrite pipeline into the existing `move` dialect.

**Architecture:** Bytecode `Program` → `BytecodeDecoder` (linear-IR SSA, virtual stack) → `stack_move` `ir.Method` → `lower_stack_move` rewrite → `move` IR (with new multi-zone `move.Measure`) → `measure_lower` rewrite (validates invariants via extended `AtomAnalysis`) → `move.EndMeasure` / existing downstream pipeline.

**Tech Stack:** Kirin IR framework (`kirin-toolchain`), Rust-backed PyO3 bytecode bindings (`bloqade.lanes.bytecode._native`), existing `move` dialect and `AtomAnalysis`.

**Reference docs:**
- Spec: `docs/superpowers/specs/2026-04-21-stack-move-dialect-design.md`
- Walkthrough: `docs/superpowers/specs/2026-04-21-bytecode-to-ssa-lowering.md`

---

## Prerequisites (blocking for Phase B only)

The decoder needs per-opcode operand accessors on the `Instruction` PyO3 binding. Currently only `Instruction.opcode: int` is exposed. Before Phase B (decoder) can be implemented, the Rust side needs accessors along the lines of:

- `Instruction.op_name() -> str` — opcode-name dispatch string (or equivalent enum)
- `Instruction.float_value() -> float` — valid on `const_float`
- `Instruction.int_value() -> int` — valid on `const_int`
- `Instruction.location_address() -> LocationAddress` — valid on `const_loc`
- `Instruction.lane_address() -> LaneAddress` — valid on `const_lane`
- `Instruction.zone_address() -> ZoneAddress` — valid on `const_zone`
- `Instruction.arity() -> int` — valid on `initial_fill`, `fill`, `move_`, `local_r`, `local_rz`, `measure`
- `Instruction.type_tag() -> int`, `.dim0() -> int`, `.dim1() -> int` — valid on `new_array`
- `Instruction.ndims() -> int` — valid on `get_item`

This is tracked separately. **Phases A, C, D, E, F, G do not depend on this** — only Phase B (decoder) does. If accessors aren't ready when Phase B is reached, those tasks are blocked pending the accessor work.

---

## File Structure

**New files:**

| File | Responsibility |
|---|---|
| `python/bloqade/lanes/dialects/stack_move.py` | The `stack_move` dialect: types + one statement per bytecode opcode with explicit SSA operands/results (Variant 2 / linear-IR style). |
| `python/bloqade/lanes/bytecode/lowering.py` | `BytecodeDecoder` class + `load_program()` entry point + `DecodeError` exception. |
| `python/bloqade/lanes/rewrite/lower_stack_move.py` | `LowerStackMove` rewrite: mechanical per-statement translation from `stack_move` → `move` / `ilist` / `py.constant` / `py.indexing` / `annotate` / `func`. Also inserts state threading. |
| `python/bloqade/lanes/rewrite/measure_lower.py` | `MeasureLower` rewrite: runs `AtomAnalysis`, enforces single-zone + single-final-measurement invariants, rewrites `move.Measure` → `move.EndMeasure`. |
| `python/tests/dialects/__init__.py` | New test package. |
| `python/tests/dialects/test_stack_move.py` | Smoke tests that every `stack_move` statement can be constructed with the expected fields/traits. |
| `python/tests/bytecode/test_decoder.py` | Per-opcode decoder unit tests + error tests. |
| `python/tests/rewrite/test_lower_stack_move.py` | Per-statement-family rewrite tests. |
| `python/tests/rewrite/test_measure_lower.py` | `measure_lower` unit tests (valid + invalid cases). |
| `python/tests/test_stack_move_e2e.py` | End-to-end test: `Program` → `load_program` → `lower_stack_move` → `measure_lower` → existing downstream pass. |

**Modified files:**

| File | Modification |
|---|---|
| `python/bloqade/lanes/dialects/move.py` | Add new `Measure` stmt — multi-zone, SSA-zone-valued, stateful. |
| `python/bloqade/lanes/analysis/atom/impl.py` | Add `@interp.impl(move.Measure)` method tracking zone sets + final-measurement count. |
| `python/tests/analysis/atom/test_atom_interpreter.py` | Add test for the new `move.Measure` analysis method. |

---

## Phase A — `stack_move` Dialect

### Task A1: Extend `bloqade.lanes.types` with new SSA type sentinels

**Files:**
- Modify: `python/bloqade/lanes/types.py`

All dialect-level SSA-type sentinels live in `bloqade.lanes.types` for reuse, alongside the existing `StateType` and `MeasurementFutureType`. This task adds a single new sentinel for arrays. The three address types (`LocationAddress`, `LaneAddress`, `ZoneAddress`) use the real Rust-backed classes from `bloqade.lanes.bytecode` directly, without a separate sentinel.

- [ ] **Step 1: Write implementation**

Append to `python/bloqade/lanes/types.py`:

```python
class Array:
    pass


ArrayType = types.PyClass(Array)
```

- [ ] **Step 2: Verify module imports cleanly**

```bash
uv run python -c "from bloqade.lanes.types import ArrayType; print(ArrayType)"
```

Expected: prints the `ArrayType` object (no ImportError).

- [ ] **Step 3: Commit**

```bash
git add python/bloqade/lanes/types.py
git commit -m "feat(types): add Array sentinel for SSA array-valued types"
```

---

### Task A2: Create the `stack_move` dialect module

**Files:**
- Create: `python/bloqade/lanes/dialects/stack_move.py`
- Create: `python/tests/dialects/__init__.py`
- Create: `python/tests/dialects/test_stack_move.py`

- [ ] **Step 1: Write failing test**

Create `python/tests/dialects/__init__.py` (empty) and `python/tests/dialects/test_stack_move.py`:

```python
from bloqade.lanes.dialects import stack_move


def test_dialect_exists():
    assert stack_move.dialect.name == "lanes.stack_move"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest python/tests/dialects/test_stack_move.py -v
```

Expected: `ModuleNotFoundError` — the dialect module doesn't exist yet.

- [ ] **Step 3: Write minimal implementation**

Create `python/bloqade/lanes/dialects/stack_move.py`. Note that `types.PyClass` takes an actual Python class, not a string. We use real Rust-backed classes for addresses and reuse the sentinel-backed types in `bloqade.lanes.types`:

```python
"""stack_move dialect — 1:1 SSA image of the bytecode."""

from kirin import ir, lowering, types
from kirin.decl import info, statement

from bloqade.lanes.bytecode import LaneAddress, LocationAddress, ZoneAddress
from bloqade.lanes.types import ArrayType, MeasurementFutureType

dialect = ir.Dialect(name="lanes.stack_move")


# ── SSA types ──────────────────────────────────────────────────────────

LocationAddressType = types.PyClass(LocationAddress)
LaneAddressType = types.PyClass(LaneAddress)
ZoneAddressType = types.PyClass(ZoneAddress)
# ArrayType and MeasurementFutureType come from bloqade.lanes.types.
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest python/tests/dialects/test_stack_move.py -v
```

Expected: PASS for `test_dialect_exists`.

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/dialects/stack_move.py python/tests/dialects/
git commit -m "feat(stack_move): add dialect skeleton"
```

---

### Task A3: Add constant statements

**Files:**
- Modify: `python/bloqade/lanes/dialects/stack_move.py`

Kirin's `@statement` machinery enforces the field/type interface of each statement at construction time, so direct per-statement tests would just duplicate framework-level checks. Real behavioural verification happens in the decoder (Phase B) and rewrite (Phase D) tests, which exercise each statement in context. These Phase A tasks therefore have no per-family tests — they just add statement definitions and re-run the Task A1 smoke test to confirm the module still imports cleanly.

- [ ] **Step 1: Write implementation**

Append to `python/bloqade/lanes/dialects/stack_move.py`:

```python
# ── Constants ──────────────────────────────────────────────────────────

@statement(dialect=dialect)
class ConstFloat(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})
    value: float = info.attribute()
    result: ir.ResultValue = info.result(types.Float)


@statement(dialect=dialect)
class ConstInt(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})
    value: int = info.attribute()
    result: ir.ResultValue = info.result(types.Int)


@statement(dialect=dialect)
class ConstLoc(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})
    value: LocationAddress = info.attribute()
    result: ir.ResultValue = info.result(LocationAddressType)


@statement(dialect=dialect)
class ConstLane(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})
    value: LaneAddress = info.attribute()
    result: ir.ResultValue = info.result(LaneAddressType)


@statement(dialect=dialect)
class ConstZone(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})
    value: ZoneAddress = info.attribute()
    result: ir.ResultValue = info.result(ZoneAddressType)
```

- [ ] **Step 2: Verify module imports cleanly**

```bash
uv run pytest python/tests/dialects/test_stack_move.py -v
```

Expected: the two Task A1 smoke tests still PASS.

- [ ] **Step 3: Commit**

```bash
git add python/bloqade/lanes/dialects/stack_move.py
git commit -m "feat(stack_move): add constant statements"
```

---

### Task A4: Add stack-manipulation statements

**Files:**
- Modify: `python/bloqade/lanes/dialects/stack_move.py`

- [ ] **Step 1: Write implementation**

Append to `python/bloqade/lanes/dialects/stack_move.py`:

```python
# ── Stack manipulation ─────────────────────────────────────────────────

@statement(dialect=dialect)
class Pop(ir.Statement):
    """Pop and discard the top of the virtual stack."""
    traits = frozenset({lowering.FromPythonCall()})
    value: ir.SSAValue = info.argument()


@statement(dialect=dialect)
class Dup(ir.Statement):
    """Duplicate the top of the virtual stack. Semantically result ≡ value;
    preserved as an explicit op to give downstream passes a hook for
    non-cloning invariants."""
    traits = frozenset({lowering.FromPythonCall()})
    value: ir.SSAValue = info.argument()
    result: ir.ResultValue = info.result()


@statement(dialect=dialect)
class Swap(ir.Statement):
    """Swap the top two virtual-stack values. out_top ≡ in_bot; out_bot ≡ in_top."""
    traits = frozenset({lowering.FromPythonCall()})
    in_top: ir.SSAValue = info.argument()
    in_bot: ir.SSAValue = info.argument()
    out_top: ir.ResultValue = info.result()
    out_bot: ir.ResultValue = info.result()
```

- [ ] **Step 2: Verify module imports cleanly**

```bash
uv run pytest python/tests/dialects/test_stack_move.py -v
```

Expected: Task A1 smoke tests still PASS.

- [ ] **Step 3: Commit**

```bash
git add python/bloqade/lanes/dialects/stack_move.py
git commit -m "feat(stack_move): add Pop/Dup/Swap statements"
```

---

### Task A5: Add atom-operation statements

**Files:**
- Modify: `python/bloqade/lanes/dialects/stack_move.py`

- [ ] **Step 1: Write implementation**

Append to `python/bloqade/lanes/dialects/stack_move.py`:

```python
# ── Atom operations ────────────────────────────────────────────────────

@statement(dialect=dialect)
class InitialFill(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})
    locations: tuple[ir.SSAValue, ...] = info.argument(type=LocationAddressType)


@statement(dialect=dialect)
class Fill(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})
    locations: tuple[ir.SSAValue, ...] = info.argument(type=LocationAddressType)


@statement(dialect=dialect)
class Move(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})
    lanes: tuple[ir.SSAValue, ...] = info.argument(type=LaneAddressType)
```

- [ ] **Step 2: Verify module imports cleanly**

```bash
uv run pytest python/tests/dialects/test_stack_move.py -v
```

Expected: Task A1 smoke tests still PASS.

- [ ] **Step 3: Commit**

```bash
git add python/bloqade/lanes/dialects/stack_move.py
git commit -m "feat(stack_move): add InitialFill/Fill/Move statements"
```

---

### Task A6: Add gate statements

**Files:**
- Modify: `python/bloqade/lanes/dialects/stack_move.py`

- [ ] **Step 1: Write implementation**

Append to `python/bloqade/lanes/dialects/stack_move.py`:

```python
# ── Gates ──────────────────────────────────────────────────────────────

@statement(dialect=dialect)
class LocalR(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})
    phi: ir.SSAValue = info.argument(type=types.Float)
    theta: ir.SSAValue = info.argument(type=types.Float)
    locations: tuple[ir.SSAValue, ...] = info.argument(type=LocationAddressType)


@statement(dialect=dialect)
class LocalRz(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})
    theta: ir.SSAValue = info.argument(type=types.Float)
    locations: tuple[ir.SSAValue, ...] = info.argument(type=LocationAddressType)


@statement(dialect=dialect)
class GlobalR(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})
    phi: ir.SSAValue = info.argument(type=types.Float)
    theta: ir.SSAValue = info.argument(type=types.Float)


@statement(dialect=dialect)
class GlobalRz(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})
    theta: ir.SSAValue = info.argument(type=types.Float)


@statement(dialect=dialect)
class CZ(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})
    zone: ir.SSAValue = info.argument(type=ZoneAddressType)
```

- [ ] **Step 2: Verify module imports cleanly**

```bash
uv run pytest python/tests/dialects/test_stack_move.py -v
```

Expected: Task A1 smoke tests still PASS.

- [ ] **Step 3: Commit**

```bash
git add python/bloqade/lanes/dialects/stack_move.py
git commit -m "feat(stack_move): add gate statements (LocalR/LocalRz/GlobalR/GlobalRz/CZ)"
```

---

### Task A7: Add measurement and control-flow statements

**Files:**
- Modify: `python/bloqade/lanes/dialects/stack_move.py`

- [ ] **Step 1: Write implementation**

Append to `python/bloqade/lanes/dialects/stack_move.py`:

```python
# ── Measurement ────────────────────────────────────────────────────────

@statement(dialect=dialect)
class Measure(ir.Statement):
    """Matches bytecode `measure(arity)` — takes location SSA values.
    Zone grouping happens during lower_stack_move."""
    traits = frozenset({lowering.FromPythonCall()})
    locations: tuple[ir.SSAValue, ...] = info.argument(type=LocationAddressType)
    result: ir.ResultValue = info.result(MeasurementFutureType)


@statement(dialect=dialect)
class AwaitMeasure(ir.Statement):
    """Synchronisation — blocks until the most recent measurement completes.

    The bytecode docs state 'block until the most recent measurement
    completes' with no documented stack effect. We treat this as a pure
    synchronisation op: takes a MeasurementFuture, produces no result.
    Extracting per-location measurement values is done via subsequent
    GetItem calls on the future. Confirm against the Rust source before
    implementation — adjust if the actual stack effect differs."""
    traits = frozenset({lowering.FromPythonCall()})
    future: ir.SSAValue = info.argument(type=MeasurementFutureType)


# ── Control flow ───────────────────────────────────────────────────────

@statement(dialect=dialect)
class Return(ir.Statement):
    traits = frozenset({lowering.FromPythonCall(), ir.IsTerminator()})


@statement(dialect=dialect)
class Halt(ir.Statement):
    """Lowered to func.Return(None) alongside Return."""
    traits = frozenset({lowering.FromPythonCall(), ir.IsTerminator()})
```

- [ ] **Step 2: Verify module imports cleanly**

```bash
uv run pytest python/tests/dialects/test_stack_move.py -v
```

Expected: Task A1 smoke tests still PASS.

- [ ] **Step 3: Commit**

```bash
git add python/bloqade/lanes/dialects/stack_move.py
git commit -m "feat(stack_move): add Measure/AwaitMeasure/Return/Halt"
```

---

### Task A8: Add array and annotation statements

**Files:**
- Modify: `python/bloqade/lanes/dialects/stack_move.py`

- [ ] **Step 1: Write implementation**

Append to `python/bloqade/lanes/dialects/stack_move.py`. Import the annotation result types from `bloqade.decoders.dialects.annotate.types` — `SetDetector` produces a `Detector` and `SetObservable` produces an `Observable`, matching the target `annotate` dialect's function signatures:

```python
# Add near the top of the file, with the other imports:
from bloqade.decoders.dialects.annotate.types import DetectorType, ObservableType
```

Then append the statement definitions:

```python
# ── Arrays ─────────────────────────────────────────────────────────────

@statement(dialect=dialect)
class NewArray(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})
    type_tag: int = info.attribute()
    dim0: int = info.attribute()
    dim1: int = info.attribute()
    result: ir.ResultValue = info.result(ArrayType)


@statement(dialect=dialect)
class GetItem(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})
    array: ir.SSAValue = info.argument(type=ArrayType)
    indices: tuple[ir.SSAValue, ...] = info.argument(type=types.Int)
    result: ir.ResultValue = info.result()  # element type is context-dependent


# ── Annotations (detectors / observables) ──────────────────────────────

@statement(dialect=dialect)
class SetDetector(ir.Statement):
    """Build a detector record from the top-of-stack array. Matches
    annotate.SetDetector's signature — produces a Detector."""
    traits = frozenset({lowering.FromPythonCall()})
    array: ir.SSAValue = info.argument(type=ArrayType)
    result: ir.ResultValue = info.result(DetectorType)


@statement(dialect=dialect)
class SetObservable(ir.Statement):
    """Build an observable record from the top-of-stack array. Matches
    annotate.SetObservable's signature — produces an Observable."""
    traits = frozenset({lowering.FromPythonCall()})
    array: ir.SSAValue = info.argument(type=ArrayType)
    result: ir.ResultValue = info.result(ObservableType)
```

- [ ] **Step 2: Verify module imports cleanly**

```bash
uv run pytest python/tests/dialects/test_stack_move.py -v
```

Expected: Task A1 smoke tests still PASS.

- [ ] **Step 3: Commit**

```bash
git add python/bloqade/lanes/dialects/stack_move.py
git commit -m "feat(stack_move): add array and annotation statements"
```

At this point the `stack_move` dialect is complete. Run the full test file and verify all smoke tests pass:

```bash
uv run pytest python/tests/dialects/test_stack_move.py -v
```

Deeper verification of statement behaviour happens naturally through the decoder tests (Phase B) and rewrite tests (Phase D), which exercise each statement in context rather than duplicating Kirin's built-in interface checks.

---

## Phase B — Bytecode Decoder

Phase B depends on the PyO3 accessor prerequisite noted at the top. Each decoder task uses the accessors; if they don't exist yet, these tasks are blocked.

### Task B1: Create decoder skeleton + smoke test

**Files:**
- Create: `python/bloqade/lanes/bytecode/lowering.py`
- Create: `python/tests/bytecode/test_decoder.py`

- [ ] **Step 1: Write failing test**

Create `python/tests/bytecode/test_decoder.py`:

```python
from bloqade.lanes.bytecode import Instruction, Program
from bloqade.lanes.bytecode.lowering import load_program


def test_empty_program_returns_method_with_empty_body():
    prog = Program(version=(1, 0), instructions=[Instruction.return_()])
    method = load_program(prog)
    assert method.sym_name == "main"
    # Body should have one return, no other statements
    block = method.callable_region.blocks[0]
    assert len(block.stmts) == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest python/tests/bytecode/test_decoder.py -v
```

Expected: FAIL with `ModuleNotFoundError` on `lowering`.

- [ ] **Step 3: Write implementation**

Create `python/bloqade/lanes/bytecode/lowering.py`:

```python
"""BytecodeDecoder — syntactic Program → stack_move ir.Method."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from kirin import ir

from bloqade.lanes.dialects import stack_move

if TYPE_CHECKING:
    from bloqade.lanes.bytecode import Instruction, Program


@dataclass
class DecodeError(Exception):
    """Raised when the decoder fails.

    Carries the offending instruction's index, opcode, and a snapshot of
    the virtual stack of SSA values at the point of failure.
    """

    instruction_index: int
    opcode_name: str
    stack_snapshot: tuple[ir.SSAValue, ...]
    reason: str

    def __str__(self) -> str:
        return (
            f"DecodeError at instruction {self.instruction_index} "
            f"({self.opcode_name}): {self.reason} "
            f"[stack depth={len(self.stack_snapshot)}]"
        )


@dataclass
class BytecodeDecoder:
    """Turn a bytecode Program into a stack_move ir.Method.

    Maintains a virtual stack of SSA values during decoding: each bytecode
    push emits a stack_move statement whose result is pushed onto the
    virtual stack, and each pop consumes the top SSA reference. Stack ops
    (Pop/Dup/Swap) emit corresponding stack_move statements (linear-IR
    style — see the design doc).
    """

    stack: list[ir.SSAValue] = field(default_factory=list)
    block: ir.Block = field(default_factory=ir.Block)

    def decode(self, program: "Program", kernel_name: str = "main") -> ir.Method:
        for idx, instr in enumerate(program.instructions):
            self._visit(idx, instr)
        return self._finalize(kernel_name)

    def _visit(self, idx: int, instr: "Instruction") -> None:
        # Dispatch placeholder — filled in later tasks
        raise NotImplementedError(f"opcode {instr.opcode} not yet handled")

    def _finalize(self, kernel_name: str) -> ir.Method:
        # Wrap self.block in a Region and an ir.Method named kernel_name.
        # See bloqade.lanes._prelude / upstream.py for examples of
        # programmatic Method construction in this codebase.
        region = ir.Region(blocks=[self.block])
        return ir.Method(
            sym_name=kernel_name,
            mod=None,
            py_func=None,
            arg_names=[],
            dialects=ir.DialectGroup([stack_move.dialect]),
            code=region,
        )


def load_program(program: "Program", kernel_name: str = "main") -> ir.Method:
    """Decode a bytecode Program into a stack_move ir.Method."""
    decoder = BytecodeDecoder()
    return decoder.decode(program, kernel_name)
```

Note: the exact `ir.Method` constructor signature may differ; check against `kirin.ir.Method` or existing `bloqade-lanes` usage (e.g. in `python/bloqade/lanes/upstream.py`). Adjust if needed.

Also append the `Return` handler (to make the smoke test pass — a program containing only `Instruction.return_()`):

```python
    def _visit(self, idx: int, instr: "Instruction") -> None:
        name = instr.op_name()  # prerequisite PyO3 accessor
        handler = getattr(self, f"_visit_{name}", None)
        if handler is None:
            raise DecodeError(idx, name, tuple(self.stack), "unknown opcode")
        handler(idx, instr)

    def _visit_return(self, idx: int, instr: "Instruction") -> None:
        self.block.stmts.append(stack_move.Return())
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest python/tests/bytecode/test_decoder.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/bytecode/lowering.py python/tests/bytecode/test_decoder.py
git commit -m "feat(bytecode): add BytecodeDecoder skeleton with return handler"
```

---

### Task B2: Decode constants (ConstFloat/Int/Loc/Lane/Zone)

**Files:**
- Modify: `python/bloqade/lanes/bytecode/lowering.py`
- Modify: `python/tests/bytecode/test_decoder.py`

- [ ] **Step 1: Write failing tests**

Append to `python/tests/bytecode/test_decoder.py`:

```python
from bloqade.lanes.bytecode import (
    Direction, Instruction, LaneAddress, LocationAddress, MoveType, Program, ZoneAddress,
)
from bloqade.lanes.dialects import stack_move


def _decode(instructions):
    prog = Program(version=(1, 0), instructions=instructions + [Instruction.return_()])
    return load_program(prog).callable_region.blocks[0]


def test_decode_const_float():
    block = _decode([Instruction.const_float(2.5)])
    assert any(
        isinstance(s, stack_move.ConstFloat) and s.value == 2.5 for s in block.stmts
    )


def test_decode_const_int():
    block = _decode([Instruction.const_int(42)])
    assert any(
        isinstance(s, stack_move.ConstInt) and s.value == 42 for s in block.stmts
    )


def test_decode_const_loc():
    block = _decode([Instruction.const_loc(0, 0, 0)])
    stmt = next(s for s in block.stmts if isinstance(s, stack_move.ConstLoc))
    assert stmt.value == LocationAddress(0, 0, 0)


def test_decode_const_lane():
    block = _decode([Instruction.const_lane(MoveType.SITE, 0, 0, 0, 0)])
    stmt = next(s for s in block.stmts if isinstance(s, stack_move.ConstLane))
    assert stmt.value == LaneAddress(MoveType.SITE, 0, 0, 0, 0)


def test_decode_const_zone():
    block = _decode([Instruction.const_zone(3)])
    stmt = next(s for s in block.stmts if isinstance(s, stack_move.ConstZone))
    assert stmt.value == ZoneAddress(3)
```

- [ ] **Step 2: Run tests**

```bash
uv run pytest python/tests/bytecode/test_decoder.py -v -k const
```

Expected: FAIL with `DecodeError(... unknown opcode)` for each.

- [ ] **Step 3: Write implementation**

Add to `BytecodeDecoder` in `python/bloqade/lanes/bytecode/lowering.py`:

```python
    def _visit_const_float(self, idx: int, instr: "Instruction") -> None:
        stmt = stack_move.ConstFloat(value=instr.float_value())
        self.block.stmts.append(stmt)
        self.stack.append(stmt.result)

    def _visit_const_int(self, idx: int, instr: "Instruction") -> None:
        stmt = stack_move.ConstInt(value=instr.int_value())
        self.block.stmts.append(stmt)
        self.stack.append(stmt.result)

    def _visit_const_loc(self, idx: int, instr: "Instruction") -> None:
        stmt = stack_move.ConstLoc(value=instr.location_address())
        self.block.stmts.append(stmt)
        self.stack.append(stmt.result)

    def _visit_const_lane(self, idx: int, instr: "Instruction") -> None:
        stmt = stack_move.ConstLane(value=instr.lane_address())
        self.block.stmts.append(stmt)
        self.stack.append(stmt.result)

    def _visit_const_zone(self, idx: int, instr: "Instruction") -> None:
        stmt = stack_move.ConstZone(value=instr.zone_address())
        self.block.stmts.append(stmt)
        self.stack.append(stmt.result)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest python/tests/bytecode/test_decoder.py -v -k const
```

Expected: all constant decoding tests PASS.

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/bytecode/lowering.py python/tests/bytecode/test_decoder.py
git commit -m "feat(bytecode): decode constant instructions"
```

---

### Task B3: Decode stack ops (Pop/Dup/Swap)

**Files:**
- Modify: `python/bloqade/lanes/bytecode/lowering.py`
- Modify: `python/tests/bytecode/test_decoder.py`

- [ ] **Step 1: Write failing tests**

Append:

```python
def test_decode_pop_consumes_top():
    block = _decode([Instruction.const_int(1), Instruction.pop()])
    assert any(isinstance(s, stack_move.Pop) for s in block.stmts)


def test_decode_dup_duplicates_top():
    block = _decode([Instruction.const_int(1), Instruction.dup()])
    dup = next(s for s in block.stmts if isinstance(s, stack_move.Dup))
    cint = next(s for s in block.stmts if isinstance(s, stack_move.ConstInt))
    assert dup.value is cint.result


def test_decode_swap_permutes_top_two():
    block = _decode([Instruction.const_int(1), Instruction.const_int(2), Instruction.swap()])
    swap = next(s for s in block.stmts if isinstance(s, stack_move.Swap))
    ints = [s for s in block.stmts if isinstance(s, stack_move.ConstInt)]
    assert swap.in_top is ints[1].result
    assert swap.in_bot is ints[0].result


def test_decode_pop_underflow_raises():
    import pytest
    from bloqade.lanes.bytecode.lowering import DecodeError
    with pytest.raises(DecodeError):
        _decode([Instruction.pop()])
```

- [ ] **Step 2: Run tests**

```bash
uv run pytest python/tests/bytecode/test_decoder.py -v -k "pop or dup or swap"
```

Expected: FAIL.

- [ ] **Step 3: Write implementation**

Add to `BytecodeDecoder`:

```python
    def _pop_or_raise(self, idx: int, instr: "Instruction") -> ir.SSAValue:
        if not self.stack:
            raise DecodeError(
                idx, instr.op_name(), tuple(self.stack), "stack underflow"
            )
        return self.stack.pop()

    def _visit_pop(self, idx: int, instr: "Instruction") -> None:
        value = self._pop_or_raise(idx, instr)
        self.block.stmts.append(stack_move.Pop(value=value))

    def _visit_dup(self, idx: int, instr: "Instruction") -> None:
        if not self.stack:
            raise DecodeError(idx, "dup", tuple(self.stack), "stack underflow")
        top = self.stack[-1]
        stmt = stack_move.Dup(value=top)
        self.block.stmts.append(stmt)
        self.stack.append(stmt.result)

    def _visit_swap(self, idx: int, instr: "Instruction") -> None:
        in_top = self._pop_or_raise(idx, instr)
        in_bot = self._pop_or_raise(idx, instr)
        stmt = stack_move.Swap(in_top=in_top, in_bot=in_bot)
        self.block.stmts.append(stmt)
        # Convention: top-of-stack last. out_bot ≡ in_top (goes below);
        # out_top ≡ in_bot (goes on top).
        self.stack.append(stmt.out_bot)
        self.stack.append(stmt.out_top)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest python/tests/bytecode/test_decoder.py -v -k "pop or dup or swap"
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/bytecode/lowering.py python/tests/bytecode/test_decoder.py
git commit -m "feat(bytecode): decode Pop/Dup/Swap with stack-underflow detection"
```

---

### Task B4: Decode atom ops (InitialFill/Fill/Move)

**Files:**
- Modify: `python/bloqade/lanes/bytecode/lowering.py`
- Modify: `python/tests/bytecode/test_decoder.py`

- [ ] **Step 1: Write failing tests**

Append:

```python
def test_decode_fill_consumes_arity_locations():
    block = _decode([
        Instruction.const_loc(0, 0, 0),
        Instruction.const_loc(0, 0, 1),
        Instruction.fill(2),
    ])
    fill = next(s for s in block.stmts if isinstance(s, stack_move.Fill))
    locs = [s for s in block.stmts if isinstance(s, stack_move.ConstLoc)]
    assert fill.locations == (locs[0].result, locs[1].result)


def test_decode_initial_fill():
    block = _decode([Instruction.const_loc(0, 0, 0), Instruction.initial_fill(1)])
    assert any(isinstance(s, stack_move.InitialFill) for s in block.stmts)


def test_decode_move_consumes_arity_lanes():
    block = _decode([
        Instruction.const_lane(MoveType.SITE, 0, 0, 0, 0),
        Instruction.move_(1),
    ])
    mv = next(s for s in block.stmts if isinstance(s, stack_move.Move))
    lane = next(s for s in block.stmts if isinstance(s, stack_move.ConstLane))
    assert mv.lanes == (lane.result,)
```

- [ ] **Step 2: Run tests**

```bash
uv run pytest python/tests/bytecode/test_decoder.py -v -k "fill or move"
```

Expected: FAIL.

- [ ] **Step 3: Write implementation**

Add to `BytecodeDecoder`:

```python
    def _pop_n(self, idx: int, instr: "Instruction", n: int) -> list[ir.SSAValue]:
        """Pop n values from the stack, newest first. Returns them in
        bottom-to-top order so the caller can pass them as a tuple
        matching 'top-of-stack = last argument' convention."""
        if len(self.stack) < n:
            raise DecodeError(
                idx, instr.op_name(), tuple(self.stack),
                f"stack underflow (need {n}, have {len(self.stack)})",
            )
        popped = [self.stack.pop() for _ in range(n)]
        popped.reverse()  # now in bottom-to-top order
        return popped

    def _visit_initial_fill(self, idx: int, instr: "Instruction") -> None:
        locs = self._pop_n(idx, instr, instr.arity())
        self.block.stmts.append(stack_move.InitialFill(locations=tuple(locs)))

    def _visit_fill(self, idx: int, instr: "Instruction") -> None:
        locs = self._pop_n(idx, instr, instr.arity())
        self.block.stmts.append(stack_move.Fill(locations=tuple(locs)))

    def _visit_move_(self, idx: int, instr: "Instruction") -> None:
        # Bytecode opcode name is `move_` to avoid the Python builtin.
        lanes = self._pop_n(idx, instr, instr.arity())
        self.block.stmts.append(stack_move.Move(lanes=tuple(lanes)))
```

Note: confirm the `op_name()` string used for the move opcode. If it returns `"move"` (not `"move_"`), rename the handler accordingly.

- [ ] **Step 4: Run tests**

```bash
uv run pytest python/tests/bytecode/test_decoder.py -v -k "fill or move"
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/bytecode/lowering.py python/tests/bytecode/test_decoder.py
git commit -m "feat(bytecode): decode atom operations (InitialFill/Fill/Move)"
```

---

### Task B5: Decode gates (LocalR/LocalRz/GlobalR/GlobalRz/CZ)

**Files:**
- Modify: `python/bloqade/lanes/bytecode/lowering.py`
- Modify: `python/tests/bytecode/test_decoder.py`

- [ ] **Step 1: Write failing tests**

Append:

```python
def test_decode_local_r():
    block = _decode([
        Instruction.const_loc(0, 0, 0),  # loc
        Instruction.const_float(0.1),    # theta
        Instruction.const_float(0.2),    # phi
        Instruction.local_r(1),
    ])
    r = next(s for s in block.stmts if isinstance(s, stack_move.LocalR))
    floats = [s for s in block.stmts if isinstance(s, stack_move.ConstFloat)]
    locs = [s for s in block.stmts if isinstance(s, stack_move.ConstLoc)]
    # bytecode pops phi first, then theta, then locations (per .pyi docstring)
    assert r.phi is floats[1].result
    assert r.theta is floats[0].result
    assert r.locations == (locs[0].result,)


def test_decode_global_rz():
    block = _decode([Instruction.const_float(0.5), Instruction.global_rz()])
    rz = next(s for s in block.stmts if isinstance(s, stack_move.GlobalRz))
    cf = next(s for s in block.stmts if isinstance(s, stack_move.ConstFloat))
    assert rz.theta is cf.result


def test_decode_cz():
    block = _decode([Instruction.const_zone(0), Instruction.cz()])
    cz = next(s for s in block.stmts if isinstance(s, stack_move.CZ))
    cz_zone = next(s for s in block.stmts if isinstance(s, stack_move.ConstZone))
    assert cz.zone is cz_zone.result
```

- [ ] **Step 2: Run tests**

```bash
uv run pytest python/tests/bytecode/test_decoder.py -v -k "local or global or cz"
```

Expected: FAIL.

- [ ] **Step 3: Write implementation**

Add to `BytecodeDecoder`. Note the pop order: per `_native.pyi` for `local_r`, the stack has (bottom→top) `[locations..., theta, phi]`, so we pop `phi` first, `theta` second, then `arity` locations.

```python
    def _visit_local_r(self, idx: int, instr: "Instruction") -> None:
        phi = self._pop_or_raise(idx, instr)
        theta = self._pop_or_raise(idx, instr)
        locs = self._pop_n(idx, instr, instr.arity())
        self.block.stmts.append(
            stack_move.LocalR(phi=phi, theta=theta, locations=tuple(locs))
        )

    def _visit_local_rz(self, idx: int, instr: "Instruction") -> None:
        theta = self._pop_or_raise(idx, instr)
        locs = self._pop_n(idx, instr, instr.arity())
        self.block.stmts.append(
            stack_move.LocalRz(theta=theta, locations=tuple(locs))
        )

    def _visit_global_r(self, idx: int, instr: "Instruction") -> None:
        phi = self._pop_or_raise(idx, instr)
        theta = self._pop_or_raise(idx, instr)
        self.block.stmts.append(stack_move.GlobalR(phi=phi, theta=theta))

    def _visit_global_rz(self, idx: int, instr: "Instruction") -> None:
        theta = self._pop_or_raise(idx, instr)
        self.block.stmts.append(stack_move.GlobalRz(theta=theta))

    def _visit_cz(self, idx: int, instr: "Instruction") -> None:
        zone = self._pop_or_raise(idx, instr)
        self.block.stmts.append(stack_move.CZ(zone=zone))
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest python/tests/bytecode/test_decoder.py -v -k "local or global or cz"
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/bytecode/lowering.py python/tests/bytecode/test_decoder.py
git commit -m "feat(bytecode): decode gate instructions"
```

---

### Task B6: Decode measurement, arrays, annotations, halt

**Files:**
- Modify: `python/bloqade/lanes/bytecode/lowering.py`
- Modify: `python/tests/bytecode/test_decoder.py`

- [ ] **Step 1: Write failing tests**

Append:

```python
def test_decode_measure():
    block = _decode([Instruction.const_loc(0, 0, 0), Instruction.measure(1)])
    m = next(s for s in block.stmts if isinstance(s, stack_move.Measure))
    loc = next(s for s in block.stmts if isinstance(s, stack_move.ConstLoc))
    assert m.locations == (loc.result,)


def test_decode_await_measure():
    # measure pushes a future; await_measure consumes it
    block = _decode([
        Instruction.const_loc(0, 0, 0),
        Instruction.measure(1),
        Instruction.await_measure(),
    ])
    aw = next(s for s in block.stmts if isinstance(s, stack_move.AwaitMeasure))
    m = next(s for s in block.stmts if isinstance(s, stack_move.Measure))
    assert aw.future is m.result


def test_decode_new_array():
    block = _decode([Instruction.new_array(type_tag=0, dim0=4)])
    na = next(s for s in block.stmts if isinstance(s, stack_move.NewArray))
    assert na.dim0 == 4


def test_decode_get_item():
    block = _decode([
        Instruction.new_array(type_tag=0, dim0=4),
        Instruction.const_int(2),
        Instruction.get_item(1),
    ])
    gi = next(s for s in block.stmts if isinstance(s, stack_move.GetItem))
    assert len(gi.indices) == 1


def test_decode_halt():
    prog = Program(version=(1, 0), instructions=[Instruction.halt()])
    method = load_program(prog)
    block = method.callable_region.blocks[0]
    assert any(isinstance(s, stack_move.Halt) for s in block.stmts)
```

- [ ] **Step 2: Run tests**

```bash
uv run pytest python/tests/bytecode/test_decoder.py -v
```

Expected: FAIL on the new tests.

- [ ] **Step 3: Write implementation**

Add to `BytecodeDecoder`:

```python
    def _visit_measure(self, idx: int, instr: "Instruction") -> None:
        locs = self._pop_n(idx, instr, instr.arity())
        stmt = stack_move.Measure(locations=tuple(locs))
        self.block.stmts.append(stmt)
        self.stack.append(stmt.result)

    def _visit_await_measure(self, idx: int, instr: "Instruction") -> None:
        # Treats await_measure as pure synchronisation — takes the future off
        # the top of the virtual stack, pushes it back so subsequent GetItem
        # calls can access measurement values. Verify the exact stack effect
        # against the Rust source and adjust if the bytecode actually pops
        # the future permanently.
        future = self._pop_or_raise(idx, instr)
        self.block.stmts.append(stack_move.AwaitMeasure(future=future))
        self.stack.append(future)

    def _visit_new_array(self, idx: int, instr: "Instruction") -> None:
        stmt = stack_move.NewArray(
            type_tag=instr.type_tag(),
            dim0=instr.dim0(),
            dim1=instr.dim1(),
        )
        self.block.stmts.append(stmt)
        self.stack.append(stmt.result)

    def _visit_get_item(self, idx: int, instr: "Instruction") -> None:
        ndims = instr.ndims()
        indices = self._pop_n(idx, instr, ndims)
        array = self._pop_or_raise(idx, instr)
        stmt = stack_move.GetItem(array=array, indices=tuple(indices))
        self.block.stmts.append(stmt)
        self.stack.append(stmt.result)

    def _visit_set_detector(self, idx: int, instr: "Instruction") -> None:
        array = self._pop_or_raise(idx, instr)
        stmt = stack_move.SetDetector(array=array)
        self.block.stmts.append(stmt)
        self.stack.append(stmt.result)

    def _visit_set_observable(self, idx: int, instr: "Instruction") -> None:
        array = self._pop_or_raise(idx, instr)
        stmt = stack_move.SetObservable(array=array)
        self.block.stmts.append(stmt)
        self.stack.append(stmt.result)

    def _visit_halt(self, idx: int, instr: "Instruction") -> None:
        self.block.stmts.append(stack_move.Halt())
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest python/tests/bytecode/test_decoder.py -v
```

Expected: all decoder tests PASS.

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/bytecode/lowering.py python/tests/bytecode/test_decoder.py
git commit -m "feat(bytecode): decode measurement, arrays, annotations, halt"
```

---

## Phase C — New `move.Measure` Statement

### Task C1: Add `move.Measure` (multi-zone SSA-based)

**Files:**
- Modify: `python/bloqade/lanes/dialects/move.py`
- Create: `python/tests/dialects/test_move_measure.py`

- [ ] **Step 1: Write failing test**

Create `python/tests/dialects/test_move_measure.py`:

```python
from kirin import ir

from bloqade.lanes.dialects import move


def test_measure_fields():
    state = ir.TestValue()
    z0 = ir.TestValue()
    z1 = ir.TestValue()
    stmt = move.Measure(current_state=state, zones=(z0, z1))
    assert stmt.current_state is state
    assert stmt.zones == (z0, z1)
    assert stmt.result is not None


def test_measure_has_consumes_state_trait():
    # The new Measure stmt is stateful like EndMeasure.
    state = ir.TestValue()
    z = ir.TestValue()
    stmt = move.Measure(current_state=state, zones=(z,))
    assert stmt.get_trait(move.ConsumesState) is not None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest python/tests/dialects/test_move_measure.py -v
```

Expected: FAIL with `AttributeError: 'move' module has no attribute 'Measure'` (or similar).

- [ ] **Step 3: Write implementation**

In `python/bloqade/lanes/dialects/move.py`, add (near the existing `EndMeasure` statement) a new `ZoneAddressType` SSA type if one doesn't exist yet, plus the `Measure` statement:

```python
# (near top, next to existing SSA types)
ZoneAddressType = types.PyClass("ZoneAddress")


@statement(dialect=dialect)
class Measure(ir.Statement):
    """Multi-zone measurement produced by lower_stack_move. Consumed by
    measure_lower, which validates single-zone + single-final-measurement
    invariants and rewrites to EndMeasure."""
    traits = frozenset({lowering.FromPythonCall(), ConsumesState(True)})
    current_state: ir.SSAValue = info.argument(StateType)
    zones: tuple[ir.SSAValue, ...] = info.argument(type=ZoneAddressType)
    result: ir.ResultValue = info.result(MeasurementFutureType)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest python/tests/dialects/test_move_measure.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/dialects/move.py python/tests/dialects/test_move_measure.py
git commit -m "feat(move): add multi-zone SSA-based Measure statement"
```

---

## Phase D — `lower_stack_move` Rewrite

### Task D1: Skeleton + state-threading infrastructure

**Files:**
- Create: `python/bloqade/lanes/rewrite/lower_stack_move.py`
- Create: `python/tests/rewrite/test_lower_stack_move.py`

This rewrite follows Kirin's `RewriteRule` interface (see `kirin/rewrite/abc.py` — `RewriteRule` with specialised `rewrite_Region` / `rewrite_Block` / `rewrite_Statement` handlers, all returning `RewriteResult`; IR mutation is in-place via `insert_before`, `replace_by`, `delete` on statements). The existing `bloqade-lanes` rewrites in `python/bloqade/lanes/rewrite/` (e.g. `state.py::RewriteLoadStore`, `place2move.py::InsertMoves`) show the same pattern and are the reference to match.

For `LowerStackMove` we override `rewrite_Block` because:
1. We need to insert the initial `move.Load()` once per block at the start.
2. State threading + attribute lifting across statements requires walk-order processing that's natural at block level.
3. Matches `RewriteLoadStore` which also processes an entire block in one pass.

- [ ] **Step 1: Write failing smoke test**

Create `python/tests/rewrite/test_lower_stack_move.py`:

```python
from kirin import ir
from kirin.rewrite import Walk

from bloqade.lanes.dialects import move, stack_move
from bloqade.lanes.rewrite.lower_stack_move import LowerStackMove


def _build_stack_move_block(stmts: list[ir.Statement]) -> ir.Block:
    block = ir.Block()
    for stmt in stmts:
        block.stmts.append(stmt)
    return block


def test_empty_block_emits_load_and_func_return():
    block = _build_stack_move_block([stack_move.Return()])
    result = Walk(LowerStackMove()).rewrite(block)
    assert result.has_done_something
    # Expect a move.Load at block start and a func.Return; the stack_move
    # Return should have been deleted.
    assert any(isinstance(s, move.Load) for s in block.stmts)
    assert not any(isinstance(s, stack_move.Return) for s in block.stmts)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest python/tests/rewrite/test_lower_stack_move.py -v
```

Expected: FAIL with `ModuleNotFoundError` on `lower_stack_move`.

- [ ] **Step 3: Write implementation**

Create `python/bloqade/lanes/rewrite/lower_stack_move.py`:

```python
"""lower_stack_move — in-place rewrite from stack_move → multi-dialect IR.

Extends Kirin's RewriteRule with a rewrite_Block handler that walks the
block's statements once and, for each stack_move statement, inserts the
corresponding target-dialect statement(s) via insert_before and deletes
the original. State threading is woven in along the way: move.Load at
block start initialises the StateType SSA value, each stateful move.*
op consumes the current state and produces a new one, and move.Store +
func.Return close out the block.

Follows the same pattern as python/bloqade/lanes/rewrite/state.py's
RewriteLoadStore.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kirin import ir
from kirin.rewrite.abc import RewriteResult, RewriteRule

from bloqade.lanes.dialects import move, stack_move


@dataclass
class LowerStackMove(RewriteRule):
    """Lower a stack_move block into a multi-dialect block in place.

    Mutable state on the rule instance, carried across the walk:
    - ssa_to_attr: stack_move SSA → underlying attribute value, for
      operands that need to be lifted into target-dialect attributes
      (addresses, rotation angles).
    - ssa_to_target: stack_move SSA → target-dialect SSA replacement,
      for values that survive as SSA (arrays, futures, detectors,
      observables).
    - state: the current StateType SSA value in the target IR.
    """

    ssa_to_attr: dict[ir.SSAValue, Any] = field(default_factory=dict)
    ssa_to_target: dict[ir.SSAValue, ir.SSAValue] = field(default_factory=dict)
    state: ir.SSAValue | None = None

    def rewrite_Block(self, block: ir.Block) -> RewriteResult:
        # Insert the initial move.Load at block start.
        load = move.Load()
        first = next(iter(block.stmts), None)
        if first is None:
            block.stmts.append(load)
        else:
            load.insert_before(first)
        self.state = load.result

        to_delete: list[ir.Statement] = []
        for stmt in list(block.stmts):
            if stmt is load:
                continue
            handler = getattr(self, f"_rewrite_{type(stmt).__name__}", None)
            if handler is None:
                # Non-stack_move statements (e.g. existing py.Constant) pass
                # through unchanged.
                continue
            handler(stmt, to_delete)

        for stmt in to_delete:
            stmt.delete()
        return RewriteResult(has_done_something=True)

    def _rewrite_Return(self, stmt: stack_move.Return, to_delete: list) -> None:
        from kirin.dialects import func
        move.Store(self.state).insert_before(stmt)
        func.Return().insert_before(stmt)
        to_delete.append(stmt)

    def _rewrite_Halt(self, stmt: stack_move.Halt, to_delete: list) -> None:
        from kirin.dialects import func
        move.Store(self.state).insert_before(stmt)
        func.Return().insert_before(stmt)
        to_delete.append(stmt)
```

- [ ] **Step 4: Run test**

```bash
uv run pytest python/tests/rewrite/test_lower_stack_move.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/rewrite/lower_stack_move.py python/tests/rewrite/test_lower_stack_move.py
git commit -m "feat(rewrite): lower_stack_move skeleton (RewriteRule + state init + Return/Halt)"
```

---

### Task D2: Rewrite constants (attribute tracking + target Const emission)

**Files:**
- Modify: `python/bloqade/lanes/rewrite/lower_stack_move.py`
- Modify: `python/tests/rewrite/test_lower_stack_move.py`

- [ ] **Step 1: Write failing tests**

Append:

```python
from kirin.dialects import py
from bloqade.lanes.bytecode import LocationAddress


def test_const_float_emits_py_constant_and_tracks_value():
    cf = stack_move.ConstFloat(value=1.5)
    block = _build_stack_move_block([cf, stack_move.Return()])
    rule = LowerStackMove()
    Walk(rule).rewrite(block)
    # py.Constant statement emitted with value 1.5, in the same block.
    assert any(
        isinstance(s, py.Constant) and s.value.unwrap() == 1.5 for s in block.stmts
    )
    # The original stack_move SSA value is mapped to its target SSA.
    assert cf.result in rule.ssa_to_target


def test_const_loc_tracks_attribute_value():
    addr = LocationAddress(0, 0, 0)
    cl = stack_move.ConstLoc(value=addr)
    block = _build_stack_move_block([cl, stack_move.Return()])
    rule = LowerStackMove()
    Walk(rule).rewrite(block)
    # The stack_move SSA is mapped to its raw attribute (for lifting into
    # downstream move.* attributes).
    assert rule.ssa_to_attr[cl.result] == addr
```

- [ ] **Step 2: Run tests**

```bash
uv run pytest python/tests/rewrite/test_lower_stack_move.py -v
```

Expected: FAIL.

- [ ] **Step 3: Write implementation**

Add to `LowerStackMove`. Each handler takes the source statement and the `to_delete` list passed down from `rewrite_Block`; emission is via `insert_before` on the original statement:

```python
    def _rewrite_ConstFloat(self, stmt: stack_move.ConstFloat, to_delete: list) -> None:
        from kirin.dialects import py
        out = py.Constant(stmt.value)
        out.insert_before(stmt)
        self.ssa_to_target[stmt.result] = out.result
        self.ssa_to_attr[stmt.result] = stmt.value
        to_delete.append(stmt)

    def _rewrite_ConstInt(self, stmt: stack_move.ConstInt, to_delete: list) -> None:
        from kirin.dialects import py
        out = py.Constant(stmt.value)
        out.insert_before(stmt)
        self.ssa_to_target[stmt.result] = out.result
        self.ssa_to_attr[stmt.result] = stmt.value
        to_delete.append(stmt)

    def _rewrite_ConstLoc(self, stmt: stack_move.ConstLoc, to_delete: list) -> None:
        # Address constants stay as decoder attributes — downstream move.*
        # statements take them as attribute values, not SSA operands.
        # We track the raw attribute value for later attribute lifting.
        self.ssa_to_attr[stmt.result] = stmt.value
        to_delete.append(stmt)

    def _rewrite_ConstLane(self, stmt: stack_move.ConstLane, to_delete: list) -> None:
        self.ssa_to_attr[stmt.result] = stmt.value
        to_delete.append(stmt)

    def _rewrite_ConstZone(self, stmt: stack_move.ConstZone, to_delete: list) -> None:
        self.ssa_to_attr[stmt.result] = stmt.value
        to_delete.append(stmt)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest python/tests/rewrite/test_lower_stack_move.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/rewrite/lower_stack_move.py python/tests/rewrite/test_lower_stack_move.py
git commit -m "feat(rewrite): lower_stack_move handles constants"
```

---

### Task D3: Rewrite stack ops (collapse)

**Files:**
- Modify: `python/bloqade/lanes/rewrite/lower_stack_move.py`
- Modify: `python/tests/rewrite/test_lower_stack_move.py`

- [ ] **Step 1: Write failing tests**

Append:

```python
def test_pop_is_dropped():
    cf = stack_move.ConstFloat(value=1.0)
    pop = stack_move.Pop(value=cf.result)
    block = _build_stack_move_block([cf, pop, stack_move.Return()])
    Walk(LowerStackMove()).rewrite(block)
    # No target statement for Pop, and the original stack_move.Pop is gone.
    assert not any(isinstance(s, stack_move.Pop) for s in block.stmts)


def test_dup_maps_result_to_input():
    cf = stack_move.ConstFloat(value=1.0)
    dup = stack_move.Dup(value=cf.result)
    block = _build_stack_move_block([cf, dup, stack_move.Return()])
    rule = LowerStackMove()
    Walk(rule).rewrite(block)
    # Dup result shares attribute value with its input.
    assert rule.ssa_to_attr[dup.result] == rule.ssa_to_attr[cf.result]


def test_swap_permutes_mappings():
    a = stack_move.ConstInt(value=1)
    b = stack_move.ConstInt(value=2)
    sw = stack_move.Swap(in_top=b.result, in_bot=a.result)
    block = _build_stack_move_block([a, b, sw, stack_move.Return()])
    rule = LowerStackMove()
    Walk(rule).rewrite(block)
    # out_top ≡ in_bot (=a); out_bot ≡ in_top (=b).
    assert rule.ssa_to_attr[sw.out_top] == 1  # a's value
    assert rule.ssa_to_attr[sw.out_bot] == 2  # b's value
```

- [ ] **Step 2: Run tests**

```bash
uv run pytest python/tests/rewrite/test_lower_stack_move.py -v
```

Expected: FAIL.

- [ ] **Step 3: Write implementation**

Add to `LowerStackMove`:

```python
    def _alias(self, dst: ir.SSAValue, src: ir.SSAValue) -> None:
        """Make dst resolve to the same target/attribute entries as src."""
        if src in self.ssa_to_target:
            self.ssa_to_target[dst] = self.ssa_to_target[src]
        if src in self.ssa_to_attr:
            self.ssa_to_attr[dst] = self.ssa_to_attr[src]

    def _rewrite_Pop(self, stmt: stack_move.Pop, to_delete: list) -> None:
        # Pop collapses — no target emission. The popped SSA value is either
        # dead (will be DCE'd) or still referenced elsewhere.
        to_delete.append(stmt)

    def _rewrite_Dup(self, stmt: stack_move.Dup, to_delete: list) -> None:
        self._alias(stmt.result, stmt.value)
        to_delete.append(stmt)

    def _rewrite_Swap(self, stmt: stack_move.Swap, to_delete: list) -> None:
        self._alias(stmt.out_top, stmt.in_bot)
        self._alias(stmt.out_bot, stmt.in_top)
        to_delete.append(stmt)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest python/tests/rewrite/test_lower_stack_move.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/rewrite/lower_stack_move.py python/tests/rewrite/test_lower_stack_move.py
git commit -m "feat(rewrite): lower_stack_move collapses Pop/Dup/Swap"
```

---

### Task D4: Rewrite atom operations with attribute lifting + state threading

**Files:**
- Modify: `python/bloqade/lanes/rewrite/lower_stack_move.py`
- Modify: `python/tests/rewrite/test_lower_stack_move.py`

- [ ] **Step 1: Write failing tests**

Append:

```python
def test_fill_lowers_to_move_fill_with_attribute_locations():
    a0 = LocationAddress(0, 0, 0)
    a1 = LocationAddress(0, 0, 1)
    cl0 = stack_move.ConstLoc(value=a0)
    cl1 = stack_move.ConstLoc(value=a1)
    fill = stack_move.Fill(locations=(cl0.result, cl1.result))
    block = _build_stack_move_block([cl0, cl1, fill, stack_move.Return()])
    Walk(LowerStackMove()).rewrite(block)
    mf = next(s for s in block.stmts if isinstance(s, move.Fill))
    assert mf.location_addresses == (a0, a1)
```

- [ ] **Step 2: Run test**

```bash
uv run pytest python/tests/rewrite/test_lower_stack_move.py::test_fill_lowers_to_move_fill_with_attribute_locations -v
```

Expected: FAIL.

- [ ] **Step 3: Write implementation**

Add to `LowerStackMove`:

```python
    def _lift_attrs(self, ssa_values: tuple[ir.SSAValue, ...]) -> tuple:
        """Resolve each stack_move SSA value back to its original attribute
        (traced through Dup / Swap). Raises if a value isn't attribute-backed."""
        out = []
        for v in ssa_values:
            if v not in self.ssa_to_attr:
                raise RuntimeError(
                    f"no attribute mapping for {v}: operand must trace back "
                    f"to a Const* statement"
                )
            out.append(self.ssa_to_attr[v])
        return tuple(out)

    def _rewrite_InitialFill(self, stmt: stack_move.InitialFill, to_delete: list) -> None:
        addrs = self._lift_attrs(stmt.locations)
        new = move.InitialFill(self.state, location_addresses=addrs)
        new.insert_before(stmt)
        self.state = new.result
        to_delete.append(stmt)

    def _rewrite_Fill(self, stmt: stack_move.Fill, to_delete: list) -> None:
        addrs = self._lift_attrs(stmt.locations)
        new = move.Fill(self.state, location_addresses=addrs)
        new.insert_before(stmt)
        self.state = new.result
        to_delete.append(stmt)

    def _rewrite_Move(self, stmt: stack_move.Move, to_delete: list) -> None:
        lanes = self._lift_attrs(stmt.lanes)
        new = move.Move(self.state, lanes=lanes)
        new.insert_before(stmt)
        self.state = new.result
        to_delete.append(stmt)
```

Note: the exact keyword arguments (`location_addresses=`, `lanes=`, etc.) for existing `move.*` statement constructors match what's already in `move.py`. If signatures differ, adjust to match.

- [ ] **Step 4: Run test**

```bash
uv run pytest python/tests/rewrite/test_lower_stack_move.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/rewrite/lower_stack_move.py python/tests/rewrite/test_lower_stack_move.py
git commit -m "feat(rewrite): lower_stack_move lifts Fill/Move/InitialFill with state threading"
```

---

### Task D5: Rewrite gates

**Files:**
- Modify: `python/bloqade/lanes/rewrite/lower_stack_move.py`
- Modify: `python/tests/rewrite/test_lower_stack_move.py`

- [ ] **Step 1: Write failing tests**

Append a test for `LocalR` that asserts the emitted `move.LocalR` has the correct `phi`, `theta`, and `location_addresses` attributes, derived from the upstream `ConstFloat`/`ConstLoc` statements. Patterns mirror `test_fill_lowers_to_move_fill_with_attribute_locations`.

```python
def test_local_r_lowers_with_attribute_lifting():
    cf_theta = stack_move.ConstFloat(value=0.1)
    cf_phi = stack_move.ConstFloat(value=0.2)
    cl = stack_move.ConstLoc(value=LocationAddress(0, 0, 0))
    lr = stack_move.LocalR(
        phi=cf_phi.result,
        theta=cf_theta.result,
        locations=(cl.result,),
    )
    block = _build_stack_move_block([cf_theta, cf_phi, cl, lr, stack_move.Return()])
    Walk(LowerStackMove()).rewrite(block)
    mr = next(s for s in block.stmts if isinstance(s, move.LocalR))
    assert mr.phi == 0.2
    assert mr.theta == 0.1
    assert mr.location_addresses == (LocationAddress(0, 0, 0),)


def test_cz_lowers_with_attribute_zone():
    from bloqade.lanes.bytecode import ZoneAddress
    cz_zone = stack_move.ConstZone(value=ZoneAddress(0))
    cz = stack_move.CZ(zone=cz_zone.result)
    block = _build_stack_move_block([cz_zone, cz, stack_move.Return()])
    Walk(LowerStackMove()).rewrite(block)
    mcz = next(s for s in block.stmts if isinstance(s, move.CZ))
    assert mcz.zone_address == ZoneAddress(0)
```

- [ ] **Step 2: Run tests**

```bash
uv run pytest python/tests/rewrite/test_lower_stack_move.py -v
```

Expected: FAIL.

- [ ] **Step 3: Write implementation**

Add to `LowerStackMove`:

```python
    def _rewrite_LocalR(self, stmt: stack_move.LocalR, to_delete: list) -> None:
        (phi,) = self._lift_attrs((stmt.phi,))
        (theta,) = self._lift_attrs((stmt.theta,))
        addrs = self._lift_attrs(stmt.locations)
        new = move.LocalR(
            self.state, phi=phi, theta=theta, location_addresses=addrs,
        )
        new.insert_before(stmt)
        self.state = new.result
        to_delete.append(stmt)

    def _rewrite_LocalRz(self, stmt: stack_move.LocalRz, to_delete: list) -> None:
        (theta,) = self._lift_attrs((stmt.theta,))
        addrs = self._lift_attrs(stmt.locations)
        new = move.LocalRz(self.state, theta=theta, location_addresses=addrs)
        new.insert_before(stmt)
        self.state = new.result
        to_delete.append(stmt)

    def _rewrite_GlobalR(self, stmt: stack_move.GlobalR, to_delete: list) -> None:
        (phi,) = self._lift_attrs((stmt.phi,))
        (theta,) = self._lift_attrs((stmt.theta,))
        new = move.GlobalR(self.state, phi=phi, theta=theta)
        new.insert_before(stmt)
        self.state = new.result
        to_delete.append(stmt)

    def _rewrite_GlobalRz(self, stmt: stack_move.GlobalRz, to_delete: list) -> None:
        (theta,) = self._lift_attrs((stmt.theta,))
        new = move.GlobalRz(self.state, theta=theta)
        new.insert_before(stmt)
        self.state = new.result
        to_delete.append(stmt)

    def _rewrite_CZ(self, stmt: stack_move.CZ, to_delete: list) -> None:
        (zone,) = self._lift_attrs((stmt.zone,))
        new = move.CZ(self.state, zone_address=zone)
        new.insert_before(stmt)
        self.state = new.result
        to_delete.append(stmt)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest python/tests/rewrite/test_lower_stack_move.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/rewrite/lower_stack_move.py python/tests/rewrite/test_lower_stack_move.py
git commit -m "feat(rewrite): lower_stack_move handles gate statements"
```

---

### Task D6: Rewrite Measure — dedup zones + emit new `move.Measure`

**Files:**
- Modify: `python/bloqade/lanes/rewrite/lower_stack_move.py`
- Modify: `python/tests/rewrite/test_lower_stack_move.py`

- [ ] **Step 1: Write failing tests**

Append:

```python
def test_measure_single_zone_emits_single_zone_measure():
    cl0 = stack_move.ConstLoc(value=LocationAddress(0, 0, 0))
    cl1 = stack_move.ConstLoc(value=LocationAddress(0, 0, 1))
    m = stack_move.Measure(locations=(cl0.result, cl1.result))
    block = _build_stack_move_block([cl0, cl1, m, stack_move.Return()])
    Walk(LowerStackMove()).rewrite(block)
    mm = next(s for s in block.stmts if isinstance(s, move.Measure))
    # One zone (both locs are in zone 0).
    assert len(mm.zones) == 1


def test_measure_multi_zone_dedups():
    # Two locations in zone 0, one in zone 1. Expect 2 zone SSA values.
    cl0 = stack_move.ConstLoc(value=LocationAddress(0, 0, 0))
    cl1 = stack_move.ConstLoc(value=LocationAddress(1, 0, 0))
    cl2 = stack_move.ConstLoc(value=LocationAddress(0, 0, 1))
    m = stack_move.Measure(locations=(cl0.result, cl1.result, cl2.result))
    block = _build_stack_move_block([cl0, cl1, cl2, m, stack_move.Return()])
    Walk(LowerStackMove()).rewrite(block)
    mm = next(s for s in block.stmts if isinstance(s, move.Measure))
    assert len(mm.zones) == 2
```

- [ ] **Step 2: Run tests**

```bash
uv run pytest python/tests/rewrite/test_lower_stack_move.py -v
```

Expected: FAIL.

- [ ] **Step 3: Write implementation**

Add to `LowerStackMove`:

```python
    def _rewrite_Measure(self, stmt: stack_move.Measure, to_delete: list) -> None:
        # Lift each location to its LocationAddress, extract distinct zone ids,
        # synthesise move.ConstZone per distinct zone, emit move.Measure(...).
        from bloqade.lanes.bytecode import ZoneAddress
        locs = self._lift_attrs(stmt.locations)
        seen_zone_ids: list[int] = []
        for loc in locs:
            if loc.zone_id not in seen_zone_ids:
                seen_zone_ids.append(loc.zone_id)
        zone_ssa: list[ir.SSAValue] = []
        for zid in seen_zone_ids:
            cz = move.ConstZone(value=ZoneAddress(zid))
            cz.insert_before(stmt)
            zone_ssa.append(cz.result)
        new = move.Measure(self.state, zones=tuple(zone_ssa))
        new.insert_before(stmt)
        self.state = new.result
        self.ssa_to_target[stmt.result] = new.result
        to_delete.append(stmt)
```

Note: this assumes a `move.ConstZone(value=ZoneAddress)` statement exists in `move.py`. If not, either add one in this task (it's trivial — attribute `value: ZoneAddress`, result of type `ZoneAddressType`) or use an inline `ZoneAddress` attribute in the `Measure` constructor. Check `move.py` and adjust.

- [ ] **Step 4: Run tests**

```bash
uv run pytest python/tests/rewrite/test_lower_stack_move.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/rewrite/lower_stack_move.py python/tests/rewrite/test_lower_stack_move.py
git commit -m "feat(rewrite): lower_stack_move deduplicates zones for Measure"
```

---

### Task D7: Rewrite AwaitMeasure, arrays, annotations

**Files:**
- Modify: `python/bloqade/lanes/rewrite/lower_stack_move.py`
- Modify: `python/tests/rewrite/test_lower_stack_move.py`

- [ ] **Step 1: Write failing tests**

Append:

```python
def test_await_measure_lowers_without_error():
    # Smoke: await_measure after measure lowers cleanly. AwaitMeasure is
    # pure synchronisation in stack_move — no target-dialect emission.
    cl = stack_move.ConstLoc(value=LocationAddress(0, 0, 0))
    m = stack_move.Measure(locations=(cl.result,))
    aw = stack_move.AwaitMeasure(future=m.result)
    block = _build_stack_move_block([cl, m, aw, stack_move.Return()])
    Walk(LowerStackMove()).rewrite(block)  # should not raise


def test_new_array_lowers_to_ilist_new():
    from kirin.dialects import ilist
    na = stack_move.NewArray(type_tag=0, dim0=4, dim1=0)
    block = _build_stack_move_block([na, stack_move.Return()])
    Walk(LowerStackMove()).rewrite(block)
    assert any(isinstance(s, ilist.New) for s in block.stmts)


def test_set_detector_lowers_to_annotate():
    from bloqade.decoders.dialects import annotate
    na = stack_move.NewArray(type_tag=0, dim0=1, dim1=0)
    sd = stack_move.SetDetector(array=na.result)
    block = _build_stack_move_block([na, sd, stack_move.Return()])
    Walk(LowerStackMove()).rewrite(block)
    assert any(isinstance(s, annotate.SetDetector) for s in block.stmts)


def test_set_observable_lowers_to_annotate():
    from bloqade.decoders.dialects import annotate
    na = stack_move.NewArray(type_tag=0, dim0=1, dim1=0)
    so = stack_move.SetObservable(array=na.result)
    block = _build_stack_move_block([na, so, stack_move.Return()])
    Walk(LowerStackMove()).rewrite(block)
    assert any(isinstance(s, annotate.SetObservable) for s in block.stmts)
```

- [ ] **Step 2: Run tests**

```bash
uv run pytest python/tests/rewrite/test_lower_stack_move.py -v
```

Expected: FAIL.

- [ ] **Step 3: Write implementation**

Add to `LowerStackMove`. Exact target-dialect construction arguments must match what the existing `ilist.New`, `py.indexing.GetItem`, `annotate.SetDetector`, and `annotate.SetObservable` constructors expect; check the Kirin / bloqade-decoders source.

```python
    def _rewrite_AwaitMeasure(self, stmt: stack_move.AwaitMeasure, to_delete: list) -> None:
        # AwaitMeasure is pure synchronisation in stack_move (no result).
        # In the existing move pipeline, measurement values are extracted
        # via GetFutureResult per (zone, location); for v1 we emit nothing
        # here, and any downstream GetItem on the future is handled in
        # _rewrite_GetItem below. Adjust if AwaitMeasure actually needs a
        # target emission (e.g. a barrier or fence) per the Rust source.
        to_delete.append(stmt)

    def _rewrite_NewArray(self, stmt: stack_move.NewArray, to_delete: list) -> None:
        from kirin.dialects import ilist
        if stmt.dim1 == 0:
            new = ilist.New(values=())  # 1-D empty; may need different signature
        else:
            new = ilist.New(values=())  # 2-D stub
        new.insert_before(stmt)
        self.ssa_to_target[stmt.result] = new.result
        to_delete.append(stmt)

    def _rewrite_GetItem(self, stmt: stack_move.GetItem, to_delete: list) -> None:
        from kirin.dialects.py import indexing
        current = self.ssa_to_target[stmt.array]
        for idx_ssa in stmt.indices:
            target_idx = self.ssa_to_target[idx_ssa]
            gi = indexing.GetItem(obj=current, index=target_idx)
            gi.insert_before(stmt)
            current = gi.result
        self.ssa_to_target[stmt.result] = current
        to_delete.append(stmt)

    def _rewrite_SetDetector(self, stmt: stack_move.SetDetector, to_delete: list) -> None:
        from kirin.dialects import ilist
        from bloqade.decoders.dialects import annotate
        measurements = self.ssa_to_target[stmt.array]
        # Coordinates default to empty per the spec — decoded bytecode
        # doesn't carry visualisation metadata.
        empty_coords = ilist.New(values=())
        empty_coords.insert_before(stmt)
        new = annotate.SetDetector(
            measurements=measurements, coordinates=empty_coords.result,
        )
        new.insert_before(stmt)
        self.ssa_to_target[stmt.result] = new.result
        to_delete.append(stmt)

    def _rewrite_SetObservable(self, stmt: stack_move.SetObservable, to_delete: list) -> None:
        from bloqade.decoders.dialects import annotate
        measurements = self.ssa_to_target[stmt.array]
        new = annotate.SetObservable(measurements=measurements)
        new.insert_before(stmt)
        self.ssa_to_target[stmt.result] = new.result
        to_delete.append(stmt)
```

Note: the `ilist.New` and indexing statements' exact constructor signatures need to be verified against the Kirin source (see `python/bloqade/lanes/dialects/move.py` neighbour imports and Kirin's `kirin.dialects.ilist` and `kirin.dialects.py.indexing`). Adjust during implementation.

- [ ] **Step 4: Run tests**

```bash
uv run pytest python/tests/rewrite/test_lower_stack_move.py -v
```

Expected: PASS (you may need to loosen or refine the tests to match the exact target-dialect constructor shapes).

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/rewrite/lower_stack_move.py python/tests/rewrite/test_lower_stack_move.py
git commit -m "feat(rewrite): lower_stack_move handles arrays/annotations/await"
```

---

## Phase E — `AtomAnalysis` Extensions

### Task E1: Track zones + final-measurement count in `AtomAnalysis`

**Files:**
- Modify: `python/bloqade/lanes/analysis/atom/impl.py`
- Modify: `python/tests/analysis/atom/test_atom_interpreter.py`

- [ ] **Step 1: Write failing test**

Append a new test to `python/tests/analysis/atom/test_atom_interpreter.py` that builds a small `move` kernel using the **new** `move.Measure` statement (not `EndMeasure`), runs the `AtomAnalysis`, and asserts:

- the analysis records the zone set for the measurement site;
- the interpreter reports the total final-measurement count for the program.

Example shape (adjust kernel construction to match the existing test style in the file):

```python
def test_atom_interpreter_tracks_measure_zones_and_count(get_arch_spec):
    # Build a @kernel that uses move.Measure instead of move.EndMeasure.
    @kernel
    def main():
        state = move.load()
        state = move.fill(state, location_addresses=(move.LocationAddress(0, 0, 0),))
        future = move.measure(state, zones=(move.ZoneAddress(0),))
        move.store(state)

    interp = atom.AtomInterpreter(main, arch_spec=get_arch_spec())
    frame, _ = interp.run(main)
    assert interp.measure_sites  # non-empty — tracked by the new analysis method
    assert interp.final_measurement_count == 1
```

Note: the exact attribute names (`measure_sites`, `final_measurement_count`) are design choices for this task. Pick readable names; use them consistently in the analysis and the test.

- [ ] **Step 2: Run test**

```bash
uv run pytest python/tests/analysis/atom/test_atom_interpreter.py -v -k measure
```

Expected: FAIL — no `@interp.impl(move.Measure)` method yet.

- [ ] **Step 3: Write implementation**

In `python/bloqade/lanes/analysis/atom/impl.py`, find the existing `@dialect.register(key="atom")` method table (search for `@move.dialect.register`) and add:

```python
    @interp.impl(move.Measure)
    def measure_impl(
        self,
        interp_: AtomInterpreter,
        frame: ForwardFrame[MoveExecution],
        stmt: move.Measure,
    ):
        current_state = frame.get(stmt.current_state)
        interp_.current_state = current_state

        # Track zone set + increment final-measurement count.
        zone_addresses = [
            # Zones here are SSA values — look up their values via the
            # interpreter frame / constant folding.
            frame.get(z_ssa) for z_ssa in stmt.zones
        ]
        interp_.measure_sites.append(
            {"stmt": stmt, "zones": tuple(zone_addresses)}
        )
        interp_.final_measurement_count += 1

        if not isinstance(current_state, AtomState):
            return (MoveExecution.bottom(),)

        # Build the MeasurementFuture by mirroring end_measure_impl: for
        # each zone, walk every location in the zone, and record any qubit
        # currently at that location.
        results: dict[layout.ZoneAddress, dict[layout.LocationAddress, int]] = {}
        for zone_val in zone_addresses:
            # frame.get(ssa) returns the abstract value for the SSA operand;
            # for a move.ConstZone-backed SSA this is the ZoneAddress itself
            # (or a PyAttr wrapping it — unwrap as end_measure_impl does).
            zone = zone_val.unwrap() if hasattr(zone_val, "unwrap") else zone_val
            result = results.setdefault(zone, {})
            for loc_addr in interp_.arch_spec.yield_zone_locations(zone):
                if (qubit_id := current_state.data.get_qubit(loc_addr)) is not None:
                    result[loc_addr] = qubit_id
        return (MeasureFuture(results),)
```

Also add `measure_sites: list[dict] = field(default_factory=list)` and `final_measurement_count: int = 0` to the `AtomInterpreter` dataclass (or equivalent state container), initialised on `run()` entry.

- [ ] **Step 4: Run test**

```bash
uv run pytest python/tests/analysis/atom/test_atom_interpreter.py -v -k measure
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/analysis/atom/impl.py python/tests/analysis/atom/test_atom_interpreter.py
git commit -m "feat(analysis): AtomAnalysis tracks move.Measure zones and count"
```

---

## Phase F — `measure_lower` Rewrite

### Task F1: Skeleton + single-zone invariant check

**Files:**
- Create: `python/bloqade/lanes/rewrite/measure_lower.py`
- Create: `python/tests/rewrite/test_measure_lower.py`

- [ ] **Step 1: Write failing test**

Create `python/tests/rewrite/test_measure_lower.py`:

```python
import pytest
from kirin import ir
from kirin.rewrite import Walk

from bloqade.lanes.dialects import move
from bloqade.lanes.rewrite.measure_lower import MeasureLower, MeasureLowerError


def test_single_zone_measure_rewrites_to_endmeasure():
    state = ir.TestValue()
    zone = ir.TestValue()
    m = move.Measure(current_state=state, zones=(zone,))
    block = ir.Block([m])
    # For this test we mock the analysis result — in real usage MeasureLower
    # is constructed via MeasureLower.from_method which runs AtomAnalysis.
    rule = MeasureLower(zone_sets={m: frozenset({0})}, final_measurement_count=1)
    Walk(rule).rewrite(block)
    # m has been replaced by a move.EndMeasure in place.
    assert not any(isinstance(s, move.Measure) for s in block.stmts)
    assert any(isinstance(s, move.EndMeasure) for s in block.stmts)


def test_multi_zone_measure_raises():
    state = ir.TestValue()
    z0, z1 = ir.TestValue(), ir.TestValue()
    m = move.Measure(current_state=state, zones=(z0, z1))
    block = ir.Block([m])
    rule = MeasureLower(zone_sets={m: frozenset({0, 1})}, final_measurement_count=1)
    with pytest.raises(MeasureLowerError):
        Walk(rule).rewrite(block)
```

- [ ] **Step 2: Run tests**

```bash
uv run pytest python/tests/rewrite/test_measure_lower.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write implementation**

Create `python/bloqade/lanes/rewrite/measure_lower.py`. This rewrite follows Kirin's `RewriteRule` interface with a `rewrite_Statement` handler — each `move.Measure` it encounters is validated and replaced in place via `stmt.replace_by(replacement)`. The existing `python/bloqade/lanes/rewrite/state.py::RewriteLoadStore` is the reference for in-place mutation idioms.

```python
"""measure_lower — validate + rewrite move.Measure to move.EndMeasure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from kirin import ir
from kirin.rewrite.abc import RewriteResult, RewriteRule

from bloqade.lanes.dialects import move


class MeasureLowerError(RuntimeError):
    """Raised when the measure_lower invariants are violated."""


@dataclass
class MeasureLower(RewriteRule):
    """Lower move.Measure stmts to move.EndMeasure.

    Requires the zone set per Measure site (from AtomAnalysis) and the
    program-wide count of final measurements. Enforces:

    1. Each move.Measure covers exactly one zone.
    2. The program contains exactly one final measurement.
    """

    zone_sets: Mapping[move.Measure, frozenset[int]]
    final_measurement_count: int

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        if not isinstance(node, move.Measure):
            return RewriteResult()
        if self.final_measurement_count != 1:
            raise MeasureLowerError(
                f"expected exactly one final measurement, "
                f"found {self.final_measurement_count}"
            )
        zones = self.zone_sets.get(node)
        if zones is None:
            raise MeasureLowerError(f"no analysis result for {node}")
        if len(zones) != 1:
            raise MeasureLowerError(
                f"move.Measure spans {len(zones)} zones; expected exactly 1"
            )
        from bloqade.lanes.bytecode import ZoneAddress
        (zone_id,) = zones
        replacement = move.EndMeasure(
            current_state=node.current_state,
            zone_addresses=(ZoneAddress(zone_id),),
        )
        node.replace_by(replacement)
        return RewriteResult(has_done_something=True)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest python/tests/rewrite/test_measure_lower.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/rewrite/measure_lower.py python/tests/rewrite/test_measure_lower.py
git commit -m "feat(rewrite): measure_lower validates zone + count, rewrites to EndMeasure"
```

---

### Task F2: Wire `MeasureLower` to `AtomAnalysis`

**Files:**
- Modify: `python/bloqade/lanes/rewrite/measure_lower.py`
- Modify: `python/tests/rewrite/test_measure_lower.py`

- [ ] **Step 1: Write failing test**

Append:

```python
def test_measure_lower_runs_analysis_end_to_end(get_arch_spec):
    """End-to-end: build a move-dialect method with a single Measure,
    call MeasureLower.from_method, and assert the Measure was rewritten to
    EndMeasure."""
    from bloqade.lanes._prelude import kernel
    from bloqade.lanes.dialects import move

    @kernel
    def main():
        state = move.load()
        state = move.fill(state, location_addresses=(move.LocationAddress(0, 0, 0),))
        future = move.measure(state, zones=(move.ZoneAddress(0),))
        move.store(state)

    from kirin.rewrite import Walk
    block = main.callable_region.blocks[0]
    rule = MeasureLower.from_method(main, arch_spec=get_arch_spec())
    Walk(rule).rewrite(block)
    assert any(isinstance(s, move.EndMeasure) for s in block.stmts)
    assert not any(isinstance(s, move.Measure) for s in block.stmts)
```

- [ ] **Step 2: Run test**

```bash
uv run pytest python/tests/rewrite/test_measure_lower.py -v -k end_to_end
```

Expected: FAIL.

- [ ] **Step 3: Write implementation**

Add a helper to `MeasureLower` that runs `AtomAnalysis` itself and populates its own inputs:

```python
    @classmethod
    def from_method(cls, method: ir.Method, arch_spec) -> "MeasureLower":
        from bloqade.lanes.analysis.atom import AtomInterpreter
        interp = AtomInterpreter(method, arch_spec=arch_spec)
        frame, _ = interp.run(method)
        zone_sets = {
            site["stmt"]: frozenset(z.zone_id for z in site["zones"])
            for site in interp.measure_sites
        }
        return cls(zone_sets=zone_sets, final_measurement_count=interp.final_measurement_count)
```

- [ ] **Step 4: Run test**

```bash
uv run pytest python/tests/rewrite/test_measure_lower.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/rewrite/measure_lower.py python/tests/rewrite/test_measure_lower.py
git commit -m "feat(rewrite): measure_lower integrates with AtomAnalysis via from_method"
```

---

## Phase G — End-to-End

### Task G1: End-to-end test

**Files:**
- Create: `python/tests/test_stack_move_e2e.py`

- [ ] **Step 1: Write failing test**

Create `python/tests/test_stack_move_e2e.py`:

```python
"""End-to-end: bytecode Program → decode → lower → measure_lower → transversal."""

from bloqade.lanes.bytecode import Instruction, LocationAddress, MoveType, Program, ZoneAddress
from bloqade.lanes.bytecode.lowering import load_program
from bloqade.lanes.rewrite.lower_stack_move import LowerStackMove
from bloqade.lanes.rewrite.measure_lower import MeasureLower


def _build_arch_spec():
    """Minimal arch spec for the end-to-end test.

    Copy the pattern from python/tests/analysis/atom/test_atom_interpreter.py's
    get_arch_spec (pytest fixture) — refactor it out into a shared helper or
    inline the body here. The arch spec needs at least one zone with one
    location matching LocationAddress(0, 0, 0) so the test's initial_fill +
    measure exercise a valid cell.
    """
    from bloqade.lanes.bytecode._native import (
        Grid as RustGrid, LocationAddress as RustLocAddr,
        Mode as RustMode, Zone as RustZone,
    )
    from bloqade.lanes import layout
    from bloqade.lanes.layout import word

    sole_word = word.Word(sites=((0, 0),))
    grid = RustGrid.from_positions([0.0], [0.0])
    zone = RustZone(
        name="test",
        grid=grid,
        site_buses=[],
        word_buses=[],
        words_with_site_buses=[0],
        sites_with_word_buses=[],
    )
    mode = RustMode(name="main", zones=[0], bitstring_order=[RustLocAddr(0, 0, 0)])
    return layout.ArchSpec.from_components(
        words=(sole_word,), zones=(zone,), modes=[mode],
    )


def test_minimal_program_runs_end_to_end():
    prog = Program(
        version=(1, 0),
        instructions=[
            Instruction.const_loc(0, 0, 0),
            Instruction.initial_fill(1),
            Instruction.const_loc(0, 0, 0),
            Instruction.measure(1),
            Instruction.await_measure(),
            Instruction.return_(),
        ],
    )
    method = load_program(prog)
    block = method.callable_region.blocks[0]

    # Lower to the move dialect in place.
    from kirin.rewrite import Walk
    Walk(LowerStackMove()).rewrite(block)

    # Apply measure_lower (also in place).
    arch_spec = _build_arch_spec()
    Walk(MeasureLower.from_method(method, arch_spec)).rewrite(block)

    # Assert: the method now contains move.EndMeasure, not move.Measure.
    from bloqade.lanes.dialects import move
    assert any(isinstance(s, move.EndMeasure) for s in block.stmts)
    assert not any(isinstance(s, move.Measure) for s in block.stmts)
```

- [ ] **Step 2: Run test**

```bash
uv run pytest python/tests/test_stack_move_e2e.py -v
```

Expected: PASS (or FAIL on a concrete bug; debug until it passes).

- [ ] **Step 3: (If failing) fix bugs; retry**

Expected: PASS.

- [ ] **Step 4: Run the full test suite to verify no regressions**

```bash
just coverage
```

Expected: the new tests PASS and all pre-existing tests still PASS.

- [ ] **Step 5: Commit**

```bash
git add python/tests/test_stack_move_e2e.py
git commit -m "test(stack_move): end-to-end bytecode → lowered move IR"
```

---

## Post-implementation

- [ ] Run full linting: `uv run black python && uv run isort python && uv run ruff check python && uv run pyright python`
- [ ] Run full Rust + Python test suite: `just test`
- [ ] Open a PR with the `breaking` label (new dialect is additive but the new `move.Measure` statement may be flagged) and `S-backport` + `backport v0.7` labels if non-breaking
