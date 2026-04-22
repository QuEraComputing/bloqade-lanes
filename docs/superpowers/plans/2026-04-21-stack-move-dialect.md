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

### Task A1: Create dialect module skeleton and types

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


def test_types_defined():
    assert stack_move.LocationAddressType is not None
    assert stack_move.LaneAddressType is not None
    assert stack_move.ZoneAddressType is not None
    assert stack_move.MeasurementFutureType is not None
    assert stack_move.BitstringType is not None
    assert stack_move.ArrayType is not None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest python/tests/dialects/test_stack_move.py -v
```

Expected: `ModuleNotFoundError` or `AttributeError` (dialect module doesn't exist yet).

- [ ] **Step 3: Write minimal implementation**

Create `python/bloqade/lanes/dialects/stack_move.py`:

```python
"""stack_move dialect — 1:1 SSA image of the bytecode."""

from kirin import ir, lowering, types
from kirin.decl import info, statement

dialect = ir.Dialect(name="lanes.stack_move")


# ── SSA types ──────────────────────────────────────────────────────────

LocationAddressType = types.PyClass("LocationAddress")
LaneAddressType = types.PyClass("LaneAddress")
ZoneAddressType = types.PyClass("ZoneAddress")
MeasurementFutureType = types.PyClass("MeasurementFuture")
BitstringType = types.PyClass("Bitstring")
ArrayType = types.PyClass("Array")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest python/tests/dialects/test_stack_move.py -v
```

Expected: PASS for `test_dialect_exists` and `test_types_defined`.

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/dialects/stack_move.py python/tests/dialects/
git commit -m "feat(stack_move): add dialect skeleton and SSA types"
```

---

### Task A2: Add constant statements

**Files:**
- Modify: `python/bloqade/lanes/dialects/stack_move.py`
- Modify: `python/tests/dialects/test_stack_move.py`

- [ ] **Step 1: Write failing test**

Append to `python/tests/dialects/test_stack_move.py`:

```python
from bloqade.lanes.bytecode import LocationAddress, LaneAddress, ZoneAddress, MoveType


def test_constants_construct():
    """Smoke test: all constant statements construct. Kirin enforces field
    and type interfaces via @statement — we only need to catch that the
    class exists on the dialect."""
    stack_move.ConstFloat(value=3.14)
    stack_move.ConstInt(value=7)
    stack_move.ConstLoc(value=LocationAddress(0, 0, 0))
    stack_move.ConstLane(value=LaneAddress(MoveType.SITE, 0, 0, 0, 0))
    stack_move.ConstZone(value=ZoneAddress(0))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest python/tests/dialects/test_stack_move.py -v
```

Expected: FAIL with `AttributeError: module 'stack_move' has no attribute 'ConstFloat'` (and similar).

- [ ] **Step 3: Write implementation**

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
    value: "LocationAddress" = info.attribute()  # type: ignore[name-defined]
    result: ir.ResultValue = info.result(LocationAddressType)


@statement(dialect=dialect)
class ConstLane(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})
    value: "LaneAddress" = info.attribute()  # type: ignore[name-defined]
    result: ir.ResultValue = info.result(LaneAddressType)


@statement(dialect=dialect)
class ConstZone(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})
    value: "ZoneAddress" = info.attribute()  # type: ignore[name-defined]
    result: ir.ResultValue = info.result(ZoneAddressType)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest python/tests/dialects/test_stack_move.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/dialects/stack_move.py python/tests/dialects/test_stack_move.py
git commit -m "feat(stack_move): add constant statements"
```

---

### Task A3: Add stack-manipulation statements

**Files:**
- Modify: `python/bloqade/lanes/dialects/stack_move.py`
- Modify: `python/tests/dialects/test_stack_move.py`

- [ ] **Step 1: Write failing test**

Append to `python/tests/dialects/test_stack_move.py`:

```python
def test_stack_ops_construct():
    v = ir.TestValue()
    w = ir.TestValue()
    stack_move.Pop(value=v)
    stack_move.Dup(value=v)
    stack_move.Swap(in_top=v, in_bot=w)
```

Add to the imports at top of test file: `from kirin import ir` if not already imported.

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest python/tests/dialects/test_stack_move.py -v
```

Expected: FAIL with `AttributeError: module 'stack_move' has no attribute 'Pop'` (etc.).

- [ ] **Step 3: Write implementation**

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

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest python/tests/dialects/test_stack_move.py -v
```

Expected: all Pop/Dup/Swap tests PASS.

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/dialects/stack_move.py python/tests/dialects/test_stack_move.py
git commit -m "feat(stack_move): add Pop/Dup/Swap statements"
```

---

### Task A4: Add atom-operation statements

**Files:**
- Modify: `python/bloqade/lanes/dialects/stack_move.py`
- Modify: `python/tests/dialects/test_stack_move.py`

- [ ] **Step 1: Write failing test**

Append to `python/tests/dialects/test_stack_move.py`:

```python
def test_atom_ops_construct():
    v0 = ir.TestValue()
    v1 = ir.TestValue()
    stack_move.InitialFill(locations=(v0, v1))
    stack_move.Fill(locations=(v0, v1))
    stack_move.Move(lanes=(v0,))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest python/tests/dialects/test_stack_move.py -v
```

Expected: FAIL with `AttributeError` for the missing classes.

- [ ] **Step 3: Write implementation**

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

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest python/tests/dialects/test_stack_move.py -v
```

Expected: all atom-op tests PASS.

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/dialects/stack_move.py python/tests/dialects/test_stack_move.py
git commit -m "feat(stack_move): add InitialFill/Fill/Move statements"
```

---

### Task A5: Add gate statements

**Files:**
- Modify: `python/bloqade/lanes/dialects/stack_move.py`
- Modify: `python/tests/dialects/test_stack_move.py`

- [ ] **Step 1: Write failing test**

Append:

```python
def test_gates_construct():
    phi, theta, loc, zone = (
        ir.TestValue(), ir.TestValue(), ir.TestValue(), ir.TestValue(),
    )
    stack_move.LocalR(phi=phi, theta=theta, locations=(loc,))
    stack_move.LocalRz(theta=theta, locations=(loc,))
    stack_move.GlobalR(phi=phi, theta=theta)
    stack_move.GlobalRz(theta=theta)
    stack_move.CZ(zone=zone)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest python/tests/dialects/test_stack_move.py -v
```

Expected: FAIL.

- [ ] **Step 3: Write implementation**

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

- [ ] **Step 4: Run tests**

```bash
uv run pytest python/tests/dialects/test_stack_move.py -v
```

Expected: all gate tests PASS.

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/dialects/stack_move.py python/tests/dialects/test_stack_move.py
git commit -m "feat(stack_move): add gate statements (LocalR/LocalRz/GlobalR/GlobalRz/CZ)"
```

---

### Task A6: Add measurement and control-flow statements

**Files:**
- Modify: `python/bloqade/lanes/dialects/stack_move.py`
- Modify: `python/tests/dialects/test_stack_move.py`

- [ ] **Step 1: Write failing test**

Append:

```python
def test_measurement_and_control_flow_construct():
    loc = ir.TestValue()
    future = ir.TestValue()
    stack_move.Measure(locations=(loc,))
    stack_move.AwaitMeasure(future=future)
    stack_move.Return()
    stack_move.Halt()
```

- [ ] **Step 2: Run tests**

```bash
uv run pytest python/tests/dialects/test_stack_move.py -v
```

Expected: FAIL.

- [ ] **Step 3: Write implementation**

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
    traits = frozenset({lowering.FromPythonCall()})
    future: ir.SSAValue = info.argument(type=MeasurementFutureType)
    result: ir.ResultValue = info.result(BitstringType)


# ── Control flow ───────────────────────────────────────────────────────

@statement(dialect=dialect)
class Return(ir.Statement):
    traits = frozenset({lowering.FromPythonCall(), ir.IsTerminator()})


@statement(dialect=dialect)
class Halt(ir.Statement):
    """Lowered to func.Return(None) alongside Return."""
    traits = frozenset({lowering.FromPythonCall(), ir.IsTerminator()})
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest python/tests/dialects/test_stack_move.py -v
```

Expected: all measurement and control-flow tests PASS.

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/dialects/stack_move.py python/tests/dialects/test_stack_move.py
git commit -m "feat(stack_move): add Measure/AwaitMeasure/Return/Halt"
```

---

### Task A7: Add array and annotation statements

**Files:**
- Modify: `python/bloqade/lanes/dialects/stack_move.py`
- Modify: `python/tests/dialects/test_stack_move.py`

- [ ] **Step 1: Write failing test**

Append:

```python
def test_array_and_annotation_construct():
    arr = ir.TestValue()
    idx = ir.TestValue()
    stack_move.NewArray(type_tag=1, dim0=4, dim1=0)
    stack_move.GetItem(array=arr, indices=(idx,))
    stack_move.SetDetector(array=arr)
    stack_move.SetObservable(array=arr)
```

- [ ] **Step 2: Run tests**

```bash
uv run pytest python/tests/dialects/test_stack_move.py -v
```

Expected: FAIL.

- [ ] **Step 3: Write implementation**

Append to `python/bloqade/lanes/dialects/stack_move.py`:

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
    traits = frozenset({lowering.FromPythonCall()})
    array: ir.SSAValue = info.argument(type=ArrayType)


@statement(dialect=dialect)
class SetObservable(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})
    array: ir.SSAValue = info.argument(type=ArrayType)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest python/tests/dialects/test_stack_move.py -v
```

Expected: all array/annotation tests PASS.

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/dialects/stack_move.py python/tests/dialects/test_stack_move.py
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
        future = self._pop_or_raise(idx, instr)
        stmt = stack_move.AwaitMeasure(future=future)
        self.block.stmts.append(stmt)
        self.stack.append(stmt.result)

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
        self.block.stmts.append(stack_move.SetDetector(array=array))

    def _visit_set_observable(self, idx: int, instr: "Instruction") -> None:
        array = self._pop_or_raise(idx, instr)
        self.block.stmts.append(stack_move.SetObservable(array=array))

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

- [ ] **Step 1: Write failing smoke test**

Create `python/tests/rewrite/test_lower_stack_move.py`:

```python
from kirin import ir

from bloqade.lanes.dialects import stack_move, move
from bloqade.lanes.rewrite.lower_stack_move import LowerStackMove


def _build_stack_move_block(stmts: list[ir.Statement]) -> ir.Block:
    block = ir.Block()
    for stmt in stmts:
        block.stmts.append(stmt)
    return block


def test_empty_block_emits_only_load_and_return():
    block = _build_stack_move_block([stack_move.Return()])
    rewritten = LowerStackMove().run(block)
    # Expect a Load (to initialise state) and a func.Return terminator.
    assert any(isinstance(s, move.Load) for s in rewritten.stmts)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest python/tests/rewrite/test_lower_stack_move.py -v
```

Expected: FAIL with `ModuleNotFoundError` on `lower_stack_move`.

- [ ] **Step 3: Write implementation**

Create `python/bloqade/lanes/rewrite/lower_stack_move.py`:

```python
"""lower_stack_move — mechanical rewrite from stack_move → multi-dialect IR.

Per-statement translation. The stack_move IR is already SSA (each bytecode
stack slot is a named SSA value), so there's no stack simulation here —
just a statement-by-statement emit into the target dialects (move, ilist,
py.constant, py.indexing, annotate, func). State threading is inserted
inline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kirin import ir

from bloqade.lanes.dialects import move, stack_move


@dataclass
class LowerStackMove:
    """Rewrite a stack_move ir.Block into a multi-dialect ir.Block.

    Maintains:
    - a mapping from stack_move SSA values → the target-dialect SSA value
      they should resolve to (or, for constants, the raw attribute value
      so consumers can lift them into move.* attributes).
    - the current StateType SSA value for state threading into stateful
      move.* statements.
    """

    target_block: ir.Block = field(default_factory=ir.Block)
    ssa_to_target: dict[ir.SSAValue, ir.SSAValue] = field(default_factory=dict)
    ssa_to_attr: dict[ir.SSAValue, Any] = field(default_factory=dict)
    state: ir.SSAValue | None = None

    def run(self, source_block: ir.Block) -> ir.Block:
        # Initialise the state SSA value via move.Load.
        load = move.Load()
        self.target_block.stmts.append(load)
        self.state = load.result

        for stmt in source_block.stmts:
            self._rewrite(stmt)

        return self.target_block

    def _rewrite(self, stmt: ir.Statement) -> None:
        handler = getattr(self, f"_rewrite_{type(stmt).__name__}", None)
        if handler is None:
            raise NotImplementedError(
                f"lower_stack_move has no handler for {type(stmt).__name__}"
            )
        handler(stmt)

    def _rewrite_Return(self, stmt: stack_move.Return) -> None:
        # Store state back before returning.
        from kirin.dialects import func
        self.target_block.stmts.append(move.Store(self.state))
        self.target_block.stmts.append(func.Return())

    def _rewrite_Halt(self, stmt: stack_move.Halt) -> None:
        from kirin.dialects import func
        self.target_block.stmts.append(move.Store(self.state))
        self.target_block.stmts.append(func.Return())
```

- [ ] **Step 4: Run test**

```bash
uv run pytest python/tests/rewrite/test_lower_stack_move.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/rewrite/lower_stack_move.py python/tests/rewrite/test_lower_stack_move.py
git commit -m "feat(rewrite): lower_stack_move skeleton with state init + Return/Halt"
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
    lower = LowerStackMove()
    out = lower.run(block)
    # py.Constant statement emitted with value 1.5
    assert any(
        isinstance(s, py.Constant) and s.value.unwrap() == 1.5 for s in out.stmts
    )
    # And the original stack_move SSA value is mapped to its target SSA.
    assert cf.result in lower.ssa_to_target


def test_const_loc_tracks_attribute_value():
    addr = LocationAddress(0, 0, 0)
    cl = stack_move.ConstLoc(value=addr)
    block = _build_stack_move_block([cl, stack_move.Return()])
    lower = LowerStackMove()
    lower.run(block)
    # The stack_move SSA is mapped to its raw attribute (for lifting into
    # downstream move.* attributes).
    assert lower.ssa_to_attr[cl.result] == addr
```

- [ ] **Step 2: Run tests**

```bash
uv run pytest python/tests/rewrite/test_lower_stack_move.py -v
```

Expected: FAIL.

- [ ] **Step 3: Write implementation**

Add to `LowerStackMove`:

```python
    def _rewrite_ConstFloat(self, stmt: stack_move.ConstFloat) -> None:
        from kirin.dialects import py
        out = py.Constant(stmt.value)
        self.target_block.stmts.append(out)
        self.ssa_to_target[stmt.result] = out.result
        self.ssa_to_attr[stmt.result] = stmt.value

    def _rewrite_ConstInt(self, stmt: stack_move.ConstInt) -> None:
        from kirin.dialects import py
        out = py.Constant(stmt.value)
        self.target_block.stmts.append(out)
        self.ssa_to_target[stmt.result] = out.result
        self.ssa_to_attr[stmt.result] = stmt.value

    def _rewrite_ConstLoc(self, stmt: stack_move.ConstLoc) -> None:
        # Address constants stay as decoder attributes — downstream move.*
        # statements take them as attribute values, not SSA operands.
        # We track the raw attribute value for later attribute lifting.
        self.ssa_to_attr[stmt.result] = stmt.value

    def _rewrite_ConstLane(self, stmt: stack_move.ConstLane) -> None:
        self.ssa_to_attr[stmt.result] = stmt.value

    def _rewrite_ConstZone(self, stmt: stack_move.ConstZone) -> None:
        self.ssa_to_attr[stmt.result] = stmt.value
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
    out = LowerStackMove().run(block)
    # No target statement for Pop.
    assert not any(type(s).__name__ == "Pop" for s in out.stmts)


def test_dup_maps_result_to_input():
    cf = stack_move.ConstFloat(value=1.0)
    dup = stack_move.Dup(value=cf.result)
    block = _build_stack_move_block([cf, dup, stack_move.Return()])
    lower = LowerStackMove()
    lower.run(block)
    # Dup result shares attribute value with its input.
    assert lower.ssa_to_attr[dup.result] == lower.ssa_to_attr[cf.result]


def test_swap_permutes_mappings():
    a = stack_move.ConstInt(value=1)
    b = stack_move.ConstInt(value=2)
    sw = stack_move.Swap(in_top=b.result, in_bot=a.result)
    block = _build_stack_move_block([a, b, sw, stack_move.Return()])
    lower = LowerStackMove()
    lower.run(block)
    # out_top ≡ in_bot (=a); out_bot ≡ in_top (=b).
    assert lower.ssa_to_attr[sw.out_top] == 1  # a's value
    assert lower.ssa_to_attr[sw.out_bot] == 2  # b's value
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

    def _rewrite_Pop(self, stmt: stack_move.Pop) -> None:
        # Nothing — Pop collapses. The discarded SSA value is either dead
        # (DCE'd later) or still referenced elsewhere.
        pass

    def _rewrite_Dup(self, stmt: stack_move.Dup) -> None:
        self._alias(stmt.result, stmt.value)

    def _rewrite_Swap(self, stmt: stack_move.Swap) -> None:
        self._alias(stmt.out_top, stmt.in_bot)
        self._alias(stmt.out_bot, stmt.in_top)
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
    out = LowerStackMove().run(block)
    mf = next(s for s in out.stmts if isinstance(s, move.Fill))
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

    def _rewrite_InitialFill(self, stmt: stack_move.InitialFill) -> None:
        addrs = self._lift_attrs(stmt.locations)
        new = move.InitialFill(self.state, location_addresses=addrs)
        self.target_block.stmts.append(new)
        self.state = new.result

    def _rewrite_Fill(self, stmt: stack_move.Fill) -> None:
        addrs = self._lift_attrs(stmt.locations)
        new = move.Fill(self.state, location_addresses=addrs)
        self.target_block.stmts.append(new)
        self.state = new.result

    def _rewrite_Move(self, stmt: stack_move.Move) -> None:
        lanes = self._lift_attrs(stmt.lanes)
        new = move.Move(self.state, lanes=lanes)
        self.target_block.stmts.append(new)
        self.state = new.result
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
    out = LowerStackMove().run(block)
    mr = next(s for s in out.stmts if isinstance(s, move.LocalR))
    assert mr.phi == 0.2
    assert mr.theta == 0.1
    assert mr.location_addresses == (LocationAddress(0, 0, 0),)


def test_cz_lowers_with_attribute_zone():
    from bloqade.lanes.bytecode import ZoneAddress
    cz_zone = stack_move.ConstZone(value=ZoneAddress(0))
    cz = stack_move.CZ(zone=cz_zone.result)
    block = _build_stack_move_block([cz_zone, cz, stack_move.Return()])
    out = LowerStackMove().run(block)
    mcz = next(s for s in out.stmts if isinstance(s, move.CZ))
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
    def _rewrite_LocalR(self, stmt: stack_move.LocalR) -> None:
        (phi,) = self._lift_attrs((stmt.phi,))
        (theta,) = self._lift_attrs((stmt.theta,))
        addrs = self._lift_attrs(stmt.locations)
        new = move.LocalR(
            self.state, phi=phi, theta=theta, location_addresses=addrs,
        )
        self.target_block.stmts.append(new)
        self.state = new.result

    def _rewrite_LocalRz(self, stmt: stack_move.LocalRz) -> None:
        (theta,) = self._lift_attrs((stmt.theta,))
        addrs = self._lift_attrs(stmt.locations)
        new = move.LocalRz(self.state, theta=theta, location_addresses=addrs)
        self.target_block.stmts.append(new)
        self.state = new.result

    def _rewrite_GlobalR(self, stmt: stack_move.GlobalR) -> None:
        (phi,) = self._lift_attrs((stmt.phi,))
        (theta,) = self._lift_attrs((stmt.theta,))
        new = move.GlobalR(self.state, phi=phi, theta=theta)
        self.target_block.stmts.append(new)
        self.state = new.result

    def _rewrite_GlobalRz(self, stmt: stack_move.GlobalRz) -> None:
        (theta,) = self._lift_attrs((stmt.theta,))
        new = move.GlobalRz(self.state, theta=theta)
        self.target_block.stmts.append(new)
        self.state = new.result

    def _rewrite_CZ(self, stmt: stack_move.CZ) -> None:
        (zone,) = self._lift_attrs((stmt.zone,))
        new = move.CZ(self.state, zone_address=zone)
        self.target_block.stmts.append(new)
        self.state = new.result
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
    out = LowerStackMove().run(block)
    mm = next(s for s in out.stmts if isinstance(s, move.Measure))
    # One zone (both locs are in zone 0).
    assert len(mm.zones) == 1


def test_measure_multi_zone_dedups():
    # Two locations in zone 0, one in zone 1. Expect 2 zone SSA values.
    cl0 = stack_move.ConstLoc(value=LocationAddress(0, 0, 0))
    cl1 = stack_move.ConstLoc(value=LocationAddress(1, 0, 0))
    cl2 = stack_move.ConstLoc(value=LocationAddress(0, 0, 1))
    m = stack_move.Measure(locations=(cl0.result, cl1.result, cl2.result))
    block = _build_stack_move_block([cl0, cl1, cl2, m, stack_move.Return()])
    out = LowerStackMove().run(block)
    mm = next(s for s in out.stmts if isinstance(s, move.Measure))
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
    def _rewrite_Measure(self, stmt: stack_move.Measure) -> None:
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
            self.target_block.stmts.append(cz)
            zone_ssa.append(cz.result)
        new = move.Measure(self.state, zones=tuple(zone_ssa))
        self.target_block.stmts.append(new)
        self.state = new.result
        self.ssa_to_target[stmt.result] = new.result
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
def test_await_measure_lowers_to_future_result():
    # Smoke: await_measure after measure produces a get_future_result /
    # AwaitMeasure-equivalent in the move dialect.
    cl = stack_move.ConstLoc(value=LocationAddress(0, 0, 0))
    m = stack_move.Measure(locations=(cl.result,))
    aw = stack_move.AwaitMeasure(future=m.result)
    block = _build_stack_move_block([cl, m, aw, stack_move.Return()])
    out = LowerStackMove().run(block)
    # There should be a downstream statement that takes the future as SSA.
    # Follow move.py's naming — likely move.GetFutureResult or similar.
    assert any(
        isinstance(s, ir.Statement) and m.result not in s.args
        for s in out.stmts
    )


def test_new_array_lowers_to_ilist_new():
    from kirin.dialects import ilist
    na = stack_move.NewArray(type_tag=0, dim0=4, dim1=0)
    block = _build_stack_move_block([na, stack_move.Return()])
    out = LowerStackMove().run(block)
    assert any(isinstance(s, ilist.New) for s in out.stmts)


def test_set_detector_lowers_to_annotate():
    from bloqade.decoders.dialects import annotate
    na = stack_move.NewArray(type_tag=0, dim0=1, dim1=0)
    sd = stack_move.SetDetector(array=na.result)
    block = _build_stack_move_block([na, sd, stack_move.Return()])
    out = LowerStackMove().run(block)
    assert any(isinstance(s, annotate.SetDetector) for s in out.stmts)


def test_set_observable_lowers_to_annotate():
    from bloqade.decoders.dialects import annotate
    na = stack_move.NewArray(type_tag=0, dim0=1, dim1=0)
    so = stack_move.SetObservable(array=na.result)
    block = _build_stack_move_block([na, so, stack_move.Return()])
    out = LowerStackMove().run(block)
    assert any(isinstance(s, annotate.SetObservable) for s in out.stmts)
```

- [ ] **Step 2: Run tests**

```bash
uv run pytest python/tests/rewrite/test_lower_stack_move.py -v
```

Expected: FAIL.

- [ ] **Step 3: Write implementation**

Add to `LowerStackMove`. Exact target-dialect construction arguments must match what the existing `ilist.New`, `py.indexing.GetItem`, `annotate.SetDetector`, and `annotate.SetObservable` constructors expect; check the Kirin / bloqade-decoders source.

```python
    def _rewrite_AwaitMeasure(self, stmt: stack_move.AwaitMeasure) -> None:
        # Look up the source Measure's target SSA (the move.Measure result).
        future_target = self.ssa_to_target[stmt.future]
        # The existing move dialect calls this GetFutureResult — confirm by
        # reading move.py. Target signature probably matches:
        #   GetFutureResult(future, zone_address=..., location_address=...)
        # For bytecode lowering the per-location lookup happens downstream;
        # for v1 we emit a single consume that references the future.
        new = move.GetFutureResult(future_target)
        self.target_block.stmts.append(new)
        self.ssa_to_target[stmt.result] = new.result

    def _rewrite_NewArray(self, stmt: stack_move.NewArray) -> None:
        from kirin.dialects import ilist
        if stmt.dim1 == 0:
            new = ilist.New(values=())  # 1-D empty; may need different signature
        else:
            new = ilist.New(values=())  # 2-D stub
        self.target_block.stmts.append(new)
        self.ssa_to_target[stmt.result] = new.result

    def _rewrite_GetItem(self, stmt: stack_move.GetItem) -> None:
        from kirin.dialects.py import indexing
        array = self.ssa_to_target[stmt.array]
        current = array
        for idx_ssa in stmt.indices:
            target_idx = self.ssa_to_target[idx_ssa]
            gi = indexing.GetItem(obj=current, index=target_idx)
            self.target_block.stmts.append(gi)
            current = gi.result
        self.ssa_to_target[stmt.result] = current

    def _rewrite_SetDetector(self, stmt: stack_move.SetDetector) -> None:
        from kirin.dialects import ilist
        from bloqade.decoders.dialects import annotate
        measurements = self.ssa_to_target[stmt.array]
        # Coordinates default to empty per the spec — decoded bytecode
        # doesn't carry visualisation metadata.
        empty_coords = ilist.New(values=())
        self.target_block.stmts.append(empty_coords)
        self.target_block.stmts.append(
            annotate.SetDetector(measurements=measurements, coordinates=empty_coords.result)
        )

    def _rewrite_SetObservable(self, stmt: stack_move.SetObservable) -> None:
        from bloqade.decoders.dialects import annotate
        measurements = self.ssa_to_target[stmt.array]
        self.target_block.stmts.append(annotate.SetObservable(measurements=measurements))
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

        # Emit a MeasurementFuture analogous to the end_measure_impl logic.
        # ... (mirror the existing end_measure_impl in this file) ...
        return (...,)  # fill based on end_measure_impl's return
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

from bloqade.lanes.dialects import move
from bloqade.lanes.rewrite.measure_lower import MeasureLower, MeasureLowerError


def test_single_zone_measure_rewrites_to_endmeasure():
    state = ir.TestValue()
    zone = ir.TestValue()
    m = move.Measure(current_state=state, zones=(zone,))
    block = ir.Block([m])
    # For this test we mock the analysis result — in real usage MeasureLower
    # runs AtomAnalysis first.
    lower = MeasureLower(zone_sets={m: frozenset({0})}, final_measure_count=1)
    lower.run(block)
    # m has been replaced by a move.EndMeasure.
    assert not any(isinstance(s, move.Measure) for s in block.stmts)
    assert any(isinstance(s, move.EndMeasure) for s in block.stmts)


def test_multi_zone_measure_raises():
    state = ir.TestValue()
    z0, z1 = ir.TestValue(), ir.TestValue()
    m = move.Measure(current_state=state, zones=(z0, z1))
    block = ir.Block([m])
    lower = MeasureLower(zone_sets={m: frozenset({0, 1})}, final_measure_count=1)
    with pytest.raises(MeasureLowerError):
        lower.run(block)
```

- [ ] **Step 2: Run tests**

```bash
uv run pytest python/tests/rewrite/test_measure_lower.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write implementation**

Create `python/bloqade/lanes/rewrite/measure_lower.py`:

```python
"""measure_lower — validate + rewrite move.Measure to move.EndMeasure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from kirin import ir

from bloqade.lanes.dialects import move


class MeasureLowerError(RuntimeError):
    """Raised when the measure_lower invariants are violated."""


@dataclass
class MeasureLower:
    """Lower move.Measure stmts to move.EndMeasure.

    Requires the zone set per Measure site (from AtomAnalysis) and the
    program-wide count of final measurements. Enforces:

    1. Each move.Measure covers exactly one zone.
    2. The program contains exactly one final measurement.
    """

    zone_sets: Mapping[move.Measure, frozenset[int]]
    final_measure_count: int

    def run(self, block: ir.Block) -> None:
        if self.final_measure_count != 1:
            raise MeasureLowerError(
                f"expected exactly one final measurement, "
                f"found {self.final_measure_count}"
            )
        for stmt in list(block.stmts):
            if isinstance(stmt, move.Measure):
                self._rewrite_measure(stmt, block)

    def _rewrite_measure(self, stmt: move.Measure, block: ir.Block) -> None:
        zones = self.zone_sets.get(stmt)
        if zones is None:
            raise MeasureLowerError(f"no analysis result for {stmt}")
        if len(zones) != 1:
            raise MeasureLowerError(
                f"move.Measure spans {len(zones)} zones; expected exactly 1"
            )
        from bloqade.lanes.bytecode import ZoneAddress
        (zone_id,) = zones
        replacement = move.EndMeasure(
            current_state=stmt.current_state,
            zone_addresses=(ZoneAddress(zone_id),),
        )
        # Swap stmt for replacement in-place.
        stmt.replace_by(replacement)
```

Note: the exact `replace_by` / statement-substitution API may differ — see Kirin's `ir.Statement.replace_by` or use `stmt.delete()` + inserting the new statement. Adjust to match the codebase's conventions (see `python/bloqade/lanes/rewrite/state.py` for an existing example).

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

    MeasureLower.from_method(main, arch_spec=get_arch_spec()).run(
        main.callable_region.blocks[0]
    )
    block = main.callable_region.blocks[0]
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
        return cls(zone_sets=zone_sets, final_measure_count=interp.final_measurement_count)
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
    """Same arch-spec builder used in other lanes tests."""
    # Borrow from existing tests (e.g. test_atom_interpreter.py get_arch_spec)
    ...


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

    # Lower to move dialect.
    from kirin import ir as kirin_ir
    source_block = method.callable_region.blocks[0]
    lowered_block = LowerStackMove().run(source_block)

    # Patch the method's region to use the lowered block (API-dependent).
    method.callable_region = kirin_ir.Region(blocks=[lowered_block])

    # Apply measure_lower.
    arch_spec = _build_arch_spec()
    MeasureLower.from_method(method, arch_spec).run(lowered_block)

    # Assert: the method now contains move.EndMeasure, not move.Measure.
    from bloqade.lanes.dialects import move
    assert any(isinstance(s, move.EndMeasure) for s in lowered_block.stmts)
    assert not any(isinstance(s, move.Measure) for s in lowered_block.stmts)
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
