# qmove Unified Frames Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every qmove method, entry kernel included, opens its state chain with `qmove.enter(frame)` and closes it with `qmove.exit`. The entry kernel's frame becomes a new `MachineFrame` marker, and `move.Load`/`move.Store` leave qmove IR.

**Architecture:** `Enter.frame` widens to `Frame | MachineFrame | None`. `thread_method` drops its `subroutine` flag and always emits the `enter`/`exit` pair. `NativeToQMove` gives the entry kernel `MachineFrame()`. The role rules become frame rules:
- **V2:** exactly one `enter` and one `exit`, at top level, and no `Load`/`Store`.
- **F5:** a whole-machine callee needs a whole-machine caller.
- **Allocation:** allowed only under `MachineFrame`.
- **F1–F5 and the policy:** apply only to partial frames.

**Tech Stack:** Python 3.10+, kirin-toolchain 0.22, bloqade-circuit 0.15, pytest.

**Spec:** `docs/superpowers/specs/2026-10-02-qmove-dialect-design.md` (revised in commit 72f8cee6). Its "Normal form", "Calls and frames → Kernel roles" and "Validation" sections describe the target behaviour.

**Baseline:** branch `claude/qmove-dialect-spec` at 72f8cee6. All files this plan touches already exist; every step replaces a file's full contents with the code shown.

## Global Constraints

- Target the **physical** pipeline only. Do not modify any module outside the qmove files listed below. The current `place`/`move` pipelines stay untouched.
- Python must stay compatible with 3.10: ruff `target-version = "py310"`.
- Always run Python tooling as `uv run --no-sync ...`. A plain `uv run` re-syncs the environment and silently replaces the locally built `_native` extension.
- Never use kirin's `is_structurally_equal` on IR with regions; use `tests._qmove_helpers.blocks_equal`.
- `kirin.ir.exception.ValidationErrorGroup` is a `BaseException`.
- User ruling: the small duplicated `_is_state` helpers stay duplicated. Do not consolidate them.
- `bloqade.gemini.physical.kernel` inlines calls unless `inline=False`.
- Commit messages follow Conventional Commits, with a `Co-Authored-By` trailer naming your model.

Lint step for every task (all four must pass before committing):

```bash
uv run --no-sync isort python && uv run --no-sync black python && uv run --no-sync ruff check python && uv run --no-sync pyright python
```

## File Structure

| File | Change |
|---|---|
| `python/bloqade/lanes/dialects/qmove/frame.py` | Add `MachineFrame`; module docstring names the three frame kinds |
| `python/bloqade/lanes/dialects/qmove/stmts.py` | `Enter.frame: Frame \| MachineFrame \| None`; docstrings for `Enter`/`Exit` describe every method |
| `python/bloqade/lanes/dialects/qmove/__init__.py` | Re-export `MachineFrame` |
| `python/bloqade/lanes/rewrite/qmove_state.py` | `thread_method(mt, *, frame)` always emits `enter`/`exit` |
| `python/bloqade/lanes/validation/qmove_input.py` | `get_input_validation(may_allocate, clones)`; allocation allowed only when `may_allocate` |
| `python/bloqade/lanes/validation/qmove.py` | `is_qmove_method` and `method_frame` replace `is_subroutine` and `subroutine_frame`; V2 becomes `check_chain_ends`; F5's whole-machine half moves into `check_calls`; F1–F5 and the policy apply to partial `Frame`s only |
| `python/bloqade/lanes/transform/native_to_qmove.py` | Entry kernel gets `MachineFrame()`; roles are `(method, frame)` pairs |
| Tests | `dialects/test_qmove.py`, `rewrite/test_qmove_state.py`, `validation/test_qmove.py`, `validation/test_qmove_input.py`, `test_transform_native_to_qmove.py` |

---

### Task 1: `MachineFrame` in the dialect

**Files:**
- Modify: `python/bloqade/lanes/dialects/qmove/frame.py`, `stmts.py`, `__init__.py`
- Test: `python/tests/dialects/test_qmove.py`

**Interfaces:**
- Produces:
  - `qmove.MachineFrame()`, a frozen, field-less, hashable dataclass. Instances compare equal.
  - `qmove.Enter(frame=...)` accepting `Frame | MachineFrame | None`.

- [ ] **Step 1: Write the failing test.** Replace the contents of `python/tests/dialects/test_qmove.py` with:

```python
from typing import Literal

from bloqade.types import MeasurementResultType, Qubit, QubitType
from kirin import ir, types
from kirin.dialects import ilist

from bloqade import squin
from bloqade.lanes.bytecode.encoding import LocationAddress, ZoneAddress
from bloqade.lanes.dialects import move, qmove
from bloqade.lanes.types import StateType

QUBITS2 = ilist.IListType[QubitType, types.Literal(2)]


def _state() -> ir.TestValue:
    return ir.TestValue(StateType)


def test_gate_statements_thread_the_state():
    s, qs = _state(), ir.TestValue(QUBITS2)
    angle = ir.TestValue(types.Float)
    for stmt in (
        qmove.CZ(s, qs, qs),
        qmove.R(s, angle, angle, qs),
        qmove.Rz(s, angle, qs),
    ):
        assert stmt.current_state is s
        assert stmt.result.type is StateType
        assert stmt.get_trait(move.ConsumesState) == move.ConsumesState(False)
        assert stmt.get_trait(move.EmitsState) == move.EmitsState(False)


def test_measure_returns_state_then_measurements():
    s, qs = _state(), ir.TestValue(QUBITS2)
    stmt = qmove.Measure(s, qs)
    assert stmt.results[0].type is StateType
    assert stmt.measurements.type.is_subseteq(
        ilist.IListType[MeasurementResultType, types.Any]
    )


def test_move_to_and_permute_keep_their_attributes():
    s, qs = _state(), ir.TestValue(QUBITS2)
    locs = ir.TestValue(types.Any)
    perm = ir.TestValue(ilist.IListType[types.Int, types.Literal(2)])
    assert (
        qmove.MoveTo(s, qs, locs, multi_move_warning=False).multi_move_warning is False
    )
    assert qmove.Permute(s, qs, perm, insert_moves=True).insert_moves is True


def test_enter_holds_an_optional_hashable_frame():
    frame = qmove.Frame(
        qmove.FrameShape(((0, 2),), scratch_slots=1),
        (LocationAddress(0, 0), LocationAddress(1, 0), LocationAddress(2, 0)),
        qmove.Effects(cz_zones=frozenset({ZoneAddress(0)})),
    )
    assert frame.shape.total_slots == 3
    assert qmove.Enter(frame=frame).frame == frame
    assert hash(qmove.Enter(frame=frame).attributes["frame"]) is not None
    assert qmove.Enter().frame is None
    assert qmove.Enter().get_trait(move.EmitsState) == move.EmitsState(True)
    assert qmove.Exit(_state()).get_trait(move.ConsumesState) == move.ConsumesState(
        False
    )


def test_effects_subset():
    z0, z1 = ZoneAddress(0), ZoneAddress(1)
    inner = qmove.Effects(cz_zones=frozenset({z0}))
    outer = qmove.Effects(cz_zones=frozenset({z0, z1}), global_pulses=True)
    assert inner.is_subset_of(outer)
    assert not outer.is_subset_of(inner)


def test_invoke_and_prepare_take_a_callee_and_inputs():
    @squin.kernel
    def sub(qs: ilist.IList[Qubit, Literal[2]]):
        squin.h(qs[0])

    s, qs = _state(), ir.TestValue(QUBITS2)
    invoke = qmove.Invoke(s, (qs,), callee=sub)
    assert invoke.callee is sub and tuple(invoke.inputs) == (qs,)
    assert len(invoke.results) == 2
    prepare = qmove.Prepare(s, (qs,), callee=sub)
    assert prepare.result.type is StateType


def test_dialect_contains_every_statement():
    assert set(qmove.dialect.stmts) == {
        qmove.CZ,
        qmove.R,
        qmove.Rz,
        qmove.MoveTo,
        qmove.Permute,
        qmove.Measure,
        qmove.Enter,
        qmove.Exit,
        qmove.Prepare,
        qmove.Invoke,
    }


def test_machine_frame_is_a_hashable_marker():
    assert qmove.MachineFrame() == qmove.MachineFrame()
    enter = qmove.Enter(frame=qmove.MachineFrame())
    assert enter.frame == qmove.MachineFrame()
    assert hash(enter.attributes["frame"]) == hash(
        qmove.Enter(frame=qmove.MachineFrame()).attributes["frame"]
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest python/tests/dialects/test_qmove.py -v`
Expected: `test_machine_frame_is_a_hashable_marker` FAILS with `AttributeError: module 'bloqade.lanes.dialects.qmove' has no attribute 'MachineFrame'`. The other 7 tests pass.

- [ ] **Step 3: Write the implementation.** Replace the contents of the three dialect files.

`python/bloqade/lanes/dialects/qmove/frame.py`:

```python
"""Subroutine frames: where a subroutine's arguments must be, and what it may do.

Every qmove method opens its chain with ``qmove.enter(frame)``. The frame is a
``Frame`` (partial: entry slots, scratch and effects), a ``MachineFrame`` (the
whole machine, as for the entry kernel), or ``None`` (a hole for synthesis).

A ``Frame`` is a *shape* (how many slots each qubit parameter takes, plus
scratch) and a *binding* of those slots to concrete locations. Keeping the two apart
leaves room for relocatable frames later without changing the IR; for now the
binding is always concrete.

Exit = entry: on return every argument atom is back in its own entry slot and
the scratch slots are empty again, so there is no separate exit layout.
"""

from __future__ import annotations

from dataclasses import dataclass

from bloqade.lanes.bytecode.encoding import LocationAddress, ZoneAddress


@dataclass(frozen=True)
class FrameShape:
    param_slots: tuple[tuple[int, int], ...]
    """``(parameter index, slot count)`` for each qubit-typed parameter.

    Indices exclude ``self``. A ``Qubit`` parameter takes one slot and an
    ``IList[Qubit, Literal[N]]`` parameter takes ``N``, in element order.
    """
    scratch_slots: int = 0

    @property
    def total_slots(self) -> int:
        return sum(count for _, count in self.param_slots) + self.scratch_slots


@dataclass(frozen=True)
class Effects:
    """Zone-wide operations a subroutine is permitted to perform."""

    cz_zones: frozenset[ZoneAddress] = frozenset()
    measure_zones: frozenset[ZoneAddress] = frozenset()
    global_pulses: bool = False

    def is_subset_of(self, other: Effects) -> bool:
        return (
            self.cz_zones <= other.cz_zones
            and self.measure_zones <= other.measure_zones
            and (other.global_pulses or not self.global_pulses)
        )


@dataclass(frozen=True)
class MachineFrame:
    """The whole machine: no footprint limit, and every effect is allowed.

    The entry kernel's frame. A method under it may allocate qubits, and only
    another whole-machine method may call it (F5).
    """


@dataclass(frozen=True)
class Frame:
    shape: FrameShape
    binding: tuple[LocationAddress, ...]
    """One location per slot: parameter slots in parameter order, then scratch."""
    effects: Effects = Effects()

    @property
    def footprint(self) -> frozenset[LocationAddress]:
        return frozenset(self.binding)
```

`python/bloqade/lanes/dialects/qmove/stmts.py`:

```python
from __future__ import annotations

from kirin import ir, types
from kirin.decl import info, statement
from kirin.dialects import ilist

from bloqade import types as bloqade_types
from bloqade.lanes.dialects.arch import LocationAddressType
from bloqade.lanes.dialects.move import ConsumesState, EmitsState, StatefulStatement
from bloqade.lanes.types import StateType

from ._dialect import dialect
from .frame import Frame, MachineFrame

N = types.TypeVar("N")
Len = types.TypeVar("Len")
QubitList = ilist.IListType[bloqade_types.QubitType, types.Any]


@statement(dialect=dialect)
class CZ(StatefulStatement):
    controls: ir.SSAValue = info.argument(ilist.IListType[bloqade_types.QubitType, N])
    targets: ir.SSAValue = info.argument(ilist.IListType[bloqade_types.QubitType, N])


@statement(dialect=dialect)
class R(StatefulStatement):
    axis_angle: ir.SSAValue = info.argument(types.Float)
    rotation_angle: ir.SSAValue = info.argument(types.Float)
    qubits: ir.SSAValue = info.argument(QubitList)


@statement(dialect=dialect)
class Rz(StatefulStatement):
    rotation_angle: ir.SSAValue = info.argument(types.Float)
    qubits: ir.SSAValue = info.argument(QubitList)


@statement(dialect=dialect)
class MoveTo(StatefulStatement):
    """Leaves quantum information unchanged; afterwards the atoms carrying
    ``qubits`` are at ``locations``."""

    qubits: ir.SSAValue = info.argument(ilist.IListType[bloqade_types.QubitType, Len])
    locations: ir.SSAValue = info.argument(ilist.IListType[LocationAddressType, Len])
    multi_move_warning: bool = info.attribute(default=True)


@statement(dialect=dialect)
class Permute(StatefulStatement):
    """Afterwards ``qubits[i]`` holds what ``qubits[perm[i]]`` held.

    ``insert_moves`` constrains how synthesis realizes it: ``False`` by
    relabeling the reference-to-atom binding with no moves, ``True`` with moves
    that restore the previous binding.
    """

    qubits: ir.SSAValue = info.argument(ilist.IListType[bloqade_types.QubitType, Len])
    perm: ir.SSAValue = info.argument(ilist.IListType[types.Int, Len])
    insert_moves: bool = info.attribute(default=False)


@statement(dialect=dialect)
class Measure(StatefulStatement):
    """Non-terminal measurement: the state continues after it."""

    qubits: ir.SSAValue = info.argument(ilist.IListType[bloqade_types.QubitType, Len])
    measurements: ir.ResultValue = info.result(
        ilist.IListType[bloqade_types.MeasurementResultType, Len]
    )


@statement(dialect=dialect)
class Enter(ir.Statement):
    """Open a method's chain; reads the machine state, plus the frame's precondition.

    Every qmove method has exactly one. ``MachineFrame()`` is the whole machine
    (the entry kernel); ``frame=None`` is a hole that later synthesis fills.
    """

    traits = frozenset({EmitsState(True)})
    frame: Frame | MachineFrame | None = info.attribute(default=None)
    result: ir.ResultValue = info.result(StateType)


@statement(dialect=dialect)
class Exit(ir.Statement):
    """Close a method's chain; writes the machine state, plus the frame's postcondition."""

    traits = frozenset({ConsumesState(False)})
    current_state: ir.SSAValue = info.argument(StateType)


@statement(dialect=dialect)
class Prepare(StatefulStatement):
    """Establish ``callee``'s frame precondition for ``inputs``."""

    callee: ir.Method = info.attribute()
    inputs: tuple[ir.SSAValue, ...] = info.argument()


@statement(dialect=dialect)
class Invoke(StatefulStatement):
    """Call a subroutine on the caller's chain.

    ``value`` always exists (a statement's result count is fixed by its
    declaration) and has the callee's return type, ``NoneType`` included.
    """

    callee: ir.Method = info.attribute()
    inputs: tuple[ir.SSAValue, ...] = info.argument()
    value: ir.ResultValue = info.result()
```

`python/bloqade/lanes/dialects/qmove/__init__.py`:

```python
from . import stmts as stmts
from ._dialect import dialect as dialect
from .frame import (
    Effects as Effects,
    Frame as Frame,
    FrameShape as FrameShape,
    MachineFrame as MachineFrame,
)
from .stmts import (
    CZ as CZ,
    Enter as Enter,
    Exit as Exit,
    Invoke as Invoke,
    Measure as Measure,
    MoveTo as MoveTo,
    Permute as Permute,
    Prepare as Prepare,
    R as R,
    Rz as Rz,
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest python/tests/dialects/test_qmove.py python/tests/validation/test_qmove.py python/tests/test_transform_native_to_qmove.py -v`
Expected: all PASS, with 8 in `dialects/test_qmove.py`. The other two files are unchanged by this task and must still pass.

- [ ] **Step 5: Lint and commit**

```bash
git add python/bloqade/lanes/dialects/qmove python/tests/dialects/test_qmove.py
git commit -m "feat(python): add a whole-machine frame marker to qmove"
```

---

### Task 2: One opener for every qmove method

This is the unification itself. It is one task because the threading change, the validation rules and the transform depend on each other. With only part of it applied, the suite is red.

**Files:**
- Modify: `python/bloqade/lanes/rewrite/qmove_state.py`, `python/bloqade/lanes/validation/qmove_input.py`, `python/bloqade/lanes/validation/qmove.py`, `python/bloqade/lanes/transform/native_to_qmove.py`
- Test: `python/tests/rewrite/test_qmove_state.py`, `python/tests/validation/test_qmove.py`, `python/tests/validation/test_qmove_input.py`, `python/tests/test_transform_native_to_qmove.py`

**Interfaces:**
- Consumes: Task 1's `qmove.MachineFrame`.
- Produces:
  - `thread_method(mt: ir.Method, *, frame: Frame | MachineFrame | None) -> None`. `frame` is a required keyword argument; the `subroutine` parameter is gone.
  - `get_input_validation(may_allocate: bool, clones: frozenset[ir.Method] = frozenset())`, with the module constant `ALLOCATION_MESSAGE`.
  - In `bloqade.lanes.validation.qmove`: `is_qmove_method(mt) -> bool`, `method_frame(mt) -> Frame | MachineFrame | None`, and `check_chain_ends(mt)`, which replaces `check_cell_access`.
  - `NativeToQMove.subroutines: Mapping[ir.Method, Frame | MachineFrame | None]`.

New messages (tests match them exactly):
- `"V2: {name} is not allowed in qmove IR; a method's chain uses enter/exit"`
- `"V2: a qmove method needs exactly one {enter|exit}, found {n}"`
- `"V2: {enter|exit} must be in the method's top-level block"`
- `"{invoke|prepare} target {name} is not a qmove method"`
- `"qmove method {name} must be called with qmove.invoke"`
- `"F5: whole-machine method {name} can only be called from a whole-machine method"`
- `"qubits may only be allocated under a whole-machine frame (the entry kernel)"`

- [ ] **Step 1: Write the failing tests.** Replace the contents of the four test files.

`python/tests/rewrite/test_qmove_state.py`:

```python
from itertools import pairwise
from typing import Literal

from bloqade.types import Qubit
from kirin import ir, rewrite, types
from kirin.dialects import func, ilist, scf
from tests._qmove_helpers import first_of, statements_of, top_level

from bloqade import squin
from bloqade.lanes.arch.gemini.physical import get_arch_spec
from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.dialects import move, qmove
from bloqade.lanes.dialects.qmove import MachineFrame
from bloqade.lanes.rewrite.native2qmove import RewriteNativeToQMove
from bloqade.lanes.rewrite.qmove_state import thread_block, thread_method
from bloqade.lanes.transform.qmove_frontend import NativeProgram, lower_to_native
from bloqade.lanes.types import StateType


def _lowered(kernel: ir.Method, subroutines=()) -> NativeProgram:
    program = lower_to_native(kernel, subroutines, get_arch_spec())
    clones = frozenset(program.subroutines.values())
    for method in (program.entry, *clones):
        rewrite.Walk(RewriteNativeToQMove(clones)).rewrite(method.code)
    return program


def _yield(region: ir.Region) -> scf.Yield:
    stmt = region.blocks[0].last_stmt
    assert isinstance(stmt, scf.Yield)
    return stmt


@squin.kernel
def straight():
    qs = squin.qalloc(2)
    squin.h(qs[0])
    squin.cz(qs[0], qs[1])
    return squin.broadcast.measure(qs)


def test_straight_line_becomes_one_chain():
    program = _lowered(straight)
    thread_method(program.entry, frame=MachineFrame())
    stmts = top_level(program.entry)
    enter, exit_ = stmts[0], stmts[-2]
    assert isinstance(enter, qmove.Enter) and enter.frame == MachineFrame()
    assert isinstance(exit_, qmove.Exit) and isinstance(stmts[-1], func.Return)
    assert not statements_of(program.entry, move.Load)
    assert not statements_of(program.entry, move.Store)
    chain = [s for s in stmts if s.args and s.args[0].type.is_subseteq(StateType)]
    for prev, nxt in pairwise(chain):
        assert nxt.args[0] is prev.results[0]


@squin.kernel
def branchy():
    qs = squin.qalloc(2)
    if squin.is_one(squin.measure(qs[0])):
        squin.x(qs[1])
    squin.h(qs[0])


def test_if_else_captures_and_yields_the_state():
    program = _lowered(branchy)
    thread_method(program.entry, frame=MachineFrame())
    branch = first_of(program.entry, scf.IfElse)
    measure = first_of(program.entry, qmove.Measure)
    assert branch.results[0].type.is_subseteq(StateType)
    assert _yield(branch.else_body).values[0] is measure.results[0]
    assert _yield(branch.then_body).values[0] is not measure.results[0]
    after = [s for s in top_level(program.entry) if isinstance(s, (qmove.R, qmove.Rz))]
    assert after[0].current_state is branch.results[0]


@squin.kernel
def loopy():
    qs = squin.qalloc(3)
    for i in range(2):
        squin.z(qs[i + 1])


def test_for_carries_the_state_ahead_of_existing_iter_args():
    program = _lowered(loopy)
    carried_before = len(first_of(program.entry, scf.For).initializers)
    assert carried_before > 0  # kirin carries `qs` through the loop
    thread_method(program.entry, frame=MachineFrame())
    loop = first_of(program.entry, scf.For)
    assert len(loop.initializers) == carried_before + 1
    assert loop.body.blocks[0].args[1].type.is_subseteq(StateType)
    assert _yield(loop.body).values[0].type.is_subseteq(StateType)
    assert first_of(program.entry, qmove.Exit).current_state is loop.results[0]


def test_hand_built_if_without_else_gets_one():
    cond = ir.TestValue(types.Bool)
    outer = ir.TestValue(StateType)
    then_block = ir.Block()
    then_block.args.append_from(types.Bool)
    load = move.Load()
    gate = qmove.Rz(load.result, ir.TestValue(types.Float), ir.TestValue(types.Any))
    then_block.stmts.extend([load, gate, move.Store(gate.result), scf.Yield()])
    branch = scf.IfElse(cond, then_block)
    assert not branch.else_body.blocks  # the constructor leaves else empty
    block = ir.Block([branch])
    final = thread_block(block, outer)
    (new_branch,) = [s for s in block.stmts if isinstance(s, scf.IfElse)]
    assert final is new_branch.results[0]
    assert _yield(new_branch.else_body).values[0] is outer


@squin.kernel
def sub(qs: ilist.IList[Qubit, Literal[2]]):
    squin.cz(qs[0], qs[1])


@squin.kernel
def calls_sub():
    qs = squin.qalloc(2)
    sub(qs)


def test_subroutine_uses_enter_and_exit():
    program = _lowered(calls_sub, [sub])
    clone = program.subroutines[sub]
    frame = qmove.Frame(
        qmove.FrameShape(((0, 2),)), (LocationAddress(0, 0), LocationAddress(1, 0))
    )
    thread_method(clone, frame=frame)
    stmts = top_level(clone)
    enter = stmts[0]
    assert isinstance(enter, qmove.Enter) and enter.frame == frame
    assert isinstance(stmts[-2], qmove.Exit)
    assert not statements_of(clone, move.Load) and not statements_of(clone, move.Store)


def test_method_without_quantum_operations_still_gets_enter_and_exit():
    @squin.kernel
    def classical(x: int) -> int:
        return x + 1

    out = classical.similar()
    thread_method(out, frame=None)
    stmts = top_level(out)
    assert isinstance(stmts[0], qmove.Enter)
    exit_ = stmts[-2]
    assert isinstance(exit_, qmove.Exit) and exit_.current_state is stmts[0].result
```

`python/tests/validation/test_qmove_input.py`:

```python
from typing import Literal

import pytest
from bloqade.types import Qubit, QubitType
from kirin import ir, types as kirin_types
from kirin.dialects import func, ilist, py, scf
from kirin.validation import ValidationSuite

from bloqade import squin
from bloqade.gemini import physical
from bloqade.gemini.common.dialects import qubit as gemini_qubit
from bloqade.gemini.logical.dialects.extensions import stmts as extensions
from bloqade.gemini.logical.dialects.operations import stmts as logical
from bloqade.lanes.arch.gemini.physical import get_arch_spec
from bloqade.lanes.transform.qmove_frontend import lower_to_native
from bloqade.lanes.validation.qmove_input import get_input_validation


def _messages(
    method: ir.Method,
    may_allocate: bool = True,
    clones: frozenset[ir.Method] = frozenset(),
) -> list[str]:
    result = ValidationSuite([get_input_validation(may_allocate, clones)]).validate(
        method
    )
    return [str(err.args[0]) for errs in result.errors.values() for err in errs]


def _native(kernel, subroutines=()):
    return lower_to_native(kernel, subroutines, get_arch_spec())


def _method(*blocks: ir.Block) -> ir.Method:
    @squin.kernel
    def stub():
        return None

    out = stub.similar()
    out.code = func.Function(
        sym_name="stub",
        signature=func.Signature(inputs=(), output=kirin_types.NoneType),
        body=ir.Region(list(blocks)),
    )
    return out


def test_supported_kernel_is_valid():
    @squin.kernel
    def k():
        qs = squin.qalloc(2)
        for q in qs:
            squin.h(q)
        if squin.is_one(squin.measure(qs[0])):
            squin.x(qs[1])
        return squin.broadcast.measure(qs)

    assert _messages(_native(k).entry) == []


def test_reset_is_rejected():
    @squin.kernel
    def k():
        qs = squin.qalloc(1)
        squin.reset(qs[0])

    assert _messages(_native(k).entry) == ["qubit.reset is not supported"]


def test_noise_is_rejected():
    @squin.kernel
    def k():
        qs = squin.qalloc(1)
        squin.depolarize(0.1, qs[0])

    (message,) = _messages(_native(k).entry)
    assert "is not supported by the qmove lowering" in message


def test_function_value_applying_gates_is_rejected():
    @squin.kernel
    def k():
        qs = squin.qalloc(2)

        def f(q: Qubit):
            squin.h(q)

        ilist.for_each(f, qs)

    assert _messages(_native(k).entry) == [
        "for_each applies gates through a function value; use a for loop"
    ]


def test_allocation_only_function_value_is_fine():
    @physical.kernel(verify=False)
    def k():
        def at(i: int):
            return gemini_qubit.new_at(0, i, 0)

        qs = ilist.map(at, ilist.range(2))
        squin.h(qs[0])

    assert _messages(_native(k).entry) == []


@squin.kernel
def allocating_sub(qs: ilist.IList[Qubit, Literal[1]]):
    extra = squin.qalloc(1)
    squin.cz(qs[0], extra[0])


@squin.kernel
def calls_allocating_sub():
    qs = squin.qalloc(1)
    allocating_sub(qs)


def test_allocation_outside_a_machine_frame_is_rejected():
    clone = _native(calls_allocating_sub, [allocating_sub]).subroutines[allocating_sub]
    assert _messages(clone, may_allocate=False) == [
        "qubits may only be allocated under a whole-machine frame (the entry kernel)"
    ]


def test_allocation_under_a_machine_frame_is_allowed():
    clone = _native(calls_allocating_sub, [allocating_sub]).subroutines[allocating_sub]
    assert _messages(clone, may_allocate=True) == []


def _logical_statements() -> list[ir.Statement]:
    qubits = ir.TestValue(ilist.IListType[QubitType, kirin_types.Literal(7)])
    angle = ir.TestValue(kirin_types.Float)
    return [
        logical.TerminalLogicalMeasurement(qubits),
        logical.Initialize(angle, angle, angle, qubits),
        extensions.StarRz(angle, qubits),
    ]


@pytest.mark.parametrize("index", range(3))
def test_logical_statements_are_rejected(index: int):
    stmt = _logical_statements()[index]
    (message,) = _messages(_method(ir.Block([stmt, func.Return()])))
    assert "logical-pipeline statement" in message


def test_call_through_a_function_value_is_rejected():
    call = func.Call(ir.TestValue(kirin_types.Any), (), kwargs=())
    assert _messages(_method(ir.Block([call, func.Return()]))) == [
        "calls through a function value are not supported"
    ]


def test_early_return_and_multiple_blocks_are_rejected():
    cond = ir.TestValue(kirin_types.Bool)
    then_block = ir.Block()
    then_block.args.append_from(kirin_types.Bool)
    then_block.stmts.append(func.Return(py.Constant(1).result))
    else_block = ir.Block()
    else_block.args.append_from(kirin_types.Bool)
    else_block.stmts.append(scf.Yield())
    branch = scf.IfElse(cond, then_block, else_block)
    messages = _messages(
        _method(ir.Block([branch, func.Return()]), ir.Block([func.Return()]))
    )
    assert "early return inside scf is not supported" in messages
    assert any("region has 2 blocks" in m for m in messages)


def test_bottom_typed_quantum_result_is_rejected():
    @squin.kernel
    def k():
        qs = squin.qalloc(3)
        # measure takes one Qubit, so its result is Bottom.
        m = squin.measure(qs)  # type: ignore[arg-type]
        squin.h(qs[0])
        return m[0]  # type: ignore[index]

    (message,) = _messages(_native(k).entry)
    assert message == (
        "measure result has no valid type (Bottom); check its argument types"
    )


def test_leftover_invoke_of_a_non_subroutine_is_rejected():
    @squin.kernel
    def helper():
        return None

    invoke = func.Invoke((), callee=helper)
    assert _messages(_method(ir.Block([invoke, func.Return()]))) == [
        "call of helper survived inlining and is not a listed subroutine"
    ]


def test_invoke_of_a_subroutine_clone_is_fine():
    @squin.kernel
    def sub(qs: ilist.IList[Qubit, Literal[1]]):
        squin.h(qs[0])

    @squin.kernel
    def k():
        qs = squin.qalloc(1)
        sub(qs)

    program = _native(k, [sub])
    clones = frozenset(program.subroutines.values())
    assert _messages(program.entry, clones=clones) == []
    # Without the clone set, the same call is a leftover.
    assert _messages(program.entry) == [
        "call of sub survived inlining and is not a listed subroutine"
    ]
```

`python/tests/validation/test_qmove.py`:

```python
from typing import Any, Literal

from bloqade.types import Qubit, QubitType
from kirin import ir, rewrite, types as kirin_types
from kirin.analysis import const
from kirin.dialects import func, ilist, py, scf
from kirin.validation import ValidationSuite
from tests._qmove_helpers import first_of

from bloqade import squin
from bloqade.gemini import physical
from bloqade.gemini.common.dialects import arrange
from bloqade.lanes.arch.gemini.physical import get_arch_spec
from bloqade.lanes.bytecode.encoding import LocationAddress, ZoneAddress
from bloqade.lanes.dialects import move, qmove
from bloqade.lanes.dialects.qmove import Effects, Frame, FrameShape, MachineFrame
from bloqade.lanes.rewrite.native2qmove import RewriteNativeToQMove
from bloqade.lanes.rewrite.qmove_state import thread_method
from bloqade.lanes.transform.qmove_frontend import NativeProgram, lower_to_native
from bloqade.lanes.validation.qmove import get_qmove_validation
from bloqade.lanes.validation.spectator import (
    SingleZonePolicy,
    SpectatorPolicy,
    ZonedPolicy,
)

ARCH = get_arch_spec()
A, B = LocationAddress(0, 0), LocationAddress(1, 0)  # an entangling pair
FAR = LocationAddress(5, 0)
Z0 = ZoneAddress(0)


def _build(
    kernel: ir.Method,
    subroutines: dict[ir.Method, Frame | MachineFrame | None] | None = None,
) -> NativeProgram:
    subroutines = subroutines or {}
    program = lower_to_native(kernel, subroutines, ARCH)
    clones = frozenset(program.subroutines.values())
    rewrite.Walk(RewriteNativeToQMove(clones)).rewrite(program.entry.code)
    thread_method(program.entry, frame=MachineFrame())
    for original, clone in program.subroutines.items():
        rewrite.Walk(RewriteNativeToQMove(clones)).rewrite(clone.code)
        thread_method(clone, frame=subroutines[original])
    return program


def _messages(method: ir.Method, policy: SpectatorPolicy | None = None) -> list[str]:
    validation = get_qmove_validation(ARCH, policy or ZonedPolicy())
    result = ValidationSuite([validation]).validate(method)
    return [str(err.args[0]) for errs in result.errors.values() for err in errs]


@squin.kernel
def sub(qs: ilist.IList[Qubit, Literal[2]]):
    squin.cz(qs[0], qs[1])


@squin.kernel
def main():
    qs = squin.qalloc(3)
    if squin.is_one(squin.measure(qs[0])):
        squin.x(qs[1])
    for q in qs:
        squin.h(q)
    sub(ilist.IList([qs[0], qs[1]]))
    return squin.broadcast.measure(qs)


PAIR = Frame(FrameShape(((0, 2),)), (A, B), Effects(cz_zones=frozenset({Z0})))


def test_lowered_program_is_valid():
    program = _build(main, {sub: PAIR})
    assert _messages(program.entry) == []
    assert _messages(program.subroutines[sub], SingleZonePolicy()) == []


def test_hole_subroutine_is_valid():
    program = _build(main, {sub: None})
    assert _messages(program.subroutines[sub]) == []


# --- V1 ---------------------------------------------------------------------


def test_v1_double_consumption():
    program = _build(main)
    measure = first_of(program.entry, qmove.Measure)
    qmove.Measure(measure.current_state, measure.qubits).insert_after(measure)
    assert any("consumed twice on one path" in m for m in _messages(program.entry))


def test_v1_dropped_update():
    program = _build(main)
    first_of(program.entry, qmove.Exit).delete()
    assert any("never used" in m for m in _messages(program.entry))


def test_v1_state_captured_into_a_loop():
    program = _build(main)
    loop = first_of(program.entry, scf.For)
    gate = first_of(loop, qmove.R)
    qmove.Rz(loop.initializers[0], gate.rotation_angle, gate.qubits).insert_before(gate)
    assert any("inside a loop body" in m for m in _messages(program.entry))


def test_v1_one_use_per_if_arm_is_fine():
    program = _build(main)
    branch = first_of(program.entry, scf.IfElse)
    else_yield = branch.else_body.blocks[0].last_stmt
    assert isinstance(else_yield, scf.Yield)
    assert len(else_yield.values[0].uses) >= 2  # then-arm gate and else-arm yield
    assert _messages(program.entry) == []


# --- V2 ---------------------------------------------------------------------


def _insert_load_store_before(anchor: ir.Statement | None) -> None:
    assert anchor is not None
    load = move.Load()
    load.insert_before(anchor)
    move.Store(load.result).insert_after(load)


def test_v2_load_and_store_are_rejected():
    program = _build(main)
    branch = first_of(program.entry, scf.IfElse)
    _insert_load_store_before(branch.then_body.blocks[0].first_stmt)
    messages = _messages(program.entry)
    for name in ("load", "store"):
        assert (
            f"V2: {name} is not allowed in qmove IR; a method's chain uses enter/exit"
            in messages
        )


def test_v2_needs_exactly_one_exit():
    program = _build(main)
    first_of(program.entry, qmove.Exit).delete()
    assert "V2: a qmove method needs exactly one exit, found 0" in _messages(
        program.entry
    )


def test_v2_enter_must_be_top_level():
    program = _build(main)
    branch = first_of(program.entry, scf.IfElse)
    anchor = branch.then_body.blocks[0].first_stmt
    assert anchor is not None
    enter = qmove.Enter(frame=MachineFrame())
    enter.insert_before(anchor)
    qmove.Exit(enter.result).insert_after(enter)
    messages = _messages(program.entry)
    assert "V2: enter must be in the method's top-level block" in messages
    assert "V2: a qmove method needs exactly one enter, found 2" in messages


# --- V3 ---------------------------------------------------------------------


def _method(*stmts: ir.Statement) -> ir.Method:
    @squin.kernel
    def stub():
        return None

    out = stub.similar()
    out.code = func.Function(
        sym_name="stub",
        signature=func.Signature(inputs=(), output=kirin_types.NoneType),
        body=ir.Region(ir.Block([*stmts, func.Return()])),
    )
    return out


def _qubits(n: int) -> ir.TestValue:
    return ir.TestValue(ilist.IListType[QubitType, kirin_types.Literal(n)])


def test_v3_cz_lengths_must_match():
    enter = qmove.Enter(frame=MachineFrame())
    cz = qmove.CZ(enter.result, _qubits(1), _qubits(2))
    messages = _messages(_method(enter, cz, qmove.Exit(cz.result)))
    assert messages == ["V3: cz has 1 controls but 2 targets"]


@physical.kernel(verify=False)
def bad_permute():
    q = squin.qalloc(2)
    arrange.permute(q, ilist.IList([0, 0]))


def test_v3_perm_must_be_a_permutation():
    assert "V3: perm (0, 0) is not a permutation" in _messages(
        _build(bad_permute).entry
    )


def test_v3_move_to_needs_one_location_per_qubit():
    # From Python, TypeInfer unifies MoveTo's Len and RefineQubitTypes reports
    # the mismatch first; this check covers hand-built IR.
    locations = py.Constant(ilist.IList([FAR]))
    locations.result.hints["const"] = const.Value(ilist.IList([FAR]))
    enter = qmove.Enter(frame=MachineFrame())
    move_to = qmove.MoveTo(enter.result, _qubits(2), locations.result)
    messages = _messages(_method(locations, enter, move_to, qmove.Exit(move_to.result)))
    assert messages == ["V3: move_to has 1 locations for 2 qubits"]


# --- call rules -------------------------------------------------------------


def test_func_invoke_of_a_qmove_method_is_rejected():
    program = _build(main, {sub: None})
    invoke = first_of(program.entry, qmove.Invoke)
    func.Invoke(tuple(invoke.inputs), callee=invoke.callee).insert_after(invoke)
    messages = _messages(program.entry)
    assert "qmove method sub must be called with qmove.invoke" in messages


def test_qmove_invoke_of_a_non_qmove_method_is_rejected():
    enter = qmove.Enter(frame=MachineFrame())
    call = qmove.Invoke(enter.result, (), callee=main)
    messages = _messages(_method(enter, call, qmove.Exit(call.result)))
    assert messages == ["invoke target main is not a qmove method"]


# --- frame rules --------------------------------------------------------------


def _frame_messages(
    frame: Frame,
    policy: SpectatorPolicy | None = None,
    kernel: ir.Method = main,
    subroutine: ir.Method = sub,
) -> list[str]:
    program = _build(kernel, {subroutine: frame})
    return _messages(program.subroutines[subroutine], policy)


def test_f1_binding_size_and_distinctness():
    messages = _frame_messages(Frame(FrameShape(((0, 2),)), (A, A, B)))
    assert "F1: binding has 3 locations for 2 slots" in messages
    assert "F1: binding locations are not distinct" in messages


def test_f1_invalid_location_and_zone():
    frame = Frame(
        FrameShape(((0, 2),)),
        (A, LocationAddress(999, 0)),
        Effects(cz_zones=frozenset({ZoneAddress(7)})),
    )
    messages = _frame_messages(frame)
    assert any(m.startswith("F1: invalid location") for m in messages)
    assert "F1: zone 7 does not exist" in messages


def test_f2_shape_must_match_parameters():
    messages = _frame_messages(Frame(FrameShape(((0, 1),)), (A,)))
    assert "F2: frame slots ((0, 1),) do not match parameters ((0, 2),)" in messages


@squin.kernel
def any_length(qs: ilist.IList[Qubit, Any]):
    squin.h(qs[0])


@squin.kernel
def calls_any_length():
    qs = squin.qalloc(2)
    any_length(qs)


def test_f2_parameter_length_must_be_static():
    frame = Frame(FrameShape(((0, 2),)), (A, B))
    messages = _frame_messages(frame, kernel=calls_any_length, subroutine=any_length)
    assert any("needs IList[Qubit, Literal[N]]" in m for m in messages)


@physical.kernel(verify=False)
def mover(qs: ilist.IList[Qubit, Literal[2]]):
    arrange.move_to(ilist.IList([qs[0]]), ilist.IList([FAR]))


# physical.kernel inlines calls by default; inline=False keeps subroutine calls.
@physical.kernel(verify=False, inline=False)
def calls_mover():
    qs = squin.qalloc(2)
    mover(qs)


def test_f3_move_to_must_stay_in_the_frame():
    frame = Frame(FrameShape(((0, 2),)), (A, B))
    messages = _frame_messages(frame, kernel=calls_mover, subroutine=mover)
    assert (
        "F3: move_to target (zone 0, word 5, site 0) is outside the frame" in messages
    )


def test_f3_global_pulse_needs_permission():
    clone = _build(main, {sub: PAIR}).subroutines[sub]
    exit_ = first_of(clone, qmove.Exit)
    pulse = move.GlobalRz(exit_.current_state, ir.TestValue(kirin_types.Float))
    pulse.insert_before(exit_)
    exit_.current_state = pulse.result
    assert "F3: global pulse is not in the frame's effects" in _messages(clone)


@physical.kernel(verify=False)
def relabels(qs: ilist.IList[Qubit, Literal[2]]):
    arrange.permute(qs, ilist.IList([1, 0]))


@physical.kernel(verify=False)
def relabels_back(qs: ilist.IList[Qubit, Literal[2]]):
    arrange.permute(qs, ilist.IList([1, 0]))
    arrange.permute(qs, ilist.IList([1, 0]))


@physical.kernel(verify=False, inline=False)
def calls_relabels():
    qs = squin.qalloc(2)
    relabels(qs)
    relabels_back(qs)


def test_f4_relabels_must_compose_to_identity():
    frame = Frame(FrameShape(((0, 2),)), (A, B))
    program = _build(calls_relabels, {relabels: frame, relabels_back: frame})
    assert any(m.startswith("F4:") for m in _messages(program.subroutines[relabels]))
    assert _messages(program.subroutines[relabels_back]) == []


@physical.kernel(verify=False)
def bad_perm_in_frame(qs: ilist.IList[Qubit, Literal[2]]):
    arrange.permute(qs, ilist.IList([0, 2]))


def test_f4_invalid_perm_does_not_crash_validation():
    frame = Frame(FrameShape(((0, 2),)), (A, B))
    messages = _frame_messages(
        frame, kernel=bad_perm_in_frame, subroutine=bad_perm_in_frame
    )
    # V3 reports "is not a permutation"; validation should not crash with IndexError.
    assert any("is not a permutation" in m for m in messages)
    assert not any(m.startswith("Validation pass") for m in messages)


def _f4_messages(kernel: ir.Method) -> list[str]:
    """F4 messages of a framed subroutine ``kernel`` that takes two qubits."""

    @physical.kernel(verify=False, inline=False)
    def entry():
        qs = squin.qalloc(2)
        kernel(qs)

    program = _build(entry, {kernel: Frame(FrameShape(((0, 2),)), (A, B))})
    return [m for m in _messages(program.subroutines[kernel]) if m.startswith("F4:")]


@physical.kernel(verify=False)
def aliased_relabels(qs: ilist.IList[Qubit, Literal[2]]):
    arrange.permute(ilist.IList([qs[0], qs[1]]), ilist.IList([1, 0]))
    arrange.permute(qs, ilist.IList([1, 0]))


def test_f4_aliased_relabels_do_not_trigger_error():
    # The first relabel acts on a fresh list of the same qubits; without qubit
    # identity F4 cannot judge it, so it abstains.
    assert _f4_messages(aliased_relabels) == []


@physical.kernel(verify=False)
def relabel_in_loop(qs: ilist.IList[Qubit, Literal[2]]):
    arrange.permute(qs, ilist.IList([1, 0]))
    for _ in range(1):
        arrange.permute(qs, ilist.IList([1, 0]))


def test_f4_abstains_when_a_relabel_is_nested():
    # The loop relabel undoes the top-level one, but only the top-level one is
    # collected: judging the composition would be a false positive.
    assert _f4_messages(relabel_in_loop) == []


@physical.kernel(verify=False)
def relabel_then_malformed(qs: ilist.IList[Qubit, Literal[2]]):
    arrange.permute(qs, ilist.IList([1, 0]))
    arrange.permute(qs, ilist.IList([0, 2]))


def test_f4_abstains_when_a_relabel_is_malformed():
    assert _f4_messages(relabel_then_malformed) == []


@physical.kernel(verify=False)
def relabel_then_unknown(
    qs: ilist.IList[Qubit, Literal[2]], perm: ilist.IList[int, Literal[2]]
):
    arrange.permute(qs, ilist.IList([1, 0]))
    arrange.permute(qs, perm)


@physical.kernel(verify=False, inline=False)
def calls_relabel_then_unknown():
    qs = squin.qalloc(2)
    relabel_then_unknown(qs, ilist.IList([1, 0]))


def test_f4_abstains_when_a_permutation_is_not_constant():
    frame = Frame(FrameShape(((0, 2),)), (A, B))
    program = _build(calls_relabel_then_unknown, {relabel_then_unknown: frame})
    messages = _messages(program.subroutines[relabel_then_unknown])
    assert not any(m.startswith("F4:") for m in messages)


@physical.kernel(verify=False)
def inner(qs: ilist.IList[Qubit, Literal[2]]):
    squin.cz(qs[0], qs[1])


@physical.kernel(verify=False, inline=False)
def outer(qs: ilist.IList[Qubit, Literal[2]]):
    inner(qs)


@physical.kernel(verify=False, inline=False)
def calls_outer():
    qs = squin.qalloc(2)
    outer(qs)


def test_f5_nested_frame_must_fit_inside():
    outer_frame = Frame(FrameShape(((0, 2),)), (A, B))
    inner_frame = Frame(FrameShape(((0, 2),)), (A, FAR))
    program = _build(calls_outer, {outer: outer_frame, inner: inner_frame})
    messages = _messages(program.subroutines[outer])
    assert "F5: frame of inner is not inside this frame" in messages


def test_policy_problems_are_reported():
    frame = Frame(FrameShape(((0, 2),)), (A, B), Effects(global_pulses=True))
    assert "ZonedPolicy: subroutines may not use global pulses" in _frame_messages(
        frame
    )


def test_f5_machine_callee_needs_a_machine_caller():
    program = _build(
        calls_outer,
        {outer: Frame(FrameShape(((0, 2),)), (A, B)), inner: MachineFrame()},
    )
    assert (
        "F5: whole-machine method inner can only be called from a whole-machine "
        "method" in _messages(program.subroutines[outer])
    )


def test_machine_callee_from_the_entry_is_fine():
    program = _build(main, {sub: MachineFrame()})
    assert _messages(program.entry) == []
    assert _messages(program.subroutines[sub]) == []
```

`python/tests/test_transform_native_to_qmove.py`:

```python
from typing import Literal

import pytest
from bloqade.types import Qubit
from kirin import ir
from kirin.dialects import func, ilist, scf
from kirin.ir.exception import ValidationErrorGroup
from kirin.validation import ValidationSuite
from tests._qmove_helpers import (
    assert_methods_match,
    erase_qmove,
    statements_of,
    top_level,
)

from bloqade import squin
from bloqade.gemini import physical
from bloqade.gemini.common.dialects import arrange, qubit as gemini_qubit
from bloqade.lanes.arch.gemini.physical import get_arch_spec
from bloqade.lanes.bytecode.encoding import LocationAddress, ZoneAddress
from bloqade.lanes.dialects import move, qmove
from bloqade.lanes.dialects.qmove import Effects, Frame, FrameShape, MachineFrame
from bloqade.lanes.transform import native_to_qmove
from bloqade.lanes.transform.native_to_qmove import NativeToQMove
from bloqade.lanes.transform.qmove_frontend import lower_to_native
from bloqade.lanes.types import StateType
from bloqade.lanes.validation.qmove import get_qmove_validation
from bloqade.lanes.validation.spectator import SingleZonePolicy, ZonedPolicy

ARCH = get_arch_spec()
A, B = LocationAddress(0, 0), LocationAddress(1, 0)


@squin.kernel
def sub(qs: ilist.IList[Qubit, Literal[2]]):
    squin.cx(qs[0], qs[1])


@squin.kernel
def nested(qs: ilist.IList[Qubit, Literal[3]]):
    for q in qs:
        if squin.is_one(squin.measure(q)):
            squin.x(q)


@squin.kernel
def main():
    qs = squin.qalloc(3)
    squin.h(qs[0])
    if squin.is_one(squin.measure(qs[0])):
        squin.x(qs[1])
    else:
        squin.z(qs[1])
    for q in qs:
        squin.h(q)
    for i in range(2):
        squin.z(qs[i + 1])
    sub(ilist.IList([qs[0], qs[1]]))
    nested(qs)
    return squin.broadcast.measure(qs)


@physical.kernel(verify=False)
def arranged():
    a = gemini_qubit.new_at(0, 0, 0)
    b = gemini_qubit.new_at(0, 2, 0)
    arrange.move_to(ilist.IList([b]), ilist.IList([B]))
    squin.cz(a, b)
    arrange.permute(ilist.IList([a, b]), ilist.IList([1, 0]))
    if squin.is_one(squin.broadcast.measure(ilist.IList([a, b]))[0]):
        squin.x(b)
    return squin.broadcast.measure(ilist.IList([a, b]))


SUBROUTINES: dict[ir.Method, Frame | None] = {sub: None, nested: None}


def _callees(entry: ir.Method, kind: type[qmove.Invoke] | type[func.Invoke]):
    return {s.callee.sym_name: s.callee for s in statements_of(entry, kind)}


def test_entry_kernel_is_one_chain():
    out = NativeToQMove(ARCH, SUBROUTINES).emit(main)
    top = top_level(out)
    enter = top[0]
    assert isinstance(enter, qmove.Enter) and enter.frame == MachineFrame()
    assert isinstance(top[-2], qmove.Exit)
    assert len(statements_of(out, qmove.Enter)) == 1
    assert not statements_of(out, move.Load) and not statements_of(out, move.Store)


def test_subroutines_are_called_on_the_chain():
    out = NativeToQMove(ARCH, SUBROUTINES).emit(main)
    callees = _callees(out, qmove.Invoke)
    assert set(callees) == {"sub", "nested"}
    for callee in callees.values():
        assert isinstance(top_level(callee)[0], qmove.Enter)


def test_pinned_frame_reaches_enter():
    frame = Frame(
        FrameShape(((0, 2),)), (A, B), Effects(cz_zones=frozenset({ZoneAddress(0)}))
    )
    subroutines: dict[ir.Method, Frame | None] = {sub: frame, nested: None}
    out = NativeToQMove(ARCH, subroutines, SingleZonePolicy()).emit(main)
    enter = top_level(_callees(out, qmove.Invoke)["sub"])[0]
    assert isinstance(enter, qmove.Enter) and enter.frame == frame


def test_loops_and_branches_carry_the_state():
    out = NativeToQMove(ARCH, SUBROUTINES).emit(main)
    for method in (out, _callees(out, qmove.Invoke)["nested"]):
        structured = statements_of(method, scf.For) + statements_of(method, scf.IfElse)
        assert structured
        for stmt in structured:
            assert stmt.results[0].type.is_subseteq(StateType)


@pytest.mark.parametrize("kernel, subroutines", [(main, SUBROUTINES), (arranged, {})])
def test_lowering_adds_only_state_plumbing(kernel, subroutines):
    reference = lower_to_native(kernel, subroutines, ARCH)
    out = NativeToQMove(ARCH, subroutines).emit(kernel)

    erase_qmove(out.callable_region.blocks[0])
    assert_methods_match(out, reference.entry)
    lowered = _callees(out, func.Invoke)
    for original, native_clone in reference.subroutines.items():
        clone = lowered[original.sym_name]
        erase_qmove(clone.callable_region.blocks[0])
        assert_methods_match(clone, native_clone)


def test_unlisted_recursion_is_rejected():
    @squin.kernel
    def rec(qs: ilist.IList[Qubit, Literal[1]], n: int):
        if n > 0:
            squin.h(qs[0])
            rec(qs, n - 1)

    @squin.kernel
    def calls_rec():
        qs = squin.qalloc(1)
        rec(qs, 3)

    with pytest.raises(ValidationErrorGroup):
        NativeToQMove(ARCH).emit(calls_rec)
    NativeToQMove(ARCH, {rec: None}).emit(calls_rec)


def test_unsupported_input_is_rejected():
    @squin.kernel
    def resets():
        qs = squin.qalloc(1)
        squin.reset(qs[0])

    with pytest.raises(ValidationErrorGroup):
        NativeToQMove(ARCH).emit(resets)


def test_mismatched_move_to_is_caught_by_type_refinement():
    @physical.kernel(verify=False)
    def bad():
        q = squin.qalloc(2)
        arrange.move_to(q, ilist.IList([A]))  # type: ignore[arg-type]

    with pytest.raises(ValidationErrorGroup):
        NativeToQMove(ARCH).emit(bad)


def test_original_kernels_are_untouched():
    before = main.print_str()
    NativeToQMove(ARCH, SUBROUTINES).emit(main)
    assert main.print_str() == before


@squin.kernel
def bottom_result():
    qs = squin.qalloc(3)
    # measure takes one Qubit, so its result is Bottom.
    m = squin.measure(qs)  # type: ignore[arg-type]
    x = m[0]  # type: ignore[index]
    squin.h(qs[0])
    return x


def test_bottom_typed_result_is_reported_as_a_type_error():
    with pytest.raises(ValidationErrorGroup) as excinfo:
        NativeToQMove(ARCH).emit(bottom_result)
    messages = [str(err.args[0]) for err in excinfo.value.errors]
    assert any("no valid type" in m and "Bottom" in m for m in messages)
    assert not any(m.startswith("V1") for m in messages)


def test_bottom_typed_value_is_not_threaded_as_state():
    out = NativeToQMove(ARCH).emit(bottom_result, no_raise=True)
    result = ValidationSuite([get_qmove_validation(ARCH, ZonedPolicy())]).validate(out)
    assert [err for errs in result.errors.values() for err in errs] == []


@squin.kernel
def sub_a(qs: ilist.IList[Qubit, Literal[2]]):
    squin.cx(qs[0], qs[1])


@squin.kernel
def sub_b(qs: ilist.IList[Qubit, Literal[2]]):
    squin.cz(qs[0], qs[1])


@squin.kernel
def calls_both():
    qs = squin.qalloc(2)
    sub_a(qs)
    sub_b(qs)


def test_frame_errors_are_reported_across_all_subroutines():
    wrong_shape = Frame(FrameShape(((0, 1),)), (A,))  # the parameter has 2 slots
    subroutines: dict[ir.Method, Frame | None] = {
        sub_a: wrong_shape,
        sub_b: wrong_shape,
    }
    with pytest.raises(ValidationErrorGroup) as excinfo:
        NativeToQMove(ARCH, subroutines).emit(calls_both)
    messages = [str(err.args[0]) for err in excinfo.value.errors]
    assert len([m for m in messages if m.startswith("F2:")]) == 2


def test_no_raise_returns_despite_frame_errors():
    wrong_shape = Frame(FrameShape(((0, 1),)), (A,))
    NativeToQMove(ARCH, {sub_a: wrong_shape}).emit(calls_both, no_raise=True)


@pytest.mark.parametrize(
    "listed", [(sub, nested, sub_a, sub_b), (sub_b, sub_a, nested, sub)]
)
def test_subroutines_are_lowered_in_listed_order(monkeypatch, listed):
    # A frozenset would order them by id hash, which varies from run to run.
    seen: list[tuple[ir.Method, ...]] = []
    real = native_to_qmove.lower_to_native

    def spy(entry, subroutines, *args, **kwargs):
        subroutines = tuple(subroutines)
        seen.append(subroutines)
        return real(entry, subroutines, *args, **kwargs)

    monkeypatch.setattr(native_to_qmove, "lower_to_native", spy)
    NativeToQMove(ARCH, dict.fromkeys(listed)).emit(main)
    assert seen == [listed]


@squin.kernel
def allocating_sub(qs: ilist.IList[Qubit, Literal[1]]):
    extra = squin.qalloc(1)
    squin.cz(qs[0], extra[0])


@squin.kernel
def calls_allocating_sub():
    qs = squin.qalloc(1)
    allocating_sub(qs)


def test_whole_machine_subroutine_may_allocate():
    out = NativeToQMove(ARCH, {allocating_sub: MachineFrame()}).emit(
        calls_allocating_sub
    )
    callee = top_level(_callees(out, qmove.Invoke)["allocating_sub"])[0]
    assert isinstance(callee, qmove.Enter) and callee.frame == MachineFrame()


def test_allocation_in_a_partial_or_unframed_subroutine_is_rejected():
    with pytest.raises(ValidationErrorGroup):
        NativeToQMove(ARCH, {allocating_sub: None}).emit(calls_allocating_sub)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --no-sync pytest python/tests/rewrite/test_qmove_state.py python/tests/validation/test_qmove_input.py python/tests/validation/test_qmove.py python/tests/test_transform_native_to_qmove.py -q`
Expected: many FAILURES. Examples are `TypeError: thread_method() missing 1 required keyword-only argument: 'subroutine'`, and assertion errors on the new V2, call-rule and allocation messages.

- [ ] **Step 3: Write the implementation.** Replace the contents of the four modules.

`python/bloqade/lanes/rewrite/qmove_state.py`:

```python
"""Thread the machine state through a method's ``load; qmove.X; store`` chains.

After ``RewriteNativeToQMove`` every stateful statement sits in its own
``load``/``store`` pair. ``thread_method`` joins them into one chain per method,
opened by ``qmove.enter(frame)`` and closed by ``qmove.exit``, and threads the
state explicitly through ``scf.IfElse`` (both arms capture it and yield it back)
and ``scf.For`` (a loop-carried value, ahead of any existing ``iter_args``). No
``load``/``store`` is left afterwards.

This is a direct recursive traversal, not ``kirin.rewrite.Walk``: ``Walk`` visits
a region's blocks in reverse and a statement's regions before the statement
(``python/tests/rewrite/test_walk_order.py``), and threading needs execution
order. ``rewrite.state.RewriteLoadStore`` cannot be reused either: it finds
stateful statements by their ``ConsumesState``/``EmitsState`` traits, and a
threaded ``scf.IfElse`` has a ``State`` result but no trait (kirin owns ``scf``),
so it would skip the branch and drop its effect.
"""

from __future__ import annotations

from kirin import ir, types
from kirin.dialects import scf

from bloqade.lanes.dialects import move, qmove
from bloqade.lanes.dialects.qmove import Frame, MachineFrame
from bloqade.lanes.types import StateType


def _is_state(value: ir.SSAValue) -> bool:
    # Bottom is a subtype of everything; a value whose type inference failed is
    # not a machine state.
    return not value.type.is_subseteq(types.Bottom) and value.type.is_subseteq(
        StateType
    )


def touches_state(stmt: ir.Statement) -> bool:
    """Whether ``stmt``'s regions contain anything that reads or writes the state."""
    for region in stmt.regions:
        for inner in region.walk():
            if isinstance(inner, (move.Load, move.Store)):
                return True
            if any(map(_is_state, inner.args)) or any(map(_is_state, inner.results)):
                return True
    return False


def _advances_state(stmt: ir.Statement) -> bool:
    return bool(
        stmt.args
        and stmt.results
        and _is_state(stmt.args[0])
        and _is_state(stmt.results[0])
    )


def thread_block(
    block: ir.Block, state: ir.SSAValue, *, skip: ir.Statement | None = None
) -> ir.SSAValue:
    """Thread ``state`` through ``block``; return the state at its end."""
    for stmt in list(block.stmts):
        if stmt is skip:
            continue
        if isinstance(stmt, move.Load):
            stmt.result.replace_by(state)
            stmt.delete()
        elif isinstance(stmt, move.Store):
            stmt.delete()
        elif isinstance(stmt, scf.IfElse) and touches_state(stmt):
            state = _thread_if_else(stmt, state)
        elif isinstance(stmt, scf.For) and touches_state(stmt):
            state = _thread_for(stmt, state)
        elif _advances_state(stmt):
            state = stmt.results[0]
    return state


def _prepend_to_yield(block: ir.Block, value: ir.SSAValue) -> None:
    old = block.last_stmt
    assert isinstance(old, scf.Yield), f"expected scf.yield, got {old}"
    old.replace_by(scf.Yield(value, *old.values))


def _replace_scf(old: ir.Statement, new: ir.Statement) -> ir.SSAValue:
    new.insert_before(old)
    for old_result, new_result in zip(old.results, new.results[1:], strict=True):
        new_result.name = old_result.name
        new_result.type = old_result.type
        old_result.replace_by(new_result)
    old.delete()
    return new.results[0]


def _thread_if_else(stmt: scf.IfElse, state: ir.SSAValue) -> ir.SSAValue:
    if not stmt.else_body.blocks:
        # Python lowering always emits an else block; hand-built IR may not.
        block = ir.Block()
        block.args.append_from(stmt.cond.type)
        block.stmts.append(scf.Yield())
        stmt.else_body.blocks.append(block)
    for region in (stmt.then_body, stmt.else_body):
        body = region.blocks[0]
        _prepend_to_yield(body, thread_block(body, state))
    # kirin fixes a statement's result count when it is built, so rebuild it.
    # Reusing the regions requires detaching them first.
    then_body, else_body = stmt.then_body, stmt.else_body
    then_body.detach()
    else_body.detach()
    return _replace_scf(stmt, scf.IfElse(stmt.cond, then_body, else_body))


def _thread_for(stmt: scf.For, state: ir.SSAValue) -> ir.SSAValue:
    body = stmt.body.blocks[0]
    carried = body.args.insert_from(1, StateType, "state")
    _prepend_to_yield(body, thread_block(body, carried))
    region = stmt.body
    region.detach()
    return _replace_scf(stmt, scf.For(stmt.iterable, region, state, *stmt.initializers))


def thread_method(mt: ir.Method, *, frame: Frame | MachineFrame | None) -> None:
    """Open the chain with ``qmove.enter(frame)``, thread it, close it with ``exit``.

    Every method gets the pair, even one with no quantum statements, so "is a
    qmove method" is the same as "has an ``enter``" and a frame is never dropped.
    """
    block = mt.callable_region.blocks[0]
    opener = qmove.Enter(frame=frame)
    first = block.first_stmt
    assert first is not None
    opener.insert_before(first)
    final = thread_block(block, opener.result, skip=opener)
    terminator = block.last_stmt
    assert terminator is not None
    qmove.Exit(final).insert_before(terminator)
```

`python/bloqade/lanes/validation/qmove_input.py`:

```python
"""Reject input the qmove lowering does not support, reporting every problem.

Runs on native IR (after ``lower_to_native``), once per method.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, ClassVar

from bloqade.native.dialects.gate import dialect as native_gate, stmts as gate
from bloqade.squin import gate as squin_gate, noise as squin_noise
from kirin import ir, types
from kirin.dialects import func, ilist, py, scf
from kirin.validation import ValidationPass

from bloqade import qubit
from bloqade.gemini.common.dialects import arrange, qubit as gemini_qubit
from bloqade.gemini.logical.dialects.extensions import dialect as logical_extensions
from bloqade.gemini.logical.dialects.operations import dialect as logical_operations

LOGICAL_DIALECTS = frozenset({logical_operations, logical_extensions})
QUANTUM_DIALECTS = (
    frozenset(
        {
            squin_gate.dialect,
            squin_noise.dialect,
            native_gate,
            qubit.dialect,
            gemini_qubit.dialect,
            arrange.dialect,
        }
    )
    | LOGICAL_DIALECTS
)
ALLOCATION = (qubit.stmts.New, gemini_qubit.stmts.NewAt)
SUPPORTED = ALLOCATION + (
    qubit.stmts.Measure,
    qubit.stmts.IsZero,
    qubit.stmts.IsOne,
    qubit.stmts.IsLost,
    gate.CZ,
    gate.R,
    gate.Rz,
    arrange.stmts.MoveTo,
    arrange.stmts.Permute,
)
HIGHER_ORDER = (ilist.Map, ilist.ForEach, ilist.Foldl, ilist.Foldr, ilist.Scan)
ALLOCATION_MESSAGE = (
    "qubits may only be allocated under a whole-machine frame (the entry kernel)"
)


def _function_code(value: ir.SSAValue) -> ir.Statement | None:
    """The body behind a function value: a lambda, or a constant method."""
    if not isinstance(value, ir.ResultValue):
        return None
    owner = value.stmt
    if isinstance(owner, func.Lambda):
        return owner
    if (
        isinstance(owner, py.Constant)
        and isinstance(owner.value, ir.PyAttr)
        and isinstance(owner.value.data, ir.Method)
    ):
        return owner.value.data.code
    return None


def _reachable(code: ir.Statement) -> Iterator[ir.Statement]:
    """Statements in ``code`` and, transitively, in every method it calls."""
    seen: set[int] = set()
    stack = [code]
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        for stmt in current.walk():
            yield stmt
            if isinstance(stmt, func.Invoke):
                stack.append(stmt.callee.code)
            elif (
                isinstance(stmt, func.Call)
                and (callee := _function_code(stmt.callee)) is not None
            ):
                stack.append(callee)


def _applies_gates(code: ir.Statement) -> bool:
    return any(
        s.dialect in QUANTUM_DIALECTS and not isinstance(s, ALLOCATION)
        for s in _reachable(code)
    )


def _allocates(code: ir.Statement) -> bool:
    return any(isinstance(s, ALLOCATION) for s in _reachable(code))


def get_input_validation(
    may_allocate: bool, clones: frozenset[ir.Method] = frozenset()
) -> type[ValidationPass]:
    """``ValidationSuite`` builds passes with no arguments, hence the factory.

    ``may_allocate`` is true for a method under a ``MachineFrame`` (the entry
    kernel, or a whole-machine subroutine). ``clones`` are the lowered subroutine
    methods; a ``func.Invoke`` of any other method is a call that survived
    inlining.
    """

    @dataclass
    class QMoveInputValidation(ValidationPass):
        MAY_ALLOCATE: ClassVar[bool] = may_allocate
        CLONES: ClassVar[frozenset[ir.Method]] = clones

        def name(self) -> str:
            return "lanes.qmove.input"

        def run(self, method: ir.Method) -> tuple[Any, list[ir.ValidationError]]:
            errors: list[ir.ValidationError] = []

            def error(node: ir.Statement, message: str) -> None:
                errors.append(ir.ValidationError(node, message))

            regions = [method.callable_region]
            for stmt in method.callable_region.walk():
                if not isinstance(stmt, func.Lambda):
                    regions.extend(stmt.regions)

                if isinstance(stmt, func.Return) and isinstance(
                    stmt.parent_stmt, (scf.IfElse, scf.For)
                ):
                    error(stmt, "early return inside scf is not supported")
                if isinstance(stmt, qubit.stmts.Reset):
                    error(stmt, "qubit.reset is not supported")
                elif stmt.dialect in LOGICAL_DIALECTS:
                    error(
                        stmt,
                        f"{stmt.name} is a logical-pipeline statement; "
                        "qmove targets the physical pipeline",
                    )
                elif stmt.dialect in QUANTUM_DIALECTS and not isinstance(
                    stmt, SUPPORTED
                ):
                    error(stmt, f"{stmt.name} is not supported by the qmove lowering")

                if (
                    stmt.dialect in QUANTUM_DIALECTS or isinstance(stmt, func.Invoke)
                ) and any(r.type.is_subseteq(types.Bottom) for r in stmt.results):
                    # Type inference found no valid type, e.g. a wrongly typed
                    # argument. Reporting it here keeps the later state checks
                    # from blaming stdlib code for the user's type error.
                    error(
                        stmt,
                        f"{stmt.name} result has no valid type (Bottom); "
                        "check its argument types",
                    )
                if isinstance(stmt, func.Call):
                    error(stmt, "calls through a function value are not supported")
                if isinstance(stmt, func.Invoke) and stmt.callee not in self.CLONES:
                    # An un-inlined call stays off the state chain, silently
                    # dropping the effects of the gates behind it.
                    error(
                        stmt,
                        f"call of {stmt.callee.sym_name} survived inlining and is "
                        "not a listed subroutine",
                    )
                if not self.MAY_ALLOCATE and isinstance(stmt, ALLOCATION):
                    error(stmt, ALLOCATION_MESSAGE)
                if (
                    isinstance(stmt, HIGHER_ORDER)
                    and (code := _function_code(stmt.fn)) is not None
                ):
                    if _applies_gates(code):
                        error(
                            stmt,
                            f"{stmt.name} applies gates through a function value; "
                            "use a for loop",
                        )
                    elif not self.MAY_ALLOCATE and _allocates(code):
                        error(stmt, ALLOCATION_MESSAGE)

            for region in regions:
                if len(region.blocks) > 1:
                    error(
                        region.parent_node or method.code,
                        f"region has {len(region.blocks)} blocks; only structured "
                        "(scf) control flow is supported",
                    )
            return None, errors

    return QMoveInputValidation
```

`python/bloqade/lanes/validation/qmove.py`:

```python
"""Structural validation of qmove IR: state rules, call rules, frame rules.

Every check is a direct walk, not a ``Forward`` analysis (which only visits
reachable code), following ``FlatBlockValidation``. Checks that need qubit
identity (distinct qubits within a statement, the general case of F4, whether a
call's frame precondition holds) are left to later analyses.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, ClassVar

from bloqade.types import QubitType
from kirin import ir, types
from kirin.analysis import const
from kirin.dialects import func, ilist, scf
from kirin.validation import ValidationPass

from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.dialects import move, qmove
from bloqade.lanes.dialects.qmove import Frame, MachineFrame
from bloqade.lanes.types import StateType
from bloqade.lanes.validation.spectator import SpectatorPolicy, format_location

Path = list[tuple[ir.Statement, int]]


def _is_state(value: ir.SSAValue) -> bool:
    # Bottom is a subtype of everything; a value whose type inference failed is
    # not a machine state.
    return not value.type.is_subseteq(types.Bottom) and value.type.is_subseteq(
        StateType
    )


def is_qmove_method(mt: ir.Method) -> bool:
    """Whether ``mt`` has been lowered to qmove: it opens its chain with ``enter``."""
    return any(isinstance(s, qmove.Enter) for s in mt.callable_region.walk())


def method_frame(mt: ir.Method) -> Frame | MachineFrame | None:
    """The frame on ``mt``'s ``enter``; ``None`` for a hole or a non-qmove method."""
    for stmt in mt.callable_region.walk():
        if isinstance(stmt, qmove.Enter):
            return stmt.frame
    return None


def _state_values(mt: ir.Method) -> Iterator[ir.SSAValue]:
    for stmt in mt.callable_region.walk():
        yield from (r for r in stmt.results if _is_state(r))
        for region in stmt.regions:
            for block in region.blocks:
                yield from (a for a in block.args if _is_state(a))


def _defining_block(value: ir.SSAValue) -> ir.Block | None:
    if isinstance(value, ir.BlockArgument):
        return value.block
    assert isinstance(value, ir.ResultValue)
    return value.stmt.parent_block


def _path(stmt: ir.Statement, top: ir.Block | None) -> Path | None:
    """``(statement, region index)`` pairs enclosing ``stmt``, below ``top``."""
    path: Path = []
    block = stmt.parent_block
    while block is not top:
        if block is None:
            return None
        region = block.parent
        if region is None:
            return None
        owner = region.parent_node
        if not isinstance(owner, ir.Statement):
            return None
        path.append((owner, owner.regions.index(region)))
        block = owner.parent_block
    path.reverse()
    return path


def _mutually_exclusive(a: Path, b: Path) -> bool:
    """Whether two uses sit in different arms of a common ``scf.IfElse``."""
    for (stmt_a, region_a), (stmt_b, region_b) in zip(a, b):
        if stmt_a is not stmt_b:
            return False
        if region_a != region_b:
            return isinstance(stmt_a, scf.IfElse)
    return False  # one path is a prefix of the other: same execution path


def check_use_def(mt: ir.Method) -> list[ir.ValidationError]:
    """V1: every state is used, and no path consumes it twice (ignoring Store)."""
    errors = []
    for value in _state_values(mt):
        node = value.stmt if isinstance(value, ir.ResultValue) else mt.code
        if not value.uses:
            errors.append(
                ir.ValidationError(
                    node, "V1: state is never used; its update is dropped"
                )
            )
            continue
        top = _defining_block(value)
        consumers: list[tuple[ir.Statement, Path]] = []
        for use in value.uses:
            if isinstance(use.stmt, move.Store):
                continue
            path = _path(use.stmt, top)
            if path is None:
                continue
            if any(isinstance(s, (scf.For, func.Lambda)) for s, _ in path):
                errors.append(
                    ir.ValidationError(
                        use.stmt,
                        "V1: state consumed inside a loop body that does not define it",
                    )
                )
            consumers.append((use.stmt, path))
        for i, (first, first_path) in enumerate(consumers):
            for second, second_path in consumers[i + 1 :]:
                if not _mutually_exclusive(first_path, second_path):
                    errors.append(
                        ir.ValidationError(
                            second,
                            f"V1: state consumed twice on one path, by {first.name} "
                            f"and {second.name}",
                        )
                    )
    return errors


def check_chain_ends(mt: ir.Method) -> list[ir.ValidationError]:
    """V2: exactly one ``enter`` and one ``exit``, top level; no ``load``/``store``."""
    errors = []
    top = mt.callable_region.blocks[0]
    stmts = list(mt.callable_region.walk())
    for stmt in stmts:
        if isinstance(stmt, (move.Load, move.Store)):
            errors.append(
                ir.ValidationError(
                    stmt,
                    f"V2: {stmt.name} is not allowed in qmove IR; "
                    "a method's chain uses enter/exit",
                )
            )
    ends = [s for s in stmts if isinstance(s, (qmove.Enter, qmove.Exit))]
    for stmt in ends:
        if stmt.parent_block is not top:
            errors.append(
                ir.ValidationError(
                    stmt, f"V2: {stmt.name} must be in the method's top-level block"
                )
            )
    stateful = any(_is_state(v) for s in stmts for v in (*s.args, *s.results))
    if ends or stateful:
        for kind in (qmove.Enter, qmove.Exit):
            count = sum(isinstance(s, kind) for s in ends)
            if count != 1:
                errors.append(
                    ir.ValidationError(
                        mt.code,
                        f"V2: a qmove method needs exactly one {kind.name}, "
                        f"found {count}",
                    )
                )
    return errors


def _const(value: ir.SSAValue) -> Any:
    hint = value.hints.get("const")
    return hint.data if isinstance(hint, const.Value) else None


def _list_len(typ: types.TypeAttribute) -> int | None:
    if (
        isinstance(typ, types.Generic)
        and typ.is_subseteq(ilist.IListType)
        and isinstance(typ.vars[1], types.Literal)
        and isinstance(typ.vars[1].data, int)
    ):
        return typ.vars[1].data
    return None


def check_statements(mt: ir.Method) -> list[ir.ValidationError]:
    """V3: per-statement checks wherever the operands are constant."""
    errors = []
    for stmt in mt.callable_region.walk():
        if isinstance(stmt, qmove.CZ):
            controls, targets = _list_len(stmt.controls.type), _list_len(
                stmt.targets.type
            )
            if controls is not None and targets is not None and controls != targets:
                errors.append(
                    ir.ValidationError(
                        stmt, f"V3: cz has {controls} controls but {targets} targets"
                    )
                )
        elif isinstance(stmt, qmove.Permute):
            perm = _const(stmt.perm)
            if perm is not None and sorted(int(p) for p in perm) != list(
                range(len(perm))
            ):
                errors.append(
                    ir.ValidationError(
                        stmt, f"V3: perm {tuple(perm)} is not a permutation"
                    )
                )
        elif isinstance(stmt, qmove.MoveTo):
            locations, count = _const(stmt.locations), _list_len(stmt.qubits.type)
            if locations is not None and count is not None and len(locations) != count:
                errors.append(
                    ir.ValidationError(
                        stmt,
                        f"V3: move_to has {len(locations)} locations for {count} qubits",
                    )
                )
    return errors


def check_calls(mt: ir.Method) -> list[ir.ValidationError]:
    """Call rules, plus F5's whole-machine half (it holds whatever the caller's frame)."""
    errors = []
    caller_is_machine = isinstance(method_frame(mt), MachineFrame)
    for stmt in mt.callable_region.walk():
        if isinstance(stmt, (qmove.Invoke, qmove.Prepare)):
            if not is_qmove_method(stmt.callee):
                errors.append(
                    ir.ValidationError(
                        stmt,
                        f"{stmt.name} target {stmt.callee.sym_name} is not a qmove "
                        "method",
                    )
                )
            elif (
                isinstance(method_frame(stmt.callee), MachineFrame)
                and not caller_is_machine
            ):
                errors.append(
                    ir.ValidationError(
                        stmt,
                        f"F5: whole-machine method {stmt.callee.sym_name} can only "
                        "be called from a whole-machine method",
                    )
                )
        elif isinstance(stmt, func.Invoke) and is_qmove_method(stmt.callee):
            errors.append(
                ir.ValidationError(
                    stmt,
                    f"qmove method {stmt.callee.sym_name} must be called with "
                    "qmove.invoke",
                )
            )
    return errors


def expected_param_slots(
    mt: ir.Method,
) -> tuple[tuple[tuple[int, int], ...], list[str]]:
    slots: list[tuple[int, int]] = []
    problems: list[str] = []
    for index, typ in enumerate(mt.arg_types):
        if typ.is_subseteq(QubitType):
            slots.append((index, 1))
        elif typ.is_subseteq(ilist.IListType[QubitType, types.Any]):
            length = _list_len(typ)
            if length is None:
                problems.append(
                    f"parameter {index} is {typ}; a framed subroutine needs "
                    "IList[Qubit, Literal[N]]"
                )
            else:
                slots.append((index, length))
    return tuple(slots), problems


def check_frame(mt: ir.Method, arch: ArchSpec) -> list[ir.ValidationError]:
    """F1-F5, for a method under a partial ``Frame``.

    A ``MachineFrame`` is the whole machine and has nothing to check; a hole
    (``None``) is not known yet. F5's whole-machine half is in ``check_calls``.

    F4 (relabels compose to the identity) is judged only when every relabel
    ``Permute`` (``insert_moves=False``) in the method is a top-level, constant,
    well-formed permutation of a parameter. Without qubit identity, a relabel
    that is nested, non-constant, malformed or on a non-parameter operand cannot
    be composed with the others, so F4 abstains for the whole method rather than
    judge an inconsistent subset.
    """
    frame = method_frame(mt)
    if not isinstance(frame, Frame):
        return []
    errors: list[ir.ValidationError] = []

    def error(node: ir.Statement, message: str) -> None:
        errors.append(ir.ValidationError(node, message))

    if len(frame.binding) != frame.shape.total_slots:
        error(
            mt.code,
            f"F1: binding has {len(frame.binding)} locations for "
            f"{frame.shape.total_slots} slots",
        )
    if len(set(frame.binding)) != len(frame.binding):
        error(mt.code, "F1: binding locations are not distinct")
    for problem in arch.check_location_group(list(set(frame.binding))):
        error(mt.code, f"F1: {problem}")
    for zone in frame.effects.cz_zones | frame.effects.measure_zones:
        if not 0 <= zone.zone_id < len(arch.zones):
            error(mt.code, f"F1: zone {zone.zone_id} does not exist")

    slots, problems = expected_param_slots(mt)
    for problem in problems:
        error(mt.code, f"F2: {problem}")
    if not problems and tuple(sorted(frame.shape.param_slots)) != slots:
        error(
            mt.code,
            f"F2: frame slots {frame.shape.param_slots} do not match parameters {slots}",
        )

    footprint = frame.footprint
    effects = frame.effects
    relabels: dict[ir.SSAValue, list[tuple[int, ...]]] = {}
    f4_abstains = False
    top = mt.callable_region.blocks[0]
    for stmt in mt.callable_region.walk():
        if isinstance(stmt, qmove.MoveTo):
            for loc in _const(stmt.locations) or ():
                if loc not in footprint:
                    error(
                        stmt,
                        f"F3: move_to target {format_location(loc)} is outside the frame",
                    )
        elif isinstance(stmt, move.CZ) and stmt.zone_address not in effects.cz_zones:
            error(
                stmt,
                f"F3: cz in zone {stmt.zone_address.zone_id} is not in the frame's effects",
            )
        elif isinstance(stmt, (move.Measure, move.EndMeasure)):
            if not set(stmt.zone_addresses) <= effects.measure_zones:
                error(stmt, "F3: measurement zone is not in the frame's effects")
        elif (
            isinstance(stmt, (move.GlobalR, move.GlobalRz))
            and not effects.global_pulses
        ):
            error(stmt, "F3: global pulse is not in the frame's effects")
        elif isinstance(stmt, move.Move):
            for lane in stmt.lanes:
                src, dst = arch.get_endpoints(lane)
                if src not in footprint or dst not in footprint:
                    error(stmt, f"F3: lane {lane} leaves the frame")
        elif isinstance(stmt, qmove.Permute) and not stmt.insert_moves:
            perm = _const(stmt.perm)
            if (
                stmt.parent_block is top
                and perm is not None
                and isinstance(stmt.qubits, ir.BlockArgument)
                and stmt.qubits.block is top
                and sorted(int(p) for p in perm) == list(range(len(perm)))
            ):
                relabels.setdefault(stmt.qubits, []).append(tuple(int(p) for p in perm))
            else:
                f4_abstains = True
        elif isinstance(stmt, qmove.Invoke):
            inner = method_frame(stmt.callee)
            if isinstance(inner, Frame) and (
                not inner.footprint <= footprint
                or not inner.effects.is_subset_of(effects)
            ):
                error(
                    stmt,
                    f"F5: frame of {stmt.callee.sym_name} is not inside this frame",
                )

    if not f4_abstains:
        for perms in relabels.values():
            # Validate all perms have the same length before composition.
            perm_len = len(perms[0])
            if any(len(p) != perm_len for p in perms):
                continue
            net = list(range(perm_len))
            for perm in perms:
                net = [net[i] for i in perm]
            if net != list(range(len(net))):
                error(
                    mt.code,
                    f"F4: relabels {perms} leave the binding permuted at exit",
                )
    return errors


def get_qmove_validation(
    arch: ArchSpec, policy: SpectatorPolicy
) -> type[ValidationPass]:
    """``ValidationSuite`` builds passes with no arguments, hence the factory."""

    @dataclass
    class QMoveValidation(ValidationPass):
        ARCH: ClassVar[ArchSpec] = arch
        POLICY: ClassVar[SpectatorPolicy] = policy

        def name(self) -> str:
            return "lanes.qmove.validation"

        def run(self, method: ir.Method) -> tuple[Any, list[ir.ValidationError]]:
            errors = (
                check_use_def(method)
                + check_chain_ends(method)
                + check_statements(method)
                + check_calls(method)
                + check_frame(method, self.ARCH)
            )
            if isinstance(frame := method_frame(method), Frame):
                errors += [
                    ir.ValidationError(method.code, problem)
                    for problem in self.POLICY.check_frame(frame, self.ARCH)
                ]
            return None, errors

    return QMoveValidation
```

`python/bloqade/lanes/transform/native_to_qmove.py`:

```python
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from kirin import ir, passes, rewrite
from kirin.ir.exception import ValidationErrorGroup
from kirin.validation import ValidationSuite
from kirin.validation.validationpass import ValidationResult

from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.dialects.qmove import Frame, MachineFrame
from bloqade.lanes.rewrite.native2qmove import RewriteNativeToQMove
from bloqade.lanes.rewrite.qmove_state import thread_method
from bloqade.lanes.rewrite.refine_qubit_types import RefineQubitTypes
from bloqade.lanes.transform.qmove_frontend import lower_to_native, unlisted_recursion
from bloqade.lanes.validation.qmove import get_qmove_validation
from bloqade.lanes.validation.qmove_input import get_input_validation
from bloqade.lanes.validation.spectator import SpectatorPolicy, ZonedPolicy


@dataclass
class NativeToQMove:
    """Lower a physical kernel to qmove IR, keeping ``scf`` and subroutine calls.

    ``subroutines`` maps each kernel to keep as a call to its pinned frame: a
    partial ``Frame``, ``MachineFrame()`` for a whole-machine subroutine (which
    may allocate, and only whole-machine methods may call), or ``None`` for a
    hole that later synthesis fills. The entry kernel always gets
    ``MachineFrame()``. Every other call is inlined;
    nothing is unrolled. The result is the entry method; subroutine clones are
    reachable through its ``qmove.invoke`` statements. No placement or move
    synthesis happens here.
    """

    arch_spec: ArchSpec
    subroutines: Mapping[ir.Method, Frame | MachineFrame | None] = field(
        default_factory=dict
    )
    policy: SpectatorPolicy = field(default_factory=ZonedPolicy)

    def emit(self, mt: ir.Method, no_raise: bool = False) -> ir.Method:
        if cycles := unlisted_recursion(mt, frozenset(self.subroutines)):
            raise ValidationErrorGroup(
                "NativeToQMove: recursive kernels must be listed as subroutines",
                errors=[
                    ir.ValidationError(
                        mt.code, f"recursive kernel is not a subroutine: {cycle}"
                    )
                    for cycle in cycles
                ],
            )

        # A tuple keeps the listed order; a frozenset would order by id hash.
        program = lower_to_native(
            mt, tuple(self.subroutines), self.arch_spec, no_raise=no_raise
        )
        roles: list[tuple[ir.Method, Frame | MachineFrame | None]] = [
            (program.entry, MachineFrame())
        ]
        roles += [
            (clone, self.subroutines[original])
            for original, clone in program.subroutines.items()
        ]

        clones = frozenset(program.subroutines.values())
        errors: list[ir.ValidationError] = []
        for method, frame in roles:
            may_allocate = isinstance(frame, MachineFrame)
            result = ValidationSuite(
                [get_input_validation(may_allocate, clones)]
            ).validate(method)
            errors += [err for errs in result.errors.values() for err in errs]
        if errors and not no_raise:
            raise ValidationErrorGroup(
                "NativeToQMove: unsupported input", errors=errors
            )

        for method, frame in roles:
            rewrite.Walk(RewriteNativeToQMove(clones)).rewrite(method.code)
            thread_method(method, frame=frame)
            rewrite.Fixpoint(rewrite.Walk(rewrite.DeadCodeElimination())).rewrite(
                method.code
            )
            passes.TypeInfer(method.dialects, no_raise=no_raise)(method)
            RefineQubitTypes(method.dialects, no_raise=no_raise)(method)

        if not no_raise:
            # Validate every method before raising, so one run reports all problems.
            validation = get_qmove_validation(self.arch_spec, self.policy)
            merged: dict[str, list[ir.ValidationError]] = {}
            for method, _ in roles:
                result = ValidationSuite([validation]).validate(method)
                for name, errs in result.errors.items():
                    merged.setdefault(name, []).extend(errs)
            ValidationResult(merged).raise_if_invalid()
            for method, _ in roles:
                method.verify()
                method.verify_type()
        return program.entry
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest python/tests/rewrite/test_qmove_state.py python/tests/validation/test_qmove_input.py python/tests/validation/test_qmove.py python/tests/test_transform_native_to_qmove.py -v`
Expected: all PASS. That is 6 in `test_qmove_state.py`, 15 in `test_qmove_input.py`, 30 in `validation/test_qmove.py` and 18 in `test_transform_native_to_qmove.py`.

Then run every qmove test file together:

```bash
uv run --no-sync pytest python/tests/dialects/test_qmove.py python/tests/test_qmove_helpers.py python/tests/rewrite/test_const_call_to_invoke.py python/tests/rewrite/test_refine_qubit_types.py python/tests/test_qmove_frontend.py python/tests/validation/test_qmove_input.py python/tests/rewrite/test_native2qmove.py python/tests/rewrite/test_qmove_state.py python/tests/validation/test_spectator.py python/tests/validation/test_qmove.py python/tests/test_transform_native_to_qmove.py python/tests/test_import_cycles.py -q
```

Expected: `119 passed`.

- [ ] **Step 5: Confirm no `Load`/`Store` reaches qmove IR.** `grep -n "move.Load\|move.Store" python/bloqade/lanes/rewrite/qmove_state.py python/bloqade/lanes/rewrite/native2qmove.py` should show them only as the transient per-statement placeholders that `thread_block` removes. They must not appear in `thread_method`.

- [ ] **Step 6: Lint and commit**

```bash
git add python/bloqade/lanes/rewrite/qmove_state.py python/bloqade/lanes/validation/qmove_input.py python/bloqade/lanes/validation/qmove.py python/bloqade/lanes/transform/native_to_qmove.py python/tests/rewrite/test_qmove_state.py python/tests/validation/test_qmove_input.py python/tests/validation/test_qmove.py python/tests/test_transform_native_to_qmove.py
git commit -m "feat(python): open every qmove method with enter/exit"
```

---

### Task 3: Final verification

- [ ] **Step 1:** Run `just test-python`. It takes about 7–10 minutes; use a timeout of up to 600000 ms.
  Expected: no failures, at least 2685 + 7 tests passing (this plan adds 7 tests).
- [ ] **Step 2:** Run the lint step.
  Expected: clean, with no reformatting of files outside this plan.
- [ ] **Step 3:** Run `git diff --name-status 72f8cee6..HEAD -- python/bloqade`.
  Expected: only `M` lines, and only for the seven modules listed in File Structure.
