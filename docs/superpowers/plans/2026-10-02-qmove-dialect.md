# qmove Dialect Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the `lanes.qmove` dialect: qubit-addressed, state-threaded quantum statements. Also add a `NativeToQMove` transform that lowers physical squin/native kernels into it. The transform keeps `scf` control flow and subroutine calls instead of unrolling and inlining them, and verifies the result.

**Architecture:** The per-method frontend runs first. It inlines every call except the listed subroutines, rewrites squin into native IR, then runs `TypeInfer` and `RefineQubitTypes`. Next comes an input check. Then each native gate is lowered locally into `load; qmove.X; store`. A recursive threading pass joins those chains and threads the state through `scf.IfElse`/`scf.For`. Subroutines open and close their chains with `qmove.enter`/`exit`, which carry an optional frame. The last step is a structural verifier (V1–V3, call rules, F1–F5) plus a pluggable spectator policy. Nothing in `place`, `move` or the existing pipelines changes.

**Tech Stack:** Python 3.10+, kirin-toolchain 0.22 (IR framework), bloqade-circuit 0.15 (squin, native gates, `AddressAnalysis`), pytest.

**Spec:** `docs/superpowers/specs/2026-10-02-qmove-dialect-design.md`. Read its "What `State` means", "Calls and frames" and "Validation" sections before starting.

## Global Constraints

- Target the **physical** pipeline only. Logical-pipeline statements (`gemini.logical.*`, `gemini.extensions.*`) are rejected, never lowered.
- Do not modify any existing module. The only existing file this plan edits is `python/tests/test_import_cycles.py`, to register new modules.
- Python must stay compatible with 3.10: ruff `target-version = "py310"`, so no PEP 695 generics and no `typing.Self`.
- Always run Python tooling as `uv run --no-sync ...`. A plain `uv run` re-syncs the environment and silently replaces the locally built `_native` extension.
- Never use kirin's `is_structurally_equal` to compare IR that contains regions. It never compares nested region contents or result types. Use `tests._qmove_helpers.blocks_equal`.
- `kirin.ir.exception.ValidationErrorGroup` subclasses `BaseException`, not `Exception`. Catch it explicitly: `pytest.raises(ValidationErrorGroup)`.
- `ValidationSuite` constructs each pass with no arguments. Configure passes through a factory that returns a class with `ClassVar`s (`get_input_validation`, `get_qmove_validation`).
- Never assign to a variadic argument field such as `scf.Yield.values`; kirin's generated setter is broken. Build a new statement and `replace_by` instead. Before rebuilding an `scf.IfElse`/`scf.For` around its existing regions, call `region.detach()` on each one.
- `bloqade.gemini.physical.kernel` inlines calls when a kernel is decorated (`inline=True` by default). A physical kernel that calls a subroutine must use `inline=False`; `squin.kernel` does not inline.
- Commit messages follow Conventional Commits (see `AGENT.md`): `feat(python): ...`, `test(python): ...`.

## Setup (once, before Task 1)

```bash
uv sync --dev --all-extras --index-strategy=unsafe-best-match
```

```bash
just develop-python
```

Check that the extension matches the sources:

```bash
uv run --no-sync python -c "import bloqade.lanes.bytecode as b; print(b.BoundStats)"
```

Expected: `<class 'bloqade.lanes.bytecode._native.BoundStats'>`. An `ImportError` means the build is stale; rerun `just develop-python`.

## File Structure

| File | Responsibility |
|---|---|
| `python/bloqade/lanes/dialects/qmove/__init__.py` | Re-exports the dialect, statements and frame types |
| `python/bloqade/lanes/dialects/qmove/_dialect.py` | `dialect = ir.Dialect("lanes.qmove")` |
| `python/bloqade/lanes/dialects/qmove/frame.py` | `FrameShape`, `Effects`, `Frame` (frozen, hashable) |
| `python/bloqade/lanes/dialects/qmove/stmts.py` | `CZ`, `R`, `Rz`, `MoveTo`, `Permute`, `Measure`, `Enter`, `Exit`, `Prepare`, `Invoke` |
| `python/bloqade/lanes/rewrite/const_call_to_invoke.py` | `ConstCallToInvoke`: `func.call` of a constant method becomes `func.invoke` |
| `python/bloqade/lanes/rewrite/refine_qubit_types.py` | `RefineQubitTypes`: narrows qubit types from `AddressAnalysis` |
| `python/bloqade/lanes/transform/qmove_frontend.py` | `lower_to_native`, `unlisted_recursion`, `NativeProgram` |
| `python/bloqade/lanes/validation/qmove_input.py` | `get_input_validation`: rejects unsupported input |
| `python/bloqade/lanes/rewrite/native2qmove.py` | `RewriteNativeToQMove`: local `load; qmove.X; store` lowering |
| `python/bloqade/lanes/rewrite/qmove_state.py` | `thread_method`, `thread_block`: state threading through `scf` |
| `python/bloqade/lanes/validation/spectator.py` | `SpectatorPolicy`, `ZonedPolicy`, `SingleZonePolicy` |
| `python/bloqade/lanes/validation/qmove.py` | `get_qmove_validation`: V1–V3, call rules, F1–F5, policy |
| `python/bloqade/lanes/transform/native_to_qmove.py` | `NativeToQMove`, the orchestrating transform |
| `python/tests/_qmove_helpers.py` | `blocks_equal`, `assert_methods_match`, `erase_qmove`, `first_of`, `statements_of`, `top_level` |
| `python/tests/test_import_cycles.py` (modify) | Register `bloqade.lanes.dialects.qmove` and `bloqade.lanes.transform.native_to_qmove` |

Tests live in `python/tests/` next to existing ones: `dialects/test_qmove.py`, `test_qmove_helpers.py`, `rewrite/test_const_call_to_invoke.py`, `rewrite/test_refine_qubit_types.py`, `test_qmove_frontend.py`, `validation/test_qmove_input.py`, `rewrite/test_native2qmove.py`, `rewrite/test_qmove_state.py`, `validation/test_spectator.py`, `validation/test_qmove.py`, `test_transform_native_to_qmove.py`.

Every task ends with this lint step, run on the files that task touched (all four must pass before committing):

```bash
uv run --no-sync isort python && uv run --no-sync black python && uv run --no-sync ruff check python && uv run --no-sync pyright python
```

---

### Task 1: The `qmove` dialect

**Files:**
- Create: `python/bloqade/lanes/dialects/qmove/_dialect.py`, `frame.py`, `stmts.py`, `__init__.py`
- Modify: `python/tests/test_import_cycles.py`, in the module list of `test_lanes_module_imports_in_fresh_interpreter`
- Test: `python/tests/dialects/test_qmove.py`

**Interfaces:**
- Consumes: `bloqade.lanes.dialects.move.StatefulStatement`, `ConsumesState`, `EmitsState`; `bloqade.lanes.types.StateType`; `bloqade.lanes.dialects.arch.LocationAddressType`.
- Produces:
  - `qmove.dialect`.
  - `qmove.FrameShape(param_slots: tuple[tuple[int, int], ...], scratch_slots: int = 0)` with `.total_slots`.
  - `qmove.Effects(cz_zones=frozenset(), measure_zones=frozenset(), global_pulses=False)` with `.is_subset_of(other)`.
  - `qmove.Frame(shape, binding: tuple[LocationAddress, ...], effects=Effects())` with `.footprint`.
  - Statements (positional constructor order shown):
    - `CZ(state, controls, targets)`
    - `R(state, axis_angle, rotation_angle, qubits)`
    - `Rz(state, rotation_angle, qubits)`
    - `MoveTo(state, qubits, locations, *, multi_move_warning=True)`
    - `Permute(state, qubits, perm, *, insert_moves=False)`
    - `Measure(state, qubits)`, with results `result` (State) and `measurements`
    - `Enter(*, frame=None)`, with result `result`
    - `Exit(state)`
    - `Prepare(state, inputs: tuple, *, callee)`
    - `Invoke(state, inputs: tuple, *, callee)`, with results `result` (State) and `value`

- [ ] **Step 1: Write the failing test**

Create `python/tests/dialects/test_qmove.py`:

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest python/tests/dialects/test_qmove.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'bloqade.lanes.dialects.qmove'`.

- [ ] **Step 3: Write the implementation**

`python/bloqade/lanes/dialects/qmove/_dialect.py`:

```python
from kirin import ir

dialect = ir.Dialect(name="lanes.qmove")
```

`python/bloqade/lanes/dialects/qmove/frame.py`:

```python
"""Subroutine frames: where a subroutine's arguments must be, and what it may do.

A frame is a *shape* (how many slots each qubit parameter takes, plus scratch)
and a *binding* of those slots to concrete locations. Keeping the two apart
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
from .frame import Frame

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
    """Open a subroutine's chain; ``move.load`` plus the frame's precondition.

    ``frame=None`` is a hole that later synthesis fills.
    """

    traits = frozenset({EmitsState(True)})
    frame: Frame | None = info.attribute(default=None)
    result: ir.ResultValue = info.result(StateType)


@statement(dialect=dialect)
class Exit(ir.Statement):
    """Close a subroutine's chain; ``move.store`` plus the frame's postcondition."""

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
from .frame import Effects as Effects, Frame as Frame, FrameShape as FrameShape
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

In `python/tests/test_import_cycles.py`, add `"bloqade.lanes.dialects.qmove",` to the parametrized module list right after `"bloqade.lanes.dialects.move",`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest python/tests/dialects/test_qmove.py python/tests/test_import_cycles.py -v`
Expected: all PASS (7 in `test_qmove.py`).

- [ ] **Step 5: Lint and commit**

Run the lint step from the File Structure section, then:

```bash
git add python/bloqade/lanes/dialects/qmove python/tests/dialects/test_qmove.py python/tests/test_import_cycles.py
git commit -m "feat(python): add the qmove dialect"
```

---

### Task 2: Test helpers: a deep IR comparator and an eraser

kirin's `is_structurally_equal` never compares nested region contents or result types. The new test shows this directly: it asserts that kirin considers two loops with different bodies equal. Every later test that compares IR uses `blocks_equal`. `erase_qmove` undoes the lowering, so Task 11 can check that the lowering only adds state plumbing.

**Files:**
- Create: `python/tests/_qmove_helpers.py`
- Test: `python/tests/test_qmove_helpers.py`

**Interfaces:**
- Consumes: Task 1's `qmove` statements.
- Produces:
  - `blocks_equal(a: ir.Block, b: ir.Block, ctx=None) -> str | None`: `None` if equal, else the first difference.
  - `assert_methods_match(got: ir.Method, expected: ir.Method) -> None`.
  - `erase_qmove(block: ir.Block) -> None`, applied in place.
  - `statements_of(node: ir.Method | ir.Statement, kind: type[T]) -> list[T]`.
  - `first_of(node, kind: type[T]) -> T`.
  - `top_level(mt: ir.Method) -> list[ir.Statement]`.

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_qmove_helpers.py`:

```python
from kirin import ir, types
from kirin.dialects import py, scf
from tests._qmove_helpers import blocks_equal

ITERABLE = ir.TestValue(types.Any)


def _loop(body_value: int) -> ir.Block:
    body = ir.Block()
    body.args.append_from(types.Int, "i")
    body.stmts.append(py.Constant(body_value))
    body.stmts.append(scf.Yield())
    return ir.Block([scf.For(ITERABLE, ir.Region(body))])


def test_identical_ir_matches():
    assert blocks_equal(_loop(1), _loop(1)) is None


def test_difference_inside_a_nested_region_is_caught():
    a, b = _loop(1), _loop(2)
    # kirin's own comparison misses this: it never looks inside nested regions.
    assert a.is_structurally_equal(b)
    assert blocks_equal(a, b) is not None


def test_result_type_difference_is_caught():
    float_typed = py.Constant(1)
    float_typed.result.type = types.Float
    assert blocks_equal(ir.Block([py.Constant(1)]), ir.Block([float_typed])) is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest python/tests/test_qmove_helpers.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'tests._qmove_helpers'`.

- [ ] **Step 3: Write the helpers**

Create `python/tests/_qmove_helpers.py`:

```python
"""Test helpers for the qmove lowering: a deep IR comparator and an eraser.

kirin's ``is_structurally_equal`` cannot compare lowered IR. ``Region``'s version
records every block pair in its context before comparing, so ``Block``'s
version returns early and nested region contents are never compared, and result
types are never compared. ``blocks_equal`` recurses into every nested region
itself, block by block.
"""

from __future__ import annotations

from typing import TypeVar

from bloqade.native.dialects.gate import stmts as gate
from kirin import ir
from kirin.dialects import func, scf

from bloqade import qubit
from bloqade.gemini.common.dialects.arrange import stmts as arrange
from bloqade.lanes.dialects import move, qmove
from bloqade.lanes.types import StateType

Context = dict[ir.SSAValue, ir.SSAValue]
T = TypeVar("T", bound=ir.Statement)


def statements_of(node: ir.Method | ir.Statement, kind: type[T]) -> list[T]:
    """Every ``kind`` statement inside ``node``, nested regions included."""
    root = node.code if isinstance(node, ir.Method) else node
    return [s for s in root.walk() if isinstance(s, kind)]


def first_of(node: ir.Method | ir.Statement, kind: type[T]) -> T:
    first, *_ = statements_of(node, kind)
    return first


def top_level(mt: ir.Method) -> list[ir.Statement]:
    return list(mt.callable_region.blocks[0].stmts)


def _attr_equal(a: ir.Attribute, b: ir.Attribute) -> bool:
    if isinstance(a, ir.PyAttr) and isinstance(b, ir.PyAttr):
        if isinstance(a.data, ir.Method) and isinstance(b.data, ir.Method):
            return a.data.sym_name == b.data.sym_name
        return a.type == b.type and a.data == b.data
    return a == b


def _stmts_equal(a: ir.Statement, b: ir.Statement, ctx: Context) -> str | None:
    if type(a) is not type(b):
        return f"{type(a).__name__} != {type(b).__name__}"
    if (len(a.args), len(a.results), len(a.regions)) != (
        len(b.args),
        len(b.results),
        len(b.regions),
    ):
        return f"{a.name}: arity differs"
    if a.attributes.keys() != b.attributes.keys():
        return f"{a.name}: attribute names differ"
    for key in a.attributes:
        if not _attr_equal(a.attributes[key], b.attributes[key]):
            return f"{a.name}: attribute {key!r} differs"
    for x, y in zip(a.args, b.args):
        if ctx.get(x, x) is not y:
            return f"{a.name}: operand differs"
    for ra, rb in zip(a.regions, b.regions):
        if len(ra.blocks) != len(rb.blocks):
            return f"{a.name}: region has {len(ra.blocks)} != {len(rb.blocks)} blocks"
        for ba, bb in zip(ra.blocks, rb.blocks):
            if (err := blocks_equal(ba, bb, ctx)) is not None:
                return f"in {a.name}: {err}"
    for x, y in zip(a.results, b.results):
        if x.type != y.type:
            return f"{a.name}: result type {x.type} != {y.type}"
        ctx[x] = y
    return None


def blocks_equal(a: ir.Block, b: ir.Block, ctx: Context | None = None) -> str | None:
    """``None`` if the blocks match, recursively; otherwise the first difference."""
    ctx = {} if ctx is None else ctx
    if len(a.args) != len(b.args):
        return f"{len(a.args)} != {len(b.args)} block arguments"
    for x, y in zip(a.args, b.args):
        if x.type != y.type:
            return f"block argument type {x.type} != {y.type}"
        ctx[x] = y
    left, right = list(a.stmts), list(b.stmts)
    if len(left) != len(right):
        return f"{len(left)} != {len(right)} statements"
    for x, y in zip(left, right):
        if (err := _stmts_equal(x, y, ctx)) is not None:
            return err
    return None


def assert_methods_match(got: ir.Method, expected: ir.Method) -> None:
    err = blocks_equal(
        got.callable_region.blocks[0], expected.callable_region.blocks[0]
    )
    if err is not None:
        raise AssertionError(
            f"{err}\n--- got ---\n{got.print_str()}\n--- expected ---\n"
            f"{expected.print_str()}"
        )


def _is_state(value: ir.SSAValue) -> bool:
    return value.type.is_subseteq(StateType)


def _native(stmt: ir.Statement) -> ir.Statement | None:
    if isinstance(stmt, qmove.CZ):
        return gate.CZ(stmt.controls, stmt.targets)
    if isinstance(stmt, qmove.R):
        return gate.R(stmt.axis_angle, stmt.rotation_angle, stmt.qubits)
    if isinstance(stmt, qmove.Rz):
        return gate.Rz(stmt.rotation_angle, stmt.qubits)
    if isinstance(stmt, qmove.MoveTo):
        return arrange.MoveTo(
            stmt.qubits, stmt.locations, multi_move_warning=stmt.multi_move_warning
        )
    if isinstance(stmt, qmove.Permute):
        return arrange.Permute(stmt.qubits, stmt.perm, insert_moves=stmt.insert_moves)
    if isinstance(stmt, qmove.Measure):
        return qubit.stmts.Measure(stmt.qubits)
    if isinstance(stmt, qmove.Invoke):
        return func.Invoke(tuple(stmt.inputs), callee=stmt.callee)
    return None


def erase_qmove(block: ir.Block) -> None:
    """Undo the qmove lowering in place: drop all state plumbing."""
    for stmt in list(block.stmts):
        if (
            isinstance(stmt, (scf.IfElse, scf.For))
            and stmt.results
            and _is_state(stmt.results[0])
        ):
            _erase_scf(stmt)
            continue
        native = _native(stmt)
        if native is None:
            continue
        native.insert_before(stmt)
        for old, new in zip(stmt.results[1:], native.results):
            new.name = old.name
            new.type = old.type
            old.replace_by(new)
        stmt.results[0].replace_by(stmt.args[0])
        stmt.delete()
    for stmt in list(block.stmts):
        if isinstance(stmt, (move.Store, qmove.Exit)):
            stmt.delete()
    for stmt in list(block.stmts):
        if isinstance(stmt, (move.Load, qmove.Enter)):
            stmt.delete()


def _erase_scf(stmt: scf.IfElse | scf.For) -> None:
    yielded: list[ir.SSAValue] = []
    for region in stmt.regions:
        body = region.blocks[0]
        erase_qmove(body)
        old_yield = body.last_stmt
        assert isinstance(old_yield, scf.Yield)
        # With the arm erased, this is the state the arm started from.
        yielded.append(old_yield.values[0])
        old_yield.replace_by(scf.Yield(*old_yield.values[1:]))
    if isinstance(stmt, scf.For):
        body = stmt.body.blocks[0]
        body.args.delete(body.args[1])
        state_in = stmt.initializers[0]
        region = stmt.body
        region.detach()
        new: ir.Statement = scf.For(stmt.iterable, region, *stmt.initializers[1:])
    else:
        state_in = yielded[0]  # the state both arms captured
        then_body, else_body = stmt.then_body, stmt.else_body
        then_body.detach()
        else_body.detach()
        new = scf.IfElse(stmt.cond, then_body, else_body)
    new.insert_before(stmt)
    stmt.results[0].replace_by(state_in)
    for old, replacement in zip(stmt.results[1:], new.results, strict=True):
        replacement.name = old.name
        replacement.type = old.type
        old.replace_by(replacement)
    stmt.delete()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest python/tests/test_qmove_helpers.py -v`
Expected: 3 PASS. `test_difference_inside_a_nested_region_is_caught` asserts kirin's comparison says *equal*. If a kirin upgrade fixes that, change that line to `assert not a.is_structurally_equal(b)`; the comparator stays.

- [ ] **Step 5: Lint and commit**

```bash
git add python/tests/_qmove_helpers.py python/tests/test_qmove_helpers.py
git commit -m "test(python): add a deep IR comparator and qmove eraser"
```

---

### Task 3: `ConstCallToInvoke`

`SquinToNative`'s `GateRule` turns each squin gate into `func.call` of a `py.constant` native kernel. `rewrite.Inline` only inlines `func.call` of a lambda. kirin's `Call2Invoke` needs const-prop hints, which are not set inside `scf.for` bodies. So without this rule, gate calls inside loops are never inlined.

**Files:**
- Create: `python/bloqade/lanes/rewrite/const_call_to_invoke.py`
- Test: `python/tests/rewrite/test_const_call_to_invoke.py`

**Interfaces:**
- Produces: `ConstCallToInvoke()`, a `RewriteRule` with no fields.

- [ ] **Step 1: Write the failing test**

Create `python/tests/rewrite/test_const_call_to_invoke.py`:

```python
from kirin import ir, rewrite, types
from kirin.dialects import func, py

from bloqade import squin
from bloqade.lanes.rewrite.const_call_to_invoke import ConstCallToInvoke


@squin.kernel
def _callee(x: int) -> int:
    return x + 1


def test_call_of_a_constant_method_becomes_invoke():
    arg = ir.TestValue(types.Int)
    const = py.Constant(_callee)
    call = func.Call(const.result, (arg,), kwargs=())
    block = ir.Block([const, call])

    assert rewrite.Walk(ConstCallToInvoke()).rewrite(block).has_done_something
    (invoke,) = [s for s in block.stmts if isinstance(s, func.Invoke)]
    assert invoke.callee is _callee
    assert tuple(invoke.inputs) == (arg,)


def test_call_of_a_non_constant_callee_is_left_alone():
    callee = ir.TestValue(types.Any)
    call = func.Call(callee, (ir.TestValue(types.Int),), kwargs=())
    block = ir.Block([call])
    assert not rewrite.Walk(ConstCallToInvoke()).rewrite(block).has_done_something


def test_rewrites_inside_loop_bodies():
    from kirin.dialects import scf

    const = py.Constant(_callee)
    body = ir.Block()
    i = body.args.append_from(types.Int, "i")
    call = func.Call(const.result, (i,), kwargs=())
    body.stmts.append(call)
    body.stmts.append(scf.Yield())
    loop = scf.For(ir.TestValue(types.Any), ir.Region(body))
    block = ir.Block([const, loop])

    assert rewrite.Walk(ConstCallToInvoke()).rewrite(block).has_done_something
    assert any(isinstance(s, func.Invoke) for s in body.stmts)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest python/tests/rewrite/test_const_call_to_invoke.py -v`
Expected: `ModuleNotFoundError: No module named 'bloqade.lanes.rewrite.const_call_to_invoke'`.

- [ ] **Step 3: Write the implementation**

Create `python/bloqade/lanes/rewrite/const_call_to_invoke.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

from kirin import ir
from kirin.dialects import func, py
from kirin.rewrite.abc import RewriteResult, RewriteRule


@dataclass
class ConstCallToInvoke(RewriteRule):
    """Rewrite ``func.call`` of a ``py.constant`` method into ``func.invoke``.

    ``kirin.rewrite.Inline`` only inlines ``func.call`` of a lambda, and
    kirin's ``Call2Invoke`` reads const-prop hints, which are not set inside
    ``scf.for`` bodies. ``SquinToNative``'s ``GateRule`` emits exactly this
    pattern, so without this rule gate calls inside loops are never inlined.
    """

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        if not isinstance(node, func.Call) or node.kwargs:
            return RewriteResult()
        callee = node.callee
        if not isinstance(callee, ir.ResultValue):
            return RewriteResult()
        const = callee.stmt
        if not isinstance(const, py.Constant) or not isinstance(const.value, ir.PyAttr):
            return RewriteResult()
        method = const.value.data
        if not isinstance(method, ir.Method):
            return RewriteResult()

        invoke = func.Invoke(tuple(node.inputs), callee=method, purity=node.purity)
        invoke.result.name = node.result.name
        invoke.result.type = node.result.type
        node.replace_by(invoke)
        return RewriteResult(has_done_something=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest python/tests/rewrite/test_const_call_to_invoke.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Lint and commit**

```bash
git add python/bloqade/lanes/rewrite/const_call_to_invoke.py python/tests/rewrite/test_const_call_to_invoke.py
git commit -m "feat(python): rewrite constant-callee calls to invokes"
```

---

### Task 4: `RefineQubitTypes`

After inlining without unrolling, qubit types are imprecise: `qalloc(3)` is typed `IList[Qubit, Any]`. Re-running `TypeInfer` cannot narrow them. The module docstring below explains why. `AddressAnalysis` knows the real registers, so this pass intersects ("meets") each qubit-valued type with the type its address implies.

**Files:**
- Create: `python/bloqade/lanes/rewrite/refine_qubit_types.py`
- Test: `python/tests/rewrite/test_refine_qubit_types.py`

**Interfaces:**
- Consumes: `bloqade.analysis.address` (bloqade-circuit).
- Produces:
  - `address_type(addr) -> types.TypeAttribute | None`.
  - `RefineQubitTypes(dialects, no_raise=...)`, a `kirin.passes.Pass`. Call it as `RefineQubitTypes(mt.dialects, no_raise=False)(mt)`. It returns a `RewriteResult`, and raises `ValidationErrorGroup` on a contradiction unless `no_raise`.

- [ ] **Step 1: Write the failing test**

Create `python/tests/rewrite/test_refine_qubit_types.py`:

```python
from typing import Literal

import pytest
from bloqade.types import Qubit, QubitType
from kirin import passes, types
from kirin.dialects import ilist, scf
from kirin.ir.exception import ValidationErrorGroup
from kirin.passes.inline import InlinePass

from bloqade import qubit, squin
from bloqade.lanes.rewrite.refine_qubit_types import RefineQubitTypes


def _inline_and_infer(kernel):
    out = kernel.similar()
    InlinePass(out.dialects).fixpoint(out)
    passes.TypeInfer(out.dialects, no_raise=False)(out)
    return out


def _map_result(mt):
    (stmt,) = [s for s in mt.callable_region.walk() if isinstance(s, ilist.Map)]
    return stmt.result


def test_qalloc_register_gets_its_length():
    @squin.kernel
    def k():
        qs = squin.qalloc(3)
        squin.h(qs[0])

    out = _inline_and_infer(k)
    assert _map_result(out).type == ilist.IListType[QubitType, types.Any]
    assert RefineQubitTypes(out.dialects, no_raise=False)(out).has_done_something
    assert _map_result(out).type == ilist.IListType[QubitType, types.Literal(3)]


def test_loop_variable_keeps_qubit():
    @squin.kernel
    def k():
        qs = squin.qalloc(2)
        for q in qs:
            squin.h(q)

    out = _inline_and_infer(k)
    RefineQubitTypes(out.dialects, no_raise=False)(out)
    (loop,) = [s for s in out.callable_region.walk() if isinstance(s, scf.For)]
    assert loop.body.blocks[0].args[0].type == QubitType


def test_subroutine_parameter_keeps_its_annotation():
    @squin.kernel
    def sub(qs: ilist.IList[Qubit, Literal[2]]):
        squin.h(qs[0])

    out = _inline_and_infer(sub)
    RefineQubitTypes(out.dialects, no_raise=False)(out)
    assert out.arg_types[0] == ilist.IListType[QubitType, types.Literal(2)]


def test_qubit_typed_value_holding_a_register_is_a_contradiction():
    @squin.kernel
    def k():
        qs = squin.qalloc(3)
        squin.h(qs[0])

    out = _inline_and_infer(k)
    _map_result(out).type = QubitType  # disagrees with its AddressReg
    with pytest.raises(ValidationErrorGroup):
        RefineQubitTypes(out.dialects, no_raise=False)(out)


def test_bottom_typed_values_are_left_alone():
    @squin.kernel
    def k():
        qs = squin.qalloc(3)
        return squin.measure(qs)  # type: ignore[arg-type]  # takes one Qubit

    out = _inline_and_infer(k)
    (measure,) = [
        s for s in out.callable_region.walk() if isinstance(s, qubit.stmts.Measure)
    ]
    assert measure.result.type.is_subseteq(types.Bottom)
    RefineQubitTypes(out.dialects, no_raise=False)(out)
    assert measure.result.type.is_subseteq(types.Bottom)


def test_running_twice_changes_nothing():
    @squin.kernel
    def k():
        qs = squin.qalloc(3)
        squin.h(qs[0])

    out = _inline_and_infer(k)
    RefineQubitTypes(out.dialects, no_raise=False)(out)
    assert not RefineQubitTypes(out.dialects, no_raise=False)(out).has_done_something
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest python/tests/rewrite/test_refine_qubit_types.py -v`
Expected: `ModuleNotFoundError: No module named 'bloqade.lanes.rewrite.refine_qubit_types'`.

- [ ] **Step 3: Write the implementation**

Create `python/bloqade/lanes/rewrite/refine_qubit_types.py`:

```python
"""Narrow qubit-valued SSA types using qubit address analysis.

Type inference cannot recover qubit types once calls are inlined without
unrolling:

* ``func.invoke`` is typed from the callee's signature. Specializing it to the
  call's arguments would mutate a signature every call site shares, so
  ``qalloc(3)`` is typed ``IList[Qubit, Any]``.
* ``py.constant`` is typed ``PyClass(type(value))`` (``!py.int``), not a
  ``Literal``; values live in the const-prop lattice instead.
* Re-inference cannot narrow an inlined type: ``TypeInference.eval_fallback``
  substitutes solved type variables into the result's *current* SSA type, and
  an inlined statement arrives already concrete (``IList[Qubit, Any]``).

``AddressAnalysis`` re-interprets each callee with its real arguments, so it does
know ``qalloc(3)`` is a three-qubit register. This pass meets each qubit-valued
SSA type with the type its address implies. It only ever narrows; a contradiction
(a value typed ``Qubit`` that holds a register) is an error. Values type
inference already typed ``Bottom`` are left alone.

``TypeInfer`` overwrites every type it infers, so run this after each
``TypeInfer``, once per method: ``AddressAnalysis`` discards callee frames.
"""

from __future__ import annotations

from dataclasses import dataclass

from bloqade.analysis import address
from bloqade.types import QubitType
from kirin import ir, types
from kirin.dialects import ilist
from kirin.ir.exception import ValidationErrorGroup
from kirin.passes import Pass
from kirin.rewrite.abc import RewriteResult


def address_type(addr: address.Address) -> types.TypeAttribute | None:
    """The type an address implies, or ``None`` if it implies nothing."""
    if isinstance(addr, (address.AddressQubit, address.UnknownQubit)):
        return QubitType
    if isinstance(addr, address.AddressReg):
        return ilist.IListType[QubitType, types.Literal(len(addr.data))]
    if isinstance(addr, address.UnknownReg):
        return ilist.IListType[QubitType, types.Any]
    if isinstance(addr, address.PartialIList) and addr.data:
        elems = [address_type(elem) for elem in addr.data]
        if any(elem is None for elem in elems):
            return None
        joined = elems[0]
        assert joined is not None
        for elem in elems[1:]:
            assert elem is not None
            joined = joined.join(elem)
        return ilist.IListType[joined, types.Literal(len(elems))]
    return None


@dataclass
class RefineQubitTypes(Pass):
    def unsafe_run(self, mt: ir.Method) -> RewriteResult:
        analysis = address.AddressAnalysis(self.dialects)
        if self.no_raise:
            frame, _ = analysis.run_no_raise(mt)
        else:
            frame, _ = analysis.run(mt)

        errors: list[ir.ValidationError] = []
        changed = False
        for value, addr in frame.entries.items():
            derived = address_type(addr)
            if derived is None:
                continue
            current = value.type
            if current.is_subseteq(types.Bottom):
                # Type inference already rejected this value. Its address is
                # derived from that Bottom type (Bottom is a subtype of every
                # IList[Qubit]), so it says nothing about what the value holds.
                continue
            refined = current.meet(derived)
            if refined.is_subseteq(types.Bottom):
                node = value.owner if isinstance(value, ir.ResultValue) else mt.code
                errors.append(
                    ir.ValidationError(
                        node,
                        f"value typed {current} holds {addr}, which needs {derived}",
                    )
                )
                continue
            if refined != current:
                value.type = refined
                changed = True

        if errors and not self.no_raise:
            for error in errors:
                error.attach(mt)
            raise ValidationErrorGroup(
                "RefineQubitTypes: qubit types contradict their addresses",
                errors=errors,
            )
        return RewriteResult(has_done_something=changed)
```

Values that type inference already typed `Bottom` are skipped on purpose. `AddressAnalysis` derives `UnknownReg` from a `Bottom` type, so taking it would re-type a rejected measurement result as a qubit register. `test_bottom_typed_values_are_left_alone` pins this.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest python/tests/rewrite/test_refine_qubit_types.py -v`
Expected: 6 PASS.

- [ ] **Step 5: Lint and commit**

```bash
git add python/bloqade/lanes/rewrite/refine_qubit_types.py python/tests/rewrite/test_refine_qubit_types.py
git commit -m "feat(python): refine qubit types from address analysis"
```

---

### Task 5: The native frontend

This step lowers a kernel and its subroutines to native IR, inlining everything else and unrolling nothing. It processes each method as its own `similar()` clone and calls `GateRule` and `DecomposeCliffordToNative` directly. It must **not** use `CallGraphPass` or `SquinToNative.emit`: those clone every callee, after which subroutine calls no longer reference the user's `Method`s, and the subroutines get silently inlined.

**Files:**
- Create: `python/bloqade/lanes/transform/qmove_frontend.py`
- Test: `python/tests/test_qmove_frontend.py`

**Interfaces:**
- Consumes: Task 3 `ConstCallToInvoke`, Task 4 `RefineQubitTypes`, Task 1 `qmove.dialect`; `bloqade.gemini.common.validation.recursion.CallGraph`/`format_cycle`; `bloqade.native.upstream.squin2native.GateRule`; `bloqade.rewrite.passes.callgraph.ReplaceMethods`.
- Produces:
  - `NativeProgram(entry: ir.Method, subroutines: dict[ir.Method, ir.Method])`, mapping original subroutine to lowered clone.
  - `unlisted_recursion(entry, subroutines: frozenset[ir.Method]) -> list[str]`.
  - `lower_to_native(entry, subroutines: Iterable[ir.Method], arch_spec: ArchSpec, *, no_raise=False) -> NativeProgram`.

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_qmove_frontend.py`:

```python
from typing import Literal

from bloqade.native.dialects.gate import stmts as gate
from bloqade.types import Qubit, QubitType
from kirin import types
from kirin.dialects import func, ilist, scf

from bloqade import squin
from bloqade.lanes.arch.gemini.physical import get_arch_spec
from bloqade.lanes.transform.qmove_frontend import lower_to_native, unlisted_recursion


@squin.kernel
def sub(qs: ilist.IList[Qubit, Literal[2]]):
    squin.cx(qs[0], qs[1])


@squin.kernel
def main():
    qs = squin.qalloc(3)
    for q in qs:
        squin.h(q)
    sub(ilist.IList([qs[0], qs[1]]))
    return squin.broadcast.measure(qs)


def _calls(mt):
    return [
        s for s in mt.callable_region.walk() if isinstance(s, (func.Invoke, func.Call))
    ]


def test_gates_inside_loops_are_native():
    program = lower_to_native(main, [sub], get_arch_spec())
    (loop,) = [
        s for s in program.entry.callable_region.walk() if isinstance(s, scf.For)
    ]
    body = list(loop.body.walk())
    assert any(isinstance(s, (gate.R, gate.Rz)) for s in body)
    assert not any(isinstance(s, (func.Invoke, func.Call)) for s in body)


def test_only_subroutine_calls_survive_and_target_the_clone():
    program = lower_to_native(main, [sub], get_arch_spec())
    clone = program.subroutines[sub]
    (call,) = _calls(program.entry)
    assert isinstance(call, func.Invoke) and call.callee is clone
    assert clone is not sub and clone.sym_name == "sub"
    assert any(isinstance(s, gate.CZ) for s in clone.callable_region.walk())


def test_originals_are_untouched():
    before = sum(1 for _ in sub.callable_region.walk())
    lower_to_native(main, [sub], get_arch_spec())
    assert sum(1 for _ in sub.callable_region.walk()) == before
    assert not any(isinstance(s, gate.CZ) for s in sub.callable_region.walk())


def test_unlisted_calls_are_inlined():
    program = lower_to_native(main, [], get_arch_spec())
    assert _calls(program.entry) == []


def test_register_types_are_refined():
    program = lower_to_native(main, [sub], get_arch_spec())
    (alloc,) = [
        s for s in program.entry.callable_region.walk() if isinstance(s, ilist.Map)
    ]
    assert alloc.result.type == ilist.IListType[QubitType, types.Literal(3)]


@squin.kernel
def rec(qs: ilist.IList[Qubit, Literal[1]], n: int):
    if n > 0:
        squin.h(qs[0])
        rec(qs, n - 1)


@squin.kernel
def calls_rec():
    qs = squin.qalloc(1)
    rec(qs, 3)


def test_unlisted_recursion_is_reported():
    (cycle,) = unlisted_recursion(calls_rec, frozenset())
    assert "rec -> rec" in cycle
    assert unlisted_recursion(calls_rec, frozenset({rec})) == []


def test_listed_recursion_calls_its_own_clone():
    program = lower_to_native(calls_rec, [rec], get_arch_spec())
    clone = program.subroutines[rec]
    (self_call,) = _calls(clone)
    assert self_call.callee is clone
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest python/tests/test_qmove_frontend.py -v`
Expected: `ModuleNotFoundError: No module named 'bloqade.lanes.transform.qmove_frontend'`.

- [ ] **Step 3: Write the implementation**

Create `python/bloqade/lanes/transform/qmove_frontend.py`:

```python
"""Lower a kernel and its subroutines to native IR, keeping subroutine calls.

Everything except the listed subroutines is inlined, without unrolling. Each
method is processed as its own ``similar()`` clone rather than through
``CallGraphPass`` / ``SquinToNative.emit``: those clone every callee and retarget
the calls, after which a subroutine call no longer references the user's
``Method`` and the subroutine would be inlined like any other call.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import chain

from bloqade.native._prelude import kernel as native_kernel
from bloqade.native.upstream.squin2native import GateRule
from bloqade.rewrite.passes.callgraph import ReplaceMethods
from kirin import ir, passes, rewrite
from kirin.rewrite import Inline

from bloqade.gemini.common.validation.recursion import CallGraph, format_cycle
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.dialects import move, qmove
from bloqade.lanes.dialects.arch import BindArchSpec
from bloqade.lanes.rewrite import clifford2native
from bloqade.lanes.rewrite.const_call_to_invoke import ConstCallToInvoke
from bloqade.lanes.rewrite.refine_qubit_types import RefineQubitTypes


@dataclass(frozen=True)
class NativeProgram:
    entry: ir.Method
    subroutines: dict[ir.Method, ir.Method]
    """Each original subroutine ``Method`` mapped to its lowered clone."""


def unlisted_recursion(
    entry: ir.Method, subroutines: frozenset[ir.Method]
) -> list[str]:
    """Call-graph cycles through a kernel that is not a subroutine.

    Checked before inlining: inlining a recursive kernel never reaches a fixpoint.
    """
    return [
        format_cycle(cycle)
        for cycle in CallGraph(entry).find_cycles()
        if any(member not in subroutines for member in cycle.members)
    ]


def lower_to_native(
    entry: ir.Method,
    subroutines: Iterable[ir.Method],
    arch_spec: ArchSpec,
    *,
    no_raise: bool = False,
) -> NativeProgram:
    subs = tuple(dict.fromkeys(subroutines))
    sub_codes = tuple(sub.code for sub in subs)
    reachable = CallGraph(entry).edges.keys()
    dialects = (
        entry.dialects.union(chain.from_iterable(m.dialects.data for m in reachable))
        .union(native_kernel)
        .add(qmove.dialect)
        .add(move.dialect)
    )

    def keep_call(code: ir.Statement) -> bool:
        # rewrite.Inline hands its heuristic the callee's func.Function, not the
        # call site.
        return all(code is not sub_code for sub_code in sub_codes)

    inline = rewrite.Fixpoint(
        rewrite.Walk(rewrite.Chain(ConstCallToInvoke(), Inline(keep_call)))
    )

    def lower(method: ir.Method) -> ir.Method:
        out = method.similar(dialects)
        inline.rewrite(out.code)
        rewrite.Walk(BindArchSpec(arch_spec)).rewrite(out.code)
        rewrite.Walk(clifford2native.DecomposeCliffordToNative()).rewrite(out.code)
        # GateRule turns each squin gate into a call of a native stdlib kernel;
        # inline again to expose the native.gate statements.
        rewrite.Walk(GateRule()).rewrite(out.code)
        inline.rewrite(out.code)
        rewrite.Fixpoint(rewrite.Walk(rewrite.DeadCodeElimination())).rewrite(out.code)
        return out

    clones = {sub: lower(sub) for sub in subs}
    new_entry = lower(entry)
    for method in (new_entry, *clones.values()):
        rewrite.Walk(ReplaceMethods(clones)).rewrite(method.code)
        passes.TypeInfer(method.dialects, no_raise=no_raise)(method)
        RefineQubitTypes(method.dialects, no_raise=no_raise)(method)
    return NativeProgram(new_entry, clones)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest python/tests/test_qmove_frontend.py -v`
Expected: 7 PASS.

- [ ] **Step 5: Lint and commit**

```bash
git add python/bloqade/lanes/transform/qmove_frontend.py python/tests/test_qmove_frontend.py
git commit -m "feat(python): lower kernels to native IR keeping subroutine calls"
```

---

### Task 6: Input validation

Rejects input the qmove lowering does not support, and reports every problem at once.

The function-value check follows calls transitively. That matters because a non-capturing nested `def` lowers to a `py.constant` method, not a `func.Lambda`, and its gates sit behind `func.invoke`s.

Kirin's Python lowering already rejects early returns inside an `if`, so the early-return check matters only for hand-built IR. That's why its test builds IR by hand.

**Files:**
- Create: `python/bloqade/lanes/validation/qmove_input.py`
- Test: `python/tests/validation/test_qmove_input.py`

**Interfaces:**
- Consumes: Task 5 `lower_to_native`, used in the tests.
- Produces: `get_input_validation(subroutine: bool) -> type[ValidationPass]`. Its name is `"lanes.qmove.input"`. It also defines the module-level tuples `QUANTUM_DIALECTS`, `LOGICAL_DIALECTS`, `ALLOCATION`, `SUPPORTED` and `HIGHER_ORDER`.

- [ ] **Step 1: Write the failing test**

Create `python/tests/validation/test_qmove_input.py`:

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


def _messages(method: ir.Method, subroutine: bool = False) -> list[str]:
    result = ValidationSuite([get_input_validation(subroutine)]).validate(method)
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


def test_allocation_in_a_subroutine_is_rejected():
    @squin.kernel
    def sub(qs: ilist.IList[Qubit, Literal[1]]):
        extra = squin.qalloc(1)
        squin.cz(qs[0], extra[0])

    @squin.kernel
    def k():
        qs = squin.qalloc(1)
        sub(qs)

    clone = _native(k, [sub]).subroutines[sub]
    assert _messages(clone, subroutine=True) == [
        "qubits may only be allocated in the entry kernel"
    ]


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest python/tests/validation/test_qmove_input.py -v`
Expected: `ModuleNotFoundError: No module named 'bloqade.lanes.validation.qmove_input'`.

- [ ] **Step 3: Write the implementation**

Create `python/bloqade/lanes/validation/qmove_input.py`:

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
from kirin import ir
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


def get_input_validation(subroutine: bool) -> type[ValidationPass]:
    """``ValidationSuite`` builds passes with no arguments, hence the factory."""

    @dataclass
    class QMoveInputValidation(ValidationPass):
        SUBROUTINE: ClassVar[bool] = subroutine

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

                if isinstance(stmt, func.Call):
                    error(stmt, "calls through a function value are not supported")
                if self.SUBROUTINE and isinstance(stmt, ALLOCATION):
                    error(stmt, "qubits may only be allocated in the entry kernel")
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
                    elif self.SUBROUTINE and _allocates(code):
                        error(stmt, "qubits may only be allocated in the entry kernel")

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

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest python/tests/validation/test_qmove_input.py -v`
Expected: 11 PASS.

- [ ] **Step 5: Lint and commit**

```bash
git add python/bloqade/lanes/validation/qmove_input.py python/tests/validation/test_qmove_input.py
git commit -m "feat(python): validate input to the qmove lowering"
```

---

### Task 7: Local lowering to qmove

Each rule is local. Its output is valid but unthreaded: every statement opens and closes its own chain. Task 8 joins the chains.

**Files:**
- Create: `python/bloqade/lanes/rewrite/native2qmove.py`
- Test: `python/tests/rewrite/test_native2qmove.py`

**Interfaces:**
- Consumes: Task 1 statements.
- Produces: `RewriteNativeToQMove(subroutines: frozenset[ir.Method] = frozenset())`, a `RewriteRule` applied with `rewrite.Walk`. `subroutines` holds the lowered subroutine **clones**.

- [ ] **Step 1: Write the failing test**

Create `python/tests/rewrite/test_native2qmove.py`:

```python
from typing import Literal, TypeVar

from bloqade.native.dialects.gate import stmts as gate
from bloqade.types import MeasurementResultType, Qubit, QubitType
from kirin import ir, rewrite, types
from kirin.dialects import func, ilist

from bloqade import qubit, squin
from bloqade.gemini.common.dialects.arrange import stmts as arrange
from bloqade.lanes.dialects import move, qmove
from bloqade.lanes.rewrite.native2qmove import RewriteNativeToQMove

QUBITS = ilist.IListType[QubitType, types.Literal(2)]
T = TypeVar("T", bound=ir.Statement)


def _lower(block: ir.Block, subroutines=frozenset()) -> list[ir.Statement]:
    rule = rewrite.Walk(RewriteNativeToQMove(subroutines))
    assert rule.rewrite(block).has_done_something
    return list(block.stmts)


def _assert_wrapped(stmts: list[ir.Statement], kind: type[T]) -> T:
    load, stateful, store = stmts
    assert isinstance(load, move.Load) and isinstance(store, move.Store)
    assert isinstance(stateful, kind)
    assert stateful.args[0] is load.result
    assert store.current_state is stateful.results[0]
    return stateful


def test_gates_are_wrapped_in_load_and_store():
    qs = ir.TestValue(QUBITS)
    angle = ir.TestValue(types.Float)
    cases: list[tuple[ir.Statement, type[ir.Statement]]] = [
        (gate.CZ(qs, qs), qmove.CZ),
        (gate.R(angle, angle, qs), qmove.R),
        (gate.Rz(angle, qs), qmove.Rz),
    ]
    for native, kind in cases:
        operands = tuple(native.args)
        stateful = _assert_wrapped(_lower(ir.Block([native])), kind)
        assert tuple(stateful.args[1:]) == operands


def test_move_to_and_permute_keep_their_attributes():
    qs = ir.TestValue(QUBITS)
    move_to = arrange.MoveTo(qs, ir.TestValue(types.Any), multi_move_warning=False)
    lowered = _assert_wrapped(_lower(ir.Block([move_to])), qmove.MoveTo)
    assert lowered.multi_move_warning is False

    permute = arrange.Permute(qs, ir.TestValue(types.Any), insert_moves=True)
    relabel = _assert_wrapped(_lower(ir.Block([permute])), qmove.Permute)
    assert relabel.insert_moves is True


def test_measure_result_is_forwarded():
    measure = qubit.stmts.Measure(ir.TestValue(QUBITS))
    measure.result.type = ilist.IListType[MeasurementResultType, types.Literal(2)]
    user = ilist.New(values=(measure.result,))
    lowered = _assert_wrapped(_lower(ir.Block([measure, user]))[:3], qmove.Measure)
    assert user.values[0] is lowered.measurements
    assert lowered.measurements.type == measure.result.type


def test_only_subroutine_invokes_are_lowered():
    @squin.kernel
    def sub(qs: ilist.IList[Qubit, Literal[2]]):
        squin.h(qs[0])

    @squin.kernel
    def helper(qs: ilist.IList[Qubit, Literal[2]]):
        squin.h(qs[0])

    qs = ir.TestValue(QUBITS)
    call_sub = func.Invoke((qs,), callee=sub)
    call_helper = func.Invoke((qs,), callee=helper)
    stmts = _lower(ir.Block([call_sub, call_helper]), frozenset({sub}))
    lowered = _assert_wrapped(stmts[:3], qmove.Invoke)
    assert lowered.callee is sub and tuple(lowered.inputs) == (qs,)
    assert stmts[3] is call_helper
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest python/tests/rewrite/test_native2qmove.py -v`
Expected: `ModuleNotFoundError: No module named 'bloqade.lanes.rewrite.native2qmove'`.

- [ ] **Step 3: Write the implementation**

Create `python/bloqade/lanes/rewrite/native2qmove.py`:

```python
"""Lower native statements to qmove, each wrapped as ``load; qmove.X; store``.

Every rule is local. The output is valid but un-threaded: each statement opens
and closes its own chain against the state cell. ``qmove_state.thread_method``
then joins the chains and threads the state through ``scf``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from bloqade.native.dialects.gate import stmts as gate
from kirin import ir
from kirin.dialects import func
from kirin.rewrite.abc import RewriteResult, RewriteRule

from bloqade import qubit
from bloqade.gemini.common.dialects.arrange import stmts as arrange
from bloqade.lanes.dialects import move, qmove


def _wrap(
    node: ir.Statement, make: Callable[[ir.SSAValue], ir.Statement]
) -> ir.Statement:
    load = move.Load()
    load.insert_before(node)
    stateful = make(load.result)
    stateful.insert_before(node)
    move.Store(stateful.results[0]).insert_before(node)
    return stateful


@dataclass
class RewriteNativeToQMove(RewriteRule):
    subroutines: frozenset[ir.Method] = field(default_factory=frozenset)
    """Lowered subroutine clones; invokes of these become ``qmove.invoke``."""

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        if isinstance(node, gate.CZ):
            _wrap(node, lambda s: qmove.CZ(s, node.controls, node.targets))
        elif isinstance(node, gate.R):
            _wrap(
                node,
                lambda s: qmove.R(s, node.axis_angle, node.rotation_angle, node.qubits),
            )
        elif isinstance(node, gate.Rz):
            _wrap(node, lambda s: qmove.Rz(s, node.rotation_angle, node.qubits))
        elif isinstance(node, arrange.MoveTo):
            _wrap(
                node,
                lambda s: qmove.MoveTo(
                    s,
                    node.qubits,
                    node.locations,
                    multi_move_warning=node.multi_move_warning,
                ),
            )
        elif isinstance(node, arrange.Permute):
            _wrap(
                node,
                lambda s: qmove.Permute(
                    s, node.qubits, node.perm, insert_moves=node.insert_moves
                ),
            )
        elif isinstance(node, qubit.stmts.Measure):
            measure = _wrap(node, lambda s: qmove.Measure(s, node.qubits))
            assert isinstance(measure, qmove.Measure)
            measure.measurements.type = node.result.type
            node.result.replace_by(measure.measurements)
        elif isinstance(node, func.Invoke) and node.callee in self.subroutines:
            invoke = _wrap(
                node, lambda s: qmove.Invoke(s, tuple(node.inputs), callee=node.callee)
            )
            assert isinstance(invoke, qmove.Invoke)
            invoke.value.type = node.result.type
            node.result.replace_by(invoke.value)
        else:
            return RewriteResult()
        node.delete()
        return RewriteResult(has_done_something=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest python/tests/rewrite/test_native2qmove.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Lint and commit**

```bash
git add python/bloqade/lanes/rewrite/native2qmove.py python/tests/rewrite/test_native2qmove.py
git commit -m "feat(python): lower native statements to qmove"
```

---

### Task 8: State threading

**Files:**
- Create: `python/bloqade/lanes/rewrite/qmove_state.py`
- Test: `python/tests/rewrite/test_qmove_state.py`

**Interfaces:**
- Consumes: Task 7 `RewriteNativeToQMove` and Task 5 `lower_to_native` (in tests); Task 2 helpers.
- Produces:
  - `touches_state(stmt) -> bool`.
  - `thread_block(block, state, *, skip=None) -> ir.SSAValue`, which returns the state at the end of the block.
  - `thread_method(mt, *, subroutine: bool, frame: Frame | None = None) -> None`.

Key facts the tests pin:
- kirin's Python lowering always gives `scf.if` an `else` block; the `scf.IfElse(cond, then)` constructor does not. So the "synthesize an `else`" branch matters only for hand-built IR.
- kirin carries a register read in a loop body as `iter_args` (`for i in range(2): squin.z(qs[i + 1])` carries `qs`). The state is inserted *ahead of* those values.

- [ ] **Step 1: Write the failing test**

Create `python/tests/rewrite/test_qmove_state.py`:

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
    thread_method(program.entry, subroutine=False)
    stmts = top_level(program.entry)
    (load,) = [s for s in stmts if isinstance(s, move.Load)]
    (store,) = [s for s in stmts if isinstance(s, move.Store)]
    assert stmts[0] is load
    assert stmts[-2] is store and isinstance(stmts[-1], func.Return)
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
    thread_method(program.entry, subroutine=False)
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
    thread_method(program.entry, subroutine=False)
    loop = first_of(program.entry, scf.For)
    assert len(loop.initializers) == carried_before + 1
    assert loop.body.blocks[0].args[1].type.is_subseteq(StateType)
    assert _yield(loop.body).values[0].type.is_subseteq(StateType)
    assert first_of(program.entry, move.Store).current_state is loop.results[0]


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
    thread_method(clone, subroutine=True, frame=frame)
    stmts = top_level(clone)
    enter = stmts[0]
    assert isinstance(enter, qmove.Enter) and enter.frame == frame
    assert isinstance(stmts[-2], qmove.Exit)
    assert not statements_of(clone, move.Load) and not statements_of(clone, move.Store)


def test_method_without_quantum_operations_is_untouched():
    @squin.kernel
    def classical(x: int) -> int:
        return x + 1

    out = classical.similar()
    before = len(top_level(out))
    thread_method(out, subroutine=False)
    assert len(top_level(out)) == before
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest python/tests/rewrite/test_qmove_state.py -v`
Expected: `ModuleNotFoundError: No module named 'bloqade.lanes.rewrite.qmove_state'`.

- [ ] **Step 3: Write the implementation**

Create `python/bloqade/lanes/rewrite/qmove_state.py`:

```python
"""Thread the machine state through a method's ``load; qmove.X; store`` chains.

After ``RewriteNativeToQMove`` every stateful statement sits in its own
``load``/``store`` pair. ``thread_method`` joins them into one chain per method
and threads the state explicitly through ``scf.IfElse`` (both arms capture it
and yield it back) and ``scf.For`` (a loop-carried value, ahead of any existing
``iter_args``).

This is a direct recursive traversal, not ``kirin.rewrite.Walk``: ``Walk`` visits
a region's blocks in reverse and a statement's regions before the statement
(``python/tests/rewrite/test_walk_order.py``), and threading needs execution
order. ``rewrite.state.RewriteLoadStore`` cannot be reused either: it finds
stateful statements by their ``ConsumesState``/``EmitsState`` traits, and a
threaded ``scf.IfElse`` has a ``State`` result but no trait (kirin owns ``scf``),
so it would skip the branch and drop its effect.
"""

from __future__ import annotations

from kirin import ir
from kirin.dialects import scf

from bloqade.lanes.dialects import move, qmove
from bloqade.lanes.dialects.qmove import Frame
from bloqade.lanes.types import StateType


def _is_state(value: ir.SSAValue) -> bool:
    return value.type.is_subseteq(StateType)


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


def thread_method(
    mt: ir.Method, *, subroutine: bool, frame: Frame | None = None
) -> None:
    """Open the chain with ``load`` (or ``qmove.enter``), thread it, and close it."""
    block = mt.callable_region.blocks[0]
    if not any(
        isinstance(s, (move.Load, move.Store)) or touches_state(s) for s in block.stmts
    ):
        return
    opener: ir.Statement = qmove.Enter(frame=frame) if subroutine else move.Load()
    first = block.first_stmt
    assert first is not None
    opener.insert_before(first)
    final = thread_block(block, opener.results[0], skip=opener)
    closer: ir.Statement = qmove.Exit(final) if subroutine else move.Store(final)
    terminator = block.last_stmt
    assert terminator is not None
    closer.insert_before(terminator)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest python/tests/rewrite/test_qmove_state.py -v`
Expected: 6 PASS.

- [ ] **Step 5: Lint and commit**

```bash
git add python/bloqade/lanes/rewrite/qmove_state.py python/tests/rewrite/test_qmove_state.py
git commit -m "feat(python): thread the machine state through qmove chains"
```

---

### Task 9: Spectator policies

**Files:**
- Create: `python/bloqade/lanes/validation/spectator.py`
- Test: `python/tests/validation/test_spectator.py`

**Interfaces:**
- Consumes: Task 1 `Frame`; `ArchSpec.get_cz_partner(loc) -> LocationAddress | None`; `bloqade.lanes.arch.ArchBlueprint` / `build_arch` (tests, for the synthetic two-zone arch).
- Produces:
  - `SpectatorPolicy` (abstract): `check_frame(frame, arch) -> list[str]`, and `check_call(frame, atoms, arch) -> list[str]`, which raises `NotImplementedError`.
  - `ZonedPolicy()` and `SingleZonePolicy()`.
  - `format_location(loc) -> str`, which renders `(zone Z, word W, site S)`.

- [ ] **Step 1: Write the failing test**

Create `python/tests/validation/test_spectator.py`:

```python
import pytest

from bloqade.lanes.arch import (
    ArchBlueprint,
    DeviceLayout,
    HypercubeSiteTopology,
    HypercubeWordTopology,
    MatchingTopology,
    ZoneSpec,
    build_arch,
)
from bloqade.lanes.arch.gemini.physical import get_arch_spec
from bloqade.lanes.bytecode.encoding import LocationAddress, ZoneAddress
from bloqade.lanes.dialects.qmove import Effects, Frame, FrameShape
from bloqade.lanes.validation.spectator import SingleZonePolicy, ZonedPolicy

ARCH = get_arch_spec()  # one zone; words (0, 1) are an entangling pair
Z0 = ZoneAddress(0)
A = LocationAddress(0, 0)
A_PARTNER = LocationAddress(1, 0)


def _frame(*binding: LocationAddress, **effects) -> Frame:
    return Frame(FrameShape(((0, len(binding)),)), binding, Effects(**effects))


def test_partner_is_where_the_tests_assume():
    assert ARCH.get_cz_partner(A) == A_PARTNER


def test_zoned_policy_rejects_only_global_pulses():
    policy = ZonedPolicy()
    assert policy.check_frame(_frame(A, cz_zones=frozenset({Z0})), ARCH) == []
    assert policy.check_frame(_frame(A, global_pulses=True), ARCH) == [
        "ZonedPolicy: subroutines may not use global pulses"
    ]


def test_single_zone_policy_accepts_a_pair_closed_footprint():
    frame = _frame(A, A_PARTNER, cz_zones=frozenset({Z0}))
    assert SingleZonePolicy().check_frame(frame, ARCH) == []


def test_single_zone_policy_requires_pair_closure():
    (problem,) = SingleZonePolicy().check_frame(
        _frame(A, cz_zones=frozenset({Z0})), ARCH
    )
    assert "CZ partner (zone 0, word 1, site 0)" in problem


def test_pair_closure_only_matters_in_cz_zones():
    assert SingleZonePolicy().check_frame(_frame(A), ARCH) == []


def test_single_zone_policy_rejects_measurement_and_global_pulses():
    frame = _frame(A, A_PARTNER, measure_zones=frozenset({Z0}), global_pulses=True)
    assert SingleZonePolicy().check_frame(frame, ARCH) == [
        "SingleZonePolicy: subroutines may not use global pulses",
        "SingleZonePolicy: subroutines may not measure",
    ]


def test_check_call_is_declared_only():
    with pytest.raises(NotImplementedError):
        ZonedPolicy().check_call(_frame(A), None, ARCH)  # type: ignore[arg-type]


def _two_zone_arch():
    """A synthetic arch: an entangling "proc" zone and a pair-less "mem" zone."""
    blueprint = ArchBlueprint(
        zones={
            "proc": ZoneSpec(
                num_rows=2,
                num_cols=2,
                entangling=True,
                word_topology=HypercubeWordTopology(),
                site_topology=HypercubeSiteTopology(),
            ),
            "mem": ZoneSpec(num_rows=2, num_cols=2),
        },
        layout=DeviceLayout(sites_per_word=4),
    )
    return build_arch(blueprint, connections={("proc", "mem"): MatchingTopology()}).arch


TWO_ZONE = _two_zone_arch()
PROC, MEM = ZoneAddress(0), ZoneAddress(1)


def test_two_zone_arch_is_shaped_as_assumed():
    assert len(TWO_ZONE.zones) == 2
    assert TWO_ZONE.get_cz_partner(LocationAddress(0, 0, 0)) == LocationAddress(1, 0, 0)
    assert TWO_ZONE.get_cz_partner(LocationAddress(0, 0, 1)) is None


def test_pair_closure_is_per_zone_on_a_synthetic_arch():
    proc_atom = LocationAddress(0, 0, 0)
    proc_partner = LocationAddress(1, 0, 0)
    mem_atom = LocationAddress(0, 0, 1)
    policy = SingleZonePolicy()
    closed = _frame(proc_atom, proc_partner, mem_atom, cz_zones=frozenset({PROC}))
    assert policy.check_frame(closed, TWO_ZONE) == []
    (problem,) = policy.check_frame(
        _frame(proc_atom, mem_atom, cz_zones=frozenset({PROC})), TWO_ZONE
    )
    assert "CZ partner (zone 0, word 1, site 0)" in problem
    # A pair-less zone needs no closure even when it is a CZ zone.
    assert (
        policy.check_frame(_frame(mem_atom, cz_zones=frozenset({MEM})), TWO_ZONE) == []
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest python/tests/validation/test_spectator.py -v`
Expected: `ModuleNotFoundError: No module named 'bloqade.lanes.validation.spectator'`.

- [ ] **Step 3: Write the implementation**

Create `python/bloqade/lanes/validation/spectator.py`:

```python
"""Spectator policies: what a subroutine frame may do to atoms outside it.

A footprint alone does not isolate a callee: ``move.CZ(zone)`` entangles every
complete pair in the zone, measurement reads every atom in its zones, and global
pulses hit every atom. How strict to be depends on the machine, so the rule is a
policy chosen per compilation. The frame records facts (its ``Effects``); the
policy decides whether they are acceptable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from bloqade.lanes.analysis.atom import AtomStateData
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode.encoding import LocationAddress, ZoneAddress
from bloqade.lanes.dialects.qmove import Frame


def format_location(loc: LocationAddress) -> str:
    return f"(zone {loc.zone_id}, word {loc.word_id}, site {loc.site_id})"


class SpectatorPolicy(ABC):
    @abstractmethod
    def check_frame(self, frame: Frame, arch: ArchSpec) -> list[str]:
        """Static: are these effects acceptable for this footprint on this arch?"""

    def check_call(
        self, frame: Frame, atoms: AtomStateData, arch: ArchSpec
    ) -> list[str]:
        """At an invoke, once atom positions exist: are the spectators safe?

        Needs synthesized IR; implemented by the subroutine synthesis work.
        """
        raise NotImplementedError("check_call needs synthesized atom positions")


class ZonedPolicy(SpectatorPolicy):
    """Strict. At a call (``check_call``, not yet implemented), no spectator may
    be anywhere in the frame's ``cz_zones`` or ``measure_zones``."""

    def check_frame(self, frame: Frame, arch: ArchSpec) -> list[str]:
        if frame.effects.global_pulses:
            return ["ZonedPolicy: subroutines may not use global pulses"]
        return []


class SingleZonePolicy(SpectatorPolicy):
    """Permissive. Spectators may share ``cz_zones`` as long as no two of them
    form a complete pair (``check_call``, not yet implemented); the footprint
    must be pair-closed so no callee atom can pair with a spectator. Measuring
    the only zone reads every atom, so subroutines may not measure."""

    def check_frame(self, frame: Frame, arch: ArchSpec) -> list[str]:
        problems = []
        if frame.effects.global_pulses:
            problems.append("SingleZonePolicy: subroutines may not use global pulses")
        if frame.effects.measure_zones:
            problems.append("SingleZonePolicy: subroutines may not measure")
        footprint = frame.footprint
        for loc in frame.binding:
            if ZoneAddress(loc.zone_id) not in frame.effects.cz_zones:
                continue
            partner = arch.get_cz_partner(loc)
            if partner is not None and partner not in footprint:
                problems.append(
                    f"SingleZonePolicy: CZ partner {format_location(partner)} of "
                    f"{format_location(loc)} is outside the frame"
                )
        return problems
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest python/tests/validation/test_spectator.py -v`
Expected: 9 PASS. These include a synthetic two-zone arch: the design must not rely on the bundled Gemini specs.

- [ ] **Step 5: Lint and commit**

```bash
git add python/bloqade/lanes/validation/spectator.py python/tests/validation/test_spectator.py
git commit -m "feat(python): add spectator policies for subroutine frames"
```

---

### Task 10: qmove validation (V1–V3, call rules, F1–F5)

**Files:**
- Create: `python/bloqade/lanes/validation/qmove.py`
- Test: `python/tests/validation/test_qmove.py`

**Interfaces:**
- Consumes: Tasks 1, 5, 7, 8, 9.
- Produces:
  - `get_qmove_validation(arch: ArchSpec, policy: SpectatorPolicy) -> type[ValidationPass]`, named `"lanes.qmove.validation"`.
  - Check functions: `check_use_def`, `check_cell_access`, `check_statements`, `check_calls` (each `(mt) -> list[ir.ValidationError]`), and `check_frame(mt, arch)`.
  - `is_subroutine(mt) -> bool` and `subroutine_frame(mt) -> Frame | None`.

V1 sees one use per `scf.IfElse` arm as one use per execution path. The verifier also has a `move_to` length check (V3), but from Python a mismatched `move_to` is caught earlier, by `RefineQubitTypes`: `TypeInfer` unifies `MoveTo`'s shared `Len`. So V3's test builds that case by hand.

- [ ] **Step 1: Write the failing test**

Create `python/tests/validation/test_qmove.py`:

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
from bloqade.lanes.dialects.qmove import Effects, Frame, FrameShape
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
    kernel: ir.Method, subroutines: dict[ir.Method, Frame | None] | None = None
) -> NativeProgram:
    subroutines = subroutines or {}
    program = lower_to_native(kernel, subroutines, ARCH)
    clones = frozenset(program.subroutines.values())
    rewrite.Walk(RewriteNativeToQMove(clones)).rewrite(program.entry.code)
    thread_method(program.entry, subroutine=False)
    for original, clone in program.subroutines.items():
        rewrite.Walk(RewriteNativeToQMove(clones)).rewrite(clone.code)
        thread_method(clone, subroutine=True, frame=subroutines[original])
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
    first_of(program.entry, move.Store).delete()
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


def test_v2_load_inside_a_region():
    program = _build(main)
    branch = first_of(program.entry, scf.IfElse)
    _insert_load_store_before(branch.then_body.blocks[0].first_stmt)
    messages = _messages(program.entry)
    assert "V2: load must be in the method's top-level block" in messages
    assert "V2: store must be in the method's top-level block" in messages


def test_v2_mixing_load_with_enter():
    program = _build(main, {sub: None})
    clone = program.subroutines[sub]
    _insert_load_store_before(clone.callable_region.blocks[0].first_stmt)
    assert "V2: method uses both load/store and enter/exit" in _messages(clone)


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
    load = move.Load()
    cz = qmove.CZ(load.result, _qubits(1), _qubits(2))
    messages = _messages(_method(load, cz, move.Store(cz.result)))
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
    load = move.Load()
    move_to = qmove.MoveTo(load.result, _qubits(2), locations.result)
    messages = _messages(_method(locations, load, move_to, move.Store(move_to.result)))
    assert messages == ["V3: move_to has 1 locations for 2 qubits"]


# --- call rules -------------------------------------------------------------


def test_func_invoke_of_a_subroutine_is_rejected():
    program = _build(main, {sub: None})
    invoke = first_of(program.entry, qmove.Invoke)
    func.Invoke(tuple(invoke.inputs), callee=invoke.callee).insert_after(invoke)
    messages = _messages(program.entry)
    assert "subroutine sub must be called with qmove.invoke" in messages


def test_qmove_invoke_of_a_non_subroutine_is_rejected():
    load = move.Load()
    call = qmove.Invoke(load.result, (), callee=main)
    messages = _messages(_method(load, call, move.Store(call.result)))
    assert messages == ["invoke target main is not a subroutine"]


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest python/tests/validation/test_qmove.py -v`
Expected: `ModuleNotFoundError: No module named 'bloqade.lanes.validation.qmove'`.

- [ ] **Step 3: Write the implementation**

Create `python/bloqade/lanes/validation/qmove.py`:

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
from bloqade.lanes.dialects.qmove import Frame
from bloqade.lanes.types import StateType
from bloqade.lanes.validation.spectator import SpectatorPolicy, format_location

Path = list[tuple[ir.Statement, int]]


def _is_state(value: ir.SSAValue) -> bool:
    return value.type.is_subseteq(StateType)


def is_subroutine(mt: ir.Method) -> bool:
    return any(isinstance(s, qmove.Enter) for s in mt.callable_region.walk())


def subroutine_frame(mt: ir.Method) -> Frame | None:
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


def check_cell_access(mt: ir.Method) -> list[ir.ValidationError]:
    """V2: load/store or enter/exit, only in the method's top-level block."""
    errors = []
    top = mt.callable_region.blocks[0]
    stmts = list(mt.callable_region.walk())
    cell = [s for s in stmts if isinstance(s, (move.Load, move.Store))]
    frame = [s for s in stmts if isinstance(s, (qmove.Enter, qmove.Exit))]
    if cell and frame:
        errors.append(
            ir.ValidationError(
                mt.code, "V2: method uses both load/store and enter/exit"
            )
        )
    for stmt in cell + frame:
        if stmt.parent_block is not top:
            errors.append(
                ir.ValidationError(
                    stmt, f"V2: {stmt.name} must be in the method's top-level block"
                )
            )
    if frame:
        for kind in (qmove.Enter, qmove.Exit):
            count = sum(isinstance(s, kind) for s in frame)
            if count != 1:
                errors.append(
                    ir.ValidationError(
                        mt.code,
                        f"V2: a subroutine needs exactly one {kind.name}, found {count}",
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
    errors = []
    for stmt in mt.callable_region.walk():
        if isinstance(stmt, (qmove.Invoke, qmove.Prepare)):
            if not is_subroutine(stmt.callee):
                errors.append(
                    ir.ValidationError(
                        stmt,
                        f"{stmt.name} target {stmt.callee.sym_name} is not a subroutine",
                    )
                )
        elif isinstance(stmt, func.Invoke) and is_subroutine(stmt.callee):
            errors.append(
                ir.ValidationError(
                    stmt,
                    f"subroutine {stmt.callee.sym_name} must be called with qmove.invoke",
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
    """F1-F5, for a framed subroutine."""
    frame = subroutine_frame(mt)
    if frame is None:
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
        elif (
            isinstance(stmt, qmove.Permute)
            and not stmt.insert_moves
            and stmt.parent_block is top
            and (perm := _const(stmt.perm)) is not None
        ):
            relabels.setdefault(stmt.qubits, []).append(tuple(int(p) for p in perm))
        elif isinstance(stmt, qmove.Invoke):
            inner = subroutine_frame(stmt.callee)
            if inner is not None and (
                not inner.footprint <= footprint
                or not inner.effects.is_subset_of(effects)
            ):
                error(
                    stmt,
                    f"F5: frame of {stmt.callee.sym_name} is not inside this frame",
                )

    for perms in relabels.values():
        net = list(range(len(perms[0])))
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
                + check_cell_access(method)
                + check_statements(method)
                + check_calls(method)
                + check_frame(method, self.ARCH)
            )
            if (frame := subroutine_frame(method)) is not None:
                errors += [
                    ir.ValidationError(method.code, problem)
                    for problem in self.POLICY.check_frame(frame, self.ARCH)
                ]
            return None, errors

    return QMoveValidation
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest python/tests/validation/test_qmove.py -v`
Expected: 22 PASS.

- [ ] **Step 5: Lint and commit**

```bash
git add python/bloqade/lanes/validation/qmove.py python/tests/validation/test_qmove.py
git commit -m "feat(python): validate qmove state, call and frame rules"
```

---

### Task 11: `NativeToQMove`

This transform ties everything together. The oracle test, `test_lowering_adds_only_state_plumbing`, erases the qmove IR and compares it block by block with the frontend's native IR.

To confirm the oracle can fail, swap `node.axis_angle` and `node.rotation_angle` in `native2qmove.py`'s `R` rule and rerun. It must fail with `r: operand differs` and `in if: r: operand differs`. Revert the swap afterwards.

**Files:**
- Create: `python/bloqade/lanes/transform/native_to_qmove.py`
- Modify: `python/tests/test_import_cycles.py`
- Test: `python/tests/test_transform_native_to_qmove.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `NativeToQMove(arch_spec, subroutines: Mapping[ir.Method, Frame | None] = {}, policy: SpectatorPolicy = ZonedPolicy())` with `.emit(mt, no_raise=False) -> ir.Method`, which returns the entry method.

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_transform_native_to_qmove.py`:

```python
from typing import Literal

import pytest
from bloqade.types import Qubit
from kirin import ir
from kirin.dialects import func, ilist, scf
from kirin.ir.exception import ValidationErrorGroup
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
from bloqade.lanes.dialects.qmove import Effects, Frame, FrameShape
from bloqade.lanes.transform.native_to_qmove import NativeToQMove
from bloqade.lanes.transform.qmove_frontend import lower_to_native
from bloqade.lanes.types import StateType
from bloqade.lanes.validation.spectator import SingleZonePolicy

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
    assert isinstance(top[0], move.Load) and isinstance(top[-2], move.Store)
    assert len(statements_of(out, move.Load)) == 1


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest python/tests/test_transform_native_to_qmove.py -v`
Expected: `ModuleNotFoundError: No module named 'bloqade.lanes.transform.native_to_qmove'`.

- [ ] **Step 3: Write the implementation**

Create `python/bloqade/lanes/transform/native_to_qmove.py`:

```python
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from kirin import ir, passes, rewrite
from kirin.ir.exception import ValidationErrorGroup
from kirin.validation import ValidationSuite

from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.dialects.qmove import Frame
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

    ``subroutines`` maps each kernel to keep as a call to its pinned frame, or to
    ``None`` for a hole that later synthesis fills. Every other call is inlined;
    nothing is unrolled. The result is the entry method; subroutine clones are
    reachable through its ``qmove.invoke`` statements. No placement or move
    synthesis happens here.
    """

    arch_spec: ArchSpec
    subroutines: Mapping[ir.Method, Frame | None] = field(default_factory=dict)
    policy: SpectatorPolicy = field(default_factory=ZonedPolicy)

    def emit(self, mt: ir.Method, no_raise: bool = False) -> ir.Method:
        subroutines = frozenset(self.subroutines)
        if cycles := unlisted_recursion(mt, subroutines):
            raise ValidationErrorGroup(
                "NativeToQMove: recursive kernels must be listed as subroutines",
                errors=[
                    ir.ValidationError(
                        mt.code, f"recursive kernel is not a subroutine: {cycle}"
                    )
                    for cycle in cycles
                ],
            )

        program = lower_to_native(mt, subroutines, self.arch_spec, no_raise=no_raise)
        roles: list[tuple[ir.Method, Frame | None, bool]] = [
            (program.entry, None, False)
        ]
        roles += [
            (clone, self.subroutines[original], True)
            for original, clone in program.subroutines.items()
        ]

        errors: list[ir.ValidationError] = []
        for method, _, is_subroutine in roles:
            result = ValidationSuite([get_input_validation(is_subroutine)]).validate(
                method
            )
            errors += [err for errs in result.errors.values() for err in errs]
        if errors and not no_raise:
            raise ValidationErrorGroup(
                "NativeToQMove: unsupported input", errors=errors
            )

        clones = frozenset(program.subroutines.values())
        for method, frame, is_subroutine in roles:
            rewrite.Walk(RewriteNativeToQMove(clones)).rewrite(method.code)
            thread_method(method, subroutine=is_subroutine, frame=frame)
            rewrite.Fixpoint(rewrite.Walk(rewrite.DeadCodeElimination())).rewrite(
                method.code
            )
            passes.TypeInfer(method.dialects, no_raise=no_raise)(method)
            RefineQubitTypes(method.dialects, no_raise=no_raise)(method)

        validation = get_qmove_validation(self.arch_spec, self.policy)
        for method, _, _ in roles:
            result = ValidationSuite([validation]).validate(method)
            if not no_raise:
                result.raise_if_invalid()
                method.verify()
                method.verify_type()
        return program.entry
```

In `python/tests/test_import_cycles.py`, add `"bloqade.lanes.transform.native_to_qmove",` to the parametrized module list right after `"bloqade.lanes.prelude",`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest python/tests/test_transform_native_to_qmove.py python/tests/test_import_cycles.py -v`
Expected: all PASS (10 in the transform test).

- [ ] **Step 5: Mutation check of the oracle**

In `python/bloqade/lanes/rewrite/native2qmove.py`, temporarily change the `gate.R` rule to `qmove.R(s, node.rotation_angle, node.axis_angle, node.qubits)`.

Run: `uv run --no-sync pytest "python/tests/test_transform_native_to_qmove.py::test_lowering_adds_only_state_plumbing" -v`
Expected: FAIL, with `AssertionError: r: operand differs` and `AssertionError: in if: r: operand differs`. Revert the change and rerun to confirm PASS.

- [ ] **Step 6: Lint and commit**

```bash
git add python/bloqade/lanes/transform/native_to_qmove.py python/tests/test_transform_native_to_qmove.py python/tests/test_import_cycles.py
git commit -m "feat(python): add the NativeToQMove transform"
```

---

### Task 12: Final verification

- [ ] **Step 1: Run every new test together**

Run:

```bash
uv run --no-sync pytest python/tests/dialects/test_qmove.py python/tests/test_qmove_helpers.py python/tests/rewrite/test_const_call_to_invoke.py python/tests/rewrite/test_refine_qubit_types.py python/tests/test_qmove_frontend.py python/tests/validation/test_qmove_input.py python/tests/rewrite/test_native2qmove.py python/tests/rewrite/test_qmove_state.py python/tests/validation/test_spectator.py python/tests/validation/test_qmove.py python/tests/test_transform_native_to_qmove.py python/tests/test_import_cycles.py -q
```

Expected: `95 passed`.

- [ ] **Step 2: Run the full Python suite**

Run: `just test-python`
Expected: the same result as on the base commit. Only new files were added, and `test_import_cycles.py` gained two entries.

- [ ] **Step 3: Run the full lint**

Run: `uv run --no-sync isort python && uv run --no-sync black python && uv run --no-sync ruff check python && uv run --no-sync pyright python`
Expected: clean, and `git diff` shows no reformatting of existing files.

- [ ] **Step 4: Confirm nothing existing changed**

Run: `git diff --name-status $(git merge-base main HEAD)..HEAD -- python/bloqade`
Expected: every line starts with `A` (added). No `M` lines.

Benchmarks are unaffected (no pipeline code changed), so the baseline CSVs need no regeneration.
