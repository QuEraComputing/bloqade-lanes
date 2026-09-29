# Rz Elimination (Virtual Z) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove every `Rz` from compiled Gemini logical programs by commuting each Z rotation along its qubit's wire into the terminal Z-basis readout, so the backend never receives a `move.local_rz` it cannot execute.

**Architecture:** One rewrite rule, `EliminateRz`, shaped like `RewriteNonCliffordToU3`: `rewrite_Statement` dispatches through a `@singledispatchmethod` registry and `Walk` supplies the traversal. Unlike that rule it carries state — a per-qubit phase *frame* (`dict[ir.SSAValue, float]`) spanning the whole program. Each `Rz` is absorbed into the frame and deleted, each `R` has its `axis_angle` shifted by it, `CZ` and `StarRz` pass through untouched (both diagonal), and the residual frame is simply never applied.

**Tech Stack:** Python 3.10+, kirin IR (`kirin.ir`, `kirin.rewrite`), `bloqade.native.dialects.gate`, `bloqade.gemini.logical.dialects.operations`, pytest, numpy (tests only).

**Spec:** `docs/superpowers/specs/2026-09-14-rz-elimination-design.md`

## Global Constraints

- Angles are in **turns**, not radians (`0.25` = 90°). `clifford2native` emits `axis ∈ {0, ¼}`, `rotation ∈ {±¼, ½}`, `Rz ∈ {±¼, ½}`.
- The frame is a **continuous `float`**, never an integer `k ∈ ℤ₄`.
- **`rewrite_Statement`, with state in the rule.** `@dataclass`, state in `field(default_factory=..., init=False)`, per-statement `@singledispatchmethod _rewrite(stmt)`, statements deleted in place. `Walk` supplies the traversal and visits statements in program order — verified, and relied upon. Do not invent a parallel record model, a reader/writer split, or a block-level loop.
- **Preconditions raise, never skip.** A skipped statement leaves an `Rz` behind and breaks the guarantee. (kirin's own `cse` uses `continue` for region-bearing statements; here that would silently lose a phase, so raise instead.)
- **No `require_clifford_angles` flag.** A non-Clifford angle has no native mapping, so it cannot reach this rule; and "logical programs are Clifford-only mid-circuit" is `GeminiLogicalValidation`'s invariant, not this rule's to re-check.
- Imports absolute from `bloqade.lanes`. snake_case files, PascalCase classes. Type annotations enforced by pyright.
- Lint before each commit: `uv run black python && uv run isort python && uv run ruff check python && uv run pyright python`.
- Commit messages follow Conventional Commits.

## Traversal: `Walk` order is relied upon

`rewrite_Statement` carries state across statements, which only works because
`Walk` visits them in program order. That is verified, not assumed: `WorkList` is
a `SimpleQueue` (FIFO — its docstring calling itself a stack is wrong), and
`populate_worklist_Block` enqueues via `first_stmt`/`next_stmt`. A five-statement
block visits 1,2,3,4,5.

Two consequences the rule must handle, both in `rewrite_Region`:

- `populate_worklist_Region` enqueues blocks **reversed** under the default
  `reverse=False`, so with two blocks the statements would be visited in reverse
  block order and the frame would accumulate backwards. Hence the single-block
  precondition. (A phase frame also has no IR representation that could cross a
  block boundary — unlike the state `stack_move2move` and `state` thread through
  block arguments.)
- The frame is reset there so that re-driving the rule is safe. Without it a
  second `Fixpoint` iteration would start from a stale frame and shift every axis
  again.

`Walk(reverse=True)` would still break the rule, and that is undefended. A
`rewrite_Block` implementation would establish its own order and remove that
dependency, at the cost of a hand-rolled statement loop; the dependency is
documented here instead.

## File Structure

| File | Responsibility |
|---|---|
| `python/bloqade/lanes/rewrite/eliminate_rz.py` (create) | The `EliminateRz` rule and its helpers. ~180 lines. |
| `python/bloqade/lanes/transform/native_to_place.py` (modify) | Gains a fifth hook `_post_unroll_rules()`, mirroring `_squin_clifford_rules()`. |
| `python/tests/rewrite/test_eliminate_rz.py` (create) | Rule behaviour on hand-built native IR. |
| `python/tests/rewrite/test_eliminate_rz_algebra.py` (create) | Unitary equivalence against the residual frame. |
| `python/tests/gemini/test_eliminate_rz_pipeline.py` (create) | End-to-end; physical-pipeline-unchanged regression. |

---

### Task 1: The rule — absorb `Rz`, shift `R`

**Files:**
- Create: `python/bloqade/lanes/rewrite/eliminate_rz.py`
- Test: `python/tests/rewrite/test_eliminate_rz.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `EliminateRzError`, and `EliminateRz()` — a `kirin.rewrite.abc.RewriteRule` driven by `rewrite.Walk`. After the walk, `rule._frame: dict[ir.SSAValue, float]` holds the residual (read by tests in Task 4).

- [ ] **Step 1: Write the failing tests**

Create `python/tests/rewrite/test_eliminate_rz.py`:

```python
"""Tests for the EliminateRz rewrite rule, on hand-built native-dialect IR."""

import pytest
from kirin import ir, rewrite, types as kirin_types
from kirin.dialects import ilist, py

from bloqade import qubit as squin_qubit, types as bloqade_types
from bloqade.native.dialects import gate as native_gate

from bloqade.lanes.rewrite.eliminate_rz import EliminateRz, EliminateRzError


def _qubits(block: ir.Block, count: int) -> list[ir.SSAValue]:
    """Append ``count`` qubit allocations to ``block`` and return their values."""
    values = []
    for _ in range(count):
        new = squin_qubit.stmts.New()
        block.stmts.append(new)
        values.append(new.result)
    return values


def _register(block: ir.Block, values) -> ir.SSAValue:
    reg = ilist.New(values=tuple(values), elem_type=bloqade_types.QubitType)
    block.stmts.append(reg)
    return reg.result


def _const(block: ir.Block, value: float) -> ir.SSAValue:
    const = py.Constant(value)
    block.stmts.append(const)
    return const.result


def _of_type(block: ir.Block, kind) -> list[ir.Statement]:
    return [stmt for stmt in block.stmts if isinstance(stmt, kind)]


def _axis(stmt) -> float:
    return stmt.axis_angle.owner.value.unwrap()


def _run(block: ir.Block):
    """Drive the rule the way the pipeline does -- a forward Walk.

    Returns the rule (so tests can read the residual frame) and the result.
    """
    rule = EliminateRz()
    return rule, rewrite.Walk(rule).rewrite(block)


def test_rz_is_deleted():
    block = ir.Block()
    (q0,) = _qubits(block, 1)
    reg = _register(block, [q0])
    angle = _const(block, 0.25)
    block.stmts.append(native_gate.stmts.Rz(rotation_angle=angle, qubits=reg))

    _, result = _run(block)

    assert result.has_done_something
    assert _of_type(block, native_gate.stmts.Rz) == []


def test_r_axis_is_shifted_by_the_pending_frame():
    block = ir.Block()
    (q0,) = _qubits(block, 1)
    reg = _register(block, [q0])
    quarter = _const(block, 0.25)
    zero = _const(block, 0.0)
    block.stmts.append(native_gate.stmts.Rz(rotation_angle=quarter, qubits=reg))
    block.stmts.append(
        native_gate.stmts.R(axis_angle=zero, rotation_angle=quarter, qubits=reg)
    )

    _run(block)

    rs = _of_type(block, native_gate.stmts.R)
    assert len(rs) == 1
    # (0.0 - 0.25) mod 1 == 0.75
    assert _axis(rs[0]) == 0.75


def test_frames_accumulate_across_multiple_rz():
    block = ir.Block()
    (q0,) = _qubits(block, 1)
    reg = _register(block, [q0])
    quarter = _const(block, 0.25)
    half = _const(block, 0.5)
    zero = _const(block, 0.0)
    block.stmts.append(native_gate.stmts.Rz(rotation_angle=quarter, qubits=reg))
    block.stmts.append(native_gate.stmts.Rz(rotation_angle=half, qubits=reg))
    block.stmts.append(
        native_gate.stmts.R(axis_angle=zero, rotation_angle=quarter, qubits=reg)
    )

    _run(block)

    (r,) = _of_type(block, native_gate.stmts.R)
    # (0.0 - 0.75) mod 1 == 0.25
    assert _axis(r) == 0.25


def test_r_on_an_untouched_qubit_is_left_alone():
    block = ir.Block()
    q0, q1 = _qubits(block, 2)
    reg0, reg1 = _register(block, [q0]), _register(block, [q1])
    quarter = _const(block, 0.25)
    zero = _const(block, 0.0)
    block.stmts.append(native_gate.stmts.Rz(rotation_angle=quarter, qubits=reg0))
    original = native_gate.stmts.R(
        axis_angle=zero, rotation_angle=quarter, qubits=reg1
    )
    block.stmts.append(original)

    _run(block)

    assert _of_type(block, native_gate.stmts.R) == [original], (
        "an untouched qubit's R must be left in place, not rebuilt"
    )


def test_cz_passes_through_and_the_frame_survives_it():
    block = ir.Block()
    q0, q1 = _qubits(block, 2)
    controls, targets = _register(block, [q0]), _register(block, [q1])
    quarter = _const(block, 0.25)
    block.stmts.append(native_gate.stmts.Rz(rotation_angle=quarter, qubits=controls))
    cz = native_gate.stmts.CZ(controls=controls, targets=targets)
    block.stmts.append(cz)

    rule, _ = _run(block)

    assert _of_type(block, native_gate.stmts.CZ) == [cz]
    # Diagonal, so it commutes with the frame exactly -- nothing changes.
    assert rule._frame[q0] == 0.25


def test_rule_is_idempotent():
    block = ir.Block()
    (q0,) = _qubits(block, 1)
    reg = _register(block, [q0])
    quarter = _const(block, 0.25)
    zero = _const(block, 0.0)
    block.stmts.append(native_gate.stmts.Rz(rotation_angle=quarter, qubits=reg))
    block.stmts.append(
        native_gate.stmts.R(axis_angle=zero, rotation_angle=quarter, qubits=reg)
    )

    _run(block)
    _, second = _run(block)

    assert not second.has_done_something
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest python/tests/rewrite/test_eliminate_rz.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bloqade.lanes.rewrite.eliminate_rz'`

- [ ] **Step 3: Write the implementation**

Create `python/bloqade/lanes/rewrite/eliminate_rz.py`:

```python
"""EliminateRz: remove ``Rz`` from native-dialect programs by phase commutation.

Scans a flat block in order carrying a per-qubit phase *frame*. Each ``Rz`` is
absorbed into the frame and deleted; each ``R`` has its axis angle shifted by the
frame of the qubits it addresses; ``CZ`` and ``StarRz`` are diagonal and pass
through untouched. Whatever frame remains when the block ends is discarded --
sound because the residual is diagonal and the device's readout is in the Z
basis.

Two exact identities do all the work (angles in turns)::

    R(phi, theta) . Rz(alpha) = Rz(alpha) . R(phi - alpha, theta)
    CZ . Rz(alpha)            = Rz(alpha) . CZ

Shaped like ``RewriteNonCliffordToU3`` -- ``rewrite_Statement`` dispatching
through ``@singledispatchmethod``, with ``Walk`` supplying the traversal -- but
carrying a frame across statements, which relies on ``Walk`` visiting them in
program order. See
``docs/superpowers/specs/2026-09-14-rz-elimination-design.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import singledispatchmethod

from kirin import ir
from kirin.dialects import func, ilist, py
from kirin.rewrite.abc import RewriteResult, RewriteRule

from bloqade import types as bloqade_types
from bloqade.native.dialects import gate as native_gate

__all__ = ["EliminateRz", "EliminateRzError"]

# Angles are grouped and cached at this many decimal places, so equal angles land
# on one dict key and can share a single constant SSA value.
_ANGLE_NDIGITS = 12


class EliminateRzError(Exception):
    """The IR violates a precondition of the Rz elimination scan."""


def _normalize(angle: float) -> float:
    """Fold an angle into [0, 1) turns and round it onto the grouping grid."""
    return round(angle % 1.0, _ANGLE_NDIGITS) % 1.0


@dataclass
class EliminateRz(RewriteRule):
    """Remove every ``Rz`` from a flat native-dialect program."""

    _frame: dict[ir.SSAValue, float] = field(default_factory=dict, init=False)
    """Pending Z phase per qubit, in turns. Spans the whole program."""

    _constants: dict[float, ir.SSAValue] = field(default_factory=dict, init=False)
    """One SSA value per distinct angle, so downstream fusion still matches."""

    _qubits: set[ir.SSAValue] = field(default_factory=set, init=False)

    def rewrite_Region(self, node: ir.Region) -> RewriteResult:
        """Require a single block, and start each walk from a clean frame.

        ``Walk`` enqueues a region before its blocks and their statements, so
        this runs first.

        The single-block requirement is not cosmetic: ``populate_worklist_Region``
        enqueues blocks *reversed* under the default ``reverse=False``, so with
        two blocks the statements would be visited in reverse block order and
        the frame would accumulate backwards. A phase frame also has no IR
        representation that could cross a block boundary -- unlike the state
        ``stack_move2move`` and ``state`` thread through block arguments.

        Resetting here makes re-driving safe: without it, a second ``Fixpoint``
        iteration would start from a stale frame and shift every axis again.
        """
        if len(node.blocks) != 1:
            raise EliminateRzError(
                f"EliminateRz requires a single-block program, found "
                f"{len(node.blocks)} blocks. A phase frame cannot cross a block "
                "boundary. Run AggressiveUnroll first."
            )
        self._frame = {}
        self._constants = {}
        self._qubits = set()
        return RewriteResult()

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        return self._rewrite(node)

    # -- helpers ----------------------------------------------------------

    def _qubit_values(
        self, stmt: ir.Statement, register: ir.SSAValue
    ) -> tuple[ir.SSAValue, ...]:
        owner = register.owner
        if not isinstance(owner, ilist.New):
            raise EliminateRzError(
                f"{stmt.name}: qubit register comes from {type(owner).__name__}, "
                "not ilist.New. EliminateRz requires the post-unroll IR shape."
            )
        values = tuple(owner.values)
        if len(set(values)) != len(values):
            raise EliminateRzError(
                f"{stmt.name} addresses a qubit more than once; no phase frame "
                "is well defined for it. This is malformed IR."
            )
        return values

    def _const_float(self, stmt: ir.Statement, value: ir.SSAValue) -> float:
        owner = value.owner
        if not isinstance(owner, py.Constant):
            raise EliminateRzError(
                f"{stmt.name}: angle is not a compile-time constant "
                f"(owner is {type(owner).__name__})."
            )
        data = owner.value.unwrap()
        if not isinstance(data, (int, float)) or isinstance(data, bool):
            raise EliminateRzError(
                f"{stmt.name}: angle constant is not numeric: {data!r}"
            )
        return float(data)

    def _constant(self, angle: float, before: ir.Statement) -> ir.SSAValue:
        """One ``py.Constant`` per distinct angle, reusing existing ones.

        ``FuseAdjacentGates`` (downstream, at the place layer) matches parameters
        by SSA *identity*, and ``circuit2place`` carries angle values through
        unchanged. Minting a fresh constant per statement would break fusion
        between statements whose angles are numerically equal.
        """
        key = _normalize(angle)
        cached = self._constants.get(key)
        if cached is not None:
            return cached
        const = py.Constant(key)
        const.insert_before(before)
        self._constants[key] = const.result
        return const.result

    # -- per-statement dispatch -------------------------------------------

    @singledispatchmethod
    def _rewrite(self, stmt: ir.Statement) -> RewriteResult:
        """Record qubit allocations, pass over everything else."""
        if len(stmt.results) == 1 and stmt.results[0].type.is_subseteq(
            bloqade_types.QubitType
        ):
            self._qubits.add(stmt.results[0])
        return RewriteResult()

    @_rewrite.register(py.Constant)
    def _(self, stmt: py.Constant) -> RewriteResult:
        data = stmt.value.unwrap()
        if isinstance(data, (int, float)) and not isinstance(data, bool):
            self._constants.setdefault(_normalize(float(data)), stmt.result)
        return RewriteResult()

    @_rewrite.register(ilist.New)
    def _(self, stmt: ilist.New) -> RewriteResult:
        return RewriteResult()

    @_rewrite.register(native_gate.stmts.Rz)
    def _(self, stmt: native_gate.stmts.Rz) -> RewriteResult:
        angle = self._const_float(stmt, stmt.rotation_angle)
        for qubit in self._qubit_values(stmt, stmt.qubits):
            self._frame[qubit] = self._frame.get(qubit, 0.0) + angle
        stmt.delete()
        return RewriteResult(has_done_something=True)

    @_rewrite.register(native_gate.stmts.R)
    def _(self, stmt: native_gate.stmts.R) -> RewriteResult:
        axis = self._const_float(stmt, stmt.axis_angle)
        qubits = self._qubit_values(stmt, stmt.qubits)

        groups: dict[float, list[ir.SSAValue]] = {}
        for qubit in qubits:
            shifted = _normalize(axis - self._frame.get(qubit, 0.0))
            groups.setdefault(shifted, []).append(qubit)

        if len(groups) == 1 and next(iter(groups)) == _normalize(axis):
            return RewriteResult()

        for shifted, group in groups.items():
            register = ilist.New(
                values=tuple(group), elem_type=bloqade_types.QubitType
            )
            register.insert_before(stmt)
            native_gate.stmts.R(
                axis_angle=self._constant(shifted, stmt),
                rotation_angle=stmt.rotation_angle,
                qubits=register.result,
            ).insert_before(stmt)

        stmt.delete()
        return RewriteResult(has_done_something=True)

    @_rewrite.register(native_gate.stmts.CZ)
    def _(self, stmt: native_gate.stmts.CZ) -> RewriteResult:
        # Diagonal: commutes with Rz on each qubit independently, so the two
        # sides' frames need not agree and nothing changes.
        return RewriteResult()

    @_rewrite.register(func.Function)
    def _(self, stmt: func.Function) -> RewriteResult:
        # Walk visits the enclosing definition too, and it carries a region --
        # so it must be registered, or the region guard below would reject the
        # program's own function statement.
        return RewriteResult()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest python/tests/rewrite/test_eliminate_rz.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Lint and commit**

```bash
uv run black python && uv run isort python && uv run ruff check python && uv run pyright python
git add python/bloqade/lanes/rewrite/eliminate_rz.py python/tests/rewrite/test_eliminate_rz.py
git commit -m "feat(rewrite): add EliminateRz, commuting Rz into the terminal readout"
```

---

### Task 2: Splitting across qubits, and constant sharing

**Files:**
- Modify: `python/bloqade/lanes/rewrite/eliminate_rz.py` (only if a test fails — Task 1's `R` handler already groups)
- Test: `python/tests/rewrite/test_eliminate_rz.py`

**Interfaces:**
- Consumes: `EliminateRz` from Task 1.
- Produces: no new names.

- [ ] **Step 1: Write the tests**

Append to `python/tests/rewrite/test_eliminate_rz.py`:

```python
def test_r_splits_when_its_qubits_carry_different_frames():
    """One pulse cannot carry two axis angles, so the statement must split."""
    block = ir.Block()
    q0, q1 = _qubits(block, 2)
    only_q0 = _register(block, [q0])
    both = _register(block, [q0, q1])
    quarter = _const(block, 0.25)
    zero = _const(block, 0.0)
    block.stmts.append(native_gate.stmts.Rz(rotation_angle=quarter, qubits=only_q0))
    block.stmts.append(
        native_gate.stmts.R(axis_angle=zero, rotation_angle=quarter, qubits=both)
    )

    _run(block)

    rs = _of_type(block, native_gate.stmts.R)
    assert len(rs) == 2
    by_axis = {_axis(r): tuple(r.qubits.owner.values) for r in rs}
    assert by_axis == {0.75: (q0,), 0.0: (q1,)}


def test_split_covers_every_original_qubit_exactly_once():
    block = ir.Block()
    q0, q1, q2 = _qubits(block, 3)
    only_q0 = _register(block, [q0])
    only_q2 = _register(block, [q2])
    all_three = _register(block, [q0, q1, q2])
    quarter = _const(block, 0.25)
    half = _const(block, 0.5)
    zero = _const(block, 0.0)
    block.stmts.append(native_gate.stmts.Rz(rotation_angle=quarter, qubits=only_q0))
    block.stmts.append(native_gate.stmts.Rz(rotation_angle=half, qubits=only_q2))
    block.stmts.append(
        native_gate.stmts.R(axis_angle=zero, rotation_angle=quarter, qubits=all_three)
    )

    _run(block)

    covered = [
        value
        for r in _of_type(block, native_gate.stmts.R)
        for value in r.qubits.owner.values
    ]
    assert sorted(map(id, covered)) == sorted(map(id, [q0, q1, q2]))


def test_qubits_with_equal_frames_stay_in_one_statement():
    block = ir.Block()
    q0, q1 = _qubits(block, 2)
    both = _register(block, [q0, q1])
    quarter = _const(block, 0.25)
    zero = _const(block, 0.0)
    block.stmts.append(native_gate.stmts.Rz(rotation_angle=quarter, qubits=both))
    block.stmts.append(
        native_gate.stmts.R(axis_angle=zero, rotation_angle=quarter, qubits=both)
    )

    _run(block)

    rs = _of_type(block, native_gate.stmts.R)
    assert len(rs) == 1
    assert tuple(rs[0].qubits.owner.values) == (q0, q1)


def test_equal_angles_share_one_constant_ssa_value():
    """FuseAdjacentGates matches on SSA identity, so equal angles must share."""
    block = ir.Block()
    q0, q1 = _qubits(block, 2)
    reg0, reg1 = _register(block, [q0]), _register(block, [q1])
    both = _register(block, [q0, q1])
    quarter = _const(block, 0.25)
    zero = _const(block, 0.0)
    block.stmts.append(native_gate.stmts.Rz(rotation_angle=quarter, qubits=both))
    block.stmts.append(
        native_gate.stmts.R(axis_angle=zero, rotation_angle=quarter, qubits=reg0)
    )
    block.stmts.append(
        native_gate.stmts.R(axis_angle=zero, rotation_angle=quarter, qubits=reg1)
    )

    _run(block)

    rs = _of_type(block, native_gate.stmts.R)
    assert len(rs) == 2
    assert rs[0].axis_angle is rs[1].axis_angle


def test_an_existing_constant_is_reused_rather_than_duplicated():
    """The shifted angle already exists in the block, so no new constant."""
    block = ir.Block()
    (q0,) = _qubits(block, 1)
    reg = _register(block, [q0])
    quarter = _const(block, 0.25)
    three_quarter = _const(block, 0.75)
    zero = _const(block, 0.0)
    block.stmts.append(native_gate.stmts.Rz(rotation_angle=quarter, qubits=reg))
    block.stmts.append(
        native_gate.stmts.R(axis_angle=zero, rotation_angle=quarter, qubits=reg)
    )

    _run(block)

    (r,) = _of_type(block, native_gate.stmts.R)
    assert r.axis_angle is three_quarter
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest python/tests/rewrite/test_eliminate_rz.py -v`
Expected: PASS (11 tests). Task 1's `R` handler already groups by shifted angle
and reuses `self._constants`, so these should pass as written. If any fail, fix
the handler — these are behaviours the spec requires, not aspirations.

- [ ] **Step 3: Commit**

```bash
git add python/tests/rewrite/test_eliminate_rz.py
git commit -m "test: cover Rz elimination splitting and constant sharing"
```

---

### Task 3: Preconditions, `Initialize`, and the terminal measurement

**Files:**
- Modify: `python/bloqade/lanes/rewrite/eliminate_rz.py`
- Test: `python/tests/rewrite/test_eliminate_rz.py`

**Interfaces:**
- Consumes: `EliminateRz`.
- Produces: handlers for `operations.StarRz`, `operations.Initialize`, `operations.TerminalLogicalMeasurement`; region and unknown-statement rejection in the default handler; `_touches_qubit(stmt) -> bool`.

- [ ] **Step 1: Write the failing tests**

Append to `python/tests/rewrite/test_eliminate_rz.py`:

```python
from bloqade.gemini.logical.dialects.operations import stmts as operations


def test_statement_carrying_a_region_raises():
    """Un-unrolled control flow must not be silently scanned past.

    This rule runs before scf2cf, so a surviving scf.For is a statement with a
    region inside one block -- not a second block. Its body closes over qubit
    values instead of taking them as arguments, so the qubit-reachability guard
    would miss it and the gates inside would vanish from the scan.
    """
    from kirin.dialects import scf

    block = ir.Block()
    _qubits(block, 1)
    block.stmts.append(
        scf.For(ir.TestValue(type=kirin_types.Any), ir.Region(ir.Block()))
    )

    with pytest.raises(EliminateRzError, match="region"):
        _run(block)


def test_multi_block_region_raises():
    """A phase frame cannot cross a block boundary.

    It has no IR representation to travel in -- unlike the state that
    stack_move2move and state.py thread through block arguments -- so a second
    block would start from zero and silently lose the first block's phases.
    """
    region = ir.Region(ir.Block())
    region.blocks.append(ir.Block())

    with pytest.raises(EliminateRzError, match="single-block"):
        EliminateRz().rewrite_Region(region)


def test_unknown_statement_touching_a_qubit_raises():
    """The scan cannot know whether an unrecognised gate is diagonal."""
    block = ir.Block()
    (q0,) = _qubits(block, 1)
    reg = _register(block, [q0])
    block.stmts.append(squin_qubit.stmts.Measure(qubits=reg))

    with pytest.raises(EliminateRzError, match="phase-neutral"):
        _run(block)


def test_non_ilist_register_raises():
    block = ir.Block()
    angle = _const(block, 0.25)
    opaque = ir.TestValue(
        type=ilist.IListType[bloqade_types.QubitType, kirin_types.Any]
    )
    block.stmts.append(native_gate.stmts.Rz(rotation_angle=angle, qubits=opaque))

    with pytest.raises(EliminateRzError, match="ilist.New"):
        _run(block)


def test_duplicate_qubit_in_one_register_raises():
    block = ir.Block()
    (q0,) = _qubits(block, 1)
    doubled = _register(block, [q0, q0])
    angle = _const(block, 0.25)
    block.stmts.append(native_gate.stmts.Rz(rotation_angle=angle, qubits=doubled))

    with pytest.raises(EliminateRzError, match="more than once"):
        _run(block)


def test_non_constant_angle_raises():
    block = ir.Block()
    (q0,) = _qubits(block, 1)
    reg = _register(block, [q0])
    angle = ir.TestValue(type=kirin_types.Float)
    block.stmts.append(native_gate.stmts.Rz(rotation_angle=angle, qubits=reg))

    with pytest.raises(EliminateRzError, match="constant"):
        _run(block)


def test_initialize_with_a_pending_frame_raises():
    """Initialize sits at the head of a wire; a mid-wire one would drop a phase."""
    block = ir.Block()
    (q0,) = _qubits(block, 1)
    reg = _register(block, [q0])
    quarter = _const(block, 0.25)
    zero = _const(block, 0.0)
    block.stmts.append(native_gate.stmts.Rz(rotation_angle=quarter, qubits=reg))
    block.stmts.append(
        operations.Initialize(theta=zero, phi=zero, lam=zero, qubits=reg)
    )

    with pytest.raises(EliminateRzError, match="Initialize"):
        _run(block)


def test_initialize_at_the_head_of_a_wire_is_fine():
    block = ir.Block()
    (q0,) = _qubits(block, 1)
    reg = _register(block, [q0])
    zero = _const(block, 0.0)
    init = operations.Initialize(theta=zero, phi=zero, lam=zero, qubits=reg)
    block.stmts.append(init)

    _run(block)

    assert _of_type(block, operations.Initialize) == [init]


def test_star_rz_is_untouched_and_the_frame_commutes_past_it():
    block = ir.Block()
    (q0,) = _qubits(block, 1)
    reg = _register(block, [q0])
    quarter = _const(block, 0.25)
    star_angle = _const(block, 0.03)
    block.stmts.append(native_gate.stmts.Rz(rotation_angle=quarter, qubits=reg))
    star = operations.StarRz(rotation_angle=star_angle, qubits=reg)
    block.stmts.append(star)

    rule, _ = _run(block)

    assert _of_type(block, operations.StarRz) == [star]
    assert rule._frame[q0] == 0.25


def test_terminal_measurement_discards_the_frame():
    block = ir.Block()
    (q0,) = _qubits(block, 1)
    reg = _register(block, [q0])
    quarter = _const(block, 0.25)
    block.stmts.append(native_gate.stmts.Rz(rotation_angle=quarter, qubits=reg))
    block.stmts.append(operations.TerminalLogicalMeasurement(qubits=reg))

    rule, _ = _run(block)

    assert rule._frame == {}


def test_frame_left_at_end_of_block_is_simply_discarded():
    """The shape left by RemovePostProcessing: no measurement statement at all."""
    block = ir.Block()
    (q0,) = _qubits(block, 1)
    reg = _register(block, [q0])
    quarter = _const(block, 0.25)
    block.stmts.append(native_gate.stmts.Rz(rotation_angle=quarter, qubits=reg))

    rule, result = _run(block)

    assert result.has_done_something
    assert _of_type(block, native_gate.stmts.Rz) == []
    assert rule._frame == {q0: 0.25}  # residual, discarded by the caller
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest python/tests/rewrite/test_eliminate_rz.py -v`
Expected: FAIL — the region, unknown-statement, `Initialize`, `StarRz`, and
measurement tests fail (no handlers yet).

`squin_qubit.stmts.Measure(qubits=reg)` is used only as a stand-in "unknown
statement that consumes a qubit register" — this rule handles
`operations.TerminalLogicalMeasurement`, not the squin measure, so it is exactly
the unregistered-but-qubit-touching case the guard exists for.

- [ ] **Step 3: Write the implementation**

Add the import:

```python
from bloqade.gemini.logical.dialects.operations import stmts as operations
```

Replace the default `_rewrite` handler with:

```python
    @singledispatchmethod
    def _rewrite(self, stmt: ir.Statement) -> RewriteResult:
        """Record qubit allocations, pass over the inert, reject the unknown."""
        if stmt.regions:
            # This rule runs *before* scf2cf, so control flow that survived
            # unrolling is still an scf.For / scf.IfElse statement holding
            # regions inside one block -- not multiple blocks. Its body closes
            # over qubit values rather than taking them as arguments, so the
            # reachability check below would not see them and the gates inside
            # would be silently skipped. No statement this rule handles carries
            # a region, so rejecting all of them is exact.
            raise EliminateRzError(
                f"{stmt.name} carries a region; EliminateRz cannot see into it, "
                "and gates inside would be silently skipped. Control flow must "
                "be fully unrolled before this rule runs."
            )

        if len(stmt.results) == 1 and stmt.results[0].type.is_subseteq(
            bloqade_types.QubitType
        ):
            self._qubits.add(stmt.results[0])
            return RewriteResult()

        if self._touches_qubit(stmt):
            raise EliminateRzError(
                f"{stmt.name} addresses a qubit but EliminateRz does not know "
                "whether it is phase-neutral. Register a handler for it, or keep "
                "it out of the logical pipeline."
            )
        return RewriteResult()
```

Add the reachability helper beside the other helpers:

```python
    def _touches_qubit(self, stmt: ir.Statement) -> bool:
        for arg in stmt.args:
            if arg in self._qubits:
                return True
            owner = arg.owner
            if isinstance(owner, ilist.New) and any(
                value in self._qubits for value in owner.values
            ):
                return True
        return False
```

And register the three `operations` statements:

```python
    @_rewrite.register(operations.StarRz)
    def _(self, stmt: operations.StarRz) -> RewriteResult:
        # Diagonal, like CZ: the frame commutes past it exactly. Its own
        # rotation is the payload of a user-requested gadget, not ours to remove.
        return RewriteResult()

    @_rewrite.register(operations.Initialize)
    def _(self, stmt: operations.Initialize) -> RewriteResult:
        pending = {
            qubit: self._frame[qubit]
            for qubit in self._qubit_values(stmt, stmt.qubits)
            if self._frame.get(qubit, 0.0) != 0.0
        }
        if pending:
            raise EliminateRzError(
                "Initialize reached with a pending phase frame. Initialize is "
                "expected at the head of a wire; a mid-wire one would need the "
                "frame absorbed into its (theta, phi, lam) instead."
            )
        return RewriteResult()

    @_rewrite.register(operations.TerminalLogicalMeasurement)
    def _(self, stmt: operations.TerminalLogicalMeasurement) -> RewriteResult:
        # The residual is diagonal and the readout is in the Z basis, so it
        # cannot shift any outcome. Drop it.
        for qubit in self._qubit_values(stmt, stmt.qubits):
            self._frame.pop(qubit, None)
        return RewriteResult()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest python/tests/rewrite/test_eliminate_rz.py -v`
Expected: PASS (22 tests)

- [ ] **Step 5: Lint and commit**

```bash
uv run black python && uv run isort python && uv run ruff check python && uv run pyright python
git add python/bloqade/lanes/rewrite/eliminate_rz.py python/tests/rewrite/test_eliminate_rz.py
git commit -m "feat(rewrite): add EliminateRz preconditions and operations handlers"
```

---

### Task 4: Verify unitary equivalence against the residual

Structural tests cannot catch a sign error in the commutation identity.

**Files:**
- Test: `python/tests/rewrite/test_eliminate_rz_algebra.py` (create)

**Interfaces:**
- Consumes: `EliminateRz`, and `rule._frame` after the walk.
- Produces: nothing importable — tests only.

- [ ] **Step 1: Write the test**

Create `python/tests/rewrite/test_eliminate_rz_algebra.py`:

```python
"""The rule must preserve the unitary up to the residual frame it leaves behind.

    U_before == Rz(residual) . U_after      (up to global phase)

A sign error in the commutation identity passes every structural test, so this
builds both circuits as matrices from the IR itself and compares them.
"""

import numpy as np
import pytest
from kirin import ir, rewrite
from kirin.dialects import ilist, py

from bloqade import qubit as squin_qubit, types as bloqade_types
from bloqade.native.dialects import gate as native_gate

from bloqade.lanes.rewrite.eliminate_rz import EliminateRz

_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]])


def _r(axis: float, rotation: float) -> np.ndarray:
    """R(axis, rotation), both angles in turns."""
    phi, theta = 2 * np.pi * axis, 2 * np.pi * rotation
    return np.cos(theta / 2) * _I - 1j * np.sin(theta / 2) * (
        np.cos(phi) * _X + np.sin(phi) * _Y
    )


def _rz(angle: float) -> np.ndarray:
    return np.diag([1.0, np.exp(1j * 2 * np.pi * angle)]).astype(complex)


def _embed(op: np.ndarray, target: int, num_qubits: int) -> np.ndarray:
    out = np.array([[1.0]], dtype=complex)
    for index in range(num_qubits):
        out = np.kron(out, op if index == target else _I)
    return out


def _cz(num_qubits: int, a: int, b: int) -> np.ndarray:
    dim = 2**num_qubits
    diagonal = np.ones(dim, dtype=complex)
    for state in range(dim):
        if (state >> (num_qubits - 1 - a)) & 1 and (state >> (num_qubits - 1 - b)) & 1:
            diagonal[state] = -1.0
    return np.diag(diagonal)


def _unitary(block: ir.Block, order: list[ir.SSAValue]) -> np.ndarray:
    """Build the block's unitary. ``order`` fixes the qubit tensor ordering."""
    index = {value: position for position, value in enumerate(order)}
    num_qubits = len(order)
    total = np.eye(2**num_qubits, dtype=complex)
    for stmt in block.stmts:
        if isinstance(stmt, native_gate.stmts.Rz):
            angle = stmt.rotation_angle.owner.value.unwrap()
            for value in stmt.qubits.owner.values:
                total = _embed(_rz(angle), index[value], num_qubits) @ total
        elif isinstance(stmt, native_gate.stmts.R):
            axis = stmt.axis_angle.owner.value.unwrap()
            rotation = stmt.rotation_angle.owner.value.unwrap()
            for value in stmt.qubits.owner.values:
                total = _embed(_r(axis, rotation), index[value], num_qubits) @ total
        elif isinstance(stmt, native_gate.stmts.CZ):
            (control,) = stmt.controls.owner.values
            (target,) = stmt.targets.owner.values
            total = _cz(num_qubits, index[control], index[target]) @ total
    return total


def _equal_up_to_phase(lhs: np.ndarray, rhs: np.ndarray) -> bool:
    position = np.unravel_index(np.argmax(np.abs(lhs)), lhs.shape)
    return np.allclose(lhs / lhs[position], rhs / rhs[position], atol=1e-9)


def _qubits(block, count):
    values = []
    for _ in range(count):
        new = squin_qubit.stmts.New()
        block.stmts.append(new)
        values.append(new.result)
    return values


def _register(block, values):
    reg = ilist.New(values=tuple(values), elem_type=bloqade_types.QubitType)
    block.stmts.append(reg)
    return reg.result


def _const(block, value):
    const = py.Constant(value)
    block.stmts.append(const)
    return const.result


def _teleportation_block():
    """The h/s/cx kernel's native gate sequence, acting on logical qubit 1."""
    block = ir.Block()
    q0, q1 = _qubits(block, 2)
    r0, r1 = _register(block, [q0]), _register(block, [q1])
    minus_quarter = _const(block, -0.25)
    zero = _const(block, 0.0)
    quarter = _const(block, 0.25)
    add = block.stmts.append
    add(native_gate.stmts.Rz(rotation_angle=minus_quarter, qubits=r1))
    add(native_gate.stmts.R(axis_angle=zero, rotation_angle=minus_quarter, qubits=r1))
    add(native_gate.stmts.Rz(rotation_angle=minus_quarter, qubits=r1))
    add(native_gate.stmts.Rz(rotation_angle=minus_quarter, qubits=r1))
    add(
        native_gate.stmts.R(
            axis_angle=quarter, rotation_angle=minus_quarter, qubits=r1
        )
    )
    add(native_gate.stmts.CZ(controls=r0, targets=r1))
    add(native_gate.stmts.R(axis_angle=quarter, rotation_angle=quarter, qubits=r1))
    return block, [q0, q1]


def _split_forcing_block():
    """Frames diverge across a broadcast R, forcing a split."""
    block = ir.Block()
    q0, q1 = _qubits(block, 2)
    only_q0 = _register(block, [q0])
    both = _register(block, [q0, q1])
    r0, r1 = _register(block, [q0]), _register(block, [q1])
    quarter = _const(block, 0.25)
    zero = _const(block, 0.0)
    add = block.stmts.append
    add(native_gate.stmts.Rz(rotation_angle=quarter, qubits=only_q0))
    add(native_gate.stmts.R(axis_angle=zero, rotation_angle=quarter, qubits=both))
    add(native_gate.stmts.CZ(controls=r0, targets=r1))
    return block, [q0, q1]


@pytest.mark.parametrize(
    "build", [_teleportation_block, _split_forcing_block], ids=["teleport", "split"]
)
def test_rule_preserves_the_unitary_up_to_the_residual(build):
    block, order = build()
    before = _unitary(block, order)

    rule, _ = _run(block)

    after = _unitary(block, order)
    residual = np.eye(2 ** len(order), dtype=complex)
    for value, angle in rule._frame.items():
        residual = _embed(_rz(angle), order.index(value), len(order)) @ residual

    assert _equal_up_to_phase(before, residual @ after)


def test_a_flipped_sign_would_be_caught():
    """Guard the guard: shifting the axis the wrong way must break the check."""
    block, order = _teleportation_block()
    before = _unitary(block, order)

    rule, _ = _run(block)
    after = _unitary(block, order)

    wrong = np.eye(4, dtype=complex)
    for value, angle in rule._frame.items():
        wrong = _embed(_rz(-angle), order.index(value), 2) @ wrong

    assert not _equal_up_to_phase(before, wrong @ after)
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest python/tests/rewrite/test_eliminate_rz_algebra.py -v`
Expected: PASS (3 tests). If `test_rule_preserves_the_unitary_up_to_the_residual`
fails, the shift in the `R` handler has the wrong sign — it must be
`axis - frame`, not `axis + frame`.

- [ ] **Step 3: Commit**

```bash
git add python/tests/rewrite/test_eliminate_rz_algebra.py
git commit -m "test: verify EliminateRz preserves the unitary up to its residual"
```

---

### Task 5: Wire it into the logical pipeline

**Files:**
- Modify: `python/bloqade/lanes/transform/native_to_place.py` (class docstring lines 41–74; `emit` around line 114; `LogicalNativeToPlace` at line 181)
- Test: `python/tests/gemini/test_eliminate_rz_pipeline.py` (create)

**Interfaces:**
- Consumes: `EliminateRz`.
- Produces: `NativeToPlaceBase._post_unroll_rules(self) -> list[RewriteRule]`, default `[]`; `LogicalNativeToPlace` overrides it to return `[EliminateRz()]`.

- [ ] **Step 1: Write the failing tests**

Create `python/tests/gemini/test_eliminate_rz_pipeline.py`:

```python
"""End-to-end: the logical pipeline must emit no local_rz from Clifford gates."""

import math

from bloqade import qubit, squin
from bloqade.gemini import logical as gemini_logical, physical as gemini_physical
from bloqade.gemini.logical import default_post_processing
from bloqade.gemini.logical.rewrite.remove_postprocessing import RemovePostProcessing

from bloqade.lanes.arch.gemini.logical import get_arch_spec as get_logical_spec
from bloqade.lanes.arch.gemini.physical import get_arch_spec as get_physical_spec
from bloqade.lanes.dialects import move
from bloqade.lanes.passes import ASAPPlacePass
from bloqade.lanes.transform import LogicalPipeline, PhysicalPipeline


def _compile(kernel, **kwargs):
    return LogicalPipeline(
        get_logical_spec(), transversal_rewrite=True, simulation=False, **kwargs
    ).emit(kernel)


def _count(method, kind) -> int:
    return sum(1 for stmt in method.callable_region.walk() if isinstance(stmt, kind))


def test_teleportation_kernel_emits_no_local_rz():
    @gemini_logical.kernel(aggressive_unroll=True)
    def kernel():
        reg = qubit.qalloc(2)
        squin.h(reg[1])
        squin.s(reg[1])
        squin.cx(reg[0], reg[1])
        gemini_logical.terminal_measure(reg)

    assert _count(_compile(kernel), move.LocalRz) == 0


def test_local_r_count_is_unchanged():
    """Rz removal must not drop or duplicate rotation pulses."""

    @gemini_logical.kernel(aggressive_unroll=True)
    def kernel():
        reg = qubit.qalloc(2)
        squin.h(reg[1])
        squin.s(reg[1])
        squin.cx(reg[0], reg[1])
        gemini_logical.terminal_measure(reg)

    assert _count(_compile(kernel), move.LocalR) == 3


def test_kernel_with_the_terminal_measure_removed_still_drops_rz():
    """The other shape: no measurement statement to discard the frame at."""

    @gemini_logical.kernel(aggressive_unroll=True)
    def kernel():
        reg = qubit.qalloc(2)
        squin.h(reg[1])
        squin.s(reg[1])
        squin.cx(reg[0], reg[1])
        default_post_processing(reg)

    stripped = kernel.similar()
    RemovePostProcessing(kernel.dialects, delete_terminal_measure=True)(stripped)

    assert _count(_compile(stripped), move.LocalRz) == 0


def test_equal_frames_still_fuse_after_lowering_to_place():
    """Constant sharing must survive the native->place boundary.

    FuseAdjacentGates matches axis angles by SSA identity and circuit2place
    carries them through unchanged, so a fresh constant per statement would stop
    these two identical gates fusing. ASAPPlacePass is the option that runs
    fusion.
    """

    @gemini_logical.kernel(aggressive_unroll=True)
    def kernel():
        reg = qubit.qalloc(2)
        squin.s(reg[0])
        squin.s(reg[1])
        squin.sqrt_x(reg[0])
        squin.sqrt_x(reg[1])
        gemini_logical.terminal_measure(reg)

    assert _count(_compile(kernel, place_opt_type=ASAPPlacePass), move.LocalR) == 1


def test_star_rz_payload_survives():
    """StarRz is a user-requested gadget, not a compiler artifact."""

    @gemini_logical.kernel(aggressive_unroll=True, verify=False)
    def kernel():
        reg = qubit.qalloc(1)
        gemini_logical.star_rz(math.pi / 16, reg[0])
        gemini_logical.terminal_measure(reg)

    assert _count(_compile(kernel), move.LocalRz) == 1


def test_physical_pipeline_is_unchanged():
    """PhysicalNativeToPlace inherits the empty hook, so nothing moves."""

    @gemini_physical.kernel(aggressive_unroll=True)
    def kernel():
        reg = qubit.qalloc(1)
        squin.s(reg[0])
        squin.measure(reg)

    out = PhysicalPipeline(get_physical_spec()).emit(kernel)

    assert _count(out, move.LocalRz) + _count(out, move.GlobalRz) > 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest python/tests/gemini/test_eliminate_rz_pipeline.py -v`
Expected: FAIL — `test_teleportation_kernel_emits_no_local_rz` finds 3.

If `test_physical_pipeline_is_unchanged` or `test_star_rz_payload_survives` fail
now for unrelated reasons (kernel spelling, validation), fix the *test* to match
the existing API first — they are baselines, not targets.

- [ ] **Step 3: Add the hook to the template**

In `python/bloqade/lanes/transform/native_to_place.py`, add to
`NativeToPlaceBase` alongside the other hooks:

```python
    def _post_unroll_rules(self) -> list[RewriteRule]:
        """Rules applied to the flat native IR, after unrolling.

        This is the only window where the program is a flat block of
        ``native.gate`` statements: ``AggressiveUnroll`` has run, and
        ``RewritePlaceOperations`` has not. Default is no rules.
        """
        return []
```

Call it in `emit`, immediately after the unroll, using the same `Walk` idiom the
neighbouring `scf2cf` line already uses:

```python
        AggressiveUnroll(out.dialects, no_raise=no_raise).fixpoint(out)

        if post_unroll_rules := self._post_unroll_rules():
            rewrite.Walk(rewrite.Chain(*post_unroll_rules)).rewrite(out.code)

        self._post_unroll_validation(out, no_raise)
```

Update the class docstring's hook list from four to five, describing
`_post_unroll_rules` in the same style as the others. It deliberately mirrors
`_squin_clifford_rules`, so the class has one way of expressing "rules to run at
stage X" rather than two.

- [ ] **Step 4: Override it in the logical subclass**

Add to `LogicalNativeToPlace`:

```python
    def _post_unroll_rules(self) -> list[RewriteRule]:
        return [EliminateRz()]
```

And import at the top of the file:

```python
from bloqade.lanes.rewrite.eliminate_rz import EliminateRz
```

`RewriteRule` and `rewrite` are already imported there — `_squin_clifford_rules`
uses both.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest python/tests/gemini/test_eliminate_rz_pipeline.py -v`
Expected: PASS (6 tests)

- [ ] **Step 6: Run the full Python suite**

Run: `uv run pytest python/tests -q -x`
Expected: PASS. Failures elsewhere that assert on `LocalRz` counts are *expected*
to change — read each one and update the assertion only if the new count is
correct for this design. Do not weaken an assertion to make it pass.

- [ ] **Step 7: Lint and commit**

```bash
uv run black python && uv run isort python && uv run ruff check python && uv run pyright python
git add python/bloqade/lanes/transform/native_to_place.py python/tests/gemini/test_eliminate_rz_pipeline.py
git commit -m "feat(lanes): run EliminateRz in the logical pipeline via a post-unroll hook"
```

---

### Task 6: Dropped

Not executed, per explicit instruction mid-run. `.github/workflows/ci.yml` runs
`just benchmark-logical` / `just benchmark-physical` on `ubuntu-latest` and
uploads each `latest_<arch>.csv` as an artifact; the committed CSV carries a
`wall_time_ms` column whose values are machine-specific, so regenerating it
locally would write this machine's timings into the repo. The intent was to
leave baseline regeneration to CI rather than do it here.

The measurement itself was not skipped: the final whole-branch review ran the
logical suite and reported the deterministic deltas (`success` and
`nodes_explored` unchanged; `move_count_events`/`move_count_lanes` a wash;
`estimated_fidelity` improved ~1.7% on the `ghz_4`/`ghz_6` cases), which are
recorded in the spec's "Risks & follow-ups" section. **`latest_logical.csv`
itself is still stale** — CI's `benchmarks (logical)` job will report a diff on
this branch until the baseline is regenerated and committed, which needs to
happen before merge but was left for the branch owner rather than done here.

Steps below are preserved for reference; none were run.

This task therefore **measures**, and records what it found. Regenerating the
committed baselines is CI's job, from the branch's own benchmark run.

**Files:**
- Modify: `docs/superpowers/specs/2026-09-14-rz-elimination-design.md` (record the measurement)
- Do NOT modify: `python/benchmarks/harness/latest_logical.csv` or `latest_physical.csv`

- [ ] **Step 1: Ensure a complete environment**

Run: `uv sync --dev --all-extras --index-strategy=unsafe-best-match`
Expected: success. Without the extras, benchmark kernels fail on import errors
that look like solver regressions.

- [ ] **Step 2: Run the logical suite for information**

Run: `just benchmark-logical`
Expected: exit 1 with a diff against the committed baseline. That exit code is
the expected outcome, not a failure — the recipe compares against the committed
CSV, and this change is supposed to move it.

- [ ] **Step 3: Capture the deterministic deltas, then restore the file**

Run: `git diff python/benchmarks/harness/latest_logical.csv`

Read the diff and write down, per changed row: the case, and the before/after of
`success`, `move_count_events`, `move_count_lanes`, `estimated_fidelity`,
`nodes_explored`, `max_depth_reached`. **Ignore `wall_time_ms` entirely** — it is
not part of the comparison and varies by machine.

Then restore the file so no local timings are committed:

```bash
git checkout -- python/benchmarks/harness/latest_logical.csv
```

Stop and report if `success` changed for any case: that is a new compile failure,
not a metric shift, and it means the rule broke something.

- [ ] **Step 4: Confirm the physical suite is untouched**

Run: `just benchmark-physical`
Expected: **exit 0, no diff.** `PhysicalNativeToPlace` inherits the empty
`_post_unroll_rules()` hook, so a physical compile must be bit-identical. A diff
here means the hook is not empty for the physical path — stop and report it as a
Task 5 defect rather than accepting the new numbers.

Then `git checkout -- python/benchmarks/harness/latest_physical.csv` if the run
touched it.

- [ ] **Step 5: Record the result in the spec**

The spec's "Risks & follow-ups" names loss of pulse parallelism as the primary
open risk and says to measure rather than assume. Add a short paragraph there
giving the observed direction and magnitude of the pulse-count change across the
suite — `move_count_events` / `move_count_lanes` totals before and after, and
whether `estimated_fidelity` moved with them. State it plainly, **including if
it is a loss**. If it is a loss on realistic kernels, say so and note that the
fallback is to reconsider whether the rule should be on by default.

- [ ] **Step 6: Confirm the working tree is clean of benchmark edits**

Run: `git status --short python/benchmarks/`
Expected: no output. If either CSV still shows as modified, restore it.

- [ ] **Step 7: Commit**

```bash
git add docs/superpowers/specs/2026-09-14-rz-elimination-design.md
git commit -m "docs: record the measured benchmark impact of Rz elimination"
```

---

## Notes for the implementer

- **Angles are turns.** `0.25` is 90°. Only the test simulator converts to radians.
- **Do not add a flag to disable the rule** in the logical pipeline. The hard
  invariant — no `Rz` reaches the backend — is the feature; an off switch
  produces IR the backend cannot run.
- **If a precondition fires on a real kernel**, that is a finding, not a
  nuisance. Report it rather than loosening the check: it means the IR has a
  shape the design did not anticipate.
- **Do not reintroduce a reader/writer split or a record model.** If a handler
  grows awkward, compare against `move2stack_move` — that structure is the one to
  match.
