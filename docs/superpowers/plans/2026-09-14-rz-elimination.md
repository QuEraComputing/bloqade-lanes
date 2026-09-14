# Rz Elimination (Virtual Z) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove every `Rz` from compiled Gemini logical programs by commuting each Z rotation along its qubit's wire into the terminal Z-basis readout, so the backend never receives a `move.local_rz` it cannot execute.

**Architecture:** One rewrite rule, `EliminateRz`, modelled directly on `move2stack_move.MoveToStackMove`: `rewrite_Block` resets per-block state and scans `list(node.stmts)` in order, dispatching each statement through a `@singledispatchmethod` registry and deferring deletions. The state is a per-qubit phase *frame* (`dict[ir.SSAValue, float]`): each `Rz` is absorbed into it and deleted, each `R` has its `axis_angle` shifted by it, `CZ` and `StarRz` pass through untouched (both diagonal), and whatever frame remains when the block ends is discarded.

**Tech Stack:** Python 3.10+, kirin IR (`kirin.ir`, `kirin.rewrite`), `bloqade.native.dialects.gate`, `bloqade.gemini.logical.dialects.operations`, pytest, numpy (tests only).

**Spec:** `docs/superpowers/specs/2026-09-14-rz-elimination-design.md`

## Global Constraints

- Angles are in **turns**, not radians (`0.25` = 90°). `clifford2native` emits `axis ∈ {0, ¼}`, `rotation ∈ {±¼, ½}`, `Rz ∈ {±¼, ½}`.
- The frame is a **continuous `float`**, never an integer `k ∈ ℤ₄`.
- **Follow `move2stack_move` exactly** for rule structure: `@dataclass`, state in `field(default_factory=..., init=False)`, reset at the top of `rewrite_Block`, per-statement `@singledispatchmethod _rewrite(stmt, to_delete)`, deletions applied in reverse after the loop. Do not invent a parallel record model or a reader/writer split.
- **Preconditions raise, never skip.** A skipped statement leaves an `Rz` behind and breaks the guarantee. (kirin's own `cse` uses `continue` for region-bearing statements; here that would silently lose a phase, so raise instead.)
- **No `require_clifford_angles` flag.** A non-Clifford angle has no native mapping, so it cannot reach this rule; and "logical programs are Clifford-only mid-circuit" is `GeminiLogicalValidation`'s invariant, not this rule's to re-check.
- Imports absolute from `bloqade.lanes`. snake_case files, PascalCase classes. Type annotations enforced by pyright.
- Lint before each commit: `uv run black python && uv run isort python && uv run ruff check python && uv run pyright python`.
- Commit messages follow Conventional Commits.

## Why a block scan rather than a per-statement peephole

Recorded because it is the one structural choice that needs defending.

The local formulation — `rewrite_Statement` matches an `Rz`, swaps it with its successor, and `Fixpoint(Walk(...))` bubbles it to the end — is the right shape for *stateless* rewrites like `RewriteNonCliffordToU3`. It does not work here. `Walk` freezes its worklist before rewriting (its own comment: *"because the rewrite pass may mutate the node thus we need to save the list of nodes to be processed first"*), so one pass moves each `Rz` exactly one position, and `Fixpoint.max_iter` defaults to 32, returning `exceeded_max_iter=True` **silently** on exhaustion. Measured on real native blocks:

```
    teleport(2q):   18 stmts,   3 Rz, earliest Rz 13 positions from the end
         ghz(6q):   37 stmts,   2 Rz, earliest Rz 28 positions from the end
  layered(8q,d4):  199 stmts,  96 Rz, earliest Rz 188 positions from the end
```

The 8-qubit depth-4 kernel needs 188 iterations against a limit of 32 — it would leave most of its 96 `Rz` in the IR and compile "successfully". ghz(6q) at 28 is worse in kind: just under the limit, so the failure would be data-dependent and would pass every small test.

An ordered `rewrite_Block` scan is O(n) and cannot silently under-apply. It is also the established shape for stateful rewrites — `move2stack_move`, `stack_move2move`, `state`, `place2move` here, and `cse`, `compactify`, `apply_type`, `wrap_const` in kirin itself.

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
- Produces: `EliminateRzError`, and `EliminateRz()` — a `kirin.rewrite.abc.RewriteRule` whose `rewrite_Block(block) -> RewriteResult` performs the scan. After it runs, `rule._frame: dict[ir.SSAValue, float]` holds the residual (read by tests in Task 4).

- [ ] **Step 1: Write the failing tests**

Create `python/tests/rewrite/test_eliminate_rz.py`:

```python
"""Tests for the EliminateRz rewrite rule, on hand-built native-dialect IR."""

import pytest
from kirin import ir, types as kirin_types
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


def test_rz_is_deleted():
    block = ir.Block()
    (q0,) = _qubits(block, 1)
    reg = _register(block, [q0])
    angle = _const(block, 0.25)
    block.stmts.append(native_gate.stmts.Rz(rotation_angle=angle, qubits=reg))

    result = EliminateRz().rewrite_Block(block)

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

    EliminateRz().rewrite_Block(block)

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

    EliminateRz().rewrite_Block(block)

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

    EliminateRz().rewrite_Block(block)

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

    rule = EliminateRz()
    rule.rewrite_Block(block)

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

    EliminateRz().rewrite_Block(block)
    second = EliminateRz().rewrite_Block(block)

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

Structured after ``move2stack_move.MoveToStackMove``: ordered ``rewrite_Block``
scan, per-block state reset at the top, ``@singledispatchmethod`` per statement,
deletions deferred. See
``docs/superpowers/specs/2026-09-14-rz-elimination-design.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import singledispatchmethod

from kirin import ir
from kirin.dialects import ilist, py
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
    """Remove every ``Rz`` from a flat native-dialect block."""

    _frame: dict[ir.SSAValue, float] = field(default_factory=dict, init=False)
    """Pending Z phase per qubit, in turns. The residual is discarded."""

    _constants: dict[float, ir.SSAValue] = field(default_factory=dict, init=False)
    """One SSA value per distinct angle, so downstream fusion still matches."""

    _qubits: set[ir.SSAValue] = field(default_factory=set, init=False)

    def rewrite_Block(self, node: ir.Block) -> RewriteResult:
        self._frame = {}
        self._constants = {}
        self._qubits = set()
        to_delete: list[ir.Statement] = []

        result = RewriteResult()
        # Snapshot before iterating: the handlers insert replacements, and those
        # must not be revisited.
        for stmt in list(node.stmts):
            result = result.join(self._rewrite(stmt, to_delete))

        for stmt in reversed(to_delete):
            stmt.delete()
        if to_delete:
            result = result.join(RewriteResult(has_done_something=True))
        return result

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
    def _rewrite(
        self, stmt: ir.Statement, to_delete: list[ir.Statement]
    ) -> RewriteResult:
        """Default: record qubit allocations, pass over everything else."""
        if len(stmt.results) == 1 and stmt.results[0].type.is_subseteq(
            bloqade_types.QubitType
        ):
            self._qubits.add(stmt.results[0])
        return RewriteResult()

    @_rewrite.register(py.Constant)
    def _(self, stmt: py.Constant, to_delete) -> RewriteResult:
        data = stmt.value.unwrap()
        if isinstance(data, (int, float)) and not isinstance(data, bool):
            self._constants.setdefault(_normalize(float(data)), stmt.result)
        return RewriteResult()

    @_rewrite.register(ilist.New)
    def _(self, stmt: ilist.New, to_delete) -> RewriteResult:
        return RewriteResult()

    @_rewrite.register(native_gate.stmts.Rz)
    def _(self, stmt: native_gate.stmts.Rz, to_delete) -> RewriteResult:
        angle = self._const_float(stmt, stmt.rotation_angle)
        for qubit in self._qubit_values(stmt, stmt.qubits):
            self._frame[qubit] = self._frame.get(qubit, 0.0) + angle
        to_delete.append(stmt)
        return RewriteResult(has_done_something=True)

    @_rewrite.register(native_gate.stmts.R)
    def _(self, stmt: native_gate.stmts.R, to_delete) -> RewriteResult:
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

        to_delete.append(stmt)
        return RewriteResult(has_done_something=True)

    @_rewrite.register(native_gate.stmts.CZ)
    def _(self, stmt: native_gate.stmts.CZ, to_delete) -> RewriteResult:
        # Diagonal: commutes with Rz on each qubit independently, so the two
        # sides' frames need not agree and nothing changes.
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

    EliminateRz().rewrite_Block(block)

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

    EliminateRz().rewrite_Block(block)

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

    EliminateRz().rewrite_Block(block)

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

    EliminateRz().rewrite_Block(block)

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

    EliminateRz().rewrite_Block(block)

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
        EliminateRz().rewrite_Block(block)


def test_unknown_statement_touching_a_qubit_raises():
    """The scan cannot know whether an unrecognised gate is diagonal."""
    block = ir.Block()
    (q0,) = _qubits(block, 1)
    reg = _register(block, [q0])
    block.stmts.append(squin_qubit.stmts.Measure(qubits=reg))

    with pytest.raises(EliminateRzError, match="phase-neutral"):
        EliminateRz().rewrite_Block(block)


def test_non_ilist_register_raises():
    block = ir.Block()
    angle = _const(block, 0.25)
    opaque = ir.TestValue(
        type=ilist.IListType[bloqade_types.QubitType, kirin_types.Any]
    )
    block.stmts.append(native_gate.stmts.Rz(rotation_angle=angle, qubits=opaque))

    with pytest.raises(EliminateRzError, match="ilist.New"):
        EliminateRz().rewrite_Block(block)


def test_duplicate_qubit_in_one_register_raises():
    block = ir.Block()
    (q0,) = _qubits(block, 1)
    doubled = _register(block, [q0, q0])
    angle = _const(block, 0.25)
    block.stmts.append(native_gate.stmts.Rz(rotation_angle=angle, qubits=doubled))

    with pytest.raises(EliminateRzError, match="more than once"):
        EliminateRz().rewrite_Block(block)


def test_non_constant_angle_raises():
    block = ir.Block()
    (q0,) = _qubits(block, 1)
    reg = _register(block, [q0])
    angle = ir.TestValue(type=kirin_types.Float)
    block.stmts.append(native_gate.stmts.Rz(rotation_angle=angle, qubits=reg))

    with pytest.raises(EliminateRzError, match="constant"):
        EliminateRz().rewrite_Block(block)


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
        EliminateRz().rewrite_Block(block)


def test_initialize_at_the_head_of_a_wire_is_fine():
    block = ir.Block()
    (q0,) = _qubits(block, 1)
    reg = _register(block, [q0])
    zero = _const(block, 0.0)
    init = operations.Initialize(theta=zero, phi=zero, lam=zero, qubits=reg)
    block.stmts.append(init)

    EliminateRz().rewrite_Block(block)

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

    rule = EliminateRz()
    rule.rewrite_Block(block)

    assert _of_type(block, operations.StarRz) == [star]
    assert rule._frame[q0] == 0.25


def test_terminal_measurement_discards_the_frame():
    block = ir.Block()
    (q0,) = _qubits(block, 1)
    reg = _register(block, [q0])
    quarter = _const(block, 0.25)
    block.stmts.append(native_gate.stmts.Rz(rotation_angle=quarter, qubits=reg))
    block.stmts.append(operations.TerminalLogicalMeasurement(qubits=reg))

    rule = EliminateRz()
    rule.rewrite_Block(block)

    assert rule._frame == {}


def test_frame_left_at_end_of_block_is_simply_discarded():
    """The shape left by RemovePostProcessing: no measurement statement at all."""
    block = ir.Block()
    (q0,) = _qubits(block, 1)
    reg = _register(block, [q0])
    quarter = _const(block, 0.25)
    block.stmts.append(native_gate.stmts.Rz(rotation_angle=quarter, qubits=reg))

    rule = EliminateRz()
    result = rule.rewrite_Block(block)

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
    def _rewrite(
        self, stmt: ir.Statement, to_delete: list[ir.Statement]
    ) -> RewriteResult:
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
    def _(self, stmt: operations.StarRz, to_delete) -> RewriteResult:
        # Diagonal, like CZ: the frame commutes past it exactly. Its own
        # rotation is the payload of a user-requested gadget, not ours to remove.
        return RewriteResult()

    @_rewrite.register(operations.Initialize)
    def _(self, stmt: operations.Initialize, to_delete) -> RewriteResult:
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
    def _(
        self, stmt: operations.TerminalLogicalMeasurement, to_delete
    ) -> RewriteResult:
        # The residual is diagonal and the readout is in the Z basis, so it
        # cannot shift any outcome. Drop it.
        for qubit in self._qubit_values(stmt, stmt.qubits):
            self._frame.pop(qubit, None)
        return RewriteResult()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest python/tests/rewrite/test_eliminate_rz.py -v`
Expected: PASS (21 tests)

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
- Consumes: `EliminateRz`, and `rule._frame` after `rewrite_Block`.
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
from kirin import ir
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

    rule = EliminateRz()
    rule.rewrite_Block(block)

    after = _unitary(block, order)
    residual = np.eye(2 ** len(order), dtype=complex)
    for value, angle in rule._frame.items():
        residual = _embed(_rz(angle), order.index(value), len(order)) @ residual

    assert _equal_up_to_phase(before, residual @ after)


def test_a_flipped_sign_would_be_caught():
    """Guard the guard: shifting the axis the wrong way must break the check."""
    block, order = _teleportation_block()
    before = _unitary(block, order)

    rule = EliminateRz()
    rule.rewrite_Block(block)
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

### Task 6: Regenerate benchmark baselines

The rule is on by default, so the deterministic benchmark metrics move.
`AGENT.md` makes regenerating them a rule for any such change.

**Files:**
- Modify: `python/benchmarks/harness/latest_logical.csv`
- Modify: `docs/superpowers/specs/2026-09-14-rz-elimination-design.md` (record the measured result)

- [ ] **Step 1: Ensure a complete environment**

Run: `uv sync --dev --all-extras --index-strategy=unsafe-best-match`
Expected: success. Without the extras, kernels fail on import errors that look
like solver regressions.

- [ ] **Step 2: Run the logical suite**

Run: `just benchmark-logical`
Expected: exit 1 with a diff. The recipe writes the CSV in place while comparing,
so the working copy is updated even on failure.

- [ ] **Step 3: Inspect the diff**

Run: `git diff python/benchmarks/harness/latest_logical.csv`

Stop if any of these do not hold:
- The `success` column is unchanged — no new failures.
- `move_count_events` / `move_count_lanes` shifts are explained by removed `Rz`
  pulses and any `R` splits.
- `estimated_fidelity` moves in the direction the pulse-count change implies.
- Ignore `wall_time_ms`; it is not compared and varies by machine.

- [ ] **Step 4: Confirm the physical suite did not move**

Run: `just benchmark-physical`
Expected: exit 0, no diff. If it moved, the hook is not empty for physical
compiles — stop and fix Task 5 rather than committing a new baseline.

- [ ] **Step 5: Confirm determinism**

Run `just benchmark-logical` again, then
`git diff python/benchmarks/harness/latest_logical.csv`.
Expected: no further change.

- [ ] **Step 6: Record the parallelism result in the spec**

The spec's primary open risk says to measure rather than assume. Add a short
paragraph under "Risks & follow-ups" in
`docs/superpowers/specs/2026-09-14-rz-elimination-design.md` giving the observed
direction and magnitude of the pulse-count change across the suite. State it
plainly, including if it is a loss.

- [ ] **Step 7: Commit**

```bash
git add python/benchmarks/harness/latest_logical.csv docs/superpowers/specs/2026-09-14-rz-elimination-design.md
git commit -m "test(benchmarks): regenerate logical baselines for Rz elimination"
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
