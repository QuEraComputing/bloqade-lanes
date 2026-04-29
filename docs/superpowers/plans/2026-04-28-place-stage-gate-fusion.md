# Place-Stage Gate Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `FuseAdjacentGates`, a place-dialect → place-dialect rewrite that fuses textually-adjacent same-op same-params (R/Rz/CZ) statements with disjoint qubit sets within a `place.StaticPlacement` body.

**Architecture:** Single `kirin.rewrite.abc.RewriteRule` subclass matching on `place.StaticPlacement`. Inside `rewrite_Statement`, perform a single linear left-to-right scan over `node.body.blocks[0].stmts`, accumulating "fusion groups" of consecutive fusable statements and emitting one merged statement per group. Fixpoint converges in one iteration; the second invocation observes no further changes.

**Tech Stack:** Python 3.10+, Kirin IR (`kirin.ir`, `kirin.rewrite`, `kirin.rewrite.abc`), pytest, `uv` for env management.

**Spec:** `docs/superpowers/specs/2026-04-28-place-stage-gate-fusion-design.md`

**Tracking issue:** [#582](https://github.com/QuEraComputing/bloqade-lanes/issues/582)

---

## File Structure

| File | Purpose |
|---|---|
| `python/bloqade/lanes/rewrite/fuse_gates.py` (new) | The `FuseAdjacentGates` rewrite rule |
| `python/tests/rewrite/test_fuse_gates.py` (new) | Unit tests, all hand-built place-dialect IR |

The rule is **not** re-exported from `python/bloqade/lanes/rewrite/__init__.py` and is **not** wired into any pipeline. Tests import directly from `bloqade.lanes.rewrite.fuse_gates`.

---

## Conventions used throughout

- Run tests with: `uv run pytest python/tests/rewrite/test_fuse_gates.py -v`
- Run a single test: append `::test_name`
- Pre-commit hooks run on commit; no `--no-verify` unless explicitly noted.
- Commit messages: Conventional Commits (`feat(rewrite): ...`, `test(rewrite): ...`).

---

## Task 1: Test scaffold + skeleton rule (no-op)

**Files:**
- Create: `python/bloqade/lanes/rewrite/fuse_gates.py`
- Create: `python/tests/rewrite/test_fuse_gates.py`

- [ ] **Step 1: Write the failing test for the skeleton rule**

Create `python/tests/rewrite/test_fuse_gates.py`:

```python
"""Tests for FuseAdjacentGates rewrite rule.

Verifies that adjacent same-opcode same-parameter quantum statements with
disjoint qubit sets inside a StaticPlacement body get fused into a single
statement covering the union of qubits.

All test IR is hand-built using kirin.ir primitives; no upstream lowering
is involved.
"""

from kirin import ir, rewrite, types as kirin_types

from bloqade import types as bloqade_types
from bloqade.lanes import types as lanes_types
from bloqade.lanes.dialects import place
from bloqade.lanes.rewrite.fuse_gates import FuseAdjacentGates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wrap_in_static_placement(
    body_block: ir.Block,
    num_qubits: int = 4,
) -> tuple[place.StaticPlacement, ir.Block]:
    """Wrap a populated body block in a StaticPlacement and an outer block.

    Caller is responsible for appending statements to ``body_block`` and for
    setting up its entry-state block argument (see ``_new_body_block``).
    Returns the StaticPlacement and the outer block used as the rewrite target.
    """
    sp_qubits = tuple(
        ir.TestValue(type=bloqade_types.QubitType) for _ in range(num_qubits)
    )
    sp = place.StaticPlacement(qubits=sp_qubits, body=ir.Region(body_block))
    outer = ir.Block([sp])
    return sp, outer


def _new_body_block() -> tuple[ir.Block, ir.SSAValue]:
    """Return an empty body block + its entry-state SSA value."""
    body_block = ir.Block()
    entry_state = body_block.args.append_from(
        lanes_types.StateType, name="entry_state"
    )
    return body_block, entry_state


def _run(outer_block: ir.Block) -> rewrite.abc.RewriteResult:
    """Apply Fixpoint(Walk(FuseAdjacentGates())) and return the result."""
    return rewrite.Fixpoint(rewrite.Walk(FuseAdjacentGates())).rewrite(outer_block)


# ---------------------------------------------------------------------------
# Skeleton: applying the rule on a single-statement body is a no-op.
# ---------------------------------------------------------------------------


def test_single_statement_body_is_unchanged():
    """A body with one R statement is left untouched by the fusion rule."""
    body_block, entry_state = _new_body_block()
    axis = ir.TestValue(type=kirin_types.Float)
    angle = ir.TestValue(type=kirin_types.Float)

    r = place.R(entry_state, axis_angle=axis, rotation_angle=angle, qubits=(0,))
    body_block.stmts.append(r)

    sp, outer = _wrap_in_static_placement(body_block)

    result = _run(outer)

    assert not result.has_done_something
    body_stmts = list(sp.body.blocks[0].stmts)
    assert len(body_stmts) == 1
    assert body_stmts[0] is r
```

- [ ] **Step 2: Run the test to verify it fails (import error)**

Run:
```bash
uv run pytest python/tests/rewrite/test_fuse_gates.py::test_single_statement_body_is_unchanged -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'bloqade.lanes.rewrite.fuse_gates'`.

- [ ] **Step 3: Create the skeleton module**

Create `python/bloqade/lanes/rewrite/fuse_gates.py`:

```python
"""FuseAdjacentGates: fuse adjacent same-op same-params R/Rz/CZ statements.

A place-dialect → place-dialect rewrite that operates on the body of a
``place.StaticPlacement``. Within that body, runs of textually-adjacent
quantum statements with the same opcode, identical non-qubit SSA arguments,
and pairwise-disjoint qubit sets are collapsed into a single statement
covering the union of the qubits.

See ``docs/superpowers/specs/2026-04-28-place-stage-gate-fusion-design.md``
for the full design.
"""

from dataclasses import dataclass

from kirin import ir
from kirin.rewrite import abc as rewrite_abc

from bloqade.lanes.dialects import place


@dataclass
class FuseAdjacentGates(rewrite_abc.RewriteRule):
    """Fuse adjacent same-op same-params R/Rz/CZ statements with disjoint qubits."""

    def rewrite_Statement(self, node: ir.Statement) -> rewrite_abc.RewriteResult:
        if not isinstance(node, place.StaticPlacement):
            return rewrite_abc.RewriteResult()
        # StaticPlacement.check() guarantees a single block.
        body_block = node.body.blocks[0]
        changed = self._fuse_block(body_block)
        return rewrite_abc.RewriteResult(has_done_something=changed)

    def _fuse_block(self, block: ir.Block) -> bool:
        _ = block
        return False
```

- [ ] **Step 4: Run the test to verify it now passes**

Run:
```bash
uv run pytest python/tests/rewrite/test_fuse_gates.py::test_single_statement_body_is_unchanged -v
```
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/rewrite/fuse_gates.py python/tests/rewrite/test_fuse_gates.py
git commit -m "feat(rewrite): scaffold FuseAdjacentGates rule (no-op)"
```

---

## Task 2: Two-way R fusion (happy path)

**Files:**
- Modify: `python/bloqade/lanes/rewrite/fuse_gates.py`
- Modify: `python/tests/rewrite/test_fuse_gates.py`

- [ ] **Step 1: Add the failing test**

Append to `python/tests/rewrite/test_fuse_gates.py`:

```python
# ---------------------------------------------------------------------------
# Two-way R fusion (happy path)
# ---------------------------------------------------------------------------


def test_two_adjacent_r_fuses():
    """Two R(state, axis=%a, angle=%φ, qubits=...) with disjoint qubits fuse.

    The merged R has state_before from the first, qubits = concat in order,
    same axis/angle SSA values; the second R is gone.
    """
    body_block, entry_state = _new_body_block()
    axis = ir.TestValue(type=kirin_types.Float)
    angle = ir.TestValue(type=kirin_types.Float)

    r1 = place.R(entry_state, axis_angle=axis, rotation_angle=angle, qubits=(0,))
    body_block.stmts.append(r1)
    r2 = place.R(r1.state_after, axis_angle=axis, rotation_angle=angle, qubits=(1, 2))
    body_block.stmts.append(r2)

    sp, outer = _wrap_in_static_placement(body_block)

    result = _run(outer)

    assert result.has_done_something
    body_stmts = list(sp.body.blocks[0].stmts)
    assert len(body_stmts) == 1
    merged = body_stmts[0]
    assert isinstance(merged, place.R)
    assert merged.qubits == (0, 1, 2)
    assert merged.axis_angle is axis
    assert merged.rotation_angle is angle
    assert merged.state_before is entry_state
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
uv run pytest python/tests/rewrite/test_fuse_gates.py::test_two_adjacent_r_fuses -v
```
Expected: FAIL — assertion `result.has_done_something` (currently False).

- [ ] **Step 3: Implement the linear scan + R-only merge**

Replace the body of `python/bloqade/lanes/rewrite/fuse_gates.py` with:

```python
"""FuseAdjacentGates: fuse adjacent same-op same-params R/Rz/CZ statements.

A place-dialect → place-dialect rewrite that operates on the body of a
``place.StaticPlacement``. Within that body, runs of textually-adjacent
quantum statements with the same opcode, identical non-qubit SSA arguments,
and pairwise-disjoint qubit sets are collapsed into a single statement
covering the union of the qubits.

See ``docs/superpowers/specs/2026-04-28-place-stage-gate-fusion-design.md``
for the full design.
"""

from dataclasses import dataclass

from kirin import ir
from kirin.rewrite import abc as rewrite_abc

from bloqade.lanes.dialects import place


@dataclass
class FuseAdjacentGates(rewrite_abc.RewriteRule):
    """Fuse adjacent same-op same-params R/Rz/CZ statements with disjoint qubits."""

    def rewrite_Statement(self, node: ir.Statement) -> rewrite_abc.RewriteResult:
        if not isinstance(node, place.StaticPlacement):
            return rewrite_abc.RewriteResult()
        body_block = node.body.blocks[0]
        changed = self._fuse_block(body_block)
        return rewrite_abc.RewriteResult(has_done_something=changed)

    def _fuse_block(self, block: ir.Block) -> bool:
        changed = False
        group: list[ir.Statement] = []

        def flush() -> bool:
            if len(group) >= 2:
                self._merge_group(group)
                group.clear()
                return True
            group.clear()
            return False

        for stmt in list(block.stmts):
            if not isinstance(stmt, place.R):
                if flush():
                    changed = True
                continue
            if not group:
                group.append(stmt)
                continue
            if self._can_extend_r(group, stmt):
                group.append(stmt)
            else:
                if flush():
                    changed = True
                group.append(stmt)
        if flush():
            changed = True
        return changed

    @staticmethod
    def _can_extend_r(group: list[ir.Statement], stmt: ir.Statement) -> bool:
        head = group[0]
        tail = group[-1]
        assert isinstance(head, place.R) and isinstance(stmt, place.R)
        if stmt.axis_angle is not head.axis_angle:
            return False
        if stmt.rotation_angle is not head.rotation_angle:
            return False
        # State-chain adjacency: the stmt's state input must be the tail's state output.
        if stmt.state_before is not tail.state_after:
            return False
        existing_qubits = {q for s in group for q in s.qubits}
        return existing_qubits.isdisjoint(stmt.qubits)

    @staticmethod
    def _merge_group(group: list[ir.Statement]) -> None:
        head = group[0]
        tail = group[-1]
        assert isinstance(head, place.R)
        all_qubits = tuple(q for s in group for q in s.qubits)
        merged = place.R(
            head.state_before,
            axis_angle=head.axis_angle,
            rotation_angle=head.rotation_angle,
            qubits=all_qubits,
        )
        tail.replace_by(merged)
        for stmt in reversed(group[:-1]):
            stmt.delete()
```

- [ ] **Step 4: Run both tests to verify they pass**

Run:
```bash
uv run pytest python/tests/rewrite/test_fuse_gates.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/rewrite/fuse_gates.py python/tests/rewrite/test_fuse_gates.py
git commit -m "feat(rewrite): fuse adjacent place.R statements with disjoint qubits"
```

---

## Task 3: R-fusion negative cases

**Files:**
- Modify: `python/tests/rewrite/test_fuse_gates.py`

These tests exercise predicate failures for `R`. They should already pass from the implementation in Task 2 — the work here is verifying the predicate, not extending it.

- [ ] **Step 1: Add the four negative-case tests**

Append to `python/tests/rewrite/test_fuse_gates.py`:

```python
# ---------------------------------------------------------------------------
# R-fusion: predicate negative cases
# ---------------------------------------------------------------------------


def test_overlapping_qubits_does_not_fuse():
    """R(qubits=[0,1]) + R(qubits=[1,2]) overlap on qubit 1 → no fusion."""
    body_block, entry_state = _new_body_block()
    axis = ir.TestValue(type=kirin_types.Float)
    angle = ir.TestValue(type=kirin_types.Float)

    r1 = place.R(entry_state, axis_angle=axis, rotation_angle=angle, qubits=(0, 1))
    body_block.stmts.append(r1)
    r2 = place.R(r1.state_after, axis_angle=axis, rotation_angle=angle, qubits=(1, 2))
    body_block.stmts.append(r2)

    sp, outer = _wrap_in_static_placement(body_block)

    result = _run(outer)

    assert not result.has_done_something
    body_stmts = list(sp.body.blocks[0].stmts)
    assert body_stmts == [r1, r2]


def test_different_axis_ssa_does_not_fuse():
    """Two R with different axis_angle SSA values → no fusion (SSA-identity)."""
    body_block, entry_state = _new_body_block()
    axis_a = ir.TestValue(type=kirin_types.Float)
    axis_b = ir.TestValue(type=kirin_types.Float)
    angle = ir.TestValue(type=kirin_types.Float)

    r1 = place.R(entry_state, axis_angle=axis_a, rotation_angle=angle, qubits=(0,))
    body_block.stmts.append(r1)
    r2 = place.R(r1.state_after, axis_angle=axis_b, rotation_angle=angle, qubits=(1,))
    body_block.stmts.append(r2)

    sp, outer = _wrap_in_static_placement(body_block)

    result = _run(outer)

    assert not result.has_done_something
    body_stmts = list(sp.body.blocks[0].stmts)
    assert body_stmts == [r1, r2]


def test_different_rotation_angle_ssa_does_not_fuse():
    """Two R with different rotation_angle SSA values → no fusion."""
    body_block, entry_state = _new_body_block()
    axis = ir.TestValue(type=kirin_types.Float)
    angle_a = ir.TestValue(type=kirin_types.Float)
    angle_b = ir.TestValue(type=kirin_types.Float)

    r1 = place.R(entry_state, axis_angle=axis, rotation_angle=angle_a, qubits=(0,))
    body_block.stmts.append(r1)
    r2 = place.R(r1.state_after, axis_angle=axis, rotation_angle=angle_b, qubits=(1,))
    body_block.stmts.append(r2)

    sp, outer = _wrap_in_static_placement(body_block)

    result = _run(outer)

    assert not result.has_done_something
    body_stmts = list(sp.body.blocks[0].stmts)
    assert body_stmts == [r1, r2]


def test_different_opcode_between_blocks_fusion():
    """R; Rz; R does NOT fuse the two Rs even though their qubits are disjoint.

    Strict adjacency: the Rz between them flushes the group.
    """
    body_block, entry_state = _new_body_block()
    axis = ir.TestValue(type=kirin_types.Float)
    angle_r = ir.TestValue(type=kirin_types.Float)
    angle_rz = ir.TestValue(type=kirin_types.Float)

    r1 = place.R(entry_state, axis_angle=axis, rotation_angle=angle_r, qubits=(0,))
    body_block.stmts.append(r1)
    rz = place.Rz(r1.state_after, rotation_angle=angle_rz, qubits=(1,))
    body_block.stmts.append(rz)
    r2 = place.R(rz.state_after, axis_angle=axis, rotation_angle=angle_r, qubits=(2,))
    body_block.stmts.append(r2)

    sp, outer = _wrap_in_static_placement(body_block)

    result = _run(outer)

    assert not result.has_done_something
    body_stmts = list(sp.body.blocks[0].stmts)
    assert body_stmts == [r1, rz, r2]
```

- [ ] **Step 2: Run the new tests to verify they pass**

Run:
```bash
uv run pytest python/tests/rewrite/test_fuse_gates.py -v
```
Expected: 6 passed.

- [ ] **Step 3: Commit**

```bash
git add python/tests/rewrite/test_fuse_gates.py
git commit -m "test(rewrite): cover R fusion predicate negative cases"
```

---

## Task 4: Rz fusion

**Files:**
- Modify: `python/bloqade/lanes/rewrite/fuse_gates.py`
- Modify: `python/tests/rewrite/test_fuse_gates.py`

- [ ] **Step 1: Add the failing Rz fusion test**

Append to `python/tests/rewrite/test_fuse_gates.py`:

```python
# ---------------------------------------------------------------------------
# Rz fusion
# ---------------------------------------------------------------------------


def test_two_adjacent_rz_fuses():
    """Two Rz(state, angle=%θ, qubits=...) with disjoint qubits fuse."""
    body_block, entry_state = _new_body_block()
    angle = ir.TestValue(type=kirin_types.Float)

    rz1 = place.Rz(entry_state, rotation_angle=angle, qubits=(0,))
    body_block.stmts.append(rz1)
    rz2 = place.Rz(rz1.state_after, rotation_angle=angle, qubits=(1, 2))
    body_block.stmts.append(rz2)

    sp, outer = _wrap_in_static_placement(body_block)

    result = _run(outer)

    assert result.has_done_something
    body_stmts = list(sp.body.blocks[0].stmts)
    assert len(body_stmts) == 1
    merged = body_stmts[0]
    assert isinstance(merged, place.Rz)
    assert merged.qubits == (0, 1, 2)
    assert merged.rotation_angle is angle
    assert merged.state_before is entry_state


def test_rz_with_different_angle_does_not_fuse():
    """Two Rz with different rotation_angle SSA values → no fusion."""
    body_block, entry_state = _new_body_block()
    angle_a = ir.TestValue(type=kirin_types.Float)
    angle_b = ir.TestValue(type=kirin_types.Float)

    rz1 = place.Rz(entry_state, rotation_angle=angle_a, qubits=(0,))
    body_block.stmts.append(rz1)
    rz2 = place.Rz(rz1.state_after, rotation_angle=angle_b, qubits=(1,))
    body_block.stmts.append(rz2)

    sp, outer = _wrap_in_static_placement(body_block)

    result = _run(outer)

    assert not result.has_done_something
    body_stmts = list(sp.body.blocks[0].stmts)
    assert body_stmts == [rz1, rz2]
```

- [ ] **Step 2: Run the Rz tests to verify they fail / partial-pass**

Run:
```bash
uv run pytest python/tests/rewrite/test_fuse_gates.py -v
```
Expected: `test_two_adjacent_rz_fuses` FAILS (assertion `result.has_done_something`); `test_rz_with_different_angle_does_not_fuse` passes (Rz isn't matched today, so it's vacuously not fused).

- [ ] **Step 3: Generalize the rule to handle Rz**

Replace `python/bloqade/lanes/rewrite/fuse_gates.py` with:

```python
"""FuseAdjacentGates: fuse adjacent same-op same-params R/Rz/CZ statements.

A place-dialect → place-dialect rewrite that operates on the body of a
``place.StaticPlacement``. Within that body, runs of textually-adjacent
quantum statements with the same opcode, identical non-qubit SSA arguments,
and pairwise-disjoint qubit sets are collapsed into a single statement
covering the union of the qubits.

See ``docs/superpowers/specs/2026-04-28-place-stage-gate-fusion-design.md``
for the full design.
"""

from dataclasses import dataclass

from kirin import ir
from kirin.rewrite import abc as rewrite_abc

from bloqade.lanes.dialects import place

# Opcodes that are eligible for fusion. Other QuantumStmts (Initialize,
# EndMeasure) and non-quantum statements (Yield, etc.) flush the current
# group and do not start a new one.
_FUSABLE_TYPES = (place.R, place.Rz)


@dataclass
class FuseAdjacentGates(rewrite_abc.RewriteRule):
    """Fuse adjacent same-op same-params R/Rz/CZ statements with disjoint qubits."""

    def rewrite_Statement(self, node: ir.Statement) -> rewrite_abc.RewriteResult:
        if not isinstance(node, place.StaticPlacement):
            return rewrite_abc.RewriteResult()
        body_block = node.body.blocks[0]
        changed = self._fuse_block(body_block)
        return rewrite_abc.RewriteResult(has_done_something=changed)

    def _fuse_block(self, block: ir.Block) -> bool:
        changed = False
        group: list[ir.Statement] = []

        def flush() -> bool:
            if len(group) >= 2:
                _merge_group(group)
                group.clear()
                return True
            group.clear()
            return False

        for stmt in list(block.stmts):
            if not isinstance(stmt, _FUSABLE_TYPES):
                if flush():
                    changed = True
                continue
            if not group:
                group.append(stmt)
                continue
            if _can_extend(group, stmt):
                group.append(stmt)
            else:
                if flush():
                    changed = True
                group.append(stmt)
        if flush():
            changed = True
        return changed


def _can_extend(group: list[ir.Statement], stmt: ir.Statement) -> bool:
    head = group[0]
    tail = group[-1]
    if type(stmt) is not type(head):
        return False
    # State-chain adjacency.
    assert isinstance(stmt, _FUSABLE_TYPES)
    assert isinstance(head, _FUSABLE_TYPES)
    assert isinstance(tail, _FUSABLE_TYPES)
    if stmt.state_before is not tail.state_after:
        return False
    if not _same_non_qubit_args(head, stmt):
        return False
    existing_qubits = {q for s in group for q in s.qubits}
    return existing_qubits.isdisjoint(stmt.qubits)


def _same_non_qubit_args(a: ir.Statement, b: ir.Statement) -> bool:
    """SSA-identity comparison of non-qubit args. Assumes type(a) is type(b)."""
    if isinstance(a, place.R):
        assert isinstance(b, place.R)
        return a.axis_angle is b.axis_angle and a.rotation_angle is b.rotation_angle
    if isinstance(a, place.Rz):
        assert isinstance(b, place.Rz)
        return a.rotation_angle is b.rotation_angle
    raise AssertionError(f"unfusable opcode in predicate: {type(a)}")


def _merge_group(group: list[ir.Statement]) -> None:
    head = group[0]
    tail = group[-1]
    if isinstance(head, place.R):
        all_qubits = tuple(q for s in group for q in s.qubits)
        merged: ir.Statement = place.R(
            head.state_before,
            axis_angle=head.axis_angle,
            rotation_angle=head.rotation_angle,
            qubits=all_qubits,
        )
    elif isinstance(head, place.Rz):
        all_qubits = tuple(q for s in group for q in s.qubits)
        merged = place.Rz(
            head.state_before,
            rotation_angle=head.rotation_angle,
            qubits=all_qubits,
        )
    else:
        raise AssertionError(f"unfusable opcode in merge: {type(head)}")
    tail.replace_by(merged)
    for stmt in reversed(group[:-1]):
        stmt.delete()
```

- [ ] **Step 4: Run all tests to verify they pass**

Run:
```bash
uv run pytest python/tests/rewrite/test_fuse_gates.py -v
```
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/rewrite/fuse_gates.py python/tests/rewrite/test_fuse_gates.py
git commit -m "feat(rewrite): extend FuseAdjacentGates to place.Rz"
```

---

## Task 5: CZ fusion with controls-then-targets re-interleaving

**Files:**
- Modify: `python/bloqade/lanes/rewrite/fuse_gates.py`
- Modify: `python/tests/rewrite/test_fuse_gates.py`

- [ ] **Step 1: Add the failing CZ tests**

Append to `python/tests/rewrite/test_fuse_gates.py`:

```python
# ---------------------------------------------------------------------------
# CZ fusion with controls-then-targets re-interleaving
# ---------------------------------------------------------------------------


def test_two_adjacent_cz_fuses_with_reinterleaved_qubits():
    """Two CZ with disjoint qubits fuse; merged.qubits = controls0+controls1+targets0+targets1.

    Verifies the controls-then-targets convention enforced by place.CZ.controls
    and place.CZ.targets (which split qubits in half) is preserved.
    """
    body_block, entry_state = _new_body_block()

    # CZ#0: controls=(0,), targets=(2,)  → qubits=(0, 2)
    cz1 = place.CZ(entry_state, qubits=(0, 2))
    body_block.stmts.append(cz1)
    # CZ#1: controls=(1,), targets=(3,)  → qubits=(1, 3)
    cz2 = place.CZ(cz1.state_after, qubits=(1, 3))
    body_block.stmts.append(cz2)

    sp, outer = _wrap_in_static_placement(body_block)

    result = _run(outer)

    assert result.has_done_something
    body_stmts = list(sp.body.blocks[0].stmts)
    assert len(body_stmts) == 1
    merged = body_stmts[0]
    assert isinstance(merged, place.CZ)
    # merged.controls = (0, 1), merged.targets = (2, 3) → qubits = (0, 1, 2, 3)
    assert merged.qubits == (0, 1, 2, 3)
    assert merged.controls == (0, 1)
    assert merged.targets == (2, 3)
    assert merged.state_before is entry_state


def test_cz_overlapping_controls_does_not_fuse():
    """Two CZ sharing a control qubit do not fuse."""
    body_block, entry_state = _new_body_block()

    cz1 = place.CZ(entry_state, qubits=(0, 2))  # control=0, target=2
    body_block.stmts.append(cz1)
    cz2 = place.CZ(cz1.state_after, qubits=(0, 3))  # control=0, target=3 (overlaps)
    body_block.stmts.append(cz2)

    sp, outer = _wrap_in_static_placement(body_block)

    result = _run(outer)

    assert not result.has_done_something
    body_stmts = list(sp.body.blocks[0].stmts)
    assert body_stmts == [cz1, cz2]


def test_cz_three_way_fusion_preserves_control_target_order():
    """Three CZ statements collapse with all controls first, then all targets."""
    body_block, entry_state = _new_body_block()

    cz1 = place.CZ(entry_state, qubits=(0, 4))  # c=0, t=4
    body_block.stmts.append(cz1)
    cz2 = place.CZ(cz1.state_after, qubits=(1, 5))  # c=1, t=5
    body_block.stmts.append(cz2)
    cz3 = place.CZ(cz2.state_after, qubits=(2, 6))  # c=2, t=6
    body_block.stmts.append(cz3)

    sp, outer = _wrap_in_static_placement(body_block)

    result = _run(outer)

    assert result.has_done_something
    body_stmts = list(sp.body.blocks[0].stmts)
    assert len(body_stmts) == 1
    merged = body_stmts[0]
    assert isinstance(merged, place.CZ)
    assert merged.qubits == (0, 1, 2, 4, 5, 6)
    assert merged.controls == (0, 1, 2)
    assert merged.targets == (4, 5, 6)
```

- [ ] **Step 2: Run the new tests to verify failure**

Run:
```bash
uv run pytest python/tests/rewrite/test_fuse_gates.py -v
```
Expected: `test_two_adjacent_cz_fuses_with_reinterleaved_qubits` and `test_cz_three_way_fusion_preserves_control_target_order` FAIL (CZ not yet handled — `result.has_done_something` is False); `test_cz_overlapping_controls_does_not_fuse` passes vacuously.

- [ ] **Step 3: Extend the rule to handle CZ**

Replace `python/bloqade/lanes/rewrite/fuse_gates.py` with:

```python
"""FuseAdjacentGates: fuse adjacent same-op same-params R/Rz/CZ statements.

A place-dialect → place-dialect rewrite that operates on the body of a
``place.StaticPlacement``. Within that body, runs of textually-adjacent
quantum statements with the same opcode, identical non-qubit SSA arguments,
and pairwise-disjoint qubit sets are collapsed into a single statement
covering the union of the qubits.

See ``docs/superpowers/specs/2026-04-28-place-stage-gate-fusion-design.md``
for the full design.
"""

from dataclasses import dataclass

from kirin import ir
from kirin.rewrite import abc as rewrite_abc

from bloqade.lanes.dialects import place

# Opcodes that are eligible for fusion. Other QuantumStmts (Initialize,
# EndMeasure) and non-quantum statements (Yield, etc.) flush the current
# group and do not start a new one.
_FUSABLE_TYPES = (place.R, place.Rz, place.CZ)


@dataclass
class FuseAdjacentGates(rewrite_abc.RewriteRule):
    """Fuse adjacent same-op same-params R/Rz/CZ statements with disjoint qubits."""

    def rewrite_Statement(self, node: ir.Statement) -> rewrite_abc.RewriteResult:
        if not isinstance(node, place.StaticPlacement):
            return rewrite_abc.RewriteResult()
        body_block = node.body.blocks[0]
        changed = self._fuse_block(body_block)
        return rewrite_abc.RewriteResult(has_done_something=changed)

    def _fuse_block(self, block: ir.Block) -> bool:
        changed = False
        group: list[ir.Statement] = []

        def flush() -> bool:
            if len(group) >= 2:
                _merge_group(group)
                group.clear()
                return True
            group.clear()
            return False

        for stmt in list(block.stmts):
            if not isinstance(stmt, _FUSABLE_TYPES):
                if flush():
                    changed = True
                continue
            if not group:
                group.append(stmt)
                continue
            if _can_extend(group, stmt):
                group.append(stmt)
            else:
                if flush():
                    changed = True
                group.append(stmt)
        if flush():
            changed = True
        return changed


def _can_extend(group: list[ir.Statement], stmt: ir.Statement) -> bool:
    head = group[0]
    tail = group[-1]
    if type(stmt) is not type(head):
        return False
    assert isinstance(stmt, _FUSABLE_TYPES)
    assert isinstance(head, _FUSABLE_TYPES)
    assert isinstance(tail, _FUSABLE_TYPES)
    if stmt.state_before is not tail.state_after:
        return False
    if not _same_non_qubit_args(head, stmt):
        return False
    existing_qubits = {q for s in group for q in s.qubits}
    return existing_qubits.isdisjoint(stmt.qubits)


def _same_non_qubit_args(a: ir.Statement, b: ir.Statement) -> bool:
    """SSA-identity comparison of non-qubit args. Assumes type(a) is type(b)."""
    if isinstance(a, place.R):
        assert isinstance(b, place.R)
        return a.axis_angle is b.axis_angle and a.rotation_angle is b.rotation_angle
    if isinstance(a, place.Rz):
        assert isinstance(b, place.Rz)
        return a.rotation_angle is b.rotation_angle
    if isinstance(a, place.CZ):
        # CZ has no non-qubit args.
        return True
    raise AssertionError(f"unfusable opcode in predicate: {type(a)}")


def _merge_group(group: list[ir.Statement]) -> None:
    head = group[0]
    tail = group[-1]
    if isinstance(head, place.R):
        all_qubits = tuple(q for s in group for q in s.qubits)
        merged: ir.Statement = place.R(
            head.state_before,
            axis_angle=head.axis_angle,
            rotation_angle=head.rotation_angle,
            qubits=all_qubits,
        )
    elif isinstance(head, place.Rz):
        all_qubits = tuple(q for s in group for q in s.qubits)
        merged = place.Rz(
            head.state_before,
            rotation_angle=head.rotation_angle,
            qubits=all_qubits,
        )
    elif isinstance(head, place.CZ):
        # Re-interleave so place.CZ.controls (first half) and place.CZ.targets
        # (second half) keep returning the right halves.
        controls = tuple(c for s in group for c in _cz_controls(s))
        targets = tuple(t for s in group for t in _cz_targets(s))
        merged = place.CZ(head.state_before, qubits=controls + targets)
    else:
        raise AssertionError(f"unfusable opcode in merge: {type(head)}")
    tail.replace_by(merged)
    for stmt in reversed(group[:-1]):
        stmt.delete()


def _cz_controls(stmt: ir.Statement) -> tuple[int, ...]:
    assert isinstance(stmt, place.CZ)
    return stmt.controls


def _cz_targets(stmt: ir.Statement) -> tuple[int, ...]:
    assert isinstance(stmt, place.CZ)
    return stmt.targets
```

- [ ] **Step 4: Run all tests**

Run:
```bash
uv run pytest python/tests/rewrite/test_fuse_gates.py -v
```
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/rewrite/fuse_gates.py python/tests/rewrite/test_fuse_gates.py
git commit -m "feat(rewrite): extend FuseAdjacentGates to place.CZ with control/target reinterleaving"
```

---

## Task 6: N-way fusion in a single rewrite pass

**Files:**
- Modify: `python/tests/rewrite/test_fuse_gates.py`

- [ ] **Step 1: Add the N-way test**

Append to `python/tests/rewrite/test_fuse_gates.py`:

```python
# ---------------------------------------------------------------------------
# N-way fusion in a single pass
# ---------------------------------------------------------------------------


def test_four_adjacent_r_collapse_in_one_pass():
    """Four adjacent fusable R statements collapse to one in a single rewrite invocation."""
    body_block, entry_state = _new_body_block()
    axis = ir.TestValue(type=kirin_types.Float)
    angle = ir.TestValue(type=kirin_types.Float)

    r1 = place.R(entry_state, axis_angle=axis, rotation_angle=angle, qubits=(0,))
    body_block.stmts.append(r1)
    r2 = place.R(r1.state_after, axis_angle=axis, rotation_angle=angle, qubits=(1,))
    body_block.stmts.append(r2)
    r3 = place.R(r2.state_after, axis_angle=axis, rotation_angle=angle, qubits=(2, 3))
    body_block.stmts.append(r3)
    r4 = place.R(r3.state_after, axis_angle=axis, rotation_angle=angle, qubits=(4,))
    body_block.stmts.append(r4)

    sp, outer = _wrap_in_static_placement(body_block)

    result = _run(outer)

    assert result.has_done_something
    body_stmts = list(sp.body.blocks[0].stmts)
    assert len(body_stmts) == 1
    merged = body_stmts[0]
    assert isinstance(merged, place.R)
    assert merged.qubits == (0, 1, 2, 3, 4)
    assert merged.state_before is entry_state
```

- [ ] **Step 2: Run the test**

Run:
```bash
uv run pytest python/tests/rewrite/test_fuse_gates.py::test_four_adjacent_r_collapse_in_one_pass -v
```
Expected: PASS — the linear scan handles N-way groups in a single sweep.

- [ ] **Step 3: Commit**

```bash
git add python/tests/rewrite/test_fuse_gates.py
git commit -m "test(rewrite): cover N-way fusion in single rewrite pass"
```

---

## Task 7: Boundary statements + idempotence + degenerate bodies

**Files:**
- Modify: `python/tests/rewrite/test_fuse_gates.py`

- [ ] **Step 1: Add boundary, idempotence, and degenerate-body tests**

Append to `python/tests/rewrite/test_fuse_gates.py`:

```python
# ---------------------------------------------------------------------------
# Boundary statements: Initialize and EndMeasure flush groups.
# ---------------------------------------------------------------------------


def test_initialize_flushes_group_does_not_start_new_one():
    """An Initialize between two R groups flushes the first and does not start a new one.

    Initialize takes its own state_before SSA value (the previous statement's
    state_after) and produces a state_after; subsequent fusable statements
    that thread through Initialize start a fresh group.
    """
    body_block, entry_state = _new_body_block()
    axis = ir.TestValue(type=kirin_types.Float)
    angle = ir.TestValue(type=kirin_types.Float)
    init_theta = ir.TestValue(type=kirin_types.Float)
    init_phi = ir.TestValue(type=kirin_types.Float)
    init_lam = ir.TestValue(type=kirin_types.Float)

    r1 = place.R(entry_state, axis_angle=axis, rotation_angle=angle, qubits=(0,))
    body_block.stmts.append(r1)
    r2 = place.R(r1.state_after, axis_angle=axis, rotation_angle=angle, qubits=(1,))
    body_block.stmts.append(r2)
    init = place.Initialize(
        r2.state_after,
        theta=init_theta,
        phi=init_phi,
        lam=init_lam,
        qubits=(2,),
    )
    body_block.stmts.append(init)
    r3 = place.R(init.state_after, axis_angle=axis, rotation_angle=angle, qubits=(3,))
    body_block.stmts.append(r3)
    r4 = place.R(r3.state_after, axis_angle=axis, rotation_angle=angle, qubits=(4,))
    body_block.stmts.append(r4)

    sp, outer = _wrap_in_static_placement(body_block, num_qubits=5)

    result = _run(outer)

    assert result.has_done_something
    body_stmts = list(sp.body.blocks[0].stmts)
    # Expected: [merged(r1+r2), init, merged(r3+r4)]
    assert len(body_stmts) == 3
    merged_first, init_seen, merged_second = body_stmts
    assert isinstance(merged_first, place.R)
    assert merged_first.qubits == (0, 1)
    assert init_seen is init
    assert isinstance(merged_second, place.R)
    assert merged_second.qubits == (3, 4)


def test_endmeasure_flushes_preceding_group():
    """An EndMeasure flushes a preceding R run; the EndMeasure itself is untouched."""
    body_block, entry_state = _new_body_block()
    axis = ir.TestValue(type=kirin_types.Float)
    angle = ir.TestValue(type=kirin_types.Float)

    r1 = place.R(entry_state, axis_angle=axis, rotation_angle=angle, qubits=(0,))
    body_block.stmts.append(r1)
    r2 = place.R(r1.state_after, axis_angle=axis, rotation_angle=angle, qubits=(1,))
    body_block.stmts.append(r2)
    em = place.EndMeasure(r2.state_after, qubits=(0, 1))
    body_block.stmts.append(em)

    sp, outer = _wrap_in_static_placement(body_block, num_qubits=2)

    result = _run(outer)

    assert result.has_done_something
    body_stmts = list(sp.body.blocks[0].stmts)
    assert len(body_stmts) == 2
    merged, em_seen = body_stmts
    assert isinstance(merged, place.R)
    assert merged.qubits == (0, 1)
    assert em_seen is em


# ---------------------------------------------------------------------------
# Idempotence and degenerate bodies.
# ---------------------------------------------------------------------------


def test_idempotence_second_application_is_noop():
    """Running the rule a second time on already-fused IR returns has_done_something=False."""
    body_block, entry_state = _new_body_block()
    axis = ir.TestValue(type=kirin_types.Float)
    angle = ir.TestValue(type=kirin_types.Float)

    r1 = place.R(entry_state, axis_angle=axis, rotation_angle=angle, qubits=(0,))
    body_block.stmts.append(r1)
    r2 = place.R(r1.state_after, axis_angle=axis, rotation_angle=angle, qubits=(1,))
    body_block.stmts.append(r2)

    _, outer = _wrap_in_static_placement(body_block)

    result_first = _run(outer)
    assert result_first.has_done_something

    # Second invocation: nothing more to do.
    result_second = _run(outer)
    assert not result_second.has_done_something


def test_empty_body_is_unchanged():
    """An empty body is a no-op."""
    body_block, entry_state = _new_body_block()

    sp, outer = _wrap_in_static_placement(body_block)

    result = _run(outer)

    assert not result.has_done_something
    assert list(sp.body.blocks[0].stmts) == []


def test_no_fusable_groups_is_unchanged():
    """A body of only non-fusable statements (Initialize + EndMeasure) is a no-op."""
    body_block, entry_state = _new_body_block()
    init_theta = ir.TestValue(type=kirin_types.Float)
    init_phi = ir.TestValue(type=kirin_types.Float)
    init_lam = ir.TestValue(type=kirin_types.Float)

    init = place.Initialize(
        entry_state,
        theta=init_theta,
        phi=init_phi,
        lam=init_lam,
        qubits=(0,),
    )
    body_block.stmts.append(init)
    em = place.EndMeasure(init.state_after, qubits=(0,))
    body_block.stmts.append(em)

    sp, outer = _wrap_in_static_placement(body_block, num_qubits=1)

    result = _run(outer)

    assert not result.has_done_something
    body_stmts = list(sp.body.blocks[0].stmts)
    assert body_stmts == [init, em]
```

- [ ] **Step 2: Run all tests**

Run:
```bash
uv run pytest python/tests/rewrite/test_fuse_gates.py -v
```
Expected: 16 passed.

- [ ] **Step 3: Run full lint pass**

Run:
```bash
uv run ruff check python/bloqade/lanes/rewrite/fuse_gates.py python/tests/rewrite/test_fuse_gates.py
uv run black --check python/bloqade/lanes/rewrite/fuse_gates.py python/tests/rewrite/test_fuse_gates.py
uv run isort --check python/bloqade/lanes/rewrite/fuse_gates.py python/tests/rewrite/test_fuse_gates.py
uv run pyright python/bloqade/lanes/rewrite/fuse_gates.py python/tests/rewrite/test_fuse_gates.py
```
Expected: all clean. If any fail, fix them in place; pre-commit hooks will catch the same issues at commit time.

- [ ] **Step 4: Commit**

```bash
git add python/tests/rewrite/test_fuse_gates.py
git commit -m "test(rewrite): cover boundary stmts, idempotence, and degenerate bodies"
```

- [ ] **Step 5: Run the fast test suite to confirm no regressions**

Run:
```bash
uv run pytest python/tests -m "not slow"
```
Expected: all fast tests pass; no regressions. (`@pytest.mark.slow` tests are deferred to CI.)

---

## Done. What's NOT in this plan

- **Wiring the rule into `compile_squin_to_*`.** Out of scope per the spec; tracked as a follow-up on issue #582.
- **Canonicalization pass that reorders commuting same-op statements adjacent.** Separate spec.
- **`Initialize` / `EndMeasure` fusion.** Separate follow-up.

## Final acceptance check (do this after Task 7)

Verify against the issue's acceptance criteria (#582):

- [ ] Module `python/bloqade/lanes/rewrite/fuse_gates.py` exports `FuseAdjacentGates`.
- [ ] Linear scan handles N-way fusion (covered in Task 6).
- [ ] Predicate enforces all four conditions (same opcode, identical non-qubit SSA args, state-chain adjacency, disjoint qubits).
- [ ] CZ merge passes the controls-then-targets ordering invariant (covered in Task 5).
- [ ] Test suite at `python/tests/rewrite/test_fuse_gates.py` covers the matrix from the design doc.
- [ ] Pass is **not** wired into the existing rewrite pipeline.
- [ ] `uv run pytest python/tests/rewrite/test_fuse_gates.py` passes.
- [ ] `uv run pyright`, `uv run ruff check`, `uv run black --check`, `uv run isort --check` are clean.
