# Rz Elimination (Virtual Z) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove every `Rz` from compiled Gemini logical programs by commuting each Z rotation along its qubit's wire into the terminal Z-basis readout, so the backend never receives a `move.local_rz` it cannot execute.

**Architecture:** A single native-dialect rewrite rule, `EliminateRz`, implementing `rewrite_Block`. It sweeps the (already flat, already unrolled) block in order carrying a per-qubit phase *frame*; each `Rz` is absorbed into the frame and deleted, each `R` has its `axis_angle` shifted by the frame, `CZ` and `StarRz` pass through untouched (both diagonal), and the residual frame is discarded when the sweep ends. The algebra lives in a pure, IR-free `sweep()` function; a thin adapter reads native statements into records and writes the results back.

**Tech Stack:** Python 3.10+, kirin IR (`kirin.ir`, `kirin.rewrite`), `bloqade.native.dialects.gate`, `bloqade.gemini.logical.dialects.operations`, pytest, numpy (tests only).

**Spec:** `docs/superpowers/specs/2026-09-14-rz-elimination-design.md`

## Global Constraints

- Angles are in **turns**, not radians (`0.25` = 90°). `clifford2native` emits `axis ∈ {0, ¼}`, `rotation ∈ {±¼, ½}`, `Rz ∈ {±¼, ½}`.
- The frame is a **continuous `float`**, never an integer `k ∈ ℤ₄`. Quantizing is explicitly rejected by the spec.
- **Preconditions raise, never skip.** A skipped statement leaves an `Rz` behind and breaks the guarantee, so the rule raises rather than returning a no-op `RewriteResult`. Rewrite rules carry no `no_raise` field, so there is nothing to suppress it.
- **This is a `RewriteRule`, not a `Pass`.** The convention here is that a transform is one rewrite rule, and passes exist to *combine* rules. `FuseAdjacentGates` is the local precedent: a stateful whole-block sweep written as a rule.
- Imports are absolute from the `bloqade.lanes` namespace. snake_case files, PascalCase classes. Type annotations are enforced by pyright.
- Lint before each commit: `uv run black python && uv run isort python && uv run ruff check python && uv run pyright python`. Pre-commit runs these too.
- Commit messages follow Conventional Commits (`feat(rewrite): ...`, `test: ...`).

## File Structure

| File | Responsibility |
|---|---|
| `python/bloqade/lanes/rewrite/eliminate_rz.py` (create) | Everything: the IR-free `sweep()` core, the native-IR reader, the writer, and the `EliminateRz` rewrite rule. One file because the parts are meaningless apart and total ~300 lines. |
| `python/bloqade/lanes/transform/native_to_place.py` (modify) | Gains the fifth hook `_post_unroll_rules()`, mirroring `_squin_clifford_rules()`; `LogicalNativeToPlace` overrides it. |
| `python/tests/rewrite/test_eliminate_rz_core.py` (create) | Core algebra — no IR at all. Qubit keys are plain strings. |
| `python/tests/rewrite/test_eliminate_rz.py` (create) | IR-level: reader preconditions, writer structure, constant sharing. |
| `python/tests/gemini/test_eliminate_rz_pipeline.py` (create) | End-to-end through `LogicalPipeline`; physical-pipeline-unchanged regression. |

---

### Task 1: Sweep core — data model and frame accumulation

**Files:**
- Create: `python/bloqade/lanes/rewrite/eliminate_rz.py`
- Test: `python/tests/rewrite/test_eliminate_rz_core.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `EliminateRzError`, `GateKind` (enum: `RZ`, `R`, `DIAGONAL`, `INIT`, `MEASURE`), `Gate(kind, qubits: tuple[Hashable, ...], angle: float = 0.0)`, `Rewrite(qubits: tuple[Hashable, ...], axis_angle: float)`, `Action(delete: bool = False, groups: tuple[Rewrite, ...] = ())`, `SweepResult(actions: tuple[Action, ...], residual: dict[Hashable, float])`, `sweep(gates: Sequence[Gate], *, require_clifford_angles: bool = True) -> SweepResult`.

- [ ] **Step 1: Write the failing tests**

Create `python/tests/rewrite/test_eliminate_rz_core.py`:

```python
"""Tests for the IR-free phase-frame sweep behind EliminateRz.

Qubit keys are plain strings here: the core never touches kirin IR, which is
the whole point of keeping it separate.
"""

from bloqade.lanes.rewrite.eliminate_rz import (
    Action,
    Gate,
    GateKind,
    Rewrite,
    sweep,
)


def test_rz_is_absorbed_into_the_frame_and_deleted():
    result = sweep([Gate(GateKind.RZ, ("q0",), 0.25)])

    assert result.actions == (Action(delete=True),)
    assert result.residual == {"q0": 0.25}


def test_r_axis_is_shifted_by_the_pending_frame():
    gates = [
        Gate(GateKind.RZ, ("q0",), 0.25),
        Gate(GateKind.R, ("q0",), 0.0),
    ]

    result = sweep(gates)

    assert result.actions[0] == Action(delete=True)
    # (0.0 - 0.25) mod 1 == 0.75
    assert result.actions[1] == Action(groups=(Rewrite(("q0",), 0.75),))


def test_r_is_left_alone_when_the_frame_is_zero():
    result = sweep([Gate(GateKind.R, ("q0",), 0.25)])

    assert result.actions == (Action(),)
    assert result.residual == {}


def test_frames_accumulate_across_multiple_rz():
    gates = [
        Gate(GateKind.RZ, ("q0",), 0.25),
        Gate(GateKind.RZ, ("q0",), 0.5),
    ]

    result = sweep(gates)

    assert result.residual == {"q0": 0.75}


def test_frames_are_tracked_per_qubit():
    gates = [
        Gate(GateKind.RZ, ("q0",), 0.25),
        Gate(GateKind.R, ("q1",), 0.0),
    ]

    result = sweep(gates)

    # q1 never saw an Rz, so its R is untouched.
    assert result.actions[1] == Action()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest python/tests/rewrite/test_eliminate_rz_core.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bloqade.lanes.rewrite.eliminate_rz'`

- [ ] **Step 3: Write the minimal implementation**

Create `python/bloqade/lanes/rewrite/eliminate_rz.py`:

```python
"""EliminateRz: remove ``Rz`` from native-dialect programs by phase commutation.

Sweeps a flat block in order carrying a per-qubit phase *frame*. Each ``Rz`` is
absorbed into the frame and deleted; each ``R`` has its axis angle shifted by the
frame of the qubits it addresses; ``CZ`` and ``StarRz`` are diagonal and pass
through untouched. Whatever frame remains when the sweep ends is discarded --
sound because the residual is diagonal and the device's readout is in the Z
basis.

Two exact identities do all the work (angles in turns)::

    R(phi, theta) . Rz(alpha) = Rz(alpha) . R(phi - alpha, theta)
    CZ . Rz(alpha)            = Rz(alpha) . CZ

See ``docs/superpowers/specs/2026-09-14-rz-elimination-design.md``.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from enum import Enum, auto

__all__ = [
    "Action",
    "EliminateRzError",
    "Gate",
    "GateKind",
    "Rewrite",
    "SweepResult",
    "sweep",
]

# Angles are compared and grouped at this many decimal places. Twelve is far
# below any physically meaningful resolution, and rounding keeps equal angles
# landing on one dict key so the constant cache can share an SSA value.
_ANGLE_NDIGITS = 12


class EliminateRzError(Exception):
    """A precondition of the Rz elimination sweep was violated."""


class GateKind(Enum):
    """How the sweep treats a statement."""

    RZ = auto()
    """Absorbed into the frame, then deleted."""

    R = auto()
    """Axis angle shifted by the frame; splits if its qubits disagree."""

    DIAGONAL = auto()
    """Commutes with Rz exactly (CZ, StarRz). Untouched."""

    INIT = auto()
    """State preparation. The frame must be zero here."""

    MEASURE = auto()
    """Terminal readout. Frames for its qubits are discarded."""


@dataclass(frozen=True)
class Gate:
    """One statement, reduced to what the sweep needs to know about it."""

    kind: GateKind
    qubits: tuple[Hashable, ...]
    angle: float = 0.0
    """``RZ``: the rotation angle. ``R``: the axis angle. Unused otherwise."""


@dataclass(frozen=True)
class Rewrite:
    """One output group of an ``R``: these qubits get this axis angle."""

    qubits: tuple[Hashable, ...]
    axis_angle: float


@dataclass(frozen=True)
class Action:
    """What to do with the statement at the same index as this action."""

    delete: bool = False
    groups: tuple[Rewrite, ...] = ()
    """Empty means "leave the statement alone"."""


@dataclass
class SweepResult:
    actions: tuple[Action, ...]
    residual: dict[Hashable, float] = field(default_factory=dict)


def _normalize(angle: float) -> float:
    """Fold an angle into [0, 1) turns and round it onto the grouping grid."""
    return round(angle % 1.0, _ANGLE_NDIGITS) % 1.0


def sweep(
    gates: Sequence[Gate], *, require_clifford_angles: bool = True
) -> SweepResult:
    """Run the phase-frame sweep over ``gates`` in circuit order.

    Returns one ``Action`` per input gate, plus the residual frame the caller is
    expected to discard. The invariant callers can test against is::

        U_before == Rz(residual) . U_after
    """
    frame: dict[Hashable, float] = {}
    actions: list[Action] = []

    for gate in gates:
        if gate.kind is GateKind.RZ:
            for qubit in gate.qubits:
                frame[qubit] = frame.get(qubit, 0.0) + gate.angle
            actions.append(Action(delete=True))
        elif gate.kind is GateKind.R:
            actions.append(_rewrite_r(gate, frame))
        else:
            actions.append(Action())

    return SweepResult(tuple(actions), frame)


def _rewrite_r(gate: Gate, frame: dict[Hashable, float]) -> Action:
    """Group ``gate``'s qubits by the axis angle each ends up with."""
    groups: dict[float, list[Hashable]] = {}
    for qubit in gate.qubits:
        axis = _normalize(gate.angle - frame.get(qubit, 0.0))
        groups.setdefault(axis, []).append(qubit)

    unchanged = len(groups) == 1 and next(iter(groups)) == _normalize(gate.angle)
    if unchanged:
        return Action()

    return Action(
        groups=tuple(
            Rewrite(tuple(qubits), axis) for axis, qubits in groups.items()
        )
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest python/tests/rewrite/test_eliminate_rz_core.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Lint**

Run: `uv run black python && uv run isort python && uv run ruff check python && uv run pyright python`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add python/bloqade/lanes/rewrite/eliminate_rz.py python/tests/rewrite/test_eliminate_rz_core.py
git commit -m "feat(rewrite): add the phase-frame sweep core for Rz elimination"
```

---

### Task 2: Sweep core — splitting, measurement, and preconditions

**Files:**
- Modify: `python/bloqade/lanes/rewrite/eliminate_rz.py`
- Test: `python/tests/rewrite/test_eliminate_rz_core.py`

**Interfaces:**
- Consumes: everything from Task 1.
- Produces: no new names. `sweep()` gains duplicate-qubit rejection, the `require_clifford_angles` check, the `INIT` zero-frame assertion, `MEASURE` frame discard, and a partition assertion on splits.

- [ ] **Step 1: Write the failing tests**

Append to `python/tests/rewrite/test_eliminate_rz_core.py`:

```python
import pytest

from bloqade.lanes.rewrite.eliminate_rz import EliminateRzError


def test_r_splits_when_its_qubits_carry_different_frames():
    gates = [
        Gate(GateKind.RZ, ("q0",), 0.25),
        Gate(GateKind.R, ("q0", "q1"), 0.0),
    ]

    result = sweep(gates)

    groups = result.actions[1].groups
    assert {group.qubits: group.axis_angle for group in groups} == {
        ("q0",): 0.75,
        ("q1",): 0.0,
    }


def test_split_groups_partition_the_original_qubits():
    gates = [
        Gate(GateKind.RZ, ("q0",), 0.25),
        Gate(GateKind.RZ, ("q2",), 0.5),
        Gate(GateKind.R, ("q0", "q1", "q2"), 0.0),
    ]

    result = sweep(gates)

    covered = [q for group in result.actions[2].groups for q in group.qubits]
    assert sorted(covered) == ["q0", "q1", "q2"]
    assert len(covered) == len(set(covered))


def test_qubits_with_equal_frames_stay_in_one_group():
    gates = [
        Gate(GateKind.RZ, ("q0", "q1"), 0.25),
        Gate(GateKind.R, ("q0", "q1"), 0.0),
    ]

    result = sweep(gates)

    assert result.actions[1].groups == (Rewrite(("q0", "q1"), 0.75),)


def test_cz_is_untouched_even_when_its_qubits_disagree():
    gates = [
        Gate(GateKind.RZ, ("q0",), 0.25),
        Gate(GateKind.DIAGONAL, ("q0", "q1")),
    ]

    result = sweep(gates)

    assert result.actions[1] == Action()
    assert result.residual == {"q0": 0.25}


def test_measure_discards_the_frame_for_its_qubits():
    gates = [
        Gate(GateKind.RZ, ("q0",), 0.25),
        Gate(GateKind.MEASURE, ("q0",)),
    ]

    result = sweep(gates)

    assert result.residual == {}


def test_duplicate_qubit_within_one_gate_raises():
    with pytest.raises(EliminateRzError, match="more than once"):
        sweep([Gate(GateKind.RZ, ("q0", "q0"), 0.25)])


def test_non_clifford_rz_angle_raises_by_default():
    with pytest.raises(EliminateRzError, match="0.1"):
        sweep([Gate(GateKind.RZ, ("q0",), 0.1)])


def test_non_clifford_r_axis_angle_raises_by_default():
    with pytest.raises(EliminateRzError, match="0.1"):
        sweep([Gate(GateKind.R, ("q0",), 0.1)])


def test_non_clifford_angles_allowed_when_the_check_is_disabled():
    result = sweep(
        [Gate(GateKind.RZ, ("q0",), 0.1)], require_clifford_angles=False
    )

    assert result.residual == {"q0": 0.1}


def test_initialize_with_a_pending_frame_raises():
    gates = [
        Gate(GateKind.RZ, ("q0",), 0.25),
        Gate(GateKind.INIT, ("q0",)),
    ]

    with pytest.raises(EliminateRzError, match="Initialize"):
        sweep(gates)


def test_initialize_with_no_pending_frame_is_fine():
    result = sweep([Gate(GateKind.INIT, ("q0",))])

    assert result.actions == (Action(),)


def test_axis_angles_stay_on_the_quarter_turn_lattice():
    """Closure: the Clifford precondition keeps the gate set finite.

    Every emitted axis angle must remain a multiple of a quarter turn, which is
    what keeps the surviving gates within {X, Y, sqrt(X), sqrt(Y)} and adjoints --
    all Steane-transversal Cliffords.
    """
    gates = [
        Gate(GateKind.RZ, ("q0",), 0.25),
        Gate(GateKind.R, ("q0",), 0.25),
        Gate(GateKind.RZ, ("q0",), 0.5),
        Gate(GateKind.R, ("q0",), 0.0),
        Gate(GateKind.RZ, ("q0",), -0.25),
        Gate(GateKind.R, ("q0",), 0.25),
    ]

    result = sweep(gates)

    emitted = [
        group.axis_angle for action in result.actions for group in action.groups
    ]
    assert emitted, "expected at least one rewritten axis angle"
    for axis in emitted:
        quarters = axis * 4
        assert abs(quarters - round(quarters)) < 1e-9, axis
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest python/tests/rewrite/test_eliminate_rz_core.py -v`
Expected: FAIL — the raising tests fail with `DID NOT RAISE`, `test_measure_discards_the_frame_for_its_qubits` fails with `{'q0': 0.25} != {}`.

- [ ] **Step 3: Write the implementation**

Add these constants and helpers to `eliminate_rz.py`, above `sweep`:

```python
# Clifford angles are multiples of a quarter turn. The values are exact today
# (`clifford2native` emits literal 0.25 / -0.25 / 0.5, all exact dyadic floats),
# so this tolerance is a backstop for odd inputs, not load-bearing.
_QUARTER_TURN_TOL = 1e-9


def _check_distinct(gate: Gate) -> None:
    if len(set(gate.qubits)) != len(gate.qubits):
        raise EliminateRzError(
            f"{gate.kind.name} addresses a qubit more than once ({gate.qubits!r}); "
            "no phase frame is well defined for it. This is malformed IR."
        )


def _check_clifford(gate: Gate) -> None:
    quarters = gate.angle * 4.0
    if abs(quarters - round(quarters)) > _QUARTER_TURN_TOL:
        raise EliminateRzError(
            f"{gate.kind.name} angle {gate.angle!r} turns is not a multiple of a "
            "quarter turn. Commuting it would move an axis angle off the Clifford "
            "lattice, producing a transversal gate that is not a logical gate. "
            "Pass require_clifford_angles=False only for unencoded programs."
        )
```

Then replace the body of `sweep` with:

```python
    frame: dict[Hashable, float] = {}
    actions: list[Action] = []

    for gate in gates:
        _check_distinct(gate)

        if gate.kind is GateKind.RZ:
            if require_clifford_angles:
                _check_clifford(gate)
            for qubit in gate.qubits:
                frame[qubit] = frame.get(qubit, 0.0) + gate.angle
            actions.append(Action(delete=True))
        elif gate.kind is GateKind.R:
            if require_clifford_angles:
                _check_clifford(gate)
            actions.append(_rewrite_r(gate, frame))
        elif gate.kind is GateKind.INIT:
            _check_no_pending_frame(gate, frame)
            actions.append(Action())
        elif gate.kind is GateKind.MEASURE:
            for qubit in gate.qubits:
                frame.pop(qubit, None)
            actions.append(Action())
        else:
            actions.append(Action())

    return SweepResult(tuple(actions), frame)
```

And add:

```python
def _check_no_pending_frame(gate: Gate, frame: dict[Hashable, float]) -> None:
    """``Initialize`` must sit at the head of a wire.

    ``GeminiLogicalValidation`` forces non-Clifford gates to be the first gate on
    a qubit, so the frame is always zero here. Asserting rather than absorbing
    means a future mid-wire ``Initialize`` fails loudly instead of silently
    dropping a phase.
    """
    pending = {q: frame[q] for q in gate.qubits if frame.get(q, 0.0) != 0.0}
    if pending:
        raise EliminateRzError(
            f"Initialize reached with a pending phase frame {pending!r}. "
            "Initialize is expected at the head of a wire; a mid-wire one would "
            "need the frame absorbed into its (theta, phi, lam) instead."
        )
```

Finally, assert the partition invariant inside `_rewrite_r`, just before the
return:

```python
    covered = tuple(q for _, qubits in groups.items() for q in qubits)
    if sorted(map(repr, covered)) != sorted(map(repr, gate.qubits)):
        raise EliminateRzError(
            f"split of {gate.kind.name} did not partition its qubits: "
            f"{covered!r} vs {gate.qubits!r}"
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest python/tests/rewrite/test_eliminate_rz_core.py -v`
Expected: PASS (16 tests)

- [ ] **Step 5: Lint and commit**

```bash
uv run black python && uv run isort python && uv run ruff check python && uv run pyright python
git add python/bloqade/lanes/rewrite/eliminate_rz.py python/tests/rewrite/test_eliminate_rz_core.py
git commit -m "feat(rewrite): add splitting and preconditions to the Rz sweep"
```

---

### Task 3: Verify the core algebra numerically

Structural tests cannot catch a sign error in the commutation identity. This task
proves the sweep preserves the unitary up to the residual it reports.

**Files:**
- Test: `python/tests/rewrite/test_eliminate_rz_algebra.py` (create)

**Interfaces:**
- Consumes: `sweep`, `Gate`, `GateKind` from Task 1–2.
- Produces: nothing importable — tests only.

- [ ] **Step 1: Write the failing test**

Create `python/tests/rewrite/test_eliminate_rz_algebra.py`:

```python
"""Numerical check that the sweep preserves the unitary up to its residual.

The invariant, with angles in turns:

    U_before == Rz(residual) . U_after     (up to global phase)

A structural test cannot catch a sign error in the commutation identity, so this
builds both circuits as matrices and compares them.
"""

import numpy as np
import pytest

from bloqade.lanes.rewrite.eliminate_rz import Gate, GateKind, sweep

_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]])


def _r(axis: float, rotation: float) -> np.ndarray:
    """R(axis, rotation) with both angles in turns."""
    phi, theta = 2 * np.pi * axis, 2 * np.pi * rotation
    axis_op = np.cos(phi) * _X + np.sin(phi) * _Y
    return np.cos(theta / 2) * _I - 1j * np.sin(theta / 2) * axis_op


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
        bit_a = (state >> (num_qubits - 1 - a)) & 1
        bit_b = (state >> (num_qubits - 1 - b)) & 1
        if bit_a and bit_b:
            diagonal[state] = -1.0
    return np.diag(diagonal)


def _equal_up_to_phase(lhs: np.ndarray, rhs: np.ndarray) -> bool:
    index = np.unravel_index(np.argmax(np.abs(lhs)), lhs.shape)
    return np.allclose(lhs / lhs[index], rhs / rhs[index], atol=1e-9)


def _build(gates, rotations, num_qubits):
    """Multiply gates in circuit order. ``rotations[i]`` is gate i's rotation."""
    total = np.eye(2**num_qubits, dtype=complex)
    for gate, rotation in zip(gates, rotations):
        if gate.kind is GateKind.RZ:
            for qubit in gate.qubits:
                total = _embed(_rz(gate.angle), qubit, num_qubits) @ total
        elif gate.kind is GateKind.R:
            for qubit in gate.qubits:
                total = _embed(_r(gate.angle, rotation), qubit, num_qubits) @ total
        elif gate.kind is GateKind.DIAGONAL:
            total = _cz(num_qubits, *gate.qubits) @ total
    return total


def _rebuild(gates, rotations, actions):
    """Turn the sweep's actions back into a gate list, as the IR writer would.

    Zips all three lists together rather than looking gates up by value: ``Gate``
    is a frozen dataclass, so two identical gates compare equal and an index
    lookup would silently return the wrong one.
    """
    out_gates, out_rotations = [], []
    for gate, rotation, action in zip(gates, rotations, actions):
        if action.delete:
            continue
        if not action.groups:
            out_gates.append(gate)
            out_rotations.append(rotation)
            continue
        for group in action.groups:
            out_gates.append(Gate(gate.kind, group.qubits, group.axis_angle))
            out_rotations.append(rotation)
    return out_gates, out_rotations


@pytest.mark.parametrize(
    "gates, rotations",
    [
        # S then sqrt(X) on one qubit
        ([Gate(GateKind.RZ, (0,), 0.25), Gate(GateKind.R, (0,), 0.0)], [0.0, 0.25]),
        # the h/s/cx teleportation sequence on logical qubit 1
        (
            [
                Gate(GateKind.RZ, (1,), -0.25),
                Gate(GateKind.R, (1,), 0.0),
                Gate(GateKind.RZ, (1,), -0.25),
                Gate(GateKind.RZ, (1,), -0.25),
                Gate(GateKind.R, (1,), 0.25),
                Gate(GateKind.DIAGONAL, (0, 1)),
                Gate(GateKind.R, (1,), 0.25),
            ],
            [0.0, -0.25, 0.0, 0.0, -0.25, 0.0, 0.25],
        ),
        # frames diverge across a broadcast R, forcing a split
        (
            [
                Gate(GateKind.RZ, (0,), 0.25),
                Gate(GateKind.R, (0, 1), 0.0),
                Gate(GateKind.DIAGONAL, (0, 1)),
            ],
            [0.0, 0.25, 0.0],
        ),
    ],
)
def test_sweep_preserves_the_unitary_up_to_its_residual(gates, rotations):
    num_qubits = 2
    before = _build(gates, rotations, num_qubits)

    result = sweep(gates)

    after_gates, after_rotations = _rebuild(gates, rotations, result.actions)
    after = _build(after_gates, after_rotations, num_qubits)

    residual = np.eye(2**num_qubits, dtype=complex)
    for qubit, angle in result.residual.items():
        residual = _embed(_rz(angle), qubit, num_qubits) @ residual

    assert _equal_up_to_phase(before, residual @ after)


def test_a_wrong_sign_would_be_caught():
    """Guard the guard: flipping the frame's sign must break the invariant."""
    gates = [Gate(GateKind.RZ, (0,), 0.25), Gate(GateKind.R, (0,), 0.0)]
    rotations = [0.0, 0.25]
    before = _build(gates, rotations, 1)

    wrong = _build(
        [Gate(GateKind.R, (0,), 0.25)], [0.25], 1
    )  # axis + frame instead of - frame
    residual = _rz(0.25)

    assert not _equal_up_to_phase(before, residual @ wrong)
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest python/tests/rewrite/test_eliminate_rz_algebra.py -v`
Expected: PASS (4 tests). If `test_sweep_preserves_the_unitary_up_to_its_residual` fails, the sign in `_rewrite_r` is wrong — it must be `gate.angle - frame`, not `+`.

- [ ] **Step 3: Commit**

```bash
git add python/tests/rewrite/test_eliminate_rz_algebra.py
git commit -m "test: verify the Rz sweep preserves the unitary up to its residual"
```

---

### Task 4: Native-IR reader

**Files:**
- Modify: `python/bloqade/lanes/rewrite/eliminate_rz.py`
- Test: `python/tests/rewrite/test_eliminate_rz.py` (create)

**Interfaces:**
- Consumes: `Gate`, `GateKind`, `EliminateRzError`.
- Produces: `read_block(block: ir.Block) -> tuple[list[Gate], list[ir.Statement]]` — parallel lists, `gates[i]` describes `owners[i]`. Qubit keys are `ir.SSAValue` objects (identity-keyed: `ir.SSAValue.__hash__` returns `id(self)`).

- [ ] **Step 1: Write the failing tests**

Create `python/tests/rewrite/test_eliminate_rz.py`:

```python
"""IR-level tests for EliminateRz: reading native statements and writing back."""

import pytest
from kirin import ir, types as kirin_types
from kirin.dialects import ilist, py

from bloqade import qubit, squin, qubit as squin_qubit, types as bloqade_types
from bloqade.gemini import logical as gemini_logical
from bloqade.native.dialects import gate as native_gate

from bloqade.lanes.rewrite.eliminate_rz import (
    EliminateRzError,
    GateKind,
    read_block,
)


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


def test_reads_rz_and_r_into_gate_records():
    block = ir.Block()
    q0, q1 = _qubits(block, 2)
    reg = _register(block, [q0])
    angle = _const(block, 0.25)
    rz = native_gate.stmts.Rz(rotation_angle=angle, qubits=reg)
    block.stmts.append(rz)
    r = native_gate.stmts.R(axis_angle=angle, rotation_angle=angle, qubits=reg)
    block.stmts.append(r)

    gates, owners = read_block(block)

    assert [gate.kind for gate in gates] == [GateKind.RZ, GateKind.R]
    assert gates[0].qubits == (q0,)
    assert gates[0].angle == 0.25
    assert owners == [rz, r]


def test_reads_cz_qubits_as_controls_then_targets():
    block = ir.Block()
    q0, q1 = _qubits(block, 2)
    controls = _register(block, [q0])
    targets = _register(block, [q1])
    cz = native_gate.stmts.CZ(controls=controls, targets=targets)
    block.stmts.append(cz)

    gates, _ = read_block(block)

    assert gates[0].kind is GateKind.DIAGONAL
    assert gates[0].qubits == (q0, q1)


def test_non_ilist_register_raises():
    block = ir.Block()
    opaque = ir.TestValue(
        type=ilist.IListType[bloqade_types.QubitType, kirin_types.Any]
    )
    angle = _const(block, 0.25)
    block.stmts.append(
        native_gate.stmts.Rz(rotation_angle=angle, qubits=opaque)
    )

    with pytest.raises(EliminateRzError, match="ilist.New"):
        read_block(block)


def test_non_constant_angle_raises():
    block = ir.Block()
    (q0,) = _qubits(block, 1)
    reg = _register(block, [q0])
    angle = ir.TestValue(type=kirin_types.Float)
    block.stmts.append(native_gate.stmts.Rz(rotation_angle=angle, qubits=reg))

    with pytest.raises(EliminateRzError, match="constant"):
        read_block(block)


def test_statements_not_touching_qubits_are_skipped():
    block = ir.Block()
    (q0,) = _qubits(block, 1)
    _const(block, 0.25)  # a bare constant is not a gate

    gates, owners = read_block(block)

    assert gates == []
    assert owners == []


def test_statement_carrying_a_region_raises():
    """Un-unrolled control flow must not be silently swept past.

    This hook runs before scf2cf, so a surviving ``scf.For`` is a statement with
    a region inside one block -- not a second block. Its body closes over qubit
    values instead of taking them as arguments, so the qubit-reachability guard
    would miss it and the gates inside would vanish from the sweep.
    """
    from kirin.dialects import scf

    block = ir.Block()
    (q0,) = _qubits(block, 1)
    body = ir.Region(ir.Block())
    loop = scf.For(ir.TestValue(type=kirin_types.Any), body)
    block.stmts.append(loop)

    with pytest.raises(EliminateRzError, match="region"):
        read_block(block)


def test_qubit_reachability_drives_the_unknown_statement_guard():
    """The safety net: an unknown statement that touches a qubit must raise.

    Fabricating a foreign dialect statement here would be more machinery than
    the behaviour is worth, so this asserts the predicate that drives the guard:
    a statement consuming a qubit register is detected, a bare constant is not.
    """
    from bloqade.lanes.rewrite.eliminate_rz import _touches_qubit

    block = ir.Block()
    (q0,) = _qubits(block, 1)
    reg = _register(block, [q0])
    quarter = _const(block, 0.25)
    gate = native_gate.stmts.Rz(rotation_angle=quarter, qubits=reg)

    assert _touches_qubit(gate, {q0}) is True
    assert _touches_qubit(py.Constant(0.25), {q0}) is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest python/tests/rewrite/test_eliminate_rz.py -v`
Expected: FAIL — `ImportError: cannot import name 'read_block'`

- [ ] **Step 3: Write the implementation**

Add to `eliminate_rz.py` (imports at the top of the file, alongside the existing ones):

```python
from kirin import ir
from kirin.dialects import ilist, py

from bloqade import qubit as squin_qubit, types as bloqade_types
from bloqade.gemini.logical.dialects.operations import stmts as operations
from bloqade.native.dialects import gate as native_gate
```

Add `"read_block"` to `__all__`, and append:

```python
def _ilist_values(stmt: ir.Statement, register: ir.SSAValue) -> tuple[ir.SSAValue, ...]:
    """Destructure a qubit register into the individual qubit SSA values.

    Post-unroll the register is always a materialized ``ilist.New``; anything
    else means the IR is not in the shape this pass requires. Raising rather
    than skipping is deliberate -- ``circuit2place`` silently returns on a
    mismatch, but a skipped statement here leaves an ``Rz`` behind.
    """
    owner = register.owner
    if not isinstance(owner, ilist.New):
        raise EliminateRzError(
            f"{stmt.name}: qubit register comes from {type(owner).__name__}, not "
            "ilist.New. EliminateRz requires the post-unroll IR shape."
        )
    return tuple(owner.values)


def _const_float(stmt: ir.Statement, value: ir.SSAValue) -> float:
    owner = value.owner
    if not isinstance(owner, py.Constant):
        raise EliminateRzError(
            f"{stmt.name}: angle is not a compile-time constant "
            f"(owner is {type(owner).__name__})."
        )
    data = owner.value.unwrap()
    if not isinstance(data, (int, float)):
        raise EliminateRzError(f"{stmt.name}: angle constant is not numeric: {data!r}")
    return float(data)


def _touches_qubit(stmt: ir.Statement, qubit_values: set[ir.SSAValue]) -> bool:
    for arg in stmt.args:
        if arg in qubit_values:
            return True
        owner = arg.owner
        if isinstance(owner, ilist.New) and any(
            value in qubit_values for value in owner.values
        ):
            return True
    return False


def read_block(block: ir.Block) -> tuple[list[Gate], list[ir.Statement]]:
    """Reduce a native-dialect block to sweep records.

    Returns parallel lists: ``gates[i]`` describes ``owners[i]``. Statements that
    do not touch a qubit are skipped; an unrecognised statement that *does* touch
    one raises, because the sweep cannot know whether it is phase-neutral.
    """
    gates: list[Gate] = []
    owners: list[ir.Statement] = []
    qubit_values: set[ir.SSAValue] = set()

    for stmt in block.stmts:
        if stmt.regions:
            # This hook runs *before* scf2cf, so control flow that survived
            # unrolling is still an scf.For / scf.IfElse statement holding
            # regions inside one block -- not multiple blocks. Its body closes
            # over qubit values rather than taking them as arguments, so the
            # qubit-reachability guard below would not see them and the gates
            # inside would be silently skipped. None of the statements this
            # sweep handles carries a region, so rejecting all of them is exact.
            raise EliminateRzError(
                f"{stmt.name} carries a region; EliminateRz cannot see into it, "
                "and gates inside would be silently skipped. Control flow must "
                "be fully unrolled before this rule runs."
            )

        if len(stmt.results) == 1 and stmt.results[0].type.is_subseteq(
            bloqade_types.QubitType
        ):
            qubit_values.add(stmt.results[0])
            continue

        if isinstance(stmt, native_gate.stmts.Rz):
            gates.append(
                Gate(
                    GateKind.RZ,
                    _ilist_values(stmt, stmt.qubits),
                    _const_float(stmt, stmt.rotation_angle),
                )
            )
        elif isinstance(stmt, native_gate.stmts.R):
            gates.append(
                Gate(
                    GateKind.R,
                    _ilist_values(stmt, stmt.qubits),
                    _const_float(stmt, stmt.axis_angle),
                )
            )
        elif isinstance(stmt, native_gate.stmts.CZ):
            gates.append(
                Gate(
                    GateKind.DIAGONAL,
                    _ilist_values(stmt, stmt.controls)
                    + _ilist_values(stmt, stmt.targets),
                )
            )
        elif isinstance(stmt, operations.StarRz):
            gates.append(Gate(GateKind.DIAGONAL, _ilist_values(stmt, stmt.qubits)))
        elif isinstance(stmt, operations.Initialize):
            gates.append(Gate(GateKind.INIT, _ilist_values(stmt, stmt.qubits)))
        elif isinstance(stmt, operations.TerminalLogicalMeasurement):
            gates.append(Gate(GateKind.MEASURE, _ilist_values(stmt, stmt.qubits)))
        elif _touches_qubit(stmt, qubit_values):
            raise EliminateRzError(
                f"{stmt.name} addresses a qubit but EliminateRz does not know "
                "whether it is phase-neutral. Add it to read_block with the right "
                "GateKind, or keep it out of the logical pipeline."
            )
        else:
            continue

        owners.append(stmt)

    return gates, owners
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest python/tests/rewrite/test_eliminate_rz.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Lint and commit**

```bash
uv run black python && uv run isort python && uv run ruff check python && uv run pyright python
git add python/bloqade/lanes/rewrite/eliminate_rz.py python/tests/rewrite/test_eliminate_rz.py
git commit -m "feat(rewrite): read native gate statements into Rz sweep records"
```

---

### Task 5: Native-IR writer and the `EliminateRz` rule

**Files:**
- Modify: `python/bloqade/lanes/rewrite/eliminate_rz.py`
- Test: `python/tests/rewrite/test_eliminate_rz.py`

**Interfaces:**
- Consumes: `read_block`, `sweep`, `Action`, `Rewrite`.
- Produces: `apply_actions(block, owners, actions) -> bool`, and `EliminateRz(require_clifford_angles: bool = True)`, a `kirin.rewrite.abc.RewriteRule` implementing `rewrite_Block` (the sweep) and `rewrite_Region` (rejects multi-block regions).

- [ ] **Step 1: Write the failing tests**

Append to `python/tests/rewrite/test_eliminate_rz.py`:

```python
from bloqade.lanes.rewrite.eliminate_rz import apply_actions, sweep


def _stmts(block: ir.Block, kind) -> list[ir.Statement]:
    return [stmt for stmt in block.stmts if isinstance(stmt, kind)]


def _rewrite(block: ir.Block) -> None:
    """Read, sweep, and write back -- what EliminateRz does to one block."""
    gates, owners = read_block(block)
    apply_actions(block, owners, sweep(gates).actions)


def test_rz_statements_are_deleted_and_r_axis_is_shifted():
    block = ir.Block()
    (q0,) = _qubits(block, 1)
    reg = _register(block, [q0])
    quarter = _const(block, 0.25)
    zero = _const(block, 0.0)
    block.stmts.append(
        native_gate.stmts.Rz(rotation_angle=quarter, qubits=reg)
    )
    block.stmts.append(
        native_gate.stmts.R(axis_angle=zero, rotation_angle=quarter, qubits=reg)
    )

    _rewrite(block)

    assert _stmts(block, native_gate.stmts.Rz) == []
    remaining = _stmts(block, native_gate.stmts.R)
    assert len(remaining) == 1
    assert remaining[0].axis_angle.owner.value.unwrap() == 0.75


def test_equal_angles_share_one_constant_ssa_value():
    """FuseAdjacentGates matches on SSA identity, so equal angles must share."""
    block = ir.Block()
    q0, q1 = _qubits(block, 2)
    reg0, reg1 = _register(block, [q0]), _register(block, [q1])
    quarter = _const(block, 0.25)
    zero = _const(block, 0.0)
    both = _register(block, [q0, q1])
    block.stmts.append(native_gate.stmts.Rz(rotation_angle=quarter, qubits=both))
    block.stmts.append(
        native_gate.stmts.R(axis_angle=zero, rotation_angle=quarter, qubits=reg0)
    )
    block.stmts.append(
        native_gate.stmts.R(axis_angle=zero, rotation_angle=quarter, qubits=reg1)
    )

    _rewrite(block)

    rs = _stmts(block, native_gate.stmts.R)
    assert len(rs) == 2
    assert rs[0].axis_angle is rs[1].axis_angle


def test_split_emits_one_statement_and_register_per_frame():
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

    _rewrite(block)

    rs = _stmts(block, native_gate.stmts.R)
    assert len(rs) == 2
    covered = [v for r in rs for v in r.qubits.owner.values]
    assert sorted(map(id, covered)) == sorted(map(id, [q0, q1]))


def test_rule_rejects_a_multi_block_region():
    """Frame tracking across a branch is undefined, so this must raise."""
    from bloqade.lanes.rewrite.eliminate_rz import EliminateRz

    region = ir.Region(ir.Block())
    region.blocks.append(ir.Block())

    with pytest.raises(EliminateRzError, match="single block"):
        EliminateRz().rewrite_Region(region)


def test_rule_sweeps_a_block_end_to_end():
    """rewrite_Block is the entry point Walk drives."""
    from bloqade.lanes.rewrite.eliminate_rz import EliminateRz

    block = ir.Block()
    (q0,) = _qubits(block, 1)
    reg = _register(block, [q0])
    quarter = _const(block, 0.25)
    zero = _const(block, 0.0)
    block.stmts.append(native_gate.stmts.Rz(rotation_angle=quarter, qubits=reg))
    block.stmts.append(
        native_gate.stmts.R(axis_angle=zero, rotation_angle=quarter, qubits=reg)
    )

    result = EliminateRz().rewrite_Block(block)

    assert result.has_done_something
    assert _stmts(block, native_gate.stmts.Rz) == []


def test_rule_is_idempotent():
    """A second application finds no Rz and reports no change.

    Matters because the rule may be run under Fixpoint or chained with others.
    """
    from bloqade.lanes.rewrite.eliminate_rz import EliminateRz

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
Expected: FAIL — `ImportError: cannot import name 'apply_actions'`

- [ ] **Step 3: Write the implementation**

Add `"apply_actions"` and `"EliminateRz"` to `__all__`, add
`from kirin.rewrite import abc as rewrite_abc` to the imports, and append:

```python
class _ConstantCache:
    """Hands out one ``py.Constant`` SSA value per distinct float.

    ``FuseAdjacentGates`` (downstream, at the place layer) matches parameters by
    SSA *identity*, and ``circuit2place`` carries angle values through unchanged.
    Minting a fresh constant per statement would therefore break fusion between
    statements whose angles are numerically equal.
    """

    def __init__(self, block: ir.Block) -> None:
        self._block = block
        self._cache: dict[float, ir.SSAValue] = {}
        for stmt in block.stmts:
            if isinstance(stmt, py.Constant):
                data = stmt.value.unwrap()
                if isinstance(data, (int, float)) and not isinstance(data, bool):
                    self._cache.setdefault(_normalize(float(data)), stmt.result)

    def get(self, angle: float, before: ir.Statement) -> ir.SSAValue:
        key = _normalize(angle)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        const = py.Constant(key)
        const.insert_before(before)
        self._cache[key] = const.result
        return const.result


def apply_actions(
    block: ir.Block,
    owners: list[ir.Statement],
    actions: tuple[Action, ...],
) -> bool:
    """Write the sweep's decisions back into ``block``. Returns True if changed."""
    cache = _ConstantCache(block)
    changed = False

    for stmt, action in zip(owners, actions):
        if action.delete:
            stmt.delete()
            changed = True
            continue
        if not action.groups:
            continue

        rotation = stmt.rotation_angle
        for group in action.groups:
            register = ilist.New(
                values=tuple(group.qubits), elem_type=bloqade_types.QubitType
            )
            register.insert_before(stmt)
            replacement = native_gate.stmts.R(
                axis_angle=cache.get(group.axis_angle, stmt),
                rotation_angle=rotation,
                qubits=register.result,
            )
            replacement.insert_before(stmt)
        stmt.delete()
        changed = True

    return changed


@dataclass
class EliminateRz(rewrite_abc.RewriteRule):
    """Remove every ``Rz`` from a flat native-dialect block.

    A single rewrite rule rather than a pass: the convention in this codebase is
    that a transform is one rule, and passes exist to *combine* rules.
    ``FuseAdjacentGates`` is the local precedent -- also a stateful whole-block
    sweep expressed as a rule.

    Raises on any precondition violation rather than returning a no-op result.
    Leaving an ``Rz`` behind produces IR the backend cannot execute, so failing
    loudly is the feature.
    """

    require_clifford_angles: bool = True

    def rewrite_Region(self, node: ir.Region) -> rewrite_abc.RewriteResult:
        """Reject control flow that survived unrolling.

        Frame tracking has no meaning across a branch: a qubit's frame entering
        a block would depend on which predecessor ran.
        """
        if len(node.blocks) != 1:
            raise EliminateRzError(
                f"EliminateRz requires a single block per region, found "
                f"{len(node.blocks)}. Run AggressiveUnroll first."
            )
        return rewrite_abc.RewriteResult()

    def rewrite_Block(self, node: ir.Block) -> rewrite_abc.RewriteResult:
        gates, owners = read_block(node)
        result = sweep(gates, require_clifford_angles=self.require_clifford_angles)
        changed = apply_actions(node, owners, result.actions)
        return rewrite_abc.RewriteResult(has_done_something=changed)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest python/tests/rewrite/test_eliminate_rz.py -v`
Expected: PASS (13 tests)

- [ ] **Step 5: Run the whole rewrite suite for regressions**

Run: `uv run pytest python/tests/rewrite -q`
Expected: PASS, no new failures.

- [ ] **Step 6: Lint and commit**

```bash
uv run black python && uv run isort python && uv run ruff check python && uv run pyright python
git add python/bloqade/lanes/rewrite/eliminate_rz.py python/tests/rewrite/test_eliminate_rz.py
git commit -m "feat(rewrite): add the EliminateRz rule and its native-IR writer"
```

---

### Task 6: Wire it into the logical pipeline via a fifth hook

**Files:**
- Modify: `python/bloqade/lanes/transform/native_to_place.py` (docstring at lines 41–74; `emit` around line 114; `LogicalNativeToPlace` at line 181)
- Test: `python/tests/gemini/test_eliminate_rz_pipeline.py` (create)

**Interfaces:**
- Consumes: `EliminateRz`.
- Produces: `NativeToPlaceBase._post_unroll_rules(self) -> list[RewriteRule]`, default `[]`; overridden in `LogicalNativeToPlace` to return `[EliminateRz()]`.

- [ ] **Step 1: Write the failing tests**

Create `python/tests/gemini/test_eliminate_rz_pipeline.py`:

```python
"""End-to-end: the logical pipeline must emit no local_rz from Clifford gates."""

from bloqade import qubit, squin
from bloqade.gemini import logical as gemini_logical

from bloqade.lanes.arch.gemini.logical import get_arch_spec as get_logical_spec
from bloqade.lanes.arch.gemini.physical import get_arch_spec as get_physical_spec
from bloqade.lanes.dialects import move
from bloqade.lanes.transform import LogicalPipeline, PhysicalPipeline


def _compile_logical(kernel):
    return LogicalPipeline(
        get_logical_spec(), transversal_rewrite=True, simulation=False
    ).emit(kernel)


def test_teleportation_kernel_emits_no_local_rz():
    @gemini_logical.kernel(aggressive_unroll=True)
    def rz_half_pi_teleport():
        reg = qubit.qalloc(2)
        squin.h(reg[1])
        squin.s(reg[1])
        squin.cx(reg[0], reg[1])
        gemini_logical.terminal_measure(reg)

    out = _compile_logical(rz_half_pi_teleport)

    local_rz = [
        stmt
        for stmt in out.callable_region.walk()
        if isinstance(stmt, move.LocalRz)
    ]
    assert local_rz == []


def test_local_r_count_is_unchanged():
    """Rz removal must not silently drop or duplicate rotation pulses."""

    @gemini_logical.kernel(aggressive_unroll=True)
    def kernel():
        reg = qubit.qalloc(2)
        squin.h(reg[1])
        squin.s(reg[1])
        squin.cx(reg[0], reg[1])
        gemini_logical.terminal_measure(reg)

    out = _compile_logical(kernel)
    local_r = [
        stmt for stmt in out.callable_region.walk() if isinstance(stmt, move.LocalR)
    ]
    assert len(local_r) == 3


def test_kernel_with_the_terminal_measure_removed_still_drops_rz():
    """The second sink shape: no measurement statement to discard the frame at.

    ``RemovePostProcessing(delete_terminal_measure=True)`` deletes the
    measurement after validation has run, so the residual frame is discarded at
    end-of-block instead. Nothing can observe a trailing diagonal in a program
    with no measurement.
    """
    from bloqade.gemini.logical import default_post_processing
    from bloqade.gemini.logical.rewrite.remove_postprocessing import (
        RemovePostProcessing,
    )

    @gemini_logical.kernel(aggressive_unroll=True)
    def kernel():
        reg = qubit.qalloc(2)
        squin.h(reg[1])
        squin.s(reg[1])
        squin.cx(reg[0], reg[1])
        default_post_processing(reg)

    stripped = kernel.similar()
    RemovePostProcessing(kernel.dialects, delete_terminal_measure=True)(stripped)

    out = _compile_logical(stripped)

    local_rz = [
        stmt for stmt in out.callable_region.walk() if isinstance(stmt, move.LocalRz)
    ]
    assert local_rz == []


def test_equal_frames_still_fuse_after_lowering_to_place():
    """Constant sharing must survive the native->place boundary.

    ``FuseAdjacentGates`` matches axis angles by SSA identity, and
    ``circuit2place`` carries the angle values through unchanged. If the rule
    minted a fresh constant per statement, these two identical gates would stop
    fusing even though their angles are numerically equal. ``ASAPPlacePass`` is
    used because it is the pipeline option that actually runs fusion.
    """
    from bloqade.lanes.passes import ASAPPlacePass

    @gemini_logical.kernel(aggressive_unroll=True)
    def kernel():
        reg = qubit.qalloc(2)
        squin.s(reg[0])
        squin.s(reg[1])
        squin.sqrt_x(reg[0])
        squin.sqrt_x(reg[1])
        gemini_logical.terminal_measure(reg)

    out = LogicalPipeline(
        get_logical_spec(),
        transversal_rewrite=True,
        simulation=False,
        place_opt_type=ASAPPlacePass,
    ).emit(kernel)

    local_r = [
        stmt for stmt in out.callable_region.walk() if isinstance(stmt, move.LocalR)
    ]
    # Both logical qubits carry the same frame, so the two sqrt(X) gates get the
    # same rewritten axis angle and must still fuse into one statement.
    assert len(local_r) == 1


def test_star_rz_payload_survives():
    """StarRz is a user-requested gadget, not a compiler artifact."""
    import math

    @gemini_logical.kernel(aggressive_unroll=True, verify=False)
    def kernel():
        reg = qubit.qalloc(1)
        gemini_logical.star_rz(math.pi / 16, reg[0])
        gemini_logical.terminal_measure(reg)

    out = _compile_logical(kernel)
    local_rz = [
        stmt for stmt in out.callable_region.walk() if isinstance(stmt, move.LocalRz)
    ]
    assert len(local_rz) == 1


def test_physical_pipeline_is_unchanged():
    """PhysicalNativeToPlace inherits the no-op hook, so nothing moves."""

    @gemini_logical.kernel(aggressive_unroll=True)
    def _unused():
        reg = qubit.qalloc(1)
        squin.h(reg[0])
        gemini_logical.terminal_measure(reg)

    # A physical compile still contains its Rz statements.
    from bloqade.gemini import physical as gemini_physical

    @gemini_physical.kernel(aggressive_unroll=True)
    def physical_kernel():
        reg = qubit.qalloc(1)
        squin.s(reg[0])
        squin.measure(reg)

    out = PhysicalPipeline(get_physical_spec()).emit(physical_kernel)
    rz = [
        stmt
        for stmt in out.callable_region.walk()
        if isinstance(stmt, (move.LocalRz, move.GlobalRz))
    ]
    assert rz, "physical compiles must still emit Rz -- the hook is a no-op there"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest python/tests/gemini/test_eliminate_rz_pipeline.py -v`
Expected: FAIL — `test_teleportation_kernel_emits_no_local_rz` finds 3 `LocalRz`.

Note: if `test_physical_pipeline_is_unchanged` or `test_star_rz_payload_survives`
fail at this point for unrelated reasons (kernel spelling, validation), fix the
*test* to match the existing API before touching the implementation — they are
baselines, not targets.

- [ ] **Step 3: Add the hook to the template**

In `python/bloqade/lanes/transform/native_to_place.py`, add the method to
`NativeToPlaceBase` next to the other hooks:

```python
    def _post_unroll_rules(self) -> list[RewriteRule]:
        """Rules applied to the flat native IR, after unrolling.

        This is the only window where the program is a flat block of
        ``native.gate`` statements: ``AggressiveUnroll`` has run, and
        ``RewritePlaceOperations`` has not. Default is no rules.
        """
        return []
```

This mirrors `_squin_clifford_rules` deliberately — same shape, same
`rewrite.Chain` treatment — so the class has one way of expressing "rules to run
at stage X" rather than two.

Call it in `emit`, immediately after the unroll, using the same `Walk` idiom the
neighbouring `scf2cf` line already uses:

```python
        AggressiveUnroll(out.dialects, no_raise=no_raise).fixpoint(out)

        if post_unroll_rules := self._post_unroll_rules():
            rewrite.Walk(rewrite.Chain(*post_unroll_rules)).rewrite(out.code)

        self._post_unroll_validation(out, no_raise)
```

Update the class docstring's hook list from four to five, adding an entry for
`_post_unroll_rules` in the same style as the others.

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

`RewriteRule` is already imported there (`from kirin.rewrite.abc import
RewriteRule`), as is `rewrite` — `_squin_clifford_rules` uses both.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest python/tests/gemini/test_eliminate_rz_pipeline.py -v`
Expected: PASS (6 tests)

If `test_equal_frames_still_fuse_after_lowering_to_place` fails with 2 statements
instead of 1, the constant cache in Task 5 is minting a fresh `py.Constant` per
statement instead of reusing one per distinct value. Fix the cache — do not
relax the assertion.

- [ ] **Step 6: Run the full Python suite**

Run: `uv run pytest python/tests -q -x`
Expected: PASS. Failures in `python/tests/gemini/` or `python/tests/rewrite/` that
assert on `LocalRz` counts are *expected* to change — read each one and update the
assertion only if the new count is correct for this design. Do not weaken an
assertion to make it pass.

- [ ] **Step 7: Lint and commit**

```bash
uv run black python && uv run isort python && uv run ruff check python && uv run pyright python
git add python/bloqade/lanes/transform/native_to_place.py python/tests/gemini/test_eliminate_rz_pipeline.py
git commit -m "feat(lanes): run EliminateRz in the logical pipeline via a post-unroll hook"
```

---

### Task 7: Regenerate benchmark baselines

The pass is on by default, so the deterministic benchmark metrics move. `AGENT.md`
makes regenerating them a rule for any such change.

**Files:**
- Modify: `python/benchmarks/harness/latest_logical.csv`
- Possibly modify: `python/benchmarks/harness/latest_physical.csv` (expected: unchanged)

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

Check, and stop if any of these do not hold:
- The `success` column is unchanged — no new failures.
- `move_count_events` / `move_count_lanes` shifts are explained by removed `Rz`
  pulses and any `R` splits.
- `estimated_fidelity` moves in the direction the pulse-count change implies.
- Ignore `wall_time_ms`; it is not part of the comparison and varies by machine.

- [ ] **Step 4: Confirm the physical suite did not move**

Run: `just benchmark-physical`
Expected: exit 0, no diff. If it moved, the hook is not a no-op for physical
compiles — stop and fix Task 6 rather than committing the new baseline.

- [ ] **Step 5: Confirm determinism**

Run: `just benchmark-logical` again, then `git diff python/benchmarks/harness/latest_logical.csv`
Expected: no further change. The deterministic columns must be identical run to run.

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

- **Angles are turns.** `0.25` is 90°. Every angle in this plan and in the IR is
  in turns; only the test simulator converts to radians.
- **Do not add a flag to disable the rule** in the logical pipeline. The hard
  invariant is the feature; an off switch produces IR the backend cannot run.
  `require_clifford_angles` is a different thing — it exists for a future
  unencoded (`PhysicalPipeline`) use, where there is no code and no
  transversality constraint.
- **If a precondition fires on a real kernel**, that is a finding, not a nuisance.
  Report it rather than loosening the check; it means the IR has a shape the
  design did not anticipate.
- **The `ilist.New` shape invariant is unenforced elsewhere** — it holds only
  because `AggressiveUnroll` ran. If Task 4's shape check fires, do not fall back
  to skipping the statement.
