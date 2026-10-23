"""MergeSingleQubitRuns: fold each qubit's run of 1-qubit gates into one rotation.

``DecomposeCliffordToNative`` lowers every source Clifford on its own, so a
source ``H . CZ . H`` written twice -- ``CX`` twice -- reaches the native window
with an ``H . H`` between the two ``CZ`` that nothing ever cancels.
``EliminateRz`` makes the ``Rz`` free but cannot touch the ``R`` pulses: the
identity pair survives as ``R(+Y) . R(-Y)``. This rule is the missing other
half (#1117).

Scans a flat block in order. Per qubit it collects a maximal *run* of ``R`` and
``Rz`` with constant angles, which is any stretch not interrupted by another
statement touching that qubit. A run is multiplied out as an SU(2) matrix and
re-emitted as at most::

    R(phi, theta) then Rz(lam)          (time order)

using the exact identity (angles in turns, ``R`` about the axis at ``phi`` in
the XY plane)::

    Rz(a) . Rx(theta) . Rz(b) = Rz(a + b) . R(-b, theta)

A product that is diagonal becomes a lone ``Rz``, and one equal to the identity
up to phase becomes nothing. The trailing ``Rz`` is not a cost: this rule runs
before ``EliminateRz``, which commutes it forward like any other. For the same
reason a run is only rewritten when that leaves its qubit with fewer ``R``
pulses -- ``R . Rz`` already is ``R`` then ``Rz``, and rebuilding it would only
churn the IR.

**Where the result goes.** A run's replacement is emitted at its *last* gate.
Nothing else touches the qubit between the first and last gate of the run, so
the position is exact; choosing the last one keeps the emitted gates in the
layer they ended in.

**Broadcasts are never split.** A broadcast gate is one parallel pulse, and
splitting it to merge part of it can cost more pulses than the merge saves.
So qubits are merged in groups: qubits whose runs consist of exactly the same
gates merge together, as one broadcast statement, and a group is merged only
if every one of its gates addresses exactly the group. A layer of ``H`` over a
register followed by another over the same register merges; ``R(a, b)``
followed by ``R(a)`` is left alone. At this level a qubit is a logical qubit,
so a gate on one of them is already a transversal pulse over its block, and
merging per logical qubit keeps it one.

**Constants only.** A gate whose angle the compiler cannot read -- a kernel
argument, a ``py.Sub`` -- has no matrix, so it ends the run on its qubits and
is left untouched.

**Anything unrecognised is a barrier on every qubit.** Ending every run is
exact whatever the statement does: it costs merges, never correctness. Only
statements known to touch nothing are skipped, and only statements whose
qubits can be read off the IR end just their own qubits' runs.

Exact up to a global phase, which is all a Clifford validation or a Tableau
simulator can see. A product of Cliffords is Clifford, so the logical
validations that ran before this rule still hold for what it emits; it runs
after ``RewriteSteaneTransversalCliffordAdjoints``, so it merges the angles
that are actually executed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from bloqade.decoders.dialects.annotate import stmts as annotate
from bloqade.native.dialects import gate as native_gate
from kirin import ir
from kirin.dialects import func, ilist, py
from kirin.dialects.py import tuple as py_tuple
from kirin.dialects.py.binop import stmts as py_binop
from kirin.rewrite.abc import RewriteResult, RewriteRule

from bloqade import qubit as squin_qubit, types as bloqade_types
from bloqade.gemini.common.dialects import qubit as gemini_qubit
from bloqade.gemini.logical.dialects.extensions import stmts as extensions
from bloqade.gemini.logical.dialects.operations import stmts as operations
from bloqade.lanes.utils import constant_float

__all__ = ["MergeSingleQubitRuns", "decompose", "gate_matrix"]

_TOL = 1e-9
"""Below this a matrix entry is zero and an angle sits on its grid point."""

_GRID = 4096
"""Angles within ``_TOL`` of a multiple of ``1 / _GRID`` turns snap onto it.

The matrix arithmetic leaves Clifford angles as ``0.24999999999999997`` and the
like. Snapping restores the literal, so ``CommonSubexpressionElimination`` can
share it with an equal angle elsewhere -- ``FuseAdjacentGates`` compares angles
by SSA identity -- and a reader sees the angle the circuit means.
"""

_INERT = (
    py.Constant,
    py.GetItem,
    py_binop.BinOp,
    py_tuple.New,
    ilist.New,
    squin_qubit.stmts.New,
    gemini_qubit.stmts.NewAt,
    func.ConstantNone,
    func.Function,
    annotate.SetDetector,
    annotate.SetObservable,
)
"""Statements that touch no qubit's state, so they end no run.

``ilist.New`` does reference qubits, but only to name a register; the gate
that consumes it is what acts on them.
"""

_TOUCHING = (
    native_gate.stmts.R,
    native_gate.stmts.Rz,
    native_gate.stmts.CZ,
    extensions.StarRz,
    operations.Initialize,
    operations.TerminalLogicalMeasurement,
    squin_qubit.stmts.Measure,
)
"""Statements whose qubits can be read off their register operands.

They end the runs of exactly those qubits. ``R`` and ``Rz`` are here for the
non-constant case; a constant one extends a run instead.
"""

_Spec = tuple[tuple[str, tuple[float, ...]], ...]
"""A merged run as emitted, in time order: ``(("R", (phi, theta)), ("Rz", (lam,)))``."""


# -- the matrix algebra --------------------------------------------------------


def gate_matrix(stmt: native_gate.stmts.R | native_gate.stmts.Rz) -> np.ndarray | None:
    """The SU(2) matrix of a constant-angle native gate, or None if not constant.

    Angles are in turns. ``Rz(t) = exp(-i pi t Z)`` and ``R(phi, theta) =
    exp(-i pi theta (cos(2 pi phi) X + sin(2 pi phi) Y))``, the conventions under
    which ``EliminateRz``'s ``R(phi, theta) . Rz(a) = Rz(a) . R(phi - a, theta)``
    holds.
    """
    if isinstance(stmt, native_gate.stmts.Rz):
        if (t := constant_float(stmt.rotation_angle)) is None:
            return None
        phase = np.exp(-1j * math.pi * t)
        return np.array([[phase, 0], [0, phase.conjugate()]])

    phi = constant_float(stmt.axis_angle)
    theta = constant_float(stmt.rotation_angle)
    if phi is None or theta is None:
        return None
    c, s = math.cos(math.pi * theta), math.sin(math.pi * theta)
    axis = np.exp(1j * math.tau * phi)
    return np.array([[c, -1j * s * axis.conjugate()], [-1j * s * axis, c]])


def _snap(turns: float, period: float = 1.0) -> float:
    """Reduce mod ``period`` and pull float noise back onto the angle grid."""
    snapped = round(turns * _GRID) / _GRID
    if abs(snapped - turns) < _TOL:
        turns = snapped
    turns %= period
    return 0.0 if abs(turns - period) < _TOL else turns


def decompose(unitary: np.ndarray) -> _Spec:
    """Write a 2x2 unitary as ``R(phi, theta)`` then ``Rz(lam)``, up to phase.

    Returns the gates in time order, dropping whichever are the identity: an
    empty tuple for the identity, a lone ``Rz`` for a diagonal matrix.
    ``theta`` lies in ``[0, 0.5]`` and ``phi`` and ``lam`` in ``[0, 1)``.
    """
    # Rz(lam) . R(phi, theta) has, up to a global phase g,
    #   u00 = g e^{-i pi lam} c          u11 = g e^{i pi lam} c
    #   u10 = -i g e^{i pi lam} e^{2 pi i phi} s
    #   u01 = -i g e^{-i pi lam} e^{-2 pi i phi} s
    # with c = cos(pi theta) and s = sin(pi theta).
    (u00, u01), (u10, u11) = unitary
    c, s = abs(u00), abs(u10)

    if s < _TOL:
        lam = _snap(np.angle(u11 / u00) / math.tau)
        return () if lam == 0.0 else (("Rz", (lam,)),)

    theta = _snap(math.atan2(s, c) / math.pi)
    if c < _TOL:
        # Only lam + 2 phi is defined here; put all of it in the axis.
        lam = 0.0
        phi = _snap(np.angle(u10 / u01) / (2 * math.tau))
    else:
        lam = _snap(np.angle(u11 / u00) / math.tau)
        phi = _snap(np.angle(1j * u10 / u00 * np.exp(-1j * math.tau * lam)) / math.tau)

    spec: _Spec = (("R", (phi, theta)),)
    return spec if lam == 0.0 else spec + (("Rz", (lam,)),)


# -- the rewrite -----------------------------------------------------------------


@dataclass
class _Run:
    gates: list[native_gate.stmts.R | native_gate.stmts.Rz] = field(
        default_factory=list
    )
    unitary: np.ndarray = field(default_factory=lambda: np.eye(2, dtype=complex))


def _register(value: ir.SSAValue) -> tuple[ir.SSAValue, ...] | None:
    """The qubits an ``ilist.New`` register names, or None for any other shape."""
    if isinstance(value, ir.ResultValue) and isinstance(value.owner, ilist.New):
        return tuple(value.owner.values)
    return None


def _touched(stmt: ir.Statement) -> tuple[ir.SSAValue, ...] | None:
    """The qubits ``stmt`` acts on, or None when that cannot be read off the IR."""
    if isinstance(stmt, native_gate.stmts.CZ):
        controls, targets = _register(stmt.controls), _register(stmt.targets)
        if controls is None or targets is None:
            return None
        return controls + targets
    return _register(stmt.qubits)  # type: ignore[attr-defined]


def _runs(block: ir.Block) -> list[tuple[ir.SSAValue, _Run]]:
    """Every maximal run in ``block``, keyed by its qubit, in the order they end."""
    open_runs: dict[ir.SSAValue, _Run] = {}
    closed: list[tuple[ir.SSAValue, _Run]] = []

    def close(qubits) -> None:
        for qubit in qubits:
            if (run := open_runs.pop(qubit, None)) is not None:
                closed.append((qubit, run))

    for stmt in block.stmts:
        if isinstance(stmt, (native_gate.stmts.R, native_gate.stmts.Rz)):
            qubits = _register(stmt.qubits)
            matrix = gate_matrix(stmt)
            if qubits is not None and matrix is not None:
                for qubit in qubits:
                    run = open_runs.setdefault(qubit, _Run())
                    run.gates.append(stmt)
                    run.unitary = matrix @ run.unitary
                continue
        if isinstance(stmt, _INERT):
            continue
        touched = _touched(stmt) if isinstance(stmt, _TOUCHING) else None
        close(list(open_runs) if touched is None else touched)
    close(list(open_runs))
    return closed


def _whole_gate_groups(
    runs: list[tuple[ir.SSAValue, _Run]],
) -> list[tuple[tuple[ir.SSAValue, ...], _Run]]:
    """Group qubits whose runs are the same gates, keeping only whole-gate groups.

    Qubits sharing a run share every gate in it, so they merge as one and stay
    one broadcast statement. A group is kept only when each of its gates
    addresses exactly the group's qubits: a gate that also reaches a qubit
    outside the group would have to be split to merge, and this rule never
    splits a broadcast -- see "Broadcasts are never split".
    """
    groups: dict[tuple[int, ...], tuple[list[ir.SSAValue], _Run]] = {}
    for qubit, run in runs:
        key = tuple(id(gate) for gate in run.gates)
        groups.setdefault(key, ([], run))[0].append(qubit)

    whole = []
    for qubits, run in groups.values():
        members = set(qubits)
        if all(set(_register(gate.qubits) or ()) == members for gate in run.gates):
            whole.append((tuple(qubits), run))
    return whole


def _worth_merging(run: _Run) -> bool:
    """Whether merging leaves this qubit with fewer ``R`` pulses.

    ``Rz`` does not count: ``EliminateRz`` removes it either way. So a run such
    as ``R . Rz``, which merges to ``R`` then ``Rz`` again, is left alone rather
    than rebuilt, since rebuilding it would gain nothing.
    """
    pulses = sum(isinstance(gate, native_gate.stmts.R) for gate in run.gates)
    merged = sum(name == "R" for name, _ in decompose(run.unitary))
    return merged < pulses


@dataclass
class MergeSingleQubitRuns(RewriteRule):
    """Merge each qubit's run of constant ``R``/``Rz`` into at most ``R`` + ``Rz``.

    **Preconditions**, the same two ``EliminateRz`` has and validated by
    ``NativeToPlaceBase.emit`` immediately before the post-unroll rules: a flat
    single-block region (``lanes.flat_block.validation``), so block order is
    execution order; and every gate register an ``ilist.New`` of distinct
    qubits (``lanes.qubit_register.validation``). A register of any other shape
    is still handled safely -- as a barrier on every qubit -- but nothing is
    merged across it.

    Acts on a whole ``ir.Block`` at once, so wrap it in ``rewrite.Walk``: the
    walk offers the block before its statements, and this rule ignores
    everything that is not a block.
    """

    def rewrite_Block(self, node: ir.Block) -> RewriteResult:
        changed = False
        for qubits, run in _whole_gate_groups(_runs(node)):
            if not _worth_merging(run):
                continue
            # At the run's last gate; see "Where the result goes".
            last = run.gates[-1]
            for name, angles in decompose(run.unitary):
                self._emit(name, angles, qubits, last)
            for gate in run.gates:
                gate.delete()
            changed = True
        return RewriteResult(has_done_something=changed)

    # -- emitting ----------------------------------------------------------

    def _emit(
        self,
        name: str,
        angles: tuple[float, ...],
        qubits: tuple[ir.SSAValue, ...],
        before: ir.Statement,
    ) -> None:
        constants = [py.Constant(angle) for angle in angles]
        for constant in constants:
            constant.insert_before(before)
        register = ilist.New(values=qubits, elem_type=bloqade_types.QubitType)
        register.insert_before(before)
        if name == "R":
            phi, theta = constants
            stmt: ir.Statement = native_gate.stmts.R(
                phi.result, theta.result, register.result
            )
        else:
            (lam,) = constants
            stmt = native_gate.stmts.Rz(lam.result, register.result)
        stmt.insert_before(before)
