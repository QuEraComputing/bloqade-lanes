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

from bloqade.native.dialects import gate as native_gate
from kirin import ir
from kirin.dialects import func, ilist, py
from kirin.rewrite.abc import RewriteResult, RewriteRule

from bloqade import types as bloqade_types

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
            register = ilist.New(values=tuple(group), elem_type=bloqade_types.QubitType)
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
