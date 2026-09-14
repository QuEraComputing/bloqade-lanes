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

The frame holds **SSA values, not floats**: a phase is whatever
``rotation_angle`` operand the ``Rz`` carried, and a shifted axis is emitted as
``py.Sub`` over those operands. That keeps the rule working on kernels whose
angles are function arguments rather than literals -- the pipeline already
carries non-constant angles end to end (``RewriteStarRz`` feeds a
``func.Invoke`` result straight into ``move.LocalRz``). When both operands do
happen to be constants the arithmetic is folded to a literal, since nothing runs
constant folding after this rule and an unfolded ``py.Sub`` would otherwise
propagate all the way into the emitted program. Duplicate angle values are left
for the ``CommonSubexpressionElimination`` already in ``NativeToPlaceBase.emit``
to merge -- both ``py.Constant`` and ``py.Sub`` are ``Pure``, so it does, and a
memo here would only re-implement it while growing with circuit depth.

Dispatch is exhaustive: every statement type reachable in this window has a
registered handler, and the default raises. There is deliberately no "does this
statement touch a qubit?" heuristic -- an unrecognised statement is a statement
whose phase behaviour we cannot know, so it is an error rather than a guess.

See ``docs/superpowers/specs/2026-09-14-rz-elimination-design.md``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from functools import singledispatchmethod

from bloqade.decoders.dialects.annotate import stmts as annotate
from bloqade.native.dialects import gate as native_gate
from kirin import ir
from kirin.dialects import func, ilist, py
from kirin.dialects.py import tuple as py_tuple
from kirin.dialects.py.binop import stmts as py_binop
from kirin.rewrite.abc import RewriteResult, RewriteRule

from bloqade import qubit as squin_qubit, types as bloqade_types
from bloqade.gemini.common.dialects import qubit as gemini_qubit
from bloqade.gemini.logical.dialects.operations import stmts as operations

__all__ = ["EliminateRz", "EliminateRzError"]


class EliminateRzError(Exception):
    """The IR violates a precondition of the Rz elimination scan."""


def _literal(value: ir.SSAValue) -> float | None:
    """The numeric literal behind ``value``, or ``None`` if it is not a constant."""
    if not isinstance(value, ir.ResultValue) or not isinstance(
        value.owner, py.Constant
    ):
        return None
    data = value.owner.value.unwrap()
    if isinstance(data, bool) or not isinstance(data, (int, float)):
        return None
    return float(data)


@dataclass
class EliminateRz(RewriteRule):
    """Remove every ``Rz`` from a flat native-dialect program."""

    _frame: dict[ir.SSAValue, ir.SSAValue] = field(default_factory=dict, init=False)
    """Pending Z phase per qubit, as the SSA value holding it. Spans the program.

    A qubit is absent until an ``Rz`` touches it, so "no pending phase" and "not
    in this dict" are the same statement.
    """

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
        return RewriteResult()

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        return self._rewrite(node)

    # -- helpers ----------------------------------------------------------

    def _qubit_values(
        self, stmt: ir.Statement, register: ir.SSAValue
    ) -> tuple[ir.SSAValue, ...]:
        if not isinstance(register, ir.ResultValue) or not isinstance(
            register.owner, ilist.New
        ):
            raise EliminateRzError(
                f"{stmt.name}: qubit register does not come from ilist.New. "
                "EliminateRz requires the post-unroll IR shape."
            )
        values = tuple(register.owner.values)
        if len(set(values)) != len(values):
            raise EliminateRzError(
                f"{stmt.name} addresses a qubit more than once; no phase frame "
                "is well defined for it. This is malformed IR."
            )
        return values

    def _combine(
        self,
        op: type[py.Add | py.Sub],
        lhs: ir.SSAValue,
        rhs: ir.SSAValue,
        before: ir.Statement,
    ) -> ir.SSAValue:
        """``lhs op rhs`` as an SSA value, folded to a literal when both are."""
        left, right = _literal(lhs), _literal(rhs)
        if left is not None and right is not None:
            folded = left + right if op is py.Add else left - right
            stmt: ir.Statement = py.Constant(folded)
        else:
            stmt = op(lhs, rhs)
        stmt.insert_before(before)
        return stmt.results[0]

    # -- per-statement dispatch -------------------------------------------
    #
    # Exhaustive by design: the default raises, and every statement type that
    # can appear in this window is registered below. An unregistered type is a
    # type whose phase behaviour is unknown, and guessing is how an Rz gets
    # silently left behind.

    @singledispatchmethod
    def _rewrite(self, stmt: ir.Statement) -> RewriteResult:
        raise EliminateRzError(
            f"{stmt.name} has no EliminateRz handler, so its effect on a phase "
            "frame is unknown. Register it explicitly -- as a no-op if it is "
            "phase-neutral -- rather than letting it pass silently."
        )

    # --- inert: carry no phase, touch no frame ---

    @_rewrite.register(py.Constant)
    @_rewrite.register(py.GetItem)
    @_rewrite.register(py_binop.BinOp)  # Add / Sub / Mult / Div / ...
    @_rewrite.register(py_tuple.New)
    @_rewrite.register(ilist.New)
    @_rewrite.register(squin_qubit.stmts.New)
    @_rewrite.register(gemini_qubit.stmts.NewAt)
    @_rewrite.register(func.Return)
    @_rewrite.register(func.ConstantNone)
    @_rewrite.register(annotate.SetDetector)
    @_rewrite.register(annotate.SetObservable)
    def _(self, stmt: ir.Statement) -> RewriteResult:
        return RewriteResult()

    @_rewrite.register(func.Function)
    def _(self, stmt: func.Function) -> RewriteResult:
        # The walk root is the program's own definition and is expected.
        #
        # A *nested* func.Function is not: Walk enqueues a statement's regions
        # into the same worklist as everything else, so a nested function's
        # block is scanned in line with the outer one, and rewrite_Region would
        # reset self._frame mid-walk -- discarding every phase the outer block
        # had accumulated, with no error raised.
        if stmt.parent_stmt is not None:
            raise EliminateRzError(
                f"{stmt.name} is a nested func.Function; EliminateRz only "
                "accepts the walk root. A nested function's region would reset "
                "the phase frame mid-walk and silently discard every pending "
                "phase from the enclosing block."
            )
        return RewriteResult()

    # --- diagonal: commute with the frame exactly, so nothing to do ---

    @_rewrite.register(native_gate.stmts.CZ)
    def _(self, stmt: native_gate.stmts.CZ) -> RewriteResult:
        # Diagonal on each qubit independently, so the two sides' frames need
        # not agree and nothing changes.
        return RewriteResult()

    @_rewrite.register(operations.StarRz)
    def _(self, stmt: operations.StarRz) -> RewriteResult:
        # Diagonal, so the frame commutes past it exactly -- passing it through
        # is correct. What we cannot do is *remove* its own rotation: STAR puts
        # an Rz on a subset of a block's physical qubits, which this rule's
        # per-logical-qubit frame does not model. That phase stays in the
        # program and reaches the backend as a physical local Rz; eliminating
        # it would need a physical-domain pass, which is out of scope here.
        warnings.warn(
            "EliminateRz does not remove the Rz introduced by the STAR gadget: "
            "it acts on a subset of a block's physical qubits, which this rule "
            "does not model. That rotation will reach the backend as a physical "
            "local Rz.",
        )
        return RewriteResult()

    # --- the actual work ---

    @_rewrite.register(native_gate.stmts.Rz)
    def _(self, stmt: native_gate.stmts.Rz) -> RewriteResult:
        for qubit in self._qubit_values(stmt, stmt.qubits):
            pending = self._frame.get(qubit)
            self._frame[qubit] = (
                stmt.rotation_angle
                if pending is None
                else self._combine(py.Add, pending, stmt.rotation_angle, stmt)
            )
        stmt.delete()
        return RewriteResult(has_done_something=True)

    @_rewrite.register(native_gate.stmts.R)
    def _(self, stmt: native_gate.stmts.R) -> RewriteResult:
        qubits = self._qubit_values(stmt, stmt.qubits)

        groups: dict[ir.SSAValue | None, list[ir.SSAValue]] = {}
        for qubit in qubits:
            groups.setdefault(self._frame.get(qubit), []).append(qubit)

        if tuple(groups) == (None,):
            # Nothing pending on any of these qubits.
            return RewriteResult()

        if len(groups) == 1:
            # Every qubit here carries the same frame, so the register is
            # unchanged and only the axis moves. Swap that one operand in
            # place: rebuilding the statement and its register would emit
            # identical IR at the cost of churn on every rewritten gate.
            (frame,) = groups
            assert frame is not None  # the all-None case returned above
            assert stmt.args[0] is stmt.axis_angle, "native.gate.R arg order changed"
            stmt.args[0] = self._combine(py.Sub, stmt.axis_angle, frame, stmt)
            return RewriteResult(has_done_something=True)

        # Frames disagree, and one pulse cannot carry two axis angles.
        for frame, group in groups.items():
            axis = (
                stmt.axis_angle
                if frame is None
                else self._combine(py.Sub, stmt.axis_angle, frame, stmt)
            )
            register = ilist.New(values=tuple(group), elem_type=bloqade_types.QubitType)
            register.insert_before(stmt)
            native_gate.stmts.R(
                axis_angle=axis,
                rotation_angle=stmt.rotation_angle,
                qubits=register.result,
            ).insert_before(stmt)

        stmt.delete()
        return RewriteResult(has_done_something=True)

    @_rewrite.register(operations.Initialize)
    def _(self, stmt: operations.Initialize) -> RewriteResult:
        for qubit in self._qubit_values(stmt, stmt.qubits):
            pending = self._frame.get(qubit)
            if pending is not None and _literal(pending) != 0.0:
                raise EliminateRzError(
                    "Initialize reached with a pending phase frame. Initialize "
                    "is expected at the head of a wire; a mid-wire one would "
                    "need the frame absorbed into its (theta, phi, lam)."
                )
        return RewriteResult()

    @_rewrite.register(operations.TerminalLogicalMeasurement)
    @_rewrite.register(squin_qubit.stmts.Measure)
    def _(
        self,
        stmt: operations.TerminalLogicalMeasurement | squin_qubit.stmts.Measure,
    ) -> RewriteResult:
        # The residual is diagonal and the readout is in the Z basis, so it
        # cannot shift any outcome. Drop it.
        #
        # Safe for a mid-circuit measurement too, not just the terminal one: a
        # Z measurement leaves the qubit in a computational basis state, where
        # any leftover Z phase is a global phase on that branch.
        for qubit in self._qubit_values(stmt, stmt.qubits):
            self._frame.pop(qubit, None)
        return RewriteResult()
