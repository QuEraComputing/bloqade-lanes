"""EliminateRz: remove ``Rz`` from native-dialect programs by phase commutation.

Scans a flat block in order carrying a per-qubit phase *frame*. Each ``Rz`` is
absorbed into the frame and deleted; each ``R`` has its axis angle shifted by the
frame of the qubits it addresses; ``CZ`` and ``StarRz`` are diagonal and pass
through untouched. Whatever frame remains when the block ends is discarded --
sound because the residual is diagonal and the device's readout is in the Z
basis. ``flush_residual`` writes it back out instead, which makes the emitted
program exactly equal the input; see that field's docstring for why a test
wants it.

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
Registering the unknown type as a no-op is not an available fix either:
``func.Invoke`` may contain gates, so a handler claiming it is phase-neutral
would be false, and no handler can be written that is true.

``no_raise`` selects what happens when the scan meets something it cannot
reason about. Under the default (``False``) it raises. Under ``True`` it
*drains* instead: the pending frame is re-emitted as real ``Rz`` statements
immediately before the offending statement, which is exact rather than merely
conservative. The rule moves each ``Rz`` *later* in the circuit, so a frame of
``alpha`` pending on a qubit says precisely that the emitted program so far,
followed by ``Rz(alpha)``, equals the original program so far -- re-emitting it
restores the original exactly, and leaves the frame empty, which is the state
the scan starts in. The cost is that such a program reaches the backend with
``Rz`` still in it and fails at the device rather than at compile time, which
is what best-effort mode buys everywhere else in this pipeline.

See ``docs/superpowers/specs/2026-09-14-rz-elimination-design.md``.
"""

from __future__ import annotations

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
from bloqade.gemini.logical.dialects.extensions import stmts as extensions
from bloqade.gemini.logical.dialects.operations import stmts as operations
from bloqade.lanes.utils import constant_float

__all__ = ["EliminateRz", "EliminateRzError"]


class EliminateRzError(ir.ValidationError):
    """The IR violates a precondition of the Rz elimination scan.

    A ``kirin`` ``ValidationError`` rather than a bare ``Exception`` so it
    carries the offending node and its source location, and renders with the
    same caret hint as every other check reachable from
    ``NativeToPlaceBase.emit``. Construct it as
    ``EliminateRzError(node, "message")``.
    """


@dataclass
class EliminateRz(RewriteRule):
    """Remove every ``Rz`` from a flat native-dialect program.

    **Preconditions**, both validated by ``NativeToPlaceBase.emit``
    immediately before this rule and neither re-checked here:

    * A single-block callable region with no nested ``func.Function``
      (``lanes.flat_block.validation``). The frame is accumulated in the order
      ``Walk`` offers statements, which matches execution order within a block
      and nowhere else, so violating this yields a different circuit rather
      than an error.
    * Every ``qubits`` operand is an ``ilist.New`` of distinct qubits
      (``lanes.qubit_register.validation``). That is why the handlers below
      read ``stmt.qubits.owner`` as an ``ilist.New`` without checking.

    A caller driving this rule directly is responsible for both.

    One ``Rz`` is deliberately left behind: the one the STAR gadget
    introduces. It acts on a subset of a block's *physical* qubits, which this
    rule's per-logical-qubit frame does not model, so it passes through and
    reaches the backend as a physical local ``Rz``. Removing it would need a
    physical-domain pass, which is out of scope here.

    That is stated here rather than raised as a warning at compile time: the
    condition is static -- a ``StarRz`` is present in the program the caller
    wrote -- and there is no action the caller could take in response, which
    is usually the sign it belongs in documentation.
    """

    no_raise: bool = False
    """Give up gracefully instead of raising when a precondition fails.

    Mirrors the flag the surrounding pipeline threads through its passes. See
    the module docstring for why giving up means *draining* the frame rather
    than skipping the statement: skipping would leave the program half
    commuted, valid, and silently wrong.
    """

    flush_residual: bool = False
    """Write the leftover frame back out as terminal ``Rz`` rather than dropping it.

    Off in production: the whole point of the pass is that the residual is
    diagonal and the readout is in the Z basis, so emitting it would put back
    an ``Rz`` the backend cannot execute for no gain.

    On, the emitted program is *exactly* the input program -- every phase this
    pass commuted forward is restored at the end of the wire it belongs to.
    That makes the commutation itself checkable against an exact-state
    reference, which discarding does not: a discarded residual is
    indistinguishable from a sign error in the commutation, since both show up
    only as a Z-type difference that a Z-basis measurement cannot see. See
    ``test_logical_clifford_lowering.py``, which runs the pass this way so its
    signed stabilizer checks cover the real rule.
    """

    _frame: dict[ir.SSAValue, ir.SSAValue] = field(default_factory=dict, init=False)
    """Pending Z phase per qubit, as the SSA value holding it. Spans the program.

    A qubit is absent until an ``Rz`` touches it, so "no pending phase" and "not
    in this dict" are the same statement.

    Starts empty and is never reset, which makes an instance **single use**:
    one rule, one program. That is how ``NativeToPlaceBase.emit`` uses it, and
    the frame models a physical quantity -- Z rotation accumulated on an atom
    over one shot -- so there is no scope smaller than the whole walk to reset
    it at. Walking the same instance over a second program would start it from
    whatever the first left pending.
    """

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        return self._rewrite(node)

    # -- helpers ----------------------------------------------------------

    def _drain(self, before: ir.Statement) -> bool:
        """Re-emit the pending frame as ``Rz`` statements, restoring the original.

        Exact, not a fallback approximation: the scan pushes each ``Rz``
        *later* in the circuit, so a pending ``alpha`` means the emitted
        program followed by ``Rz(alpha)`` equals the original. Writing it back
        here undoes exactly what was taken out, and leaves the frame empty --
        the state the scan starts in, so it is safe to carry on afterwards.

        Returns whether anything was emitted. An empty frame drains to nothing,
        which is the common case when the scan gives up before any ``Rz`` has
        been absorbed, and callers must not then claim to have changed the IR.
        """
        if not self._frame:
            return False

        for qubit, angle in self._frame.items():
            register = ilist.New(values=(qubit,), elem_type=bloqade_types.QubitType)
            register.insert_before(before)
            native_gate.stmts.Rz(angle, register.result).insert_before(before)
        self._frame = {}
        return True

    def _give_up(self, message: str, stmt: ir.Statement) -> RewriteResult:
        """Raise, or under ``no_raise`` drain the frame and leave ``stmt`` alone.

        **Always call this as** ``return self._give_up(...)``. It both performs
        the give-up and reports it, and the report cannot be reconstructed by
        the caller: only ``_drain`` knows whether the frame held anything to
        write back. Calling it without returning -- inside a loop, say --
        silently discards that, and there is no meaningful way to combine two
        of them, since giving up twice in one statement is not a thing that
        happens: the first drain empties the frame.

        Both call sites obey this. Keep it that way; if a third needs to
        check several things first, decide with ``any``/``all`` and give up
        once, as the ``Initialize`` handler does.
        """
        if not self.no_raise:
            raise EliminateRzError(stmt, message)
        return RewriteResult(has_done_something=self._drain(stmt))

    def _combine(
        self,
        op: type[py.Add | py.Sub],
        lhs: ir.SSAValue,
        rhs: ir.SSAValue,
        before: ir.Statement,
    ) -> ir.SSAValue:
        """``lhs op rhs`` as an SSA value, folded to a literal when both are."""
        left, right = constant_float(lhs), constant_float(rhs)
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
        return self._give_up(
            f"{stmt.name} has no EliminateRz handler, so its effect on a phase "
            "frame is unknown. Register it explicitly -- as a no-op if it is "
            "phase-neutral -- rather than letting it pass silently. Note that "
            "not every statement can be registered: func.Invoke may contain "
            "gates, so no handler for it would be truthful.",
            stmt,
        )

    # --- inert: carry no phase, touch no frame ---

    @_rewrite.register(py.Constant)
    @_rewrite.register(py.GetItem)
    @_rewrite.register(py_binop.BinOp)  # Add / Sub / Mult / Div / ...
    @_rewrite.register(py_tuple.New)
    @_rewrite.register(ilist.New)
    @_rewrite.register(squin_qubit.stmts.New)
    @_rewrite.register(gemini_qubit.stmts.NewAt)
    @_rewrite.register(func.ConstantNone)
    # The walk root's own definition. A *nested* one would break the frame,
    # but lanes.flat_block.validation rejects that before this rule runs.
    @_rewrite.register(func.Function)
    @_rewrite.register(annotate.SetDetector)
    @_rewrite.register(annotate.SetObservable)
    def _(self, stmt: ir.Statement) -> RewriteResult:
        return RewriteResult()

    @_rewrite.register(func.Return)
    def _(self, stmt: func.Return) -> RewriteResult:
        if self.flush_residual:
            return RewriteResult(has_done_something=self._drain(stmt))
        return RewriteResult()

    # --- diagonal: commute with the frame exactly, so nothing to do ---

    @_rewrite.register(native_gate.stmts.CZ)
    @_rewrite.register(extensions.StarRz)
    def _(self, stmt: native_gate.stmts.CZ | extensions.StarRz) -> RewriteResult:
        # The frame commutes past a diagonal gate exactly, so there is nothing
        # to do. ``CZ`` is diagonal on each qubit independently, so the two
        # sides' frames need not agree. ``StarRz``'s own rotation survives into
        # the emitted program -- see the class docstring.
        return RewriteResult()

    # --- the actual work ---

    @_rewrite.register(native_gate.stmts.Rz)
    def _(self, stmt: native_gate.stmts.Rz) -> RewriteResult:
        source: ilist.New = stmt.qubits.owner  # type: ignore[reportAssignmentType]
        for qubit in source.values:
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
        source: ilist.New = stmt.qubits.owner  # type: ignore[reportAssignmentType]

        groups: dict[ir.SSAValue | None, list[ir.SSAValue]] = {}
        for qubit in source.values:
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
            shifted = self._combine(py.Sub, stmt.axis_angle, frame, stmt)
            stmt.args.set_item(stmt.args.get_slice("axis_angle").start, shifted)
            return RewriteResult(has_done_something=True)

        # Frames disagree, and one pulse cannot carry two axis angles.
        for frame, group in groups.items():
            axis = (
                stmt.axis_angle
                if frame is None
                else self._combine(py.Sub, stmt.axis_angle, frame, stmt)
            )
            register = ilist.New(values=tuple(group), elem_type=source.elem_type)
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
        source: ilist.New = stmt.qubits.owner  # type: ignore[reportAssignmentType]
        # Any pending phase here means an upstream invariant broke, so the test
        # is just "is one pending" -- not "is one pending that fails to be a
        # literal zero", which is what it used to be. That old form rejected a
        # symbolic frame (an angle passed as a function argument) simply
        # because a non-constant has no literal to compare, even though the
        # SSA-valued frame exists precisely to support those; it rejected a
        # frame of exactly 1.0 turns, the identity; and when it passed it left
        # the entry in the dict, so later gates on that qubit paid for a
        # semantically empty frame.
        #
        # Raising rather than popping is deliberate. _RewriteU3ToInitialize
        # documents that it "assumes there are no other U3 gates acting on the
        # same qubits later in the circuit", and there is no logical reset
        # statement, so a mid-wire Initialize is not a valid program. Popping
        # would quietly accept one and discard the phase it carries.
        if any(qubit in self._frame for qubit in source.values):
            return self._give_up(
                f"{stmt.name} reached with a pending phase frame. Initialize "
                "sits at the head of a wire -- there is no logical reset -- "
                "so a phase pending here means an earlier rewrite emitted a "
                "mid-wire Initialize, violating the assumption documented on "
                "_RewriteU3ToInitialize.",
                stmt,
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
        source: ilist.New = stmt.qubits.owner  # type: ignore[reportAssignmentType]
        if self.flush_residual:
            return RewriteResult(has_done_something=self._drain(stmt))
        for qubit in source.values:
            self._frame.pop(qubit, None)
        return RewriteResult()
