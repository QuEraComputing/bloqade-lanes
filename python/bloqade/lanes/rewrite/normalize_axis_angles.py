"""Reduce constant ``R`` axis angles mod one turn, so equal angles can share.

``EliminateRz`` folds each commuted phase by plain addition, so a qubit that
sees N ``Rz(0.25)`` layers ends up with axis angles 0.25, 0.5, ... 0.25*N --
each a distinct ``py.Constant``, growing without bound with circuit depth.

Nothing downstream rejects an out-of-range angle: the Rust bytecode core
pushes ``local_r``/``global_r`` operands as plain ``const.f64``, and
``move2squin`` lowers ``move.LocalR`` to a ``U3`` that is periodic in phi and
lam anyway. The cost is sharing, not correctness. The checks that let two
gates fuse or reorder together are **identity**-based, not value-based --
``fuse_gates`` compares ``stmt.axis_angle is head.axis_angle`` and
``reorder_static_placement`` keys on ``id(stmt.axis_angle)`` -- so two axes
that are physically the same angle but differ by a whole turn can never be
merged into one SSA value by CSE, and therefore never fuse.

That pins where this runs: after ``EliminateRz`` and *before* the
``CommonSubexpressionElimination`` in ``NativeToPlaceBase.emit``. Bolted onto
the end of the pipeline it would produce tidy angles and recover none of the
fusion, which is the whole point.

Two deliberate restrictions:

* **Axis angles only.** For ``R(phi, theta)`` with angles in turns, phi has
  period exactly 1, so ``phi mod 1`` is an identity with no caveat. Adding a
  turn to *theta* gives ``-R(phi, theta)``; the period is 2. That sign is a
  global phase a Tableau simulator cannot observe, so modding it would be
  defensible -- but it stops being an identity, and ``EliminateRz`` only ever
  grows axis angles, so there is nothing to gain by taking it on.

* **Constant operands only.** Normalizing a symbolic angle means emitting a
  live ``py.Mod``, and nothing runs constant folding after this point in
  ``emit`` -- it would survive into the emitted program. The symbolic branch
  does not have the problem being fixed either: an unfolded chain never
  produces a ladder of distinct constants for CSE to fail on.
"""

from __future__ import annotations

from bloqade.native.dialects import gate as native_gate
from kirin import ir
from kirin.dialects import py
from kirin.rewrite.abc import RewriteResult, RewriteRule

__all__ = ["NormalizeGateAxisAngles"]


class NormalizeGateAxisAngles(RewriteRule):
    """Rewrite each constant ``R`` axis angle to its representative in [0, 1)."""

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        if not isinstance(node, native_gate.stmts.R):
            return RewriteResult()

        axis = node.axis_angle
        if not isinstance(axis, ir.ResultValue) or not isinstance(
            axis.owner, py.Constant
        ):
            return RewriteResult()
        value = axis.owner.value.unwrap()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return RewriteResult()

        normalized = float(value) % 1.0
        if normalized == value:
            return RewriteResult()

        # Insert a fresh constant rather than mutating the existing one. By
        # this point constants are shared across *roles*: the real post-unroll
        # IR contains a single `py.Constant 0.25` feeding both the axis angle
        # and the rotation angle of the same statement, and `HoistConstants`
        # running just before makes that more likely, not less. Rewriting the
        # constant in place would silently move the rotation angle too. The
        # CSE and DCE later in `emit` merge the duplicates and sweep the
        # originals.
        (constant := py.Constant(normalized)).insert_before(node)
        node.args.set_item(node.args.get_slice("axis_angle").start, constant.result)
        return RewriteResult(has_done_something=True)
