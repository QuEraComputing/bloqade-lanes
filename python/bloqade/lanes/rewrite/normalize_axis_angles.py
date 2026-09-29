"""Reduce constant ``R`` axis angles mod one turn, so equal angles can share.

``EliminateRz`` folds each commuted phase by plain addition, so a qubit that
sees N ``Rz(0.25)`` layers ends up with axis angles 0.25, 0.5, ... 0.25*N --
each a distinct ``py.Constant``, growing without bound with circuit depth.

Nothing downstream rejects an out-of-range angle: the Rust bytecode core
pushes ``local_r``/``global_r`` operands as a plain ``cpu::cpu.const f64``,
and ``move2squin`` lowers ``move.LocalR`` to a ``U3`` that is periodic in phi and
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

**Axis angles only.** For ``R(phi, theta)`` with angles in turns, phi has
period exactly 1, so ``phi mod 1`` is an identity with no caveat. Adding a
turn to *theta* gives ``-R(phi, theta)``; the period is 2. That sign is a
global phase a Tableau simulator cannot observe, so modding it would be
defensible -- but it stops being an identity, and ``EliminateRz`` only ever
grows axis angles, so there is nothing to gain by taking it on.

**Constants are reduced now; everything else gets a ``py.Mod``.** Only the
constant case can recover sharing: ``fuse_gates`` and
``reorder_static_placement`` compare axis angles by *identity*, so two axes
that are physically equal are only seen to be equal if the compiler computes
them and CSE merges the results. A ``py.Mod`` is opaque to that -- the value
exists but ``constant_float`` cannot read it, exactly as for a kernel argument --
so deferring the constant case would defeat the rule's own purpose. Deferring
the symbolic case costs nothing, because there was never anything to read.

Two consequences, both stated on the class and neither guarded against:
a downstream pass cannot assume a constant axis angle, and this rule is not
idempotent.
"""

from __future__ import annotations

from bloqade.native.dialects import gate as native_gate
from kirin import ir
from kirin.dialects import py
from kirin.rewrite.abc import RewriteResult, RewriteRule

from bloqade.lanes.utils import constant_float

__all__ = ["NormalizeGateAxisAngles"]


class NormalizeGateAxisAngles(RewriteRule):
    """Rewrite each ``R`` axis angle to a representative in [0, 1).

    A constant is reduced now, by the compiler. Anything else gets a
    ``py.Mod`` and is reduced by whoever runs the program.

    **This rule does not preserve a constant axis angle.** A pass downstream
    that needs to read the number must handle a non-constant operand or
    validate the shape it requires -- it cannot assume one. Nothing in the
    logical pipeline is affected today, since every axis angle reaching this
    rule is a constant, but that is a property of Clifford kernels rather than
    a guarantee this rule offers.

    **This rule is not idempotent.** ``constant_float`` cannot see through a
    ``py.Mod``, so a second pass over an already-normalized symbolic axis
    wraps it again. Run it once; do not put it under ``Fixpoint``, which would
    nest a ``py.Mod`` per iteration until ``max_iter``.
    """

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        if not isinstance(node, native_gate.stmts.R):
            return RewriteResult()

        axis = node.axis_angle
        if (value := constant_float(axis)) is not None:
            normalized = value % 1.0
            if normalized == value:
                return RewriteResult()
            # Insert a fresh constant rather than mutating the existing one. By
            # this point constants are shared across *roles*: the real
            # post-unroll IR contains a single `py.Constant 0.25` feeding both
            # the axis angle and the rotation angle of the same statement, and
            # `HoistConstants` running just before makes that more likely, not
            # less. Rewriting the constant in place would silently move the
            # rotation angle too.
            replacement: ir.Statement = py.Constant(normalized)
        else:
            # The compiler cannot read this one, so defer the reduction rather
            # than skip it. `py.Mod` is `Pure`, so the CSE in `emit` merges
            # identical ones and the duplicate `1.0` constants along with them.
            (one := py.Constant(1.0)).insert_before(node)
            replacement = py.Mod(axis, one.result)

        replacement.insert_before(node)
        node.args.set_item(
            node.args.get_slice("axis_angle").start, replacement.results[0]
        )
        return RewriteResult(has_done_something=True)
