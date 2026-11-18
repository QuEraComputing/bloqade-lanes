from bloqade.squin import gate
from kirin import ir
from kirin.dialects import py
from kirin.rewrite import abc as rewrite_abc

from bloqade.lanes.dialects.steane713.extensions import stmts as extensions
from bloqade.lanes.utils import constant_float


class RewriteSteaneTransversalCliffordAdjoints(rewrite_abc.RewriteRule):
    """Swap adjoints for Steane transversal sqrt-X and sqrt-Z/S gates.

    This is specific to the Steane-code logical-operator convention used by
    the Gemini logical pipeline. Because the logical operators are odd-weight
    representatives,

        \\bar X = X_1 X_2 X_3
        \\bar Y = i \\bar X \\bar Z = -Y_1 Y_2 Y_3

    single-qubit Clifford conjugation signs accumulate across the transversal
    support. In particular, sqrt(X) Z sqrt(X)^dag = Y on one physical qubit
    implies

        Z_1 Z_2 Z_3 -> Y_1 Y_2 Y_3 = -\\bar Y

    so transversal physical sqrt(X) implements logical -sqrt(X). Using
    transversal sqrt(X)^dag instead gives

        Z_1 Z_2 Z_3 -> -Y_1 Y_2 Y_3 = \\bar Y

    which matches the desired logical Pauli action. The same odd-weight sign
    issue applies to sqrt(Z), represented in Squin IR as S/S^dag.
    """

    def rewrite_Statement(self, node: ir.Statement) -> rewrite_abc.RewriteResult:
        if not isinstance(node, (gate.stmts.SqrtX, gate.stmts.S)):
            return rewrite_abc.RewriteResult()

        node.replace_by(type(node)(node.qubits, adjoint=not node.adjoint))
        return rewrite_abc.RewriteResult(has_done_something=True)


class RewriteSteaneStarRAxisConvention(rewrite_abc.RewriteRule):
    """Negate each ``StarR`` axis into the convention ``EliminateRz`` works in.

    ``star_r``'s ``axis_angle`` is the *logical* axis. Under the transversal
    rewrite the native Z frame ``EliminateRz`` absorbs is physical: logical
    ``S`` is emitted as transversal ``S^dag`` (see
    ``RewriteSteaneTransversalCliffordAdjoints``), so a physical frame
    ``alpha`` is a logical ``Rz(-alpha)``, and absorbing it moves the logical
    axis by ``+alpha``. ``EliminateRz``'s ``R`` handler subtracts the frame, so
    storing ``-axis`` makes the one shared handler move the logical axis
    correctly: ``-(axis) - alpha = -(axis + alpha)``.

    The STAR support also has ``Y_L = -Ybar`` on the code space, so at
    quarter-turn axes ``-axis`` is exactly the physical axis of the gadget.
    Other axes are pre-distorted from ``-axis`` by ``RewriteStarR``.

    Mutates the operand in place, so ``Walk`` cannot revisit a rewritten node
    and negate it twice.
    """

    def rewrite_Statement(self, node: ir.Statement) -> rewrite_abc.RewriteResult:
        if not isinstance(node, extensions.StarR):
            return rewrite_abc.RewriteResult()

        axis = node.axis_angle
        if (value := constant_float(axis)) is not None:
            negated: ir.Statement = py.Constant(-value)
        else:
            # 0 - axis rather than USub: EliminateRz's dispatch is exhaustive
            # and already registers BinOp as inert.
            (zero := py.Constant(0.0)).insert_before(node)
            negated = py.Sub(zero.result, axis)
        negated.insert_before(node)
        node.args.set_item(node.args.get_slice("axis_angle").start, negated.result)
        return rewrite_abc.RewriteResult(has_done_something=True)
