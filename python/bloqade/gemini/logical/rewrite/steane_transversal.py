from bloqade.squin import gate
from kirin import ir
from kirin.rewrite import abc as rewrite_abc


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
