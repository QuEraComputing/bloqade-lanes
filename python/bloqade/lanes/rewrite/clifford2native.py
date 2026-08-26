"""Decompose composite squin Cliffords into the neutral-atom native gate set.

Gemini runs three primitives: a rotation ``R(axis_angle, rotation_angle)``, a
``Rz(angle)``, and an entangling ``CZ``.  Seven squin Clifford statements map
onto one of those one-to-one --

======================  ==========================
squin statement         neutral-atom primitive
======================  ==========================
``X``                   ``R(0, 1/2)``
``Y``                   ``R(1/4, 1/2)``
``Z``                   ``Rz(1/2)``
``S`` / ``S_adj``       ``Rz(+/- 1/4)``
``SqrtX`` / adj         ``R(0, +/- 1/4)``
``SqrtY`` / adj         ``R(1/4, +/- 1/4)``
``CZ``                  ``CZ``
======================  ==========================

-- while ``H``, ``CX``, ``CY`` and ``Swap`` are composites that have to be
expanded first.

Upstream ``SquinToNative`` does expand them, but *inside* the
``bloqade.native`` standard-library kernels (``broadcast.cy`` and friends), one
level below squin.  Anything the composites expand to is therefore invisible to
squin-statement rewrites, which is how logical ``squin.cy`` ended up as
``Zbar(control) . CY``: the Steane transversal correction had no ``SqrtX``
statement to flip, because ``cy``'s two sqrt(X) layers only existed inside the
native kernel (QuEraComputing/bloqade-internal#404).

This rule performs the same expansion at the *statement* level instead, so
every gate the hardware will actually run is visible as squin IR before
``SquinToNative`` turns it into rotations.  The decompositions below are the
ones in ``bloqade.native.stdlib.broadcast``; keep them in sync.
"""

from bloqade.squin.gate import stmts as gate
from kirin import ir
from kirin.dialects import py
from kirin.rewrite import abc as rewrite_abc

__all__ = ["DecomposeCliffordToNative"]


class DecomposeCliffordToNative(rewrite_abc.RewriteRule):
    """Expand ``H``, ``CX``, ``CY`` and ``Swap`` into the squin Clifford
    statements that lower one-to-one onto Gemini's native gate set.

    Idempotent: the expansions only emit statements the rule does not match.

    Ordering
    --------
    The ``_hadamard`` / ``_controlled_*`` / ``_swap`` helpers return their
    statements in **circuit order** -- element 0 runs first -- and ``_expand``
    preserves that order in the IR.

    Their docstrings state the identity as an **operator product**, which reads
    the other way round, right to left.  So

        CX = sqrt(Y) . CZ . sqrt(Y)^dag      (operator product)

    is emitted as, and appears in the IR as,

        sqrt_y_adj(targets); cz(controls, targets); sqrt_y(targets)

    The two are reverses of each other by convention, not by accident: the
    conjugator adjacent to the ket in the product is the gate applied *first*.
    Read the formula when you want to check the conjugation direction, and the
    returned list when you want to know what the hardware runs.
    """

    def rewrite_Statement(self, node: ir.Statement) -> rewrite_abc.RewriteResult:
        if isinstance(node, gate.H):
            return self._expand(node, self._hadamard(node))
        if isinstance(node, gate.CX):
            return self._expand(node, self._controlled_x(node))
        if isinstance(node, gate.CY):
            return self._expand(node, self._controlled_y(node))
        if isinstance(node, gate.Swap):
            return self._expand(node, self._swap(node))
        return rewrite_abc.RewriteResult()

    @staticmethod
    def _expand(
        node: ir.Statement, replacement: list[ir.Statement]
    ) -> rewrite_abc.RewriteResult:
        # Each insert lands immediately before `node` and therefore after the
        # previously inserted one, so the IR ends up in `replacement` order --
        # circuit order, not the reversed operator-product order the
        # decomposition docstrings use. See the class docstring.
        for stmt in replacement:
            stmt.insert_before(node)
        # Clifford gate statements carry no results, so nothing to reconnect.
        node.delete()
        return rewrite_abc.RewriteResult(has_done_something=True)

    @staticmethod
    def _hadamard(node: gate.H) -> list[ir.Statement]:
        """``H = S . sqrt(X) . S`` (up to global phase).

        A palindrome, so the operator product and the emitted circuit order
        read alike here.
        """
        return [
            gate.S(node.qubits),
            gate.SqrtX(node.qubits),
            gate.S(node.qubits),
        ]

    @staticmethod
    def _controlled_x(node: gate.CX) -> list[ir.Statement]:
        """``CX = sqrt(Y) . CZ . sqrt(Y)^dag``, conjugating on the target.

        Operator product; emitted in circuit order as
        ``sqrt_y_adj; cz; sqrt_y``.
        """
        return [
            gate.SqrtY(node.targets, adjoint=True),
            gate.CZ(node.controls, node.targets),
            gate.SqrtY(node.targets),
        ]

    @staticmethod
    def _controlled_y(node: gate.CY) -> list[ir.Statement]:
        """``CY = sqrt(X)^dag . CZ . sqrt(X)``, conjugating on the target.

        Operator product; emitted in circuit order as
        ``sqrt_x; cz; sqrt_x_adj``.  The transversal rewrite flips both sqrt(X)
        layers on the logical path, which is what makes logical CY come out as
        CY rather than ``Zbar(control) . CY``.
        """
        return [
            gate.SqrtX(node.targets),
            gate.CZ(node.controls, node.targets),
            gate.SqrtX(node.targets, adjoint=True),
        ]

    @staticmethod
    def _swap(node: gate.Swap) -> list[ir.Statement]:
        """Three ``CZ`` rounds interleaved with sqrt(Y) layers, in circuit order.

        The leading ``py.Add`` is the ``qubits1 + qubits2`` concatenation the
        two middle sqrt(Y) layers broadcast over; it has to be emitted before
        its uses.
        """
        both = py.Add(node.qubits1, node.qubits2)
        return [
            both,
            gate.SqrtY(node.qubits2),
            gate.CZ(node.qubits1, node.qubits2),
            gate.SqrtY(both.result),
            gate.CZ(node.qubits2, node.qubits1),
            gate.SqrtY(both.result),
            gate.CZ(node.qubits1, node.qubits2),
            gate.SqrtY(node.qubits2),
        ]
