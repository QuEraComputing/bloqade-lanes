"""Tests for the DecomposeCliffordToNative rewrite rule.

Verifies that the composite squin Cliffords (H, CX, CY, Swap) expand into the
squin statements that map one-to-one onto Gemini's native gate set, that the
one-to-one Cliffords are left alone, and that the expansions agree with the
``bloqade.native.stdlib.broadcast`` kernels they mirror.

All test IR is hand-built using kirin.ir primitives; no upstream lowering
is involved.
"""

import bloqade.squin as squin
from bloqade.squin.gate import stmts as gate
from bloqade.test_utils import assert_nodes
from kirin import ir, rewrite
from kirin.analysis import const
from kirin.dialects import ilist, py

from bloqade import qubit
from bloqade.lanes.arch.gemini.physical import (
    get_arch_spec as get_physical_arch_spec,
)
from bloqade.lanes.dialects import place
from bloqade.lanes.rewrite.clifford2native import DecomposeCliffordToNative
from bloqade.lanes.transform import PhysicalNativeToPlace


def _rewrite(block: ir.Block):
    return rewrite.Walk(DecomposeCliffordToNative()).rewrite(block)


def test_hadamard_expands_to_s_sqrt_x_s():
    qubits = ir.TestValue()
    block = ir.Block([gate.H(qubits)])

    expected = ir.Block(
        [
            gate.S(qubits),
            gate.SqrtX(qubits),
            gate.S(qubits),
        ]
    )

    assert _rewrite(block).has_done_something
    assert_nodes(block, expected)


def test_cx_expands_to_sqrt_y_conjugated_cz():
    controls = ir.TestValue()
    targets = ir.TestValue()
    block = ir.Block([gate.CX(controls, targets)])

    expected = ir.Block(
        [
            gate.SqrtY(targets, adjoint=True),
            gate.CZ(controls, targets),
            gate.SqrtY(targets),
        ]
    )

    assert _rewrite(block).has_done_something
    assert_nodes(block, expected)


def test_cy_expands_to_sqrt_x_conjugated_cz():
    """CY's two sqrt(X) layers are the ones the Steane transversal rewrite has
    to flip; before this decomposition they only existed inside the native
    stdlib kernel (bloqade-internal#404)."""
    controls = ir.TestValue()
    targets = ir.TestValue()
    block = ir.Block([gate.CY(controls, targets)])

    expected = ir.Block(
        [
            gate.SqrtX(targets),
            gate.CZ(controls, targets),
            gate.SqrtX(targets, adjoint=True),
        ]
    )

    assert _rewrite(block).has_done_something
    assert_nodes(block, expected)


def test_swap_expands_to_three_cz_rounds():
    qubits1 = ir.TestValue()
    qubits2 = ir.TestValue()
    block = ir.Block([gate.Swap(qubits1, qubits2)])

    expected_both = py.Add(qubits1, qubits2)
    expected = ir.Block(
        [
            expected_both,
            gate.SqrtY(qubits2),
            gate.CZ(qubits1, qubits2),
            gate.SqrtY(expected_both.result),
            gate.CZ(qubits2, qubits1),
            gate.SqrtY(expected_both.result),
            gate.CZ(qubits1, qubits2),
            gate.SqrtY(qubits2),
        ]
    )

    assert _rewrite(block).has_done_something
    assert_nodes(block, expected)


def test_one_to_one_cliffords_are_untouched():
    """Gates that already map onto a single native primitive must survive
    unchanged — otherwise the rule would not be idempotent."""
    qubits = ir.TestValue()
    controls = ir.TestValue()
    block = ir.Block(
        [
            gate.X(qubits),
            gate.Y(qubits),
            gate.Z(qubits),
            gate.S(qubits),
            gate.S(qubits, adjoint=True),
            gate.SqrtX(qubits),
            gate.SqrtX(qubits, adjoint=True),
            gate.SqrtY(qubits),
            gate.SqrtY(qubits, adjoint=True),
            gate.CZ(controls, qubits),
        ]
    )

    assert not _rewrite(block).has_done_something


def test_rule_is_idempotent():
    controls = ir.TestValue()
    targets = ir.TestValue()
    block = ir.Block(
        [
            gate.H(targets),
            gate.CX(controls, targets),
            gate.CY(controls, targets),
        ]
    )

    assert _rewrite(block).has_done_something
    stmts_after_first = [type(stmt) for stmt in block.stmts]

    assert not _rewrite(block).has_done_something
    assert [type(stmt) for stmt in block.stmts] == stmts_after_first


def _lowered_gate_signature(mt) -> list[str]:
    """Summarize a lowered kernel as its ordered place-level gate ops, with
    const-folded angles."""

    def angle(value) -> str:
        hint = value.hints.get("const")
        return str(hint.data) if isinstance(hint, const.Value) else "?"

    signature = []
    for stmt in mt.callable_region.walk():
        if isinstance(stmt, place.CZ):
            signature.append(f"CZ{stmt.qubits}")
        elif isinstance(stmt, place.R):
            signature.append(
                f"R({angle(stmt.axis_angle)},{angle(stmt.rotation_angle)}){stmt.qubits}"
            )
        elif isinstance(stmt, place.Rz):
            signature.append(f"Rz({angle(stmt.rotation_angle)}){stmt.qubits}")
    return signature


class _NoDecomposition(PhysicalNativeToPlace):
    """Physical lowering with the Clifford decomposition disabled, so the
    composites fall through to the ``bloqade.native`` stdlib kernels the way
    they did before this rule existed."""

    def _squin_clifford_rules(self):
        return []


def test_decomposition_matches_the_native_stdlib_expansion():
    """The statement-level decompositions must reproduce exactly what the
    ``bloqade.native.stdlib.broadcast`` kernels expand to.

    Compiling the same kernel with and without the rule has to give the same
    lowered gate sequence; if the two ever drift, this is where it shows up.
    """

    @squin.kernel
    def kernel():
        q = qubit.qalloc(4)
        squin.h(q[0])
        squin.swap(q[0], q[1])
        squin.cx(q[1], q[2])
        squin.cy(q[2], q[3])
        squin.broadcast.swap(ilist.IList([q[0], q[2]]), ilist.IList([q[1], q[3]]))
        squin.broadcast.measure(q)

    arch_spec = get_physical_arch_spec()
    decomposed = PhysicalNativeToPlace(arch_spec=arch_spec).emit(kernel, no_raise=True)
    via_stdlib = _NoDecomposition(arch_spec=arch_spec).emit(kernel, no_raise=True)

    signature = _lowered_gate_signature(decomposed)
    assert signature, "expected the kernel to lower to some gates"
    assert signature == _lowered_gate_signature(via_stdlib)


def test_non_clifford_gates_are_left_to_squin_to_native():
    """T / rotations / U3 / CCZ are not this rule's business."""
    angle = ir.TestValue()
    qubits = ir.TestValue()
    controls1 = ir.TestValue()
    controls2 = ir.TestValue()
    block = ir.Block(
        [
            gate.T(qubits),
            gate.Rx(angle, qubits),
            gate.Ry(angle, qubits),
            gate.Rz(angle, qubits),
            gate.U3(angle, angle, angle, qubits),
            gate.CCZ(controls1, controls2, qubits),
        ]
    )

    assert not _rewrite(block).has_done_something
