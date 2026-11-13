"""Compiler validation for STAR-X and Rz-producing logical gates."""

import pytest
from kirin import ir
from kirin.dialects import func, ilist, py
from kirin.ir.exception import ValidationErrorGroup
from kirin.validation import ValidationSuite

from bloqade import squin, types
from bloqade.lanes.dialects import steane713
from bloqade.lanes.dialects.steane713 import extensions
from bloqade.lanes.validation.star_rx_rz import StarRxRzGateValidation

from ._steane_kernel import kernel as steane_kernel


def test_h_with_star_rx_rejected():
    with pytest.raises(ValidationErrorGroup, match="StarRx.*H"):

        @steane_kernel
        def main():
            q = squin.qalloc(1)
            squin.h(q[0])
            extensions.star_rx(0.125, ilist.IList([q[0]]))
            steane713.terminal_measure(q)

        ValidationSuite([StarRxRzGateValidation]).validate(main).raise_if_invalid()


def test_first_gate_rz_with_star_rx_rejected():
    with pytest.raises(ValidationErrorGroup, match="StarRx.*Rz"):

        @steane_kernel
        def main():
            q = squin.qalloc(1)
            squin.rz(0.25, q[0])
            extensions.star_rx(0.125, ilist.IList([q[0]]))
            steane713.terminal_measure(q)

        ValidationSuite([StarRxRzGateValidation]).validate(main).raise_if_invalid()


def test_z_with_star_rx_rejected():
    with pytest.raises(ValidationErrorGroup, match="StarRx.*Z"):

        @steane_kernel
        def main():
            q = squin.qalloc(1)
            squin.z(q[0])
            extensions.star_rx(0.125, ilist.IList([q[0]]))
            steane713.terminal_measure(q)

        ValidationSuite([StarRxRzGateValidation]).validate(main).raise_if_invalid()


def test_s_with_star_rx_rejected():
    with pytest.raises(ValidationErrorGroup, match="StarRx.*S"):

        @steane_kernel
        def main():
            q = squin.qalloc(1)
            squin.s(q[0])
            extensions.star_rx(0.125, ilist.IList([q[0]]))
            steane713.terminal_measure(q)

        ValidationSuite([StarRxRzGateValidation]).validate(main).raise_if_invalid()


def test_s_adjoint_with_star_rx_rejected():
    with pytest.raises(ValidationErrorGroup, match="StarRx.*S"):

        @steane_kernel
        def main():
            q = squin.qalloc(1)
            squin.s_adj(q[0])
            extensions.star_rx(0.125, ilist.IList([q[0]]))
            steane713.terminal_measure(q)

        ValidationSuite([StarRxRzGateValidation]).validate(main).raise_if_invalid()


def test_h_after_star_rx_is_still_rejected():
    with pytest.raises(ValidationErrorGroup, match="StarRx.*H"):

        @steane_kernel
        def main():
            q = squin.qalloc(1)
            extensions.star_rx(0.125, ilist.IList([q[0]]))
            squin.h(q[0])
            steane713.terminal_measure(q)

        ValidationSuite([StarRxRzGateValidation]).validate(main).raise_if_invalid()


def test_conflicting_gate_on_different_qubit_is_still_rejected():
    with pytest.raises(ValidationErrorGroup, match="StarRx.*S"):

        @steane_kernel
        def main():
            q = squin.qalloc(2)
            squin.s(q[1])
            extensions.star_rx(0.125, ilist.IList([q[0]]))
            steane713.terminal_measure(q)

        ValidationSuite([StarRxRzGateValidation]).validate(main).raise_if_invalid()


def test_conflict_in_non_inlined_static_helper_is_rejected():
    @steane_kernel(unroll=False)
    def rotate(q: types.Qubit):
        squin.h(q)

    @steane_kernel(unroll=False)
    def main():
        q = squin.qalloc(1)
        rotate(q[0])
        extensions.star_rx(0.125, ilist.IList([q[0]]))
        steane713.terminal_measure(q)

    result = ValidationSuite([StarRxRzGateValidation]).validate(main)
    assert not result.is_valid
    assert any(isinstance(stmt, func.Invoke) for stmt in main.callable_region.walk())
    (error,) = result.errors["Steane StarRx Rz Gate Validation"]
    assert "StarRx" in error.args[0] and "H" in error.args[0]


def test_conflict_in_static_ilist_map_helper_is_rejected():
    @steane_kernel(unroll=False)
    def rotate(q: types.Qubit):
        squin.h(q)
        return q

    @steane_kernel(unroll=False)
    def main():
        q = squin.qalloc(1)
        ilist.map(rotate, q)
        extensions.star_rx(0.125, ilist.IList([q[0]]))
        steane713.terminal_measure(q)

    assert any(isinstance(stmt, ilist.Map) for stmt in main.callable_region.walk())
    result = ValidationSuite([StarRxRzGateValidation]).validate(main)
    assert not result.is_valid
    (error,) = result.errors["Steane StarRx Rz Gate Validation"]
    assert "StarRx" in error.args[0] and "H" in error.args[0]


def test_stored_but_never_called_helper_does_not_conflict():
    @steane_kernel(unroll=False)
    def rotate(q: types.Qubit):
        squin.h(q)

    @steane_kernel(unroll=False)
    def main():
        q = squin.qalloc(1)
        _callbacks = (rotate,)
        extensions.star_rx(0.125, ilist.IList([q[0]]))
        steane713.terminal_measure(q)

    result = ValidationSuite([StarRxRzGateValidation]).validate(main)
    assert result.is_valid


def test_star_rx_with_cx_is_allowed():
    @steane_kernel
    def main():
        q = squin.qalloc(2)
        squin.cx(q[0], q[1])
        extensions.star_rx(0.125, ilist.IList([q[0]]))
        steane713.terminal_measure(q)

    ValidationSuite([StarRxRzGateValidation]).validate(main).raise_if_invalid()


def test_star_rx_with_cz_is_allowed():
    @steane_kernel
    def main():
        q = squin.qalloc(2)
        squin.cz(q[0], q[1])
        extensions.star_rx(0.125, ilist.IList([q[0]]))
        steane713.terminal_measure(q)

    ValidationSuite([StarRxRzGateValidation]).validate(main).raise_if_invalid()


def test_star_rx_with_star_rz_is_allowed():
    @steane_kernel
    def main():
        q = squin.qalloc(1)
        extensions.star_rz(0.25, ilist.IList([q[0]]))
        extensions.star_rx(0.125, ilist.IList([q[0]]))
        steane713.terminal_measure(q)

    ValidationSuite([StarRxRzGateValidation]).validate(main).raise_if_invalid()


def test_h_without_star_rx_is_allowed():
    @steane_kernel
    def main():
        q = squin.qalloc(1)
        squin.h(q[0])
        steane713.terminal_measure(q)

    ValidationSuite([StarRxRzGateValidation]).validate(main).raise_if_invalid()


@pytest.mark.parametrize("known_callee", [True, False])
def test_star_rx_validation_resolves_constant_indirect_calls(known_callee):
    @steane_kernel(unroll=False)
    def rotate(q: types.Qubit):
        squin.h(q)

    @steane_kernel(unroll=False)
    def main(callback: ir.Method):
        q = squin.qalloc(1)
        rotate(q[0])
        rotate(q[0])
        extensions.star_rx(0.125, ilist.IList([q[0]]))
        steane713.terminal_measure(q)

    calls = [
        stmt
        for stmt in main.callable_region.walk()
        if isinstance(stmt, func.Invoke) and stmt.callee is rotate
    ]
    assert len(calls) == 2
    # Exercise indirect calls without depending on frontend call folding.
    # Repeated references must visit the helper only once.
    callee = py.Constant(rotate)
    callee.insert_before(calls[0])
    target = callee.result if known_callee else main.callable_region.blocks[0].args[1]
    for call in calls:
        call.replace_by(func.Call(target, call.inputs, ()))

    result = ValidationSuite([StarRxRzGateValidation]).validate(main)
    if known_callee:
        assert not result.is_valid
        (error,) = result.errors["Steane StarRx Rz Gate Validation"]
        assert "StarRx" in error.args[0] and "H" in error.args[0]
    else:
        # This pass only checks statically reachable gates. Dynamic-call
        # legality is the responsibility of the other kernel validators.
        assert result.is_valid
