"""Definition-time validation for STAR-X and Rz-producing logical gates."""

import pytest
from kirin.dialects import func, ilist
from kirin.ir.exception import ValidationErrorGroup
from kirin.validation import ValidationSuite

from bloqade import squin, types
from bloqade.gemini import logical
from bloqade.gemini.logical.validation.star_rx_rz import StarRxRzGateValidation


def test_h_with_star_rx_rejected_at_definition():
    with pytest.raises(ValidationErrorGroup, match="StarRx.*H"):

        @logical.kernel()
        def main():
            q = squin.qalloc(1)
            squin.h(q[0])
            logical.extensions.star_rx(0.125, q[0])
            logical.terminal_measure(q)


def test_first_gate_rz_with_star_rx_rejected_at_definition():
    with pytest.raises(ValidationErrorGroup, match="StarRx.*Rz"):

        @logical.kernel()
        def main():
            q = squin.qalloc(1)
            squin.rz(0.25, q[0])
            logical.extensions.star_rx(0.125, q[0])
            logical.terminal_measure(q)


def test_z_with_star_rx_rejected_at_definition():
    with pytest.raises(ValidationErrorGroup, match="StarRx.*Z"):

        @logical.kernel()
        def main():
            q = squin.qalloc(1)
            squin.z(q[0])
            logical.extensions.star_rx(0.125, q[0])
            logical.terminal_measure(q)


def test_s_with_star_rx_rejected_at_definition():
    with pytest.raises(ValidationErrorGroup, match="StarRx.*S"):

        @logical.kernel()
        def main():
            q = squin.qalloc(1)
            squin.s(q[0])
            logical.extensions.star_rx(0.125, q[0])
            logical.terminal_measure(q)


def test_s_adjoint_with_star_rx_rejected_at_definition():
    with pytest.raises(ValidationErrorGroup, match="StarRx.*S"):

        @logical.kernel()
        def main():
            q = squin.qalloc(1)
            squin.s_adj(q[0])
            logical.extensions.star_rx(0.125, q[0])
            logical.terminal_measure(q)


def test_h_after_star_rx_is_still_rejected():
    with pytest.raises(ValidationErrorGroup, match="StarRx.*H"):

        @logical.kernel()
        def main():
            q = squin.qalloc(1)
            logical.extensions.star_rx(0.125, q[0])
            squin.h(q[0])
            logical.terminal_measure(q)


def test_conflicting_gate_on_different_qubit_is_still_rejected():
    with pytest.raises(ValidationErrorGroup, match="StarRx.*S"):

        @logical.kernel()
        def main():
            q = squin.qalloc(2)
            squin.s(q[1])
            logical.extensions.star_rx(0.125, q[0])
            logical.terminal_measure(q)


def test_conflict_in_non_inlined_static_helper_is_rejected():
    @logical.kernel(verify=False)
    def rotate(q: types.Qubit):
        squin.h(q)

    @logical.kernel(aggressive_unroll=False, inline=False, fold=False, verify=False)
    def main():
        q = squin.qalloc(1)
        rotate(q[0])
        logical.extensions.star_rx(0.125, q[0])
        logical.terminal_measure(q)

    result = ValidationSuite([StarRxRzGateValidation]).validate(main)
    assert not result.is_valid
    assert any(isinstance(stmt, func.Invoke) for stmt in main.callable_region.walk())
    (error,) = result.errors["Gemini Logical StarRx Rz Gate Validation"]
    assert "StarRx" in error.args[0] and "H" in error.args[0]


def test_conflict_in_static_ilist_map_helper_is_rejected():
    @logical.kernel(verify=False)
    def rotate(q: types.Qubit):
        squin.h(q)
        return q

    @logical.kernel(aggressive_unroll=False, inline=False, fold=False, verify=False)
    def main():
        q = squin.qalloc(1)
        ilist.map(rotate, q)
        logical.extensions.star_rx(0.125, q[0])
        logical.terminal_measure(q)

    assert any(isinstance(stmt, ilist.Map) for stmt in main.callable_region.walk())
    result = ValidationSuite([StarRxRzGateValidation]).validate(main)
    assert not result.is_valid
    (error,) = result.errors["Gemini Logical StarRx Rz Gate Validation"]
    assert "StarRx" in error.args[0] and "H" in error.args[0]


def test_stored_but_never_called_helper_does_not_conflict():
    @logical.kernel(verify=False)
    def rotate(q: types.Qubit):
        squin.h(q)

    @logical.kernel(aggressive_unroll=False, inline=False, fold=False, verify=False)
    def main():
        q = squin.qalloc(1)
        _callbacks = (rotate,)
        logical.extensions.star_rx(0.125, q[0])
        logical.terminal_measure(q)

    result = ValidationSuite([StarRxRzGateValidation]).validate(main)
    assert result.is_valid


def test_star_rx_with_cx_is_allowed():
    @logical.kernel()
    def main():
        q = squin.qalloc(2)
        squin.cx(q[0], q[1])
        logical.extensions.star_rx(0.125, q[0])
        logical.terminal_measure(q)

    assert main is not None


def test_star_rx_with_cz_is_allowed():
    @logical.kernel()
    def main():
        q = squin.qalloc(2)
        squin.cz(q[0], q[1])
        logical.extensions.star_rx(0.125, q[0])
        logical.terminal_measure(q)

    assert main is not None


def test_star_rx_with_star_rz_is_allowed():
    @logical.kernel()
    def main():
        q = squin.qalloc(1)
        logical.extensions.star_rz(0.25, q[0])
        logical.extensions.star_rx(0.125, q[0])
        logical.terminal_measure(q)

    assert main is not None


def test_h_without_star_rx_is_allowed():
    @logical.kernel()
    def main():
        q = squin.qalloc(1)
        squin.h(q[0])
        logical.terminal_measure(q)

    assert main is not None


def test_verify_false_skips_star_rx_gate_validation():
    @logical.kernel(verify=False)
    def main():
        q = squin.qalloc(1)
        squin.h(q[0])
        logical.extensions.star_rx(0.125, q[0])
        logical.terminal_measure(q)

    assert main is not None
