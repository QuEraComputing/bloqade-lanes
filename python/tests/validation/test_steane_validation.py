import bloqade.squin as squin
import pytest
from bloqade.analysis.address import AddressAnalysis
from bloqade.types import Qubit
from kirin.dialects import func, ilist
from kirin.ir.exception import ValidationErrorGroup
from kirin.validation import ValidationSuite

from bloqade.lanes.dialects import steane713
from bloqade.lanes.dialects.steane713 import extensions
from bloqade.lanes.validation.steane import (
    SteaneLogicalValidation,
    _SteaneLogicalValidationAnalysis,
)
from bloqade.lanes.validation.terminal_measurement import (
    SteaneTerminalMeasurementValidation,
    _SteaneTerminalMeasurementValidationAnalysis,
)

from ._steane_kernel import kernel as steane_kernel


def test_if_stmt_invalid():
    @steane_kernel
    def main():
        q = squin.qalloc(3)

        squin.h(q[0])

        for i in range(10):
            squin.x(q[1])

        m = squin.qubit.measure(q[1])

        q2 = squin.qalloc(5)
        squin.x(q2[0])

        if m:
            squin.x(q[1])

        m2 = squin.qubit.measure(q[2])
        if m2:
            squin.y(q[2])

    addr_frame, _ = AddressAnalysis(main.dialects).run(main)
    frame, _ = _SteaneLogicalValidationAnalysis(
        main.dialects, addr_frame=addr_frame
    ).run_no_raise(main)

    main.print(analysis=frame.entries)

    validator = ValidationSuite([SteaneLogicalValidation])
    validation_result = validator.validate(main)

    with pytest.raises(ValidationErrorGroup):
        validation_result.raise_if_invalid()


def test_qalloc_and_terminal_measure_type_valid():
    @steane_kernel
    def main():
        q = squin.qalloc(3)
        steane713.terminal_measure(q)

    validator = ValidationSuite([SteaneTerminalMeasurementValidation])
    validation_result = validator.validate(main)

    validation_result.raise_if_invalid()


def test_missing_terminal_measurement_is_invalid():
    @steane_kernel
    def main():
        q = squin.qalloc(1)
        squin.x(q[0])

    validator = ValidationSuite([SteaneTerminalMeasurementValidation])

    with pytest.raises(
        ValidationErrorGroup,
        match="must contain exactly one.*terminal_measure",
    ):
        validator.validate(main).raise_if_invalid()


def test_terminal_measurement():
    @steane_kernel
    def not_all_qubits_consumed():
        qs = squin.qalloc(3)
        sub_qs = qs[0:2]
        tm = steane713.terminal_measure(sub_qs)
        return tm

    validator = ValidationSuite([SteaneTerminalMeasurementValidation])
    validation_result = validator.validate(not_all_qubits_consumed)

    with pytest.raises(ValidationErrorGroup):
        validation_result.raise_if_invalid()

    @steane_kernel
    def terminal_measure_kernel(q):
        return steane713.terminal_measure(q)

    @steane_kernel
    def terminal_measure_in_kernel():
        q = squin.qalloc(10)
        sub_qs = q[:2]
        m = terminal_measure_kernel(sub_qs)
        return m

    validator = ValidationSuite([SteaneTerminalMeasurementValidation])
    validation_result = validator.validate(terminal_measure_in_kernel)

    with pytest.raises(ValidationErrorGroup):
        validation_result.raise_if_invalid()


def test_dynamic_call_through_parameter_does_not_crash_validation():
    """A dynamic `func.Call` must not blow up the terminal measurement pass.

    `measurement.impls.Func` subclasses the address analysis' `func` method
    table, so the measurement analysis *inherits* a `func.Call` impl that calls
    `run_lattice` on the interpreter. Without the override on
    `_SteaneTerminalMeasurementValidationAnalysis`, a kernel carrying a dynamic
    call into this pass fails with
    `AttributeError: ... has no attribute 'run_lattice'`.

    Built with `squin.kernel`, not `gemini.logical.kernel`: the latter now
    rejects a call through a parameter outright (`gemini.no_opaque_call`), so a
    Gemini kernel can no longer carry a dynamic call this far. A kernel from
    another dialect group still can, and that is the case the override has to
    survive. No other kernel in the suite performs a dynamic call, so nothing
    else covers this.
    """
    from bloqade.analysis.address.impls import Func as AddressFuncMethodTable

    from bloqade.lanes.validation.terminal_measurement import (
        Func as MeasurementFuncMethodTable,
    )

    # The coupling is what makes the override mandatory. If this stops holding,
    # re-check whether the override is still needed before removing it.
    assert issubclass(MeasurementFuncMethodTable, AddressFuncMethodTable)

    @squin.kernel
    def higher_order(f):
        f()

    # Guard the premise: the point of the test is a dynamic call reaching the
    # pass, which is silently lost if lowering ever folds this one away.
    assert any(isinstance(stmt, func.Call) for stmt in higher_order.code.walk())

    validator = ValidationSuite([SteaneTerminalMeasurementValidation])
    validator.validate(higher_order).raise_if_invalid()


def test_every_gate_after_terminal_measurement_is_reported():
    @steane_kernel
    def main():
        q = squin.qalloc(2)
        squin.h(q[0])
        steane713.terminal_measure(q)
        squin.x(q[0])
        squin.cz(q[0], q[1])
        extensions.star_rz(0.125, ilist.IList([q[1]]))

    validation_result = ValidationSuite([SteaneTerminalMeasurementValidation]).validate(
        main
    )
    errors = [str(e) for errs in validation_result.errors.values() for e in errs]
    assert len(errors) == 3, errors
    assert all("after the terminal measurement" in e for e in errors)
    assert any("Gate x " in e for e in errors)
    assert any("Gate cz " in e for e in errors)
    assert any("Gate star_rz " in e for e in errors)


def test_gate_after_terminal_measurement_in_non_inlined_callee_is_invalid():
    @steane_kernel(unroll=False)
    def apply_x(q: Qubit):
        squin.x(q)

    @steane_kernel(unroll=False)
    def main():
        q = squin.qalloc(1)
        steane713.terminal_measure(q)
        apply_x(q[0])

    # Guard the premise: the gate must still sit behind a call.
    assert any(isinstance(stmt, func.Invoke) for stmt in main.code.walk())

    with pytest.raises(ValidationErrorGroup, match="Gate x is applied after"):
        ValidationSuite([SteaneTerminalMeasurementValidation]).validate(
            main
        ).raise_if_invalid()


def test_every_gate_statement_is_checked_after_terminal_measurement():
    """A gate missing from the impl table would be accepted after measurement."""
    from bloqade.squin import gate

    registry = steane_kernel.registry.interpreter(
        keys=_SteaneTerminalMeasurementValidationAnalysis.keys
    )
    covered = {sig.head for sig in registry}
    missing = {stmt.__name__ for stmt in gate.dialect.stmts if stmt not in covered}
    assert not missing, f"gates without a post-measurement check: {missing}"


def test_every_extension_statement_is_checked_after_terminal_measurement():
    """The same guard for `gemini.logical.extensions`.

    The extension statements act on qubits, so applying one after the terminal
    measurement is the same mistake a gate is -- but they are a separate dialect,
    so the sweep above does not reach them. Without this, a second extension
    would be silently accepted after the measurement.
    """
    from bloqade.lanes.dialects.steane713 import extensions as extensions

    registry = steane_kernel.registry.interpreter(
        keys=_SteaneTerminalMeasurementValidationAnalysis.keys
    )
    covered = {sig.head for sig in registry}
    missing = {
        stmt.__name__ for stmt in extensions.dialect.stmts if stmt not in covered
    }
    assert not missing, f"extensions without a post-measurement check: {missing}"


@pytest.mark.parametrize("prior_operation", ["none", "h", "swap", "star_rz"])
def test_non_clifford_allowed_only_for_initial_preparation(prior_operation):
    @steane_kernel
    def main():
        q = squin.qalloc(2)
        if prior_operation == "h":
            squin.h(q[1])
        elif prior_operation == "swap":
            squin.swap(q[0], q[1])
        elif prior_operation == "star_rz":
            extensions.star_rz(0.125, ilist.IList([q[1]]))
        squin.u3(0.1, 0.2, 0.3, q[1])
        steane713.terminal_measure(q)

    _, errors = SteaneLogicalValidation().run(main)
    if prior_operation == "none":
        assert errors == []
    else:
        assert len(errors) == 1
        assert "Non-clifford gate u3" in str(errors[0])


def test_dynamic_loop_is_reported():
    @steane_kernel(unroll=False)
    def main(n: int):
        q = squin.qalloc(1)
        for _ in range(n):
            squin.x(q[0])
        steane713.terminal_measure(q)

    _, errors = SteaneLogicalValidation().run(main)
    assert any("Non-constant iterable" in str(error) for error in errors)


def test_allocation_over_capacity_is_reported():
    @steane_kernel
    def main():
        q = squin.qalloc(11)
        steane713.terminal_measure(q)

    _, errors = SteaneLogicalValidation().run(main)
    assert any("Qubit allocations exceeded 10" in str(error) for error in errors)
    assert any("kernel allocates 11 qubits" in str(error) for error in errors)


def test_repeated_terminal_measurement_is_reported():
    @steane_kernel
    def main():
        q = squin.qalloc(1)
        steane713.terminal_measure(q)
        return steane713.terminal_measure(q)

    result = ValidationSuite([SteaneTerminalMeasurementValidation]).validate(main)
    with pytest.raises(ValidationErrorGroup, match="Multiple terminal measurements"):
        result.raise_if_invalid()


def test_non_terminal_measurement_is_reported():
    @steane_kernel
    def main():
        q = squin.qalloc(1)
        squin.measure(q[0])
        return steane713.terminal_measure(q)

    result = ValidationSuite([SteaneTerminalMeasurementValidation]).validate(main)
    with pytest.raises(ValidationErrorGroup, match="Non-terminal measurements"):
        result.raise_if_invalid()
