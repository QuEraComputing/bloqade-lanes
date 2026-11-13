NO_RESULTS = "Measurement ID Analysis failed to produce the necessary results"
import bloqade.squin as squin
from kirin.dialects import ilist
from kirin.validation import ValidationSuite

from bloqade.lanes.dialects import steane713
from bloqade.lanes.validation.terminal_measurement import (
    SteaneTerminalMeasurementValidation,
)

from ._steane_kernel import kernel as steane_kernel


def _run(method):
    return ValidationSuite([SteaneTerminalMeasurementValidation]).validate(method)


def _messages(result):
    # `args[0] if args else str(e)` matches `test_recursion_validation`: an error
    # built without a message would otherwise raise `IndexError` here and hide
    # whatever actually went wrong.
    return [
        str(e.args[0]) if e.args else str(e)
        for errors in result.errors.values()
        for e in errors
    ]


def test_untyped_register_parameter_is_reported_not_asserted():
    """`reg` is untyped, so its type is `AnyType` rather than `IList[_, Len]`."""

    @steane_kernel
    def subroutine(reg):
        return steane713.terminal_measure(reg)

    messages = _messages(_run(subroutine))

    assert any(NO_RESULTS in m for m in messages), messages
    # The tell for the old behaviour: the suite catching a pass that raised.
    assert not any("Validation pass" in m and "failed" in m for m in messages)
    assert not any("AssertionError" in m for m in messages)


def test_uninlined_callee_measurement_is_reported_not_raised():
    """The measurement lives in a callee, so it is evaluated in a nested frame.

    `Frame.get` raises on the miss; the impl has to route that into its own
    "no usable result" branch.
    """

    @steane_kernel
    def subroutine(reg):
        return steane713.terminal_measure(reg)

    @steane_kernel(unroll=False)
    def main():
        qbs = ilist.IList([squin.qubit.new(), squin.qubit.new()])
        squin.cz(qbs[0], qbs[1])
        return subroutine(qbs)

    messages = _messages(_run(main))

    assert any(NO_RESULTS in m for m in messages), messages
    assert not any("Validation pass" in m and "failed" in m for m in messages)
    assert not any("InterpreterError" in m for m in messages)


def test_a_well_formed_kernel_still_validates():
    """The guards must not swallow the real analysis on a valid kernel."""

    @steane_kernel
    def main():
        qbs = ilist.IList([squin.qubit.new(), squin.qubit.new()])
        squin.cz(qbs[0], qbs[1])
        return steane713.terminal_measure(qbs)

    result = _run(main)
    assert result.is_valid, _messages(result)
