"""Tests for GeminiLogicalArgumentValidation and its wiring into the group.

The pass is scoped to *programs* -- kernels that allocate their own qubits --
because `gemini.logical.kernel` decorates sub-kernels too, and a sub-kernel
taking arguments is a supported pattern. These pin both sides of that line.
"""

import pytest
from kirin.ir.exception import ValidationErrorGroup
from kirin.validation import ValidationSuite

from bloqade import squin
from bloqade.gemini import logical
from bloqade.gemini.logical.validation.arguments import (
    GeminiLogicalArgumentValidation,
)


def _validate(method):
    return ValidationSuite([GeminiLogicalArgumentValidation]).validate(method)


# --- programs -----------------------------------------------------------------


def test_a_program_taking_no_arguments_is_valid():
    @logical.kernel
    def main():
        q = squin.qalloc(2)
        squin.h(q[0])
        logical.terminal_measure(q)

    assert _validate(main).is_valid


def test_a_program_taking_an_argument_is_rejected_at_definition():
    """The group wiring: this raises out of the decorator, not at run time."""
    with pytest.raises(ValidationErrorGroup, match="must take no arguments"):

        @logical.kernel
        def main(n: int):
            q = squin.qalloc(2)
            squin.h(q[0])
            logical.terminal_measure(q)


def test_the_message_names_the_offending_parameter():
    @logical.kernel(verify=False)
    def main(theta: float):
        q = squin.qalloc(2)
        squin.h(q[0])
        logical.terminal_measure(q)

    result = _validate(main)
    (error,) = result.errors["Gemini Logical Argument Validation"]

    assert "'theta'" in error.args[0]
    assert "'main'" in error.args[0]


# --- sub-kernels --------------------------------------------------------------


def test_a_sub_kernel_may_take_arguments():
    """It is handed its qubits, so it is not a program -- see the module docstring."""

    @logical.kernel
    def flip(q):
        squin.x(q)

    assert _validate(flip).is_valid


def test_a_sub_kernel_with_arguments_still_compiles_into_a_program():
    """The pattern end to end: the helper is inlined, the program takes nothing."""

    @logical.kernel
    def flip(q):
        squin.x(q)

    @logical.kernel(aggressive_unroll=True)
    def main():
        q = squin.qalloc(2)
        flip(q[0])
        flip(q[1])
        logical.terminal_measure(q)

    assert _validate(main).is_valid


def test_a_sub_kernel_that_allocates_is_treated_as_a_program():
    """Allocation is the line. A helper that allocates is a program by this
    suite's definition -- the same one that makes it owe a terminal measure."""

    @logical.kernel(verify=False)
    def allocates(n: int):
        q = squin.qalloc(2)
        squin.h(q[0])
        logical.terminal_measure(q)

    assert not _validate(allocates).is_valid
