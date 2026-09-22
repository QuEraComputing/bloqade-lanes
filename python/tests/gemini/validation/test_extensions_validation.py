"""Tests for NoLogicalExtensionsValidation and its wiring.

The point of the `gemini.logical.extensions` namespace is that the statements in
it are usable but gated: they leave the logical subspace, so they are only
meaningful where the caller post-selects on the surrounding error-correction
checks. These pin both halves of that -- the pass rejects them, and the pass is
wired into the submission path but *not* into `@logical.kernel` or the
simulation path.
"""

import math

import pytest
from bloqade.core.device.task_builder import TaskBuilder, TaskFinalizeError
from kirin.validation import ValidationSuite

from bloqade import squin, types
from bloqade.gemini import GeminiLogicalDevice, logical
from bloqade.gemini.compile import run_squin_kernel_validation
from bloqade.gemini.logical.validation.extensions import (
    NoLogicalExtensionsValidation,
)

THETA = math.pi / 16


def _validate(method):
    return ValidationSuite([NoLogicalExtensionsValidation]).validate(method)


def _program_using_star_rz():
    @logical.kernel
    def main():
        q = squin.qalloc(1)
        squin.h(q[0])
        logical.extensions.star_rz(THETA, q[0])
        logical.terminal_measure(q)

    return main


def _program_without_extensions():
    @logical.kernel
    def main():
        q = squin.qalloc(1)
        squin.h(q[0])
        logical.terminal_measure(q)

    return main


# --- the pass itself ----------------------------------------------------------


def test_a_program_without_extensions_is_valid():
    assert _validate(_program_without_extensions()).is_valid


def test_a_program_using_star_rz_is_rejected():
    assert not _validate(_program_using_star_rz()).is_valid


def test_the_message_names_the_statement_and_the_kernel():
    result = _validate(_program_using_star_rz())
    (error,) = result.errors["Gemini Logical Extensions Validation"]

    assert "gemini.logical.extensions.star_rz" in error.args[0]
    assert "'main'" in error.args[0]


def test_every_use_is_reported_rather_than_just_the_first():
    """The report-everything contract the validations next door keep."""

    @logical.kernel
    def main():
        q = squin.qalloc(2)
        logical.extensions.star_rz(THETA, q[0])
        logical.extensions.star_rz(THETA, q[1])
        logical.terminal_measure(q)

    result = _validate(main)

    assert len(result.errors["Gemini Logical Extensions Validation"]) == 2


def test_a_use_inside_an_inlined_sub_kernel_is_caught():
    """Decoration inlines the helper, so the statement is in `main`'s own region
    by the time any suite runs -- there is no way to smuggle one in."""

    @logical.kernel
    def rotate(q: types.Qubit):
        logical.extensions.star_rz(THETA, q)

    @logical.kernel(aggressive_unroll=True)
    def main():
        q = squin.qalloc(1)
        rotate(q[0])
        logical.terminal_measure(q)

    assert not _validate(main).is_valid


# --- the wiring ---------------------------------------------------------------


def test_the_kernel_decorator_does_not_run_this_pass():
    """Gating the feature at definition time would be the opposite of gating
    it: the extensions would be unusable everywhere, including in simulation."""
    _program_using_star_rz()  # must not raise


def test_the_device_suite_rejects_a_program_using_star_rz():
    suite = GeminiLogicalDevice().validation_suite
    assert suite is not None

    result = suite.validate(_program_using_star_rz())

    assert not result.is_valid
    assert any(
        "gemini.logical.extensions.star_rz" in error.args[0]
        for errors in result.errors.values()
        for error in errors
    )


def test_the_device_suite_accepts_a_program_without_extensions():
    suite = GeminiLogicalDevice().validation_suite
    assert suite is not None

    assert suite.validate(_program_without_extensions()).is_valid


def _dry_run(method):
    """Finalize a builder through the device without submitting anything.

    This is where the device's suite actually runs: `run_async` validates and
    serializes exactly as a submission does, then returns without
    authenticating or calling QLAM.
    """
    builder = TaskBuilder()
    builder.add_subtask(method, num_shots=1)
    GeminiLogicalDevice().run_async(builder, dry_run=True)


def test_submitting_a_program_using_star_rz_is_rejected():
    """End to end: the guard fires where a user meets it, not only in a suite."""
    with pytest.raises(TaskFinalizeError, match="gemini.logical.extensions.star_rz"):
        _dry_run(_program_using_star_rz())


def test_submitting_a_program_without_extensions_is_accepted():
    _dry_run(_program_without_extensions())  # must not raise


def test_the_simulation_path_still_accepts_star_rz():
    """`GeminiLogicalSimulator` compiles through `run_squin_kernel_validation`,
    which deliberately omits this pass -- post-selecting on the detectors is
    the user's job there, and that is the supported use of the gadget."""
    assert run_squin_kernel_validation(_program_using_star_rz()).is_valid
