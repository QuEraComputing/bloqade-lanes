"""Tests for KernelArgumentValidation."""

import typing

from kirin import types as kirin_types
from kirin.dialects import ilist
from kirin.validation import ValidationSuite

from bloqade import qubit, squin, types as bloqade_types
from bloqade.lanes.validation.kernel_args import (
    KernelArgumentValidation,
    get_validation,
)

QUBIT = bloqade_types.QubitType
QUBIT_LIST = ilist.IListType[QUBIT, kirin_types.Any]


def _messages(
    method, allowed_types: frozenset[kirin_types.TypeAttribute] | None
) -> list[str]:
    """Run a pass bound to ``allowed_types`` and return one message per error."""
    _, errors = get_validation(allowed_types)().run(method)
    return [error.args[0] for error in errors]


# --- the no-arguments variant -------------------------------------------------


def test_a_kernel_with_no_arguments_is_valid():
    @squin.kernel
    def no_args():
        return None

    assert _messages(no_args, None) == []


def test_an_argument_is_rejected_when_none_are_allowed():
    @squin.kernel
    def one_arg(a: int):
        return None

    (message,) = _messages(one_arg, None)
    assert "must take no arguments" in message
    assert "'a'" in message


def test_every_argument_is_named_in_the_single_no_args_error():
    """One edit fixes the whole signature, so it reports once and lists them."""

    @squin.kernel
    def three_args(a: int, b: float, c: int):
        return None

    (message,) = _messages(three_args, None)
    assert "declares 3" in message
    assert "'a', 'b', 'c'" in message


def test_the_suite_checks_the_no_args_variant_by_default():
    """`ValidationSuite` calls `pass_cls()`, which is the no-arguments case."""

    @squin.kernel
    def one_arg(a: int):
        return None

    result = ValidationSuite([KernelArgumentValidation]).validate(one_arg)

    assert not result.is_valid
    assert result.error_count() == 1


# --- the allowed-types variant ------------------------------------------------


def test_an_allowed_argument_type_is_valid():
    @squin.kernel
    def takes_int(a: int):
        return None

    assert _messages(takes_int, frozenset({kirin_types.Int})) == []


def test_a_disallowed_argument_type_is_rejected():
    @squin.kernel
    def takes_float(a: float):
        return None

    (message,) = _messages(takes_float, frozenset({kirin_types.Int}))
    assert "argument 'a'" in message
    assert "has type float" in message
    assert "int" in message


def test_every_offending_argument_is_reported():
    """Compiler-style: show all the problems, not just the first."""

    @squin.kernel
    def mixed(a: int, b: float, c: str):
        return None

    messages = _messages(mixed, frozenset({kirin_types.Int}))

    assert len(messages) == 2
    assert "argument 'b'" in messages[0]
    assert "argument 'c'" in messages[1]


def test_a_subtype_of_an_allowed_type_is_valid():
    """`int` satisfies an allowance of `Number`; the check is subtyping."""

    @squin.kernel
    def takes_int(a: int):
        return None

    assert _messages(takes_int, frozenset({kirin_types.Number})) == []


def test_a_supertype_of_an_allowed_type_is_rejected():
    """The reverse direction: the kernel accepts more than the consumer supplies.

    `int | str` is a strict supertype of `int`, so a consumer that only ever
    supplies ints would still be handing this kernel a signature it cannot
    honour in full. Subtyping is directional on purpose.
    """

    @squin.kernel
    def takes_either(a: int | str):
        return None

    assert _messages(takes_either, frozenset({kirin_types.Int})) != []


def test_a_qubit_register_satisfies_a_generic_allowance():
    """The shape that matters here: a concrete list length under `IList[_, Any]`."""

    @squin.kernel
    def takes_register(reg: ilist.IList[qubit.Qubit, typing.Literal[2]]):
        return None

    assert _messages(takes_register, frozenset({QUBIT_LIST})) == []


def test_any_of_several_allowed_types_is_enough():
    @squin.kernel
    def mixed(a: int, b: float):
        return None

    allowed = frozenset({kirin_types.Int, kirin_types.Float})

    assert _messages(mixed, allowed) == []


def test_an_unannotated_argument_is_rejected_with_its_own_message():
    """It lowers to `Any`, a subtype of nothing -- say so, don't quote AnyType()."""

    @squin.kernel
    def unannotated(a):
        return None

    (message,) = _messages(unannotated, frozenset({kirin_types.Int}))
    assert "is unannotated" in message
    assert "AnyType" not in message


def test_an_unannotated_argument_is_valid_when_any_is_allowed():
    @squin.kernel
    def unannotated(a):
        return None

    assert _messages(unannotated, frozenset({kirin_types.Any})) == []


def test_an_empty_allowance_rejects_every_argument_individually():
    """Same effect as `None`, reached by a different route: one error per arg."""

    @squin.kernel
    def two_args(a: int, b: float):
        return None

    messages = _messages(two_args, frozenset())

    assert len(messages) == 2
    assert all("<nothing>" in message for message in messages)


def test_an_empty_allowance_still_passes_a_kernel_with_no_arguments():
    @squin.kernel
    def no_args():
        return None

    assert _messages(no_args, frozenset()) == []


# --- the factory --------------------------------------------------------------


def test_the_factory_binds_allowed_types_for_a_suite():
    """`ValidationSuite` constructs with no arguments, so the set must be bound."""

    @squin.kernel
    def takes_float(a: float):
        return None

    validation = get_validation(frozenset({kirin_types.Int}))
    result = ValidationSuite([validation]).validate(takes_float)

    assert not result.is_valid
    assert result.error_count() == 1


def test_the_factory_leaves_the_base_class_alone():
    """A bound subclass must not leak its allowance onto the shared base.

    `ALLOWED_TYPES` is a class variable, so a factory that assigned it in place
    rather than on a fresh subclass would silently rewire every other suite
    using the no-arguments default.
    """
    get_validation(frozenset({kirin_types.Int}))

    assert KernelArgumentValidation.ALLOWED_TYPES is None


def test_the_factory_accepts_a_valid_kernel():
    @squin.kernel
    def takes_int(a: int):
        return None

    validation = get_validation(frozenset({kirin_types.Int}))

    assert ValidationSuite([validation]).validate(takes_int).is_valid


def test_the_factory_also_takes_none_for_the_no_args_variant():
    @squin.kernel
    def one_arg(a: int):
        return None

    validation = get_validation(None)
    result = ValidationSuite([validation]).validate(one_arg)

    assert not result.is_valid
    assert "must take no arguments" in result.errors[validation().name()][0].args[0]


def test_the_pass_reports_errors_the_suite_can_attach():
    """`ValidationSuite` calls `err.attach(method)`, which needs a real IRNode.

    A `BlockArgument` is an `SSAValue`, not an `IRNode`; anchoring there would
    raise inside the error constructor and surface as a pass crash instead of a
    validation failure.
    """

    @squin.kernel
    def one_arg(a: int):
        return None

    result = ValidationSuite([KernelArgumentValidation]).validate(one_arg)
    (error,) = result.errors["lanes.kernel_args.validation"]

    assert error.method is one_arg
    assert "Validation pass" not in error.args[0]
