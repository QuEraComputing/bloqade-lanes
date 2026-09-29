"""Validate the arguments a kernel declares.

**Why this exists.** A compiled kernel is a fixed sequence of physical
operations: the atoms it allocates, the moves it performs, and the
measurements it reads are all decided at compile time. A parameter is a value
supplied by a caller at *run* time, and there is no caller left by the time the
program reaches the hardware -- so a kernel that takes one is asking for
something the backend cannot give it.

Two shapes of that requirement show up, and this pass covers both through one
knob, the ``ALLOWED_TYPES`` class variable:

**No arguments at all** (``ALLOWED_TYPES = None``). The entry point a pipeline
compiles has nothing to be called *from*. Any parameter is unsatisfiable, and
the check is on the signature rather than on a use, because a parameter that
is never read is just as unsatisfiable as one that is.

**Arguments drawn from a fixed set** (``ALLOWED_TYPES = {...}``). A kernel meant
to be invoked from another kernel, or to be specialized before lowering, may
legitimately take arguments -- but only of the types the consumer knows how to
supply. An argument type passes when it is a *subtype* of some allowed type,
so declaring ``IList[Qubit, Literal[2]]`` satisfies an allowance of
``IList[Qubit, Any]``, while the reverse is rejected: a kernel that accepts
more than the consumer can supply is the defect this catches.

**Unannotated parameters are rejected** whenever ``Any`` itself is not
allowed. An unannotated parameter lowers to ``AnyType``, which is a subtype of
nothing but itself -- so there is no allowed type it can be shown to satisfy,
and the pass cannot vouch for it. That is a real failure rather than a
technicality, but it reads as a confusing one, so it gets its own message
naming the missing annotation instead of quoting ``AnyType()`` back at the
user.

**Why errors are anchored to ``method.code``** and not to the offending
argument: ``ir.ValidationError`` takes an ``IRNode``, and a ``BlockArgument``
is an ``SSAValue``, not an ``IRNode``. Passing one raises inside the
constructor, which ``ValidationSuite`` would swallow and re-report as a pass
crash. The parameter is named in the message instead.

One error per offending argument, so a signature with three bad parameters
reports three problems rather than stopping at the first -- the same
report-everything contract the validations next door keep. The no-arguments
case is the exception: every parameter is equally at fault and the fix is a
single edit, so it reports once and lists them.

Because ``ALLOWED_TYPES`` is a class variable rather than a constructor
argument, the parameterized variants come from :func:`get_validation`, which
builds the bound class at runtime. See its docstring for why a
``ValidationSuite`` leaves no other way in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from kirin import ir, types
from kirin.validation import ValidationPass

NO_ARGUMENTS_HELP = (
    "a compiled kernel has no caller to supply them; close over the values as "
    "compile-time constants, or compute them inside the kernel body"
)

ANNOTATE_HELP = (
    "annotate the parameter with one of the allowed types; an unannotated "
    "parameter lowers to `Any`, which cannot be shown to satisfy any of them"
)


def _render(ty: types.TypeAttribute) -> str:
    """``str`` for a type, with ``AnyType()`` spelled the way users write it."""
    return "Any" if ty == types.Any else str(ty)


def _render_allowed(allowed: frozenset[types.TypeAttribute]) -> str:
    # Sorted so the message is stable across runs -- a frozenset is not ordered.
    return ", ".join(sorted(_render(ty) for ty in allowed)) or "<nothing>"


def _argument_names(method: ir.Method) -> list[str]:
    """The declared parameter names, excluding the implicit ``self``.

    ``Method.arg_names`` carries the self name at index 0 and is typed
    optional, so fall back to the block arguments -- which always line up with
    ``Method.args`` -- when it is missing or the wrong length.
    """
    names = method.arg_names
    if names is not None and len(names) == len(method.args) + 1:
        return list(names[1:])

    return [arg.name or f"arg{i}" for i, arg in enumerate(method.args)]


@dataclass
class KernelArgumentValidation(ValidationPass):
    """Require a kernel's arguments to be drawn from ``ALLOWED_TYPES``.

    ``ALLOWED_TYPES = None`` -- the default this class carries -- means the
    kernel must declare no arguments at all, so the class is directly usable in
    a suite for that variant. :func:`get_validation` builds the parameterized
    variants. An empty frozenset has the same effect as ``None`` by a different
    route: no argument can satisfy it, so every one is reported individually.
    """

    ALLOWED_TYPES: ClassVar[frozenset[types.TypeAttribute] | None] = None

    def name(self) -> str:
        return "lanes.kernel_args.validation"

    def run(self, method: ir.Method) -> tuple[Any, list[ir.ValidationError]]:
        arguments = method.args
        if not arguments:
            return None, []

        kernel = method.sym_name or "<lambda>"
        names = _argument_names(method)
        allowed_types = self.ALLOWED_TYPES

        if allowed_types is None:
            declared = ", ".join(f"'{name}'" for name in names)
            return None, [
                ir.ValidationError(
                    method.code,
                    f"kernel '{kernel}' must take no arguments, but declares "
                    f"{len(arguments)}: {declared}",
                    help=NO_ARGUMENTS_HELP,
                )
            ]

        allowed = _render_allowed(allowed_types)
        errors: list[ir.ValidationError] = []

        for name, argument in zip(names, arguments):
            if any(argument.type.is_subseteq(ty) for ty in allowed_types):
                continue

            if argument.type == types.Any:
                errors.append(
                    ir.ValidationError(
                        method.code,
                        f"argument '{name}' of kernel '{kernel}' is unannotated, "
                        f"so its type cannot be checked against the allowed "
                        f"argument types: {allowed}",
                        help=ANNOTATE_HELP,
                    )
                )
                continue

            errors.append(
                ir.ValidationError(
                    method.code,
                    f"argument '{name}' of kernel '{kernel}' has type "
                    f"{_render(argument.type)}, which is not one of the allowed "
                    f"argument types: {allowed}",
                )
            )

        return None, errors


def get_validation(
    allowed_types: frozenset[types.TypeAttribute] | None,
) -> type[ValidationPass]:
    """Build an argument-validation pass bound to ``allowed_types``.

    ``kirin``'s :class:`ValidationSuite` instantiates each pass with no
    arguments (``pass_cls()``), so a pass cannot take ``allowed_types`` as a
    constructor argument. We close over it and stash it on a :data:`ClassVar`,
    yielding a no-arg-constructible pass class that the suite can drive
    directly -- the same trick
    :func:`bloqade.lanes.validation.address.get_validation` uses.

    The no-arguments variant needs no factory: it is what the base class
    carries, so ``ValidationSuite([KernelArgumentValidation])`` already checks
    it.
    """

    @dataclass
    class Validation(KernelArgumentValidation):
        """Validates a kernel's arguments against a fixed set of types."""

        ALLOWED_TYPES: ClassVar[frozenset[types.TypeAttribute] | None] = allowed_types

    return Validation
