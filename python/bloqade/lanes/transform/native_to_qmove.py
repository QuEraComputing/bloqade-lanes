from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from kirin import ir, passes, rewrite
from kirin.ir.exception import ValidationErrorGroup
from kirin.validation import ValidationSuite
from kirin.validation.validationpass import ValidationResult

from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.dialects.qmove import Frame, MachineFrame
from bloqade.lanes.rewrite.native2qmove import RewriteNativeToQMove
from bloqade.lanes.rewrite.qmove_state import thread_method
from bloqade.lanes.rewrite.refine_qubit_types import RefineQubitTypes
from bloqade.lanes.transform.qmove_frontend import (
    lower_to_native,
    recursive_allocation,
    unlisted_recursion,
)
from bloqade.lanes.validation.qmove import get_qmove_validation
from bloqade.lanes.validation.qmove_input import get_input_validation
from bloqade.lanes.validation.spectator import SpectatorPolicy, ZonedPolicy

RECURSIVE_ALLOCATION_MESSAGE = (
    "kirin's constant propagation does not support allocation through a "
    "function value, such as squin.qalloc, inside recursion; allocate outside "
    "the recursion and pass the qubits in, or use qubit.new()"
)


@dataclass
class NativeToQMove:
    """Lower a physical kernel to qmove IR, keeping ``scf`` and subroutine calls.

    ``subroutines`` maps each kernel to keep as a call to its pinned frame: a
    partial ``Frame``, ``MachineFrame()`` for a whole-machine subroutine (which
    may allocate, and only whole-machine methods may call), or ``None`` for a
    hole that later synthesis fills. The entry kernel always gets
    ``MachineFrame()``. Every other call is inlined; nothing is unrolled. The
    result is the entry method; subroutine clones are reachable through its
    ``qmove.invoke`` statements. No placement or move synthesis happens here.
    """

    arch_spec: ArchSpec
    subroutines: Mapping[ir.Method, Frame | MachineFrame | None] = field(
        default_factory=dict
    )
    policy: SpectatorPolicy = field(default_factory=ZonedPolicy)

    def emit(self, mt: ir.Method, no_raise: bool = False) -> ir.Method:
        recursion = [
            f"recursive kernel is not a subroutine: {cycle}"
            for cycle in unlisted_recursion(mt, frozenset(self.subroutines))
        ]
        if not no_raise:
            # Under no_raise, kirin's HintConst swallows the error instead and
            # drops the method's constant hints.
            recursion += [
                f"{cycle}: {RECURSIVE_ALLOCATION_MESSAGE}"
                for cycle in recursive_allocation(mt, tuple(self.subroutines))
            ]
        if recursion:
            raise ValidationErrorGroup(
                "NativeToQMove: unsupported recursion",
                errors=[ir.ValidationError(mt.code, msg) for msg in recursion],
            )

        # A tuple keeps the listed order; a frozenset would order by id hash.
        program = lower_to_native(
            mt, tuple(self.subroutines), self.arch_spec, no_raise=no_raise
        )
        roles: list[tuple[ir.Method, Frame | MachineFrame | None]] = [
            (program.entry, MachineFrame())
        ]
        roles += [
            (clone, self.subroutines[original])
            for original, clone in program.subroutines.items()
        ]

        clones = frozenset(program.subroutines.values())
        errors: list[ir.ValidationError] = []
        for method, frame in roles:
            may_allocate = isinstance(frame, MachineFrame)
            result = ValidationSuite(
                [get_input_validation(may_allocate, clones)]
            ).validate(method)
            errors += [err for errs in result.errors.values() for err in errs]
        if errors and not no_raise:
            raise ValidationErrorGroup(
                "NativeToQMove: unsupported input", errors=errors
            )

        for method, frame in roles:
            rewrite.Walk(RewriteNativeToQMove(clones)).rewrite(method.code)
            thread_method(method, frame=frame)
            rewrite.Fixpoint(rewrite.Walk(rewrite.DeadCodeElimination())).rewrite(
                method.code
            )
            passes.TypeInfer(method.dialects, no_raise=no_raise)(method)
            RefineQubitTypes(method.dialects, no_raise=no_raise)(method)

        if not no_raise:
            # Validate every method before raising, so one run reports all problems.
            validation = get_qmove_validation(self.arch_spec, self.policy)
            merged: dict[str, list[ir.ValidationError]] = {}
            for method, _ in roles:
                result = ValidationSuite([validation]).validate(method)
                for name, errs in result.errors.items():
                    merged.setdefault(name, []).extend(errs)
            ValidationResult(merged).raise_if_invalid()
            for method, _ in roles:
                method.verify()
                method.verify_type()
        return program.entry
