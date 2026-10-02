from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from kirin import ir, passes, rewrite
from kirin.ir.exception import ValidationErrorGroup
from kirin.validation import ValidationSuite
from kirin.validation.validationpass import ValidationResult

from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.dialects.qmove import Frame
from bloqade.lanes.rewrite.native2qmove import RewriteNativeToQMove
from bloqade.lanes.rewrite.qmove_state import thread_method
from bloqade.lanes.rewrite.refine_qubit_types import RefineQubitTypes
from bloqade.lanes.transform.qmove_frontend import lower_to_native, unlisted_recursion
from bloqade.lanes.validation.qmove import get_qmove_validation
from bloqade.lanes.validation.qmove_input import get_input_validation
from bloqade.lanes.validation.spectator import SpectatorPolicy, ZonedPolicy


@dataclass
class NativeToQMove:
    """Lower a physical kernel to qmove IR, keeping ``scf`` and subroutine calls.

    ``subroutines`` maps each kernel to keep as a call to its pinned frame, or to
    ``None`` for a hole that later synthesis fills. Every other call is inlined;
    nothing is unrolled. The result is the entry method; subroutine clones are
    reachable through its ``qmove.invoke`` statements. No placement or move
    synthesis happens here.
    """

    arch_spec: ArchSpec
    subroutines: Mapping[ir.Method, Frame | None] = field(default_factory=dict)
    policy: SpectatorPolicy = field(default_factory=ZonedPolicy)

    def emit(self, mt: ir.Method, no_raise: bool = False) -> ir.Method:
        if cycles := unlisted_recursion(mt, frozenset(self.subroutines)):
            raise ValidationErrorGroup(
                "NativeToQMove: recursive kernels must be listed as subroutines",
                errors=[
                    ir.ValidationError(
                        mt.code, f"recursive kernel is not a subroutine: {cycle}"
                    )
                    for cycle in cycles
                ],
            )

        # A tuple keeps the listed order; a frozenset would order by id hash.
        program = lower_to_native(
            mt, tuple(self.subroutines), self.arch_spec, no_raise=no_raise
        )
        roles: list[tuple[ir.Method, Frame | None, bool]] = [
            (program.entry, None, False)
        ]
        roles += [
            (clone, self.subroutines[original], True)
            for original, clone in program.subroutines.items()
        ]

        clones = frozenset(program.subroutines.values())
        errors: list[ir.ValidationError] = []
        for method, _, is_subroutine in roles:
            result = ValidationSuite(
                [get_input_validation(is_subroutine, clones)]
            ).validate(method)
            errors += [err for errs in result.errors.values() for err in errs]
        if errors and not no_raise:
            raise ValidationErrorGroup(
                "NativeToQMove: unsupported input", errors=errors
            )

        for method, frame, is_subroutine in roles:
            rewrite.Walk(RewriteNativeToQMove(clones)).rewrite(method.code)
            thread_method(method, subroutine=is_subroutine, frame=frame)
            rewrite.Fixpoint(rewrite.Walk(rewrite.DeadCodeElimination())).rewrite(
                method.code
            )
            passes.TypeInfer(method.dialects, no_raise=no_raise)(method)
            RefineQubitTypes(method.dialects, no_raise=no_raise)(method)

        if not no_raise:
            # Validate every method before raising, so one run reports all problems.
            validation = get_qmove_validation(self.arch_spec, self.policy)
            merged: dict[str, list[ir.ValidationError]] = {}
            for method, _, _ in roles:
                result = ValidationSuite([validation]).validate(method)
                for name, errs in result.errors.items():
                    merged.setdefault(name, []).extend(errs)
            ValidationResult(merged).raise_if_invalid()
            for method, _, _ in roles:
                method.verify()
                method.verify_type()
        return program.entry
