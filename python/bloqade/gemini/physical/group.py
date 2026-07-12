from typing import Annotated

from bloqade.analysis import address
from bloqade.analysis.validation.simple_nocloning import FlatKernelNoCloningValidation
from bloqade.rewrite.passes import AggressiveUnroll
from bloqade.squin.rewrite import WrapAddressAnalysis
from kirin import ir, rewrite
from kirin.passes import Default
from kirin.passes.inline import InlinePass
from kirin.validation import ValidationSuite
from typing_extensions import Doc

from bloqade.gemini import common as gemini_common
from bloqade.gemini.logical.group import kernel as logical_kernel
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.dialects import arch as arch_dialect


@ir.dialect_group(logical_kernel.union([gemini_common.dialects.movement, arch_dialect]))
def kernel(self):
    """Gemini kernel with user-directed atom movement.

    Extends the logical ``kernel`` with the movement dialect (``move_to`` /
    ``permute``) and the ``bloqade.lanes.arch`` dialect (``loc`` / ``cz_partner``
    / location-attribute reads); use this when your kernel calls
    ``movement.move_to(...)`` / location primitives.
    """
    address_analysis = address.AddressAnalysis(dialects=self)

    def run_pass(
        mt,
        *,
        verify: Annotated[
            bool, Doc("run `verify` before running passes, default is `True`")
        ] = True,
        typeinfer: Annotated[
            bool,
            Doc("run type inference and apply the inferred type to IR, default `True`"),
        ] = True,
        fold: Annotated[bool, Doc("run folding passes")] = True,
        aggressive: Annotated[
            bool, Doc("run aggressive folding passes if `fold=True`")
        ] = False,
        inline: Annotated[bool, Doc("inline function calls, default `True`")] = True,
        aggressive_unroll: Annotated[
            bool,
            Doc(
                "Run aggressive inlining and unrolling pass on the IR, default `False`"
            ),
        ] = False,
        no_raise: Annotated[bool, Doc("do not raise exception during analysis")] = True,
        num_physical_qubits: Annotated[
            int, Doc("number of physical qubits per logical qubit")
        ] = 7,
        arch_spec: Annotated[
            ArchSpec | None,
            Doc(
                "architecture spec; reserved for future kernel-level checks. "
                "Address validation now runs inside PhysicalPipeline.emit."
            ),
        ] = None,
    ) -> None:
        # stop circular import problems
        from bloqade.gemini.logical.rewrite.qubit_count import InsertQubitCount

        if arch_spec is None:
            from bloqade.lanes.arch.gemini.logical import get_arch_spec

            arch_spec = get_arch_spec()

        if inline and not aggressive_unroll:
            InlinePass(mt.dialects, no_raise=no_raise).fixpoint(mt)

        if aggressive_unroll:
            AggressiveUnroll(mt.dialects, no_raise=no_raise).fixpoint(mt)
        else:
            default_pass = Default(
                self,
                verify=verify,
                fold=fold,
                aggressive=aggressive,
                typeinfer=typeinfer,
                no_raise=no_raise,
            )

            default_pass.fixpoint(mt)

        if no_raise:
            runner = address_analysis.run_no_raise
        else:
            runner = address_analysis.run

        address_frame, _ = runner(mt)

        rewrite.Walk(
            rewrite.Chain(
                WrapAddressAnalysis(address_frame.entries),
                InsertQubitCount(num_physical_qubits),
            )
        ).rewrite(mt.code)

        if verify:
            # stop circular import problems
            from bloqade.gemini.common.validation.duplicate_address import (
                DuplicateAddressValidation,
            )
            from bloqade.gemini.logical.validation.clifford.analysis import (
                GeminiLogicalValidation,
            )
            from bloqade.gemini.logical.validation.measurement.analysis import (
                GeminiTerminalMeasurementValidation,
            )

            validator = ValidationSuite(
                [
                    GeminiLogicalValidation,
                    GeminiTerminalMeasurementValidation,
                    FlatKernelNoCloningValidation,
                    DuplicateAddressValidation,
                ]
            )
            validator.validate(mt).raise_if_invalid()

            mt.verify()

    return run_pass
