from __future__ import annotations

from dataclasses import dataclass, field

import bloqade.qubit as squin_qubit
from bloqade.analysis.validation.simple_nocloning import FlatKernelNoCloningValidation
from bloqade.native.dialects import gate as native_gate
from bloqade.native.upstream.squin2native import SquinToNative
from bloqade.rewrite.passes import AggressiveUnroll
from bloqade.rewrite.passes.callgraph import CallGraphPass
from bloqade.squin.rewrite.non_clifford_to_U3 import RewriteNonCliffordToU3
from kirin import ir, passes, rewrite
from kirin.dialects.scf import scf2cf
from kirin.ir.exception import ValidationErrorGroup
from kirin.ir.method import Method
from kirin.rewrite.abc import RewriteRule
from kirin.validation import ValidationSuite

from bloqade.gemini.common.dialects import qubit as gemini_qubit
from bloqade.gemini.common.validation.duplicate_address import (
    DuplicateAddressValidation,
)
from bloqade.gemini.common.validation.terminal_measure import (
    PhysicalTerminalMeasurementValidation,
)
from bloqade.gemini.logical.rewrite.initialize import _RewriteU3ToInitialize
from bloqade.gemini.logical.rewrite.steane_transversal import (
    RewriteSteaneTransversalCliffordAdjoints,
)
from bloqade.gemini.logical.validation.clifford.analysis import GeminiLogicalValidation
from bloqade.gemini.logical.validation.measurement.analysis import (
    GeminiTerminalMeasurementValidation,
)
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.dialects import place
from bloqade.lanes.rewrite import circuit2place
from bloqade.lanes.validation.address import Validation as AddressValidation


@dataclass
class NativeToPlaceBase:
    """Template-method base for the squin-native → place compilation stage.

    Subclasses override up to three hooks; all other steps are shared:

    * ``_pre_native_rewrites(mt, out, no_raise)`` — called after ``out`` is
      created (dialect-extended copy of ``mt``) but before ``SquinToNative``.
      Default is a no-op.  Logical subclass runs ``ValidationSuite`` on ``mt``
      and applies callgraph rewrites to ``out``.

    * ``_post_unroll_validation(out)`` — called after ``AggressiveUnroll``,
      before ``ScfToCfRule``.  Default is a no-op.  Physical subclass runs
      ``PhysicalTerminalMeasurementValidation``.

    * ``_lower_qubits(out)`` — called after the optional address/duplicate
      validation block.  Must be overridden: raises ``NotImplementedError``.
      Physical subclass runs ``RewriteQubitsToPinnedQubits`` +
      ``RewritePhysicalMeasure``; logical subclass runs the four initialize
      rewrites.

    The ``arch_spec`` field controls whether post-unroll address and duplicate
    validation runs (the ``if self.arch_spec is not None`` block).  Both
    ``PhysicalPipeline`` and ``LogicalPipeline`` always supply an
    ``arch_spec`` (defaulting to their respective Gemini arch specs), so this
    validation is unconditional for both pipelines.  Set ``arch_spec=None``
    only when constructing a ``NativeToPlaceBase`` subclass directly and
    you explicitly want to skip address validation.
    """

    arch_spec: ArchSpec | None = field(default=None)

    def _pre_native_rewrites(self, mt: Method, out: Method, no_raise: bool) -> Method:
        return out

    def _post_unroll_validation(self, out: Method, no_raise: bool) -> None:
        pass

    def _lower_qubits(self, out: Method) -> None:
        raise NotImplementedError

    def emit(self, mt: Method, no_raise: bool = True) -> Method:
        out = mt.similar(mt.dialects.add(place))
        out = self._pre_native_rewrites(mt, out, no_raise)

        out = SquinToNative().emit(out, no_raise=no_raise)
        AggressiveUnroll(out.dialects, no_raise=no_raise).fixpoint(out)
        self._post_unroll_validation(out, no_raise)

        rewrite.Walk(scf2cf.ScfToCfRule()).rewrite(out.code)
        rewrite.Walk(circuit2place.HoistConstants()).rewrite(out.code)

        if self.arch_spec is not None:
            validation_errors: list[ir.ValidationError] = []
            _, per_stmt_errors = AddressValidation(arch_spec=self.arch_spec).run(out)
            validation_errors.extend(per_stmt_errors)
            _, dup_errors = DuplicateAddressValidation().run(out)
            validation_errors.extend(dup_errors)
            if validation_errors:
                raise ValidationErrorGroup(
                    f"Gemini IR validation failed with {len(validation_errors)} error(s)",
                    errors=validation_errors,
                )

        self._lower_qubits(out)

        rewrite.Walk(circuit2place.RewritePlaceOperations()).rewrite(out.code)
        rewrite.Walk(
            rewrite.Chain(
                rewrite.DeadCodeElimination(),
                rewrite.CommonSubexpressionElimination(),
            )
        ).rewrite(out.code)
        out = out.similar(
            out.dialects.discard(native_gate).discard(gemini_qubit).discard(squin_qubit)
        )
        passes.TypeInfer(out.dialects, no_raise=no_raise)(out)

        if not no_raise:
            out.verify()
            out.verify_type()

        return out


@dataclass
class NativeToPlace(NativeToPlaceBase):
    """Neutral squin -> place lowering.

    No logical-initialize rewrites and no physical pinned-qubit lowering — the
    "generic" path that reproduces the legacy ``upstream.NativeToPlace(
    logical_initialize=False)`` behavior. ``arch_spec`` defaults to ``None`` so
    the post-unroll address/duplicate validation block is skipped (as the legacy
    generic path did). Used by the entropy-trace visualizer and by callers that
    want a plain squin->place lowering.
    """

    def _lower_qubits(self, out: Method) -> None:
        rewrite.Walk(circuit2place.InitializeNewQubits()).rewrite(out.code)


@dataclass
class PhysicalNativeToPlace(NativeToPlaceBase):
    def _post_unroll_validation(self, out: Method, no_raise: bool) -> None:
        _, errors = PhysicalTerminalMeasurementValidation().run(out)
        if errors and not no_raise:
            raise ValidationErrorGroup(
                f"Physical circuit validation failed with {len(errors)} error(s)",
                errors=errors,
            )

    def _lower_qubits(self, out: Method) -> None:
        rewrite.Walk(circuit2place.RewriteQubitsToPinnedQubits()).rewrite(out.code)
        rewrite.Walk(circuit2place.RewritePhysicalMeasure()).rewrite(out.code)


@dataclass
class LogicalNativeToPlace(NativeToPlaceBase):
    transversal_rewrite: bool = False

    def _pre_native_rewrites(self, mt: Method, out: Method, no_raise: bool) -> Method:
        validator = ValidationSuite(
            [
                GeminiLogicalValidation,
                GeminiTerminalMeasurementValidation,
                FlatKernelNoCloningValidation,
            ]
        )
        result = validator.validate(mt)
        if not result.is_valid and not no_raise:
            result.raise_if_invalid()

        rules: list[RewriteRule] = []
        if self.transversal_rewrite:
            # For [[7,1,3]] Steane code, logical sqrt-X and sqrt-Z are implemented
            # as transversal sqrt-X-adj and sqrt-Z-adj, respectively.
            rules.append(rewrite.Walk(RewriteSteaneTransversalCliffordAdjoints()))
        rules += [
            rewrite.Walk(RewriteNonCliffordToU3()),
            rewrite.Walk(_RewriteU3ToInitialize()),
        ]
        CallGraphPass(mt.dialects, rewrite.Chain(*rules))(out)
        return out

    def _lower_qubits(self, out: Method) -> None:
        rewrite.Walk(circuit2place.RewriteInitializeToLogicalInitialize()).rewrite(
            out.code
        )
        rewrite.Walk(circuit2place.RewriteLogicalInitializeToNewLogical()).rewrite(
            out.code
        )
        rewrite.Walk(circuit2place.CleanUpLogicalInitialize()).rewrite(out.code)
        rewrite.Walk(circuit2place.InitializeNewQubits()).rewrite(out.code)
