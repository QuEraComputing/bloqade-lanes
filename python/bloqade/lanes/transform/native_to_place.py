from __future__ import annotations

from dataclasses import dataclass, field

import bloqade.qubit as squin_qubit
from bloqade.analysis.validation.simple_nocloning import FlatKernelNoCloningValidation
from bloqade.native.dialects import gate as native_gate
from bloqade.native.upstream.squin2native import SquinToNative
from bloqade.rewrite.passes import AggressiveUnroll
from bloqade.rewrite.passes.callgraph import CallGraphPass
from bloqade.squin.rewrite.non_clifford_to_U3 import RewriteNonCliffordToU3
from kirin import passes, rewrite
from kirin.dialects.scf import scf2cf
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
from bloqade.lanes.dialects.arch import BindArchSpec
from bloqade.lanes.rewrite import circuit2place, clifford2native
from bloqade.lanes.utils import raise_if_statements_outside_dialect_group
from bloqade.lanes.validation.address import get_validation


@dataclass
class NativeToPlaceBase:
    """Template-method base for the squin-native → place compilation stage.

    Subclasses override up to four hooks; all other steps are shared:

    * ``_pre_native_rewrites(mt, out, no_raise)`` — called after ``out`` is
      created (dialect-extended copy of ``mt``) but before ``SquinToNative``.
      Default is a no-op.  Logical subclass runs ``ValidationSuite`` on ``mt``
      and applies callgraph rewrites to ``out``.

    * ``_squin_clifford_rules()`` — squin→squin rules run over the call graph
      as the last step before ``SquinToNative``.  The base implementation
      decomposes composite Cliffords into the neutral-atom gate set; the
      logical subclass appends the Steane transversal adjoint swap.

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

    def _squin_clifford_rules(self) -> list[RewriteRule]:
        """Squin→squin rules applied just before ``SquinToNative``.

        Composite Cliffords (H, CX, CY, Swap) are expanded here rather than
        inside the ``bloqade.native`` stdlib kernels, so that any rule appended
        by a subclass sees the individual sqrt(X)/sqrt(Y)/S layers the hardware
        will run instead of the opaque composite statement.
        """
        return [rewrite.Walk(clifford2native.DecomposeCliffordToNative())]

    def _post_unroll_validation(self, out: Method, no_raise: bool) -> None:
        pass

    def _lower_qubits(self, out: Method) -> None:
        raise NotImplementedError

    def emit(self, mt: Method, no_raise: bool = True) -> Method:
        out = mt.similar(mt.dialects.add(place))
        out = self._pre_native_rewrites(mt, out, no_raise)

        if self.arch_spec is not None:
            # Bind arch_spec on every arch-resolved statement (Loc, CzPartner) reachable
            # so const-prop resolves them during AggressiveUnroll.
            CallGraphPass(out.dialects, rewrite.Walk(BindArchSpec(self.arch_spec)))(out)

        if squin_clifford_rules := self._squin_clifford_rules():
            CallGraphPass(
                out.dialects,
                rewrite.Chain(*squin_clifford_rules),
                no_raise=no_raise,
            )(out)

        out = SquinToNative().emit(out, no_raise=no_raise)
        AggressiveUnroll(out.dialects, no_raise=no_raise).fixpoint(out)

        self._post_unroll_validation(out, no_raise)

        rewrite.Walk(scf2cf.ScfToCfRule()).rewrite(out.code)
        rewrite.Walk(circuit2place.HoistConstants()).rewrite(out.code)

        if self.arch_spec is not None:
            suite = ValidationSuite(
                [DuplicateAddressValidation, get_validation(self.arch_spec)]
            )
            suite.validate(out).raise_if_invalid()

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
            # verify() does not police dialect-group membership, so a gate or
            # qubit statement the rewrites above missed would slip through the
            # discard() and only fail lazily downstream. Check it explicitly.
            raise_if_statements_outside_dialect_group(out, type(self).__name__)
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
        if no_raise:
            return
        suite = ValidationSuite([PhysicalTerminalMeasurementValidation])
        suite.validate(out).raise_if_invalid()

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

        rules: list[RewriteRule] = [
            rewrite.Walk(RewriteNonCliffordToU3()),
            rewrite.Walk(_RewriteU3ToInitialize()),
        ]
        CallGraphPass(mt.dialects, rewrite.Chain(*rules))(out)
        return out

    def _squin_clifford_rules(self) -> list[RewriteRule]:
        rules = super()._squin_clifford_rules()
        if self.transversal_rewrite:
            # For [[7,1,3]] Steane code, logical sqrt-X and sqrt-Z are implemented
            # as transversal sqrt-X-adj and sqrt-Z-adj, respectively.
            #
            # This has to run *after* the Clifford decomposition above, not in
            # _pre_native_rewrites: squin.cy's two sqrt(X) layers do not exist
            # as statements until CY has been expanded, so flipping earlier
            # leaves logical CY as Zbar(control) . CY
            # (QuEraComputing/bloqade-internal#404).
            rules.append(rewrite.Walk(RewriteSteaneTransversalCliffordAdjoints()))
        return rules

    def _lower_qubits(self, out: Method) -> None:
        rewrite.Walk(circuit2place.RewriteInitializeToLogicalInitialize()).rewrite(
            out.code
        )
        rewrite.Walk(circuit2place.RewriteLogicalInitializeToNewLogical()).rewrite(
            out.code
        )
        rewrite.Walk(circuit2place.CleanUpLogicalInitialize()).rewrite(out.code)
        rewrite.Walk(circuit2place.InitializeNewQubits()).rewrite(out.code)
