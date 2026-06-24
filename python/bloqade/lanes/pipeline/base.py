from __future__ import annotations

from dataclasses import dataclass, field

import bloqade.qubit as squin_qubit
from bloqade.analysis import address
from bloqade.native.dialects import gate as native_gate
from bloqade.native.upstream.squin2native import SquinToNative
from bloqade.rewrite.passes import AggressiveUnroll
from bloqade.rewrite.passes.aggressive_unroll import Fold
from kirin import passes, rewrite
from kirin.dialects.scf import scf2cf
from kirin.ir.method import Method
from kirin.rewrite.abc import RewriteRule
from kirin.validation import ValidationSuite

from bloqade.gemini.common.dialects import qubit as gemini_qubit
from bloqade.gemini.common.dialects.movement.rewrite import ResolveCzPartner
from bloqade.gemini.common.validation.duplicate_address import (
    DuplicateAddressValidation,
)
from bloqade.lanes.analysis import layout, placement
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.dialects import move, place
from bloqade.lanes.rewrite import circuit2place, place2move, resolve_pinned, state
from bloqade.lanes.validation.address import get_validation


@dataclass
class _NativeToPlaceBase:
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
    only when constructing a ``_NativeToPlaceBase`` subclass directly and
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

        if self.arch_spec is not None:
            # Resolve movement.cz_partner against the arch spec, then re-fold so
            # the resulting constant locations propagate into move_to lists
            # before they are validated / lowered.
            rewrite.Walk(ResolveCzPartner(self.arch_spec)).rewrite(out.code)
            Fold(out.dialects, no_raise=no_raise)(out)

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
        passes.TypeInfer(out.dialects, no_raise=True)(out)

        if not no_raise:
            out.verify()
            out.verify_type()

        return out


@dataclass
class _PlaceToMove:
    """Shared place → move compilation stage for both pipelines.

    The only difference between the physical and logical pipelines at this
    stage is whether ``InsertInitialize`` is included in the rewrite rules.
    Pass ``insert_initialize=True`` for the logical pipeline.
    """

    layout_heuristic: layout.LayoutHeuristicABC
    placement_strategy: placement.PlacementStrategyABC
    insert_initialize: bool = False

    def emit(self, mt: Method, no_raise: bool = True) -> Method:
        out = mt.similar(mt.dialects.add(move))

        address_analysis = address.AddressAnalysis(out.dialects)
        if no_raise:
            address_frame, _ = address_analysis.run_no_raise(out)
            all_qubits = tuple(range(address_analysis.next_address))
            initial_layout = layout.LayoutAnalysis(
                out.dialects,
                self.layout_heuristic,
                address_frame.entries,
                all_qubits,
            ).get_layout_no_raise(out)
        else:
            address_frame, _ = address_analysis.run(out)
            all_qubits = tuple(range(address_analysis.next_address))
            initial_layout = layout.LayoutAnalysis(
                out.dialects,
                self.layout_heuristic,
                address_frame.entries,
                all_qubits,
            ).get_layout(out)

        rewrite.Walk(
            resolve_pinned.ResolvePinnedAddresses(
                address_entries=address_frame.entries,
                initial_layout=initial_layout,
            )
        ).rewrite(out.code)

        placement_analysis = placement.PlacementAnalysis(
            out.dialects,
            initial_layout,
            address_frame.entries,
            self.placement_strategy,
        )
        if no_raise:
            placement_frame, _ = placement_analysis.run_no_raise(out)
        else:
            placement_frame, _ = placement_analysis.run(out)

        rules: list[RewriteRule] = [place2move.InsertFill()]
        if self.insert_initialize:
            rules.append(place2move.InsertInitialize())
        rules += [
            place2move.InsertMoves(placement_frame.entries),
            place2move.RewriteGates(placement_frame.entries),
            place2move.InsertMeasure(placement_frame.entries),
        ]
        rewrite.Walk(rewrite.Chain(*rules)).rewrite(out.code)

        rewrite.Walk(
            rewrite.Chain(
                place2move.LiftMoveStatements(), place2move.DeleteInitialize()
            )
        ).rewrite(out.code)
        rewrite.Walk(place2move.RemoveNoOpStaticPlacements()).rewrite(out.code)
        rewrite.Fixpoint(rewrite.Walk(rewrite.CFGCompactify())).rewrite(out.code)

        state.InsertBlockArgs().rewrite(out.code)
        rewrite.Walk(state.RewriteBranches()).rewrite(out.code)
        rewrite.Walk(state.RewriteLoadStore()).rewrite(out.code)

        rewrite.Fixpoint(
            rewrite.Walk(
                rewrite.Chain(
                    place2move.DeleteQubitNew(), rewrite.DeadCodeElimination()
                )
            )
        ).rewrite(out.code)

        passes.TypeInfer(out.dialects)(out)
        if not no_raise:
            out.verify()
            out.verify_type()

        return out
