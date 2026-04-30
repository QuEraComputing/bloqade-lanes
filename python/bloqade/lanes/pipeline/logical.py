from __future__ import annotations

from dataclasses import dataclass, field
from itertools import chain
from typing import Callable

import bloqade.qubit as squin_qubit
from bloqade.analysis import address
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
from kirin.validation import ValidationSuite

from bloqade.gemini.common.dialects import qubit as gemini_qubit
from bloqade.gemini.common.validation.duplicate_address import (
    DuplicateAddressValidation,
)
from bloqade.gemini.logical.rewrite.initialize import _RewriteU3ToInitialize
from bloqade.gemini.logical.validation.clifford.analysis import GeminiLogicalValidation
from bloqade.gemini.logical.validation.measurement.analysis import (
    GeminiTerminalMeasurementValidation,
)
from bloqade.lanes.analysis import layout, placement
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode.encoding import LaneAddress
from bloqade.lanes.dialects import move, place
from bloqade.lanes.heuristics.logical.layout import LogicalLayoutHeuristic
from bloqade.lanes.heuristics.logical.placement import LogicalPlacementStrategyNoHome
from bloqade.lanes.rewrite import circuit2place, place2move, resolve_pinned, state
from bloqade.lanes.validation.address import Validation as AddressValidation


def _default_merge_heuristic(region_a: ir.Region, region_b: ir.Region) -> bool:
    return all(
        isinstance(stmt, (place.R, place.Rz, place.Yield))
        for stmt in chain(region_a.walk(), region_b.walk())
    )


@dataclass
class _LogicalNativeToPlace:
    merge_heuristic: Callable[[ir.Region, ir.Region], bool] = field(
        default=_default_merge_heuristic
    )
    arch_spec: ArchSpec | None = field(default=None)

    def emit(self, mt: Method, no_raise: bool = True) -> Method:
        validator = ValidationSuite(
            [
                GeminiLogicalValidation,
                GeminiTerminalMeasurementValidation,
                FlatKernelNoCloningValidation,
            ]
        )
        result = validator.validate(mt)
        if not result.is_valid:
            result.raise_if_invalid()

        out = mt.similar(mt.dialects.add(place))

        rule = rewrite.Chain(
            rewrite.Walk(RewriteNonCliffordToU3()),
            rewrite.Walk(_RewriteU3ToInitialize()),
        )
        CallGraphPass(mt.dialects, rule)(out)

        out = SquinToNative().emit(out, no_raise=no_raise)
        AggressiveUnroll(out.dialects, no_raise=no_raise).fixpoint(out)
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

        rewrite.Walk(circuit2place.RewriteInitializeToLogicalInitialize()).rewrite(
            out.code
        )
        rewrite.Walk(circuit2place.RewriteLogicalInitializeToNewLogical()).rewrite(
            out.code
        )
        rewrite.Walk(circuit2place.CleanUpLogicalInitialize()).rewrite(out.code)
        rewrite.Walk(circuit2place.InitializeNewQubits()).rewrite(out.code)

        rewrite.Walk(circuit2place.RewritePlaceOperations()).rewrite(out.code)
        rewrite.Walk(
            rewrite.Chain(
                rewrite.DeadCodeElimination(),
                rewrite.CommonSubexpressionElimination(),
            )
        ).rewrite(out.code)
        rewrite.Walk(circuit2place.HoistConstants()).rewrite(out.code)
        rewrite.Fixpoint(
            rewrite.Walk(circuit2place.MergePlacementRegions(self.merge_heuristic))
        ).rewrite(out.code)
        rewrite.Walk(circuit2place.HoistNewQubitsUp()).rewrite(out.code)
        rewrite.Fixpoint(
            rewrite.Walk(circuit2place.MergePlacementRegions(self.merge_heuristic))
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
class _LogicalPlaceToMove:
    layout_heuristic: layout.LayoutHeuristicABC
    placement_strategy: placement.PlacementStrategyABC
    insert_return_moves: bool = True
    revert_initial_position: Callable[
        [dict[ir.SSAValue, placement.AtomState], place.StaticPlacement],
        tuple[tuple[LaneAddress, ...], ...] | None,
    ] = place2move.palindrome_move_layers

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

        rules = [
            place2move.InsertFill(),
            place2move.InsertInitialize(),
            place2move.InsertMoves(placement_frame.entries),
            place2move.RewriteGates(placement_frame.entries),
            place2move.InsertMeasure(placement_frame.entries),
        ]
        rewrite.Walk(rewrite.Chain(*rules)).rewrite(out.code)

        if self.insert_return_moves:
            rewrite.Walk(
                place2move.InsertReturnMoves(
                    placement_analysis=placement_frame.entries,
                    revert_initial_position=self.revert_initial_position,
                )
            ).rewrite(out.code)

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
        out.verify()
        out.verify_type()

        return out


@dataclass
class LogicalPipeline:
    """Compile a logical squin kernel to the move dialect.

    Includes Clifford/measurement/no-cloning validation and the full logical
    initialization sequence (InsertInitialize for LogicalQubits).
    """

    layout_heuristic: layout.LayoutHeuristicABC = field(
        default_factory=LogicalLayoutHeuristic
    )
    placement_strategy: placement.PlacementStrategyABC = field(
        default_factory=LogicalPlacementStrategyNoHome
    )
    insert_return_moves: bool = True
    merge_heuristic: Callable[[ir.Region, ir.Region], bool] = field(
        default=_default_merge_heuristic
    )
    arch_spec: ArchSpec | None = None

    def emit(self, mt: Method, no_raise: bool = True) -> Method:
        out = _LogicalNativeToPlace(
            merge_heuristic=self.merge_heuristic,
            arch_spec=self.arch_spec,
        ).emit(mt, no_raise=no_raise)

        out = _LogicalPlaceToMove(
            layout_heuristic=self.layout_heuristic,
            placement_strategy=self.placement_strategy,
            insert_return_moves=self.insert_return_moves,
        ).emit(out, no_raise=no_raise)

        return out
