from dataclasses import dataclass, field
from typing import Callable

from bloqade.analysis import address
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

from bloqade.gemini.common.dialects import qubit as gemini_qubit
from bloqade.gemini.logical.rewrite.initialize import _RewriteU3ToInitialize
from bloqade.lanes.analysis import layout, placement
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode.encoding import LaneAddress
from bloqade.lanes.dialects import move, place
from bloqade.lanes.rewrite import circuit2place, place2move, resolve_pinned, state
from bloqade.lanes.rewrite.fuse_gates import FuseAdjacentGates
from bloqade.lanes.rewrite.reorder_static_placement import (
    ReorderStaticPlacement,
    asap_reorder_policy,
)
from bloqade.lanes.rewrite.split_static_placement import (
    SplitStaticPlacement,
    cz_layer_split_policy,
)


@dataclass
class NativeToPlace:
    logical_initialize: bool = True
    arch_spec: ArchSpec | None = field(default=None)
    fuse_gates: bool = False

    def emit(self, mt: Method, no_raise: bool = True):
        out = mt.similar(mt.dialects.add(place))
        if self.logical_initialize:
            rule = rewrite.Chain(
                rewrite.Walk(
                    RewriteNonCliffordToU3(),
                ),
                rewrite.Walk(
                    _RewriteU3ToInitialize(),
                ),
            )
            CallGraphPass(mt.dialects, rule)(out)

        out = SquinToNative().emit(out, no_raise=no_raise)
        AggressiveUnroll(out.dialects, no_raise=no_raise).fixpoint(out)
        rewrite.Walk(scf2cf.ScfToCfRule()).rewrite(out.code)

        rewrite.Walk(circuit2place.HoistConstants()).rewrite(out.code)

        if self.arch_spec is not None:
            from bloqade.gemini.common.validation.duplicate_address import (
                DuplicateAddressValidation,
            )
            from bloqade.lanes.validation.address import Validation

            errors: list[ir.ValidationError] = []
            _, per_stmt_errors = Validation(arch_spec=self.arch_spec).run(out)
            errors.extend(per_stmt_errors)
            _, dup_errors = DuplicateAddressValidation().run(out)
            errors.extend(dup_errors)
            if errors:
                message = f"Gemini IR validation failed with {len(errors)} error(s)"
                raise ValidationErrorGroup(message, errors=errors)

        if self.logical_initialize:
            rewrite.Walk(circuit2place.RewriteInitializeToLogicalInitialize()).rewrite(
                out.code
            )

            rewrite.Walk(circuit2place.RewriteLogicalInitializeToNewLogical()).rewrite(
                out.code
            )
            rewrite.Walk(circuit2place.CleanUpLogicalInitialize()).rewrite(out.code)

        # Must run unconditionally: NewAt statements can appear in physical
        # circuits (not just logical ones gated on arch_spec). A follow-up PR
        # will separate the logical and physical compilation pipelines properly.
        rewrite.Walk(circuit2place.InitializeNewQubits()).rewrite(out.code)

        rewrite.Walk(
            circuit2place.RewritePlaceOperations(),
        ).rewrite(out.code)

        rewrite.Walk(
            rewrite.Chain(
                rewrite.DeadCodeElimination(), rewrite.CommonSubexpressionElimination()
            )
        ).rewrite(out.code)

        rewrite.Walk(circuit2place.HoistConstants()).rewrite(out.code)

        rewrite.Fixpoint(
            rewrite.Walk(
                circuit2place.MergeStaticPlacement(circuit2place.gate_only_merge)
            ),
        ).rewrite(out.code)

        rewrite.Walk(circuit2place.HoistNewQubitsUp()).rewrite(out.code)

        rewrite.Fixpoint(
            rewrite.Walk(
                circuit2place.MergeStaticPlacement(circuit2place.gate_only_merge)
            ),
        ).rewrite(out.code)
        rewrite.Walk(ReorderStaticPlacement(asap_reorder_policy)).rewrite(out.code)
        if self.fuse_gates:
            rewrite.Fixpoint(rewrite.Walk(FuseAdjacentGates())).rewrite(out.code)
        rewrite.Walk(SplitStaticPlacement(cz_layer_split_policy)).rewrite(out.code)

        out = out.similar(out.dialects.discard(native_gate).discard(gemini_qubit))
        passes.TypeInfer(out.dialects, no_raise=True)(out)

        if not no_raise:
            out.verify()
            out.verify_type()

        return out


@dataclass
class PlaceToMove:
    layout_heuristic: layout.LayoutHeuristicABC
    placement_strategy: placement.PlacementStrategyABC
    insert_return_moves: bool = True
    revert_initial_position: Callable[
        [dict[ir.SSAValue, placement.AtomState], place.StaticPlacement],
        tuple[tuple[LaneAddress, ...], ...] | None,
    ] = place2move.palindrome_move_layers
    logical_initialize: bool = True

    def emit(self, mt: Method, no_raise: bool = True):
        out = mt.similar(mt.dialects.add(move))
        address_analysis = address.AddressAnalysis(out.dialects)
        if no_raise:
            address_frame, _ = address_analysis.run_no_raise(out)
            all_qubits = tuple(range(address_analysis.next_address))
            initial_layout = layout.LayoutAnalysis(
                out.dialects, self.layout_heuristic, address_frame.entries, all_qubits
            ).get_layout_no_raise(out)

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
            placement_frame, _ = placement_analysis.run_no_raise(out)
        else:
            address_frame, _ = address_analysis.run(out)
            all_qubits = tuple(range(address_analysis.next_address))
            initial_layout = layout.LayoutAnalysis(
                out.dialects, self.layout_heuristic, address_frame.entries, all_qubits
            ).get_layout(out)

            rewrite.Walk(
                resolve_pinned.ResolvePinnedAddresses(
                    address_entries=address_frame.entries,
                    initial_layout=initial_layout,
                )
            ).rewrite(out.code)

            placement_frame, _ = placement.PlacementAnalysis(
                out.dialects,
                initial_layout,
                address_frame.entries,
                self.placement_strategy,
            ).run(out)

        rules: list[RewriteRule] = [place2move.InsertFill()]
        if self.logical_initialize:
            # Insert logical initialize operations based on the address frame and initial layout.
            rules.append(place2move.InsertInitialize())

        rules.extend(
            (
                place2move.InsertMoves(placement_frame.entries),
                place2move.RewriteGates(placement_frame.entries),
                place2move.InsertMeasure(placement_frame.entries),
            )
        )
        rule = rewrite.Chain(*rules)
        rewrite.Walk(rule).rewrite(out.code)

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


def squin_to_move(
    mt: ir.Method,
    layout_heuristic: layout.LayoutHeuristicABC,
    placement_strategy: placement.PlacementStrategyABC,
    insert_return_moves: bool = True,
    revert_initial_position: Callable[
        [dict[ir.SSAValue, placement.AtomState], place.StaticPlacement],
        tuple[tuple[LaneAddress, ...], ...] | None,
    ] = place2move.palindrome_move_layers,
    no_raise: bool = True,
    logical_initialize: bool = True,
    fuse_gates: bool = False,
) -> ir.Method:
    """
    Compile a squin kernel to move dialect.

    Args:
        mt (ir.Method): The Squin kernel to compile.
        layout_heuristic (layout.LayoutHeuristicABC): The layout heuristic to use.
        placement_strategy (placement.PlacementStrategyABC): The placement strategy to use.
        insert_return_moves (bool, optional): Whether to insert return moves. Defaults to True.
        revert_initial_position (Callable, optional): Callback returning move
            layers to insert near the end of each static placement region.
            Defaults to palindrome_move_layers.
        no_raise (bool, optional): Whether to suppress exceptions during compilation. Defaults to True.
        logical_initialize (bool, optional): Whether to apply rewrites that insert logical qubit initialization operations; when False, these rewrites are skipped. Defaults to True.
        fuse_gates (bool, optional): Whether to fuse adjacent compatible gates before splitting into CZ-anchored layers. Defaults to True.

    Returns:
        ir.Method: The compiled move dialect method.
    """

    arch_spec: ArchSpec | None = getattr(layout_heuristic, "arch_spec", None)
    out = NativeToPlace(
        logical_initialize=logical_initialize,
        arch_spec=arch_spec,
        fuse_gates=fuse_gates,
    ).emit(mt, no_raise=no_raise)
    out = PlaceToMove(
        layout_heuristic=layout_heuristic,
        placement_strategy=placement_strategy,
        insert_return_moves=insert_return_moves,
        revert_initial_position=revert_initial_position,
        logical_initialize=logical_initialize,
    ).emit(out, no_raise=no_raise)

    passes.TypeInfer(mt.dialects)(out)
    out.verify()
    out.verify_type()

    return out
