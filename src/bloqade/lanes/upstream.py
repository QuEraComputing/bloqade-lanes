from dataclasses import dataclass
from itertools import chain
from typing import Callable

from bloqade.analysis import address
from bloqade.native.dialects import gate as native_gate
from bloqade.rewrite.passes import AggressiveUnroll
from kirin import ir, passes, rewrite
from kirin.dialects.scf import scf2cf
from kirin.ir.method import Method

from bloqade.lanes.analysis import layout, placement
from bloqade.lanes.dialects import move, place
from bloqade.lanes.rewrite import circuit2place, place2move, state


def default_merge_heuristic(region_a: ir.Region, region_b: ir.Region) -> bool:
    return all(
        isinstance(stmt, (place.R, place.Rz, place.Yield))
        for stmt in chain(region_a.walk(), region_b.walk())
    )


@dataclass
class NativeToPlace:
    merge_heuristic: Callable[[ir.Region, ir.Region], bool] = default_merge_heuristic

    def emit(self, mt: Method, no_raise: bool = True):
        out = mt.similar(mt.dialects.add(place).discard(native_gate))
        AggressiveUnroll(out.dialects, no_raise=no_raise).fixpoint(out)
        rewrite.Walk(scf2cf.ScfToCfRule()).rewrite(out.code)

        rewrite.Walk(circuit2place.HoistConstants()).rewrite(out.code)

        rewrite.Walk(circuit2place.RewriteInitializeToLogicalInitialize()).rewrite(
            out.code
        )

        rewrite.Walk(circuit2place.RewriteLogicalInitializeToNewLogical()).rewrite(
            out.code
        )

        rewrite.Walk(circuit2place.InitializeNewQubits()).rewrite(out.code)
        rewrite.Walk(circuit2place.CleanUpLogicalInitialize()).rewrite(out.code)

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
            rewrite.Walk(circuit2place.MergePlacementRegions(self.merge_heuristic)),
        ).rewrite(out.code)

        rewrite.Walk(circuit2place.HoistNewQubitsUp()).rewrite(out.code)

        rewrite.Fixpoint(
            rewrite.Walk(circuit2place.MergePlacementRegions(self.merge_heuristic)),
        ).rewrite(out.code)

        passes.TypeInfer(out.dialects)(out)

        out.verify()
        out.verify_type()

        return out


@dataclass
class PlaceToMove:
    layout_heristic: layout.LayoutHeuristicABC
    placement_strategy: placement.PlacementStrategyABC
    move_scheduler: place2move.MoveSchedulerABC
    insert_palindrome_moves: bool = True

    def emit(self, mt: Method, no_raise: bool = True):
        out = mt.similar(mt.dialects.add(move))
        address_analysis = address.AddressAnalysis(out.dialects)
        if no_raise:
            address_frame, _ = address_analysis.run_no_raise(out)
            all_qubits = tuple(range(address_analysis.next_address))
            initial_layout = layout.LayoutAnalysis(
                out.dialects, self.layout_heristic, address_frame.entries, all_qubits
            ).get_layout_no_raise(out)

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
                out.dialects, self.layout_heristic, address_frame.entries, all_qubits
            ).get_layout(out)
            placement_frame, _ = placement.PlacementAnalysis(
                out.dialects,
                initial_layout,
                address_frame.entries,
                self.placement_strategy,
            ).run(out)

        rule = rewrite.Chain(
            place2move.InsertFill(initial_layout),
            place2move.InsertInitialize(address_frame.entries, initial_layout),
            place2move.InsertMoves(self.move_scheduler, placement_frame.entries),
            place2move.RewriteGates(placement_frame.entries),
            place2move.InsertMeasure(placement_frame.entries),
        )
        rewrite.Walk(rule).rewrite(out.code)

        if self.insert_palindrome_moves:
            rewrite.Walk(place2move.InsertPalindromeMoves()).rewrite(out.code)

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
