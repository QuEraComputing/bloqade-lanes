from dataclasses import dataclass
from itertools import chain
from typing import Callable

from bloqade.analysis import address
from bloqade.rewrite.passes import AggressiveUnroll
from kirin import ir, passes, rewrite
from kirin.ir.method import Method

from bloqade.lanes.analysis import layout, placement
from bloqade.lanes.dialects import move, place
from bloqade.lanes.passes.canonicalize import CanonicalizeNative
from bloqade.lanes.rewrite import native2place, place2move


def default_merge_heuristic(region_a: ir.Region, region_b: ir.Region) -> bool:
    return all(
        isinstance(stmt, (place.R, place.Rz, place.Yield))
        for stmt in chain(region_a.walk(), region_b.walk())
    )


@dataclass
class NativeToCircuit:
    merge_heuristic: Callable[[ir.Region, ir.Region], bool] = default_merge_heuristic

    def emit(self, mt: Method, no_raise: bool = True):
        out = mt.similar(mt.dialects.add(place))
        AggressiveUnroll(out.dialects, no_raise=no_raise).fixpoint(out)
        CanonicalizeNative(out.dialects, no_raise=no_raise).fixpoint(out)
        rewrite.Walk(
            native2place.RewritePlaceOperations(),
        ).rewrite(out.code)

        rewrite.Fixpoint(
            rewrite.Walk(native2place.MergePlacementRegions(self.merge_heuristic))
        ).rewrite(out.code)
        passes.TypeInfer(out.dialects)(out)

        return out


@dataclass
class CircuitToMove:
    layout_heristic: layout.LayoutHeuristicABC
    placement_strategy: placement.PlacementStrategyABC
    move_scheduler: place2move.MoveSchedulerABC
    insert_palindrome_moves: bool = True

    def emit(self, mt: Method, no_raise: bool = True):
        out = mt.similar(mt.dialects.add(move))

        if no_raise:
            address_frame, _ = address.AddressAnalysis(out.dialects).run_no_raise(out)
            initial_layout = layout.LayoutAnalysis(
                out.dialects, self.layout_heristic, address_frame.entries
            ).get_layout_no_raise(out)
            placement_frame, _ = placement.PlacementAnalysis(
                out.dialects,
                initial_layout,
                address_frame.entries,
                self.placement_strategy,
            ).run_no_raise(out)
        else:
            address_frame, _ = address.AddressAnalysis(out.dialects).run(out)
            initial_layout = layout.LayoutAnalysis(
                out.dialects, self.layout_heristic, address_frame.entries
            ).get_layout(out)
            placement_frame, _ = placement.PlacementAnalysis(
                out.dialects,
                initial_layout,
                address_frame.entries,
                self.placement_strategy,
            ).run(out)

        placement_analysis = placement_frame.entries
        args = (self.move_scheduler, placement_analysis)
        rule = rewrite.Chain(
            place2move.InsertInitialize(initial_layout),
            place2move.InsertMoves(*args),
            place2move.RewriteCZ(*args),
            place2move.RewriteR(*args),
            place2move.RewriteRz(*args),
        )
        rewrite.Walk(rule).rewrite(out.code)

        if self.insert_palindrome_moves:
            rewrite.Walk(place2move.InsertPalindromeMoves()).rewrite(out.code)

        rewrite.Walk(
            rewrite.Chain(
                place2move.LiftMoveStatements(), place2move.RemoveNoOpStaticPlacements()
            )
        ).rewrite(out.code)

        rewrite.Fixpoint(
            rewrite.Walk(
                rewrite.Chain(
                    place2move.DeleteQubitNew(), rewrite.DeadCodeElimination()
                )
            )
        ).rewrite(out.code)
        passes.TypeInfer(out.dialects)(out)

        return out
