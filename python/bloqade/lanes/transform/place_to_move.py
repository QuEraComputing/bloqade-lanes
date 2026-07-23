from __future__ import annotations

from dataclasses import dataclass

from bloqade.analysis import address
from kirin import passes, rewrite
from kirin.ir.method import Method
from kirin.rewrite.abc import RewriteRule

from bloqade.lanes.analysis import layout, placement
from bloqade.lanes.dialects import move
from bloqade.lanes.rewrite import place2move, resolve_pinned, state


@dataclass
class PlaceToMove:
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

        passes.TypeInfer(out.dialects, no_raise=no_raise)(out)
        if not no_raise:
            out.verify()
            out.verify_type()

        return out
