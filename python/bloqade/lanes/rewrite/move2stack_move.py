"""move2stack_move — in-place rewrite from move dialect → stack_move dialect.

Inverse of RewriteStackMoveToMove in stack_move2move.py.

Strips move.Load / move.Store state threading, materialises address
attributes as stack_move.Const* SSA values, converts py.Constant
float/int values to stack_move.ConstFloat/Int, and reconstructs
stack_move.Measure + stack_move.AwaitMeasure from the
move.Measure + move.GetFutureResult chain + ilist.New pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import singledispatchmethod

from bloqade.decoders.dialects import annotate as _annotate
from kirin import ir
from kirin.dialects import ilist as kirin_ilist, py as kirin_py
from kirin.rewrite.abc import RewriteResult, RewriteRule

from bloqade.lanes.dialects import move, stack_move


@dataclass
class RewriteMoveToStackMove(RewriteRule):
    """Rewrite a move-dialect block into stack_move dialect in place.

    Mutable state carried across the block walk:
    - _first_fill_emitted: True after the first move.Fill is processed,
      so subsequent fills lower to stack_move.Fill instead of InitialFill.
    - _gfr_results: set of SSA results produced by move.GetFutureResult
      statements, used to detect the measurement-bundle ilist.New.
    - _future_to_sm_measure: maps move.Measure.future SSA → the emitted
      stack_move.Measure statement, for AwaitMeasure reconstruction.
    """

    _first_fill_emitted: bool = field(default=False, init=False)
    _gfr_results: set[ir.SSAValue] = field(default_factory=set, init=False)
    _future_to_sm_measure: dict[ir.SSAValue, stack_move.Measure] = field(
        default_factory=dict, init=False
    )

    def rewrite_Block(self, node: ir.Block) -> RewriteResult:
        to_delete: list[ir.Statement] = []
        for stmt in list(node.stmts):
            self._rewrite(stmt, to_delete)
        for stmt in reversed(to_delete):
            stmt.delete()
        return RewriteResult(has_done_something=True)

    @singledispatchmethod
    def _rewrite(self, stmt: ir.Statement, to_delete: list[ir.Statement]) -> None:
        """Default: unknown statements pass through unchanged."""
        pass

    @_rewrite.register(move.Load)
    def _(self, stmt: move.Load, to_delete: list[ir.Statement]) -> None:
        to_delete.append(stmt)

    @_rewrite.register(move.Store)
    def _(self, stmt: move.Store, to_delete: list[ir.Statement]) -> None:
        to_delete.append(stmt)

    @_rewrite.register(move.Fill)
    def _(self, stmt: move.Fill, to_delete: list[ir.Statement]) -> None:
        loc_consts = tuple(
            stack_move.ConstLoc(value=addr) for addr in stmt.location_addresses
        )
        for lc in loc_consts:
            lc.insert_before(stmt)
        cls = stack_move.Fill if self._first_fill_emitted else stack_move.InitialFill
        new = cls(locations=tuple(lc.result for lc in loc_consts))
        new.insert_before(stmt)
        self._first_fill_emitted = True
        to_delete.append(stmt)

    @_rewrite.register(move.Move)
    def _(self, stmt: move.Move, to_delete: list[ir.Statement]) -> None:
        lane_consts = tuple(stack_move.ConstLane(value=addr) for addr in stmt.lanes)
        for lc in lane_consts:
            lc.insert_before(stmt)
        new = stack_move.Move(lanes=tuple(lc.result for lc in lane_consts))
        new.insert_before(stmt)
        to_delete.append(stmt)

    @_rewrite.register(kirin_py.Constant)
    def _(self, stmt: kirin_py.Constant, to_delete: list[ir.Statement]) -> None:
        val = stmt.value.unwrap()
        if isinstance(val, bool):
            return  # bool subclasses int — pass through unchanged
        if isinstance(val, float):
            new: ir.Statement = stack_move.ConstFloat(value=val)
        elif isinstance(val, int):
            new = stack_move.ConstInt(value=val)
        else:
            return  # non-numeric py.Constant passes through unchanged
        new.insert_before(stmt)
        stmt.result.replace_by(new.result)
        to_delete.append(stmt)

    @_rewrite.register(move.LocalR)
    def _(self, stmt: move.LocalR, to_delete: list[ir.Statement]) -> None:
        loc_consts = tuple(
            stack_move.ConstLoc(value=addr) for addr in stmt.location_addresses
        )
        for lc in loc_consts:
            lc.insert_before(stmt)
        new = stack_move.LocalR(
            axis_angle=stmt.axis_angle,
            rotation_angle=stmt.rotation_angle,
            locations=tuple(lc.result for lc in loc_consts),
        )
        new.insert_before(stmt)
        to_delete.append(stmt)

    @_rewrite.register(move.LocalRz)
    def _(self, stmt: move.LocalRz, to_delete: list[ir.Statement]) -> None:
        loc_consts = tuple(
            stack_move.ConstLoc(value=addr) for addr in stmt.location_addresses
        )
        for lc in loc_consts:
            lc.insert_before(stmt)
        new = stack_move.LocalRz(
            rotation_angle=stmt.rotation_angle,
            locations=tuple(lc.result for lc in loc_consts),
        )
        new.insert_before(stmt)
        to_delete.append(stmt)

    @_rewrite.register(move.GlobalR)
    def _(self, stmt: move.GlobalR, to_delete: list[ir.Statement]) -> None:
        new = stack_move.GlobalR(
            axis_angle=stmt.axis_angle,
            rotation_angle=stmt.rotation_angle,
        )
        new.insert_before(stmt)
        to_delete.append(stmt)

    @_rewrite.register(move.GlobalRz)
    def _(self, stmt: move.GlobalRz, to_delete: list[ir.Statement]) -> None:
        new = stack_move.GlobalRz(rotation_angle=stmt.rotation_angle)
        new.insert_before(stmt)
        to_delete.append(stmt)

    @_rewrite.register(move.CZ)
    def _(self, stmt: move.CZ, to_delete: list[ir.Statement]) -> None:
        zone_const = stack_move.ConstZone(value=stmt.zone_address)
        zone_const.insert_before(stmt)
        new = stack_move.CZ(zone=zone_const.result)
        new.insert_before(stmt)
        to_delete.append(stmt)

    @_rewrite.register(move.Measure)
    def _(self, stmt: move.Measure, to_delete: list[ir.Statement]) -> None:
        zone_consts = tuple(
            stack_move.ConstZone(value=addr) for addr in stmt.zone_addresses
        )
        for zc in zone_consts:
            zc.insert_before(stmt)
        new = stack_move.Measure(zones=tuple(zc.result for zc in zone_consts))
        new.insert_before(stmt)
        self._future_to_sm_measure[stmt.future] = new
        to_delete.append(stmt)

    @_rewrite.register(move.GetFutureResult)
    def _(self, stmt: move.GetFutureResult, to_delete: list[ir.Statement]) -> None:
        self._gfr_results.add(stmt.result)
        to_delete.append(stmt)

    @_rewrite.register(kirin_ilist.New)
    def _(self, stmt: kirin_ilist.New, to_delete: list[ir.Statement]) -> None:
        values = tuple(stmt.values)
        if not values:
            return  # empty ilist (e.g. coordinates placeholder) — pass through
        if not all(v in self._gfr_results for v in values):
            return  # non-measurement ilist — pass through
        # All values are GetFutureResult outputs: this is the measurement bundle.
        futures: set[ir.SSAValue] = set()
        for v in values:
            if isinstance(v, ir.ResultValue):
                owner = v.owner
                if isinstance(owner, move.GetFutureResult):
                    futures.add(owner.measurement_future)
        if len(futures) != 1:
            return
        (move_future,) = futures
        sm_measure = self._future_to_sm_measure.get(move_future)
        if sm_measure is None:
            return
        aw = stack_move.AwaitMeasure(future=sm_measure.results[0])
        aw.insert_before(stmt)
        stmt.result.replace_by(aw.result)
        to_delete.append(stmt)

    @_rewrite.register(_annotate.stmts.SetDetector)
    def _(
        self, stmt: _annotate.stmts.SetDetector, to_delete: list[ir.Statement]
    ) -> None:
        new = stack_move.SetDetector(array=stmt.measurements)
        new.insert_before(stmt)
        stmt.result.replace_by(new.result)
        to_delete.append(stmt)

    @_rewrite.register(_annotate.stmts.SetObservable)
    def _(
        self, stmt: _annotate.stmts.SetObservable, to_delete: list[ir.Statement]
    ) -> None:
        new = stack_move.SetObservable(array=stmt.measurements)
        new.insert_before(stmt)
        stmt.result.replace_by(new.result)
        to_delete.append(stmt)
