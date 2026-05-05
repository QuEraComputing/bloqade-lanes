"""move2stack_move — in-place rewrite from move dialect → stack_move dialect.

Inverse of RewriteStackMoveToMove in stack_move2move.py.

Strips move.Load / move.Store state threading, materialises address
attributes as stack_move.Const* SSA values, converts py.Constant
float/int values to stack_move.ConstFloat/Int, and reconstructs
stack_move.Measure + stack_move.AwaitMeasure + stack_move.GetItem from
the move.Measure + move.GetFutureResult pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import singledispatchmethod

from bloqade.decoders.dialects import annotate as _annotate
from bloqade.decoders.dialects.annotate.types import MeasurementResultType
from kirin import ir
from kirin.dialects import ilist as kirin_ilist, py as kirin_py
from kirin.rewrite.abc import RewriteResult, RewriteRule

from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode.encoding import LocationAddress, ZoneAddress
from bloqade.lanes.dialects import move, stack_move

# Reverse of stack_move.TYPE_TAG: Kirin TypeAttribute → bytecode type_tag int.
# MeasurementResultType has no dedicated tag yet; use Int (1) as a placeholder.
_TYPE_TO_TAG: dict[object, int] = {v: k for k, v in stack_move.TYPE_TAG.items()}
_TYPE_TO_TAG[MeasurementResultType] = 1


def _elem_type_tag(value: ir.SSAValue) -> int:
    """Return the bytecode type_tag for the element type of *value*."""
    tag = _TYPE_TO_TAG.get(value.type)
    if tag is not None:
        return tag
    # Fallback for unrecognised types (e.g. future MeasurementResultType tag).
    return 1


@dataclass
class RewriteMoveToStackMove(RewriteRule):
    """Rewrite a move-dialect block into stack_move dialect in place.

    Constructor args:
    - arch_spec: required. GetFutureResult lowering uses
      ``arch_spec.yield_zone_locations`` and ``arch_spec.get_zone_index``
      to resolve the flat index of a (zone, location) pair into the
      AwaitMeasure result array.

    Mutable state carried across the block walk:
    - _first_fill_emitted: True after the first move.Fill is processed,
      so subsequent fills lower to stack_move.Fill instead of InitialFill.
    - _future_to_await: maps move.Measure.future SSA → (AwaitMeasure result
      SSA, zone_addresses tuple), for GetFutureResult index resolution.
    """

    arch_spec: ArchSpec
    _first_fill_emitted: bool = field(default=False, init=False)
    _future_to_await: dict[ir.SSAValue, tuple[ir.SSAValue, tuple[ZoneAddress, ...]]] = (
        field(default_factory=dict, init=False)
    )

    def rewrite_Block(self, node: ir.Block) -> RewriteResult:
        self._first_fill_emitted = False
        self._future_to_await = {}
        to_delete: list[ir.Statement] = []
        result = RewriteResult()
        for stmt in list(node.stmts):
            result = result.join(self._rewrite(stmt, to_delete))
        for stmt in reversed(to_delete):
            stmt.delete()
        if to_delete:
            result = result.join(RewriteResult(has_done_something=True))
        return result

    def _measure_flat_index(
        self,
        zone_addresses: tuple[ZoneAddress, ...],
        target_zone: ZoneAddress,
        target_loc: LocationAddress,
    ) -> int:
        """Flat index of (target_zone, target_loc) in the measurement array."""
        offset = 0
        for zone in zone_addresses:
            if zone == target_zone:
                within = self.arch_spec.get_zone_index(target_loc, zone)
                if within is None:
                    raise ValueError(
                        f"location {target_loc!r} not found in zone {zone!r}"
                    )
                return offset + within
            offset += sum(1 for _ in self.arch_spec.yield_zone_locations(zone))
        raise ValueError(
            f"zone {target_zone!r} not found in measurement zone_addresses"
        )

    @singledispatchmethod
    def _rewrite(
        self, stmt: ir.Statement, to_delete: list[ir.Statement]
    ) -> RewriteResult:
        """Default: unknown statements pass through unchanged."""
        return RewriteResult()

    @_rewrite.register(move.Load)
    def _(self, stmt: move.Load, to_delete: list[ir.Statement]) -> RewriteResult:
        to_delete.append(stmt)
        return RewriteResult()

    @_rewrite.register(move.Store)
    def _(self, stmt: move.Store, to_delete: list[ir.Statement]) -> RewriteResult:
        to_delete.append(stmt)
        return RewriteResult()

    @_rewrite.register(move.Fill)
    def _(self, stmt: move.Fill, to_delete: list[ir.Statement]) -> RewriteResult:
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
        return RewriteResult(has_done_something=True)

    @_rewrite.register(move.Move)
    def _(self, stmt: move.Move, to_delete: list[ir.Statement]) -> RewriteResult:
        lane_consts = tuple(stack_move.ConstLane(value=addr) for addr in stmt.lanes)
        for lc in lane_consts:
            lc.insert_before(stmt)
        new = stack_move.Move(lanes=tuple(lc.result for lc in lane_consts))
        new.insert_before(stmt)
        to_delete.append(stmt)
        return RewriteResult(has_done_something=True)

    @_rewrite.register(kirin_py.Constant)
    def _(
        self, stmt: kirin_py.Constant, to_delete: list[ir.Statement]
    ) -> RewriteResult:
        val = stmt.value.unwrap()
        if isinstance(val, bool):
            return RewriteResult()  # bool subclasses int — pass through unchanged
        if isinstance(val, float):
            new: ir.Statement = stack_move.ConstFloat(value=val)
        elif isinstance(val, int):
            new = stack_move.ConstInt(value=val)
        else:
            return RewriteResult()  # non-numeric py.Constant passes through unchanged
        new.insert_before(stmt)
        stmt.result.replace_by(new.result)
        to_delete.append(stmt)
        return RewriteResult(has_done_something=True)

    @_rewrite.register(move.LocalR)
    def _(self, stmt: move.LocalR, to_delete: list[ir.Statement]) -> RewriteResult:
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
        return RewriteResult(has_done_something=True)

    @_rewrite.register(move.LocalRz)
    def _(self, stmt: move.LocalRz, to_delete: list[ir.Statement]) -> RewriteResult:
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
        return RewriteResult(has_done_something=True)

    @_rewrite.register(move.GlobalR)
    def _(self, stmt: move.GlobalR, to_delete: list[ir.Statement]) -> RewriteResult:
        new = stack_move.GlobalR(
            axis_angle=stmt.axis_angle,
            rotation_angle=stmt.rotation_angle,
        )
        new.insert_before(stmt)
        to_delete.append(stmt)
        return RewriteResult(has_done_something=True)

    @_rewrite.register(move.GlobalRz)
    def _(self, stmt: move.GlobalRz, to_delete: list[ir.Statement]) -> RewriteResult:
        new = stack_move.GlobalRz(rotation_angle=stmt.rotation_angle)
        new.insert_before(stmt)
        to_delete.append(stmt)
        return RewriteResult(has_done_something=True)

    @_rewrite.register(move.CZ)
    def _(self, stmt: move.CZ, to_delete: list[ir.Statement]) -> RewriteResult:
        zone_const = stack_move.ConstZone(value=stmt.zone_address)
        zone_const.insert_before(stmt)
        new = stack_move.CZ(zone=zone_const.result)
        new.insert_before(stmt)
        to_delete.append(stmt)
        return RewriteResult(has_done_something=True)

    @_rewrite.register(move.Measure)
    def _(self, stmt: move.Measure, to_delete: list[ir.Statement]) -> RewriteResult:
        zone_consts = tuple(
            stack_move.ConstZone(value=addr) for addr in stmt.zone_addresses
        )
        for zc in zone_consts:
            zc.insert_before(stmt)
        sm_measure = stack_move.Measure(zones=tuple(zc.result for zc in zone_consts))
        sm_measure.insert_before(stmt)
        aw = stack_move.AwaitMeasure(future=sm_measure.results[0])
        aw.insert_before(stmt)
        self._future_to_await[stmt.future] = (aw.result, stmt.zone_addresses)
        to_delete.append(stmt)
        return RewriteResult(has_done_something=True)

    @_rewrite.register(move.GetFutureResult)
    def _(
        self, stmt: move.GetFutureResult, to_delete: list[ir.Statement]
    ) -> RewriteResult:
        await_result, zone_addresses = self._future_to_await[stmt.measurement_future]
        idx = self._measure_flat_index(
            zone_addresses, stmt.zone_address, stmt.location_address
        )
        idx_const = stack_move.ConstInt(value=idx)
        idx_const.insert_before(stmt)
        gi = stack_move.GetItem(array=await_result, indices=(idx_const.result,))
        gi.insert_before(stmt)
        stmt.result.replace_by(gi.result)
        to_delete.append(stmt)
        return RewriteResult(has_done_something=True)

    @_rewrite.register(kirin_ilist.New)
    def _(self, stmt: kirin_ilist.New, to_delete: list[ir.Statement]) -> RewriteResult:
        values = tuple(stmt.values)
        if not values:
            return RewriteResult()  # empty ilist placeholder — pass through

        # 2-D: all values come from inner stack_move.NewArray rows emitted
        # earlier in this same block walk (inner ilist.New → NewArray with
        # replace_by already applied, so stmt.values now point at NewArray
        # results).
        if all(
            isinstance(v, ir.ResultValue) and isinstance(v.owner, stack_move.NewArray)
            for v in values
        ):
            inner_stmts: list[stack_move.NewArray] = [v.owner for v in values]  # type: ignore[union-attr]
            dim0 = len(inner_stmts)
            dim1 = inner_stmts[0].dim0
            type_tag = inner_stmts[0].type_tag
            flat_values = tuple(v for row in inner_stmts for v in row.values)
            new_2d = stack_move.NewArray(
                values=flat_values, type_tag=type_tag, dim0=dim0, dim1=dim1
            )
            new_2d.insert_before(stmt)
            for inner in inner_stmts:
                inner.delete()
            stmt.result.replace_by(new_2d.result)
            to_delete.append(stmt)
            return RewriteResult(has_done_something=True)

        # 1-D: infer type_tag from the first element.
        type_tag = _elem_type_tag(values[0])
        new_1d = stack_move.NewArray(
            values=values, type_tag=type_tag, dim0=len(values), dim1=0
        )
        new_1d.insert_before(stmt)
        stmt.result.replace_by(new_1d.result)
        to_delete.append(stmt)
        return RewriteResult(has_done_something=True)

    @_rewrite.register(_annotate.stmts.SetDetector)
    def _(
        self, stmt: _annotate.stmts.SetDetector, to_delete: list[ir.Statement]
    ) -> RewriteResult:
        new = stack_move.SetDetector(array=stmt.measurements)
        new.insert_before(stmt)
        stmt.result.replace_by(new.result)
        to_delete.append(stmt)
        return RewriteResult(has_done_something=True)

    @_rewrite.register(_annotate.stmts.SetObservable)
    def _(
        self, stmt: _annotate.stmts.SetObservable, to_delete: list[ir.Statement]
    ) -> RewriteResult:
        new = stack_move.SetObservable(array=stmt.measurements)
        new.insert_before(stmt)
        stmt.result.replace_by(new.result)
        to_delete.append(stmt)
        return RewriteResult(has_done_something=True)
