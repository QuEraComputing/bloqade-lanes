"""stack_move2move — in-place rewrite from stack_move → multi-dialect IR.

Extends Kirin's RewriteRule with a rewrite_Block handler that walks the
block's statements once and, for each stack_move statement, inserts the
corresponding target-dialect statement(s) via insert_before and deletes
the original. State threading is woven in along the way: move.Load at
block start initialises the StateType SSA value, each stateful move.*
op consumes the current state and produces a new one, and move.Store +
func.Return close out the block.

Follows the same pattern as python/bloqade/lanes/rewrite/state.py's
RewriteLoadStore.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import singledispatchmethod
from typing import Any, TypeVar

from kirin import ir
from kirin.rewrite.abc import RewriteResult, RewriteRule

from bloqade.lanes.dialects import move, stack_move
from bloqade.lanes.layout.encoding import LaneAddress, LocationAddress, ZoneAddress
from bloqade.lanes.utils import no_none_elements_tuple

# Generic TypeVar used by the _lift_attrs helper to propagate the concrete
# attribute type through runtime isinstance checks.
T = TypeVar("T")


@dataclass
class RewriteStackMoveToMove(RewriteRule):
    """Rewrite a stack_move block into a multi-dialect block in place.

    Mutable state on the rule instance, carried across the walk:
    - ssa_to_attr: stack_move SSA → raw Python attribute value (float,
      int, LocationAddress, LaneAddress, ZoneAddress) for operands that
      need to be lifted into target-dialect attributes (addresses,
      rotation angles). SSA-to-attribute can't be expressed through SSA
      rewiring because attributes aren't SSA values, so we carry an
      explicit mapping. The value type is Any because the lifted values
      span heterogeneous scalar and address types.
    - state: the current StateType SSA value in the target IR.

    For SSA-valued outputs (arrays, futures, detectors, observables,
    constants that emit py.Constant), we use the Kirin idiom
    `old_ssa.replace_by(new_ssa)` to redirect all uses in place — no
    second mapping needed. This matches state.RewriteLoadStore's
    `next_use.replace_by(current_use)` pattern.
    """

    ssa_to_attr: dict[ir.SSAValue, Any] = field(default_factory=dict)
    state: ir.SSAValue | None = None

    def rewrite_Block(self, node: ir.Block) -> RewriteResult:
        # Insert the initial move.Load at block start.
        load = move.Load()
        first = next(iter(node.stmts), None)
        if first is None:
            node.stmts.append(load)
        else:
            load.insert_before(first)
        self.state = load.result

        to_delete: list[ir.Statement] = []
        for stmt in list(node.stmts):
            if stmt is load:
                continue
            # Non-stack_move statements (e.g. existing py.Constant) pass
            # through unchanged via the singledispatchmethod base case.
            self._rewrite(stmt, to_delete)

        # Delete in reverse order: consumers before producers so that when
        # a Const* stmt is deleted its result already has no uses. Without
        # this, e.g. stack_move.Fill(locations=(const_loc.result,)) would
        # still reference the const_loc result when const_loc.delete() runs.
        for stmt in reversed(to_delete):
            stmt.delete()
        return RewriteResult(has_done_something=True)

    @singledispatchmethod
    def _rewrite(self, stmt: ir.Statement, to_delete: list[ir.Statement]) -> None:
        """Default: non-stack_move statements pass through unchanged."""
        pass

    @_rewrite.register(stack_move.Return)
    def _(self, stmt: stack_move.Return, to_delete: list[ir.Statement]) -> None:
        from kirin.dialects import func

        assert self.state is not None
        move.Store(self.state).insert_before(stmt)
        # The stack_move.Return.value operand has already been rewired by
        # earlier replace_by calls on its defining Const* / stack op, so
        # we can thread it directly through to func.Return.
        func.Return(stmt.value).insert_before(stmt)
        to_delete.append(stmt)

    @_rewrite.register(stack_move.Halt)
    def _(self, stmt: stack_move.Halt, to_delete: list[ir.Statement]) -> None:
        from kirin.dialects import func

        assert self.state is not None
        move.Store(self.state).insert_before(stmt)
        none_stmt = func.ConstantNone()
        none_stmt.insert_before(stmt)
        func.Return(none_stmt.result).insert_before(stmt)
        to_delete.append(stmt)

    @_rewrite.register(stack_move.ConstFloat)
    def _(self, stmt: stack_move.ConstFloat, to_delete: list[ir.Statement]) -> None:
        from kirin.dialects import py

        out = py.Constant(stmt.value)
        out.insert_before(stmt)
        # Redirect all SSA uses of the old stack_move.ConstFloat result to
        # the new py.Constant result in place.
        stmt.result.replace_by(out.result)
        # Consumers that need the raw float as an attribute (e.g.
        # _rewrite_LocalR building a rotation_angle= kwarg) look up ssa_to_attr.
        # The key is the new SSA, because replace_by rewired the operands
        # stored on downstream statements to point there.
        self.ssa_to_attr[out.result] = stmt.value
        to_delete.append(stmt)

    @_rewrite.register(stack_move.ConstInt)
    def _(self, stmt: stack_move.ConstInt, to_delete: list[ir.Statement]) -> None:
        from kirin.dialects import py

        out = py.Constant(stmt.value)
        out.insert_before(stmt)
        stmt.result.replace_by(out.result)
        self.ssa_to_attr[out.result] = stmt.value
        to_delete.append(stmt)

    @_rewrite.register(stack_move.ConstLoc)
    def _(self, stmt: stack_move.ConstLoc, to_delete: list[ir.Statement]) -> None:
        # Address constants stay as decoder attributes — downstream move.*
        # statements take them as attribute values, not SSA operands.
        # We track the raw attribute value for later attribute lifting.
        self.ssa_to_attr[stmt.result] = stmt.value
        to_delete.append(stmt)

    @_rewrite.register(stack_move.ConstLane)
    def _(self, stmt: stack_move.ConstLane, to_delete: list[ir.Statement]) -> None:
        self.ssa_to_attr[stmt.result] = stmt.value
        to_delete.append(stmt)

    @_rewrite.register(stack_move.ConstZone)
    def _(self, stmt: stack_move.ConstZone, to_delete: list[ir.Statement]) -> None:
        self.ssa_to_attr[stmt.result] = stmt.value
        to_delete.append(stmt)

    @_rewrite.register(stack_move.Pop)
    def _(self, stmt: stack_move.Pop, to_delete: list[ir.Statement]) -> None:
        # Pop collapses — no target emission. The popped SSA value remains
        # on its definition; if nothing else references it, it becomes
        # dead and a later DCE pass cleans it up.
        to_delete.append(stmt)

    @_rewrite.register(stack_move.Dup)
    def _(self, stmt: stack_move.Dup, to_delete: list[ir.Statement]) -> None:
        # Dup is a semantic identity — redirect all uses of the result to
        # the input in place.
        stmt.result.replace_by(stmt.value)
        to_delete.append(stmt)

    @_rewrite.register(stack_move.Swap)
    def _(self, stmt: stack_move.Swap, to_delete: list[ir.Statement]) -> None:
        # Swap is a permutation — out_top ≡ in_bot, out_bot ≡ in_top.
        stmt.out_top.replace_by(stmt.in_bot)
        stmt.out_bot.replace_by(stmt.in_top)
        to_delete.append(stmt)

    # ── Attribute lifting ─────────────────────────────────────────────

    def _try_lift(self, v: ir.SSAValue, attr_type: type[T]) -> T | None:
        """Look up an SSA value's backing attribute and return it if its
        concrete type matches ``attr_type``; otherwise return None (for both
        missing-mapping and wrong-type cases)."""
        data = self.ssa_to_attr.get(v)
        if data is None:
            return None
        return data if isinstance(data, attr_type) else None

    def _lift_attrs(
        self,
        ssa_values: tuple[ir.SSAValue, ...],
        attr_type: type[T],
    ) -> tuple[T, ...]:
        """Resolve each stack_move SSA operand to its backing Python-class
        attribute value, verifying the concrete type matches ``attr_type``.

        Raises:
            RuntimeError: if an SSA operand isn't attribute-backed (i.e.
                didn't come from a stack_move.Const*), or if its attribute
                has a type that doesn't match ``attr_type``.
        """
        raws: tuple[T | None, ...] = tuple(
            self._try_lift(v, attr_type) for v in ssa_values
        )
        if no_none_elements_tuple(raws):
            return raws
        for v, r in zip(ssa_values, raws):
            if r is None:
                if v not in self.ssa_to_attr:
                    raise RuntimeError(
                        f"no attribute mapping for {v}: operand must "
                        f"trace back to a Const* statement"
                    )
                raise RuntimeError(
                    f"attribute type mismatch for {v}: expected "
                    f"{attr_type.__name__}"
                )
        # Unreachable: the loop above raises on any None encountered.
        raise RuntimeError("_lift_attrs: unreachable")

    # ── Atom operations ───────────────────────────────────────────────

    @_rewrite.register(stack_move.InitialFill)
    def _(self, stmt: stack_move.InitialFill, to_delete: list[ir.Statement]) -> None:
        assert self.state is not None
        addrs = self._lift_attrs(stmt.locations, LocationAddress)
        # move dialect has no InitialFill — both stack_move.InitialFill
        # and stack_move.Fill lower to move.Fill. The InitialFill
        # distinction exists only at the bytecode/stack_move layer for
        # validation (InitialFillNotFirstError).
        new = move.Fill(self.state, location_addresses=addrs)
        new.insert_before(stmt)
        self.state = new.result
        to_delete.append(stmt)

    @_rewrite.register(stack_move.Fill)
    def _(self, stmt: stack_move.Fill, to_delete: list[ir.Statement]) -> None:
        assert self.state is not None
        addrs = self._lift_attrs(stmt.locations, LocationAddress)
        new = move.Fill(self.state, location_addresses=addrs)
        new.insert_before(stmt)
        self.state = new.result
        to_delete.append(stmt)

    @_rewrite.register(stack_move.Move)
    def _(self, stmt: stack_move.Move, to_delete: list[ir.Statement]) -> None:
        assert self.state is not None
        lanes = self._lift_attrs(stmt.lanes, LaneAddress)
        new = move.Move(self.state, lanes=lanes)
        new.insert_before(stmt)
        self.state = new.result
        to_delete.append(stmt)

    # ── Gates ─────────────────────────────────────────────────────────

    @_rewrite.register(stack_move.LocalR)
    def _(self, stmt: stack_move.LocalR, to_delete: list[ir.Statement]) -> None:
        assert self.state is not None
        # stack_move.LocalR and move.LocalR share axis_angle/rotation_angle
        # SSA args; stack_move additionally carries locations as an SSA
        # tuple, which lowers to move.LocalR's location_addresses attribute.
        # The angle SSAs were rewired to the new py.Constant results by
        # _rewrite_ConstFloat (via replace_by), so we can forward them
        # directly.
        addrs = self._lift_attrs(stmt.locations, LocationAddress)
        new = move.LocalR(
            self.state,
            axis_angle=stmt.axis_angle,
            rotation_angle=stmt.rotation_angle,
            location_addresses=addrs,
        )
        new.insert_before(stmt)
        self.state = new.result
        to_delete.append(stmt)

    @_rewrite.register(stack_move.LocalRz)
    def _(self, stmt: stack_move.LocalRz, to_delete: list[ir.Statement]) -> None:
        assert self.state is not None
        addrs = self._lift_attrs(stmt.locations, LocationAddress)
        new = move.LocalRz(
            self.state,
            rotation_angle=stmt.rotation_angle,
            location_addresses=addrs,
        )
        new.insert_before(stmt)
        self.state = new.result
        to_delete.append(stmt)

    @_rewrite.register(stack_move.GlobalR)
    def _(self, stmt: stack_move.GlobalR, to_delete: list[ir.Statement]) -> None:
        assert self.state is not None
        new = move.GlobalR(
            self.state,
            axis_angle=stmt.axis_angle,
            rotation_angle=stmt.rotation_angle,
        )
        new.insert_before(stmt)
        self.state = new.result
        to_delete.append(stmt)

    @_rewrite.register(stack_move.GlobalRz)
    def _(self, stmt: stack_move.GlobalRz, to_delete: list[ir.Statement]) -> None:
        assert self.state is not None
        new = move.GlobalRz(self.state, rotation_angle=stmt.rotation_angle)
        new.insert_before(stmt)
        self.state = new.result
        to_delete.append(stmt)

    @_rewrite.register(stack_move.Measure)
    def _(self, stmt: stack_move.Measure, to_delete: list[ir.Statement]) -> None:
        assert self.state is not None
        # Zones are direct operands on stack_move.Measure (matching the
        # Rust validator's sim_measure: pop `arity` zones, push `arity`
        # futures). Lift each zone SSA operand back to its ZoneAddress
        # attribute and deduplicate by zone id, preserving first-seen
        # order, so move.Measure receives the distinct set as an
        # attribute tuple.
        zone_attrs = self._lift_attrs(stmt.zones, ZoneAddress)
        seen_zone_ids: list[int] = []
        for zone in zone_attrs:
            if zone.zone_id not in seen_zone_ids:
                seen_zone_ids.append(zone.zone_id)
        distinct_zones = tuple(ZoneAddress(zid) for zid in seen_zone_ids)

        # Emit move.Measure with the attribute-tuple zones. move.Measure
        # is a StatefulStatement subclass, so it produces two results:
        # the inherited new-state (threaded forward) and the measurement
        # future.
        new = move.Measure(self.state, zone_addresses=distinct_zones)
        new.insert_before(stmt)
        self.state = new.result

        # stack_move.Measure produces `arity` MeasurementFutureType
        # results (one per input zone), while move.Measure produces a
        # single measurement future covering all distinct zones.
        # Redirect every per-zone future to the single move.Measure
        # future — safe under measure_lower's single-final-measurement
        # invariant (no two consumers survive lowering).
        for stmt_result in stmt.results:
            stmt_result.replace_by(new.future)
        to_delete.append(stmt)

    @_rewrite.register(stack_move.AwaitMeasure)
    def _(self, stmt: stack_move.AwaitMeasure, to_delete: list[ir.Statement]) -> None:
        from kirin.dialects import ilist

        # v1 stub: emit an empty ilist as a placeholder for the measurement
        # result array. The future operand (stmt.future) is already rewired
        # to the move.Measure.future result by _rewrite_Measure, but it's
        # dropped here — proper lowering would need per-location
        # GetFutureResult expansion, which isn't wired up in this PR. See
        # follow-up issue.
        placeholder = ilist.New(values=())
        placeholder.insert_before(stmt)
        stmt.result.replace_by(placeholder.result)
        to_delete.append(stmt)

    # ── Arrays / annotations ──────────────────────────────────────────

    @_rewrite.register(stack_move.NewArray)
    def _(self, stmt: stack_move.NewArray, to_delete: list[ir.Statement]) -> None:
        from kirin.dialects import ilist

        # Forward element SSA operands directly into the ilist.New. The
        # bytecode's new_array pops dim0 * max(dim1, 1) values as the
        # array's initial elements (per the Rust validator) — they've
        # already been rewired by earlier replace_by calls on their
        # defining Const* / ilist.New / py.Constant predecessors, so
        # stmt.values points at valid target-dialect SSA values.
        new = ilist.New(values=tuple(stmt.values))
        new.insert_before(stmt)
        stmt.result.replace_by(new.result)
        to_delete.append(stmt)

    @_rewrite.register(stack_move.GetItem)
    def _(self, stmt: stack_move.GetItem, to_delete: list[ir.Statement]) -> None:
        from kirin.dialects.py import indexing

        # Chained single-dim indexing: for each index SSA, emit
        # py.indexing.GetItem(obj=current, index=idx). The final result
        # replaces the stack_move.GetItem result for all downstream uses.
        current = stmt.array
        for idx_ssa in stmt.indices:
            gi = indexing.GetItem(obj=current, index=idx_ssa)
            gi.insert_before(stmt)
            current = gi.result
        stmt.result.replace_by(current)
        to_delete.append(stmt)

    @_rewrite.register(stack_move.SetDetector)
    def _(self, stmt: stack_move.SetDetector, to_delete: list[ir.Statement]) -> None:
        from bloqade.decoders.dialects import annotate
        from kirin.dialects import ilist

        # annotate.SetDetector requires a coordinates ilist; v1 emits an
        # empty ilist. If coordinate provenance is added to stack_move
        # later, thread it through here.
        empty_coords = ilist.New(values=())
        empty_coords.insert_before(stmt)
        new = annotate.stmts.SetDetector(
            measurements=stmt.array,
            coordinates=empty_coords.result,
        )
        new.insert_before(stmt)
        stmt.result.replace_by(new.result)
        to_delete.append(stmt)

    @_rewrite.register(stack_move.SetObservable)
    def _(self, stmt: stack_move.SetObservable, to_delete: list[ir.Statement]) -> None:
        from bloqade.decoders.dialects import annotate

        new = annotate.stmts.SetObservable(measurements=stmt.array)
        new.insert_before(stmt)
        stmt.result.replace_by(new.result)
        to_delete.append(stmt)

    @_rewrite.register(stack_move.CZ)
    def _(self, stmt: stack_move.CZ, to_delete: list[ir.Statement]) -> None:
        assert self.state is not None
        (zone,) = self._lift_attrs((stmt.zone,), ZoneAddress)
        new = move.CZ(self.state, zone_address=zone)
        new.insert_before(stmt)
        self.state = new.result
        to_delete.append(stmt)
