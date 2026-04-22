"""lower_stack_move — in-place rewrite from stack_move → multi-dialect IR.

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
from typing import Any, TypeVar

from kirin import ir
from kirin.rewrite.abc import RewriteResult, RewriteRule

from bloqade.lanes.dialects import move, stack_move
from bloqade.lanes.utils import no_none_elements_tuple  # noqa: F401

# TypeVar and no_none_elements_tuple are consumed by the _lift_attrs helper
# introduced in Phase D2+ tasks; kept here so subsequent patches stay minimal.
T = TypeVar("T")


@dataclass
class LowerStackMove(RewriteRule):
    """Lower a stack_move block into a multi-dialect block in place.

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
            handler = getattr(self, f"_rewrite_{type(stmt).__name__}", None)
            if handler is None:
                # Non-stack_move statements (e.g. existing py.Constant) pass
                # through unchanged.
                continue
            handler(stmt, to_delete)

        for stmt in to_delete:
            stmt.delete()
        return RewriteResult(has_done_something=True)

    def _rewrite_Return(
        self, stmt: stack_move.Return, to_delete: list[ir.Statement]
    ) -> None:
        from kirin.dialects import func

        assert self.state is not None
        move.Store(self.state).insert_before(stmt)
        func.Return().insert_before(stmt)
        to_delete.append(stmt)

    def _rewrite_Halt(
        self, stmt: stack_move.Halt, to_delete: list[ir.Statement]
    ) -> None:
        from kirin.dialects import func

        assert self.state is not None
        move.Store(self.state).insert_before(stmt)
        func.Return().insert_before(stmt)
        to_delete.append(stmt)

    def _rewrite_ConstFloat(
        self, stmt: stack_move.ConstFloat, to_delete: list[ir.Statement]
    ) -> None:
        from kirin.dialects import py

        out = py.Constant(stmt.value)
        out.insert_before(stmt)
        # Redirect all SSA uses of the old stack_move.ConstFloat result to
        # the new py.Constant result in place.
        stmt.result.replace_by(out.result)
        # Consumers that need the raw float as an attribute (e.g.
        # _rewrite_LocalR building a theta= kwarg) look up ssa_to_attr.
        # The key is the new SSA, because replace_by rewired the operands
        # stored on downstream statements to point there.
        self.ssa_to_attr[out.result] = stmt.value
        to_delete.append(stmt)

    def _rewrite_ConstInt(
        self, stmt: stack_move.ConstInt, to_delete: list[ir.Statement]
    ) -> None:
        from kirin.dialects import py

        out = py.Constant(stmt.value)
        out.insert_before(stmt)
        stmt.result.replace_by(out.result)
        self.ssa_to_attr[out.result] = stmt.value
        to_delete.append(stmt)

    def _rewrite_ConstLoc(
        self, stmt: stack_move.ConstLoc, to_delete: list[ir.Statement]
    ) -> None:
        # Address constants stay as decoder attributes — downstream move.*
        # statements take them as attribute values, not SSA operands.
        # We track the raw attribute value for later attribute lifting.
        self.ssa_to_attr[stmt.result] = stmt.value
        to_delete.append(stmt)

    def _rewrite_ConstLane(
        self, stmt: stack_move.ConstLane, to_delete: list[ir.Statement]
    ) -> None:
        self.ssa_to_attr[stmt.result] = stmt.value
        to_delete.append(stmt)

    def _rewrite_ConstZone(
        self, stmt: stack_move.ConstZone, to_delete: list[ir.Statement]
    ) -> None:
        self.ssa_to_attr[stmt.result] = stmt.value
        to_delete.append(stmt)
