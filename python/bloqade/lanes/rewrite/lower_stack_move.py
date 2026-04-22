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
from typing import TypeVar

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
    - ssa_to_attr: stack_move SSA → Kirin attribute value (ir.Data) for
      operands that need to be lifted into target-dialect attributes
      (addresses, rotation angles). Every Kirin attribute obeys the
      ir.Data interface, so the map's value type is ir.Data directly.
      SSA-to-attribute can't be expressed through SSA rewiring because
      attributes aren't SSA values, so we carry an explicit mapping.
    - state: the current StateType SSA value in the target IR.

    For SSA-valued outputs (arrays, futures, detectors, observables,
    constants that emit py.Constant), we use the Kirin idiom
    `old_ssa.replace_by(new_ssa)` to redirect all uses in place — no
    second mapping needed. This matches state.RewriteLoadStore's
    `next_use.replace_by(current_use)` pattern.
    """

    ssa_to_attr: dict[ir.SSAValue, ir.Data] = field(default_factory=dict)
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
