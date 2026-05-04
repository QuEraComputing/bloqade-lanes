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

from kirin import ir
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
