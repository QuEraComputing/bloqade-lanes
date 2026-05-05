"""ReorderStaticPlacement rewrite rule."""

from dataclasses import dataclass
from typing import Callable

from kirin import ir
from kirin.rewrite import abc

from bloqade.lanes.dialects import place
from bloqade.lanes.rewrite.reorder_static_placement.types import _SchedulableStmt
from bloqade.lanes.types import StateType


@dataclass
class ReorderStaticPlacement(abc.RewriteRule):
    """Reorder quantum statements within a StaticPlacement using a pluggable policy.

    The policy receives all schedulable statements from the body (R, Rz, CZ,
    Initialize, EndMeasure — everything except the trailing Yield) and returns
    them in the desired order.  Barrier handling (Initialize, EndMeasure) is the
    policy's responsibility; ``asap_reorder_policy`` segments on barriers and
    schedules each segment independently.

    If the body contains any statement type outside that supported set the
    rewriter skips the node rather than silently dropping unknown statements.
    """

    reorder_policy: Callable[[list[_SchedulableStmt]], list[_SchedulableStmt]]

    def rewrite_Statement(self, node: ir.Statement) -> abc.RewriteResult:
        if not isinstance(node, place.StaticPlacement):
            return abc.RewriteResult()

        body_block = node.body.blocks[0]
        old_yield = body_block.last_stmt
        assert isinstance(old_yield, place.Yield)

        _supported = (place.R, place.Rz, place.CZ, place.Initialize, place.EndMeasure)
        stmts: list[_SchedulableStmt] = []
        for s in body_block.stmts:
            if isinstance(s, place.Yield):
                continue
            if not isinstance(s, _supported):
                return abc.RewriteResult()
            stmts.append(s)

        if not stmts:
            return abc.RewriteResult()

        new_stmts = self.reorder_policy(stmts)

        if [id(s) for s in new_stmts] == [id(s) for s in stmts]:
            return abc.RewriteResult()

        new_body = ir.Region(new_block := ir.Block())
        curr_state = new_block.args.append_from(StateType, "entry_state")

        for stmt in new_stmts:
            remapped = stmt.from_stmt(stmt, args=(curr_state, *stmt.args[1:]))
            new_block.stmts.append(remapped)
            curr_state = remapped.state_after
            for old_r, new_r in zip(stmt.results[1:], remapped.results[1:]):
                old_r.replace_by(new_r)

        new_block.stmts.append(place.Yield(curr_state, *old_yield.classical_results))

        new_sp = place.StaticPlacement(node.qubits, new_body)
        new_sp.insert_before(node)

        for old_r, new_r in zip(node.results, new_sp.results, strict=True):
            old_r.replace_by(new_r)

        node.delete()
        return abc.RewriteResult(has_done_something=True)
