from dataclasses import dataclass
from functools import singledispatchmethod

from kirin import ir
from kirin.dialects import cf
from kirin.rewrite.abc import RewriteResult, RewriteRule

from bloqade.lanes.dialects import move
from bloqade.lanes.types import StateType


@dataclass
class RewriteLoadStore(RewriteRule):

    def rewrite_Block(self, node: ir.Block) -> RewriteResult:
        if len(node.args) == 0 or not (
            current_use := node.args[0]
        ).type.is_structurally_equal(StateType):
            current_use = None

        to_delete: list[ir.Statement] = []
        state_stored = False

        for stmt in node.stmts:

            consume_trait = stmt.get_trait(move.ConsumesState)
            if consume_trait is not None and consume_trait.terminates:
                current_use = None

            emit_trait = stmt.get_trait(move.EmitsState)
            if emit_trait is not None:
                next_use = emit_trait.get_state_result(stmt)
                if emit_trait.originates and current_use is not None:
                    next_use.replace_by(current_use)
                    to_delete.append(stmt)
                else:
                    state_stored = False
                    current_use = next_use

            if isinstance(stmt, move.Store):
                if state_stored:
                    to_delete.append(stmt)
                else:
                    state_stored = True

        for stmt in to_delete:
            stmt.delete()

        return RewriteResult(has_done_something=True)


class InsertBlockArgs(RewriteRule):
    def rewrite_Statement(self, node: ir.Statement):
        callable_stmt_trait = node.get_trait(ir.CallableStmtInterface)

        if callable_stmt_trait is None:
            return RewriteResult()

        region = callable_stmt_trait.get_callable_region(node)

        has_done_something = False
        for block in region.blocks[1:]:
            if len(args := block.args) == 0 or not args[0].type.is_structurally_equal(
                StateType
            ):
                block.args.insert_from(0, StateType, "current_state")
                has_done_something = True

        return RewriteResult(has_done_something=has_done_something)


@dataclass
class RewriteBranches(RewriteRule):

    def rewrite_Statement(self, node: ir.Statement):
        return self.rewrite_node(node)

    @singledispatchmethod
    def rewrite_node(self, node: ir.Statement) -> RewriteResult:
        return RewriteResult()

    @rewrite_node.register(cf.Branch)
    def rewrite_Branch(self, node: cf.Branch):
        (current_state := move.Load()).insert_before(node)
        node.replace_by(
            cf.Branch(
                successor=node.successor,
                arguments=(current_state.result,) + node.arguments,
            )
        )
        return RewriteResult(has_done_something=True)

    @rewrite_node.register(cf.ConditionalBranch)
    def rewrite_ConditionalBranch(self, node: cf.ConditionalBranch):
        (current_state := move.Load()).insert_before(node)
        node.replace_by(
            cf.ConditionalBranch(
                cond=node.cond,
                then_successor=node.then_successor,
                then_arguments=(current_state.result,) + node.then_arguments,
                else_successor=node.else_successor,
                else_arguments=(current_state.result,) + node.else_arguments,
            )
        )
        return RewriteResult(has_done_something=True)
