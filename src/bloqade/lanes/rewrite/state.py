from dataclasses import dataclass
from functools import singledispatchmethod
from typing import TypeGuard

from kirin import ir
from kirin.dialects import cf
from kirin.rewrite.abc import RewriteResult, RewriteRule

from bloqade.lanes.dialects import move
from bloqade.lanes.types import StateType


@dataclass
class RewriteLoad(RewriteRule):

    @staticmethod
    def is_load(stmt: ir.Statement) -> TypeGuard[move.Load]:
        return isinstance(stmt, move.Load)

    def rewrite_Block(self, node: ir.Block) -> RewriteResult:
        if not (current_use := node.args[0]).type.is_structurally_equal(StateType):
            current_use = None

        to_delete = []
        for load_stmt in filter(self.is_load, node.stmts):
            result = load_stmt.result
            if current_use is not None:
                result.replace_by(current_use)
                to_delete.append(load_stmt)

            unique_stmts = set(use.stmt for use in result.uses)
            if len(unique_stmts) == 0:
                continue

            stmt = unique_stmts.pop()
            if isinstance(stmt, move.StatefulStatement):
                current_use = stmt.result
            else:
                current_use = None

        for load_stmt in to_delete:
            load_stmt.delete()

        return RewriteResult(has_done_something=True)


class InsertBlockArgs(RewriteRule):
    def rewrite_Statement(self, node: ir.Statement):
        callable_stmt_trait = node.get_trait(ir.CallableStmtInterface)

        if callable_stmt_trait is None:
            return RewriteResult()

        region = callable_stmt_trait.get_callable_region(node)

        for block in region.blocks[1:]:
            block.args.insert_from(0, StateType, "current_state")

        return RewriteResult(has_done_something=True)


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
