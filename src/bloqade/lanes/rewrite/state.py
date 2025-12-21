from dataclasses import dataclass, field
from functools import singledispatchmethod

from kirin import ir
from kirin.dialects import cf
from kirin.rewrite.abc import RewriteResult, RewriteRule

from bloqade.lanes.dialects import move
from bloqade.lanes.types import StateType


@dataclass
class FixUpStateFlow(RewriteRule):
    current_states: ir.SSAValue | None = field(default=None, init=False)

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        return self.rewrite_node(node)

    @singledispatchmethod
    def rewrite_node(self, node: ir.Statement) -> RewriteResult:
        return RewriteResult()

    @rewrite_node.register(move.Load)
    def rewrite_Load(self, node: move.Load):
        if len(uses := node.result.uses) != 1:
            # Something is wrong, we cannot fix up
            return RewriteResult()

        (use,) = uses

        if not isinstance(stmt := use.stmt, move.StatefulStatement):
            return RewriteResult()

        if self.current_states is None:
            self.current_states = stmt.result
            return RewriteResult()
        else:
            stmt.current_state.replace_by(self.current_states)
            self.current_states = stmt.result
            return RewriteResult(has_done_something=True)

    @rewrite_node.register(move.Store)
    def rewrite_Store(self, node: move.Store):
        if self.current_states is None:
            # Something is wrong, we cannot fix up
            return RewriteResult()

        node.current_state.replace_by(self.current_states)
        self.current_state = None
        return RewriteResult(has_done_something=True)

    @rewrite_node.register(cf.Branch)
    def rewrite_Branch(self, node: cf.Branch):
        if self.current_states is None:
            return RewriteResult()

        node.replace_by(
            cf.Branch(
                successor=node.successor,
                arguments=(self.current_states,) + node.arguments,
            )
        )
        self.current_states = None
        return RewriteResult(has_done_something=True)

    @rewrite_node.register(cf.ConditionalBranch)
    def rewrite_ConditionalBranch(self, node: cf.ConditionalBranch):
        if self.current_states is None:
            return RewriteResult()

        node.replace_by(
            cf.ConditionalBranch(
                cond=node.cond,
                then_successor=node.then_successor,
                then_arguments=(self.current_states,) + node.then_arguments,
                else_successor=node.else_successor,
                else_arguments=(self.current_states,) + node.else_arguments,
            )
        )

        self.current_states = None
        return RewriteResult(has_done_something=True)

    def is_entry_block(self, node: ir.Block) -> bool:
        if (parent_stmt := node.parent_stmt) is None:
            return False

        callable_stmt_trait = parent_stmt.get_trait(ir.CallableStmtInterface)

        if callable_stmt_trait is None:
            return False

        parent_region = callable_stmt_trait.get_callable_region(parent_stmt)
        return parent_region._block_idx[node] == 0

    def rewrite_Block(self, node: ir.Block):
        if self.is_entry_block(node):
            return RewriteResult()

        if self.current_states is not None:
            # something has gone wrong, we cannot fix up
            return RewriteResult()

        self.current_states = node.args.insert_from(0, StateType, "current_state")
        return RewriteResult(has_done_something=True)
