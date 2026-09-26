"""InlineDup: canonicalise ``stack_move.Dup`` out of the IR.

A ``Dup`` is an identity: both of its results, ``top`` and ``below``, are its
operand. In SSA form that copy says nothing a second use of the operand does
not, so this rule gives each copy's consumers the operand itself and deletes
the ``Dup`` — as kirin's ``InlineAlias`` does for ``py.Alias``. It works for
any operand, where ``ConstantFold`` + DCE only remove a ``Dup`` of a constant.

``RewriteStackMoveToMove`` runs it first: ``move`` keeps no stack, so a copy
means nothing there. ``stackify`` does not — it would still be correct, since
the operand now has a consumer per copy and is spilled to a local and
reloaded for each, but a decoded ``dup`` would no longer come back out as
itself.
"""

from dataclasses import dataclass

from kirin import ir
from kirin.rewrite.abc import RewriteResult, RewriteRule

from bloqade.lanes.dialects import stack_move


@dataclass
class InlineDup(RewriteRule):
    """Replace every use of a ``Dup``'s copies by its operand, and delete it."""

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        if not isinstance(node, stack_move.Dup):
            return RewriteResult()
        node.top.replace_by(node.value)
        node.below.replace_by(node.value)
        node.delete()
        return RewriteResult(has_done_something=True)
