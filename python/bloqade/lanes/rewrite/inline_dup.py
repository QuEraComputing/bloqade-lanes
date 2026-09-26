"""InlineDup: canonicalise ``stack_move.Dup`` out of the IR.

A ``Dup`` is an identity: both of its results, ``top`` and ``below``, are its
operand. In SSA form that copy says nothing a second use of the operand does
not, so this rule gives each copy's consumers the operand itself and deletes
the ``Dup`` — as kirin's ``InlineAlias`` does for ``py.Alias``. It works for
any operand, where ``ConstantFold`` + DCE only remove a ``Dup`` of a constant.

What the ``Dup`` did on the stack is ``stackify``'s to redo: the operand now
has a consumer per copy, so it is spilled to a local and reloaded for each.
That is correct but is not the program that was decoded, which is why
``stackify`` does not run this itself; run it where the IR should say what a
program computes rather than how it keeps its stack.
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
