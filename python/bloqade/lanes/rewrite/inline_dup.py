"""InlineDup: canonicalise ``stack_move.Dup`` out of the IR.

A ``Dup`` is an identity: both of its results, ``top`` and ``below``, are its
operand. In SSA form that copy says nothing a second use of the operand does
not, so this rule gives each copy's consumers the operand itself and deletes
the ``Dup`` — as kirin's ``InlineAlias`` does for ``py.Alias``. It is the
one thing that does: a ``Dup`` is not ``Pure``, so DCE and ``ConstantFold``
leave it. ``load_program(..., inline_dup=True)`` runs it on a decoded kernel.

``RewriteStackMoveToMove`` lowers each ``Dup`` with it: ``move`` keeps no
stack, so a copy means nothing there. ``stackify`` uses it for one case only,
and only in a block that is not already a stack program: a ``Dup`` of a
constant, since a copy of a constant is that constant, which it re-creates in
front of each consumer. It keeps every other ``Dup``. Inlining those too
would still be correct, since the operand then has a consumer per copy and is
moved to a local and reloaded for each, but it would spend a local where the
``dup`` needed none.
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
