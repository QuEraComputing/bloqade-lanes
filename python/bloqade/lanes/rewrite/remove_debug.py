from dataclasses import dataclass

from kirin import ir
from kirin.dialects import debug
from kirin.rewrite.abc import RewriteResult, RewriteRule


@dataclass
class RemoveDebugStatements(RewriteRule):
    """Delete all kirin debug.Info statements."""

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        if not isinstance(node, debug.Info):
            return RewriteResult()
        node.delete()
        return RewriteResult(has_done_something=True)
