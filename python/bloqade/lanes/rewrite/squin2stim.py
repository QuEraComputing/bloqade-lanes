from kirin.dialects import func
from kirin.ir import Statement
from kirin.rewrite.abc import RewriteResult, RewriteRule


class RemoveReturn(RewriteRule):
    def rewrite_Statement(self, node: Statement) -> RewriteResult:
        if not isinstance(node, func.Return):
            return RewriteResult()

        (none_stmt := func.ConstantNone()).insert_before(node)
        node.replace_by(func.Return(none_stmt.result))
        return RewriteResult(has_done_something=True)
