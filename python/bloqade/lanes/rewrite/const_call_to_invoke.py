from __future__ import annotations

from dataclasses import dataclass

from kirin import ir
from kirin.dialects import func, py
from kirin.rewrite.abc import RewriteResult, RewriteRule


@dataclass
class ConstCallToInvoke(RewriteRule):
    """Rewrite ``func.call`` of a ``py.constant`` method into ``func.invoke``.

    ``kirin.rewrite.Inline`` only inlines ``func.call`` of a lambda, and
    kirin's ``Call2Invoke`` reads const-prop hints, which are not set inside
    ``scf.for`` bodies. ``SquinToNative``'s ``GateRule`` emits exactly this
    pattern, so without this rule gate calls inside loops are never inlined.
    """

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        if not isinstance(node, func.Call) or node.kwargs:
            return RewriteResult()
        callee = node.callee
        if not isinstance(callee, ir.ResultValue):
            return RewriteResult()
        const = callee.stmt
        if not isinstance(const, py.Constant) or not isinstance(const.value, ir.PyAttr):
            return RewriteResult()
        method = const.value.data
        if not isinstance(method, ir.Method):
            return RewriteResult()

        invoke = func.Invoke(tuple(node.inputs), callee=method, purity=node.purity)
        invoke.result.name = node.result.name
        invoke.result.type = node.result.type
        node.replace_by(invoke)
        return RewriteResult(has_done_something=True)
