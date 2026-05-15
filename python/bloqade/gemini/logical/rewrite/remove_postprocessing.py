from dataclasses import dataclass

from bloqade.types import MeasurementResultType
from kirin import types
from kirin.dialects import func, ilist
from kirin.ir import Method
from kirin.ir.nodes.stmt import Statement
from kirin.passes import Pass
from kirin.rewrite import Walk
from kirin.rewrite.abc import RewriteResult, RewriteRule

from bloqade.gemini.logical.dialects.operations.stmts import TerminalLogicalMeasurement


@dataclass
class _DeleteBelowTerminalMeasure(RewriteRule):
    has_seen_terminal_measure: bool = False
    returns_terminal_measure: bool = False

    def rewrite_Statement(self, node: Statement) -> RewriteResult:
        if isinstance(node, func.Function):
            # NOTE: this is a flat kernel, so the only function we should
            # have is the callable_region of the kernel; don't delete that
            return RewriteResult()

        if isinstance(node, TerminalLogicalMeasurement):
            self.has_seen_terminal_measure = True
            for use in node.result.uses:
                if isinstance(use.stmt, func.Return):
                    self.returns_terminal_measure = True
            return RewriteResult()

        if not self.has_seen_terminal_measure:
            # we are still above the terminal measurement
            return RewriteResult()

        if isinstance(node, func.Return) and self.returns_terminal_measure:
            return RewriteResult()

        # NOTE: we need to use unsafe deletion here since the node may have
        # uses below, but any statements that use it will be deleted, except
        # for the return which is handled above
        node.delete(safe=False)
        return RewriteResult(has_done_something=True)


class _InsertTerminalMeasureReturn(RewriteRule):
    terminal_measure_type = ilist.IListType[
        ilist.IListType[MeasurementResultType, types.Any], types.Any
    ]

    def rewrite_Statement(self, node: Statement) -> RewriteResult:
        if isinstance(node, func.Function):
            if node.signature.output.is_subseteq(self.terminal_measure_type):
                return RewriteResult()
            # need to update signature to return None
            new_signature = func.Signature(
                node.signature.inputs, self.terminal_measure_type
            )
            node.signature = new_signature
            return RewriteResult(has_done_something=True)

        if not isinstance(node, TerminalLogicalMeasurement):
            return RewriteResult()

        for use in node.result.uses:
            if isinstance(use.stmt, func.Return):
                return RewriteResult()

        ret = func.Return(node.result)
        ret.insert_after(node)

        return RewriteResult(has_done_something=True)


class RemovePostProcessing(Pass):
    """Remove post-processing steps, i.e. everything below a TerminalMeasure statement
    in a logical kernel.

    The return value is changed to return the TerminalMeasure result.

    **NOTE**: Expects a flat logical kernel. Otherwise may lead to incorrect results.
    """

    def unsafe_run(self, mt: Method) -> RewriteResult:
        result = Walk(_DeleteBelowTerminalMeasure()).rewrite(mt.code)
        Walk(_InsertTerminalMeasureReturn()).rewrite(mt.code)
        return result
