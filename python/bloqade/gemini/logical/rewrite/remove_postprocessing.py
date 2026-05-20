from dataclasses import dataclass, field

from bloqade.types import MeasurementResultType
from kirin import types
from kirin.dialects import func, ilist
from kirin.ir import Method
from kirin.ir.nodes.stmt import Statement
from kirin.passes import Pass, TypeInfer
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


class _DeleteTerminalMeasure(RewriteRule):
    """Deletes the terminal measurement after the post processing
    has been deleted. This MUST happen after _DeleteBelowTerminalMeasure
    has been run on the kernel.
    """

    def rewrite_Statement(self, node: Statement) -> RewriteResult:
        # statement is not a terminal measurement, skip
        if not isinstance(node, func.Function):
            return RewriteResult()

        last_stmt = node.body.blocks[-1].last_stmt

        if isinstance(last_stmt, TerminalLogicalMeasurement):
            # In this case the _DeleteBelowTerminalMeasure
            # Has deleted the return and left only the Terminal measurement
            # we delete this and add back a return value
            (none_stmt := func.ConstantNone()).insert_after(last_stmt)
            func.Return(none_stmt).insert_before(last_stmt)
            last_stmt.delete()
            node.signature = func.Signature(node.signature.inputs, types.NoneType)

            return RewriteResult(has_done_something=True)

        elif isinstance(last_stmt, func.Return) and isinstance(
            owner := last_stmt.value.owner, TerminalLogicalMeasurement
        ):
            # on this case there is no post processing didn't
            # do anything so we need to just replace the TerminalLogicalMeasurement with None
            owner.replace_by(func.ConstantNone())
            node.signature = func.Signature(node.signature.inputs, types.NoneType)
            return RewriteResult(has_done_something=True)

        return RewriteResult()


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


@dataclass
class RemovePostProcessing(Pass):
    """Remove post-processing steps, i.e. everything below a TerminalMeasure statement
    in a logical kernel.

    The return value is changed to return the TerminalMeasure result.

    if delete_terminal_measure is true the return value is None and the Terminal measurement
    is deleted.

    **NOTE**: Expects a flat logical kernel. Otherwise may lead to incorrect results.
    """

    delete_terminal_measure: bool = field(kw_only=True, default=False)

    def unsafe_run(self, mt: Method) -> RewriteResult:
        result = Walk(_DeleteBelowTerminalMeasure()).rewrite(mt.code)
        if self.delete_terminal_measure:
            result = result.join(_DeleteTerminalMeasure().rewrite(mt.code))
        else:
            result = result.join(Walk(_InsertTerminalMeasureReturn()).rewrite(mt.code))

        TypeInfer(mt.dialects)(mt)
        return result
