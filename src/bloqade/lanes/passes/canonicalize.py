from dataclasses import dataclass

from bloqade.gemini.dialects.logical.stmts import Initialize
from bloqade.native.dialects.gate import stmts as gates
from kirin import ir
from kirin.dialects import ilist, py
from kirin.passes.hint_const import HintConst
from kirin.rewrite import abc

from bloqade import qubit


class HoistConstants(abc.RewriteRule):
    """This rewrite rule hoists all constant values to the top of the kernel."""

    TYPES = (
        gates.CZ,
        gates.R,
        gates.Rz,
        Initialize,
        qubit.stmts.Measure,
        qubit.stmts.Reset,
    )

    def is_pure(self, node: ir.Statement) -> bool:
        return (
            node.has_trait(ir.Pure)
            or (maybe_pure := node.get_trait(ir.MaybePure)) is not None
            and maybe_pure.is_pure(node)
        )

    def rewrite_Statement(self, node: ir.Statement) -> abc.RewriteResult:
        if not (
            isinstance(node, py.Constant)
            and (parent_block := node.parent_block) is not None
            and (first_stmt := parent_block.first_stmt) is not None
        ):
            return abc.RewriteResult()

        node.detach()
        node.insert_before(first_stmt)

        return abc.RewriteResult(has_done_something=True)
