from dataclasses import dataclass

from bloqade.native.dialects.gate import stmts as gates
from kirin import ir, passes, rewrite
from kirin.passes.hint_const import HintConst
from kirin.rewrite import abc

from bloqade import qubit


class HoistClassicalStatements(abc.RewriteRule):
    """This rewrite rule shift any classical statements that are pure
    (quantum statements are never pure) to the beginning of the block.
    swapping the other with quantum statements. This is useful after
    rewriting the native operations to placement operations,
    so that we can merge the placement regions together.

    Note that this rule also works with subroutines that contain
    quantum statements because these are also not pure

    """

    TYPES = (
        gates.CZ,
        gates.R,
        gates.Rz,
        qubit.stmts.Measure,
        qubit.stmts.Reset,
        qubit.stmts.New,
    )

    def is_pure(self, node: ir.Statement) -> bool:
        return (
            node.has_trait(ir.Pure)
            or (maybe_pure := node.get_trait(ir.MaybePure)) is not None
            and maybe_pure.is_pure(node)
        )

    def rewrite_Statement(self, node: ir.Statement) -> abc.RewriteResult:
        if not (
            isinstance(node, self.TYPES)
            and (next_node := node.next_stmt) is not None
            and not next_node.has_trait(ir.IsTerminator)
            and set(node.results).isdisjoint(next_node.args)
            and self.is_pure(next_node)
        ):
            return abc.RewriteResult()

        next_node.detach()
        next_node.insert_before(node)
        return abc.RewriteResult(has_done_something=True)


@dataclass
class CanonicalizeNative(passes.Pass):

    def unsafe_run(self, mt: ir.Method) -> abc.RewriteResult:
        result = HintConst(mt.dialects)(mt)
        result = rewrite.Walk(HoistClassicalStatements()).rewrite(mt.code).join(result)

        return result
