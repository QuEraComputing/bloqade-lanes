from dataclasses import dataclass

from kirin import decl, ir, rewrite, types
from kirin.decl import info
from kirin.dialects import ilist, py
from kirin.rewrite import abc as rewrite_abc

from bloqade.lanes.dialects import move
from bloqade.lanes.layout.encoding import (
    Direction,
    MoveType,
)

from .impls import generate_arch


def get_arch_spec():
    return generate_arch(hypercube_dims=1, word_size_y=5)


dialect = ir.Dialect("gemini.logical")


@decl.statement(dialect=dialect)
class SiteBusMove(ir.Statement):
    y_mask: ir.SSAValue = info.argument(ilist.IListType[types.Int])
    word: int = info.attribute()
    bus_id: int = info.attribute()
    direction: Direction = info.attribute()


@decl.statement(dialect=dialect)
class WordBusMove(ir.Statement):
    y_mask: ir.SSAValue = info.argument(ilist.IListType[types.Int])
    direction: Direction = info.attribute()


@dataclass
class RewriteMoves(rewrite_abc.RewriteRule):

    def get_address_info(self, node: move.Move):

        move_type = node.lanes[0].move_type
        direction = node.lanes[0].direction
        word = node.lanes[0].word_id
        bus_id = node.lanes[0].bus_id

        y_positions = [lane.site_id // 2 for lane in node.lanes]

        y_mask = ilist.IList([i in y_positions for i in range(5)])

        (y_mask_stmt := py.Constant(y_mask)).insert_before(node)

        return move_type, y_mask_stmt.result, word, bus_id, direction

    def rewrite_Statement(self, node: ir.Statement):
        if not (isinstance(node, move.Move) and len(node.lanes) > 0):
            return rewrite_abc.RewriteResult()

        # This assumes validation has already occurred so only valid moves are present
        move_type, y_mask_ref, word, bus_id, direction = self.get_address_info(node)

        if move_type is MoveType.SITE:
            node.replace_by(
                SiteBusMove(
                    y_mask_ref,
                    word=word,
                    bus_id=bus_id,
                    direction=direction,
                )
            )
        elif move_type is MoveType.WORD:
            node.replace_by(
                WordBusMove(
                    y_mask_ref,
                    direction=direction,
                )
            )
        else:
            raise AssertionError("Unsupported move type for rewrite")

        return rewrite_abc.RewriteResult(has_done_something=True)


class SpecializeGemini:

    def emit(self, mt: ir.Method, no_raise=True) -> ir.Method:
        out = mt.similar(dialects=mt.dialects.add(dialect))

        rewrite.Walk(RewriteMoves()).rewrite(out.code)

        if not no_raise:
            out.verify()

        return out
