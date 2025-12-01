from kirin import decl, ir, types
from kirin.decl import info
from kirin.dialects import ilist, py
from kirin.rewrite import abc as rewrite_abc

from bloqade.lanes.dialects import move
from bloqade.lanes.layout.encoding import (
    Direction,
    LaneAddress,
    SiteLaneAddress,
    WordLaneAddress,
)

from .impls import generate_arch


def get_arch_spec():
    return generate_arch(hypercube_dims=1, word_size_y=5)


dialect = ir.Dialect("gemini.logical")


@decl.statement(dialect=dialect)
class SiteBusMove(ir.Statement):
    y_positions: ir.SSAValue = info.argument(ilist.IListType[types.Int])
    word: int = info.attribute()
    bus_id: int = info.attribute()
    direction: Direction = info.attribute()


@decl.statement(dialect=dialect)
class WordBusMove(ir.Statement):
    y_positions: ir.SSAValue = info.argument(ilist.IListType[types.Int])
    direction: Direction = info.attribute()


class RewriteMoves(rewrite_abc.RewriteRule):

    def get_y_positions(self, lane: LaneAddress):
        return lane.site_id // 2

    def get_address_info(self, node: move.Move):

        if len(node.lanes) == 0:
            return None, None, None, None
        direction = node.lanes[0].direction
        word = node.lanes[0].word_id
        bus_id = node.lanes[0].bus_id

        if not all(lane.word_id == word for lane in node.lanes):
            word = None
        if not all(lane.bus_id == bus_id for lane in node.lanes):
            bus_id = None
        if not all(lane.direction == direction for lane in node.lanes):
            direction = None

        y_positions_list = ilist.IList(
            [self.get_y_positions(lane) for lane in node.lanes]
        )
        (y_positions_stmt := py.Constant(y_positions_list)).insert_before(node)

        return y_positions_stmt.result, word, bus_id, direction

    def rewrite_Statement(self, node: ir.Statement):
        if not isinstance(node, move.Move):
            return rewrite_abc.RewriteResult()

        y_positions, word, bus_id, direction = self.get_address_info(node)

        if y_positions is None or word is None or bus_id is None or direction is None:
            return rewrite_abc.RewriteResult()

        if types.is_tuple_of(node.lanes, SiteLaneAddress):
            node.replace_by(
                SiteBusMove(
                    y_positions,
                    word=word,
                    bus_id=bus_id,
                    direction=direction,
                )
            )
            return rewrite_abc.RewriteResult(has_done_something=True)
        elif types.is_tuple_of(node.lanes, WordLaneAddress):
            if word != 0 or bus_id != 0:
                return rewrite_abc.RewriteResult()

            node.replace_by(
                WordBusMove(
                    y_positions,
                    direction=direction,
                )
            )
            return rewrite_abc.RewriteResult(has_done_something=True)
        else:
            return rewrite_abc.RewriteResult()
