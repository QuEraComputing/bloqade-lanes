from dataclasses import dataclass, field

from kirin import decl, ir, types
from kirin.decl import info
from kirin.dialects import ilist, py
from kirin.rewrite import abc as rewrite_abc

from bloqade.lanes.dialects import move
from bloqade.lanes.layout.arch import ArchSpec
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
    y_positions: ir.SSAValue = info.argument(ilist.IListType[types.Int])
    word: int = info.attribute()
    bus_id: int = info.attribute()
    direction: Direction = info.attribute()


@decl.statement(dialect=dialect)
class WordBusMove(ir.Statement):
    y_positions: ir.SSAValue = info.argument(ilist.IListType[types.Int])
    direction: Direction = info.attribute()


@dataclass
class RewriteMoves(rewrite_abc.RewriteRule):
    """Re"""

    arch_spec: ArchSpec = field(default_factory=get_arch_spec)

    def get_address_info(self, node: move.Move):

        move_type = node.lanes[0].move_type
        direction = node.lanes[0].direction
        word = node.lanes[0].word_id
        bus_id = node.lanes[0].bus_id

        y_positions_list = ilist.IList([lane.site_id // 2 for lane in node.lanes])
        (y_positions_stmt := py.Constant(y_positions_list)).insert_before(node)

        return move_type, y_positions_stmt.result, word, bus_id, direction

    def rewrite_Statement(self, node: ir.Statement):
        if not (isinstance(node, move.Move) and len(node.lanes) > 0):
            return rewrite_abc.RewriteResult()

        # This assumes validation has already occurred so only valid moves are present
        move_type, y_positions, word, bus_id, direction = self.get_address_info(node)

        if move_type is MoveType.SITE:
            node.replace_by(
                SiteBusMove(
                    y_positions,
                    word=word,
                    bus_id=bus_id,
                    direction=direction,
                )
            )
        elif move_type is MoveType.WORD:
            node.replace_by(
                WordBusMove(
                    y_positions,
                    direction=direction,
                )
            )
        else:
            raise AssertionError("Unsupported move type for rewrite")

        return rewrite_abc.RewriteResult(has_done_something=True)
