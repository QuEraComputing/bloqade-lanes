from dataclasses import dataclass

from kirin import ir, types
from kirin.dialects import ilist, py
from kirin.rewrite import abc as rewrite_abc

from bloqade.lanes.dialects import move
from bloqade.lanes.heuristics.fixed import get_coordinate
from bloqade.lanes.layout.encoding import MoveType

from . import stmts


@dataclass
class RewriteMoves(rewrite_abc.RewriteRule):

    def get_address_info(self, node: move.Move):

        move_type = node.lanes[0].move_type
        direction = node.lanes[0].direction
        word = node.lanes[0].word_id
        bus_id = node.lanes[0].bus_id

        y_positions = [get_coordinate(lane.site_id)[1] for lane in node.lanes]

        y_mask = ilist.IList([i in y_positions for i in range(5)])

        (y_mask_stmt := py.Constant(y_mask)).insert_before(node)

        return move_type, y_mask_stmt.result, word, bus_id, direction

    def rewrite_Statement(self, node: ir.Statement):
        if not isinstance(node, move.Move):
            return rewrite_abc.RewriteResult()

        if len(node.lanes) == 0:
            node.delete()
            return rewrite_abc.RewriteResult(has_done_something=True)

        # This assumes validation has already occurred so only valid moves are present
        move_type, y_mask_ref, word, bus_id, direction = self.get_address_info(node)
        node.result.replace_by(node.current_state)
        if move_type is MoveType.SITE:
            node.replace_by(
                stmts.SiteBusMove(
                    current_state=node.current_state,
                    y_mask=y_mask_ref,
                    word=word,
                    bus_id=bus_id,
                    direction=direction,
                )
            )
        elif move_type is MoveType.WORD:
            node.replace_by(
                stmts.WordBusMove(
                    current_state=node.current_state,
                    y_mask=y_mask_ref,
                    direction=direction,
                )
            )
        else:
            raise AssertionError("Unsupported move type for rewrite")

        return rewrite_abc.RewriteResult(has_done_something=True)


class RewriteFill(rewrite_abc.RewriteRule):
    def rewrite_Statement(self, node: ir.Statement):
        if not isinstance(node, move.Fill):
            return rewrite_abc.RewriteResult()

        logical_addresses_ilist = ilist.IList(
            [(addr.word_id, addr.site_id) for addr in node.location_addresses],
            elem=types.Tuple[types.Int, types.Int],
        )
        (logical_addresses_stmt := py.Constant(logical_addresses_ilist)).insert_before(
            node
        )
        node.result.replace_by(node.current_state)
        node.replace_by(
            stmts.Fill(
                current_state=node.current_state,
                logical_addresses=logical_addresses_stmt.result,
            )
        )
        return rewrite_abc.RewriteResult(has_done_something=True)


class RewriteInitialize(rewrite_abc.RewriteRule):
    def rewrite_Statement(self, node: ir.Statement):
        if not isinstance(node, move.LogicalInitialize):
            return rewrite_abc.RewriteResult()

        groups: dict[
            tuple[ir.SSAValue, ir.SSAValue, ir.SSAValue], list[tuple[int, int]]
        ] = {}
        for location_addr, theta, phi, lam in zip(
            node.location_addresses, node.thetas, node.phis, node.lams
        ):
            groups.setdefault((theta, phi, lam), []).append(
                (location_addr.word_id, location_addr.site_id)
            )

        def make_ilist(list_of_tuples: list[tuple[int, int]]) -> ilist.IList:
            return ilist.IList(list_of_tuples, elem=types.Tuple[types.Int, types.Int])

        thetas, phis, lams = ([key[i] for key in groups.keys()] for i in range(3))
        logical_addresses = ilist.IList(list(map(make_ilist, groups.values())))

        (thetas_stmt := ilist.New(thetas)).insert_before(node)
        (phis_stmt := ilist.New(phis)).insert_before(node)
        (lams_stmt := ilist.New(lams)).insert_before(node)
        (logical_addresses_stmt := py.Constant(logical_addresses)).insert_before(node)
        node.result.replace_by(node.current_state)
        node.replace_by(
            stmts.LogicalInitialize(
                current_state=node.current_state,
                thetas=thetas_stmt.result,
                phis=phis_stmt.result,
                lams=lams_stmt.result,
                logical_addresses=logical_addresses_stmt.result,
            )
        )
        return rewrite_abc.RewriteResult(has_done_something=True)
