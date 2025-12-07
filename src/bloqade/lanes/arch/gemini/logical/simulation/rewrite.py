from dataclasses import dataclass

from kirin import ir
from kirin.rewrite import abc as rewrite_abc

from bloqade.lanes.dialects import move
from bloqade.lanes.layout.encoding import LocationAddress


@dataclass
class RewriteFill(rewrite_abc.RewriteRule):
    def get_locations(self, address: LocationAddress):
        if address.word_id == 0:
            return tuple(
                LocationAddress(word_id, address.site_id) for word_id in range(7)
            )
        elif address.word_id == 1:
            return tuple(
                LocationAddress(word_id, address.site_id) for word_id in range(9, 16, 1)
            )
        else:
            return (address,)

    def rewrite_Statement(self, node: ir.Statement):
        if not isinstance(node, move.Fill):
            return rewrite_abc.RewriteResult()

        transversal_addresses = sum(
            map(self.get_locations, node.location_addresses), ()
        )

        node.replace_by(move.Fill(location_addresses=transversal_addresses))

        return rewrite_abc.RewriteResult(has_done_something=True)
