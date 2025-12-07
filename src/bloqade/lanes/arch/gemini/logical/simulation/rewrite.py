from dataclasses import dataclass, replace
from typing import TypeVar

from kirin import ir
from kirin.rewrite import abc as rewrite_abc

from bloqade.lanes.dialects import move
from bloqade.lanes.layout.encoding import LaneAddress, LocationAddress

AddressType = TypeVar("AddressType", bound=LocationAddress | LaneAddress)


def physical_word_id(address: AddressType) -> tuple[AddressType, ...]:
    if address.word_id == 0:
        return tuple(replace(address, word_id=word_id) for word_id in range(7))
    elif address.word_id == 1:
        return tuple(replace(address, word_id=word_id) for word_id in range(9, 16, 1))
    else:
        return (address,)


@dataclass
class RewriteFill(rewrite_abc.RewriteRule):

    def rewrite_Statement(self, node: ir.Statement):
        if not isinstance(node, move.Fill):
            return rewrite_abc.RewriteResult()

        transversal_addresses = sum(map(physical_word_id, node.location_addresses), ())

        node.replace_by(move.Fill(location_addresses=transversal_addresses))

        return rewrite_abc.RewriteResult(has_done_something=True)
