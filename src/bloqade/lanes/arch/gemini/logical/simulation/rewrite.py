from dataclasses import dataclass, replace
from itertools import chain
from typing import Callable, Iterator, TypeVar

from kirin import ir
from kirin.rewrite import abc as rewrite_abc

from bloqade.lanes.dialects import move
from bloqade.lanes.layout.encoding import LaneAddress, LocationAddress

AddressType = TypeVar("AddressType", bound=LocationAddress | LaneAddress)


def physical_word_id(address: AddressType) -> Iterator[AddressType]:
    if address.word_id == 0:
        yield from (replace(address, word_id=word_id) for word_id in range(7))
    elif address.word_id == 1:
        yield from (replace(address, word_id=word_id) for word_id in range(9, 16, 1))
    else:
        yield address


@dataclass
class RewriteLocations(rewrite_abc.RewriteRule):

    transform_location: Callable[[LocationAddress], Iterator[LocationAddress]] = (
        physical_word_id
    )

    def rewrite_Statement(self, node: ir.Statement):
        if not isinstance(
            node, (move.Fill, move.LocalR, move.LocalRz, move.Initialize)
        ):
            return rewrite_abc.RewriteResult()

        physical_addresses = tuple(
            chain.from_iterable(map(self.transform_location, node.location_addresses))
        )

        attributes: dict[str, ir.Attribute] = {
            "location_addresses": ir.PyAttr(physical_addresses)
        }

        node.replace_by(node.from_stmt(node, attributes=attributes))
        return rewrite_abc.RewriteResult(has_done_something=True)


@dataclass
class RewriteMoves(rewrite_abc.RewriteRule):
    transform_lanes: Callable[[LaneAddress], Iterator[LaneAddress]] = physical_word_id

    def rewrite_Statement(self, node: ir.Statement):
        if not isinstance(node, move.Move):
            return rewrite_abc.RewriteResult()

        physical_lanes = tuple(
            chain.from_iterable(map(self.transform_lanes, node.lanes))
        )

        node.replace_by(move.Move(lanes=physical_lanes))

        return rewrite_abc.RewriteResult(has_done_something=True)
