from dataclasses import dataclass
from itertools import chain

from kirin import ir, types
from kirin.dialects import ilist, py
from kirin.rewrite import abc as rewrite_abc
from typing_extensions import Callable, Iterable

from bloqade.lanes.dialects import move, place
from bloqade.lanes.layout.encoding import LaneAddress, LocationAddress

from .utils import no_none_elements


@dataclass
class RewriteLogicalInitialize(rewrite_abc.RewriteRule):
    transform_location: Callable[[LocationAddress], Iterable[LocationAddress] | None]

    def rewrite_Statement(self, node: ir.Statement):
        if not isinstance(node, (move.LogicalInitialize)):
            return rewrite_abc.RewriteResult()

        iterators = list(map(self.transform_location, node.location_addresses))

        if not no_none_elements(iterators):
            return rewrite_abc.RewriteResult()

        node.replace_by(
            move.PhysicalInitialize(
                thetas=node.thetas,
                phis=node.phis,
                lams=node.lams,
                location_addresses=tuple(map(tuple, iterators)),
            )
        )
        return rewrite_abc.RewriteResult(has_done_something=True)


@dataclass
class RewriteLocations(rewrite_abc.RewriteRule):
    transform_location: Callable[[LocationAddress], Iterable[LocationAddress] | None]

    def rewrite_Statement(self, node: ir.Statement):
        if not isinstance(node, (move.Fill, move.LocalR, move.LocalRz)):
            return rewrite_abc.RewriteResult()

        iterators = list(map(self.transform_location, node.location_addresses))

        if not no_none_elements(iterators):
            return rewrite_abc.RewriteResult()

        physical_addresses = tuple(chain.from_iterable(iterators))

        attributes: dict[str, ir.Attribute] = {
            "location_addresses": ir.PyAttr(
                physical_addresses,
                pytype=types.Tuple[types.Vararg(types.PyClass(LocationAddress))],
            )
        }

        node.replace_by(node.from_stmt(node, attributes=attributes))
        return rewrite_abc.RewriteResult(has_done_something=True)


@dataclass
class RewriteMoves(rewrite_abc.RewriteRule):
    transform_lane: Callable[[LaneAddress], Iterable[LaneAddress] | None]

    def rewrite_Statement(self, node: ir.Statement):
        if not isinstance(node, move.Move):
            return rewrite_abc.RewriteResult()

        iterators = list(map(self.transform_lane, node.lanes))

        if not no_none_elements(iterators):
            return rewrite_abc.RewriteResult()

        physical_lanes = tuple(chain.from_iterable(iterators))

        node.replace_by(move.Move(lanes=physical_lanes))

        return rewrite_abc.RewriteResult(has_done_something=True)


@dataclass
class RewriteGetItem(rewrite_abc.RewriteRule):
    transform_lane: Callable[[LocationAddress], Iterable[LocationAddress] | None]

    def rewrite_Statement(self, node: ir.Statement):
        if not (
            isinstance(node, py.GetItem)
            and isinstance(index_stmt := node.index.owner, move.GetZoneIndex)
        ):
            return rewrite_abc.RewriteResult()

        iterator = self.transform_lane(index_stmt.location_address)

        if iterator is None:
            return rewrite_abc.RewriteResult()

        new_results: list[ir.ResultValue] = []
        for address in iterator:
            (
                inew_index_stmt := move.GetZoneIndex(
                    zone_address=index_stmt.zone_address, location_address=address
                )
            ).insert_before(index_stmt)
            (new_stmt := py.GetItem(node.obj, inew_index_stmt.result)).insert_before(
                node
            )
            new_results.append(new_stmt.result)

        node.replace_by(ilist.New(tuple(new_results)))

        return rewrite_abc.RewriteResult(has_done_something=True)


class RewriteLogicalToPhysicalConversion(rewrite_abc.RewriteRule):
    """Note that this rewrite is to be combined with RewriteGetMeasurementResult."""

    def rewrite_Statement(self, node: ir.Statement) -> rewrite_abc.RewriteResult:
        if not isinstance(node, place.ConvertToPhysicalMeasurements):
            return rewrite_abc.RewriteResult()

        node.replace_by(ilist.New(tuple(node.args)))
        return rewrite_abc.RewriteResult(has_done_something=True)
