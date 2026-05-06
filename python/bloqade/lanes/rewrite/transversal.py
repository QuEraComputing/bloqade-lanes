from dataclasses import dataclass
from itertools import chain

from kirin import ir, types
from kirin.dialects import func, ilist, math as kmath
from kirin.rewrite import abc as rewrite_abc
from typing_extensions import Callable, Iterable

from bloqade.lanes.bytecode.encoding import LaneAddress, LocationAddress
from bloqade.lanes.dialects import move, place
from bloqade.lanes.prelude import kernel
from bloqade.lanes.utils import no_none_elements


@kernel(verify=False)
def steane_star_theta(theta: float) -> float:
    return -kmath.copysign(
        2 * kmath.atan(kmath.fabs(kmath.tan(theta / 2)) ** (1 / 3)),
        theta,
    )


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
                node.current_state,
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

        node.replace_by(move.Move(node.current_state, lanes=physical_lanes))

        return rewrite_abc.RewriteResult(has_done_something=True)


@dataclass
class RewriteStarRz(rewrite_abc.RewriteRule):
    """Lower logical STAR rotations to physical local Rz rotations.

    v1 supports the k=3 Steane STAR protocol. The logical target angle is
    converted to the physical STAR angle with Kirin math IR, so the angle may
    be either a literal or an SSA value.
    """

    transform_location: Callable[[LocationAddress], Iterable[LocationAddress] | None]

    def _theta_star_ir(self, node: move.StarRz) -> ir.SSAValue:
        theta_star = func.Invoke((node.rotation_angle,), callee=steane_star_theta)
        theta_star.insert_before(node)
        return theta_star.result

    def _physical_support(
        self, logical_address: LocationAddress, support: tuple[int, int, int]
    ) -> tuple[LocationAddress, ...] | None:
        iterator = self.transform_location(logical_address)
        if iterator is None:
            return None

        physical_addresses = tuple(iterator)
        if any(index >= len(physical_addresses) for index in support):
            return None

        return tuple(physical_addresses[index] for index in support)

    def rewrite_Statement(self, node: ir.Statement):
        if not isinstance(node, move.StarRz):
            return rewrite_abc.RewriteResult()

        iterators = [
            self._physical_support(address, node.qubit_indices)
            for address in node.location_addresses
        ]

        if not no_none_elements(iterators):
            return rewrite_abc.RewriteResult()

        theta_star = self._theta_star_ir(node)
        physical_addresses = tuple(chain.from_iterable(iterators))
        node.replace_by(
            move.LocalRz(
                node.current_state,
                theta_star,
                location_addresses=physical_addresses,
            )
        )
        return rewrite_abc.RewriteResult(has_done_something=True)


@dataclass
class RewriteGetItem(rewrite_abc.RewriteRule):
    transform_lane: Callable[[LocationAddress], Iterable[LocationAddress] | None]

    def rewrite_Statement(self, node: ir.Statement):
        if not (isinstance(node, move.GetFutureResult)):
            return rewrite_abc.RewriteResult()

        iterator = self.transform_lane(node.location_address)

        if iterator is None:
            return rewrite_abc.RewriteResult()

        new_results: list[ir.ResultValue] = []
        for address in iterator:
            new_node = move.GetFutureResult(
                node.measurement_future,
                zone_address=node.zone_address,
                location_address=address,
            )
            new_results.append(new_node.result)
            new_node.insert_before(node)

        node.replace_by(ilist.New(tuple(new_results)))

        return rewrite_abc.RewriteResult(has_done_something=True)


class RewriteLogicalToPhysicalConversion(rewrite_abc.RewriteRule):
    """Note that this rewrite is to be combined with RewriteGetMeasurementResult."""

    def rewrite_Statement(self, node: ir.Statement) -> rewrite_abc.RewriteResult:
        if not isinstance(node, place.ConvertToPhysicalMeasurements):
            return rewrite_abc.RewriteResult()

        node.replace_by(ilist.New(tuple(node.args)))
        return rewrite_abc.RewriteResult(has_done_something=True)
