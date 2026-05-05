from dataclasses import dataclass
from itertools import chain

from kirin import ir, types
from kirin.dialects import ilist, math as kmath, py
from kirin.rewrite import abc as rewrite_abc
from typing_extensions import Callable, Iterable

from bloqade.lanes.bytecode.encoding import LaneAddress, LocationAddress
from bloqade.lanes.dialects import move, place
from bloqade.lanes.utils import no_none_elements


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

    def _theta_star_ir(self, node: move.StarRz) -> ir.SSAValue:
        # theta_star = -copysign(2 * atan(abs(tan(theta / 2)) ** (1 / 3)), theta)
        half = py.Constant(2.0)
        exponent = py.Constant(1.0 / 3.0)
        two = py.Constant(2.0)
        neg_one = py.Constant(-1.0)
        theta_over_two = py.Div(node.rotation_angle, half.result)
        tan_half_theta = kmath.stmts.tan(theta_over_two.result)
        abs_tan = kmath.stmts.fabs(tan_half_theta.result)
        root = kmath.stmts.pow(abs_tan.result, exponent.result)
        atan_root = kmath.stmts.atan(root.result)
        magnitude = py.Mult(two.result, atan_root.result)
        signed_magnitude = kmath.stmts.copysign(magnitude.result, node.rotation_angle)
        theta_star = py.Mult(neg_one.result, signed_magnitude.result)

        for stmt in (
            half,
            exponent,
            two,
            neg_one,
            theta_over_two,
            tan_half_theta,
            abs_tan,
            root,
            atan_root,
            magnitude,
            signed_magnitude,
            theta_star,
        ):
            stmt.insert_before(node)

        return theta_star.result

    def _physical_support(
        self,
        logical_address: LocationAddress,
        support: tuple[int, int, int],
    ) -> tuple[LocationAddress, ...] | None:
        if logical_address.site_id >= 2:
            return None
        base = logical_address.site_id * 7
        return tuple(
            logical_address.replace(site_id=base + physical_index)
            for physical_index in support
        )

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
