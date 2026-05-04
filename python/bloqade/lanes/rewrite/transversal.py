import math
from dataclasses import dataclass
from itertools import chain
from typing import Any, cast

from kirin import ir, types
from kirin.analysis import const
from kirin.dialects import ilist, py
from kirin.rewrite import abc as rewrite_abc
from typing_extensions import Callable, Iterable

from bloqade.lanes.bytecode.encoding import LaneAddress, LocationAddress
from bloqade.lanes.dialects import move, place
from bloqade.lanes.star import steane_star_theta
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

    v1 supports only literal target logical angles and the k=3 Steane STAR
    protocol.
    """

    def _literal_angle(self, node: move.StarRz) -> float:
        hint = node.rotation_angle.hints.get("const")
        if isinstance(hint, const.Value) and isinstance(hint.data, (int, float)):
            theta = float(hint.data)
            if not math.isfinite(theta):
                raise ValueError("star_rz theta must be finite")
            return theta

        owner = node.rotation_angle.owner
        if not isinstance(owner, py.Constant):
            raise ValueError(
                "star_rz only supports literal/compile-time theta values in v1; "
                "runtime kernel-argument theta is not supported"
            )
        value = cast(Any, owner.value).data
        if not isinstance(value, (int, float)):
            raise ValueError(
                "star_rz only supports literal/compile-time theta values in v1; "
                "runtime kernel-argument theta is not supported"
            )
        theta = float(value)
        if not math.isfinite(theta):
            raise ValueError("star_rz theta must be finite")
        return theta

    def _physical_support(
        self,
        logical_address: LocationAddress,
        support: tuple[int, int, int],
    ) -> tuple[LocationAddress, ...]:
        if logical_address.site_id >= 2:
            raise ValueError(
                f"star_rz expected a logical site id 0 or 1, got "
                f"{logical_address.site_id}"
            )
        base = logical_address.site_id * 7
        return tuple(
            logical_address.replace(site_id=base + physical_index)
            for physical_index in support
        )

    def rewrite_Statement(self, node: ir.Statement):
        if not isinstance(node, move.StarRz):
            return rewrite_abc.RewriteResult()

        theta_star = steane_star_theta(self._literal_angle(node))
        angle_stmt = py.Constant(theta_star)
        angle_stmt.insert_before(node)

        physical_addresses = tuple(
            chain.from_iterable(
                self._physical_support(address, node.qubit_indices)
                for address in node.location_addresses
            )
        )

        node.replace_by(
            move.LocalRz(
                node.current_state,
                angle_stmt.result,
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
