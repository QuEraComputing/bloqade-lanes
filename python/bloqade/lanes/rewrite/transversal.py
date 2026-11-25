import math
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import chain

from kirin import ir, types
from kirin.dialects import func, ilist, math as kmath, py
from kirin.rewrite import abc as rewrite_abc

from bloqade.lanes.bytecode.encoding import LaneAddress, LocationAddress
from bloqade.lanes.dialects import move, place
from bloqade.lanes.prelude import kernel
from bloqade.lanes.utils import constant_float, no_none_elements


@kernel(verify=False)
def steane_star_theta(theta_turns: float) -> float:
    theta = theta_turns * kmath.tau
    theta_star = -kmath.copysign(
        2 * kmath.atan(kmath.fabs(kmath.tan(theta / 2)) ** (1 / 3)),
        theta,
    )
    return theta_star / kmath.tau


# --- STAR-R at an arbitrary axis ------------------------------------------
#
# Post-selected on a trivial syndrome, physical LocalR(phi_p, t) on a weight-3
# Steane support acts on the code space as
#
#     c^3 + i s^3 (cos^3(phi_p) Xbar - sin^3(phi_p) Ybar),   c, s = cos, sin(t/2)
#
# (mixed X/Y terms leave the code space; Y_L = -Ybar). That is a logical
# rotation about atan2(-sin^3 phi_p, cos^3 phi_p) with
# tan(theta_L / 2) = -r tan^3(t / 2), r = sqrt(cos^6 phi_p + sin^6 phi_p).
# Inverting it gives the physical axis and angle for a logical target. At
# quarter turns r = 1 and the physical axis is -phi_L, so the gadget reduces
# to steane_star_theta on the negated axis.
#
# These take the axis in EliminateRz's convention: ``axis_turns`` is the
# *negated* logical axis (see RewriteSteaneStarRAxisConvention).


if sys.version_info >= (3, 11):
    _cbrt = math.cbrt
else:

    def _cbrt(x: float) -> float:
        """Real cube root; ``math.cbrt`` is new in Python 3.11."""
        return math.copysign(abs(x) ** (1 / 3), x)


def _is_quarter_turn(axis_turns: float) -> bool:
    return abs(4 * axis_turns - round(4 * axis_turns)) < 1e-12


def steane_star_r_physical(
    axis_turns: float, theta_turns: float
) -> tuple[float, float]:
    """Physical (axis, angle) in turns for a STAR-R gadget, at compile time.

    ``axis_turns`` is the negated logical axis. Quarter turns are snapped
    exactly: the cube root would otherwise turn ``cos(pi/2) ~ 6e-17`` into an
    axis error of ~1e-6.
    """
    theta = theta_turns * math.tau
    if _is_quarter_turn(axis_turns):
        return (
            axis_turns % 1.0,
            -math.copysign(2 * math.atan(abs(math.tan(theta / 2)) ** (1 / 3)), theta)
            / math.tau,
        )
    phi_l = -axis_turns * math.tau
    phi_p = math.atan2(-_cbrt(math.sin(phi_l)), _cbrt(math.cos(phi_l)))
    r = math.hypot(math.cos(phi_p) ** 3, math.sin(phi_p) ** 3)
    theta_p = -math.copysign(
        2 * math.atan((abs(math.tan(theta / 2)) / r) ** (1 / 3)), theta
    )
    return (phi_p / math.tau) % 1.0, theta_p / math.tau


# The bodies of these two kernels run in Kirin's interpreter, never as Python,
# so line coverage cannot see them. They are exercised against the
# constant-folding path in test_symbolic_star_r_kernels_match_compile_time_folding.
#
# Precision: unlike steane_star_r_physical, these do not snap quarter turns. A
# symbolic axis that is a quarter turn at runtime goes through the cube root,
# which turns e.g. cos(pi/2) ~ 6e-17 into ~4e-6, so the physical axis comes out
# ~6e-7 to 9e-7 turns off at logical 1/4, 1/2, 3/4 (exact at 0). Constant
# axes are snapped exactly in RewriteStarR before these are ever used.
@kernel(verify=False)  # pragma: no cover
def steane_star_physical_axis(axis_turns: float) -> float:
    phi_l = -axis_turns * kmath.tau
    c = kmath.cos(phi_l)
    s = kmath.sin(phi_l)
    c3 = kmath.copysign(kmath.fabs(c) ** (1 / 3), c)
    s3 = kmath.copysign(kmath.fabs(s) ** (1 / 3), s)
    return kmath.atan2(-s3, c3) / kmath.tau


@kernel(verify=False)  # pragma: no cover
def steane_star_physical_theta(physical_axis_turns: float, theta_turns: float) -> float:
    phi_p = physical_axis_turns * kmath.tau
    theta = theta_turns * kmath.tau
    c = kmath.cos(phi_p)
    s = kmath.sin(phi_p)
    r = kmath.sqrt(c**6 + s**6)
    theta_star = -kmath.copysign(
        2 * kmath.atan((kmath.fabs(kmath.tan(theta / 2)) / r) ** (1 / 3)),
        theta,
    )
    return theta_star / kmath.tau


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

    v1 supports the k=3 Steane STAR protocol. The logical target angle is stored
    in SQuIn IR turn units; the STAR formula converts through radians internally
    and returns the physical STAR angle in turns.
    """

    transform_location: Callable[[LocationAddress], Iterable[LocationAddress] | None]

    def _theta_star_ir(self, node: move.StarRz | move.StarR) -> ir.SSAValue:
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
class RewriteStarR(RewriteStarRz):
    """Lower STAR-R to physical local ``R(axis, theta*)`` rotations on its support.

    The axis arriving here is the *negated* logical axis: under
    ``transversal_rewrite``, ``RewriteSteaneStarRAxisConvention`` negates it
    before ``EliminateRz``, and ``EliminateRz`` then shifts it to absorb any
    pending Z frame. Nothing is pre-distorted before this point; this rule
    does it.

    At a quarter-turn axis the negated axis already *is* the physical axis, so
    it passes through unchanged and the gadget is
    ``LocalR(axis, steane_star_theta(rotation_angle))``. At any other axis both
    the axis and the angle are pre-distorted here (see
    ``steane_star_r_physical``) so the post-selected logical gate is the
    requested one. Constants are folded at compile time; symbolic operands go
    through the ``steane_star_physical_*`` kernels.

    **Assumes every frame ``EliminateRz`` absorbed into the axis was a quarter
    turn.** The pre-distortion is nonlinear and commutes only with
    quarter-turn shifts, so shifting before it (as the pipeline does) equals
    shifting the physical axis only then. Validated logical programs satisfy
    this: ``SteaneLogicalValidation`` rejects mid-circuit ``Rz``/``T``/``U3``/
    ``Rx``/``Ry``, so frames come only from ``Z``/``S``/``S_adj``/``H``. That
    validation is not enforced under ``no_raise=True`` (``LogicalPipeline``'s
    default); a non-quarter-turn frame there silently gives a misaligned
    gadget. Plain transversal ``R`` has the same limitation: transversal
    ``Rz(alpha)`` is a logical operation only for quarter-turn ``alpha``.
    """

    def _physical_axis_and_theta(
        self, node: move.StarR
    ) -> tuple[ir.SSAValue, ir.SSAValue]:
        axis = constant_float(node.axis_angle)
        theta = constant_float(node.rotation_angle)
        if axis is not None and _is_quarter_turn(axis):
            return node.axis_angle, self._theta_star_ir(node)
        if axis is not None and theta is not None:
            phi_p, theta_p = steane_star_r_physical(axis, theta)
            (axis_stmt := py.Constant(phi_p)).insert_before(node)
            (theta_stmt := py.Constant(theta_p)).insert_before(node)
            return axis_stmt.result, theta_stmt.result
        phi_p_stmt = func.Invoke((node.axis_angle,), callee=steane_star_physical_axis)
        phi_p_stmt.insert_before(node)
        theta_p_stmt = func.Invoke(
            (phi_p_stmt.result, node.rotation_angle), callee=steane_star_physical_theta
        )
        theta_p_stmt.insert_before(node)
        return phi_p_stmt.result, theta_p_stmt.result

    def rewrite_Statement(self, node: ir.Statement):
        if not isinstance(node, move.StarR):
            return rewrite_abc.RewriteResult()

        supports = [
            self._physical_support(address, node.qubit_indices)
            for address in node.location_addresses
        ]
        if not no_none_elements(supports):
            return rewrite_abc.RewriteResult()

        axis, theta_star = self._physical_axis_and_theta(node)
        node.replace_by(
            move.LocalR(
                node.current_state,
                axis,
                theta_star,
                location_addresses=tuple(chain.from_iterable(supports)),
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
