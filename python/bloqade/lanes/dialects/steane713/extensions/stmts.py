from bloqade.types import QubitType
from kirin import exception, ir, lowering, types
from kirin.decl import info, statement
from kirin.dialects import ilist

from bloqade.lanes.dialects.steane713.extensions._dialect import dialect
from bloqade.lanes.dialects.steane713.star import (
    DEFAULT_STEANE_STAR_SUPPORT,
    validate_steane_star_support,
)


@statement(dialect=dialect)
class StarRz(ir.Statement):
    """STAR/TMR logical-Z rotation injection primitive.

    ``rotation_angle`` is the target logical angle in SQuIn IR turn units.
    ``qubits`` is an ``IList`` of logical qubits.
    The final logical-to-physical lowering computes the physical STAR angle for
    k=3 and emits physical Rz rotations on ``qubit_indices``.
    """

    traits = frozenset({lowering.FromPythonCall()})
    rotation_angle: ir.SSAValue = info.argument(types.Float)
    qubits: ir.SSAValue = info.argument(ilist.IListType[QubitType, types.Any])
    qubit_indices: tuple[int, int, int] = info.attribute(
        default=DEFAULT_STEANE_STAR_SUPPORT
    )

    def check(self) -> None:
        try:
            validate_steane_star_support(self.qubit_indices)
        except ValueError as exc:
            raise exception.StaticCheckError(str(exc)) from exc


@statement(dialect=dialect)
class StarR(ir.Statement):
    """STAR/TMR logical XY-plane rotation injection primitive.

    Mirrors native ``R``: ``axis_angle`` is the rotation axis in the XY plane
    and ``rotation_angle`` the target logical angle, both in SQuIn IR turns.
    STAR-X is ``axis_angle == 0``. The physical lowering applies local
    ``R(axis, theta*)`` rotations on the selected Steane support.

    Any axis is allowed. At a quarter turn (0, 1/4, 1/2, 3/4) the physical
    gadget is ``R(-axis, steane_star_theta(rotation_angle))``. At any other
    axis, ``RewriteStarR`` pre-distorts both the physical axis and the angle so
    that the post-selected logical gate is still the requested rotation (see
    ``steane_star_r_physical``); the acceptance probability then depends on the
    axis. ``EliminateRz`` shifts the axis to absorb a pending Z frame.
    """

    traits = frozenset({lowering.FromPythonCall()})
    axis_angle: ir.SSAValue = info.argument(types.Float)
    rotation_angle: ir.SSAValue = info.argument(types.Float)
    qubits: ir.SSAValue = info.argument(ilist.IListType[QubitType, types.Any])
    qubit_indices: tuple[int, int, int] = info.attribute(
        default=DEFAULT_STEANE_STAR_SUPPORT
    )

    def check(self) -> None:
        try:
            validate_steane_star_support(self.qubit_indices)
        except ValueError as exc:
            raise exception.StaticCheckError(str(exc)) from exc
