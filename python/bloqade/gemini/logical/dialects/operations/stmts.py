from bloqade.types import MeasurementResultType, QubitType
from kirin import exception, ir, lowering, types
from kirin.decl import info, statement
from kirin.dialects import ilist

from bloqade.gemini.star import (
    DEFAULT_STEANE_STAR_SUPPORT,
    validate_steane_star_support,
)

from ._dialect import dialect


@statement(dialect=dialect)
class Initialize(ir.Statement):
    """Initialize a list of logical qubits to an arbitrary state.

    Args:
        phi (float): Angle for rotation around the Z axis
        theta (float): angle for rotation around the Y axis
        phi (float): angle for rotation around the Z axis
        qubits (IList[QubitType, Len]): The list of logical qubits to initialize

    """

    traits = frozenset({})
    theta: ir.SSAValue = info.argument(types.Float)
    phi: ir.SSAValue = info.argument(types.Float)
    lam: ir.SSAValue = info.argument(types.Float)
    qubits: ir.SSAValue = info.argument(ilist.IListType[QubitType, types.Any])


Len = types.TypeVar("Len")


@statement(dialect=dialect)
class TerminalLogicalMeasurement(ir.Statement):
    """Perform measurements on a list of logical qubits.

    Measurements are returned as a nested list where each member list
    contains the individual measurement results for the constituent physical qubits per logical qubit.

    Args:
        qubits (IList[QubitType, Len]): The list of logical qubits

    Returns:
        IList[IList[MeasurementResultType, CodeN], Len]: A nested list containing the measurement results,
            where each inner list corresponds to the measurements of the physical qubits that make up each logical qubit.
    """

    traits = frozenset({lowering.FromPythonCall()})
    qubits: ir.SSAValue = info.argument(ilist.IListType[QubitType, Len])
    num_physical_qubits: int | None = info.attribute(
        types.Int, init=False, default_factory=lambda: None
    )
    result: ir.ResultValue = info.result(
        ilist.IListType[ilist.IListType[MeasurementResultType, types.Any], Len]
    )


@statement(dialect=dialect)
class StarRz(ir.Statement):
    """STAR/TMR logical-Z rotation injection primitive.

    ``rotation_angle`` is the target logical angle. ``qubits`` is an ``IList``
    of logical qubits.
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
        if self.qubits.type.is_structurally_equal(
            types.Bottom
        ) or self.qubits.type.is_subseteq(QubitType):
            raise exception.StaticCheckError(
                "star_rz expects qubits to be an ilist.IList[Qubit]. "
                "For one logical qubit, wrap it as ilist.IList([q]); for a "
                "register, pass the register directly."
            )
        try:
            validate_steane_star_support(self.qubit_indices)
        except ValueError as exc:
            raise exception.StaticCheckError(str(exc)) from exc
