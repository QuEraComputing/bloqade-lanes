from typing import cast

from bloqade.types import MeasurementResultType, QubitType
from kirin import ir, lowering, types
from kirin.decl import info, statement
from kirin.dialects import ilist

from ._dialect import dialect

DEFAULT_STEANE_STAR_SUPPORT = (4, 5, 6)
VALID_STEANE_STAR_SUPPORTS = frozenset(
    {
        (0, 1, 5),
        (0, 2, 4),
        (0, 3, 6),
        (1, 2, 6),
        (1, 3, 4),
        (2, 3, 5),
        DEFAULT_STEANE_STAR_SUPPORT,
    }
)


def validate_steane_star_support(
    qubit_indices: tuple[int, ...] | None,
) -> tuple[int, int, int]:
    support = DEFAULT_STEANE_STAR_SUPPORT if qubit_indices is None else qubit_indices
    out = tuple(support)
    if not all(isinstance(index, int) and not isinstance(index, bool) for index in out):
        raise ValueError("qubit_indices must contain integer physical qubit indices")
    if out not in VALID_STEANE_STAR_SUPPORTS:
        valid = ", ".join(
            str(support) for support in sorted(VALID_STEANE_STAR_SUPPORTS)
        )
        raise ValueError(
            f"qubit_indices must be a valid Steane weight-3 logical-Z support; "
            f"got {out}. Valid Steane supports are: {valid}"
        )
    return cast(tuple[int, int, int], out)


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


@statement(dialect=dialect, init=False)
class StarRz(ir.Statement):
    """STAR/TMR logical-Z rotation injection primitive.

    ``rotation_angle`` is the target logical angle. ``qubits`` may be either one
    logical qubit SSA value or an ``IList`` of logical qubits. The final
    logical-to-physical lowering computes the physical STAR angle for k=3 and
    emits physical Rz rotations on ``qubit_indices``.
    """

    traits = frozenset({lowering.FromPythonCall()})
    rotation_angle: ir.SSAValue = info.argument(types.Float)
    qubits: ir.SSAValue = info.argument(types.Any)
    qubit_indices: tuple[int, int, int] = info.attribute(
        default=DEFAULT_STEANE_STAR_SUPPORT
    )

    def __init__(
        self,
        rotation_angle: ir.SSAValue,
        qubits: ir.SSAValue,
        *,
        qubit_indices: tuple[int, int, int] | tuple[int, ...] | None = None,
    ):
        super().__init__(
            args=(rotation_angle, qubits),
            args_slice={"rotation_angle": 0, "qubits": 1},
            result_types=(),
        )
        self.qubit_indices = validate_steane_star_support(qubit_indices)
