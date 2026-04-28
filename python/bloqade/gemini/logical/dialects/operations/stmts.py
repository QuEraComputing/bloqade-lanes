from bloqade.types import MeasurementResultType, QubitType
from kirin import ir, lowering, types
from kirin.decl import info, statement
from kirin.dialects import ilist

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
class NewAt(ir.Statement):
    """Allocate a new logical qubit pinned to the given physical address.

    The three int args MUST be compile-time constants (enforced by validation
    in Phase E). The constant values are read by the circuit→place rewrite
    chain (Phase D's D2/D3) and stamped into place.NewLogicalQubit.location_address.
    """

    traits = frozenset({lowering.FromPythonCall()})
    zone_id: ir.SSAValue = info.argument(types.Int)
    word_id: ir.SSAValue = info.argument(types.Int)
    site_id: ir.SSAValue = info.argument(types.Int)
    qubit: ir.ResultValue = info.result(QubitType)
