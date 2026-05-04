from typing import Any, TypeVar

from bloqade.types import MeasurementResult, Qubit
from kirin import lowering
from kirin.dialects import ilist

from .stmts import StarRz, TerminalLogicalMeasurement

Len = TypeVar("Len")


@lowering.wraps(TerminalLogicalMeasurement)
def terminal_measure(
    qubits: ilist.IList[Qubit, Len],
) -> ilist.IList[ilist.IList[MeasurementResult, Any], Len]:
    """Perform measurements on a list of logical qubits.

    Measurements are returned as a nested list where each member list
    contains the individual measurement results for the constituent physical qubits per logical qubit.

    Args:
        qubits (IList[Qubit, Len]): The list of logical qubits to measure.

    Returns:
        IList[IList[MeasurementResult, CodeN], Len]: A nested list containing the measurement results,
            where each inner list corresponds to the measurements of the physical qubits that make up each logical qubit.
    """
    ...


@lowering.wraps(StarRz)
def star_rz(
    theta: float,
    qubits: Qubit | ilist.IList[Qubit, Len],
    qubit_indices: tuple[int, int, int] | tuple[int, ...] | None = None,
) -> None:
    """Apply a STAR/TMR logical-Z rotation injection primitive.

    Args:
        theta: Target logical rotation angle.
        qubits: One logical qubit or a list of logical qubits.
        qubit_indices: Optional Steane weight-3 logical-Z support. Defaults to
            ``(4, 5, 6)``.
    """
    ...
