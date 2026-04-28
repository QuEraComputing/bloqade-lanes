from typing import Any, TypeVar

from bloqade.types import MeasurementResult, Qubit
from kirin import lowering
from kirin.dialects import ilist

from .stmts import NewAt, TerminalLogicalMeasurement

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


@lowering.wraps(NewAt)
def new_at(zone_id: int, word_id: int, site_id: int) -> Qubit:
    """Allocate a logical qubit pinned to (zone_id, word_id, site_id).

    All three arguments must be compile-time constants. Use of non-constant
    values raises a validation error before lowering.
    """
    ...
