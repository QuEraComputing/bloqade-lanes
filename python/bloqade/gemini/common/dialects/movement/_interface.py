from typing import TypeVar, overload

from bloqade.types import Qubit
from kirin import lowering
from kirin.dialects import ilist

from bloqade.lanes.bytecode.encoding import LocationAddress

from .stmts import Loc, MoveTo


@lowering.wraps(Loc)
def loc(zone_id: int, word_id: int, site_id: int) -> LocationAddress:
    """Construct a LocationAddress for use with move_to inside a kernel body.

    All three arguments must be compile-time-constant integers.
    """
    ...


Len = TypeVar("Len")


@overload
def move_to(
    qubits: ilist.IList[Qubit, Len],
    locations: ilist.IList[LocationAddress, Len],
    multi_move_warning: bool = True,
): ...


@overload
def move_to(
    qubits: list[Qubit],
    locations: list[LocationAddress],
    multi_move_warning: bool = True,
): ...


@lowering.wraps(MoveTo)
def move_to(
    qubits,
    locations,
    multi_move_warning=True,
) -> None:
    """Move the given qubits to the specified LocationAddress destinations.

    Must immediately precede a CZ gate (or another move_to call).
    Compilation fails if the move is physically infeasible.
    """
    ...
