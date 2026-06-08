from bloqade.types import Qubit
from kirin import lowering

from bloqade.lanes.bytecode.encoding import LocationAddress

from .stmts import Loc, MoveTo


@lowering.wraps(Loc)
def loc(zone_id: int, word_id: int, site_id: int) -> LocationAddress:
    """Construct a LocationAddress for use with move_to inside a kernel body.

    All three arguments must be compile-time-constant integers.
    """
    ...


@lowering.wraps(MoveTo)
def move_to(
    qubits: list[Qubit],
    locations: list[LocationAddress],
    multi_move_warning: bool = True,
) -> None:
    """Move the given qubits to the specified LocationAddress destinations.

    Must immediately precede a CZ gate (or another move_to call).
    Compilation fails if the move is physically infeasible.
    """
    ...
