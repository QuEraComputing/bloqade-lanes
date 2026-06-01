from bloqade.types import Qubit
from kirin import lowering

from bloqade.lanes.bytecode.encoding import LocationAddress

from .stmts import MoveTo


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
