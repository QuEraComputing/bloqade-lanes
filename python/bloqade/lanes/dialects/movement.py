from kirin import ir, lowering, types
from kirin.decl import info, statement
from kirin.dialects import ilist
from kirin.lowering.python.binding import wraps

from bloqade import types as bloqade_types
from bloqade.lanes.bytecode.encoding import LocationAddress

dialect = ir.Dialect(name="lanes.movement")

LocationAddressType = types.PyClass(LocationAddress)


@statement(dialect=dialect)
class MoveTo(ir.Statement):
    name = "move_to"
    traits = frozenset({lowering.FromPythonCall()})
    qubits: ir.SSAValue = info.argument(
        type=ilist.IListType[bloqade_types.QubitType, types.Any]
    )
    locations: ir.SSAValue = info.argument(
        type=ilist.IListType[LocationAddressType, types.Any]
    )
    multi_move_warning: bool = info.attribute(default=True)


@wraps(MoveTo)
def move_to(
    qubits: list,
    locations: list,
    multi_move_warning: bool = True,
) -> None:
    """Move the given qubits to the specified LocationAddress destinations.

    Must immediately precede a CZ gate (or another move_to call).
    Compilation fails if the move is physically infeasible.
    """
    ...
