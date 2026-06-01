from bloqade.types import QubitType
from kirin import ir, lowering, types
from kirin.decl import info, statement
from kirin.dialects import ilist

from bloqade.lanes.bytecode.encoding import LocationAddress

from ._dialect import dialect

LocationAddressType = types.PyClass(LocationAddress)
Len = types.TypeVar("Len")


@statement(dialect=dialect)
class MoveTo(ir.Statement):
    """User-facing move_to directive: move qubits to specified LocationAddress destinations.

    Lowered from a Python call by FromPythonCall. RewritePlaceOperations rewrites this
    to a StaticPlacement(place.MoveTo) after const-folding the locations ilist.
    """

    name = "move_to"
    traits = frozenset({lowering.FromPythonCall()})
    qubits: ir.SSAValue = info.argument(type=ilist.IListType[QubitType, Len])
    locations: ir.SSAValue = info.argument(
        type=ilist.IListType[LocationAddressType, Len]
    )
    multi_move_warning: bool = info.attribute(default=True)
