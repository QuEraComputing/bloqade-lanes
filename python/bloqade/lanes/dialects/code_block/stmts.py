from bloqade.types import QubitType
from kirin import ir, lowering, types
from kirin.decl import info, statement
from kirin.dialects import ilist

from ._dialect import dialect


@statement(dialect=dialect)
class Register(ir.Statement):
    """Declare ``qubits`` as one code block, positions in list order.

    Has no runtime effect. ``PhysicalNativeToPlace`` validates every
    ``Register`` after unrolling, stamps a ``CodeBlockTag`` onto each member's
    ``place.NewPinnedQubit`` and then deletes the statement.
    """

    name = "register"
    traits = frozenset({lowering.FromPythonCall()})
    qubits: ir.SSAValue = info.argument(ilist.IListType[QubitType, types.Any])
