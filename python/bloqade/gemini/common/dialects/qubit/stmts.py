from bloqade.types import QubitType
from kirin import ir, lowering, types
from kirin.decl import info, statement

from ._dialect import dialect


@statement(dialect=dialect)
class NewAt(ir.Statement):
    """Allocate a new qubit pinned to the given physical grid coordinate.

    The three int args MUST be compile-time constants (enforced by validation
    in ``bloqade.gemini.analysis.new_at_validation``). The circuit→place
    rewrite resolves the coordinate through the active architecture and stamps
    the resulting physical address into ``place.NewLogicalQubit``.

    Belongs to the ``gemini.common`` dialect so it can be used from both
    logical and physical kernels.
    """

    traits = frozenset({lowering.FromPythonCall()})
    zone: ir.SSAValue = info.argument(types.Int)
    row: ir.SSAValue = info.argument(types.Int)
    col: ir.SSAValue = info.argument(types.Int)
    qubit: ir.ResultValue = info.result(QubitType)
