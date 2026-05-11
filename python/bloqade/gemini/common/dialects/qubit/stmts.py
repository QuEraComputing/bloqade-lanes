from bloqade.types import QubitType
from kirin import ir, lowering, types
from kirin.decl import info, statement

from ._dialect import dialect


@statement(dialect=dialect)
class NewAt(ir.Statement):
    """Allocate a new qubit pinned to the given physical address.

    The three int args MUST be compile-time constants (enforced by validation
    in ``bloqade.gemini.analysis.new_at_validation``). The constant values are
    read by the circuit→place rewrite chain and stamped into
    ``place.NewLogicalQubit.location_address``.

    Belongs to the ``gemini.common`` dialect so it can be used from both
    logical and physical kernels.
    """

    traits = frozenset({lowering.FromPythonCall()})
    zone_id: ir.SSAValue = info.argument(types.Int)
    word_id: ir.SSAValue = info.argument(types.Int)
    site_id: ir.SSAValue = info.argument(types.Int)
    qubit: ir.ResultValue = info.result(QubitType)
