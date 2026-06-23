from bloqade.types import QubitType
from kirin import ir, lowering, types
from kirin.decl import info, statement
from kirin.dialects import ilist

from bloqade.lanes.bytecode.encoding import LocationAddress

from ._dialect import dialect

LocationAddressType = types.PyClass(LocationAddress)
Len = types.TypeVar("Len")


@statement(dialect=dialect)
class Loc(ir.Statement):
    """Construct a LocationAddress from three integer SSA values."""

    name = "loc"
    traits = frozenset({ir.Pure(), lowering.FromPythonCall()})
    zone_id: ir.SSAValue = info.argument(types.Int)
    word_id: ir.SSAValue = info.argument(types.Int)
    site_id: ir.SSAValue = info.argument(types.Int)
    location: ir.ResultValue = info.result(LocationAddressType)


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


@statement(dialect=dialect)
class Permute(ir.Statement):
    """User-facing permute directive: move qubits into a permutation of their
    own current locations. qubits[i] moves to the current location of
    qubits[perm[i]].

    Lowered from a Python call by FromPythonCall. RewritePlaceOperations rewrites
    this to a StaticPlacement(place.Permute) after const-folding the perm ilist.
    """

    name = "permute"
    traits = frozenset({lowering.FromPythonCall()})
    qubits: ir.SSAValue = info.argument(type=ilist.IListType[QubitType, Len])
    perm: ir.SSAValue = info.argument(type=ilist.IListType[types.Int, Len])


@statement(dialect=dialect)
class WordId(ir.Statement):
    """Read the word_id of a LocationAddress (produced by the GetAttr rewrite)."""

    name = "word_id"
    traits = frozenset({ir.Pure()})
    address: ir.SSAValue = info.argument(LocationAddressType)
    result: ir.ResultValue = info.result(types.Int)


@statement(dialect=dialect)
class SiteId(ir.Statement):
    """Read the site_id of a LocationAddress (produced by the GetAttr rewrite)."""

    name = "site_id"
    traits = frozenset({ir.Pure()})
    address: ir.SSAValue = info.argument(LocationAddressType)
    result: ir.ResultValue = info.result(types.Int)


@statement(dialect=dialect)
class ZoneId(ir.Statement):
    """Read the zone_id of a LocationAddress (produced by the GetAttr rewrite)."""

    name = "zone_id"
    traits = frozenset({ir.Pure()})
    address: ir.SSAValue = info.argument(LocationAddressType)
    result: ir.ResultValue = info.result(types.Int)
