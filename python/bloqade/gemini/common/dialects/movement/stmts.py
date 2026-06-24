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
class CzPartner(ir.Statement):
    """Resolve the CZ blockade-partner LocationAddress of a location.

    ``movement.cz_partner(loc)`` returns the location an atom must occupy to
    be CZ-entangled with an atom at ``loc``. Lowered from a Python call; it
    cannot const-fold on its own (the partner relation lives in the arch
    spec, not in the kernel), so it is resolved at compile time by
    ``ResolveCzPartner`` — which has the pipeline's ``arch_spec`` — into a
    constant ``Loc``. Any ``CzPartner`` that survives resolution (its address
    did not const-fold) is reported by the existing location validation.
    """

    # Deliberately NOT ir.Pure. The partner relation lives in the arch spec
    # (available only at compile time, via ResolveCzPartner), so cz_partner has
    # no const-fold impl. If it were marked Pure, constant folding would hoist
    # an enclosing ``ilist.map`` closure into an opaque constant ``Method``,
    # burying the cz_partner statement inside a body that ResolveCzPartner's
    # ``Walk`` never descends into (and that ``CallGraphPass`` does not follow,
    # since it doesn't treat ``ilist.map``'s ``fn`` operand as a call edge) — so
    # it would never be resolved. Leaving it impure keeps the map's closure
    # inline, where the rewrite reaches it once the unroller expands the map.
    # Supporting Pure here needs a deeper fix: call-graph rewrites must descend
    # into ``ilist.map`` function bodies — tracked in QuEraComputing/
    # bloqade-circuit#830.
    name = "cz_partner"
    traits = frozenset({lowering.FromPythonCall()})
    address: ir.SSAValue = info.argument(LocationAddressType)
    result: ir.ResultValue = info.result(LocationAddressType)


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
