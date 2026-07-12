from kirin import ir, lowering, types
from kirin.decl import info, statement

from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode.encoding import LocationAddress

from ._dialect import dialect

LocationAddressType = types.PyClass(LocationAddress)


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
class CzPartner(ir.Statement):
    """Resolve the CZ blockade-partner LocationAddress of a location.

    ``arch.cz_partner(loc)`` returns the location an atom must occupy to be
    CZ-entangled with an atom at ``loc``. The result is materialized by
    standard const-prop: once ``arch_spec`` is bound (by the pipeline's
    ``BindCzPartnerArchSpec`` pass) and the ``address`` operand is constant,
    the registered constprop impl returns a ``const.Value(LocationAddress)``
    so the rest of the fold pipeline propagates it (e.g. into a ``move_to``
    locations list).

    Any ``CzPartner`` that survives resolution (because ``arch_spec`` was not
    bound, ``address`` did not const-fold, or the architecture has no partner
    for that location) keeps the downstream ``move_to`` non-const, which the
    existing move_to validation reports.
    """

    name = "cz_partner"
    traits = frozenset({ir.Pure(), lowering.FromPythonCall()})
    address: ir.SSAValue = info.argument(LocationAddressType)
    arch_spec: ArchSpec | None = info.attribute(default=None)
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
