from kirin import ir, lowering, types
from kirin.decl import info, statement

from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode.encoding import LocationAddress

from ._dialect import dialect

LocationAddressType = types.PyClass(LocationAddress)


@statement(init=False)
class ArchResolvedStmt(ir.Statement):
    """Base for arch-dialect statements resolved against a bound ``ArchSpec``.

    Holds ``arch_spec`` (populated by the pipeline's ``BindArchSpec`` pass before
    folding). While unbound, const-prop and the interpreter leave the statement
    unresolved; once ``arch_spec`` is bound and the operands are constant they
    materialize the ``LocationAddress`` result. Subclasses: ``Loc``, ``CzPartner``.
    """

    arch_spec: ArchSpec | None = info.attribute(default=None)


@statement(dialect=dialect)
class Loc(ArchResolvedStmt):
    """Construct a LocationAddress from a ``(zone, row, col)`` grid coordinate.

    Resolved against the architecture's addressing scheme
    (``ArchSpec.location_at``) once ``arch_spec`` is bound by ``BindArchSpec``;
    ``zone`` / ``row`` / ``col`` must be compile-time constants.
    """

    name = "loc"
    traits = frozenset({ir.Pure(), lowering.FromPythonCall()})
    zone: ir.SSAValue = info.argument(types.Int)
    row: ir.SSAValue = info.argument(types.Int)
    col: ir.SSAValue = info.argument(types.Int)
    result: ir.ResultValue = info.result(LocationAddressType)


@statement(dialect=dialect)
class CzPartner(ArchResolvedStmt):
    """Resolve the CZ blockade-partner LocationAddress of a location.

    ``arch.cz_partner(loc)`` returns the location an atom must occupy to be
    CZ-entangled with an atom at ``loc``. The result is materialized by standard
    const-prop: once ``arch_spec`` is bound (by ``BindArchSpec``) and the
    ``address`` operand is constant, the registered constprop impl returns a
    ``const.Value(LocationAddress)`` so the rest of the fold pipeline propagates
    it (e.g. into a ``move_to`` locations list).

    Any ``CzPartner`` that survives resolution (because ``arch_spec`` was not
    bound, ``address`` did not const-fold, or the architecture has no partner
    for that location) keeps the downstream ``move_to`` non-const, which the
    existing move_to validation reports.
    """

    name = "cz_partner"
    traits = frozenset({ir.Pure(), lowering.FromPythonCall()})
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
