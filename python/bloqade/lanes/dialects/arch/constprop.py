from kirin import interp
from kirin.analysis import const
from kirin.analysis.const.prop import Frame

from bloqade.lanes.bytecode.encoding import LocationAddress

from ._dialect import dialect
from .stmts import CzPartner, Loc


@dialect.register(key="constprop")
class ArchResolvedConstProp(interp.MethodTable):
    """Const-prop for arch-resolved statements (``Loc`` / ``CzPartner``).

    Materializes the ``LocationAddress`` as a ``const.Value`` once ``arch_spec``
    is bound (by ``BindArchSpec`` in the pipeline) and the operands are constant.
    While still unresolved (arch spec unbound, or operands not yet constant) it
    returns ``top()`` so the statement stays non-const and folds later. If the
    arch API resolves to ``None`` — no CZ partner, or no atom at the requested
    ``(zone, row, col)`` — it raises ``InterpreterError`` so the error surfaces
    during analysis (consistent with the interpreter impls in ``impl.py``).
    """

    @interp.impl(CzPartner)
    def cz_partner(self, _, frame: Frame, stmt: CzPartner):
        if stmt.arch_spec is None:
            return (const.Result.top(),)
        addr = frame.get(stmt.address)
        if not isinstance(addr, const.Value) or not isinstance(
            addr.data, LocationAddress
        ):
            return (const.Result.top(),)
        partner = stmt.arch_spec.get_cz_partner(addr.data)
        if partner is None:
            raise interp.InterpreterError(
                f"cz_partner: no CZ partner for {addr.data!r} in the architecture spec."
            )
        return (const.Value(partner),)

    @interp.impl(Loc)
    def loc(self, _, frame: Frame, stmt: Loc):
        if stmt.arch_spec is None:
            return (const.Result.top(),)
        zone_v = frame.get(stmt.zone)
        row_v = frame.get(stmt.row)
        col_v = frame.get(stmt.col)
        if not (
            isinstance(zone_v, const.Value)
            and isinstance(row_v, const.Value)
            and isinstance(col_v, const.Value)
        ):
            return (const.Result.top(),)
        zone, row, col = zone_v.data, row_v.data, col_v.data
        if not (
            isinstance(zone, int) and isinstance(row, int) and isinstance(col, int)
        ):
            return (const.Result.top(),)
        location = stmt.arch_spec.location_at(zone, row, col)
        if location is None:
            raise interp.InterpreterError(
                f"loc: no location address at (zone={zone}, row={row}, col={col}) "
                "in the architecture spec."
            )
        return (const.Value(location),)
