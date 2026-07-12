from kirin import interp
from kirin.analysis import const
from kirin.analysis.const.prop import Frame

from bloqade.lanes.bytecode.encoding import LocationAddress

from ._dialect import dialect
from .stmts import CzPartner


@dialect.register(key="constprop")
class CzPartnerConstProp(interp.MethodTable):
    """Resolve ``arch.cz_partner(loc)`` during const propagation.

    Returns the partner location as ``const.Value`` once ``arch_spec`` is
    bound (by ``BindCzPartnerArchSpec`` in the pipeline) and the ``address``
    operand is itself a constant ``LocationAddress``. Returns ``top()`` for
    every other case so downstream consumers stay non-const — that's how
    unresolved ``CzPartner`` surfaces as a compilation error today (via the
    existing move_to "locations must be compile-time constants" check).
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
            return (const.Result.top(),)
        return (const.Value(partner),)
