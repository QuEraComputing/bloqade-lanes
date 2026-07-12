from kirin import interp
from kirin.interp import Frame

from ._dialect import dialect
from .stmts import CzPartner, Loc, SiteId, WordId, ZoneId


@dialect.register
class _LocInterpreter(interp.MethodTable):
    @interp.impl(Loc)
    def loc(self, interp_: interp.Interpreter, frame: Frame, stmt: Loc):
        from bloqade.lanes.bytecode.encoding import LocationAddress

        z = frame.get(stmt.zone_id)
        w = frame.get(stmt.word_id)
        s = frame.get(stmt.site_id)
        return (LocationAddress(zone_id=z, word_id=w, site_id=s),)

    @interp.impl(WordId)
    def word_id(self, interp_: interp.Interpreter, frame: Frame, stmt: WordId):
        return (frame.get(stmt.address).word_id,)

    @interp.impl(SiteId)
    def site_id(self, interp_: interp.Interpreter, frame: Frame, stmt: SiteId):
        return (frame.get(stmt.address).site_id,)

    @interp.impl(ZoneId)
    def zone_id(self, interp_: interp.Interpreter, frame: Frame, stmt: ZoneId):
        return (frame.get(stmt.address).zone_id,)

    @interp.impl(CzPartner)
    def cz_partner(self, interp_: interp.Interpreter, frame: Frame, stmt: CzPartner):
        """Concrete evaluation of ``cz_partner`` used by the ilist constprop's
        ``try_eval_const_pure`` when unrolling pure lambdas at pipeline time.

        Requires ``arch_spec`` to be bound (by ``BindCzPartnerArchSpec``) and
        ``address`` to be a concrete ``LocationAddress`` in the interpreter
        frame. Raises ``NotImplementedError`` otherwise — at kernel-decoration
        time (arch_spec not yet set) this causes ``run_no_raise`` to return an
        empty frame, which gracefully skips const-folding without corrupting the
        IR.
        """
        from bloqade.lanes.bytecode.encoding import LocationAddress

        if stmt.arch_spec is None:
            raise NotImplementedError(
                "CzPartner has no arch_spec bound; "
                "run BindCzPartnerArchSpec before AggressiveUnroll."
            )
        address = frame.get(stmt.address)
        if not isinstance(address, LocationAddress):
            raise NotImplementedError(
                f"CzPartner address is not a concrete LocationAddress: {address!r}"
            )
        partner = stmt.arch_spec.get_cz_partner(address)
        if partner is None:
            raise NotImplementedError(
                f"No CZ partner for {address!r} in the architecture spec."
            )
        return (partner,)
