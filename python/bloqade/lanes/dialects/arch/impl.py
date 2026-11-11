from kirin import interp
from kirin.interp import Frame

from bloqade.lanes.bytecode.encoding import LocationAddress

from ._dialect import dialect
from .stmts import CzPartner, Loc, SiteId, WordId, ZoneId


@dialect.register
class _ArchInterpreter(interp.MethodTable):
    @interp.impl(Loc)
    def loc(self, interp_: interp.Interpreter, frame: Frame, stmt: Loc):
        """Concrete evaluation of ``loc`` — resolve a ``(zone, row, col)`` grid
        coordinate to a ``LocationAddress`` via the bound arch spec.

        Requires ``arch_spec`` to be bound (by ``BindArchSpec``) and the
        coordinates to be concrete in the interpreter frame. Raises
        ``InterpreterError`` otherwise. During const-folding (``run_no_raise``)
        this is caught and the frame falls back to bottom, so an unbound arch
        spec at kernel-decoration time gracefully skips folding without
        corrupting the IR; during analysis (``no_raise=False``) it surfaces as
        an error, consistent with the const-prop table in ``constprop.py``.
        """
        if stmt.arch_spec is None:
            raise interp.InterpreterError(
                "Loc has no arch_spec bound; run BindArchSpec before AggressiveUnroll."
            )
        zone = frame.get(stmt.zone)
        row = frame.get(stmt.row)
        col = frame.get(stmt.col)
        location = stmt.arch_spec.location_at(zone, row, col)
        if location is None:
            raise interp.InterpreterError(
                f"loc: no location address at (zone={zone}, row={row}, col={col}) "
                "in the architecture spec."
            )
        return (location,)

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

        Requires ``arch_spec`` to be bound (by ``BindArchSpec``) and ``address``
        to be a concrete ``LocationAddress`` in the interpreter frame. Raises
        ``InterpreterError`` otherwise (see ``loc`` for the rationale).
        """
        if stmt.arch_spec is None:
            raise interp.InterpreterError(
                "CzPartner has no arch_spec bound; "
                "run BindArchSpec before AggressiveUnroll."
            )
        address = frame.get(stmt.address)
        if not isinstance(address, LocationAddress):
            raise interp.InterpreterError(
                f"CzPartner address is not a concrete LocationAddress: {address!r}"
            )
        partner = stmt.arch_spec.get_cz_partner(address)
        if partner is None:
            raise interp.InterpreterError(
                f"cz_partner: no CZ partner for {address!r} in the architecture spec."
            )
        return (partner,)
