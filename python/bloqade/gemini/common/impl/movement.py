from kirin import interp
from kirin.interp import Frame

from bloqade.gemini.common.dialects.movement import dialect as movement_dialect
from bloqade.gemini.common.dialects.movement.stmts import Loc, SiteId, WordId, ZoneId


@movement_dialect.register
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
