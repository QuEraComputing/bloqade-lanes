from kirin import decl, ir, types
from kirin.decl import info
from kirin.dialects import ilist

from bloqade.lanes.bytecode.encoding import Direction
from bloqade.lanes.dialects.move import StatefulStatement

dialect = ir.Dialect("gemini.logical")


@decl.statement(dialect=dialect)
class Fill(StatefulStatement):
    logical_addresses: ir.SSAValue = info.argument(
        ilist.IListType[types.Tuple[types.Int, types.Int], types.Any]
    )


NumLocations = types.TypeVar("NumLocations")


@decl.statement(dialect=dialect)
class LogicalInitialize(StatefulStatement):
    thetas: ir.SSAValue = info.argument(ilist.IListType[types.Float, NumLocations])
    phis: ir.SSAValue = info.argument(ilist.IListType[types.Float, NumLocations])
    lams: ir.SSAValue = info.argument(ilist.IListType[types.Float, NumLocations])
    logical_addresses: ir.SSAValue = info.argument(
        ilist.IListType[types.Tuple[types.Int, types.Int], NumLocations]
    )


SitesPerWord = types.TypeVar("SitesPerWord")


@decl.statement(dialect=dialect)
class SiteBusMove(StatefulStatement):
    y_mask: ir.SSAValue = info.argument(ilist.IListType[types.Bool, SitesPerWord])
    word: int = info.attribute()
    bus_id: int = info.attribute()
    direction: Direction = info.attribute()
    sites_per_word: int = info.attribute()


@decl.statement(dialect=dialect)
class WordBusMove(StatefulStatement):
    y_mask: ir.SSAValue = info.argument(ilist.IListType[types.Bool, SitesPerWord])
    direction: Direction = info.attribute()
    sites_per_word: int = info.attribute()
