from kirin import decl, ir, types
from kirin.decl import info
from kirin.dialects import ilist

from bloqade.lanes.layout.encoding import Direction

dialect = ir.Dialect("gemini.logical")


@decl.statement(dialect=dialect)
class Fill(ir.Statement):
    filled_logical_addresses: ir.SSAValue = info.argument(
        ilist.IListType[types.Tuple[types.Int, types.Int], types.Any]
    )
    vacant_logical_addresses: ir.SSAValue = info.argument(
        ilist.IListType[types.Tuple[types.Int, types.Int], types.Any]
    )


NumLocations = types.TypeVar("NumLocations")


@decl.statement(dialect=dialect)
class LogicalInitialize(ir.Statement):
    thetas: ir.SSAValue = info.argument(ilist.IListType[types.Float, NumLocations])
    phis: ir.SSAValue = info.argument(ilist.IListType[types.Float, NumLocations])
    lams: ir.SSAValue = info.argument(ilist.IListType[types.Float, NumLocations])
    logical_addresses: ir.SSAValue = info.argument(
        ilist.IListType[types.Tuple[types.Int, types.Int], NumLocations]
    )


@decl.statement(dialect=dialect)
class SiteBusMove(ir.Statement):
    y_mask: ir.SSAValue = info.argument(ilist.IListType[types.Bool, types.Literal(5)])
    word: int = info.attribute()
    bus_id: int = info.attribute()
    direction: Direction = info.attribute()


@decl.statement(dialect=dialect)
class WordBusMove(ir.Statement):
    y_mask: ir.SSAValue = info.argument(ilist.IListType[types.Bool, types.Literal(5)])
    direction: Direction = info.attribute()
