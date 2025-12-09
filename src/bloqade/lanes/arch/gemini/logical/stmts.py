from bloqade.geometry.dialects import filled
from kirin import decl, ir, types
from kirin.decl import info
from kirin.dialects import ilist

from bloqade.lanes.layout.encoding import Direction

dialect = ir.Dialect("gemini.logical")


@decl.statement(dialect=dialect)
class Fill(ir.Statement):
    filled: ir.SSAValue = info.argument(
        ilist.IListType[types.Tuple[types.Int, types.Int], types.Any]
    )
    vacant: ir.SSAValue = info.argument(
        ilist.IListType[types.Tuple[types.Int, types.Int], types.Any]
    )


NumGates = types.TypeVar("NumGates")
NumRows = types.TypeVar("NumRows")
NumCols = types.TypeVar("NumCols")


@decl.statement(dialect=dialect)
class LogicalInitialize(ir.Statement):
    location_groups: ir.SSAValue = info.argument(
        ilist.IListType[filled.FilledGridType[NumRows, NumCols], NumGates]
    )
    thetas: ir.SSAValue = info.argument(ilist.IListType[types.Float, NumGates])
    phis: ir.SSAValue = info.argument(ilist.IListType[types.Float, NumGates])
    lams: ir.SSAValue = info.argument(ilist.IListType[types.Float, NumGates])


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
