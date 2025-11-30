from kirin import decl, ir, types
from kirin.decl import info
from kirin.dialects import ilist

from bloqade.lanes.layout.encoding import Direction

from .impls import generate_arch


def get_arch_spec():
    return generate_arch(hypercube_dims=1, word_size_y=5)


dialect = ir.Dialect("gemini.logical")


@decl.statement(dialect=dialect)
class SiteBusMove(ir.Statement):
    y_positions: ir.SSAValue = info.argument(ilist.IListType[types.Int])
    word: int = info.attribute()
    bus_id: int = info.attribute()
    direction: Direction = info.attribute()


@decl.statement(dialect=dialect)
class WordBusMove(ir.Statement):
    y_positions: ir.SSAValue = info.argument(ilist.IListType[types.Int])
    direction: Direction = info.attribute()
