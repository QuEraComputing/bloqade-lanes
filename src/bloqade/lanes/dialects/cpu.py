from kirin import ir, types
from kirin.decl import info, statement

dialect = ir.Dialect(name="lowlevel.cpu")


@statement(dialect=dialect)
class StaticFloat(ir.Statement):
    traits = frozenset({ir.ConstantLike()})

    value: float = info.attribute(type=types.Float)
    result: ir.ResultValue = info.result(type=types.Float)
