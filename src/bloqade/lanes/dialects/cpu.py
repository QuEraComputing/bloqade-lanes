from kirin import ir, types
from kirin.decl import info, statement

from .execute import LowLevelStmt

dialect = ir.Dialect(name="lowlevel.cpu")


@statement(dialect=dialect)
class StaticFloat(LowLevelStmt):
    traits = frozenset({ir.ConstantLike()})

    value: float = info.attribute(type=types.Float)
    result: ir.ResultValue = info.result(type=types.Float)


@statement(dialect=dialect)
class StaticInt(LowLevelStmt):
    traits = frozenset({ir.ConstantLike()})

    value: int = info.attribute(type=types.Int)

    result: ir.ResultValue = info.result(type=types.Int)
