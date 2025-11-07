from kirin import ir, types
from kirin.decl import info, statement

from ..layout.encoding import LocationAddress, MoveType
from .lowlevel import ExitLowLevel, LowLevelStmt

dialect = ir.Dialect(name="bytecode.move")


@statement(dialect=dialect)
class CZ(LowLevelStmt):
    pass


@statement(dialect=dialect)
class LocalR(LowLevelStmt):
    physical_addr: tuple[LocationAddress, ...] = info.attribute()

    axis_angle: ir.SSAValue = info.argument(type=types.Float)
    rotation_angle: ir.SSAValue = info.argument(type=types.Float)


@statement(dialect=dialect)
class GlobalR(LowLevelStmt):
    axis_angle: ir.SSAValue = info.argument(type=types.Float)
    rotation_angle: ir.SSAValue = info.argument(type=types.Float)


@statement(dialect=dialect)
class localRz(LowLevelStmt):
    physical_addr: tuple[LocationAddress, ...] = info.attribute()
    rotation_angle: ir.SSAValue = info.argument(type=types.Float)


@statement(dialect=dialect)
class GlobalRz(LowLevelStmt):
    rotation_angle: ir.SSAValue = info.argument(type=types.Float)


class Move(LowLevelStmt):
    lanes: tuple[MoveType, ...] = info.attribute()


@statement(dialect=dialect)
class TerminalMeasure(ExitLowLevel):
    physical_addr: tuple[LocationAddress, ...] = info.attribute()
