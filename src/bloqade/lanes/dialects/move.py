from kirin import ir, types
from kirin.decl import info, statement

from ..layout.encoding import LocationAddress, MoveType
from .execute import ExitLowLevel, QuantumStmt

dialect = ir.Dialect(name="lowlevel.move")


@statement(dialect=dialect)
class CZ(QuantumStmt):
    pass


@statement(dialect=dialect)
class LocalR(QuantumStmt):
    physical_addr: tuple[LocationAddress, ...] = info.attribute()

    axis_angle: ir.SSAValue = info.argument(type=types.Float)
    rotation_angle: ir.SSAValue = info.argument(type=types.Float)


@statement(dialect=dialect)
class GlobalR(QuantumStmt):
    axis_angle: ir.SSAValue = info.argument(type=types.Float)
    rotation_angle: ir.SSAValue = info.argument(type=types.Float)


@statement(dialect=dialect)
class LocalRz(QuantumStmt):
    physical_addr: tuple[LocationAddress, ...] = info.attribute()
    rotation_angle: ir.SSAValue = info.argument(type=types.Float)


@statement(dialect=dialect)
class GlobalRz(QuantumStmt):
    rotation_angle: ir.SSAValue = info.argument(type=types.Float)


class Move(QuantumStmt):
    lanes: tuple[MoveType, ...] = info.attribute()


@statement(dialect=dialect)
class TerminalMeasure(ExitLowLevel):
    physical_addr: tuple[LocationAddress, ...] = info.attribute()
