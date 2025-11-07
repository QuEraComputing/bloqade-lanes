from kirin import ir, types
from kirin.decl import info, statement

from ..layout.encoding import MoveType, PhysicalAddress
from .bytecode import ByteCodeStmt, ExitRegion

dialect = ir.Dialect(name="bytecode.move")


@statement(dialect=dialect)
class CZ(ByteCodeStmt):
    pass


@statement(dialect=dialect)
class LocalR(ByteCodeStmt):
    physical_addr: tuple[PhysicalAddress, ...] = info.attribute()

    axis_angle: ir.SSAValue = info.argument(type=types.Float)
    rotation_angle: ir.SSAValue = info.argument(type=types.Float)


@statement(dialect=dialect)
class GlobalR(ByteCodeStmt):
    axis_angle: ir.SSAValue = info.argument(type=types.Float)
    rotation_angle: ir.SSAValue = info.argument(type=types.Float)


@statement(dialect=dialect)
class localRz(ByteCodeStmt):
    physical_addr: tuple[PhysicalAddress, ...] = info.attribute()
    rotation_angle: ir.SSAValue = info.argument(type=types.Float)


@statement(dialect=dialect)
class GlobalRz(ByteCodeStmt):
    rotation_angle: ir.SSAValue = info.argument(type=types.Float)


class Move(ByteCodeStmt):
    lanes: tuple[MoveType, ...] = info.attribute()


@statement(dialect=dialect)
class TerminalMeasure(ExitRegion):
    physical_addr: tuple[PhysicalAddress, ...] = info.attribute()
