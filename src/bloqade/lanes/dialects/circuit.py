from kirin import ir, types
from kirin.decl import info, statement

from .bytecode import ByteCodeStmt, ExitRegion

dialect = ir.Dialect(name="bytecode.circuit")


@statement(dialect=dialect)
class CZ(ByteCodeStmt):
    pairs: tuple[tuple[int, int], ...] = info.attribute()


@statement(dialect=dialect)
class R(ByteCodeStmt):
    inputs: tuple[int, ...] = info.attribute()

    axis_angle: ir.SSAValue = info.argument(type=types.Float)
    rotation_angle: ir.SSAValue = info.argument(type=types.Float)


@statement(dialect=dialect)
class Rz(ByteCodeStmt):
    inputs: tuple[int, ...] = info.attribute()
    rotation_angle: ir.SSAValue = info.argument(type=types.Float)


@statement(dialect=dialect)
class TerminalMeasure(ExitRegion):
    qubits: tuple[int, ...] = info.attribute()
