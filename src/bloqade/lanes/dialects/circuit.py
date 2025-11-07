from kirin import ir, types
from kirin.decl import info, statement

from .execute import ExitLowLevel, LowLevelStmt

dialect = ir.Dialect(name="lowlevel.circuit")


@statement(dialect=dialect)
class CZ(LowLevelStmt):
    pairs: tuple[tuple[int, int], ...] = info.attribute()


@statement(dialect=dialect)
class R(LowLevelStmt):
    inputs: tuple[int, ...] = info.attribute()

    axis_angle: ir.SSAValue = info.argument(type=types.Float)
    rotation_angle: ir.SSAValue = info.argument(type=types.Float)


@statement(dialect=dialect)
class Rz(LowLevelStmt):
    inputs: tuple[int, ...] = info.attribute()
    rotation_angle: ir.SSAValue = info.argument(type=types.Float)


@statement(dialect=dialect)
class TerminalMeasure(ExitLowLevel):
    qubits: tuple[int, ...] = info.attribute()
