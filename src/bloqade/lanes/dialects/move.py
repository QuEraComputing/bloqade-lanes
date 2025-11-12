from kirin import ir, types
from kirin.decl import info, statement

from bloqade import types as bloqade_types

from ..layout.encoding import LocationAddress, LaneAddress, ZoneAddress
from ..types import MeasurementFutureType

dialect = ir.Dialect(name="lowlevel.move")


@statement(dialect=dialect)
class Initialize(ir.Statement):
    location_addresses: tuple[LocationAddress, ...] = info.attribute()


@statement(dialect=dialect)
class CZ(ir.Statement):
    zone_address: ZoneAddress = info.attribute()


@statement(dialect=dialect)
class LocalR(ir.Statement):
    location_addresses: tuple[LocationAddress, ...] = info.attribute()
    axis_angle: ir.SSAValue = info.argument(type=types.Float)
    rotation_angle: ir.SSAValue = info.argument(type=types.Float)


@statement(dialect=dialect)
class GlobalR(ir.Statement):
    axis_angle: ir.SSAValue = info.argument(type=types.Float)
    rotation_angle: ir.SSAValue = info.argument(type=types.Float)


@statement(dialect=dialect)
class LocalRz(ir.Statement):
    location_addresses: tuple[LocationAddress, ...] = info.attribute()
    rotation_angle: ir.SSAValue = info.argument(type=types.Float)


@statement(dialect=dialect)
class GlobalRz(ir.Statement):
    rotation_angle: ir.SSAValue = info.argument(type=types.Float)


@statement(dialect=dialect)
class Move(ir.Statement):
    lanes: tuple[LaneAddress, ...] = info.attribute()


@statement(dialect=dialect)
class EndMeasure(ir.Statement):
    zone_address: ZoneAddress = info.attribute()

    result: ir.ResultValue = info.result(MeasurementFutureType)


@statement(dialect=dialect)
class GetMeasurementResult(ir.Statement):
    measurement_future: ir.SSAValue = info.argument(MeasurementFutureType)
    index: int = info.attribute()

    result: ir.ResultValue = info.result(type=bloqade_types.MeasurementResultType)
