from kirin import ir, lowering, types
from kirin.decl import info, statement
from kirin.lowering.python.binding import wraps

from bloqade import types as bloqade_types

from ..layout.encoding import LaneAddress, LocationAddress, ZoneAddress
from ..types import MeasurementFuture, MeasurementFutureType

dialect = ir.Dialect(name="lanes.move")


@statement(dialect=dialect)
class Initialize(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})
    location_addresses: tuple[LocationAddress, ...] = info.attribute()


@statement(dialect=dialect)
class CZ(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})

    zone_address: ZoneAddress = info.attribute()


@statement(dialect=dialect)
class LocalR(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})

    location_addresses: tuple[LocationAddress, ...] = info.attribute()
    axis_angle: ir.SSAValue = info.argument(type=types.Float)
    rotation_angle: ir.SSAValue = info.argument(type=types.Float)


@statement(dialect=dialect)
class GlobalR(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})

    axis_angle: ir.SSAValue = info.argument(type=types.Float)
    rotation_angle: ir.SSAValue = info.argument(type=types.Float)


@statement(dialect=dialect)
class LocalRz(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})

    location_addresses: tuple[LocationAddress, ...] = info.attribute()
    rotation_angle: ir.SSAValue = info.argument(type=types.Float)


@statement(dialect=dialect)
class GlobalRz(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})

    rotation_angle: ir.SSAValue = info.argument(type=types.Float)


@statement(dialect=dialect)
class Move(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})

    lanes: tuple[LaneAddress, ...] = info.attribute()


@statement(dialect=dialect)
class EndMeasure(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})

    zone_address: ZoneAddress = info.attribute()

    result: ir.ResultValue = info.result(MeasurementFutureType)


@statement(dialect=dialect)
class GetMeasurementResult(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})

    measurement_future: ir.SSAValue = info.argument(MeasurementFutureType)
    index: int = info.attribute()

    result: ir.ResultValue = info.result(type=bloqade_types.MeasurementResultType)


@wraps(Initialize)
def initialize(*, location_addresses: tuple[LocationAddress, ...]) -> None: ...


@wraps(CZ)
def cz(*, zone_address: ZoneAddress) -> None: ...


@wraps(LocalR)
def local_r(
    axis_angle: float,
    rotation_angle: float,
    *,
    location_addresses: tuple[LocationAddress, ...],
) -> None: ...


@wraps(GlobalR)
def global_r(axis_angle: float, rotation_angle: float) -> None: ...


@wraps(LocalRz)
def local_rz(
    rotation_angle: float, *, location_addresses: tuple[LocationAddress, ...]
) -> None: ...


@wraps(GlobalRz)
def global_rz(rotation_angle: float) -> None: ...


@wraps(Move)
def move(*, lanes: tuple[LaneAddress, ...]) -> None: ...


@wraps(EndMeasure)
def end_measure(*, zone_address: ZoneAddress) -> MeasurementFuture: ...
