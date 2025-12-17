from typing import Any

from kirin import ir, lowering, types
from kirin.decl import info, statement
from kirin.dialects import ilist
from kirin.lowering.python.binding import wraps

from bloqade import types as bloqade_types

from ..layout.encoding import LaneAddress, LocationAddress, ZoneAddress
from ..types import MeasurementFuture, MeasurementFutureType

dialect = ir.Dialect(name="lanes.move")


@statement(dialect=dialect)
class Fill(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})

    location_addresses: tuple[LocationAddress, ...] = info.attribute()


@statement(dialect=dialect)
class LogicalInitialize(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})

    location_addresses: tuple[LocationAddress, ...] = info.attribute()
    thetas: tuple[ir.SSAValue, ...] = info.argument(type=types.Float)
    phis: tuple[ir.SSAValue, ...] = info.argument(type=types.Float)
    lams: tuple[ir.SSAValue, ...] = info.argument(type=types.Float)


@statement(dialect=dialect)
class PhysicalInitialize(ir.Statement):
    """Placeholder for when rewriting to simulation"""

    traits = frozenset({lowering.FromPythonCall()})

    location_addresses: tuple[tuple[LocationAddress, ...], ...] = info.attribute()
    thetas: tuple[ir.SSAValue, ...] = info.argument(type=types.Float)
    phis: tuple[ir.SSAValue, ...] = info.argument(type=types.Float)
    lams: tuple[ir.SSAValue, ...] = info.argument(type=types.Float)


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

    zone_addresses: tuple[ZoneAddress, ...] = info.attribute()
    result: ir.ResultValue = info.result(MeasurementFutureType)


@statement(dialect=dialect)
class GetFutureResult(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})

    measurement_future: ir.SSAValue = info.argument(MeasurementFutureType)
    zone_address: ZoneAddress = info.attribute()

    result: ir.ResultValue = info.result(
        type=ilist.IListType[bloqade_types.MeasurementResultType, types.Any]
    )


@statement(dialect=dialect)
class GetZoneIndex(ir.Statement):
    """Get the index of a location within a zone"""

    traits = frozenset({lowering.FromPythonCall(), ir.Pure()})

    zone_address: ZoneAddress = info.attribute()
    location_address: LocationAddress = info.attribute()

    result: ir.ResultValue = info.result(type=types.Int)


@wraps(Fill)
def fill(*, location_addresses: tuple[LocationAddress, ...]) -> None: ...


@wraps(LogicalInitialize)
def logical_initialize(
    thetas: tuple[float, ...],
    phis: tuple[float, ...],
    lams: tuple[float, ...],
    *,
    location_addresses: tuple[LocationAddress, ...],
) -> None: ...


@wraps(PhysicalInitialize)
def physical_initialize(
    thetas: tuple[float, ...],
    phis: tuple[float, ...],
    lams: tuple[float, ...],
    *,
    location_addresses: tuple[tuple[LocationAddress, ...], ...],
) -> None: ...


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
def end_measure(*, zone_addresses: tuple[ZoneAddress, ...]) -> MeasurementFuture: ...


@wraps(GetFutureResult)
def get_future_result(
    measurement_future: MeasurementFuture, *, zone_address: ZoneAddress
) -> ilist.IList[bloqade_types.MeasurementResult, Any]: ...


@wraps(GetZoneIndex)
def get_zone_index(
    *, zone_address: ZoneAddress, location_address: LocationAddress
) -> int: ...
