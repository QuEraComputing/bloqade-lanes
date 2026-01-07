from dataclasses import dataclass
from typing import Any

from kirin import ir, lowering, types
from kirin.decl import info, statement
from kirin.dialects import ilist
from kirin.lowering.python.binding import wraps

from bloqade import types as bloqade_types

from ..layout.encoding import LaneAddress, LocationAddress, ZoneAddress
from ..types import MeasurementFuture, MeasurementFutureType, State, StateType

dialect = ir.Dialect(name="lanes.move")


@dataclass(frozen=True)
class ConsumesState(ir.Trait):
    terminates: bool
    argument_index: int = 0

    def get_state_argument(self, stmt: ir.Statement) -> ir.SSAValue:
        return stmt.args[self.argument_index]


@dataclass(frozen=True)
class EmitsState(ir.Trait):
    originates: bool
    result_index: int = 0

    def get_state_result(self, stmt: ir.Statement) -> ir.ResultValue:
        return stmt.results[0]


@statement(dialect=dialect)
class Load(ir.Statement):
    """Load a previously stored atom state."""

    traits = frozenset({lowering.FromPythonCall(), EmitsState(True, 0)})

    result: ir.ResultValue = info.result(StateType)


@statement(dialect=dialect)
class Store(ir.Statement):
    traits = frozenset({lowering.FromPythonCall(), ConsumesState(False)})

    current_state: ir.SSAValue = info.argument(StateType)


@statement(dialect=None)
class StatefulStatement(ir.Statement):
    """Base class for statements that modify the atom state."""

    traits = frozenset(
        {lowering.FromPythonCall(), ConsumesState(False), EmitsState(False)}
    )

    current_state: ir.SSAValue = info.argument(StateType)
    result: ir.ResultValue = info.result(StateType)


@statement(dialect=dialect)
class Fill(StatefulStatement):

    location_addresses: tuple[LocationAddress, ...] = info.attribute()


@statement(dialect=dialect)
class LogicalInitialize(StatefulStatement):
    location_addresses: tuple[LocationAddress, ...] = info.attribute()
    thetas: tuple[ir.SSAValue, ...] = info.argument(type=types.Float)
    phis: tuple[ir.SSAValue, ...] = info.argument(type=types.Float)
    lams: tuple[ir.SSAValue, ...] = info.argument(type=types.Float)


@statement(dialect=dialect)
class PhysicalInitialize(StatefulStatement):
    """Placeholder for when rewriting to simulation"""

    location_addresses: tuple[tuple[LocationAddress, ...], ...] = info.attribute()
    thetas: tuple[ir.SSAValue, ...] = info.argument(type=types.Float)
    phis: tuple[ir.SSAValue, ...] = info.argument(type=types.Float)
    lams: tuple[ir.SSAValue, ...] = info.argument(type=types.Float)


@statement(dialect=dialect)
class CZ(StatefulStatement):
    zone_address: ZoneAddress = info.attribute()


@statement(dialect=dialect)
class LocalR(StatefulStatement):
    location_addresses: tuple[LocationAddress, ...] = info.attribute()
    axis_angle: ir.SSAValue = info.argument(type=types.Float)
    rotation_angle: ir.SSAValue = info.argument(type=types.Float)


@statement(dialect=dialect)
class GlobalR(StatefulStatement):
    axis_angle: ir.SSAValue = info.argument(type=types.Float)
    rotation_angle: ir.SSAValue = info.argument(type=types.Float)


@statement(dialect=dialect)
class LocalRz(StatefulStatement):
    location_addresses: tuple[LocationAddress, ...] = info.attribute()
    rotation_angle: ir.SSAValue = info.argument(type=types.Float)


@statement(dialect=dialect)
class GlobalRz(StatefulStatement):
    rotation_angle: ir.SSAValue = info.argument(type=types.Float)


@statement(dialect=dialect)
class Move(StatefulStatement):
    lanes: tuple[LaneAddress, ...] = info.attribute()


@statement(dialect=dialect)
class EndMeasure(ir.Statement):
    """Start a measurement over the specified zones. Returns a MeasurementFuture."""

    traits = frozenset({lowering.FromPythonCall(), ConsumesState(True)})
    current_state: ir.SSAValue = info.argument(StateType)
    zone_addresses: tuple[ZoneAddress, ...] = info.attribute()
    result: ir.ResultValue = info.result(MeasurementFutureType)


@statement(dialect=dialect)
class GetFutureResult(ir.Statement):
    """Get the measurement results from a measurement future"""

    traits = frozenset({lowering.FromPythonCall()})

    measurement_future: ir.SSAValue = info.argument(MeasurementFutureType)
    zone_address: ZoneAddress = info.attribute()
    location_address: LocationAddress = info.attribute()

    result: ir.ResultValue = info.result(type=bloqade_types.MeasurementResultType)


@wraps(Load)
def load() -> State: ...


@wraps(Store)
def store(current_state: State) -> None: ...


@wraps(Fill)
def fill(
    current_state: State, *, location_addresses: tuple[LocationAddress, ...]
) -> State: ...


@wraps(LogicalInitialize)
def logical_initialize(
    current_state: State,
    thetas: tuple[float, ...],
    phis: tuple[float, ...],
    lams: tuple[float, ...],
    *,
    location_addresses: tuple[LocationAddress, ...],
) -> State: ...


@wraps(PhysicalInitialize)
def physical_initialize(
    current_state: State,
    thetas: tuple[float, ...],
    phis: tuple[float, ...],
    lams: tuple[float, ...],
    *,
    location_addresses: tuple[tuple[LocationAddress, ...], ...],
) -> State: ...


@wraps(CZ)
def cz(current_state: State, *, zone_address: ZoneAddress) -> State: ...


@wraps(LocalR)
def local_r(
    current_state: State,
    axis_angle: float,
    rotation_angle: float,
    *,
    location_addresses: tuple[LocationAddress, ...],
) -> State: ...


@wraps(GlobalR)
def global_r(
    current_state: State, axis_angle: float, rotation_angle: float
) -> State: ...


@wraps(LocalRz)
def local_rz(
    current_state: State,
    rotation_angle: float,
    *,
    location_addresses: tuple[LocationAddress, ...],
) -> State: ...


@wraps(GlobalRz)
def global_rz(current_state: State, rotation_angle: float) -> State: ...


@wraps(Move)
def move(current_state: State, *, lanes: tuple[LaneAddress, ...]) -> State: ...


@wraps(EndMeasure)
def end_measure(
    current_state: State, *, zone_addresses: tuple[ZoneAddress, ...]
) -> MeasurementFuture: ...


@wraps(GetFutureResult)
def get_future_result(
    measurement_future: MeasurementFuture,
    *,
    zone_address: ZoneAddress,
    location_address: LocationAddress,
) -> ilist.IList[bloqade_types.MeasurementResult, Any]: ...
