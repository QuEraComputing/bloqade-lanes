"""stack_move dialect — 1:1 SSA image of the bytecode."""

from kirin import ir, lowering, types
from kirin.decl import info, statement

from bloqade.lanes.bytecode import LaneAddress, LocationAddress, ZoneAddress
from bloqade.lanes.types import ArrayType, MeasurementFutureType  # noqa: F401

dialect = ir.Dialect(name="lanes.stack_move")


# ── SSA types ──────────────────────────────────────────────────────────

LocationAddressType = types.PyClass(LocationAddress)
LaneAddressType = types.PyClass(LaneAddress)
ZoneAddressType = types.PyClass(ZoneAddress)
# ArrayType and MeasurementFutureType come from bloqade.lanes.types.


# ── Constants ──────────────────────────────────────────────────────────


@statement(dialect=dialect)
class ConstFloat(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})
    value: float = info.attribute()
    result: ir.ResultValue = info.result(types.Float)


@statement(dialect=dialect)
class ConstInt(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})
    value: int = info.attribute()
    result: ir.ResultValue = info.result(types.Int)


@statement(dialect=dialect)
class ConstLoc(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})
    value: LocationAddress = info.attribute()
    result: ir.ResultValue = info.result(LocationAddressType)


@statement(dialect=dialect)
class ConstLane(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})
    value: LaneAddress = info.attribute()
    result: ir.ResultValue = info.result(LaneAddressType)


@statement(dialect=dialect)
class ConstZone(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})
    value: ZoneAddress = info.attribute()
    result: ir.ResultValue = info.result(ZoneAddressType)
