"""stack_move dialect — 1:1 SSA image of the bytecode."""

from kirin import ir, lowering, types  # noqa: F401
from kirin.decl import info, statement  # noqa: F401

from bloqade.lanes.bytecode import LaneAddress, LocationAddress, ZoneAddress
from bloqade.lanes.types import ArrayType, MeasurementFutureType  # noqa: F401

dialect = ir.Dialect(name="lanes.stack_move")


# ── SSA types ──────────────────────────────────────────────────────────

LocationAddressType = types.PyClass(LocationAddress)
LaneAddressType = types.PyClass(LaneAddress)
ZoneAddressType = types.PyClass(ZoneAddress)
# ArrayType and MeasurementFutureType come from bloqade.lanes.types.
