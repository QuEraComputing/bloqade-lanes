"""stack_move dialect — 1:1 SSA image of the bytecode."""

from kirin import ir, lowering, types
from kirin.decl import info, statement

from bloqade.lanes.bytecode import LaneAddress, LocationAddress, ZoneAddress
from bloqade.lanes.types import ArrayType  # noqa: F401
from bloqade.lanes.types import MeasurementFutureType

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


# ── Stack manipulation ─────────────────────────────────────────────────


@statement(dialect=dialect)
class Pop(ir.Statement):
    """Pop and discard the top of the virtual stack."""

    traits = frozenset({lowering.FromPythonCall()})
    value: ir.SSAValue = info.argument()


@statement(dialect=dialect)
class Dup(ir.Statement):
    """Duplicate the top of the virtual stack. Semantically result ≡ value;
    preserved as an explicit op to give downstream passes a hook for
    non-cloning invariants."""

    traits = frozenset({lowering.FromPythonCall()})
    value: ir.SSAValue = info.argument()
    result: ir.ResultValue = info.result()


@statement(dialect=dialect)
class Swap(ir.Statement):
    """Swap the top two virtual-stack values. out_top ≡ in_bot; out_bot ≡ in_top."""

    traits = frozenset({lowering.FromPythonCall()})
    in_top: ir.SSAValue = info.argument()
    in_bot: ir.SSAValue = info.argument()
    out_top: ir.ResultValue = info.result()
    out_bot: ir.ResultValue = info.result()


# ── Atom operations ────────────────────────────────────────────────────


@statement(dialect=dialect)
class InitialFill(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})
    locations: tuple[ir.SSAValue, ...] = info.argument(type=LocationAddressType)


@statement(dialect=dialect)
class Fill(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})
    locations: tuple[ir.SSAValue, ...] = info.argument(type=LocationAddressType)


@statement(dialect=dialect)
class Move(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})
    lanes: tuple[ir.SSAValue, ...] = info.argument(type=LaneAddressType)


# ── Gates ──────────────────────────────────────────────────────────────


@statement(dialect=dialect)
class LocalR(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})
    phi: ir.SSAValue = info.argument(type=types.Float)
    theta: ir.SSAValue = info.argument(type=types.Float)
    locations: tuple[ir.SSAValue, ...] = info.argument(type=LocationAddressType)


@statement(dialect=dialect)
class LocalRz(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})
    theta: ir.SSAValue = info.argument(type=types.Float)
    locations: tuple[ir.SSAValue, ...] = info.argument(type=LocationAddressType)


@statement(dialect=dialect)
class GlobalR(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})
    phi: ir.SSAValue = info.argument(type=types.Float)
    theta: ir.SSAValue = info.argument(type=types.Float)


@statement(dialect=dialect)
class GlobalRz(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})
    theta: ir.SSAValue = info.argument(type=types.Float)


@statement(dialect=dialect)
class CZ(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})
    zone: ir.SSAValue = info.argument(type=ZoneAddressType)


# ── Measurement ────────────────────────────────────────────────────────


@statement(dialect=dialect)
class Measure(ir.Statement):
    """Matches bytecode `measure(arity)` — takes location SSA values.
    Zone grouping happens during lower_stack_move."""

    traits = frozenset({lowering.FromPythonCall()})
    locations: tuple[ir.SSAValue, ...] = info.argument(type=LocationAddressType)
    result: ir.ResultValue = info.result(MeasurementFutureType)


@statement(dialect=dialect)
class AwaitMeasure(ir.Statement):
    """Synchronisation — blocks until the most recent measurement completes.

    The bytecode docs state 'block until the most recent measurement
    completes' with no documented stack effect. We treat this as a pure
    synchronisation op: takes a MeasurementFuture, produces no result.
    Extracting per-location measurement values is done via subsequent
    GetItem calls on the future. Confirm against the Rust source before
    implementation — adjust if the actual stack effect differs."""

    traits = frozenset({lowering.FromPythonCall()})
    future: ir.SSAValue = info.argument(type=MeasurementFutureType)


# ── Control flow ───────────────────────────────────────────────────────


@statement(dialect=dialect)
class Return(ir.Statement):
    traits = frozenset({lowering.FromPythonCall(), ir.IsTerminator()})


@statement(dialect=dialect)
class Halt(ir.Statement):
    """Lowered to func.Return(None) alongside Return."""

    traits = frozenset({lowering.FromPythonCall(), ir.IsTerminator()})
