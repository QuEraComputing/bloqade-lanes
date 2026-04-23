"""stack_move dialect — 1:1 SSA image of the bytecode."""

from bloqade.decoders.dialects.annotate.types import DetectorType, ObservableType
from kirin import ir, lowering, types
from kirin.decl import info, statement

from bloqade.lanes.layout.encoding import LaneAddress, LocationAddress, ZoneAddress
from bloqade.lanes.types import ArrayType, MeasurementFutureType

dialect = ir.Dialect(name="lanes.stack_move")


# ── SSA types ──────────────────────────────────────────────────────────

LocationAddressType = types.PyClass(LocationAddress)
LaneAddressType = types.PyClass(LaneAddress)
ZoneAddressType = types.PyClass(ZoneAddress)
# ArrayType and MeasurementFutureType come from bloqade.lanes.types.

# Type variables for stack-manipulation invariants:
#   Dup preserves the top-of-stack type (T → T).
#   Swap permutes the top two types ((TopType, BottomType) → (BottomType, TopType)).
T = types.TypeVar("T")
TopType = types.TypeVar("TopType")
BottomType = types.TypeVar("BottomType")


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
    value: ir.SSAValue = info.argument(types.Any)


@statement(dialect=dialect)
class Dup(ir.Statement):
    """Duplicate the top of the virtual stack. Semantically result ≡ value;
    preserved as an explicit op to give downstream passes a hook for
    non-cloning invariants."""

    traits = frozenset({lowering.FromPythonCall()})
    value: ir.SSAValue = info.argument(T)
    result: ir.ResultValue = info.result(T)


@statement(dialect=dialect)
class Swap(ir.Statement):
    """Swap the top two virtual-stack values. out_top ≡ in_bot; out_bot ≡ in_top."""

    traits = frozenset({lowering.FromPythonCall()})
    in_top: ir.SSAValue = info.argument(TopType)
    in_bot: ir.SSAValue = info.argument(BottomType)
    out_top: ir.ResultValue = info.result(BottomType)
    out_bot: ir.ResultValue = info.result(TopType)


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
    """Return the top-of-stack value from the current program."""

    traits = frozenset({lowering.FromPythonCall(), ir.IsTerminator()})
    value: ir.SSAValue = info.argument()


@statement(dialect=dialect)
class Halt(ir.Statement):
    """Lowered to func.Return(None) alongside Return."""

    traits = frozenset({lowering.FromPythonCall(), ir.IsTerminator()})


# ── Arrays ─────────────────────────────────────────────────────────────


@statement(dialect=dialect)
class NewArray(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})
    type_tag: int = info.attribute()
    dim0: int = info.attribute()
    dim1: int = info.attribute()
    result: ir.ResultValue = info.result(ArrayType)


@statement(dialect=dialect)
class GetItem(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})
    array: ir.SSAValue = info.argument(type=ArrayType)
    indices: tuple[ir.SSAValue, ...] = info.argument(type=types.Int)
    result: ir.ResultValue = info.result()  # element type is context-dependent


# ── Annotations (detectors / observables) ──────────────────────────────


@statement(dialect=dialect)
class SetDetector(ir.Statement):
    """Build a detector record from the top-of-stack array. Matches
    annotate.SetDetector's signature — produces a Detector."""

    traits = frozenset({lowering.FromPythonCall()})
    array: ir.SSAValue = info.argument(type=ArrayType)
    result: ir.ResultValue = info.result(DetectorType)


@statement(dialect=dialect)
class SetObservable(ir.Statement):
    """Build an observable record from the top-of-stack array. Matches
    annotate.SetObservable's signature — produces an Observable."""

    traits = frozenset({lowering.FromPythonCall()})
    array: ir.SSAValue = info.argument(type=ArrayType)
    result: ir.ResultValue = info.result(ObservableType)
