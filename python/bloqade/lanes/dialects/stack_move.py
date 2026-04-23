"""stack_move dialect — 1:1 SSA image of the bytecode."""

import typing

from bloqade.decoders.dialects.annotate.types import (
    DetectorType,
    MeasurementResultType,
    ObservableType,
)
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

# Type variables for the parameterised ArrayType — used by NewArray (result
# type) and GetItem (array element type flows through to result).
ElemType = types.TypeVar("ElemType")
Dim0Type = types.TypeVar("Dim0Type")
Dim1Type = types.TypeVar("Dim1Type")


# Mapping from bytecode type_tag byte (see
# crates/bloqade-lanes-bytecode-core/src/bytecode/value.rs) to the Kirin
# element type used when building NewArray's parameterised result type.
# For TAG_ARRAY_REF (nested array), we use the parameterised ArrayType
# with three Any slots — element shape is not tracked across nesting.
TYPE_TAG: dict[int, types.TypeAttribute] = {
    0: types.Float,
    1: types.Int,
    2: ArrayType[types.Any, types.Any, types.Any],
    3: LocationAddressType,
    4: LaneAddressType,
    5: ZoneAddressType,
    6: MeasurementFutureType,
    7: DetectorType,
    8: ObservableType,
}


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
    axis_angle: ir.SSAValue = info.argument(type=types.Float)
    rotation_angle: ir.SSAValue = info.argument(type=types.Float)
    locations: tuple[ir.SSAValue, ...] = info.argument(type=LocationAddressType)


@statement(dialect=dialect)
class LocalRz(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})
    rotation_angle: ir.SSAValue = info.argument(type=types.Float)
    locations: tuple[ir.SSAValue, ...] = info.argument(type=LocationAddressType)


@statement(dialect=dialect)
class GlobalR(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})
    axis_angle: ir.SSAValue = info.argument(type=types.Float)
    rotation_angle: ir.SSAValue = info.argument(type=types.Float)


@statement(dialect=dialect)
class GlobalRz(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})
    rotation_angle: ir.SSAValue = info.argument(type=types.Float)


@statement(dialect=dialect)
class CZ(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})
    zone: ir.SSAValue = info.argument(type=ZoneAddressType)


# ── Measurement ────────────────────────────────────────────────────────


@statement(dialect=dialect)
class Measure(ir.Statement):
    """Matches bytecode `measure(arity)` — takes location SSA values.
    Zone grouping happens during stack_move2move."""

    traits = frozenset({lowering.FromPythonCall()})
    locations: tuple[ir.SSAValue, ...] = info.argument(type=LocationAddressType)
    result: ir.ResultValue = info.result(MeasurementFutureType)


@statement(dialect=dialect)
class AwaitMeasure(ir.Statement):
    """Synchronisation — consumes a measurement future (linear) and
    produces an array ref of measurement results.

    Matches the bytecode's `await_measure`: pops a measure future,
    pushes an array ref containing the measurement results. The
    element type is MeasurementResult; shape is unknown until actual
    measurement resolution (hence Any/Any dimensions)."""

    traits = frozenset({lowering.FromPythonCall()})
    future: ir.SSAValue = info.argument(type=MeasurementFutureType)
    result: ir.ResultValue = info.result(
        ArrayType[MeasurementResultType, types.Any, types.Any]
    )


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


@statement(dialect=dialect, init=False)
class NewArray(ir.Statement):
    """Create an array.

    Bytecode stack effect: doesn't pop (empty-array placeholder).  Python
    construction accepts ``values=()`` in that case or a non-empty tuple
    when authoring stack_move IR directly from Python.

    The result type is a fully-parameterised ``ArrayType[ElemType,
    Literal[dim0], Literal[dim1]]`` derived from the ``type_tag`` byte
    (via ``TYPE_TAG``) and the two dimension attributes.
    """

    traits = frozenset({lowering.FromPythonCall()})
    values: tuple[ir.SSAValue, ...]
    type_tag: int = info.attribute()
    dim0: int = info.attribute()
    dim1: int = info.attribute()
    result: ir.ResultValue = info.result(ArrayType[ElemType, Dim0Type, Dim1Type])

    def __init__(
        self,
        values: typing.Sequence[ir.SSAValue],
        type_tag: int,
        dim0: int,
        dim1: int,
    ) -> None:
        elem_type = TYPE_TAG[type_tag]
        result_type = ArrayType[
            elem_type,
            types.Literal(dim0),
            types.Literal(dim1),
        ]
        super().__init__(
            args=tuple(values),
            args_slice={"values": slice(0, len(values))},
            result_types=(result_type,),
            attributes={
                "type_tag": ir.PyAttr(type_tag),
                "dim0": ir.PyAttr(dim0),
                "dim1": ir.PyAttr(dim1),
            },
        )


@statement(dialect=dialect)
class GetItem(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})
    array: ir.SSAValue = info.argument(type=ArrayType[ElemType, types.Any, types.Any])
    indices: tuple[ir.SSAValue, ...] = info.argument(type=types.Int)
    result: ir.ResultValue = info.result(ElemType)


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
