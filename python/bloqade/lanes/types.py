import typing

from kirin import types


class State:
    pass


StateType = types.PyClass(State)


class MeasurementFuture:
    pass


MeasurementFutureType = types.PyClass(MeasurementFuture)


ElemType = typing.TypeVar("ElemType")
Dim0Type = typing.TypeVar("Dim0Type")
Dim1Type = typing.TypeVar("Dim1Type")


class Array(typing.Generic[ElemType, Dim0Type, Dim1Type]):
    pass


ArrayType = types.Generic(
    Array,
    types.TypeVar("ElemType"),
    types.TypeVar("Dim0Type"),
    types.TypeVar("Dim1Type"),
)
