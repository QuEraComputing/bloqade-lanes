import abc
from dataclasses import dataclass, field
from typing import Any, final

from kirin.lattice import (
    BoundedLattice,
    IsSubsetEqMixin,
    SimpleJoinMixin,
    SimpleMeetMixin,
    SingletonMeta,
)
from typing_extensions import Self

from bloqade.lanes import layout

from ._visitor import _ElemVisitor
from .atom_state_data import AtomStateData


@dataclass
class MoveExecution(
    SimpleJoinMixin["MoveExecution"],
    SimpleMeetMixin["MoveExecution"],
    IsSubsetEqMixin["MoveExecution"],
    _ElemVisitor,
    BoundedLattice["MoveExecution"],
):
    @classmethod
    def top(cls) -> "MoveExecution":
        return Unknown()

    @classmethod
    def bottom(cls) -> "MoveExecution":
        return Bottom()

    @abc.abstractmethod
    def copy(self: Self) -> "Self": ...

    def join(self, other: "MoveExecution") -> "MoveExecution":
        method = getattr(self, f"join_{type(other).__name__}", None)
        if method is not None:
            return method(other)
        else:
            return super().join(other)

    def meet(self, other: "MoveExecution") -> "MoveExecution":
        method = getattr(self, f"meet_{type(other).__name__}", None)
        if method is not None:
            return method(other)
        else:
            return super().meet(other)


@final
@dataclass
class Unknown(MoveExecution, metaclass=SingletonMeta):
    def is_structurally_equal(
        self, other: MoveExecution, context: dict | None = None
    ) -> bool:
        return isinstance(other, Unknown)

    def copy(self):
        return self


@final
@dataclass
class Bottom(MoveExecution, metaclass=SingletonMeta):
    def is_structurally_equal(
        self, other: MoveExecution, context: dict | None = None
    ) -> bool:
        return isinstance(other, Bottom)

    def copy(self):
        return self


@final
@dataclass
class Value(MoveExecution):
    value: Any

    def is_subseteq_Value(self, elem: "Value") -> bool:
        return self.value == elem.value

    def is_structurally_equal(
        self, other: MoveExecution, context: dict | None = None
    ) -> bool:
        return isinstance(other, Value) and self.value == other.value

    def copy(self):
        return Value(self.value)


@final
@dataclass
class AtomState(MoveExecution):
    data: AtomStateData = field(default_factory=AtomStateData)

    def is_structurally_equal(
        self, other: MoveExecution, context: dict | None = None
    ) -> bool:
        return self == other

    def is_subseteq_AtomState(self, elem: "AtomState") -> bool:
        return self.data == elem.data

    def copy(self):
        return AtomState(self.data.copy())


@final
@dataclass
class MeasureFuture(MoveExecution):
    results: dict[layout.ZoneAddress, dict[layout.LocationAddress, int]]

    def copy(self):
        return MeasureFuture(self.results.copy())

    def is_subseteq_MeasureFuture(self, elem: "MeasureFuture") -> bool:
        return self.results == elem.results


@final
@dataclass
class MeasureResult(MoveExecution):
    qubit_id: int

    def copy(self):
        return MeasureResult(self.qubit_id)

    def is_subseteq_MeasureResult(self, elem: "MeasureResult") -> bool:
        return self.qubit_id == elem.qubit_id


@final
@dataclass
class IListResult(MoveExecution):
    data: tuple[MoveExecution, ...]

    def copy(self):
        return IListResult(tuple(item.copy() for item in self.data))

    def is_subseteq_IListResult(self, elem: "IListResult") -> bool:
        if len(self.data) != len(elem.data):
            return False
        return all(a.is_subseteq(b) for a, b in zip(self.data, elem.data))

    def join_IListResult(self, other: "IListResult") -> "MoveExecution":
        if len(self.data) != len(other.data):
            return Unknown()

        return IListResult(tuple(a.join(b) for a, b in zip(self.data, other.data)))

    def meet_IListResult(self, other: "IListResult") -> "MoveExecution":
        if len(self.data) != len(other.data):
            return Bottom()

        return IListResult(tuple(a.meet(b) for a, b in zip(self.data, other.data)))
