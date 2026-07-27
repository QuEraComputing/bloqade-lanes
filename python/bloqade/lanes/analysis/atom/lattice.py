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

from bloqade.lanes.bytecode.encoding import LocationAddress, ZoneAddress

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
    results: dict[ZoneAddress, dict[LocationAddress, int]]
    measurement_count: int

    def copy(self):
        return MeasureFuture(self.results.copy(), self.measurement_count)

    def is_subseteq_MeasureFuture(self, elem: "MeasureFuture") -> bool:
        return (
            self.results == elem.results
            and self.measurement_count == elem.measurement_count
        )


@final
@dataclass
class MeasureResult(MoveExecution):
    """A single terminal-measurement result.

    Attributes:
        measurement_id: Global measurement-record index, i.e. the order in
            which the physical measurement operation executes (the first
            measurement is ``0``, the second ``1``, ...). This is the column
            index into the raw per-shot measurement array returned by the
            simulator/hardware backend and is what post-processing must use
            to look up the result. It coincides with ``qubit_id`` only when
            qubits happen to be measured once in allocation order.
        qubit_id: Physical qubit that occupied ``location_address`` at
            measurement time. Selects which physical qubit SSA value to
            measure during lowering, but is *not* a valid index into the raw
            measurement array.
        location_address: Hardware location that was measured.
    """

    measurement_id: int
    qubit_id: int
    location_address: LocationAddress

    def copy(self):
        return MeasureResult(self.measurement_id, self.qubit_id, self.location_address)

    def is_subseteq_MeasureResult(self, elem: "MeasureResult") -> bool:
        return (
            self.measurement_id == elem.measurement_id
            and self.qubit_id == elem.qubit_id
            and self.location_address == elem.location_address
        )


@final
@dataclass
class DetectorResult(MoveExecution):
    data: MoveExecution

    def copy(self):
        return DetectorResult(self.data.copy())

    def is_subseteq_DetectorResult(self, elem: "DetectorResult") -> bool:
        return self.data.is_subseteq(elem.data)

    def join_DetectorResult(self, other: "DetectorResult") -> "MoveExecution":
        return DetectorResult(self.data.join(other.data))

    def meet_DetectorResult(self, other: "DetectorResult") -> "MoveExecution":
        return DetectorResult(self.data.meet(other.data))


@final
@dataclass
class ObservableResult(MoveExecution):
    data: MoveExecution

    def copy(self):
        return ObservableResult(self.data.copy())

    def is_subseteq_ObservableResult(self, elem: "ObservableResult") -> bool:
        return self.data.is_subseteq(elem.data)

    def join_ObservableResult(self, other: "ObservableResult") -> "MoveExecution":
        return ObservableResult(self.data.join(other.data))

    def meet_ObservableResult(self, other: "ObservableResult") -> "MoveExecution":
        return ObservableResult(self.data.meet(other.data))


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


@dataclass
class TupleResult(MoveExecution):
    data: tuple[MoveExecution, ...]

    def copy(self):
        return TupleResult(tuple(item.copy() for item in self.data))

    def is_subseteq_TupleResult(self, elem: "TupleResult") -> bool:
        if len(self.data) != len(elem.data):
            return False
        return all(a.is_subseteq(b) for a, b in zip(self.data, elem.data))

    def join_TupleResult(self, other: "TupleResult") -> "MoveExecution":
        if len(self.data) != len(other.data):
            return Unknown()

        return TupleResult(tuple(a.join(b) for a, b in zip(self.data, other.data)))

    def meet_TupleResult(self, other: "TupleResult") -> "MoveExecution":
        if len(self.data) != len(other.data):
            return Bottom()

        return TupleResult(tuple(a.meet(b) for a, b in zip(self.data, other.data)))
