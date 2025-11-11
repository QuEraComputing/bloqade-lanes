from typing import final
from dataclasses import dataclass

from bloqade.analysis.address.lattice import Address, AddressQubit
from kirin.lattice import (
    SingletonMeta,
    BoundedLattice,
    SimpleJoinMixin,
    SimpleMeetMixin,
)
from bloqade.lanes.layout.encoding import LocationAddress, MoveType

@dataclass
class AtomsState(
    SimpleJoinMixin["AtomsState"],
    SimpleMeetMixin["AtomsState"],
    BoundedLattice["AtomsState"],
):

    @classmethod
    def bottom(cls) -> "AtomsState":
        return NotState()

    @classmethod
    def top(cls) -> "AtomsState":
        return AnyState()


@final
@dataclass
class NotState(AtomsState, metaclass=SingletonMeta):

    def is_subseteq(self, other: AtomsState) -> bool:
        return True


@final
@dataclass
class AnyState(AtomsState, metaclass=SingletonMeta):

    def is_subseteq(self, other: AtomsState) -> bool:
        return isinstance(other, AnyState)


@dataclass(frozen=True)
class LocalID:
    index: int
        

@final
@dataclass
class ConcreteState(AtomsState):
    occupied: frozenset[LocationAddress]
    layout: dict[LocalID, LocationAddress]
    moves: dict[LocalID, list[MoveType]]

    def is_subseteq(self, other: AtomsState) -> bool:
        return self == other

    @property
    def num_qubits(self):
        return len(self.layout)
