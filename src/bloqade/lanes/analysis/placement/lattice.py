from dataclasses import dataclass
from typing import final

from kirin.lattice import (
    BoundedLattice,
    SimpleJoinMixin,
    SimpleMeetMixin,
    SingletonMeta,
)

from bloqade.lanes.layout.encoding import LocationAddress


@dataclass
class AtomState(
    SimpleJoinMixin["AtomState"],
    SimpleMeetMixin["AtomState"],
    BoundedLattice["AtomState"],
):

    @classmethod
    def bottom(cls) -> "AtomState":
        return NotState()

    @classmethod
    def top(cls) -> "AtomState":
        return AnyState()


@final
@dataclass
class NotState(AtomState, metaclass=SingletonMeta):

    def is_subseteq(self, other: AtomState) -> bool:
        return True


@final
@dataclass
class AnyState(AtomState, metaclass=SingletonMeta):

    def is_subseteq(self, other: AtomState) -> bool:
        return isinstance(other, AnyState)


@final
@dataclass
class ConcreteState(AtomState):
    occupied: frozenset[LocationAddress]
    """Stores the set of occupied locations with atoms not participating in this static circuit."""
    layout: tuple[LocationAddress, ...]
    """Stores the current location of the ith qubit argument in layout[i]."""
    move_count: tuple[int, ...]
    """Stores the number of moves each atom has undergone."""

    def __post_init__(self):
        assert self.occupied.isdisjoint(
            self.layout
        ), "layout can't containe occupied location addresses"
        assert len(set(self.layout)) == len(
            self.layout
        ), "Atoms can't occupy the same location"

    def is_subseteq(self, other: AtomState) -> bool:
        return (
            isinstance(other, ConcreteState)
            and self.occupied == other.occupied
            and self.layout == other.layout
        )

    def get_qubit_id(self, location: LocationAddress) -> int | None:
        try:
            return self.layout.index(location)
        except ValueError:
            return None
