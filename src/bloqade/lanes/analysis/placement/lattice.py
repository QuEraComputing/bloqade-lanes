from dataclasses import dataclass
from typing import final

from kirin.lattice import (
    BoundedLattice,
    SimpleJoinMixin,
    SimpleMeetMixin,
    SingletonMeta,
)

from bloqade.lanes.layout import LocationAddress, ZoneAddress


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


@final
@dataclass
class ExecuteCZ(ConcreteState):
    """Defines the state representing the placement of
    atoms before/after executing CZ gate pulse.

    NOTE: you can specify multiple entnangling zones to be active
    in a single ExecuteCZ state in cases where there are multiple entangling
    zones that can be used in parallel.

    """

    active_cz_zones: frozenset[ZoneAddress]
    """The set of CZ zones that need to execute for this round of CZ gates."""

    @classmethod
    def from_concrete_state(
        cls, state: ConcreteState, active_cz_zones: frozenset[ZoneAddress]
    ) -> "ExecuteCZ":
        return cls(
            occupied=state.occupied,
            layout=state.layout,
            move_count=state.move_count,
            active_cz_zones=active_cz_zones,
        )

    def is_subseteq(self, other: AtomState) -> bool:
        return (
            super().is_subseteq(other)
            and isinstance(other, ExecuteCZ)
            and self.active_cz_zones == other.active_cz_zones
        )


@final
@dataclass
class ExecuteMeasure(ConcreteState):
    """A state representing measurement placements.

    NOTE: Depending on the placement of the atoms you may need to specify
    which atoms are measured by which zone. This is done via the zone_maps field, such that
    `zone_maps[i]` gives the zone that measures the ith qubit.

    """

    zone_maps: tuple[ZoneAddress, ...]

    @classmethod
    def from_concrete_state(
        cls, state: ConcreteState, zone_maps: tuple[ZoneAddress, ...]
    ) -> "ExecuteMeasure":
        return cls(
            occupied=state.occupied,
            layout=state.layout,
            move_count=state.move_count,
            zone_maps=zone_maps,
        )

    def is_subseteq(self, other: AtomState) -> bool:
        return (
            super().is_subseteq(other)
            and isinstance(other, ExecuteMeasure)
            and self.zone_maps == other.zone_maps
        )
