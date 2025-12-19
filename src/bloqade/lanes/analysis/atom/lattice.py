from dataclasses import dataclass, field, replace
from typing import final

from kirin.lattice import (
    BoundedLattice,
    SimpleJoinMixin,
    SimpleMeetMixin,
    SingletonMeta,
)

from bloqade.lanes.layout import LaneAddress, LocationAddress, ZoneAddress
from bloqade.lanes.layout.arch import ArchSpec


@dataclass
class AtomStateLattice(
    SimpleJoinMixin["AtomStateLattice"],
    SimpleMeetMixin["AtomStateLattice"],
    BoundedLattice["AtomStateLattice"],
):
    @classmethod
    def top(cls) -> "AtomStateLattice":
        return UnknownAtomState()

    @classmethod
    def bottom(cls) -> "AtomStateLattice":
        return InvalidAtomState()


@final
@dataclass
class UnknownAtomState(AtomStateLattice, metaclass=SingletonMeta):

    def is_subseteq(self, other: AtomStateLattice) -> bool:
        return True


@final
@dataclass
class InvalidAtomState(AtomStateLattice, metaclass=SingletonMeta):

    def is_subseteq(self, other: AtomStateLattice) -> bool:
        return isinstance(other, InvalidAtomState)


@final
@dataclass
class ConcreteState(AtomStateLattice):
    locations: tuple[LocationAddress, ...]
    prev_lanes: dict[int, LaneAddress] = field(default_factory=dict)

    def is_subseteq(self, other: AtomStateLattice) -> bool:
        return (
            isinstance(other, ConcreteState)
            and self.locations == other.locations
            and self.prev_lanes == other.prev_lanes
        )

    def update(
        self,
        qubits: dict[int, LocationAddress],
        prev_lanes: dict[int, LaneAddress] | None = None,
    ):
        if prev_lanes is None:
            prev_lanes = {}

        new_locations = tuple(
            qubits.get(i, original_location)
            for i, original_location in enumerate(self.locations)
        )
        return replace(self, locations=new_locations, prev_lanes=prev_lanes)

    def get_qubit(self, location: LocationAddress):
        if location in self.locations:
            return self.locations.index(location)

    def get_qubit_pairing(self, zone_address: ZoneAddress, arch_spec: ArchSpec):

        controls: list[int] = []
        targets: list[int] = []
        unpaired: list[int] = []
        visited: set[int] = set()
        word_ids = arch_spec.zones[zone_address.zone_id]

        for qubit_index, address in enumerate(self.locations):
            if qubit_index in visited:
                continue

            visited.add(qubit_index)
            if (word_id := address.word_id) not in word_ids:
                continue

            site = arch_spec.words[word_id][address.site_id]

            if site.cz_pair is None:
                unpaired.append(qubit_index)
                continue

            target_site = site.cz_pair
            target_id = self.get_qubit(LocationAddress(word_id, target_site))
            if target_id is None:
                unpaired.append(qubit_index)
                continue

            controls.append(qubit_index)
            targets.append(target_id)
            visited.add(target_id)

        return controls, targets, unpaired
