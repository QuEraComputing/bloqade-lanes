from dataclasses import dataclass, field
from typing import Any, final

from kirin.interp import InterpreterError
from kirin.lattice import (
    BoundedLattice,
    SimpleJoinMixin,
    SimpleMeetMixin,
    SingletonMeta,
)

from bloqade.lanes.layout import LaneAddress, LocationAddress, ZoneAddress
from bloqade.lanes.layout.arch import ArchSpec


@dataclass
class MoveExecution(
    SimpleJoinMixin["MoveExecution"],
    SimpleMeetMixin["MoveExecution"],
    BoundedLattice["MoveExecution"],
):
    @classmethod
    def top(cls) -> "MoveExecution":
        return Unknown()

    @classmethod
    def bottom(cls) -> "MoveExecution":
        return Bottom()


@final
@dataclass
class Unknown(MoveExecution, metaclass=SingletonMeta):

    def is_subseteq(self, other: MoveExecution) -> bool:
        return True


@final
@dataclass
class Bottom(MoveExecution, metaclass=SingletonMeta):
    def is_subseteq(self, other: MoveExecution) -> bool:
        return isinstance(other, Bottom)


@final
@dataclass
class Value(MoveExecution):
    value: Any

    def is_subseteq(self, other: MoveExecution) -> bool:
        return isinstance(other, Value) and self.value == other.value


@final
@dataclass
class AtomState(MoveExecution):
    locations_to_qubit: dict[LocationAddress, int] = field(
        repr=False, default_factory=dict
    )
    qubit_to_locations: dict[int, LocationAddress] = field(default_factory=dict)
    prev_lanes: dict[int, LaneAddress] = field(default_factory=dict)

    def is_subseteq(self, other: MoveExecution) -> bool:
        return (
            isinstance(other, AtomState)
            and self.locations_to_qubit == other.locations_to_qubit
            and self.prev_lanes == other.prev_lanes
        )

    def add_atoms(self, *locations: LocationAddress):
        if not self.locations_to_qubit.keys().isdisjoint(locations):
            raise InterpreterError("Attempted to add atom to occupied location")

        qubit_to_locations = self.qubit_to_locations.copy()
        locations_to_qubit = self.locations_to_qubit.copy()

        for current_qubit, location in enumerate(
            locations, len(self.locations_to_qubit)
        ):
            qubit_to_locations[current_qubit] = location
            locations_to_qubit[location] = current_qubit

        return AtomState(
            locations_to_qubit=locations_to_qubit,
            qubit_to_locations=qubit_to_locations,
        )

    def update(
        self,
        updates: dict[int, LocationAddress],
        prev_lanes: dict[int, LaneAddress] | None = None,
    ):
        if prev_lanes is None:
            prev_lanes = {}

        locations_to_qubit = self.locations_to_qubit.copy()
        qubit_to_location = self.qubit_to_locations.copy()
        for qubit, new_location in updates.items():
            old_location = self.qubit_to_locations.pop(qubit)
            locations_to_qubit.pop(old_location)
            qubit_to_location[qubit] = new_location
            locations_to_qubit[new_location] = qubit

        return AtomState(
            locations_to_qubit=locations_to_qubit,
            qubit_to_locations=qubit_to_location,
            prev_lanes=prev_lanes,
        )

    def get_qubit(self, location: LocationAddress):
        return self.locations_to_qubit.get(location)

    def get_qubit_pairing(self, zone_address: ZoneAddress, arch_spec: ArchSpec):

        controls: list[int] = []
        targets: list[int] = []
        unpaired: list[int] = []
        visited: set[int] = set()
        word_ids = arch_spec.zones[zone_address.zone_id]

        for qubit_index, address in self.qubit_to_locations.items():
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


@final
@dataclass
class MeasureFuture(MoveExecution):
    current_state: AtomState

    def is_subseteq(self, other: MoveExecution) -> bool:
        return isinstance(other, MeasureFuture) and self.current_state.is_subseteq(
            other.current_state
        )


@final
@dataclass
class MeasureResult(MoveExecution):
    qubit_id: int

    def is_subseteq(self, other: "MoveExecution") -> bool:
        return isinstance(other, MeasureResult) and self.qubit_id == other.qubit_id


@final
@dataclass
class IListResult(MoveExecution):
    data: tuple[MoveExecution, ...]

    def is_subseteq(self, other: MoveExecution) -> bool:
        return (
            isinstance(other, IListResult)
            and len(self.data) == len(other.data)
            and all(s.is_subseteq(o) for s, o in zip(self.data, other.data))
        )
