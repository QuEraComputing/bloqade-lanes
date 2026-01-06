from dataclasses import dataclass, field
from functools import cached_property

from kirin.interp import InterpreterError

from bloqade.lanes.layout import LaneAddress, LocationAddress, ZoneAddress
from bloqade.lanes.layout.arch import ArchSpec
from bloqade.lanes.layout.path import PathFinder


@dataclass(frozen=True)
class AtomStateData:
    locations_to_qubit: dict[LocationAddress, int] = field(default_factory=dict)
    """Mapping from location to qubit id."""
    qubit_to_locations: dict[int, LocationAddress] = field(default_factory=dict)
    """Mapping from qubit id to its current location."""
    collision: dict[int, int] = field(default_factory=dict)
    """Mapping from qubit id to another qubit id that it has collided with in this state."""
    prev_lanes: dict[int, LaneAddress] = field(default_factory=dict)
    """Mapping from qubit id to the lane it took to reach this state."""
    move_count: dict[int, int] = field(default_factory=dict)
    """Mapping from qubit id to number of moves it has had."""

    @classmethod
    def new(cls, locations: dict[int, LocationAddress] | list[LocationAddress]):
        if isinstance(locations, list):
            locations = {i: loc for i, loc in enumerate(locations)}

        qubit_to_locations = {}
        locations_to_qubit = {}

        for qubit, location in locations.items():
            qubit_to_locations[qubit] = location
            locations_to_qubit[location] = qubit

        return cls(
            locations_to_qubit=locations_to_qubit,
            qubit_to_locations=qubit_to_locations,
        )

    @cached_property
    def _hash(self):
        return hash(
            (
                AtomStateData,
                frozenset(self.locations_to_qubit.items()),
                frozenset(self.qubit_to_locations.items()),
                frozenset(self.collision.items()),
                frozenset(self.prev_lanes.items()),
                frozenset(self.move_count.items()),
            )
        )

    def __hash__(self):
        return self._hash

    def add_atoms(self, locations: dict[int, LocationAddress]):
        if not self.qubit_to_locations.keys().isdisjoint(locations.keys()):
            raise InterpreterError("Attempted to add atom that already exists")

        if not self.locations_to_qubit.keys().isdisjoint(locations.values()):
            raise InterpreterError("Attempted to add atom to occupied location")

        qubit_to_locations = self.qubit_to_locations.copy()
        locations_to_qubit = self.locations_to_qubit.copy()

        for current_qubit, location in locations.items():
            qubit_to_locations[current_qubit] = location
            locations_to_qubit[location] = current_qubit

        return AtomStateData(
            locations_to_qubit=locations_to_qubit,
            qubit_to_locations=qubit_to_locations,
        )

    def apply_moves(
        self,
        lanes: tuple[LaneAddress, ...],
        path_finder: PathFinder,
    ):
        qubit_to_locations = self.qubit_to_locations.copy()
        locations_to_qubit = self.locations_to_qubit.copy()
        collisions = self.collision.copy()
        moves_count = self.move_count.copy()
        prev_lanes: dict[int, LaneAddress] = {}

        for lane in lanes:
            src, dst = path_finder.get_endpoints(lane)
            if src is None or dst is None:
                return None

            qubit = locations_to_qubit.pop(src, None)

            if qubit is None:
                continue

            moves_count[qubit] = moves_count.get(qubit, 0) + 1
            prev_lanes[qubit] = lane

            if (other_qubit := locations_to_qubit.pop(dst, None)) is None:
                qubit_to_locations[qubit] = dst
                locations_to_qubit[dst] = qubit
            else:
                del qubit_to_locations[qubit]
                del qubit_to_locations[other_qubit]
                collisions[qubit] = other_qubit

        return AtomStateData(
            locations_to_qubit=locations_to_qubit,
            qubit_to_locations=qubit_to_locations,
            prev_lanes=prev_lanes,
            collision=collisions,
            move_count=moves_count,
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

    def copy(self):
        return AtomStateData(
            locations_to_qubit=self.locations_to_qubit.copy(),
            qubit_to_locations=self.qubit_to_locations.copy(),
            collision=self.collision.copy(),
            prev_lanes=self.prev_lanes.copy(),
            move_count=self.move_count.copy(),
        )
