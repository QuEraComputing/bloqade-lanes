from dataclasses import dataclass, field, replace

from kirin import ir
from kirin.analysis import ForwardExtra, ForwardFrame
from kirin.lattice.empty import EmptyLattice
from kirin.worklist import WorkList
from matplotlib.axes import Axes

from bloqade.lanes.layout import LaneAddress, LocationAddress, ZoneAddress
from bloqade.lanes.layout.arch import ArchSpec
from bloqade.lanes.layout.path import PathFinder


class AtomStateType:
    pass


@dataclass(frozen=True)
class AtomState(AtomStateType):
    locations: tuple[LocationAddress, ...]
    prev_lanes: dict[int, LaneAddress] = field(default_factory=dict)

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

    def draw_atoms(self, arch_spec: ArchSpec, ax: Axes | None = None, **kwargs):
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        for location in self.locations:
            x_pos, y_pos = zip(
                *arch_spec.words[location.word_id].site_positions(location.site_id)
            )
            ax.scatter(x_pos, y_pos, **kwargs)

    def draw_moves(self, arch_spec: ArchSpec, ax: Axes | None = None, **kwargs):
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        for lane in self.prev_lanes.values():
            start, end = arch_spec.get_endpoints(lane)
            start_pos = arch_spec.words[start.word_id].site_positions(start.site_id)
            end_pos = arch_spec.words[end.word_id].site_positions(end.site_id)
            for (x_start, y_start), (x_end, y_end) in zip(start_pos, end_pos):
                ax.quiver(
                    [x_start],
                    [y_start],
                    [x_end - x_start],
                    [y_end - y_start],
                    angles="xy",
                    scale_units="xy",
                    scale=1.0,
                    **kwargs,
                )


@dataclass(frozen=True)
class UnknownAtomState(AtomStateType):
    pass


@dataclass
class AtomFrame(ForwardFrame[EmptyLattice]):
    atom_states: WorkList[AtomStateType] = field(default_factory=WorkList)
    atom_state_map: dict[ir.Statement, AtomStateType] = field(
        default_factory=dict, init=False
    )
    initial_states: dict[ir.Block, AtomStateType] = field(
        default_factory=dict, init=False
    )
    current_state: AtomStateType = field(default_factory=UnknownAtomState, init=False)

    def set_state_for_stmt(self, stmt: ir.Statement):
        if stmt in self.atom_state_map:
            self.atom_state_map[stmt] = UnknownAtomState()
        else:
            self.atom_state_map[stmt] = self.current_state


@dataclass
class AtomInterpreter(ForwardExtra[AtomFrame, EmptyLattice]):
    lattice = EmptyLattice

    arch_spec: ArchSpec = field(kw_only=True)
    path_finder: PathFinder = field(init=False)
    keys = ("atom",)

    def __post_init__(self):
        super().__post_init__()
        self.path_finder = PathFinder(self.arch_spec)

    def method_self(self, method) -> EmptyLattice:
        return EmptyLattice.bottom()

    def initialize_frame(
        self, node: ir.Statement, *, has_parent_access: bool = False
    ) -> AtomFrame:
        return AtomFrame(node, has_parent_access=has_parent_access)

    def eval_fallback(self, frame: AtomFrame, node: ir.Statement):
        return tuple(EmptyLattice.bottom() for _ in node.results)
