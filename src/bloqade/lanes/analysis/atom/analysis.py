from dataclasses import dataclass, field, replace

from bloqade.lanes.layout.path import PathFinder
from kirin import ir
from kirin.analysis import ForwardExtra, ForwardFrame
from kirin.interp.value import Successor
from kirin.lattice.empty import EmptyLattice
from kirin.worklist import WorkList

from bloqade.lanes.layout.arch import ArchSpec
from bloqade.lanes.layout.encoding import LocationAddress


class AtomStateType:
    pass


@dataclass(frozen=True)
class AtomState(AtomStateType):
    locations: tuple[LocationAddress, ...]

    def update(self, qubits: dict[int, LocationAddress]):
        new_locations = tuple(
            qubits.get(i, original_location)
            for i, original_location in enumerate(self.locations)
        )
        return replace(self, locations=new_locations)

    def get_qubit(self, location: LocationAddress):
        if location in self.locations:
            return self.locations.index(location)


@dataclass(frozen=True)
class UnknownAtomState(AtomStateType):
    pass


@dataclass
class AtomFrame(ForwardFrame[EmptyLattice]):
    atom_states: WorkList[AtomStateType] = field(default_factory=WorkList)
    atom_visited: dict[ir.Block, set[tuple[AtomStateType, Successor[EmptyLattice]]]] = (
        field(default_factory=dict)
    )
    prev_states: list[AtomStateType] = field(default_factory=list)

    @property
    def current_state(self) -> AtomStateType:
        if len(self.prev_states) > 0:
            return self.prev_states[-1]
        return UnknownAtomState()

    @current_state.setter
    def current_state(self, value: AtomStateType) -> None:
        if self.current_state != value:
            self.prev_states.append(value)


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
