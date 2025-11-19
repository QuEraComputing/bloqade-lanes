from dataclasses import dataclass, field, replace

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
    current_state: AtomStateType = field(default_factory=UnknownAtomState)


@dataclass
class AtomInterpreter(ForwardExtra[AtomFrame, EmptyLattice]):
    lattice = EmptyLattice

    arch_spec: ArchSpec = field(kw_only=True)
    keys = ("atom",)

    def method_self(self, method) -> EmptyLattice:
        return EmptyLattice.bottom()

    def initialize_frame(
        self, node: ir.Statement, *, has_parent_access: bool = False
    ) -> AtomFrame:
        return AtomFrame(node, has_parent_access=has_parent_access)

    def eval_fallback(self, frame: AtomFrame, node: ir.Statement):
        return tuple(EmptyLattice.bottom() for _ in node.results)
