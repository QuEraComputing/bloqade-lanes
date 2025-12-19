from dataclasses import dataclass, field

from kirin import ir
from kirin.analysis import ForwardExtra, ForwardFrame
from kirin.lattice.empty import EmptyLattice
from kirin.worklist import WorkList

from bloqade.lanes.layout.arch import ArchSpec
from bloqade.lanes.layout.path import PathFinder

from .lattice import AtomStateLattice, UnknownAtomState


@dataclass
class AtomFrame(ForwardFrame[EmptyLattice]):
    atom_states: WorkList[AtomStateLattice] = field(default_factory=WorkList)
    atom_state_map: dict[ir.Statement, AtomStateLattice] = field(
        default_factory=dict, init=False
    )
    initial_states: dict[ir.Block, AtomStateLattice] = field(
        default_factory=dict, init=False
    )
    current_state: AtomStateLattice = field(
        default_factory=UnknownAtomState, init=False
    )

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
