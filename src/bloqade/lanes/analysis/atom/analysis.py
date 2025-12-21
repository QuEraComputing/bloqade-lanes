from dataclasses import dataclass, field

from kirin import ir
from kirin.analysis import Forward
from kirin.analysis.forward import ForwardFrame
from typing_extensions import Self

from bloqade.lanes.layout.arch import ArchSpec
from bloqade.lanes.layout.path import PathFinder

from .lattice import AtomState, MoveExecution


@dataclass
class AtomInterpreter(Forward[MoveExecution]):
    lattice = MoveExecution

    arch_spec: ArchSpec = field(kw_only=True)
    path_finder: PathFinder = field(init=False)
    current_state: MoveExecution = field(init=False)
    keys = ("atom",)

    def __post_init__(self):
        super().__post_init__()
        self.path_finder = PathFinder(self.arch_spec)

    def initialize(self) -> Self:
        self.current_state = AtomState()
        return super().initialize()

    def method_self(self, method) -> MoveExecution:
        return MoveExecution.bottom()

    def eval_fallback(self, frame: ForwardFrame[MoveExecution], node: ir.Statement):
        return tuple(MoveExecution.bottom() for _ in node.results)
