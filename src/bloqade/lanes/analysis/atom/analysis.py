from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from kirin import ir
from kirin.analysis import Forward
from kirin.analysis.forward import ForwardFrame
from typing_extensions import Self

from bloqade.lanes.layout.arch import ArchSpec
from bloqade.lanes.layout.path import PathFinder

from .lattice import AtomState, MoveExecution


def _default_best_state_cost(state: AtomState) -> float:
    """Average of move counts plus standard deviation.

    More weight is added to the standard deviation to prefer a balanced number
    of moves across atoms.
    """
    if len(state.data.collision) > 0:
        return float("inf")

    move_counts = np.array(
        list(
            state.data.move_count.get(qubit, 0)
            for qubit in state.data.qubit_to_locations.keys()
        )
    )
    return 0.1 * np.mean(move_counts).astype(float) + np.std(move_counts).astype(float)


@dataclass
class AtomInterpreter(Forward[MoveExecution]):
    lattice = MoveExecution

    arch_spec: ArchSpec = field(kw_only=True)
    path_finder: PathFinder = field(init=False)
    current_state: MoveExecution = field(init=False)
    best_state_cost: Callable[[AtomState], float] = field(
        kw_only=True, default=_default_best_state_cost
    )
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
