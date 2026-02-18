from dataclasses import dataclass, field
from typing import Callable, Sequence, TypeVar, cast

import numpy as np
from kirin import ir
from kirin.analysis import Forward
from kirin.analysis.forward import ForwardFrame
from typing_extensions import Self

from bloqade.lanes.layout.arch import ArchSpec
from bloqade.lanes.layout.path import PathFinder

from ._post_processing import constructor_function
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


RetType = TypeVar("RetType")


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

    def get_post_processing(self, method: ir.Method[..., RetType]):
        _, output = self.run(method)

        func = cast(Callable[[Sequence[bool]], RetType], constructor_function(output))

        if func is None:
            return None

        def post_processing(measurement_results: Sequence[Sequence[bool]]):
            yield from map(func, measurement_results)

        return post_processing
