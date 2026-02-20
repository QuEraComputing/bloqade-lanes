from dataclasses import dataclass, field
from typing import Callable, Generator, Generic, Sequence, TypeVar, cast

import numpy as np
from kirin import ir
from kirin.analysis import Forward
from kirin.analysis.forward import ForwardFrame
from typing_extensions import Self

from bloqade.lanes.layout.arch import ArchSpec
from bloqade.lanes.layout.path import PathFinder
from bloqade.lanes.utils import no_none_elements_tuple

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
class PostProcessing(Generic[RetType]):
    emit_return: Callable[[Sequence[Sequence[bool]]], Generator[RetType, None, None]]
    emit_detectors: Callable[
        [Sequence[Sequence[bool]]], Generator[list[bool], None, None]
    ]
    emit_observables: Callable[
        [Sequence[Sequence[bool]]], Generator[list[bool], None, None]
    ]


@dataclass
class AtomInterpreter(Forward[MoveExecution]):
    lattice = MoveExecution

    arch_spec: ArchSpec = field(kw_only=True)
    path_finder: PathFinder = field(init=False)
    current_state: MoveExecution = field(init=False)
    best_state_cost: Callable[[AtomState], float] = field(
        kw_only=True, default=_default_best_state_cost
    )
    _detectors: list[MoveExecution] = field(init=False, default_factory=list)
    _observables: list[MoveExecution] = field(init=False, default_factory=list)
    keys = ("atom",)

    def __post_init__(self):
        super().__post_init__()
        self.path_finder = PathFinder(self.arch_spec)

    def initialize(self) -> Self:
        self.current_state = AtomState()
        self._detectors.clear()
        self._observables.clear()
        return super().initialize()

    def method_self(self, method) -> MoveExecution:
        return MoveExecution.bottom()

    def eval_fallback(self, frame: ForwardFrame[MoveExecution], node: ir.Statement):
        return tuple(MoveExecution.bottom() for _ in node.results)

    def get_post_processing(
        self, method: ir.Method[..., RetType]
    ) -> PostProcessing[RetType]:
        _, output = self.run(method)

        func = cast(Callable[[Sequence[bool]], RetType], constructor_function(output))

        if func is None:
            raise ValueError("Unable to infer return result value from method output")

        def post_processing_return(measurement_results: Sequence[Sequence[bool]]):
            yield from map(func, measurement_results)

        detector_funcs: tuple[Callable[[Sequence[bool]], bool] | None, ...] = tuple(
            map(constructor_function, self._detectors)
        )
        if not no_none_elements_tuple(detector_funcs):
            raise ValueError("Unable to infer detector measurement values")

        def detectors(measurement_results: Sequence[Sequence[bool]]):
            yield from (
                list(func(measurement_shot) for func in detector_funcs)
                for measurement_shot in measurement_results
            )

        observable_funcs: tuple[Callable[[Sequence[bool]], bool] | None, ...] = tuple(
            map(constructor_function, self._observables)
        )
        if not no_none_elements_tuple(observable_funcs):
            raise ValueError("Unable to infer observable measurement values")

        def observables(measurement_results: Sequence[Sequence[bool]]):
            yield from (
                list(func(measurement_shot) for func in observable_funcs)
                for measurement_shot in measurement_results
            )

        return PostProcessing(post_processing_return, detectors, observables)
