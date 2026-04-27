from dataclasses import dataclass, field
from typing import Callable, Generator, Generic, Sequence, TypeVar, cast

import numpy as np
from kirin import ir
from kirin.analysis import Forward
from kirin.analysis.forward import ForwardFrame
from typing_extensions import Self

from bloqade.lanes.layout.arch import ArchSpec
from bloqade.lanes.utils import no_none_elements_tuple

from . import _shot_remapping
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
    current_state: MoveExecution = field(init=False)
    best_state_cost: Callable[[AtomState], float] = field(
        kw_only=True, default=_default_best_state_cost
    )
    _detectors: list[MoveExecution] = field(init=False, default_factory=list)
    _observables: list[MoveExecution] = field(init=False, default_factory=list)
    measure_sites: list[dict] = field(init=False, default_factory=list)
    final_measurement_count: int = field(init=False, default=0)
    keys = ("atom",)

    def __post_init__(self):
        super().__post_init__()

    def initialize(self) -> Self:
        self.current_state = AtomState()
        self._detectors.clear()
        self._observables.clear()
        self.measure_sites.clear()
        self.final_measurement_count = 0
        return super().initialize()

    def method_self(self, method) -> MoveExecution:
        return MoveExecution.bottom()

    def eval_fallback(self, frame: ForwardFrame[MoveExecution], node: ir.Statement):
        return tuple(MoveExecution.bottom() for _ in node.results)

    def get_shot_remapping(
        self, method: ir.Method, *, no_raise: bool = True
    ) -> _shot_remapping.ShotRemappingOk | _shot_remapping.ShotRemappingErr:
        """Run the analysis on ``method`` and return the flat Zone-0
        bitstring index list (in row-major order over the nested
        ``IListResult[IListResult[MeasureResult]]`` return shape) as a
        ``ShotRemappingOk``. On failure, returns ``ShotRemappingErr``
        carrying a ``ShotRemappingDiagnostic``.

        Convenience wrapper around the standalone
        ``bloqade.lanes.analysis.atom._shot_remapping.get_shot_remapping``;
        see that function's docstring for the contract on the analysis
        output shape, the meaning of the returned indices, and the
        diagnostic emitted on failure.

        ``method``'s return value is expected to refine to
        ``IListResult[IListResult[MeasureResult]]`` — the shape produced
        by lowering a logical ``terminal_measure`` (or any kernel that
        returns a nested ilist of measurement results) through the
        atom-analysis chain. Callers (typically the compiler service)
        are responsible for surfacing the diagnostic in the failure
        case; a failure here is a compiler-pipeline regression, not a
        user error.

        Args:
            method: kirin method to analyse.
            no_raise: when ``True`` (default), an analysis crash is
                caught by ``Forward.run_no_raise`` and falls through
                into the standard ``ShotRemappingErr`` path with the
                ``Bottom`` lattice as the offending value, so callers
                see a single failure shape. Flip to ``False`` when
                debugging an analysis-side bug to let the original
                exception propagate.
        """
        run_method = self.run_no_raise if no_raise else self.run
        _, output = run_method(method)
        return _shot_remapping.get_shot_remapping(output, self.arch_spec)

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
