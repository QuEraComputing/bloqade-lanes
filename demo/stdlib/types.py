from __future__ import annotations

from typing import Any, Literal, Protocol, TypeAlias, TypedDict, overload

import stim
import tsim as tsim_backend
from kirin import ir

KirinKernel: TypeAlias = ir.Method[..., Any]
SquinKernel: TypeAlias = KirinKernel
TsimCircuit: TypeAlias = tsim_backend.Circuit
MeasurementMap: TypeAlias = list[list[int]]


class FidelitySummary(TypedDict):
    """Summary statistics for a reconstructed single-qubit fidelity.

    Attributes:
        point: Point estimate of the fidelity.
        median: Median of the fidelity uncertainty distribution.
        low: Lower bound of the reported uncertainty interval.
        high: Upper bound of the reported uncertainty interval.
        error: Symmetric error bar derived from ``low`` and ``high``.
        bloch: Reconstructed Bloch-vector components after sign correction.
    """

    point: float
    median: float
    low: float
    high: float
    error: float
    bloch: tuple[float, float, float]


# NOTE: technically, this type won't be exposed to the user
class PosteriorFidelitySummary(TypedDict):
    """Posterior fidelity summary without the reconstructed Bloch vector.

    Attributes:
        point: Posterior mean fidelity.
        median: Posterior median fidelity.
        low: Lower credible interval bound.
        high: Upper credible interval bound.
        error: Symmetric error bar derived from the credible interval.
    """

    point: float
    median: float
    low: float
    high: float
    error: float


class DetectorObservableResult(Protocol):
    """Object exposing detector and observable samples.

    Attributes:
        detectors: Detector sample data.
        observables: Observable sample data.
    """

    @property
    def detectors(self) -> object: ...

    @property
    def observables(self) -> object: ...


# TODO: not sure if I like this overload logic; do we need it?
class SimulatorTask(Protocol):
    """Simulator task protocol used by sampling helpers.

    Implementations are expected to provide a ``run`` method compatible with
    ``GeminiLogicalSimulatorTask.run``.
    """

    @overload
    def run(
        self,
        shots: int,
        with_noise: bool = True,
        *,
        run_detectors: Literal[False] = ...,
    ) -> object: ...

    @overload
    def run(
        self,
        shots: int,
        with_noise: bool = True,
        *,
        run_detectors: Literal[True],
    ) -> DetectorObservableResult: ...

    @overload
    def run(
        self,
        shots: int,
        with_noise: bool = True,
        *,
        run_detectors: bool,
    ) -> object | DetectorObservableResult: ...


class DetectorErrorModelTask(Protocol):
    """Protocol for objects that expose a Stim detector error model.
    Used to denote tasks that just need to expose a detector error model (ex: to feed into a decoder).

    Attributes:
        detector_error_model: Stim detector error model for the task.
    """

    @property
    def detector_error_model(self) -> stim.DetectorErrorModel: ...
