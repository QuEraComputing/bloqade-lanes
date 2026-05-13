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
    point: float
    median: float
    low: float
    high: float
    error: float
    bloch: tuple[float, float, float]


# NOTE: technically, this type won't be exposed to the user
class PosteriorFidelitySummary(TypedDict):
    point: float
    median: float
    low: float
    high: float
    error: float


class DetectorObservableResult(Protocol):
    @property
    def detectors(self) -> object: ...

    @property
    def observables(self) -> object: ...


# TODO: not sure if I like this overload logic; do we need it?
class SimulatorTask(Protocol):
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
    @property
    def detector_error_model(self) -> stim.DetectorErrorModel: ...
