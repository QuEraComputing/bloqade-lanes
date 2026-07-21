from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    TypeVar,
    cast,
    overload,
)

import numpy as np
from bloqade.analysis.fidelity import FidelityAnalysis
from kirin import ir
from stim import DetectorErrorModel

from .simulator_backend import (
    AbstractSimulatorBackend,
    BackendSample,
    _get_tsim_circuit,
)

if TYPE_CHECKING:
    import tsim as tsim_backend  # type: ignore[reportMissingImports]

    from bloqade.lanes.analysis import atom
    from bloqade.lanes.arch.spec import ArchSpec

RetType = TypeVar("RetType")


def _validate_seed(seed: int | None) -> None:
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63
    ):
        raise ValueError("seed must be a non-bool int in the range [0, 2**63).")


@dataclass(frozen=True)
class DetectorResult:
    """Detector and observable outcomes from simulator sampling."""

    _detector_error_model: DetectorErrorModel
    _fidelity_min: float
    _fidelity_max: float
    _detectors: list[list[bool]]
    _observables: list[list[bool]]

    def fidelity_bounds(self) -> tuple[float, float]:
        return (self._fidelity_min, self._fidelity_max)

    @property
    def detector_error_model(self) -> DetectorErrorModel:
        return self._detector_error_model

    @property
    def detectors(self) -> tuple[tuple[bool, ...], ...]:
        return tuple(tuple(shot) for shot in self._detectors)

    @property
    def observables(self) -> tuple[tuple[bool, ...], ...]:
        return tuple(tuple(shot) for shot in self._observables)


@dataclass(frozen=True)
class Result(Generic[RetType]):
    """Measurements, post-processed values, fidelity, and a guaranteed DEM."""

    _raw_measurements: list[list[bool]]
    _detector_error_model: DetectorErrorModel
    _post_processing: atom.PostProcessing[RetType]
    _fidelity_min: float
    _fidelity_max: float

    def fidelity_bounds(self) -> tuple[float, float]:
        return (self._fidelity_min, self._fidelity_max)

    @property
    def detector_error_model(self) -> DetectorErrorModel:
        return self._detector_error_model

    @cached_property
    def return_values(self) -> list[RetType]:
        return list(self._post_processing.emit_return(self._raw_measurements))

    @cached_property
    def detectors(self) -> list[list[bool]]:
        return list(self._post_processing.emit_detectors(self._raw_measurements))

    @cached_property
    def measurements(self) -> list[list[bool]]:
        return list(map(list, self._raw_measurements))

    @cached_property
    def observables(self) -> list[list[bool]]:
        return list(self._post_processing.emit_observables(self._raw_measurements))


class _SimulatorTaskBase(Generic[RetType]):
    """Shared execution runtime for compiled logical and physical tasks."""

    physical_arch_spec: ArchSpec
    physical_move_kernel: ir.Method[[], RetType]
    _post_processing: atom.PostProcessing[RetType]

    @property
    def _backend(self) -> AbstractSimulatorBackend:
        return cast(Any, self).simulator.backend

    @property
    def _physical_kernel(self) -> ir.Method[[], RetType]:
        return cast(Any, self).physical_squin_kernel

    @property
    def _noiseless_physical_kernel(self) -> ir.Method[[], RetType]:
        return cast(Any, self).noiseless_physical_squin_kernel

    @cached_property
    def _thread_pool_executor(self) -> ThreadPoolExecutor:
        return ThreadPoolExecutor()

    @cached_property
    def tsim_circuit(self) -> tsim_backend.Circuit:
        """Tsim-compatible circuit for the noisy physical kernel."""
        return _get_tsim_circuit(self._backend, self._physical_kernel)

    @cached_property
    def noiseless_tsim_circuit(self) -> tsim_backend.Circuit:
        """Tsim-compatible circuit for the noiseless physical kernel."""
        return _get_tsim_circuit(self._backend, self._noiseless_physical_kernel)

    @cached_property
    def measurement_sampler(self):
        return self.tsim_circuit.compile_sampler()

    @cached_property
    def noiseless_measurement_sampler(self):
        return self.noiseless_tsim_circuit.compile_sampler()

    @cached_property
    def detector_sampler(self):
        return self.tsim_circuit.compile_detector_sampler()

    @cached_property
    def noiseless_detector_sampler(self):
        return self.noiseless_tsim_circuit.compile_detector_sampler()

    @cached_property
    def detector_error_model(self) -> DetectorErrorModel:
        """Guaranteed detector error model for the noisy physical kernel."""
        return self._backend._detector_error_model(self._physical_kernel)

    def visualize(self, animated: bool = False, interactive: bool = True):
        from bloqade.lanes.visualize import animated_debugger, debugger

        if animated:
            animated_debugger(
                self.physical_move_kernel,
                self.physical_arch_spec,
                interactive=interactive,
            )
        else:
            debugger(
                self.physical_move_kernel,
                self.physical_arch_spec,
                interactive=interactive,
            )

    def fidelity_bounds(self) -> tuple[float, float]:
        analysis = FidelityAnalysis(self._physical_kernel.dialects)
        analysis.run(self._physical_kernel)

        min_fidelity = 1.0
        max_fidelity = 1.0
        for gate_fidelity in analysis.gate_fidelities:
            min_fidelity *= gate_fidelity.min
            max_fidelity *= gate_fidelity.max
        return min_fidelity, max_fidelity

    @staticmethod
    def _normalize_matrix(payload: Any, *, name: str, shots: int) -> list[list[bool]]:
        try:
            array = np.asarray(payload, dtype=bool)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Backend returned invalid {name} samples") from exc
        if array.ndim != 2:
            raise ValueError(f"Backend {name} samples must be a two-dimensional array")
        if array.shape[0] != shots:
            raise ValueError(
                f"Backend returned {array.shape[0]} {name} rows for {shots} shots"
            )
        return array.tolist()

    @overload
    def run(
        self,
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: Literal[False] = ...,
        seed: int | None = None,
    ) -> Result[RetType]: ...

    @overload
    def run(
        self,
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: Literal[True],
        seed: int | None = None,
    ) -> DetectorResult: ...

    @overload
    def run(
        self,
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: bool,
        seed: int | None = None,
    ) -> Result[RetType] | DetectorResult: ...

    def run(
        self,
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: bool = False,
        seed: int | None = None,
    ) -> Result[RetType] | DetectorResult:
        """Sample through the configured backend and normalize its result."""
        # Build the guaranteed DEM before beginning a potentially expensive
        # sampling request. This also fails early when Tsim is unavailable.
        _validate_seed(seed)
        detector_error_model = self.detector_error_model
        physical_kernel = (
            self._physical_kernel if with_noise else self._noiseless_physical_kernel
        )
        sample = self._backend.sample(
            physical_kernel, shots=shots, run_detectors=run_detectors, seed=seed
        )
        fidelity_min, fidelity_max = self.fidelity_bounds()

        if run_detectors:
            return self._detector_result(
                sample,
                shots=shots,
                detector_error_model=detector_error_model,
                fidelity_min=fidelity_min,
                fidelity_max=fidelity_max,
            )

        if sample.measurements is None:
            raise ValueError("Backend did not return measurement samples")
        measurements = self._normalize_matrix(
            sample.measurements, name="measurement", shots=shots
        )
        return Result(
            measurements,
            detector_error_model,
            self._post_processing,
            fidelity_min,
            fidelity_max,
        )

    def _detector_result(
        self,
        sample: BackendSample,
        *,
        shots: int,
        detector_error_model: DetectorErrorModel,
        fidelity_min: float,
        fidelity_max: float,
    ) -> DetectorResult:
        has_detectors = sample.detectors is not None
        has_observables = sample.observables is not None
        if has_detectors or has_observables:
            if not (has_detectors and has_observables):
                raise ValueError(
                    "Backend must return detector and observable samples together"
                )
            detectors = self._normalize_matrix(
                sample.detectors, name="detector", shots=shots
            )
            observables = self._normalize_matrix(
                sample.observables, name="observable", shots=shots
            )
        elif sample.measurements is not None:
            measurements = self._normalize_matrix(
                sample.measurements, name="measurement", shots=shots
            )
            detectors = list(self._post_processing.emit_detectors(measurements))
            observables = list(self._post_processing.emit_observables(measurements))
        else:
            raise ValueError("Backend did not return detector or measurement samples")

        return DetectorResult(
            _detector_error_model=detector_error_model,
            _fidelity_min=fidelity_min,
            _fidelity_max=fidelity_max,
            _detectors=detectors,
            _observables=observables,
        )

    @overload
    def run_async(
        self,
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: Literal[False] = ...,
        seed: int | None = None,
    ) -> Future[Result[RetType]]: ...

    @overload
    def run_async(
        self,
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: Literal[True],
        seed: int | None = None,
    ) -> Future[DetectorResult]: ...

    @overload
    def run_async(
        self,
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: bool,
        seed: int | None = None,
    ) -> Future[Result[RetType]] | Future[DetectorResult]: ...

    def run_async(
        self,
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: bool = False,
        seed: int | None = None,
    ) -> Future[Result[RetType]] | Future[DetectorResult]:
        _validate_seed(seed)
        if run_detectors:
            return cast(
                Future[DetectorResult],
                self._thread_pool_executor.submit(
                    self.run, shots, with_noise, run_detectors=True, seed=seed
                ),
            )
        return cast(
            Future[Result[RetType]],
            self._thread_pool_executor.submit(
                self.run, shots, with_noise, run_detectors=False, seed=seed
            ),
        )
