from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Protocol,
    Sequence,
    TypeVar,
    cast,
)

import numpy as np
from bloqade.analysis.fidelity import FidelityAnalysis
from kirin import ir
from stim import DetectorErrorModel

from .simulator_backend import (
    AbstractSimulatorBackend,
    _get_tsim_circuit,
)

if TYPE_CHECKING:
    import tsim as tsim_backend  # type: ignore[reportMissingImports]

    from bloqade.lanes.analysis import atom
    from bloqade.lanes.arch.spec import ArchSpec

RetType = TypeVar("RetType")
ResultRetType = TypeVar("ResultRetType", covariant=True)


class SimulatorResult(Protocol[ResultRetType]):
    """Common interface for simulator results."""

    def fidelity_bounds(self) -> tuple[float, float]: ...

    @property
    def detector_error_model(self) -> DetectorErrorModel: ...

    @property
    def return_values(self) -> Sequence[ResultRetType]: ...

    @property
    def detectors(self) -> Sequence[Sequence[bool]]: ...

    @property
    def measurements(self) -> Sequence[Sequence[bool]]: ...

    @property
    def observables(self) -> Sequence[Sequence[bool]]: ...


@dataclass(frozen=True)
class DetectorResult(Generic[ResultRetType]):
    """Detector and observable outcomes from simulator sampling."""

    _detector_error_model: DetectorErrorModel
    _fidelity_min: float
    _fidelity_max: float
    _detectors: list[list[bool]]
    _observables: list[list[bool]]

    def fidelity_bounds(self) -> tuple[float, float]:
        """Return the upper and lower fidelity bounds.

        Returns:
            tuple[float, float]: The (min, max) fidelity bounds.

        """
        return (self._fidelity_min, self._fidelity_max)

    @property
    def detector_error_model(self) -> DetectorErrorModel:
        """The STIM detector error model corresponding to the physical noise circuit.

        Returns:
            DetectorErrorModel: The STIM detector error model.

        """
        return self._detector_error_model

    @property
    def return_values(self) -> Sequence[ResultRetType]:
        raise ValueError(
            "kernel return values are unavailable for detector-only results"
        )

    @property
    def detectors(self) -> tuple[tuple[bool, ...], ...]:
        """The detector outcomes from the simulation.

        Returns:
            tuple[tuple[bool, ...], ...]: The detector outcomes, one tuple per shot.

        """
        return tuple(tuple(shot) for shot in self._detectors)

    @property
    def measurements(self) -> Sequence[Sequence[bool]]:
        raise ValueError("Raw measurements are unavailable for detector-only results")

    @property
    def observables(self) -> tuple[tuple[bool, ...], ...]:
        """The observable outcomes from the simulation.

        Returns:
            tuple[tuple[bool, ...], ...]: The observable outcomes, one tuple per shot.

        """
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
        """Return the upper and lower fidelity bounds.

        Note: The upper and lower bounds are related to branching logic in the kernel.

        Returns:
            tuple[float, float]: The (min, max) fidelity bounds.

        """
        return (self._fidelity_min, self._fidelity_max)

    @property
    def detector_error_model(self) -> DetectorErrorModel:
        """The STIM detector error model corresponding to the physical noise circuit.

        Returns:
            DetectorErrorModel: The STIM detector error model.

        """
        return self._detector_error_model

    @property
    def return_values(self) -> list[RetType]:
        """The return values of the kernel.

        Returns:
            list[RetType]: The return values, one per shot.

        """
        return self._return_values

    @cached_property
    def _return_values(self) -> list[RetType]:
        return list(self._post_processing.emit_return(self._raw_measurements))

    @property
    def detectors(self) -> list[list[bool]]:
        """The detector outcomes from the simulation.

        Returns:
            list[list[bool]]: The detector outcomes, one list per shot.

        """
        return self._detectors

    @cached_property
    def _detectors(self) -> list[list[bool]]:
        return list(self._post_processing.emit_detectors(self._raw_measurements))

    @property
    def measurements(self) -> list[list[bool]]:
        """The raw measurement outcomes used to compute detectors and observables.

        Returns:
            list[list[bool]]: The raw measurement outcomes, one list per shot.

        """
        return self._measurements

    @cached_property
    def _measurements(self) -> list[list[bool]]:
        return list(map(list, self._raw_measurements))

    @property
    def observables(self) -> list[list[bool]]:
        """The observable outcomes from the simulation.

        Returns:
            list[list[bool]]: The observable outcomes, one list per shot.

        """
        return self._observables

    @cached_property
    def _observables(self) -> list[list[bool]]:
        return list(self._post_processing.emit_observables(self._raw_measurements))


class _SimulatorTaskBase(Generic[RetType]):
    """Shared execution runtime for compiled logical and physical tasks."""

    physical_arch_spec: ArchSpec
    physical_move_kernel: ir.Method[[], RetType]
    _post_processing: atom.PostProcessing[RetType]

    @property
    def _backend(self) -> AbstractSimulatorBackend:
        return cast(Any, self).backend

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
        """The noisy tsim measurement sampler."""
        return self.tsim_circuit.compile_sampler()

    @cached_property
    def noiseless_measurement_sampler(self):
        """The noiseless tsim measurement sampler."""
        return self.noiseless_tsim_circuit.compile_sampler()

    @cached_property
    def detector_sampler(self):
        """The noisy tsim detector sampler."""
        return self.tsim_circuit.compile_detector_sampler()

    @cached_property
    def noiseless_detector_sampler(self):
        """The noiseless tsim detector sampler."""
        return self.noiseless_tsim_circuit.compile_detector_sampler()

    @cached_property
    def detector_error_model(self) -> DetectorErrorModel:
        """Guaranteed detector error model for the noisy physical kernel."""
        return self._backend._detector_error_model(self._physical_kernel)

    def visualize(self, animated: bool = False, interactive: bool = True):
        """Visualize the physical move kernel using the built-in debugger.

        Args:
            animated (bool): Whether to use the animated debugger. Defaults to False.
            interactive (bool): Whether to enable interactive mode. Defaults to True.

        """
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
        """Compute the fidelity bounds for the physical squin kernel.

        Returns:
            tuple[float, float]: The (min, max) fidelity bounds.

        """
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

    def run(
        self,
        shots: int = 1,
        with_noise: bool = True,
    ) -> SimulatorResult[RetType]:
        """Run the kernel and get simulation results.

        Args:
            shots (int): Number of shots to run. Defaults to 1.
            with_noise (bool): Whether to include noise in the simulation. Defaults to True.

        Returns:
            SimulatorResult[RetType]: The simulation result containing measurements, detectors, and observables.

        """
        # Build the guaranteed DEM before beginning a potentially expensive
        # sampling request. This also fails early when Tsim is unavailable.
        detector_error_model = self.detector_error_model
        physical_kernel = (
            self._physical_kernel if with_noise else self._noiseless_physical_kernel
        )
        sample = self._backend.sample(physical_kernel, shots=shots)
        fidelity_min, fidelity_max = self.fidelity_bounds()

        has_measurements = sample.measurements is not None
        has_detectors = sample.detectors is not None
        has_observables = sample.observables is not None

        if has_measurements and not has_detectors and not has_observables:
            measurements = self._normalize_matrix(
                sample.measurements, name="measurement", shots=shots
            )
            return cast(
                SimulatorResult[RetType],
                Result(
                    measurements,
                    detector_error_model,
                    self._post_processing,
                    fidelity_min,
                    fidelity_max,
                ),
            )

        if not has_measurements and has_detectors and has_observables:
            detectors = self._normalize_matrix(
                sample.detectors, name="detector", shots=shots
            )
            observables = self._normalize_matrix(
                sample.observables, name="observable", shots=shots
            )
            return DetectorResult(
                _detector_error_model=detector_error_model,
                _fidelity_min=fidelity_min,
                _fidelity_max=fidelity_max,
                _detectors=detectors,
                _observables=observables,
            )

        raise ValueError(
            "Backend samples must be measurement-only or detector+observable-only"
        )

    def run_async(
        self,
        shots: int = 1,
        with_noise: bool = True,
    ) -> Future[SimulatorResult[RetType]]:
        """Run the kernel asynchronously and get simulation results.

        Args:
            shots (int): Number of shots to run. Defaults to 1.
            with_noise (bool): Whether to include noise in the simulation. Defaults to True.

        Returns:
            Future[SimulatorResult[RetType]]: A future resolving to the full simulation result.

        """
        return cast(
            Future[SimulatorResult[RetType]],
            self._thread_pool_executor.submit(self.run, shots, with_noise),
        )
