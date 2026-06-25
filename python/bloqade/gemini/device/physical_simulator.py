from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Generic,
    Literal,
    TypeVar,
    overload,
)

import numpy as np
from bloqade.analysis.fidelity import FidelityAnalysis
from kirin import ir, passes, rewrite
from stim import DetectorErrorModel

from .simulator import DetectorResult

if TYPE_CHECKING:
    import tsim as tsim_backend  # type: ignore[reportMissingImports]

    from bloqade.lanes.arch.spec import ArchSpec
    from bloqade.lanes.rewrite.move2squin.noise import NoiseModelABC

RetType = TypeVar("RetType")


def _default_noise_model() -> "NoiseModelABC":
    from bloqade.lanes.noise_model import generate_simple_noise_model

    return generate_simple_noise_model()


def _default_arch_spec() -> "ArchSpec":
    from bloqade.lanes.arch.gemini.physical import get_arch_spec

    return get_arch_spec()


def _tsim():
    try:
        from bloqade import tsim
    except ImportError as exc:
        raise ImportError(
            "Gemini physical simulation requires the optional `tsim` extra. "
            "Install it with `bloqade-lanes[tsim]` or `uv sync --extra tsim`."
        ) from exc

    return tsim


@dataclass(frozen=True)
class PhysicalResult:
    """Simulation result for physical SQuIn programs."""

    _raw_measurements: list[list[bool]]
    _tsim_circuit: tsim_backend.Circuit
    _detector_error_model: DetectorErrorModel
    _fidelity_min: float
    _fidelity_max: float

    def fidelity_bounds(self) -> tuple[float, float]:
        """Return the lower and upper fidelity bounds."""
        return (self._fidelity_min, self._fidelity_max)

    @property
    def detector_error_model(self) -> DetectorErrorModel:
        """The STIM detector error model corresponding to the noisy circuit."""
        return self._detector_error_model

    @cached_property
    def measurements(self) -> list[list[bool]]:
        """The raw measurement outcomes from the measurement sampler."""
        return list(map(list, self._raw_measurements))

    @cached_property
    def _detector_observable_samples(self) -> tuple[np.ndarray, np.ndarray]:
        converter = self._tsim_circuit.compile_m2d_converter(skip_reference_sample=True)
        return converter.convert(
            measurements=np.asarray(self._raw_measurements, dtype=bool),
            separate_observables=True,
        )

    @cached_property
    def detectors(self) -> list[list[bool]]:
        """The detector outcomes derived from circuit annotations."""
        return self._detector_observable_samples[0].tolist()

    @cached_property
    def observables(self) -> list[list[bool]]:
        """The observable outcomes derived from circuit annotations."""
        return self._detector_observable_samples[1].tolist()


@dataclass(frozen=True)
class PhysicalSimulatorTask(Generic[RetType]):
    """A compiled simulation task for physical SQuIn programs."""

    source_squin_kernel: ir.Method[[], RetType]
    """The input physical SQuIn kernel."""
    noise_model: NoiseModelABC
    """The physical noise model to insert into the SQuIn kernel."""
    physical_arch_spec: ArchSpec = field(repr=False)
    """The physical architecture specification."""
    physical_move_kernel: ir.Method[[], RetType] = field(repr=False)
    """The physical move kernel compiled from the source SQuIn kernel."""
    _thread_pool_executor: ThreadPoolExecutor = field(
        default_factory=ThreadPoolExecutor, init=False
    )

    @cached_property
    def physical_squin_kernel(self) -> ir.Method[[], RetType]:
        """The physical SQuIn kernel with noise channels."""
        from bloqade.lanes.transform import MoveToSquinPhysical

        return MoveToSquinPhysical(
            arch_spec=self.physical_arch_spec,
            noise_model=self.noise_model,
        ).emit(self.physical_move_kernel)

    @cached_property
    def noiseless_physical_squin_kernel(self) -> ir.Method[[], RetType]:
        """The physical SQuIn kernel without noise channels."""
        from bloqade.lanes.transform import MoveToSquinPhysical

        return MoveToSquinPhysical(
            arch_spec=self.physical_arch_spec,
        ).emit(self.physical_move_kernel)

    @cached_property
    def tsim_circuit(self) -> tsim_backend.Circuit:
        """The tsim circuit corresponding to the noisy physical SQuIn kernel."""
        from bloqade.lanes.rewrite.squin2stim import RemoveReturn

        physical_squin_kernel = self.physical_squin_kernel.similar()
        rewrite.Walk(RemoveReturn()).rewrite(physical_squin_kernel.code)
        return _tsim().Circuit(physical_squin_kernel)

    @cached_property
    def noiseless_tsim_circuit(self) -> tsim_backend.Circuit:
        """The tsim circuit corresponding to the noiseless physical SQuIn kernel."""
        from bloqade.lanes.rewrite.squin2stim import RemoveReturn

        noiseless_kernel = self.noiseless_physical_squin_kernel.similar()
        rewrite.Walk(RemoveReturn()).rewrite(noiseless_kernel.code)
        return _tsim().Circuit(noiseless_kernel)

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
    def detector_error_model(self):
        """The STIM detector error model corresponding to the noisy tsim circuit."""
        return self.tsim_circuit.detector_error_model(approximate_disjoint_errors=True)

    def visualize(self, animated: bool = False, interactive: bool = True):
        """Visualize the compiled physical move kernel."""
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
        """Compute fidelity bounds for the noisy physical SQuIn kernel."""
        analysis = FidelityAnalysis(self.physical_squin_kernel.dialects)
        analysis.run(self.physical_squin_kernel)

        max_fidelity = 1.0
        min_fidelity = 1.0

        for gate_fid in analysis.gate_fidelities:
            min_fidelity *= gate_fid.min
            max_fidelity *= gate_fid.max

        return min_fidelity, max_fidelity

    @overload
    def run(
        self,
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: Literal[False] = ...,
    ) -> PhysicalResult: ...

    @overload
    def run(
        self,
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: Literal[True],
    ) -> DetectorResult: ...

    @overload
    def run(
        self,
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: bool,
    ) -> PhysicalResult | DetectorResult: ...

    def run(
        self,
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: bool = False,
    ) -> PhysicalResult | DetectorResult:
        """Run the task and return measurement or detector samples."""
        if run_detectors:
            return self._run_detectors(shots, with_noise)

        circuit = self.tsim_circuit if with_noise else self.noiseless_tsim_circuit
        if circuit.is_clifford:
            sampler = circuit.stim_circuit.compile_sampler()
        else:
            sampler = (
                self.measurement_sampler
                if with_noise
                else self.noiseless_measurement_sampler
            )

        raw_results = sampler.sample(shots=shots).tolist()
        fidelity_min, fidelity_max = self.fidelity_bounds()
        return PhysicalResult(
            raw_results,
            circuit,
            self.detector_error_model,
            fidelity_min,
            fidelity_max,
        )

    def _run_detectors(self, shots: int = 1, with_noise: bool = True) -> DetectorResult:
        """Run the detector sampler for detector and observable samples."""
        circuit = self.tsim_circuit if with_noise else self.noiseless_tsim_circuit
        if circuit.is_clifford:
            sampler = circuit.stim_circuit.compile_sampler()
            m2det = circuit.compile_m2d_converter(skip_reference_sample=True)
            samples = sampler.sample(shots=shots)
            det_obs = m2det.convert(measurements=samples, separate_observables=True)
        else:
            sampler = (
                self.detector_sampler if with_noise else self.noiseless_detector_sampler
            )
            det_obs: tuple[np.ndarray, np.ndarray] = sampler.sample(
                shots=shots, separate_observables=True
            )

        fidelity_min, fidelity_max = self.fidelity_bounds()
        return DetectorResult(
            _detector_error_model=self.detector_error_model,
            _fidelity_min=fidelity_min,
            _fidelity_max=fidelity_max,
            _detectors=det_obs[0].tolist(),
            _observables=det_obs[1].tolist(),
        )

    @overload
    def run_async(
        self,
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: Literal[False] = ...,
    ) -> Future[PhysicalResult]: ...

    @overload
    def run_async(
        self,
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: Literal[True],
    ) -> Future[DetectorResult]: ...

    def run_async(
        self,
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: bool = False,
    ) -> Future[PhysicalResult] | Future[DetectorResult]:
        """Run the task asynchronously."""
        if run_detectors:
            return self._thread_pool_executor.submit(
                self._run_detectors, shots, with_noise
            )
        return self._thread_pool_executor.submit(self.run, shots, with_noise)


@dataclass
class PhysicalSimulator:
    """Simulator for programs written directly at the physical SQuIn level."""

    noise_model: NoiseModelABC = field(default_factory=_default_noise_model)
    """The physical noise model used for simulation."""
    arch_spec: ArchSpec = field(default_factory=_default_arch_spec)
    """The physical architecture specification used for compilation."""

    def task(
        self,
        physical_kernel: ir.Method[[], RetType],
        place_opt_type: type[passes.Pass] | None = None,
    ) -> PhysicalSimulatorTask[RetType]:
        """Compile a physical SQuIn kernel into a reusable simulation task."""
        from bloqade.lanes.passes import SequentialPlacePass
        from bloqade.lanes.pipeline import PhysicalPipeline

        if place_opt_type is None:
            place_opt_type = SequentialPlacePass

        physical_pipeline = PhysicalPipeline(
            arch_spec=self.arch_spec,
            place_opt_type=place_opt_type,
        )
        physical_move_kernel = physical_pipeline.emit(physical_kernel, no_raise=False)

        return PhysicalSimulatorTask(
            physical_kernel,
            self.noise_model,
            self.arch_spec,
            physical_move_kernel,
        )

    @overload
    def run(
        self,
        physical_kernel: ir.Method[[], RetType],
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: Literal[False] = ...,
    ) -> PhysicalResult: ...

    @overload
    def run(
        self,
        physical_kernel: ir.Method[[], RetType],
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: Literal[True],
    ) -> DetectorResult: ...

    @overload
    def run(
        self,
        physical_kernel: ir.Method[[], RetType],
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: bool,
    ) -> PhysicalResult | DetectorResult: ...

    def run(
        self,
        physical_kernel: ir.Method[[], RetType],
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: bool = False,
    ) -> PhysicalResult | DetectorResult:
        """Compile and run a physical SQuIn kernel."""
        return self.task(physical_kernel).run(
            shots, with_noise, run_detectors=run_detectors
        )

    @overload
    def run_async(
        self,
        physical_kernel: ir.Method[[], RetType],
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: Literal[False] = ...,
    ) -> Future[PhysicalResult]: ...

    @overload
    def run_async(
        self,
        physical_kernel: ir.Method[[], RetType],
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: Literal[True],
    ) -> Future[DetectorResult]: ...

    def run_async(
        self,
        physical_kernel: ir.Method[[], RetType],
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: bool = False,
    ) -> Future[PhysicalResult] | Future[DetectorResult]:
        """Compile and run a physical SQuIn kernel asynchronously."""
        task = self.task(physical_kernel)
        if run_detectors:
            return task.run_async(shots, with_noise, run_detectors=True)
        return task.run_async(shots, with_noise)

    def visualize(
        self,
        physical_kernel: ir.Method[[], RetType],
        animated: bool = False,
        interactive: bool = True,
    ):
        """Visualize the compiled physical move kernel."""
        self.task(physical_kernel).visualize(animated=animated, interactive=interactive)

    def physical_squin_kernel(
        self, physical_kernel: ir.Method[[], RetType]
    ) -> ir.Method[[], RetType]:
        """Compile the source physical SQuIn kernel to a noisy physical SQuIn kernel."""
        return self.task(physical_kernel).physical_squin_kernel

    def physical_move_kernel(
        self, physical_kernel: ir.Method[[], RetType]
    ) -> ir.Method[[], RetType]:
        """Compile the source physical SQuIn kernel to the physical move dialect."""
        return self.task(physical_kernel).physical_move_kernel

    def tsim_circuit(
        self, physical_kernel: ir.Method[[], RetType], with_noise: bool = True
    ) -> tsim_backend.Circuit:
        """Compile the physical SQuIn kernel to a tsim circuit."""
        task = self.task(physical_kernel)
        if with_noise:
            return task.tsim_circuit
        return task.noiseless_tsim_circuit

    def fidelity_bounds(
        self, physical_kernel: ir.Method[[], RetType]
    ) -> tuple[float, float]:
        """Get the fidelity bounds for the physical SQuIn kernel."""
        return self.task(physical_kernel).fidelity_bounds()
