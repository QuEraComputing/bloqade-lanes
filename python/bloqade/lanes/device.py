from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Callable, Generic, Literal, TypeVar, Union, overload

import numpy as np
import tsim as tsim_backend
from bloqade.analysis.fidelity import FidelityAnalysis
from kirin import ir, rewrite
from stim import DetectorErrorModel

from bloqade import tsim
from bloqade.lanes.analysis import atom
from bloqade.lanes.arch.gemini.impls import generate_arch_hypercube
from bloqade.lanes.arch.gemini.logical import steane7_initialize
from bloqade.lanes.cudaq_integration import cudaq_to_squin, is_cudaq_kernel
from bloqade.lanes.layout.arch import ArchSpec
from bloqade.lanes.logical_mvp import (
    _find_qubit_ssas,
    append_measurements_and_annotations,
    compile_squin_to_move,
    run_squin_kernel_validation,
)
from bloqade.lanes.noise_model import generate_simple_noise_model
from bloqade.lanes.rewrite.move2squin.noise import NoiseModelABC
from bloqade.lanes.rewrite.squin2stim import RemoveReturn
from bloqade.lanes.steane_defaults import steane7_m2dets, steane7_m2obs
from bloqade.lanes.transform import MoveToSquin

RetType = TypeVar("RetType")


@dataclass(frozen=True)
class DetectorResult:
    """Result from the detector sampler containing only detector and observable outcomes."""

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
    def detectors(self) -> tuple[list[bool], ...]:
        """The detector outcomes from the simulation.

        Returns:
            tuple[list[bool], ...]: The detector outcomes, one list per shot.

        """
        return tuple(self._detectors)

    @property
    def observables(self) -> tuple[list[bool], ...]:
        """The observable outcomes from the simulation.

        Returns:
            tuple[list[bool], ...]: The observable outcomes, one list per shot.

        """
        return tuple(self._observables)


@dataclass(frozen=True)
class Result(Generic[RetType]):
    """Simulation result including measurement outcomes, detector error model, post-processing, and fidelity bounds."""

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

    @cached_property
    def return_values(self) -> list[RetType]:
        """The return values of the logical kernel.

        Returns:
            list[RetType]: The return values, one per shot.

        """
        return list(self._post_processing.emit_return(self._raw_measurements))

    @cached_property
    def detectors(self) -> list[list[bool]]:
        """The detector outcomes from the simulation.

        Returns:
            list[list[bool]]: The detector outcomes, one list per shot.

        """
        return list(self._post_processing.emit_detectors(self._raw_measurements))

    @cached_property
    def measurements(self) -> list[list[bool]]:
        """The raw measurement outcomes used to compute detectors and observables.

        Returns:
            list[list[bool]]: The raw measurement outcomes, one list per shot.

        """
        return list(map(list, self._raw_measurements))

    @cached_property
    def observables(self) -> list[list[bool]]:
        """The observable outcomes from the simulation.

        Returns:
            list[list[bool]]: The observable outcomes, one list per shot.

        """
        return list(self._post_processing.emit_observables(self._raw_measurements))


@dataclass(frozen=True)
class GeminiLogicalSimulatorTask(Generic[RetType]):
    """A compiled simulation task for the Gemini logical simulator.

    Created by :meth:`GeminiLogicalSimulator.task`. The squin-to-move compilation
    and post-processing extraction are performed eagerly at construction time.
    Simulation artifacts (physical squin kernel, stim circuits, samplers, detector
    error model) are computed lazily on first access since they depend on the
    noise model.
    """

    logical_squin_kernel: ir.Method[[], RetType]
    """The input logical squin kernel to be executed on the Gemini architecture."""
    noise_model: NoiseModelABC
    """The noise model to be inserted into the physical squin kernel."""
    physical_arch_spec: ArchSpec = field(repr=False)
    """The physical architecture specification."""
    physical_move_kernel: ir.Method[[], RetType] = field(repr=False)
    """The physical move kernel that executes the logical squin kernel on the physical architecture."""
    _post_processing: atom.PostProcessing[RetType] = field(repr=False)
    """The post-processing object for extracting detectors, observables, and return values."""
    _thread_pool_executor: ThreadPoolExecutor = field(
        default_factory=ThreadPoolExecutor, init=False
    )

    @cached_property
    def physical_squin_kernel(self) -> ir.Method[[], RetType]:
        """The physical squin kernel corresponding to the physical move kernel, including noise."""
        return MoveToSquin(
            self.physical_arch_spec,
            steane7_initialize,
            self.noise_model,
        ).emit(self.physical_move_kernel)

    @cached_property
    def tsim_circuit(self) -> tsim_backend.Circuit:
        """The tsim circuit corresponding to the physical squin kernel."""
        physical_squin_kernel = self.physical_squin_kernel.similar()
        rewrite.Walk(RemoveReturn()).rewrite(physical_squin_kernel.code)
        return tsim.Circuit(physical_squin_kernel)

    @cached_property
    def noiseless_tsim_circuit(self) -> tsim_backend.Circuit:
        """The noiseless tsim circuit."""
        return self.tsim_circuit.without_noise()

    @cached_property
    def measurement_sampler(self):
        """The tsim measurement sampler."""
        return self.tsim_circuit.compile_sampler()

    @cached_property
    def noiseless_measurement_sampler(self):
        """The noiseless tsim measurement sampler."""
        return self.noiseless_tsim_circuit.compile_sampler()

    @cached_property
    def detector_sampler(self):
        """The tsim detector sampler."""
        return self.tsim_circuit.compile_detector_sampler()

    @cached_property
    def noiseless_detector_sampler(self):
        """The noiseless tsim detector sampler."""
        return self.noiseless_tsim_circuit.compile_detector_sampler()

    @cached_property
    def detector_error_model(self):
        """The STIM detector error model corresponding to the tsim circuit."""
        return self.tsim_circuit.detector_error_model(approximate_disjoint_errors=True)

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
    ) -> Result[RetType]: ...

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
    ) -> Result[RetType] | DetectorResult: ...

    def run(
        self,
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: bool = False,
    ) -> Result[RetType] | DetectorResult:
        """Run the kernel and get simulation results.

        Args:
            shots (int): Number of shots to run. Defaults to 1.
            with_noise (bool): Whether to include noise in the simulation. Defaults to True.
            run_detectors (bool): When ``True``, use the detector sampler instead of
                the measurement sampler for faster detector/observable sampling.
                Defaults to False.

        Returns:
            Result[RetType]: When ``run_detectors=False``, the full simulation result
                including measurement outcomes, return values, detectors, and observables.
            DetectorResult: When ``run_detectors=True``, the result containing only
                detector and observable outcomes.

        """
        if run_detectors:
            return self._run_detectors(shots, with_noise)

        if with_noise:
            raw_results = self.measurement_sampler.sample(shots=shots).tolist()
        else:
            raw_results = self.noiseless_measurement_sampler.sample(
                shots=shots
            ).tolist()

        fidelity_min, fidelity_max = self.fidelity_bounds()
        return Result(
            raw_results,
            self.detector_error_model,
            self._post_processing,
            fidelity_min,
            fidelity_max,
        )

    def _run_detectors(self, shots: int = 1, with_noise: bool = True) -> DetectorResult:
        """Run the detector sampler for faster detector/observable sampling.

        This skips the full measurement sampler and directly samples detector
        and observable outcomes, which is significantly faster when only
        detectors and observables are needed.

        Args:
            shots (int): Number of shots to run. Defaults to 1.
            with_noise (bool): Whether to include noise in the simulation. Defaults to True.

        Returns:
            DetectorResult: The result containing detector and observable outcomes.

        """
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
    ) -> Future[Result[RetType]]: ...

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
    ) -> Future[Result[RetType]] | Future[DetectorResult]:
        """Run the kernel asynchronously and get simulation results.

        Args:
            shots (int): Number of shots to run. Defaults to 1.
            with_noise (bool): Whether to include noise in the simulation. Defaults to True.
            run_detectors (bool): When ``True``, use the detector sampler instead of
                the measurement sampler. Defaults to False.

        Returns:
            Future[Result[RetType]]: When ``run_detectors=False``, a future resolving
                to the full simulation result.
            Future[DetectorResult]: When ``run_detectors=True``, a future resolving
                to the detector result.

        """
        if run_detectors:
            return self._thread_pool_executor.submit(
                self._run_detectors, shots, with_noise
            )
        return self._thread_pool_executor.submit(self.run, shots, with_noise)


@dataclass
class GeminiLogicalSimulator:
    """Logical simulator targeting the Gemini neutral-atom architecture.

    This is the primary entry point for compiling and simulating logical quantum
    circuits on the Gemini architecture. Use :meth:`task` to compile a kernel into
    a reusable :class:`GeminiLogicalSimulatorTask`, or :meth:`run` for one-shot
    compile-and-execute convenience.
    """

    noise_model: NoiseModelABC = field(default_factory=generate_simple_noise_model)
    """The noise model used for simulation. Defaults to :func:`generate_simple_noise_model`."""

    def task(
        self,
        logical_kernel: Union[ir.Method[[], RetType], Callable[..., Any]],
        m2dets: list[list[int]] | None = None,
        m2obs: list[list[int]] | None = None,
    ) -> GeminiLogicalSimulatorTask[RetType]:
        """Create a simulation task for the given kernel.

        Eagerly compiles the kernel through squin-to-move and extracts post-processing.
        For CUDA-Q kernels, detector and observable annotation matrices default to
        Steane [[7,1,3]] parity checks when not provided.

        Args:
            logical_kernel (Union[ir.Method[[], RetType], Callable[..., Any]]): The logical
                squin or CUDA-Q kernel to compile and run.
            m2dets (list[list[int]] | None): Binary measurement-to-detector matrix.
                For CUDA-Q kernels, defaults to Steane [[7,1,3]] detectors if ``None``.
            m2obs (list[list[int]] | None): Binary measurement-to-observable matrix.
                For CUDA-Q kernels, defaults to Steane [[7,1,3]] observables if ``None``.

        Returns:
            GeminiLogicalSimulatorTask[RetType]: The compiled simulation task.

        """
        if is_cudaq_kernel(logical_kernel):
            logical_squin_kernel: ir.Method = cudaq_to_squin(logical_kernel)

            # Default to Steane [[7,1,3]] annotation matrices when not provided
            if m2dets is None and m2obs is None:
                num_qubits = len(_find_qubit_ssas(logical_squin_kernel))
                m2dets = steane7_m2dets(num_qubits)
                m2obs = steane7_m2obs(num_qubits)

            append_measurements_and_annotations(logical_squin_kernel, m2dets, m2obs)
        elif isinstance(logical_kernel, ir.Method):
            logical_squin_kernel = logical_kernel
            if m2dets is not None or m2obs is not None:
                append_measurements_and_annotations(logical_squin_kernel, m2dets, m2obs)
        else:
            raise ValueError(f"Unknown kernel type {type(logical_kernel)}")

        run_squin_kernel_validation(logical_squin_kernel).raise_if_invalid()

        physical_arch_spec = generate_arch_hypercube(4)
        physical_move_kernel = compile_squin_to_move(
            logical_squin_kernel, transversal_rewrite=True
        )
        post_processing = atom.AtomInterpreter(
            physical_move_kernel.dialects, arch_spec=physical_arch_spec
        ).get_post_processing(physical_move_kernel)

        return GeminiLogicalSimulatorTask(
            logical_squin_kernel,
            self.noise_model,
            physical_arch_spec,
            physical_move_kernel,
            post_processing,
        )

    @overload
    def run(
        self,
        logical_squin_kernel: ir.Method[[], RetType],
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: Literal[False] = ...,
    ) -> Result[RetType]: ...

    @overload
    def run(
        self,
        logical_squin_kernel: ir.Method[[], RetType],
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: Literal[True],
    ) -> DetectorResult: ...

    @overload
    def run(
        self,
        logical_squin_kernel: ir.Method[[], RetType],
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: bool,
    ) -> Result[RetType] | DetectorResult: ...

    def run(
        self,
        logical_squin_kernel: ir.Method[[], RetType],
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: bool = False,
    ) -> Result[RetType] | DetectorResult:
        """Run the kernel and get simulation results.

        Args:
            logical_squin_kernel (ir.Method[[], RetType]): The logical squin kernel to run.
            shots (int): Number of shots to run. Defaults to 1.
            with_noise (bool): Whether to include noise in the simulation. Defaults to True.
            run_detectors (bool): When ``True``, use the detector sampler instead of
                the measurement sampler for faster detector/observable sampling.
                Defaults to False.

        Returns:
            Result[RetType]: When ``run_detectors=False``, the full simulation result.
            DetectorResult: When ``run_detectors=True``, the result containing only
                detector and observable outcomes.

        """
        return self.task(logical_squin_kernel).run(
            shots, with_noise, run_detectors=run_detectors
        )

    @overload
    def run_async(
        self,
        logical_squin_kernel: ir.Method[[], RetType],
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: Literal[False] = ...,
    ) -> Future[Result[RetType]]: ...

    @overload
    def run_async(
        self,
        logical_squin_kernel: ir.Method[[], RetType],
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: Literal[True],
    ) -> Future[DetectorResult]: ...

    def run_async(
        self,
        logical_squin_kernel: ir.Method[[], RetType],
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: bool = False,
    ) -> Future[Result[RetType]] | Future[DetectorResult]:
        """Run the kernel asynchronously and get simulation results.

        Args:
            logical_squin_kernel (ir.Method[[], RetType]): The logical squin kernel to run.
            shots (int): Number of shots to run. Defaults to 1.
            with_noise (bool): Whether to include noise in the simulation. Defaults to True.
            run_detectors (bool): When ``True``, use the detector sampler instead of
                the measurement sampler. Defaults to False.

        Returns:
            Future[Result[RetType]]: When ``run_detectors=False``, a future resolving
                to the full simulation result.
            Future[DetectorResult]: When ``run_detectors=True``, a future resolving
                to the detector result.

        """
        task = self.task(logical_squin_kernel)
        if run_detectors:
            return task.run_async(shots, with_noise, run_detectors=True)
        return task.run_async(shots, with_noise)

    def visualize(
        self,
        logical_squin_kernel: ir.Method[[], RetType],
        animated: bool = False,
        interactive: bool = True,
    ):
        """Visualize the physical move kernel using the built-in debugger.

        Args:
            logical_squin_kernel (ir.Method[[], RetType]): The logical squin kernel to visualize.
            animated (bool): Whether to use the animated debugger. Defaults to False.
            interactive (bool): Whether to enable interactive mode. Defaults to True.

        """
        self.task(logical_squin_kernel).visualize(
            animated=animated, interactive=interactive
        )

    def physical_squin_kernel(
        self, logical_squin_kernel: ir.Method[[], RetType]
    ) -> ir.Method[[], RetType]:
        """Compile the logical squin kernel to the physical squin kernel.

        Args:
            logical_squin_kernel (ir.Method[[], RetType]): The logical squin kernel to compile.

        Returns:
            ir.Method[[], RetType]: The physical squin kernel.

        """
        return self.task(logical_squin_kernel).physical_squin_kernel

    def physical_move_kernel(
        self, logical_squin_kernel: ir.Method[[], RetType]
    ) -> ir.Method[[], RetType]:
        """Compile the logical squin kernel to the physical move kernel.

        Args:
            logical_squin_kernel (ir.Method[[], RetType]): The logical squin kernel to compile.

        Returns:
            ir.Method[[], RetType]: The physical move kernel.

        """
        return self.task(logical_squin_kernel).physical_move_kernel

    def tsim_circuit(
        self, logical_squin_kernel: ir.Method[[], RetType], with_noise: bool = True
    ) -> tsim_backend.Circuit:
        """Compile the logical squin kernel to the tsim circuit.

        Args:
            logical_squin_kernel (ir.Method[[], RetType]): The logical squin kernel to compile.
            with_noise (bool): Whether to include noise in the tsim circuit. Defaults to True.

        Returns:
            tsim.Circuit: The compiled tsim circuit.

        """
        if with_noise:
            return self.task(logical_squin_kernel).tsim_circuit
        else:
            return self.task(logical_squin_kernel).noiseless_tsim_circuit

    def fidelity_bounds(
        self, logical_squin_kernel: ir.Method[[], RetType]
    ) -> tuple[float, float]:
        """Get the fidelity bounds for the logical squin kernel.

        Args:
            logical_squin_kernel (ir.Method[[], RetType]): The logical squin kernel to analyze.

        Returns:
            tuple[float, float]: The (min, max) fidelity bounds.

        """
        return self.task(logical_squin_kernel).fidelity_bounds()
