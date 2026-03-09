from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cached_property
from typing import Generic, TypeVar

import numpy as np
import tsim as tsim_backend
from bloqade.analysis.fidelity import FidelityAnalysis
from kirin import ir, rewrite
from stim import DetectorErrorModel

from bloqade import tsim
from bloqade.lanes.analysis import atom
from bloqade.lanes.arch.gemini.impls import generate_arch_hypercube
from bloqade.lanes.arch.gemini.logical import steane7_initialize
from bloqade.lanes.logical_mvp import compile_squin_to_move, run_squin_kernel_validation
from bloqade.lanes.noise_model import generate_simple_noise_model
from bloqade.lanes.rewrite.move2squin.noise import NoiseModelABC
from bloqade.lanes.rewrite.squin2stim import RemoveReturn
from bloqade.lanes.transform import MoveToSquin

RetType = TypeVar("RetType")


@dataclass(frozen=True)
class Result(Generic[RetType]):
    """Simulation result including detector error model, fidelity bounds, and sampling outcomes.

    When constructed via the default measurement sampler path, ``measurements``,
    ``return_values``, ``detectors``, and ``observables`` are all available.

    When constructed via the ``no_measurements=True`` path, only ``detectors``
    and ``observables`` are populated directly from the detector sampler.
    Accessing ``measurements`` or ``return_values`` in this mode raises
    ``ValueError``.
    """

    _detector_error_model: DetectorErrorModel
    _fidelity_min: float
    _fidelity_max: float
    _raw_measurements: list[list[bool]] | None = None
    _post_processing: atom.PostProcessing[RetType] | None = None
    _detectors: list[list[bool]] | None = None
    _observables: list[list[bool]] | None = None

    def fidelity_bounds(self) -> tuple[float, float]:
        """Return the upper and lower fidelity bounds.

        Note: The upper and lower bounds are related to and branching logic in the kernel.

        """
        return (self._fidelity_min, self._fidelity_max)

    @property
    def detector_error_model(self) -> DetectorErrorModel:
        """The STIM detector error model corresponding to the physical noise circuit."""
        return self._detector_error_model

    @property
    def return_values(self) -> list[RetType]:
        """The return values of the logical kernel.

        Raises:
            ValueError: If the result was produced with ``no_measurements=True``.
        """
        if self._post_processing is None or self._raw_measurements is None:
            raise ValueError("return values not accessible with `no_measurements=True`")
        return list(self._post_processing.emit_return(self._raw_measurements))

    @property
    def detectors(self) -> list[list[bool]]:
        """The detector outcomes from the simulation."""
        if self._detectors is not None:
            return self._detectors
        if self._post_processing is None or self._raw_measurements is None:
            raise ValueError(
                "detectors not accessible with `no_measurements=True`; "
                "use the detector sampler API on the task instead"
            )
        return list(self._post_processing.emit_detectors(self._raw_measurements))

    @property
    def measurements(self) -> list[list[bool]]:
        """The raw measurement outcomes used to compute detectors and observables.

        Raises:
            ValueError: If the result was produced with ``no_measurements=True``.
        """
        if self._raw_measurements is None:
            raise ValueError("measurements not accessible with `no_measurements=True`")
        return list(map(list, self._raw_measurements))

    @property
    def observables(self) -> list[list[bool]]:
        """The observable outcomes from the simulation."""
        if self._observables is not None:
            return self._observables
        if self._post_processing is None or self._raw_measurements is None:
            raise ValueError(
                "observables not accessible with `no_measurements=True`; "
                "use the detector sampler API on the task instead"
            )
        return list(self._post_processing.emit_observables(self._raw_measurements))


@dataclass(frozen=True)
class GeminiLogicalSimulatorTask(Generic[RetType]):
    logical_squin_kernel: ir.Method[[], RetType]
    """The input logical squin kernel to be executed on the Gemini architecture."""
    noise_model: NoiseModelABC
    """The noise model to be inserted into the physical squin kernel."""
    no_measurements: bool = False
    """When True, skip the measurement sampler and use the detector sampler instead."""
    _thread_pool_executor: ThreadPoolExecutor = field(
        default_factory=ThreadPoolExecutor, init=False
    )

    def __post_init__(self):
        if not self.no_measurements:
            assert isinstance(self._post_processing, atom.PostProcessing)

    @cached_property
    def physical_arch_spec(self):
        """The physical architecture specification."""
        return generate_arch_hypercube(4)

    @cached_property
    def physical_move_kernel(self) -> ir.Method[[], RetType]:
        """The physical move kernel that execute the logical squin kernel on the physical architecture."""
        return compile_squin_to_move(
            self.logical_squin_kernel, transversal_rewrite=True
        )

    @cached_property
    def _post_processing(self):
        return atom.AtomInterpreter(
            self.physical_move_kernel.dialects, arch_spec=self.physical_arch_spec
        ).get_post_processing(self.physical_move_kernel)

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

        Args
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
        analysis = FidelityAnalysis(self.physical_squin_kernel.dialects)
        analysis.run(self.physical_squin_kernel)

        max_fidelity = 1.0
        min_fidelity = 1.0

        for gate_fid in analysis.gate_fidelities:
            min_fidelity *= gate_fid.min
            max_fidelity *= gate_fid.max

        return min_fidelity, max_fidelity

    def run(self, shots: int = 1, with_noise: bool = True) -> Result[RetType]:
        """Run the kernel and get simulation results.

        When ``no_measurements=True``, the detector sampler is used instead of
        the full measurement sampler for improved performance. In this mode the
        returned ``Result`` provides ``detectors`` and ``observables`` but
        accessing ``measurements`` or ``return_values`` will raise ``ValueError``.

        Args:
            shots (int): Number of shots to run. Defaults to 1.
            with_noise (bool): Whether to include noise in the simulation. Defaults to True.

        Returns:
            Result: The simulation result including detector error model, fidelity bounds,
                and sampling outcomes.

        """
        fidelity_min, fidelity_max = self.fidelity_bounds()

        if self.no_measurements:
            sampler = (
                self.detector_sampler if with_noise else self.noiseless_detector_sampler
            )
            det_obs: tuple[np.ndarray, np.ndarray] = sampler.sample(
                shots=shots, separate_observables=True
            )
            detectors = det_obs[0].tolist()
            observables = det_obs[1].tolist()
            return Result(
                _detector_error_model=self.detector_error_model,
                _fidelity_min=fidelity_min,
                _fidelity_max=fidelity_max,
                _detectors=detectors,
                _observables=observables,
            )

        if with_noise:
            raw_results = self.measurement_sampler.sample(shots=shots).tolist()
        else:
            raw_results = self.noiseless_measurement_sampler.sample(
                shots=shots
            ).tolist()

        return Result(
            _detector_error_model=self.detector_error_model,
            _fidelity_min=fidelity_min,
            _fidelity_max=fidelity_max,
            _raw_measurements=raw_results,
            _post_processing=self._post_processing,
        )

    def run_async(
        self, shots: int = 1, with_noise: bool = True
    ) -> Future[Result[RetType]]:
        """Run the kernel asynchronously and get simulation results.

        Args:
            shots (int): Number of shots to run. Defaults to 1.
            with_noise (bool): Whether to include noise in the simulation. Defaults to True.

        Returns:
            Future[Result]: A future that will resolve to the simulation result including
                measurement outcomes, detector error model, post-processing, and fidelity bounds.
        """

        def _runner(
            task: GeminiLogicalSimulatorTask[RetType], shots: int, with_noise: bool
        ) -> Result[RetType]:
            return task.run(shots, with_noise)

        return self._thread_pool_executor.submit(_runner, self, shots, with_noise)


@dataclass
class GeminiLogicalSimulator:
    noise_model: NoiseModelABC = field(default_factory=generate_simple_noise_model)
    no_measurements: bool = False
    """When True, skip the measurement sampler and use the detector sampler instead.
    The kernel must have a None return type when this is enabled."""

    def task(
        self, logical_squin_kernel: ir.Method[[], RetType]
    ) -> GeminiLogicalSimulatorTask[RetType]:
        """Create a simulation task for the given kernel.

        Args:
            logical_squin_kernel: The logical squin kernel to compile and run.

        Raises:
            ValueError: If ``no_measurements=True`` and the kernel return type
                is not ``None``.

        """
        run_squin_kernel_validation(logical_squin_kernel).raise_if_invalid()
        if self.no_measurements:
            from kirin import types

            if not logical_squin_kernel.return_type.is_structurally_equal(
                types.NoneType
            ):
                raise ValueError(
                    "Kernel must have a None return type when " "`no_measurements=True`"
                )
        return GeminiLogicalSimulatorTask(
            logical_squin_kernel,
            self.noise_model,
            no_measurements=self.no_measurements,
        )

    def run(
        self,
        logical_squin_kernel: ir.Method[[], RetType],
        shots: int = 1,
        with_noise: bool = True,
    ) -> Result[RetType]:
        return self.task(logical_squin_kernel).run(shots, with_noise)

    def run_async(
        self,
        logical_squin_kernel: ir.Method[[], RetType],
        shots: int = 1,
        with_noise: bool = True,
    ) -> Future[Result[RetType]]:
        return self.task(logical_squin_kernel).run_async(shots, with_noise)

    def visualize(
        self,
        logical_squin_kernel: ir.Method[[], RetType],
        animated: bool = False,
        interactive: bool = True,
    ):
        """Visualize the physical move kernel using the built-in debugger.

        Args
            logical_squin_kernel (ir.Method): The logical squin kernel to visualize.
            animated (bool): Whether to use the animated debugger. Defaults to False.
            interactive (bool): Whether to enable interactive mode. Defaults to True.

        """
        self.task(logical_squin_kernel).visualize(
            animated=animated, interactive=interactive
        )

    def physical_squin_kernel(
        self, logical_squin_kernel: ir.Method[[], RetType]
    ) -> ir.Method[[], RetType]:
        """Compile the logical squin kernel to the physical squin kernel."""
        return self.task(logical_squin_kernel).physical_squin_kernel

    def physical_move_kernel(
        self, logical_squin_kernel: ir.Method[[], RetType]
    ) -> ir.Method[[], RetType]:
        """Compile the logical squin kernel to the physical move kernel."""
        return self.task(logical_squin_kernel).physical_move_kernel

    def tsim_circuit(
        self, logical_squin_kernel: ir.Method[[], RetType], with_noise: bool = True
    ) -> tsim_backend.Circuit:
        """Compile the logical squin kernel to the tsim circuit.

        Args:
            logical_squin_kernel (ir.Method): The logical squin kernel to compile.
            with_noise (bool): Whether to include noise in the tsim circuit. Defaults to True.

        """
        if with_noise:
            return self.task(logical_squin_kernel).tsim_circuit
        else:
            return self.task(logical_squin_kernel).noiseless_tsim_circuit

    def fidelity_bounds(
        self, logical_squin_kernel: ir.Method[[], RetType]
    ) -> tuple[float, float]:
        """Get the fidelity bounds for the logical squin kernel."""
        return self.task(logical_squin_kernel).fidelity_bounds()
