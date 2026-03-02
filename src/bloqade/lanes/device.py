from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cached_property
from typing import Generic, TypeVar

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
    """Simulation result including measurement outcomes, detector error model, post-processing, and fidelity bounds."""

    _raw_measurements: list[list[bool]]
    _detector_error_model: DetectorErrorModel
    _post_processing: atom.PostProcessing[RetType]
    _fidelity_min: float
    _fidelity_max: float

    def fidelity_bounds(self) -> tuple[float, float]:
        """Return the upper and lower fidelity bounds.

        Note: The upper and lower bounds are related to and branching logic in the kernel.

        """
        return (self._fidelity_min, self._fidelity_max)

    @property
    def detector_error_model(self) -> DetectorErrorModel:
        """The STIM detector error model corresponding to the physical noise circuit."""
        return self._detector_error_model

    @cached_property
    def return_values(self) -> list[RetType]:
        """The return values of the logical kernel"""
        return list(self._post_processing.emit_return(self._raw_measurements))

    @cached_property
    def detectors(self) -> list[list[bool]]:
        """The detector outcomes from the simulation."""
        return list(self._post_processing.emit_detectors(self._raw_measurements))

    @cached_property
    def measurements(self) -> list[list[bool]]:
        """The raw measurement outcomes used to compute detectors and observables."""
        return list(self._raw_measurements)

    @cached_property
    def observables(self) -> list[list[bool]]:
        """The observable outcomes from the simulation."""
        return list(self._post_processing.emit_observables(self._raw_measurements))


@dataclass(frozen=True)
class GeminiLogicalSimulatorTask(Generic[RetType]):
    logical_squin_kernel: ir.Method[[], RetType]
    """The input logical squin kernel to be executed on the Gemini architecture."""
    noise_model: NoiseModelABC
    """The noise model to be inserted into the physical squin kernel."""
    _thread_pool_executor: ThreadPoolExecutor = field(
        default_factory=ThreadPoolExecutor, init=False
    )

    def __post_init__(self):
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

        Args:
            shots (int): Number of shots to run. Defaults to 1.
            with_noise (bool): Whether to include noise in the simulation. Defaults to True.

        Returns:
            Result: The simulation result including measurement outcomes, detector error model, post-processing, and fidelity

        """

        if with_noise:
            raw_results = self.measurement_sampler.sample(shots=shots).tolist()
        else:
            raw_results = self.noiseless_measurement_sampler.sample(
                shots=shots
            ).tolist()

        return Result(
            raw_results,
            self.detector_error_model,
            self._post_processing,
            *self.fidelity_bounds(),
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

    def task(
        self, logical_squin_kernel: ir.Method[[], RetType]
    ) -> GeminiLogicalSimulatorTask[RetType]:
        run_squin_kernel_validation(logical_squin_kernel).raise_if_invalid()
        return GeminiLogicalSimulatorTask(
            logical_squin_kernel,
            self.noise_model,
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
