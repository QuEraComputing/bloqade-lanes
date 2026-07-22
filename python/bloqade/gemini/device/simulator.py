from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass, field
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Generic,
    TypeVar,
)

from kirin import ir

from ._task_runtime import (
    DetectorResult as DetectorResult,
    Result as Result,
    SimulatorResult,
    _SimulatorTaskBase,
)
from .simulator_backend import AbstractSimulatorBackend, TsimSimulatorBackend

if TYPE_CHECKING:
    import tsim as tsim_backend  # type: ignore[reportMissingImports]

    from bloqade.lanes.analysis import atom
    from bloqade.lanes.arch.spec import ArchSpec
    from bloqade.lanes.rewrite.move2squin.noise import LogicalNoiseModelABC

RetType = TypeVar("RetType")


def _default_noise_model() -> LogicalNoiseModelABC:
    from bloqade.lanes.noise_model import generate_logical_noise_model

    return generate_logical_noise_model()


@dataclass(frozen=True)
class GeminiLogicalSimulatorTask(_SimulatorTaskBase[RetType], Generic[RetType]):
    """A compiled simulation task for the Gemini logical simulator.

    Created by :meth:`GeminiLogicalSimulator.task`. The squin-to-move compilation
    and post-processing extraction are performed eagerly at construction time.
    Simulation artifacts (physical squin kernel, stim circuits, samplers, detector
    error model) are computed lazily on first access since they depend on the
    noise model.
    """

    logical_squin_kernel: ir.Method[[], RetType]
    """The input logical squin kernel to be executed on the Gemini architecture."""
    noise_model: LogicalNoiseModelABC
    """The noise model to be inserted into the physical squin kernel."""
    physical_arch_spec: ArchSpec = field(repr=False)
    """The physical architecture specification."""
    physical_move_kernel: ir.Method[[], RetType] = field(repr=False)
    """The physical move kernel that executes the logical squin kernel on the physical architecture."""
    _post_processing: atom.PostProcessing[RetType] = field(repr=False)
    """The post-processing object for extracting detectors, observables, and return values."""
    backend: AbstractSimulatorBackend = field(default_factory=TsimSimulatorBackend)

    @cached_property
    def physical_squin_kernel(self) -> ir.Method[[], RetType]:
        """The physical squin kernel with noise channels."""
        from bloqade.lanes.transform import MoveToSquinLogical

        return MoveToSquinLogical(
            arch_spec=self.physical_arch_spec,
            noise_model=self.noise_model,
            add_noise=True,
        ).emit(self.physical_move_kernel)

    @cached_property
    def noiseless_physical_squin_kernel(self) -> ir.Method[[], RetType]:
        """The physical squin kernel without noise channels."""
        from bloqade.lanes.transform import MoveToSquinLogical

        return MoveToSquinLogical(
            arch_spec=self.physical_arch_spec,
            noise_model=self.noise_model,
            add_noise=False,
        ).emit(self.physical_move_kernel)


@dataclass
class GeminiLogicalSimulator:
    """This is the primary entry point for compiling and simulating logical quantum
    circuits on the Gemini architecture. Use :meth:`task` to compile a kernel into
    a reusable :class:`GeminiLogicalSimulatorTask`, or :meth:`run` for one-shot
    compile-and-execute convenience.

    CUDA-Q users must convert kernels and add any desired detector/observable
    annotations externally.
    """

    noise_model: LogicalNoiseModelABC = field(default_factory=_default_noise_model)
    """The noise model used for simulation. Defaults to :func:`generate_logical_noise_model`."""
    backend: AbstractSimulatorBackend = field(default_factory=TsimSimulatorBackend)
    """Sampling backend for tasks created by this simulator."""

    def task(
        self,
        logical_kernel: ir.Method[[], RetType],
    ) -> GeminiLogicalSimulatorTask[RetType]:
        """Create a simulation task for the given kernel.

        Eagerly compiles the kernel through squin-to-move and extracts post-processing.
        Kernels are expected to be SQuIN kernels (for example, CUDA-Q kernels are expected
        to be first converted to SQuIN).

        CUDA-Q conversion and any desired detector/observable annotations must be
        completed externally.

        Args:
            logical_kernel (ir.Method[[], RetType]): The logical
                squin kernel to compile and run.

        Returns:
            GeminiLogicalSimulatorTask[RetType]: The compiled simulation task.
        """
        if not isinstance(logical_kernel, ir.Method):
            raise TypeError("GeminiLogicalSimulator.task() requires a Squin ir.Method")

        from bloqade.lanes.logical_mvp import compile_task

        (
            logical_squin_kernel,
            physical_arch_spec,
            physical_move_kernel,
            post_processing,
        ) = compile_task(logical_kernel)
        return GeminiLogicalSimulatorTask(
            logical_squin_kernel,
            self.noise_model,
            physical_arch_spec,
            physical_move_kernel,
            post_processing,
            self.backend,
        )

    def run(
        self,
        logical_kernel: ir.Method[[], RetType],
        shots: int = 1,
        with_noise: bool = True,
    ) -> SimulatorResult[RetType]:
        """Run the kernel and get simulation results.

        Args:
            logical_kernel (ir.Method[[], RetType]): The logical squin kernel to run.
            shots (int): Number of shots to run. Defaults to 1.
            with_noise (bool): Whether to include noise in the simulation. Defaults to True.

        Returns:
            SimulatorResult[RetType]: The full simulation result.

        """
        return self.task(logical_kernel).run(shots, with_noise)

    def run_async(
        self,
        logical_kernel: ir.Method[[], RetType],
        shots: int = 1,
        with_noise: bool = True,
    ) -> Future[SimulatorResult[RetType]]:
        """Run the kernel asynchronously and get simulation results.

        Args:
            logical_kernel (ir.Method[[], RetType]): The logical squin kernel to run.
            shots (int): Number of shots to run. Defaults to 1.
            with_noise (bool): Whether to include noise in the simulation. Defaults to True.

        Returns:
            Future[SimulatorResult[RetType]]: A future resolving to the full simulation result.

        """
        return self.task(logical_kernel).run_async(shots, with_noise)

    def visualize(
        self,
        logical_kernel: ir.Method[[], RetType],
        animated: bool = False,
        interactive: bool = True,
    ):
        """Visualize the physical move kernel using the built-in debugger.

        Args:
            logical_kernel (ir.Method[[], RetType]): The logical squin kernel to visualize.
            animated (bool): Whether to use the animated debugger. Defaults to False.
            interactive (bool): Whether to enable interactive mode. Defaults to True.

        """
        self.task(logical_kernel).visualize(animated=animated, interactive=interactive)

    def physical_squin_kernel(
        self, logical_kernel: ir.Method[[], RetType]
    ) -> ir.Method[[], RetType]:
        """Compile the logical squin kernel to the physical squin kernel.

        Args:
            logical_kernel (ir.Method[[], RetType]): The logical squin kernel to compile.

        Returns:
            ir.Method[[], RetType]: The physical squin kernel.

        """
        return self.task(logical_kernel).physical_squin_kernel

    def physical_move_kernel(
        self, logical_kernel: ir.Method[[], RetType]
    ) -> ir.Method[[], RetType]:
        """Compile the logical squin kernel to the physical move kernel.

        Args:
            logical_kernel (ir.Method[[], RetType]): The logical squin kernel to compile.

        Returns:
            ir.Method[[], RetType]: The physical move kernel.

        """
        return self.task(logical_kernel).physical_move_kernel

    def tsim_circuit(
        self, logical_kernel: ir.Method[[], RetType], with_noise: bool = True
    ) -> tsim_backend.Circuit:
        """Compile the logical squin kernel to the tsim circuit.

        Args:
            logical_kernel (ir.Method[[], RetType]): The logical squin kernel to compile.
            with_noise (bool): Whether to include noise in the tsim circuit. Defaults to True.

        Returns:
            tsim.Circuit: The compiled tsim circuit.

        """
        task = self.task(logical_kernel)
        if with_noise:
            return task.tsim_circuit
        return task.noiseless_tsim_circuit

    def fidelity_bounds(
        self, logical_kernel: ir.Method[[], RetType]
    ) -> tuple[float, float]:
        """Get the fidelity bounds for the logical squin kernel.

        Args:
            logical_kernel (ir.Method[[], RetType]): The logical squin kernel to analyze.

        Returns:
            tuple[float, float]: The (min, max) fidelity bounds.

        """
        return self.task(logical_kernel).fidelity_bounds()
