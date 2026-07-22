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
    """Compiled logical task with lazy logical-to-physical lowering."""

    logical_squin_kernel: ir.Method[[], RetType]
    noise_model: LogicalNoiseModelABC
    physical_arch_spec: ArchSpec = field(repr=False)
    physical_move_kernel: ir.Method[[], RetType] = field(repr=False)
    _post_processing: atom.PostProcessing[RetType] = field(repr=False)
    backend: AbstractSimulatorBackend = field(default_factory=TsimSimulatorBackend)

    @cached_property
    def physical_squin_kernel(self) -> ir.Method[[], RetType]:
        from bloqade.lanes.transform import MoveToSquinLogical

        return MoveToSquinLogical(
            arch_spec=self.physical_arch_spec,
            noise_model=self.noise_model,
            add_noise=True,
        ).emit(self.physical_move_kernel)

    @cached_property
    def noiseless_physical_squin_kernel(self) -> ir.Method[[], RetType]:
        from bloqade.lanes.transform import MoveToSquinLogical

        return MoveToSquinLogical(
            arch_spec=self.physical_arch_spec,
            noise_model=self.noise_model,
            add_noise=False,
        ).emit(self.physical_move_kernel)


@dataclass
class GeminiLogicalSimulator:
    """Simulator for caller-prepared logical Squin ``ir.Method`` kernels.

    CUDA-Q users must convert kernels and add any desired detector/observable
    annotations externally.
    """

    noise_model: LogicalNoiseModelABC = field(default_factory=_default_noise_model)
    backend: AbstractSimulatorBackend = field(default_factory=TsimSimulatorBackend)

    def task(
        self,
        logical_kernel: ir.Method[[], RetType],
    ) -> GeminiLogicalSimulatorTask[RetType]:
        """Compile a caller-prepared logical Squin ``ir.Method`` into a task.

        CUDA-Q conversion and any desired detector/observable annotations must be
        completed externally.
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
        *,
        seed: int | None = None,
    ) -> SimulatorResult[RetType]:
        return self.task(logical_kernel).run(shots, with_noise, seed=seed)

    def run_async(
        self,
        logical_kernel: ir.Method[[], RetType],
        shots: int = 1,
        with_noise: bool = True,
        *,
        seed: int | None = None,
    ) -> Future[SimulatorResult[RetType]]:
        return self.task(logical_kernel).run_async(shots, with_noise, seed=seed)

    def visualize(
        self,
        logical_kernel: ir.Method[[], RetType],
        animated: bool = False,
        interactive: bool = True,
    ):
        self.task(logical_kernel).visualize(animated=animated, interactive=interactive)

    def physical_squin_kernel(
        self, logical_kernel: ir.Method[[], RetType]
    ) -> ir.Method[[], RetType]:
        return self.task(logical_kernel).physical_squin_kernel

    def physical_move_kernel(
        self, logical_kernel: ir.Method[[], RetType]
    ) -> ir.Method[[], RetType]:
        return self.task(logical_kernel).physical_move_kernel

    def tsim_circuit(
        self, logical_kernel: ir.Method[[], RetType], with_noise: bool = True
    ) -> tsim_backend.Circuit:
        task = self.task(logical_kernel)
        if with_noise:
            return task.tsim_circuit
        return task.noiseless_tsim_circuit

    def fidelity_bounds(
        self, logical_kernel: ir.Method[[], RetType]
    ) -> tuple[float, float]:
        return self.task(logical_kernel).fidelity_bounds()
