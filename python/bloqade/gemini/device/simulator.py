from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass, field
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Literal,
    TypeVar,
    Union,
    overload,
)

from kirin import ir

from ._task_runtime import (
    DetectorResult,
    Result,
    _SimulatorTaskBase,
)
from .simulator_backend import AbstractSimulatorBackend, TsimSimulatorBackend

if TYPE_CHECKING:
    import tsim as tsim_backend  # type: ignore[reportMissingImports]

    from bloqade.lanes.analysis import atom
    from bloqade.lanes.arch.spec import ArchSpec
    from bloqade.lanes.rewrite.move2squin.noise import LogicalNoiseModelABC

RetType = TypeVar("RetType")
LogicalKernel = Union[ir.Method[[], RetType], Callable[..., Any]]


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
    simulator: GeminiLogicalSimulator = field(repr=False)

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
    """Logical Gemini simulator configured with one reusable backend."""

    noise_model: LogicalNoiseModelABC = field(default_factory=_default_noise_model)
    backend: AbstractSimulatorBackend = field(default_factory=TsimSimulatorBackend)
    m2dets: list[list[int]] | None = None
    m2obs: list[list[int]] | None = None

    def task(
        self,
        logical_kernel: LogicalKernel[RetType],
    ) -> GeminiLogicalSimulatorTask[RetType]:
        """Compile a logical SQuIn or CUDA-Q kernel into a reusable task."""
        from bloqade.lanes.logical_mvp import compile_task

        (
            logical_squin_kernel,
            physical_arch_spec,
            physical_move_kernel,
            post_processing,
        ) = compile_task(logical_kernel, self.m2dets, self.m2obs)
        return GeminiLogicalSimulatorTask(
            logical_squin_kernel,
            self.noise_model,
            physical_arch_spec,
            physical_move_kernel,
            post_processing,
            self,
        )

    @overload
    def run(
        self,
        logical_kernel: LogicalKernel[RetType],
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: Literal[False] = ...,
    ) -> Result[RetType]: ...

    @overload
    def run(
        self,
        logical_kernel: LogicalKernel[RetType],
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: Literal[True],
    ) -> DetectorResult: ...

    @overload
    def run(
        self,
        logical_kernel: LogicalKernel[RetType],
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: bool,
    ) -> Result[RetType] | DetectorResult: ...

    def run(
        self,
        logical_kernel: LogicalKernel[RetType],
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: bool = False,
    ) -> Result[RetType] | DetectorResult:
        return self.task(logical_kernel).run(
            shots, with_noise, run_detectors=run_detectors
        )

    @overload
    def run_async(
        self,
        logical_kernel: LogicalKernel[RetType],
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: Literal[False] = ...,
    ) -> Future[Result[RetType]]: ...

    @overload
    def run_async(
        self,
        logical_kernel: LogicalKernel[RetType],
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: Literal[True],
    ) -> Future[DetectorResult]: ...

    @overload
    def run_async(
        self,
        logical_kernel: LogicalKernel[RetType],
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: bool,
    ) -> Future[Result[RetType]] | Future[DetectorResult]: ...

    def run_async(
        self,
        logical_kernel: LogicalKernel[RetType],
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: bool = False,
    ) -> Future[Result[RetType]] | Future[DetectorResult]:
        task = self.task(logical_kernel)
        if run_detectors:
            return task.run_async(shots, with_noise, run_detectors=True)
        return task.run_async(shots, with_noise, run_detectors=False)

    def visualize(
        self,
        logical_kernel: LogicalKernel[RetType],
        animated: bool = False,
        interactive: bool = True,
    ):
        self.task(logical_kernel).visualize(animated=animated, interactive=interactive)

    def physical_squin_kernel(
        self, logical_kernel: LogicalKernel[RetType]
    ) -> ir.Method[[], RetType]:
        return self.task(logical_kernel).physical_squin_kernel

    def physical_move_kernel(
        self, logical_kernel: LogicalKernel[RetType]
    ) -> ir.Method[[], RetType]:
        return self.task(logical_kernel).physical_move_kernel

    def tsim_circuit(
        self, logical_kernel: LogicalKernel[RetType], with_noise: bool = True
    ) -> tsim_backend.Circuit:
        task = self.task(logical_kernel)
        if with_noise:
            return task.tsim_circuit
        return task.noiseless_tsim_circuit

    def fidelity_bounds(
        self, logical_kernel: LogicalKernel[RetType]
    ) -> tuple[float, float]:
        return self.task(logical_kernel).fidelity_bounds()
