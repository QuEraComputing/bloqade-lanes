from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, TypeVar, Union, cast

from kirin import ir

from .abstract_simulator import (
    AbstractSimulator,
    CliffTSimulatorTask,
    DetectorResult as DetectorResult,
    Result as Result,
    TsimSimulatorTask,
)

RetType = TypeVar("RetType")


@dataclass(frozen=True)
class GeminiLogicalSimulatorTask(TsimSimulatorTask[RetType]):
    """A compiled simulation task for the Gemini logical simulator.

    Created by :meth:`GeminiLogicalSimulator.task`. The squin-to-move compilation
    and post-processing extraction are performed eagerly at construction time.
    Simulation artifacts (physical squin kernel, stim circuits, samplers, detector
    error model) are computed lazily on first access since they depend on the
    noise model.
    """

    pass


@dataclass(frozen=True)
class GeminiLogicalCliffTSimulatorTask(CliffTSimulatorTask[RetType]):
    """A CliffT-backed compiled simulation task for Gemini logical programs."""

    pass


@dataclass
class GeminiLogicalSimulator(AbstractSimulator):
    """Logical simulator targeting the Gemini neutral-atom architecture.

    This is the primary entry point for compiling and simulating logical quantum
    circuits on the Gemini architecture. Use :meth:`task` to compile a kernel into
    a reusable :class:`GeminiLogicalSimulatorTask`, or :meth:`run` for one-shot
    compile-and-execute convenience.
    """

    backend: str = "tsim"
    """Deprecated compatibility backend selector. Prefer GeminiLogicalCliffTSimulator."""

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
        from bloqade.lanes.logical_mvp import compile_task

        (
            logical_squin_kernel,
            physical_arch_spec,
            physical_move_kernel,
            post_processing,
        ) = compile_task(logical_kernel, m2dets, m2obs)

        # NOTE: kept for backwards compatibility only
        if self.backend == "clifft":
            from bloqade.gemini.decoding.tasks import _CliffTSimulatorTask

            return cast(
                GeminiLogicalSimulatorTask[RetType],
                _CliffTSimulatorTask(
                    logical_squin_kernel,
                    self.noise_model,
                    physical_arch_spec,
                    physical_move_kernel,
                    post_processing,
                    seed=self.seed,
                ),
            )
        if self.backend != "tsim":
            raise ValueError("backend must be either 'tsim' or 'clifft'.")

        return GeminiLogicalSimulatorTask(
            logical_squin_kernel,
            self.noise_model,
            physical_arch_spec,
            physical_move_kernel,
            post_processing,
            seed=self.seed,
        )


@dataclass
class GeminiLogicalCliffTSimulator(AbstractSimulator):
    """CliffT-backed logical simulator targeting the Gemini architecture."""

    def task(
        self,
        logical_kernel: Union[ir.Method[[], RetType], Callable[..., Any]],
        m2dets: list[list[int]] | None = None,
        m2obs: list[list[int]] | None = None,
    ) -> GeminiLogicalCliffTSimulatorTask[RetType]:
        """Create a CliffT-backed simulation task for the given logical kernel.

        Args:
            logical_kernel (Union[ir.Method[[], RetType], Callable[..., Any]]): The logical
                squin or CUDA-Q kernel to compile and run.
            m2dets (list[list[int]] | None): Binary measurement-to-detector matrix.
            m2obs (list[list[int]] | None): Binary measurement-to-observable matrix.

        Returns:
            GeminiLogicalCliffTSimulatorTask[RetType]: The compiled simulation task.

        """
        from bloqade.lanes.logical_mvp import compile_task

        (
            logical_squin_kernel,
            physical_arch_spec,
            physical_move_kernel,
            post_processing,
        ) = compile_task(logical_kernel, m2dets, m2obs)

        return GeminiLogicalCliffTSimulatorTask(
            logical_squin_kernel,
            self.noise_model,
            physical_arch_spec,
            physical_move_kernel,
            post_processing,
            seed=self.seed,
        )
