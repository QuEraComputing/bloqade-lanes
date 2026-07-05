from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeVar

from kirin import ir, passes

from .abstract_simulator import (
    AbstractSimulator,
    CliffTSimulatorTask,
    Result,
    TsimSimulatorTask,
)

if TYPE_CHECKING:
    from bloqade.lanes.analysis import atom
    from bloqade.lanes.analysis.placement import PlacementStrategyABC
    from bloqade.lanes.arch.spec import ArchSpec

RetType = TypeVar("RetType")
PhysicalResult = Result


def _default_arch_spec() -> "ArchSpec":
    from bloqade.lanes.arch.gemini.physical import get_arch_spec

    return get_arch_spec()


@dataclass(frozen=True)
class PhysicalSimulatorTask(TsimSimulatorTask[RetType]):
    """A compiled simulation task for physical SQuIn programs."""

    @property
    def source_squin_kernel(self) -> ir.Method[[], RetType]:
        """The input physical SQuIn kernel."""
        return self.logical_squin_kernel


@dataclass(frozen=True)
class PhysicalCliffTSimulatorTask(CliffTSimulatorTask[RetType]):
    """A CliffT-backed compiled simulation task for physical SQuIn programs."""

    @property
    def source_squin_kernel(self) -> ir.Method[[], RetType]:
        """The input physical SQuIn kernel."""
        return self.logical_squin_kernel


@dataclass
class PhysicalSimulator(AbstractSimulator):
    """Simulator for programs written directly at the physical SQuIn level."""

    arch_spec: ArchSpec = field(default_factory=_default_arch_spec)
    """The physical architecture specification used for compilation."""

    def task(
        self,
        physical_kernel: ir.Method[[], RetType],
        place_opt_type: type[passes.Pass] | None = None,
        placement_strategy: "PlacementStrategyABC | None" = None,
    ) -> PhysicalSimulatorTask[RetType]:
        """Compile a physical SQuIn kernel into a reusable simulation task.

        Args:
            physical_kernel (ir.Method[[], RetType]): The physical SQuIn kernel to compile.
            place_opt_type (type[passes.Pass] | None): Optional placement pass class.
            placement_strategy (PlacementStrategyABC | None): Optional placement strategy.

        Returns:
            PhysicalSimulatorTask[RetType]: The compiled simulation task.

        """
        physical_move_kernel, post_processing = self._compile_physical_task(
            physical_kernel, place_opt_type, placement_strategy
        )
        return PhysicalSimulatorTask(
            physical_kernel,
            self.noise_model,
            self.arch_spec,
            physical_move_kernel,
            post_processing,
            seed=self.seed,
        )

    def _compile_physical_task(
        self,
        physical_kernel: ir.Method[[], RetType],
        place_opt_type: type[passes.Pass] | None = None,
        placement_strategy: "PlacementStrategyABC | None" = None,
    ) -> tuple[ir.Method[[], RetType], "atom.PostProcessing[RetType]"]:
        """Compile a physical SQuIn kernel and extract post-processing."""
        from bloqade.lanes.analysis import atom
        from bloqade.lanes.passes import SequentialPlacePass
        from bloqade.lanes.pipeline import PhysicalPipeline

        if place_opt_type is None:
            place_opt_type = SequentialPlacePass

        physical_pipeline = PhysicalPipeline(
            arch_spec=self.arch_spec,
            place_opt_type=place_opt_type,
            placement_strategy=placement_strategy,
        )
        physical_move_kernel = physical_pipeline.emit(physical_kernel, no_raise=False)
        post_processing = atom.AtomInterpreter(
            physical_move_kernel.dialects, arch_spec=self.arch_spec
        ).get_post_processing(physical_move_kernel)
        return physical_move_kernel, post_processing


@dataclass
class PhysicalCliffTSimulator(AbstractSimulator):
    """CliffT-backed simulator for physical SQuIn programs."""

    arch_spec: ArchSpec = field(default_factory=_default_arch_spec)
    """The physical architecture specification used for compilation."""

    def task(
        self,
        physical_kernel: ir.Method[[], RetType],
        place_opt_type: type[passes.Pass] | None = None,
        placement_strategy: "PlacementStrategyABC | None" = None,
    ) -> PhysicalCliffTSimulatorTask[RetType]:
        """Compile a physical SQuIn kernel into a reusable CliffT-backed task.

        Args:
            physical_kernel (ir.Method[[], RetType]): The physical SQuIn kernel to compile.
            place_opt_type (type[passes.Pass] | None): Optional placement pass class.
            placement_strategy (PlacementStrategyABC | None): Optional placement strategy.

        Returns:
            PhysicalCliffTSimulatorTask[RetType]: The compiled simulation task.

        """
        physical_move_kernel, post_processing = self._compile_physical_task(
            physical_kernel, place_opt_type, placement_strategy
        )
        return PhysicalCliffTSimulatorTask(
            physical_kernel,
            self.noise_model,
            self.arch_spec,
            physical_move_kernel,
            post_processing,
            seed=self.seed,
        )

    def _compile_physical_task(
        self,
        physical_kernel: ir.Method[[], RetType],
        place_opt_type: type[passes.Pass] | None = None,
        placement_strategy: "PlacementStrategyABC | None" = None,
    ) -> tuple[ir.Method[[], RetType], "atom.PostProcessing[RetType]"]:
        """Compile a physical kernel using the shared physical simulator helper."""
        return PhysicalSimulator(
            noise_model=self.noise_model, arch_spec=self.arch_spec
        )._compile_physical_task(physical_kernel, place_opt_type, placement_strategy)
