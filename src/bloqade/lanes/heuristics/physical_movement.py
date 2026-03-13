from dataclasses import dataclass, field

from bloqade.lanes import layout
from bloqade.lanes.analysis.placement import (
    ConcreteState,
    SingleZonePlacementStrategyABC,
)
from bloqade.lanes.arch.gemini.physical import get_arch_spec as get_physical_arch_spec


@dataclass
class PhysicalGreedyPlacementStrategy(SingleZonePlacementStrategyABC):
    """Temporary no-op physical placement placeholder.

    This keeps the physical placement wiring importable while movement
    synthesis is split into a follow-up PR.
    """

    arch_spec: layout.ArchSpec = field(default_factory=get_physical_arch_spec)

    def validate_initial_layout(
        self,
        initial_layout: tuple[layout.LocationAddress, ...],
    ) -> None:
        _ = initial_layout

    def desired_cz_layout(
        self,
        state: ConcreteState,
        controls: tuple[int, ...],
        targets: tuple[int, ...],
    ) -> ConcreteState:
        _ = controls, targets
        return state

    def compute_moves(
        self,
        state_before: ConcreteState,
        state_after: ConcreteState,
    ) -> tuple[tuple[layout.LaneAddress, ...], ...]:
        _ = state_before, state_after
        return ()
