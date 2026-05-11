import abc
from dataclasses import dataclass

from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode.encoding import LaneAddress, LocationAddress, ZoneAddress

from .lattice import AtomState, ConcreteState, ExecuteCZ, ExecuteMeasure


def assert_single_cz_zone(arch_spec: ArchSpec, strategy_name: str) -> None:
    """Validate that the arch has exactly one CZ-capable zone.

    The current placement strategies emit CZ pulses in every zone with
    ``entangling_pairs`` (via ``ArchSpec.cz_zone_addresses``), which is
    correct only when there is one such zone. Multi-zone CZ scheduling
    requires deriving the active zones from the specific (controls,
    targets) of each CZ layer and is not yet implemented; until then we
    fail loudly at construction time so callers don't silently get extra
    CZ pulses on unrelated zones.
    """
    cz_zones = arch_spec.cz_zone_addresses
    if len(cz_zones) != 1:
        zone_ids = sorted(z.zone_id for z in cz_zones)
        raise ValueError(
            f"{strategy_name} requires exactly one CZ-capable zone, "
            f"found {len(cz_zones)} (zone ids: {zone_ids}). "
            "Multi-zone CZ scheduling is not yet supported by this strategy."
        )


@dataclass
class PlacementStrategyABC(abc.ABC):

    arch_spec: ArchSpec

    @abc.abstractmethod
    def validate_initial_layout(
        self,
        initial_layout: tuple[LocationAddress, ...],
    ) -> None: ...

    @abc.abstractmethod
    def cz_placements(
        self,
        state: AtomState,
        controls: tuple[int, ...],
        targets: tuple[int, ...],
        lookahead_cz_layers: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = (),
    ) -> AtomState: ...

    @abc.abstractmethod
    def sq_placements(
        self,
        state: AtomState,
        qubits: tuple[int, ...],
    ) -> AtomState: ...

    @abc.abstractmethod
    def measure_placements(
        self,
        state: AtomState,
        qubits: tuple[int, ...],
    ) -> AtomState: ...


class SingleZonePlacementStrategyABC(PlacementStrategyABC):

    def __post_init__(self) -> None:
        assert_single_cz_zone(self.arch_spec, type(self).__name__)

    @abc.abstractmethod
    def compute_moves(
        self,
        state_before: ConcreteState,
        state_after: ConcreteState,
    ) -> tuple[tuple[LaneAddress, ...], ...]: ...

    @abc.abstractmethod
    def desired_cz_layout(
        self,
        state: ConcreteState,
        controls: tuple[int, ...],
        targets: tuple[int, ...],
    ) -> ConcreteState: ...

    def cz_placements(
        self,
        state: AtomState,
        controls: tuple[int, ...],
        targets: tuple[int, ...],
        lookahead_cz_layers: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = (),
    ) -> AtomState:
        _ = lookahead_cz_layers
        if len(controls) != len(targets) or state == AtomState.bottom():
            return AtomState.bottom()

        if not isinstance(state, ConcreteState):
            return AtomState.top()

        desired_state = self.desired_cz_layout(state, controls, targets)
        move_layers = self.compute_moves(state, desired_state)

        return ExecuteCZ(
            occupied=state.occupied,
            layout=desired_state.layout,
            move_count=tuple(
                mc + int(src != dst)
                for mc, src, dst in zip(
                    state.move_count, state.layout, desired_state.layout
                )
            ),
            active_cz_zones=self.arch_spec.cz_zone_addresses,
            move_layers=move_layers,
        )

    def sq_placements(self, state: AtomState, qubits: tuple[int, ...]) -> AtomState:
        if isinstance(state, ConcreteState):
            # Strip CZ-specific metadata so non-CZ statements do not inherit stale move layers in downstream rewrite passes.
            return ConcreteState(
                occupied=state.occupied,
                layout=state.layout,
                move_count=state.move_count,
            )
        return state  # No movement needed for single zone

    def measure_placements(
        self, state: AtomState, qubits: tuple[int, ...]
    ) -> AtomState:
        if not isinstance(state, ConcreteState):
            return state

        # all qubits must be measured
        if len(qubits) != len(state.layout):
            return AtomState.bottom()

        return ExecuteMeasure(
            occupied=state.occupied,
            layout=state.layout,
            move_count=state.move_count,
            zone_maps=tuple(ZoneAddress(loc.zone_id) for loc in state.layout),
        )
