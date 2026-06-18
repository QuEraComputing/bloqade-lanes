import abc
from dataclasses import dataclass

from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode.encoding import LaneAddress, LocationAddress, ZoneAddress

from .lattice import (
    AtomState,
    ConcreteState,
    ExecuteCZ,
    ExecuteCZReturn,
    ExecuteMeasure,
    UserMoved,
)


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

    def _strip_user_moved(self, state: AtomState) -> AtomState:
        """Strip UserMoved back to plain ConcreteState.

        Non-palindrome strategies call this in sq_placements so that
        accumulated_move_layers and pre_user_layout (only needed for the
        palindrome return at the next CZ) don't leak past SQ gates.
        PalindromePlacementStrategy.sq_placements bypasses this so that
        UserMoved survives SQ gates and cz_placements can still read
        accumulated_move_layers.
        """
        if isinstance(state, ConcreteState):
            return ConcreteState(
                occupied=state.occupied,
                layout=state.layout,
                move_count=state.move_count,
            )
        return state

    def _unwrap_cz_input(self, state: AtomState) -> AtomState:
        """Normalize state before calling the CZ solver.

        ``ExecuteCZReturn`` encodes a palindrome home position in
        ``initial_layout``; the next CZ must start from there, not from the
        staging positions stored in ``layout``.  Plain ``ExecuteCZ`` and
        ``ConcreteState`` are used as-is (atoms are already at the desired
        starting position).
        """
        if isinstance(state, ExecuteCZReturn):
            return ConcreteState(
                occupied=state.occupied,
                layout=state.initial_layout,
                move_count=state.move_count,
            )
        return state

    def move_to_placements(
        self,
        state: AtomState,
        qubits: tuple[int, ...],
        locations: tuple[LocationAddress, ...],
    ) -> AtomState:
        """User-directed atom movement placement.

        Stub — implemented by concrete strategies that support user-directed
        movement.  Raises NotImplementedError by default so existing strategies
        fail loudly if reached before the full impl lands.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement move_to_placements"
        )

    def compute_rearrangement(
        self,
        state_before: "ConcreteState",
        state_after: "ConcreteState",
    ) -> "tuple[tuple[LaneAddress, ...], ...]":
        """Compute move layers to route atoms from state_before to state_after.

        Used by UserMoved rewrite path to synthesize move layers for
        user-directed movement.  Raises NotImplementedError by default;
        physical strategies override this with solver-backed implementations.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement compute_rearrangement"
        )


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
        state = self._unwrap_cz_input(state)
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
        return self._strip_user_moved(state)

    def measure_placements(
        self, state: AtomState, qubits: tuple[int, ...]
    ) -> AtomState:
        if isinstance(state, UserMoved):
            return AtomState.bottom()  # move_to before measurement is invalid
        if not isinstance(state, ConcreteState):
            return state

        # all qubits must be measured
        if len(qubits) != len(state.layout):
            return AtomState.bottom()

        # Order layout/zone_maps/move_count by the measurement order ``qubits``
        # so each emitted measurement result lines up with the location of the
        # qubit it measures. ``qubits`` is identity for un-merged blocks but a
        # permutation once StaticPlacement blocks are merged (always_merge),
        # which remaps qubit indices; reading state.layout positionally would
        # then pair a result with the wrong patch.
        return ExecuteMeasure(
            occupied=state.occupied,
            layout=tuple(state.layout[i] for i in qubits),
            move_count=tuple(state.move_count[i] for i in qubits),
            zone_maps=tuple(ZoneAddress(state.layout[i].zone_id) for i in qubits),
        )

    def move_to_placements(
        self,
        state: AtomState,
        qubits: tuple[int, ...],
        locations: tuple[LocationAddress, ...],
    ) -> AtomState:
        if state == AtomState.bottom():
            return AtomState.bottom()
        if not isinstance(state, ConcreteState):
            return AtomState.top()

        # Occupancy check: destinations must not be held by unmoved qubits
        moved_set = set(qubits)
        for dest in locations:
            for idx, current_loc in enumerate(state.layout):
                if current_loc == dest and idx not in moved_set:
                    return AtomState.bottom()  # unmoved qubit at destination

        # Build target layout
        new_layout = list(state.layout)
        for qubit_idx, dest in zip(qubits, locations):
            new_layout[qubit_idx] = dest
        target_layout = tuple(new_layout)

        # Synthetic target ConcreteState for move synthesis
        target_state = ConcreteState(
            occupied=state.occupied,
            layout=target_layout,
            move_count=state.move_count,
        )

        try:
            new_layers = self.compute_moves(state, target_state)
        except RuntimeError:
            return AtomState.bottom()  # synthesizer failure

        # Accumulate layers across consecutive MoveTo calls
        if isinstance(state, UserMoved):
            accumulated = state.accumulated_move_layers + new_layers
            pre_user = state.pre_user_layout
        else:
            accumulated = new_layers
            pre_user = state.layout

        return UserMoved.from_concrete_state(
            target_state,
            move_layers=new_layers,
            accumulated_move_layers=accumulated,
            pre_user_layout=pre_user,
        )


class PalindromePlacementStrategy(PlacementStrategyABC):
    """Wraps any PlacementStrategyABC to emit ExecuteCZReturn for every CZ.

    On ``cz_placements``: unwraps ``ExecuteCZReturn`` input to the home
    ``ConcreteState`` (via ``initial_layout``), delegates to ``inner``, then
    wraps the resulting ``ExecuteCZ`` in ``ExecuteCZReturn``.

    On ``sq_placements`` and ``measure_placements``: unwraps
    ``ExecuteCZReturn`` to the home ``ConcreteState`` before delegating, so
    the inner strategy always sees a plain ``ConcreteState``.

    This replaces the ``insert_return_moves`` flag: choosing this strategy
    enables palindrome moves; using the bare inner strategy disables them.
    """

    def __init__(self, *, inner: PlacementStrategyABC) -> None:
        self.inner = inner

    @property  # type: ignore[reportIncompatibleVariableOverride]
    def arch_spec(self) -> ArchSpec:  # type: ignore[reportIncompatibleVariableOverride]
        return self.inner.arch_spec

    def _unwrap(self, state: AtomState) -> AtomState:
        """Return home ConcreteState when state is ExecuteCZReturn, else pass through."""
        if isinstance(state, ExecuteCZReturn):
            return ConcreteState(
                occupied=state.occupied,
                layout=state.initial_layout,
                move_count=state.move_count,
            )
        return state

    def validate_initial_layout(
        self, initial_layout: tuple[LocationAddress, ...]
    ) -> None:
        self.inner.validate_initial_layout(initial_layout)

    def cz_placements(
        self,
        state: AtomState,
        controls: tuple[int, ...],
        targets: tuple[int, ...],
        lookahead_cz_layers: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = (),
    ) -> AtomState:
        home = self._unwrap(state)
        result = self.inner.cz_placements(home, controls, targets, lookahead_cz_layers)
        if not isinstance(result, ExecuteCZ) or not isinstance(home, ConcreteState):
            return result

        if isinstance(home, UserMoved):
            return ExecuteCZReturn(
                occupied=result.occupied,
                layout=result.layout,
                move_count=result.move_count,
                active_cz_zones=result.active_cz_zones,
                move_layers=result.move_layers,
                user_move_layers=home.accumulated_move_layers,
                initial_layout=home.pre_user_layout,
            )

        return ExecuteCZReturn(
            occupied=result.occupied,
            layout=result.layout,
            move_count=result.move_count,
            active_cz_zones=result.active_cz_zones,
            move_layers=result.move_layers,
            initial_layout=home.layout,
        )

    def sq_placements(self, state: AtomState, qubits: tuple[int, ...]) -> AtomState:
        if isinstance(state, UserMoved):
            # Do NOT call _strip_user_moved here: UserMoved must survive SQ gates
            # so cz_placements can read accumulated_move_layers and include the
            # user-move history in the palindrome return.
            return state
        return self.inner.sq_placements(self._unwrap(state), qubits)

    def measure_placements(
        self, state: AtomState, qubits: tuple[int, ...]
    ) -> AtomState:
        return self.inner.measure_placements(self._unwrap(state), qubits)

    def move_to_placements(
        self,
        state: AtomState,
        qubits: tuple[int, ...],
        locations: tuple[LocationAddress, ...],
    ) -> AtomState:
        return self.inner.move_to_placements(self._unwrap(state), qubits, locations)
