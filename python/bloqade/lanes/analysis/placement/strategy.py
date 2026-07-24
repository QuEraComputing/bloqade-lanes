import abc
from dataclasses import dataclass

from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode.encoding import LaneAddress, LocationAddress, ZoneAddress

from .exceptions import PlacementError
from .lattice import (
    AtomState,
    ConcreteState,
    ExecuteCZ,
    ExecuteCZReturn,
    ExecuteMeasure,
    Permuted,
    UserMoved,
)


def _resolve_permute_locations(
    layout: tuple[LocationAddress, ...],
    qubits: tuple[int, ...],
    permutation: tuple[int, ...],
) -> tuple[LocationAddress, ...]:
    """Resolve a permutation to target locations against the current layout.

    ``qubits`` are layout indices; ``permutation`` permutes the *positions* of
    ``qubits``. For each value ``j`` in ``permutation`` (an index into
    ``qubits``), the target is the current location of that qubit,
    ``layout[qubits[j]]``.
    """
    return tuple(layout[qubits[j]] for j in permutation)


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

    def measure_placements(
        self,
        state: AtomState,
        qubits: tuple[int, ...],
    ) -> AtomState:
        """Place a terminal measurement of all ``qubits``.

        ``layout`` stays canonical (indexed by qubit id) — it must not be
        permuted, since reordering it would relabel qubits. The measurement
        order is carried by ``place.EndMeasure.qubits`` and applied when the
        measurement is lowered (see ``place2move.InsertMeasure``), which indexes
        this canonical layout by ``qubits`` so each result reads the location of
        the qubit it measures (``qubits`` is a permutation once StaticPlacement
        blocks are merged).

        A ``UserMoved`` state is accepted here: a user-directed move that ends
        at a measurement is committed — the atoms stay at their moved layout and
        are measured there, and the compiler tracks the final positions within
        the single zone. ``UserMoved`` is a ``ConcreteState`` subclass, so it
        flows through the concrete-state path below. ``PalindromePlacementStrategy``
        overrides this to reject ``UserMoved`` instead: under palindrome a
        user-move is only committed by a following CZ, never a measurement.
        """
        if not isinstance(state, ConcreteState):
            return state
        if len(qubits) != len(state.layout):
            raise PlacementError(
                f"terminal measurement must measure all {len(state.layout)} "
                f"qubits in the block, got {len(qubits)}"
            )
        return ExecuteMeasure(
            occupied=state.occupied,
            layout=state.layout,
            move_count=state.move_count,
            zone_maps=tuple(ZoneAddress(loc.zone_id) for loc in state.layout),
        )

    def _strip_user_moved(self, state: AtomState) -> AtomState:
        """Normalise any ConcreteState subtype back to a plain ConcreteState.

        Despite the name, this strips the extra metadata of *any* ConcreteState
        subclass — UserMoved's ``accumulated_move_layers``/``pre_user_layout`` as
        well as ExecuteCZ/ExecuteCZReturn's move/CZ metadata — keeping only
        ``occupied``/``layout``/``move_count``. Non-palindrome strategies call
        this in sq_placements so that metadata (only needed for the palindrome
        return at the next CZ) does not leak past SQ gates.
        PalindromePlacementStrategy.sq_placements bypasses this so that UserMoved
        survives SQ gates and cz_placements can still read accumulated_move_layers.
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


class MoveToPlacementStrategyABC(PlacementStrategyABC):
    """Base class for strategies that support user-directed ``MoveTo``.

    Adds the ``compute_moves`` primitive (route atoms between two layouts) and a
    shared ``move_to_placements`` built on top of it. Strategies that do not
    support user-directed movement should extend ``PlacementStrategyABC``
    directly; the ``MoveTo`` placement interpreter returns ``bottom`` for any
    strategy that is not a ``MoveToPlacementStrategyABC``.
    """

    @abc.abstractmethod
    def compute_moves(
        self,
        state_before: ConcreteState,
        state_after: ConcreteState,
    ) -> tuple[tuple[LaneAddress, ...], ...]:
        """Compute the AOD move layers routing atoms from ``state_before`` to
        ``state_after``."""

    def move_to_placements(
        self,
        state: AtomState,
        qubits: tuple[int, ...],
        locations: tuple[LocationAddress, ...],
    ) -> AtomState:
        """Place ``qubits`` at ``locations`` (user-directed movement).

        Produces a ``UserMoved`` state whose ``accumulated_move_layers`` grows
        across consecutive MoveTo calls within an inter-CZ segment and whose
        ``pre_user_layout`` pins the segment's home position for the eventual
        palindrome return.
        """
        if state == AtomState.bottom():
            return AtomState.bottom()
        if not isinstance(state, ConcreteState):
            return AtomState.top()

        # Occupancy check: reject any request that would produce an invalid
        # target layout (which ConcreteState's __post_init__ would otherwise
        # assert on). Three ways to invalidate:
        #   1. two moved qubits mapped to the same destination,
        #   2. destination overlaps an external atom in state.occupied,
        #   3. destination held by an unmoved qubit.
        if len(set(locations)) != len(locations):
            raise PlacementError(
                f"move_to maps multiple qubits to the same destination: {locations}"
            )
        moved_set = set(qubits)
        for dest in locations:
            if dest in state.occupied:
                raise PlacementError(
                    f"move_to destination {dest} is occupied by an external atom"
                )
            for idx, current_loc in enumerate(state.layout):
                if current_loc == dest and idx not in moved_set:
                    raise PlacementError(
                        f"move_to destination {dest} is held by unmoved qubit {idx}"
                    )

        new_layout = list(state.layout)
        for qubit_idx, dest in zip(qubits, locations):
            new_layout[qubit_idx] = dest
        target_state = ConcreteState(
            occupied=state.occupied,
            layout=tuple(new_layout),
            move_count=state.move_count,
        )

        try:
            new_layers = self.compute_moves(state, target_state)
        except RuntimeError as exc:
            raise PlacementError(
                f"move synthesizer could not route qubits {qubits} to {locations}: "
                f"{exc}"
            ) from exc

        # Accumulate across consecutive MoveTo calls in the same segment.
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

    def permute_placements(
        self,
        state: AtomState,
        qubits: tuple[int, ...],
        permutation: tuple[int, ...],
        insert_moves: bool = False,
    ) -> AtomState:
        """Permute ``qubits`` — a logical relabel where ``qubits[i]`` ends up
        referring to what was ``qubits[permutation[i]]``.

        ``insert_moves=False`` (default) is a *lazy* relabel: no atoms move, the
        ``layout`` is permuted so each qubit id points at the location of the
        atom it now denotes, and the quantum information is permuted for free (a
        later transversal move absorbs the physical permutation). Produces a
        plain ``ConcreteState`` with the permuted layout.

        ``insert_moves=True`` additionally *commits* the physical moves that
        route the atoms so the permutation is realized now, pinning the layout
        back to the pre-permute layout (each qid keeps its slot, now holding the
        routed atom). Produces a ``Permuted`` state. This is only valid under a
        non-palindrome strategy (``PalindromePlacementStrategy`` rejects it).
        """
        if state == AtomState.bottom():
            return AtomState.bottom()
        if not isinstance(state, ConcreteState):
            return AtomState.top()

        locations = _resolve_permute_locations(state.layout, qubits, permutation)

        # Occupancy check: destinations must not be held by unmoved qubits. In a
        # permutation every destination is held by a moving qubit, so this is
        # only reachable via malformed input (``permutation`` is not a genuine
        # permutation of ``qubits``).
        moved_set = set(qubits)
        for dest in locations:
            for idx, current_loc in enumerate(state.layout):
                if current_loc == dest and idx not in moved_set:
                    raise PlacementError(
                        f"permute destination {dest} is held by unmoved qubit "
                        f"{idx}; permutation {permutation} of qubits {qubits} is "
                        "not a valid permutation"
                    )

        relabeled_layout = list(state.layout)
        for qubit_idx, dest in zip(qubits, locations):
            relabeled_layout[qubit_idx] = dest

        if not insert_moves:
            # Lazy relabel: permute the layout (references) only, emit no moves.
            if isinstance(state, UserMoved):
                # Preserve the user-move history through the relabel: the
                # accumulated physical layers are lane-indexed and unchanged,
                # but pre_user_layout is qubit-id indexed and must permute
                # alongside layout so the palindrome return maps each qubit id
                # to the correct home slot for the atom it now denotes.
                pre_user_values = _resolve_permute_locations(
                    state.pre_user_layout, qubits, permutation
                )
                permuted_pre_user = list(state.pre_user_layout)
                for qubit_idx, source_val in zip(qubits, pre_user_values):
                    permuted_pre_user[qubit_idx] = source_val
                return UserMoved(
                    occupied=state.occupied,
                    layout=tuple(relabeled_layout),
                    move_count=state.move_count,
                    move_layers=(),  # relabel adds no physical moves here
                    accumulated_move_layers=state.accumulated_move_layers,
                    pre_user_layout=tuple(permuted_pre_user),
                )
            return ConcreteState(
                occupied=state.occupied,
                layout=tuple(relabeled_layout),
                move_count=state.move_count,
            )

        # Active: commit the physical permutation. Route atoms to the relabeled
        # positions, then pin the layout back to the pre-permute layout (the
        # atoms are permuted among their slots; each qid keeps its slot).
        target_state = ConcreteState(
            occupied=state.occupied,
            layout=tuple(relabeled_layout),
            move_count=state.move_count,
        )
        try:
            move_layers = self.compute_moves(state, target_state)
        except RuntimeError as exc:
            raise PlacementError(
                f"move synthesizer could not route the committed permutation "
                f"{permutation} of qubits {qubits}: {exc}"
            ) from exc
        return Permuted(
            occupied=state.occupied,
            layout=state.layout,
            move_count=state.move_count,
            move_layers=move_layers,
        )


class SingleZonePlacementStrategyABC(MoveToPlacementStrategyABC):

    def __post_init__(self) -> None:
        assert_single_cz_zone(self.arch_spec, type(self).__name__)

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
        if state == AtomState.bottom():
            return AtomState.bottom()
        if len(controls) != len(targets):
            raise PlacementError(
                f"CZ has mismatched control/target counts: "
                f"{len(controls)} controls, {len(targets)} targets"
            )
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
        # Strip CZ/user-move metadata so non-CZ statements do not inherit stale
        # move layers in downstream rewrite passes. ``measure_placements`` is
        # inherited from ``PlacementStrategyABC`` (canonical-layout ExecuteMeasure).
        return self._strip_user_moved(state)


class PalindromePlacementStrategy(MoveToPlacementStrategyABC):
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

    def __init__(self, *, inner: PlacementStrategyABC, lookahead: int = 1) -> None:
        if lookahead < 0:
            raise ValueError(f"lookahead must be >= 0, got {lookahead}")
        self.inner = inner
        # ``lookahead``: number of CZ layers forwarded to ``inner.cz_placements``.
        # Defaults to ``1`` (only the current CZ layer). Palindrome moves always
        # return atoms to the pre-CZ home before the next CZ, so future CZ layers
        # cannot inform the current placement — looking ahead only inflates the
        # inner solver's search without changing the (forced) home return. A
        # larger value forwards that many leading layers; ``0`` forwards none.
        self.lookahead = lookahead

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
        result = self.inner.cz_placements(
            home, controls, targets, lookahead_cz_layers[: self.lookahead]
        )
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
        if isinstance(state, UserMoved):
            # Under palindrome, a user-move is only committed by a following CZ
            # (stage + return). Reaching a measurement with a pending user-move
            # is invalid. (The base strategy concretises it instead.)
            raise PlacementError(
                "PalindromePlacementStrategy reached a measurement with a "
                "pending user-directed move; under palindrome a user-move must "
                "be committed by a following CZ, not a measurement"
            )
        return self.inner.measure_placements(self._unwrap(state), qubits)

    def compute_moves(
        self,
        state_before: ConcreteState,
        state_after: ConcreteState,
    ) -> tuple[tuple[LaneAddress, ...], ...]:
        if not isinstance(self.inner, MoveToPlacementStrategyABC):
            raise NotImplementedError(
                f"inner strategy {type(self.inner).__name__} does not support "
                "user-directed movement (not a MoveToPlacementStrategyABC)"
            )
        return self.inner.compute_moves(state_before, state_after)

    def move_to_placements(
        self,
        state: AtomState,
        qubits: tuple[int, ...],
        locations: tuple[LocationAddress, ...],
    ) -> AtomState:
        # Palindrome supports MoveTo only when its inner does; otherwise the
        # segment is infeasible for user-directed movement.
        if not isinstance(self.inner, MoveToPlacementStrategyABC):
            raise PlacementError(
                f"PalindromePlacementStrategy inner strategy "
                f"{type(self.inner).__name__} does not support user-directed "
                "movement (not a MoveToPlacementStrategyABC), but a move_to was "
                "requested"
            )
        return self.inner.move_to_placements(self._unwrap(state), qubits, locations)

    def permute_placements(
        self,
        state: AtomState,
        qubits: tuple[int, ...],
        permutation: tuple[int, ...],
        insert_moves: bool = False,
    ) -> AtomState:
        # The default relabel-only permute is fine under palindrome: it just
        # permutes the layout (references) with no physical moves. But
        # ``insert_moves=True`` commits a physical permutation, which is
        # incompatible with the palindrome model — every move is returned to the
        # pre-move home at the next CZ, so committed moves would be (partly)
        # undone. Reject that combination loudly.
        if insert_moves:
            raise NotImplementedError(
                "permute(insert_moves=True) commits a physical permutation, which "
                "PalindromePlacementStrategy cannot express (it returns every move "
                "to the pre-move home at each CZ). Use a non-palindrome (no-return) "
                "placement strategy, or the default relabel-only permute."
            )
        if not isinstance(self.inner, MoveToPlacementStrategyABC):
            raise PlacementError(
                f"PalindromePlacementStrategy inner strategy "
                f"{type(self.inner).__name__} does not support user-directed "
                "movement (not a MoveToPlacementStrategyABC), but a permute was "
                "requested"
            )
        return self.inner.permute_placements(
            self._unwrap(state), qubits, permutation, insert_moves=False
        )
