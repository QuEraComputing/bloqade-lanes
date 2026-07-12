from dataclasses import dataclass, field
from typing import final

from kirin.lattice import (
    BoundedLattice,
    SimpleJoinMixin,
    SimpleMeetMixin,
    SingletonMeta,
)

from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode.encoding import LaneAddress, LocationAddress, ZoneAddress


@dataclass
class AtomState(
    SimpleJoinMixin["AtomState"],
    SimpleMeetMixin["AtomState"],
    BoundedLattice["AtomState"],
):

    @classmethod
    def bottom(cls) -> "AtomState":
        return NotState()

    @classmethod
    def top(cls) -> "AtomState":
        return AnyState()

    def get_move_layers(self) -> tuple[tuple[LaneAddress, ...], ...]:
        return ()

    def get_reverse_moves(self) -> tuple[tuple[LaneAddress, ...], ...]:
        return ()


@final
@dataclass
class NotState(AtomState, metaclass=SingletonMeta):

    def is_subseteq(self, other: AtomState) -> bool:
        return True


@final
@dataclass
class AnyState(AtomState, metaclass=SingletonMeta):

    def is_subseteq(self, other: AtomState) -> bool:
        return isinstance(other, AnyState)


@dataclass
class ConcreteState(AtomState):
    occupied: frozenset[LocationAddress]
    """Stores the set of occupied locations with atoms not participating in this static circuit."""
    layout: tuple[LocationAddress, ...]
    """Stores the current location of the ith qubit argument in layout[i]."""
    move_count: tuple[int, ...]
    """Stores the number of moves each atom has undergone."""

    def __post_init__(self):
        assert self.occupied.isdisjoint(
            self.layout
        ), "layout can't containe occupied location addresses"
        assert len(set(self.layout)) == len(
            self.layout
        ), "Atoms can't occupy the same location"

    def is_subseteq(self, other: AtomState) -> bool:
        return (
            isinstance(other, ConcreteState)
            and self.occupied == other.occupied
            and self.layout == other.layout
        )

    def get_qubit_id(self, location: LocationAddress) -> int | None:
        try:
            return self.layout.index(location)
        except ValueError:
            return None


@dataclass
class ExecuteCZ(ConcreteState):
    """Defines the state representing the placement of
    atoms before/after executing CZ gate pulse.

    NOTE: you can specify multiple entnangling zones to be active
    in a single ExecuteCZ state in cases where there are multiple entangling
    zones that can be used in parallel.

    """

    active_cz_zones: frozenset[ZoneAddress]
    """The set of CZ zones that need to execute for this round of CZ gates."""
    move_layers: tuple[tuple[LaneAddress, ...], ...] = ()
    """The layers of moves that need to be executed to reach this state."""

    def get_move_layers(self) -> tuple[tuple[LaneAddress, ...], ...]:
        return self.move_layers

    @classmethod
    def from_concrete_state(
        cls, state: ConcreteState, active_cz_zones: frozenset[ZoneAddress]
    ) -> "ExecuteCZ":
        return cls(
            occupied=state.occupied,
            layout=state.layout,
            move_count=state.move_count,
            active_cz_zones=active_cz_zones,
        )

    def is_subseteq(self, other: AtomState) -> bool:
        return (
            super().is_subseteq(other)
            and isinstance(other, ExecuteCZ)
            and self.active_cz_zones == other.active_cz_zones
        )

    def verify(
        self, arch_spec: ArchSpec, controls: tuple[int, ...], targets: tuple[int, ...]
    ):
        """Returns True if the current atom configuration will execute the provided entangled pairs."""
        if len(targets) != len(controls):
            return False

        for control, target in zip(controls, targets):
            if control < 0 or control >= len(self.layout):
                return False
            if target < 0 or target >= len(self.layout):
                return False

            c_addr = self.layout[control]
            t_addr = self.layout[target]

            if (arch_spec.get_cz_partner(c_addr) != t_addr) and (
                arch_spec.get_cz_partner(t_addr) != c_addr
            ):
                return False

        return True


@final
@dataclass
class ExecuteMeasure(ConcreteState):
    """A state representing measurement placements.

    NOTE: Depending on the placement of the atoms you may need to specify
    which atoms are measured by which zone. This is done via the zone_maps field, such that
    `zone_maps[i]` gives the zone that measures the ith qubit.

    """

    zone_maps: tuple[ZoneAddress, ...]
    """The mapping from qubit index to the zone that measures it."""
    move_layers: tuple[tuple[LaneAddress, ...], ...] = ()
    """The layers of moves that need to be executed to reach this state."""

    def get_move_layers(self) -> tuple[tuple[LaneAddress, ...], ...]:
        return self.move_layers

    @classmethod
    def from_concrete_state(
        cls, state: ConcreteState, zone_maps: tuple[ZoneAddress, ...]
    ) -> "ExecuteMeasure":
        return cls(
            occupied=state.occupied,
            layout=state.layout,
            move_count=state.move_count,
            zone_maps=zone_maps,
        )

    def is_subseteq(self, other: AtomState) -> bool:
        return (
            super().is_subseteq(other)
            and isinstance(other, ExecuteMeasure)
            and self.zone_maps == other.zone_maps
        )


@final
@dataclass
class ExecuteCZReturn(ExecuteCZ):
    """ExecuteCZ with palindrome return moves encoded at analysis time.

    Produced by ``PalindromePlacementStrategy`` so that ``InsertMoves`` can
    emit both forward moves (before the CZ gate) and return moves (after the
    CZ gate) in a single pass, without a separate ``InsertReturnMoves`` pass.

    ``initial_layout`` records the atom layout *before* the forward moves were
    applied.  ``PalindromePlacementStrategy`` uses this when the next CZ gate
    arrives and its input state is an ``ExecuteCZReturn`` — it extracts the
    home ``ConcreteState`` so the next placement starts from the correct
    position.

    ``return_move_layers`` is the strict palindrome of ``move_layers``: layer
    order reversed, and within each layer the lane order is also reversed with
    each lane's direction flipped.  Reversing the within-layer order is required
    for correctness when a forward layer contains dependent lanes — i.e. atom A
    vacates position X and atom B immediately enters X in the same layer.  On
    the return, B must be processed first (so B vacates X before A tries to
    re-enter it); preserving the forward order would cause A to collide with B.
    The value is computed eagerly from ``move_layers`` in ``__post_init__``.
    """

    initial_layout: tuple[LocationAddress, ...] = field(kw_only=True)
    """Atom layout before forward moves were applied (the home position)."""

    user_move_layers: tuple[tuple[LaneAddress, ...], ...] = field(
        kw_only=True, default=()
    )
    """User-move layers already emitted forward at MoveTo sites.
    Included in return_move_layers to palindrome the full inter-CZ segment."""

    return_move_layers: tuple[tuple[LaneAddress, ...], ...] = field(init=False)
    """Strict palindrome of the combined forward sequence
    ``user_move_layers + move_layers``; computed at construction."""

    def __post_init__(self) -> None:
        super().__post_init__()
        # Forward order within an inter-CZ segment is user moves (emitted at the
        # MoveTo sites) followed by compiler moves (emitted before the CZ), i.e.
        # ``user_move_layers + move_layers``. The palindrome return is the strict
        # reverse of that single combined sequence: layer order reversed and,
        # within each layer, lane order reversed with each lane's direction
        # flipped. Treating both halves uniformly yields
        # ``reverse(move_layers) + reverse(user_move_layers)`` and avoids the
        # within-layer-ordering mismatch that arises from reversing them apart.
        forward_layers = self.user_move_layers + self.move_layers
        self.return_move_layers = tuple(
            tuple(lane.reverse() for lane in reversed(layer))
            for layer in reversed(forward_layers)
        )

    def get_reverse_moves(self) -> tuple[tuple[LaneAddress, ...], ...]:
        return self.return_move_layers

    def is_subseteq(self, other: AtomState) -> bool:
        return (
            super().is_subseteq(other)
            and isinstance(other, ExecuteCZReturn)
            and self.initial_layout == other.initial_layout
            and self.user_move_layers == other.user_move_layers
        )


@final
@dataclass
class UserMoved(ConcreteState):
    """State produced by a user-directed place.MoveTo statement.

    - `move_layers`: AOD layers for *this* MoveTo only; read by InsertMoves
      to emit forward Move IR at the MoveTo site.
    - `accumulated_move_layers`: all user-move layers since the last CZ (or
      start), for palindrome return at the next CZ.
    - `pre_user_layout`: atom layout before the first user move in this
      inter-CZ segment; the palindrome home position.
    """

    move_layers: tuple[tuple[LaneAddress, ...], ...] = field(kw_only=True)
    accumulated_move_layers: tuple[tuple[LaneAddress, ...], ...] = field(kw_only=True)
    pre_user_layout: tuple[LocationAddress, ...] = field(kw_only=True)

    def get_move_layers(self) -> tuple[tuple[LaneAddress, ...], ...]:
        return self.move_layers

    def is_subseteq(self, other: AtomState) -> bool:
        return (
            super().is_subseteq(other)
            and isinstance(other, UserMoved)
            and self.move_layers == other.move_layers
            and self.accumulated_move_layers == other.accumulated_move_layers
            and self.pre_user_layout == other.pre_user_layout
        )

    @classmethod
    def from_concrete_state(
        cls,
        state: ConcreteState,
        move_layers: tuple[tuple[LaneAddress, ...], ...],
        accumulated_move_layers: tuple[tuple[LaneAddress, ...], ...],
        pre_user_layout: tuple[LocationAddress, ...],
    ) -> "UserMoved":
        return cls(
            occupied=state.occupied,
            layout=state.layout,
            move_count=state.move_count,
            move_layers=move_layers,
            accumulated_move_layers=accumulated_move_layers,
            pre_user_layout=pre_user_layout,
        )


@final
@dataclass
class Relabeled(ConcreteState):
    """State produced by a ``place.Permute`` with ``relabel=True`` (an active
    qubit permutation).

    The physical permutation is *committed*: ``move_layers`` are emitted at the
    permute site (``get_move_layers``) but are **not** palindrome-returned. The
    permutation cycles atoms among the same set of slots, so each qubit id is
    pinned back to its pre-permute slot — ``layout`` is therefore unchanged, and
    the permutation surfaces as a relabel: the atom now sitting at qid ``i``'s
    slot is a different one, so downstream ops on qid ``i`` act on the permuted
    quantum information.

    Unlike ``UserMoved`` (palindrome-pending, rejected at a terminal measure),
    ``Relabeled`` is a committed ``ConcreteState``: it flows through subsequent
    SQ / CZ / measure normally, and ``_strip_user_moved`` collapses it to a plain
    ``ConcreteState`` once its forward moves have been emitted at the permute
    site.
    """

    move_layers: tuple[tuple[LaneAddress, ...], ...] = field(kw_only=True, default=())
    """Forward AOD layers realizing the physical permutation; emitted once at
    the permute site and not returned."""

    def get_move_layers(self) -> tuple[tuple[LaneAddress, ...], ...]:
        return self.move_layers

    def is_subseteq(self, other: AtomState) -> bool:
        return (
            super().is_subseteq(other)
            and isinstance(other, Relabeled)
            and self.move_layers == other.move_layers
        )
