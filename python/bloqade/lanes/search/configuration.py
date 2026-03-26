"""Configuration node for the atom move search tree."""

from __future__ import annotations

from dataclasses import dataclass, field

from bloqade.lanes.layout import LaneAddress, LocationAddress

# Canonical hashable type for atom configurations.
# Order-independent: frozenset of (qubit_id, location) pairs.
Configuration = frozenset[tuple[int, LocationAddress]]


@dataclass
class ConfigurationNode:
    """A node in the configuration tree representing a valid atom placement.

    Each node tracks which qubits are at which physical locations, how this
    configuration was reached (parent + moves), and what configurations are
    reachable from here (children).
    """

    configuration: dict[int, LocationAddress]
    """Mapping of qubit ID to current physical location."""

    parent: ConfigurationNode | None = None
    """The parent node that produced this configuration, or None for the root."""

    parent_moves: frozenset[LaneAddress] | None = None
    """The move set applied to the parent to produce this configuration."""

    children: dict[frozenset[LaneAddress], ConfigurationNode] = field(
        default_factory=dict
    )
    """Children keyed by the move set that produced them."""

    depth: int = 0
    """Distance from the root node."""

    _config_key: Configuration = field(init=False, repr=False, compare=False)
    _occupied_locations: frozenset[LocationAddress] = field(
        init=False, repr=False, compare=False
    )
    _qubit_at_location: dict[LocationAddress, int] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        self._config_key = frozenset(self.configuration.items())
        self._occupied_locations = frozenset(self.configuration.values())
        self._qubit_at_location = {loc: qid for qid, loc in self.configuration.items()}

    @property
    def config_key(self) -> Configuration:
        """Canonical hashable key for this configuration.

        Two nodes with the same atom placement (regardless of history)
        produce the same config_key.
        """
        return self._config_key

    @property
    def occupied_locations(self) -> frozenset[LocationAddress]:
        """The set of physical locations currently occupied by atoms."""
        return self._occupied_locations

    def is_occupied(self, location: LocationAddress) -> bool:
        """Check whether a physical location has an atom."""
        return location in self._occupied_locations

    def get_qubit_at(self, location: LocationAddress) -> int | None:
        """Return the qubit ID at a location, or None if empty."""
        return self._qubit_at_location.get(location)

    def path_to_root(self) -> list[frozenset[LaneAddress]]:
        """Walk from this node to the root, returning move sets in root-to-leaf order.

        The returned list has length equal to this node's depth. Each element
        is the frozenset of lane addresses applied at that step.
        """
        moves: list[frozenset[LaneAddress]] = []
        node = self
        while node.parent is not None:
            assert node.parent_moves is not None
            moves.append(node.parent_moves)
            node = node.parent
        moves.reverse()
        return moves

    def to_move_program(self) -> tuple[tuple[LaneAddress, ...], ...]:
        """Convert the path from root to this node into a move program.

        Returns a tuple of tuples, where each inner tuple is a sorted
        sequence of lane addresses for one move step. Sorting is by
        encoded lane value for deterministic ordering.
        """
        return tuple(
            tuple(sorted(ms, key=lambda lane: lane.encode()))
            for ms in self.path_to_root()
        )
