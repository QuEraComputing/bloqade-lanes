"""Configuration tree for exploring valid atom move programs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from bloqade.lanes.layout import Direction, LaneAddress, LocationAddress, MoveType
from bloqade.lanes.layout.arch import ArchSpec
from bloqade.lanes.layout.path import PathFinder
from bloqade.lanes.search.configuration import Configuration, ConfigurationNode

if TYPE_CHECKING:
    from bloqade.lanes.search.generators import MoveGenerator


class InvalidMoveError(Exception):
    """Raised when a generator produces an invalid move set in strict mode."""


@dataclass
class ConfigurationTree:
    """Tree that explores the space of valid atom configurations.

    Starting from an initial placement, the tree manages the transposition
    table and validates move sets. Move generation and node expansion are
    delegated to MoveGenerator implementations.

    NOTE: If deadlock density is high, consider refactoring to a DAG
    (directed acyclic graph) where nodes can have multiple parents.
    This enables backward propagation of deadlock information — when a
    subtree is exhausted, all parents are notified and can prune early.
    Currently the transposition table prevents re-expanding seen
    configurations, but does not propagate deadlock status upstream.
    """

    arch_spec: ArchSpec
    root: ConfigurationNode
    path_finder: PathFinder = field(init=False, repr=False)
    seen: dict[Configuration, ConfigurationNode] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.path_finder = PathFinder(self.arch_spec)
        self.seen[self.root.config_key] = self.root

    @classmethod
    def from_initial_placement(
        cls,
        arch_spec: ArchSpec,
        placement: dict[int, LocationAddress],
    ) -> ConfigurationTree:
        """Create a tree from an initial qubit placement.

        Args:
            arch_spec: Architecture specification for lane validation.
            placement: Mapping of qubit IDs to their initial locations.

        Returns:
            A new ConfigurationTree rooted at the given placement.
        """
        root = ConfigurationNode(configuration=dict(placement))
        return cls(arch_spec=arch_spec, root=root)

    def valid_lanes(
        self,
        node: ConfigurationNode,
        move_type: MoveType | None = None,
        bus_id: int | None = None,
        direction: Direction | None = None,
    ) -> frozenset[LaneAddress]:
        """Return all valid individual lane addresses from a configuration.

        A lane is valid if its source is occupied and its destination
        is not occupied. Optionally filter by move_type, bus_id, and
        direction — None means include all.

        Args:
            node: The configuration to query.
            move_type: Filter to this move type, or None for all.
            bus_id: Filter to this bus ID, or None for all.
            direction: Filter to this direction, or None for all.

        Returns:
            frozenset of valid LaneAddress values.
        """
        occupied = node.occupied_locations
        result: set[LaneAddress] = set()

        for _qid, src_loc in node.configuration.items():
            src_idx = self.path_finder.physical_address_map.get(src_loc)
            if src_idx is None:
                continue

            for dst_idx in self.path_finder.site_graph.successor_indices(src_idx):
                dst_loc = self.path_finder.physical_addresses[dst_idx]
                if dst_loc in occupied:
                    continue

                lane = self.path_finder.site_graph.get_edge_data(src_idx, dst_idx)
                if lane is None:
                    continue

                # Apply filters
                if move_type is not None and lane.move_type != move_type:
                    continue
                if bus_id is not None and lane.bus_id != bus_id:
                    continue
                if direction is not None and lane.direction != direction:
                    continue

                result.add(lane)

        return frozenset(result)

    def apply_move_set(
        self,
        node: ConfigurationNode,
        move_set: frozenset[LaneAddress],
        strict: bool = True,
    ) -> ConfigurationNode | None:
        """Apply a move set to a node, returning a new child or None.

        Resolves lane endpoints, checks for collisions, and creates
        a child node if valid.

        Args:
            node: The node to apply moves to.
            move_set: The set of lane addresses to apply.
            strict: If True (default), raises InvalidMoveError when a
                move set causes a collision. If False, silently returns
                None for invalid moves.

        Returns:
            A new ConfigurationNode, or None if:
            - The move is invalid and strict=False
            - The configuration was already reached via a different
              branch at equal-or-lesser depth (transposition table)

        Raises:
            InvalidMoveError: If strict=True and the move set causes a
                collision or contains an invalid lane address.
        """
        new_config = dict(node.configuration)
        occupied = node.occupied_locations

        for lane in move_set:
            try:
                src, dst = self.arch_spec.get_endpoints(lane)
            except Exception as e:
                if strict:
                    raise InvalidMoveError(f"Invalid lane address {lane!r}: {e}") from e
                return None

            qid = node.get_qubit_at(src)
            if qid is None:
                continue

            # Bus src and dst are disjoint sets, so within a single-bus
            # move set, two sources cannot map to the same destination.
            # We only need to check against stationary atoms.
            if dst in occupied:
                if strict:
                    blocker = node.get_qubit_at(dst)
                    raise InvalidMoveError(
                        f"Collision: qubit {qid} moving to {dst!r} "
                        f"which is occupied by qubit {blocker}"
                    )
                return None

            new_config[qid] = dst

        # Check transposition table
        new_node = ConfigurationNode(
            configuration=new_config,
            parent=node,
            parent_moves=move_set,
            depth=node.depth + 1,
        )
        key = new_node.config_key

        if key in self.seen:
            existing = self.seen[key]
            if existing.depth <= new_node.depth:
                return None

        # Register in transposition table and add as child
        self.seen[key] = new_node
        node.children[move_set] = new_node
        return new_node

    def expand_node(
        self,
        node: ConfigurationNode,
        generator: MoveGenerator,
        strict: bool = True,
    ) -> list[ConfigurationNode]:
        """Expand a node using the given generator.

        Generates candidate move sets, validates each (collision checks,
        transposition table), and creates child nodes. Nodes already
        seen at equal-or-lesser depth are skipped.

        Args:
            node: The node to expand.
            generator: Produces candidate move sets.
            strict: If True, raises on invalid moves. If False, skips them.

        Returns:
            List of newly created child nodes (may be empty).
        """
        children: list[ConfigurationNode] = []
        for move_set in generator.generate(node, self):
            child = self.apply_move_set(node, move_set, strict=strict)
            if child is not None:
                children.append(child)
        return children
