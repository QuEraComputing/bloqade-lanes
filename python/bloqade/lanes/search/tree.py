"""Configuration tree for exploring valid atom move programs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from bloqade.lanes.layout import LaneAddress, LocationAddress
from bloqade.lanes.layout.arch import ArchSpec
from bloqade.lanes.layout.path import PathFinder
from bloqade.lanes.search.configuration import Configuration, ConfigurationNode

if TYPE_CHECKING:
    from bloqade.lanes.search.generators import MoveGenerator


@dataclass
class ConfigurationTree:
    """Tree that explores the space of valid atom configurations.

    Starting from an initial placement, the tree expands by applying
    valid move sets to generate child configurations. A transposition
    table prevents re-expanding configurations already seen at
    equal-or-lesser depth.

    Move generation is delegated to a MoveGenerator, which produces
    candidate move sets. Validation (lane validity, collision checks,
    transposition table) is handled by _apply_move_set.
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

    def _enumerate_single_moves(
        self, node: ConfigurationNode
    ) -> list[tuple[LaneAddress, LocationAddress, LocationAddress]]:
        """Enumerate all valid single-lane moves from a configuration.

        For each occupied location, finds outgoing edges in the path
        finder's site graph where the destination is unoccupied.

        Returns:
            List of (lane_address, src_location, dst_location) triples.
        """
        occupied = node.occupied_locations
        result: list[tuple[LaneAddress, LocationAddress, LocationAddress]] = []

        for _qid, src_loc in node.configuration.items():
            src_node_idx = self.path_finder.physical_address_map.get(src_loc)
            if src_node_idx is None:
                continue

            for dst_node_idx in self.path_finder.site_graph.successor_indices(
                src_node_idx
            ):
                dst_loc = self.path_finder.physical_addresses[dst_node_idx]
                if dst_loc in occupied:
                    continue

                lane = self.path_finder.site_graph.get_edge_data(
                    src_node_idx, dst_node_idx
                )
                if lane is not None:
                    result.append((lane, src_loc, dst_loc))

        return result

    def _apply_move_set(
        self,
        node: ConfigurationNode,
        move_set: frozenset[LaneAddress],
    ) -> ConfigurationNode | None:
        """Apply a move set to a node, returning a new child or None.

        Resolves lane endpoints, checks for collisions, and creates
        a child node if valid. This is the single validation point —
        MoveGenerators may yield invalid candidates that are filtered here.

        Returns:
            A new ConfigurationNode, or None if:
            - An occupied source has an occupied destination (collision)
            - Two atoms would move to the same destination
            - The configuration was already seen at equal-or-lesser depth
        """
        new_config = dict(node.configuration)
        destinations: set[LocationAddress] = set()
        occupied = node.occupied_locations

        for lane in move_set:
            src, dst = self.arch_spec.get_endpoints(lane)

            qid = node.get_qubit_at(src)
            if qid is None:
                # No atom at source — no-op lane
                continue

            # Check collision: destination already occupied by non-moving atom
            # or two atoms moving to the same destination
            if dst in occupied or dst in destinations:
                return None

            destinations.add(dst)
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
    ) -> list[ConfigurationNode]:
        """Expand a node by generating and validating candidate move sets.

        The generator produces candidates; _apply_move_set validates each
        (collision checks, transposition table) and creates child nodes.

        Args:
            node: The node to expand.
            generator: The move generator to use for candidate enumeration.

        Returns:
            List of newly created child nodes (may be empty if deadlocked).
        """
        children: list[ConfigurationNode] = []
        for move_set in generator.generate(node, self):
            child = self._apply_move_set(node, move_set)
            if child is not None:
                children.append(child)
        return children
