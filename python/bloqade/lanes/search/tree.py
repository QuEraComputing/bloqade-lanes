"""Configuration tree for exploring valid atom move programs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from bloqade.lanes.layout import LaneAddress, LocationAddress
from bloqade.lanes.layout.arch import ArchSpec
from bloqade.lanes.layout.path import PathFinder
from bloqade.lanes.search.configuration import Configuration, ConfigurationNode


@dataclass
class ConfigurationTree:
    """Tree that explores the space of valid atom configurations.

    Starting from an initial placement, the tree expands by applying
    valid move sets to generate child configurations. A transposition
    table prevents re-expanding configurations already seen at
    equal-or-lesser depth.
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

        for qid, src_loc in node.configuration.items():
            src_node_idx = self.path_finder.physical_address_map.get(src_loc)
            if src_node_idx is None:
                continue

            # Iterate outgoing edges from this node in the site graph
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

    def _enumerate_compatible_move_sets(
        self, node: ConfigurationNode
    ) -> Iterator[frozenset[LaneAddress]]:
        """Yield valid move sets (compatible lane groups) from a configuration.

        A move set is a group of lanes that:
        1. Share the same move_type, bus_id, and direction
        2. Have occupied sources and unoccupied destinations
        3. Pass ArchSpec.check_lanes validation (including AOD constraints)
        4. Have no destination collisions after all moves resolve

        NOTE: This method is a placeholder. The enumeration logic for
        compatible move sets has non-trivial grouping constraints
        (AOD geometry, bus membership) that require careful design.
        See issue #298 for discussion.

        Yields:
            frozenset[LaneAddress] — each valid parallel move set.
        """
        raise NotImplementedError(
            "_enumerate_compatible_move_sets requires design discussion. "
            "See issue #298."
        )

    def _apply_move_set(
        self,
        node: ConfigurationNode,
        move_set: frozenset[LaneAddress],
    ) -> ConfigurationNode | None:
        """Apply a move set to a node, returning a new child or None.

        Resolves all lane endpoints, checks for collisions, and creates
        a child node if the resulting configuration is valid and not
        previously seen at equal-or-lesser depth.

        Returns:
            A new ConfigurationNode, or None if:
            - Any lane endpoint cannot be resolved
            - Two atoms would occupy the same location after moves
            - The configuration was already seen at equal-or-lesser depth
        """
        # Build the new configuration by applying moves
        new_config = dict(node.configuration)
        moved_qubits: set[int] = set()
        destinations: dict[LocationAddress, int] = {}

        for lane in move_set:
            endpoints = self.arch_spec.get_endpoints(lane)
            src, dst = endpoints

            qid = node.get_qubit_at(src)
            if qid is None:
                # No atom at source — skip this lane
                continue

            # Check destination collision with another moving atom
            if dst in destinations:
                return None  # Two atoms moving to same destination

            # Check destination collision with a stationary atom
            stationary_qid = node.get_qubit_at(dst)
            if stationary_qid is not None and stationary_qid not in moved_qubits:
                return None  # Would collide with stationary atom

            moved_qubits.add(qid)
            destinations[dst] = qid
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
                return None  # Already seen at equal-or-lesser depth

        # Register in transposition table and add as child
        self.seen[key] = new_node
        node.children[move_set] = new_node
        return new_node

    def expand_node(self, node: ConfigurationNode) -> list[ConfigurationNode]:
        """Expand a node by generating all valid children.

        Enumerates all valid move sets from the node's configuration,
        applies each, and returns the list of new child nodes.

        Returns:
            List of newly created child nodes (may be empty if deadlocked).
        """
        children: list[ConfigurationNode] = []
        for move_set in self._enumerate_compatible_move_sets(node):
            child = self._apply_move_set(node, move_set)
            if child is not None:
                children.append(child)
        return children
