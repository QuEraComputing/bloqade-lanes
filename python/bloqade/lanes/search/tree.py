"""Configuration tree for exploring valid atom move programs."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, product
from typing import Iterator

from bloqade.lanes.layout import (
    Direction,
    LaneAddress,
    LocationAddress,
    MoveType,
)
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
        self,
        node: ConfigurationNode,
        max_x_capacity: int | None = None,
        max_y_capacity: int | None = None,
    ) -> Iterator[frozenset[LaneAddress]]:
        """Yield valid move sets (compatible lane groups) from a configuration.

        For each (move_type, bus_id, direction) group, enumerates all valid
        rectangles of source positions on the bus. A valid rectangle is a
        Cartesian product of X and Y positions where no position in the
        rectangle is occupied by a stationary atom. The full rectangle of
        lanes is yielded if at least one source has an atom.

        The AOD moves the entire rectangle — lanes are generated for every
        source in the rectangle, not just occupied ones. Empty source
        positions produce no-op lanes (no atom to transport).

        Args:
            node: The configuration to expand from.
            max_x_capacity: Maximum number of unique X positions the AOD
                can address. None means unlimited.
            max_y_capacity: Maximum number of unique Y positions the AOD
                can address. None means unlimited.

        Yields:
            frozenset[LaneAddress] — each valid parallel move set.
        """
        occupied = node.occupied_locations

        # Enumerate site buses
        for bus_id, bus in enumerate(self.arch_spec.site_buses):
            for direction in (Direction.FORWARD, Direction.BACKWARD):
                # Source positions: every (word_id, src_site) for words with site buses
                src_locs = [
                    LocationAddress(w, s)
                    for w in self.arch_spec.has_site_buses
                    for s in bus.src
                ]
                yield from self._rectangles_to_move_sets(
                    src_locs,
                    occupied,
                    MoveType.SITE,
                    bus_id,
                    direction,
                    max_x_capacity,
                    max_y_capacity,
                )

        # Enumerate word buses
        for bus_id, bus in enumerate(self.arch_spec.word_buses):
            for direction in (Direction.FORWARD, Direction.BACKWARD):
                # Source positions: every (src_word, site_id) for sites with word buses
                src_locs = [
                    LocationAddress(w, s)
                    for w in bus.src
                    for s in self.arch_spec.has_word_buses
                ]
                yield from self._rectangles_to_move_sets(
                    src_locs,
                    occupied,
                    MoveType.WORD,
                    bus_id,
                    direction,
                    max_x_capacity,
                    max_y_capacity,
                )

    def _rectangles_to_move_sets(
        self,
        src_locs: list[LocationAddress],
        occupied: frozenset[LocationAddress],
        move_type: MoveType,
        bus_id: int,
        direction: Direction,
        max_x_capacity: int | None,
        max_y_capacity: int | None,
    ) -> Iterator[frozenset[LaneAddress]]:
        """Enumerate valid rectangles from source locations and yield move sets.

        A valid rectangle is a Cartesian product of X and Y positions from
        the source locations. At least one position in the rectangle must
        be occupied by an atom. No position in the rectangle may be occupied
        by a stationary atom (one not in the rectangle's source set).

        Args:
            src_locs: All source locations for this bus.
            occupied: Currently occupied locations.
            move_type: SITE or WORD.
            bus_id: Bus identifier.
            direction: FORWARD or BACKWARD.
            max_x_capacity: AOD X capacity limit (None = unlimited).
            max_y_capacity: AOD Y capacity limit (None = unlimited).

        Yields:
            frozenset[LaneAddress] for each valid rectangle.
        """

        if not src_locs:
            return

        # Build position lookups
        pos_to_loc: dict[tuple[float, float], LocationAddress] = {}
        unique_xs: set[float] = set()
        unique_ys: set[float] = set()
        for loc in src_locs:
            x, y = self.arch_spec.get_position(loc)
            pos_to_loc[(x, y)] = loc
            unique_xs.add(x)
            unique_ys.add(y)

        sorted_xs = sorted(unique_xs)
        sorted_ys = sorted(unique_ys)

        # Pre-build lane addresses and cache which are invalid (destination occupied)
        loc_to_lane: dict[LocationAddress, LaneAddress] = {}
        invalid_locs: set[LocationAddress] = set()
        for loc in src_locs:
            lane = LaneAddress(move_type, loc.word_id, loc.site_id, bus_id, direction)
            loc_to_lane[loc] = lane

            # If source is occupied, check if destination is also occupied
            if loc in occupied:
                _, dst = self.arch_spec.get_endpoints(lane)
                if dst in occupied:
                    invalid_locs.add(loc)

        # Enumerate all X × Y subset combinations within capacity
        max_nx = max_x_capacity if max_x_capacity is not None else len(sorted_xs)
        max_ny = max_y_capacity if max_y_capacity is not None else len(sorted_ys)

        for nx, ny in product(range(1, max_nx + 1), range(1, max_ny + 1)):
            for ny in range(1, min(max_ny, len(sorted_ys)) + 1):
                yield from self._enumerate_xy_combinations(
                    combinations(sorted_xs, nx),
                    combinations(sorted_ys, ny),
                    pos_to_loc,
                    loc_to_lane,
                    invalid_locs,
                    occupied,
                )

    def _enumerate_xy_combinations(
        self,
        x_subsets: Iterator[tuple[float, ...]],
        y_subsets: Iterator[tuple[float, ...]],
        pos_to_loc: dict[tuple[float, float], LocationAddress],
        loc_to_lane: dict[LocationAddress, LaneAddress],
        invalid_locs: set[LocationAddress],
        occupied: frozenset[LocationAddress],
    ) -> Iterator[frozenset[LaneAddress]]:
        """Yield valid move sets for all nx × ny rectangles.

        NOTE for Rust port: replace itertools.combinations with Gosper's
        hack for bitmask enumeration of exactly-k-set-bits subsets. This
        avoids iterating all 2^n masks when capacity is small. See #298.
        """

        for x_subset, y_subset in product(x_subsets, y_subsets):
            # Build the rectangle and check validity
            lanes: list[LaneAddress] = []
            valid = True
            has_atom = False

            for loc in map(pos_to_loc.get, product(x_subset, y_subset)):
                if loc is not None and loc not in invalid_locs:
                    # is valid
                    lanes.append(loc_to_lane[loc])
                    has_atom = has_atom or loc in occupied
                else:
                    valid = False

            if valid and has_atom:
                yield frozenset(lanes)

    def _apply_move_set(
        self,
        node: ConfigurationNode,
        move_set: frozenset[LaneAddress],
    ) -> ConfigurationNode | None:
        """Apply a move set to a node, returning a new child or None.

        Resolves lane endpoints and moves atoms. Collision checking is
        handled during move set enumeration (bus src/dst are disjoint,
        so no swap-through conflicts within the same bus group).

        Returns:
            A new ConfigurationNode, or None if the configuration was
            already seen at equal-or-lesser depth.
        """
        # Build the new configuration by applying moves
        new_config = dict(node.configuration)

        for lane in move_set:
            src, dst = self.arch_spec.get_endpoints(lane)

            qid = node.get_qubit_at(src)
            if qid is None:
                # No atom at source — no-op lane
                continue

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

    def expand_node(
        self,
        node: ConfigurationNode,
        max_x_capacity: int | None = None,
        max_y_capacity: int | None = None,
    ) -> list[ConfigurationNode]:
        """Expand a node by generating all valid children.

        Enumerates all valid move sets from the node's configuration,
        applies each, and returns the list of new child nodes.

        Args:
            node: The node to expand.
            max_x_capacity: AOD X capacity limit (None = unlimited).
            max_y_capacity: AOD Y capacity limit (None = unlimited).

        Returns:
            List of newly created child nodes (may be empty if deadlocked).
        """
        children: list[ConfigurationNode] = []
        for move_set in self._enumerate_compatible_move_sets(
            node, max_x_capacity, max_y_capacity
        ):
            child = self._apply_move_set(node, move_set)
            if child is not None:
                children.append(child)
        return children
