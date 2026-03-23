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
                collision (occupied destination or duplicate destination).
        """
        new_config = dict(node.configuration)
        destinations: set[LocationAddress] = set()
        occupied = node.occupied_locations

        for lane in move_set:
            src, dst = self.arch_spec.get_endpoints(lane)

            qid = node.get_qubit_at(src)
            if qid is None:
                continue

            if dst in occupied or dst in destinations:
                if strict:
                    blocker = node.get_qubit_at(dst)
                    raise InvalidMoveError(
                        f"Collision: qubit {qid} moving to {dst!r} "
                        f"which is occupied by qubit {blocker}"
                    )
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
