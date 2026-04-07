"""Goal predicates, type aliases, and shared infrastructure for traversals."""

from __future__ import annotations

__all__ = [
    "CostFunction",
    "GoalPredicate",
    "HeuristicFunction",
    "PriorityEntry",
    "SearchResult",
    "partial_placement_goal",
    "placement_goal",
    "zone_goal",
]

from dataclasses import dataclass, field
from typing import Callable

from bloqade.lanes.layout import LocationAddress
from bloqade.lanes.layout.arch import ArchSpec
from bloqade.lanes.search.configuration import ConfigurationNode

GoalPredicate = Callable[[ConfigurationNode], bool]
"""Returns True if the node satisfies the search goal."""

CostFunction = Callable[[ConfigurationNode], float]
"""Computes the accumulated cost of reaching a node. Lower is better."""

HeuristicFunction = Callable[[ConfigurationNode], float]
"""Estimates the cost from a node to the goal. Lower is better."""


@dataclass(init=False)
class SearchResult:
    """Result of a search strategy."""

    nodes_expanded: int
    """Total number of nodes expanded during search."""

    max_depth_reached: int
    """Maximum depth reached during search."""

    goal_nodes: tuple[ConfigurationNode, ...]
    """Goal nodes found during search, ordered best to worst."""

    def __init__(
        self,
        *,
        nodes_expanded: int,
        max_depth_reached: int,
        goal_nodes: tuple[ConfigurationNode, ...] = (),
        goal_node: ConfigurationNode | None = None,
    ) -> None:
        # Canonicalize all results into goal_nodes so single/multi-solution
        # handling uses one representation everywhere.
        if goal_node is not None:
            if goal_nodes and goal_nodes[0] is not goal_node:
                raise ValueError(
                    "goal_node must match first item in goal_nodes when both are set"
                )
            goal_nodes = (goal_node, *goal_nodes[1:]) if goal_nodes else (goal_node,)

        self.nodes_expanded = nodes_expanded
        self.max_depth_reached = max_depth_reached
        self.goal_nodes = goal_nodes

    @property
    def goal_node(self) -> ConfigurationNode | None:
        """Best goal node, or None if no goal was found."""
        if not self.goal_nodes:
            return None
        return self.goal_nodes[0]


@dataclass(order=True)
class PriorityEntry:
    """Heap entry for priority-based search."""

    priority: float
    node: ConfigurationNode = field(compare=False)


def placement_goal(target: dict[int, LocationAddress]) -> GoalPredicate:
    """Goal: all specified qubits are at their target locations.

    Every qubit in `target` must be at the exact location. Qubits not
    in `target` are ignored.

    Args:
        target: Mapping of qubit ID to desired location.

    Returns:
        A GoalPredicate that returns True when all targets are met.
    """

    def goal(node: ConfigurationNode) -> bool:
        return all(node.configuration.get(qid) == loc for qid, loc in target.items())

    return goal


def partial_placement_goal(
    target: dict[int, LocationAddress],
    min_placed: int | None = None,
) -> GoalPredicate:
    """Goal: at least some qubits are at their target locations.

    If `min_placed` is None, all qubits in `target` must be placed
    (same as `placement_goal`). Otherwise, at least `min_placed`
    qubits must be at their target.

    Args:
        target: Mapping of qubit ID to desired location.
        min_placed: Minimum number of qubits that must be at their
            target. None means all.

    Returns:
        A GoalPredicate.
    """
    required = min_placed if min_placed is not None else len(target)

    def goal(node: ConfigurationNode) -> bool:
        placed = sum(
            1 for qid, loc in target.items() if node.configuration.get(qid) == loc
        )
        return placed >= required

    return goal


def zone_goal(zone_id: int, arch_spec: ArchSpec) -> GoalPredicate:
    """Goal: all qubits are located in the specified zone.

    A qubit is "in the zone" if its location's word_id is in the
    zone's word list.

    Args:
        zone_id: The zone to target.
        arch_spec: Architecture specification (for zone → word mapping).

    Returns:
        A GoalPredicate that returns True when all qubits are in the zone.
    """
    zone = arch_spec.zones[zone_id]
    zone_words = set(zone.words_with_site_buses)

    def goal(node: ConfigurationNode) -> bool:
        return all(loc.word_id in zone_words for loc in node.configuration.values())

    return goal
