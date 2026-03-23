"""Search strategies for the configuration tree."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Callable

from bloqade.lanes.search.configuration import ConfigurationNode
from bloqade.lanes.search.generators import MoveGenerator
from bloqade.lanes.search.tree import ConfigurationTree

GoalPredicate = Callable[[ConfigurationNode], bool]
"""Returns True if the node satisfies the search goal."""

HeuristicFunction = Callable[[ConfigurationNode], float]
"""Estimates the cost from a node to the goal. Lower is better."""


@dataclass
class SearchResult:
    """Result of a search strategy."""

    goal_node: ConfigurationNode | None
    """The node that satisfied the goal, or None if not found."""

    nodes_expanded: int
    """Total number of nodes expanded during search."""

    max_depth_reached: int
    """Maximum depth reached during search."""


@dataclass(order=True)
class _PriorityEntry:
    """Heap entry for priority-based search."""

    priority: float
    node: ConfigurationNode = field(compare=False)


def greedy_best_first(
    tree: ConfigurationTree,
    generator: MoveGenerator,
    goal: GoalPredicate,
    heuristic: HeuristicFunction,
    max_expansions: int | None = None,
) -> SearchResult:
    """Greedy best-first search using heuristic only.

    Expands the node with the lowest heuristic value first. Does not
    consider accumulated path cost — purely greedy. Fast but not
    guaranteed to find the optimal (shortest) solution.

    Args:
        tree: The configuration tree to search.
        generator: Move generator for producing candidates.
        goal: Predicate that returns True for goal configurations.
        heuristic: Estimates cost to goal. Lower is better.
        max_expansions: Maximum number of nodes to expand before
            giving up. None means no limit.

    Returns:
        SearchResult with the goal node (or None if not found).
    """
    # Check if root is already the goal
    if goal(tree.root):
        return SearchResult(goal_node=tree.root, nodes_expanded=0, max_depth_reached=0)

    frontier: list[_PriorityEntry] = []
    heapq.heappush(frontier, _PriorityEntry(heuristic(tree.root), tree.root))

    nodes_expanded = 0
    max_depth = 0

    while frontier:
        if max_expansions is not None and nodes_expanded >= max_expansions:
            break

        entry = heapq.heappop(frontier)
        node = entry.node

        nodes_expanded += 1
        max_depth = max(max_depth, node.depth)

        for child in tree.expand_node(node, generator):
            if goal(child):
                return SearchResult(
                    goal_node=child,
                    nodes_expanded=nodes_expanded,
                    max_depth_reached=child.depth,
                )
            heapq.heappush(frontier, _PriorityEntry(heuristic(child), child))

    return SearchResult(
        goal_node=None,
        nodes_expanded=nodes_expanded,
        max_depth_reached=max_depth,
    )
