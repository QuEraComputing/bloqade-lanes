"""Tree traversal strategies for the configuration search."""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from bloqade.lanes.search.configuration import ConfigurationNode
from bloqade.lanes.search.generators import MoveGenerator
from bloqade.lanes.search.tree import ConfigurationTree

GoalPredicate = Callable[[ConfigurationNode], bool]
"""Returns True if the node satisfies the search goal."""

CostFunction = Callable[[ConfigurationNode], float]
"""Computes the accumulated cost of reaching a node. Lower is better."""

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


def bfs(
    tree: ConfigurationTree,
    generator: MoveGenerator,
    goal: GoalPredicate,
    max_expansions: int | None = None,
    max_depth: int | None = None,
) -> SearchResult:
    """Breadth-first search.

    Explores nodes level by level (shortest path first). Guarantees
    finding the shallowest goal if one exists within the depth limit.

    Args:
        tree: The configuration tree to search.
        generator: Move generator for producing candidates.
        goal: Predicate that returns True for goal configurations.
        max_expansions: Maximum nodes to expand. None means no limit.
        max_depth: Maximum depth to explore. None means no limit.

    Returns:
        SearchResult with the goal node (or None if not found).
    """
    if goal(tree.root):
        return SearchResult(goal_node=tree.root, nodes_expanded=0, max_depth_reached=0)

    frontier: deque[ConfigurationNode] = deque([tree.root])
    nodes_expanded = 0
    reached_depth = 0

    while frontier:
        if max_expansions is not None and nodes_expanded >= max_expansions:
            break

        node = frontier.popleft()
        nodes_expanded += 1
        reached_depth = max(reached_depth, node.depth)

        if max_depth is not None and node.depth >= max_depth:
            continue

        for child in tree.expand_node(node, generator, strict=False):
            if goal(child):
                return SearchResult(
                    goal_node=child,
                    nodes_expanded=nodes_expanded,
                    max_depth_reached=child.depth,
                )
            frontier.append(child)

    return SearchResult(
        goal_node=None,
        nodes_expanded=nodes_expanded,
        max_depth_reached=reached_depth,
    )


def astar(
    tree: ConfigurationTree,
    generator: MoveGenerator,
    goal: GoalPredicate,
    heuristic: HeuristicFunction,
    cost: CostFunction | None = None,
    max_expansions: int | None = None,
) -> SearchResult:
    """A* search.

    Expands the node with the lowest `cost(node) + heuristic(node)`
    first. With an admissible heuristic (never overestimates), A*
    guarantees finding the optimal (lowest cost) solution.

    Args:
        tree: The configuration tree to search.
        generator: Move generator for producing candidates.
        goal: Predicate that returns True for goal configurations.
        heuristic: Estimates cost to goal. Must be admissible for
            optimality. Lower is better.
        cost: Accumulated cost function. Defaults to node depth if None.
        max_expansions: Maximum nodes to expand. None means no limit.

    Returns:
        SearchResult with the goal node (or None if not found).
    """
    cost_fn: CostFunction = cost if cost is not None else lambda node: float(node.depth)

    if goal(tree.root):
        return SearchResult(goal_node=tree.root, nodes_expanded=0, max_depth_reached=0)

    frontier: list[_PriorityEntry] = []
    f_score = cost_fn(tree.root) + heuristic(tree.root)
    heapq.heappush(frontier, _PriorityEntry(f_score, tree.root))

    nodes_expanded = 0
    max_depth = 0

    while frontier:
        if max_expansions is not None and nodes_expanded >= max_expansions:
            break

        entry = heapq.heappop(frontier)
        node = entry.node

        nodes_expanded += 1
        max_depth = max(max_depth, node.depth)

        for child in tree.expand_node(node, generator, strict=False):
            if goal(child):
                return SearchResult(
                    goal_node=child,
                    nodes_expanded=nodes_expanded,
                    max_depth_reached=child.depth,
                )
            f_score = cost_fn(child) + heuristic(child)
            heapq.heappush(frontier, _PriorityEntry(f_score, child))

    return SearchResult(
        goal_node=None,
        nodes_expanded=nodes_expanded,
        max_depth_reached=max_depth,
    )


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

        for child in tree.expand_node(node, generator, strict=False):
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
