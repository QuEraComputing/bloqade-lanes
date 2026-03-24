"""Greedy best-first search traversal."""

from __future__ import annotations

__all__ = ["greedy_best_first"]

import heapq

from bloqade.lanes.search.generators import MoveGenerator
from bloqade.lanes.search.traversal.goal import (
    GoalPredicate,
    HeuristicFunction,
    PriorityEntry,
    SearchResult,
)
from bloqade.lanes.search.tree import ConfigurationTree


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

    frontier: list[PriorityEntry] = []
    heapq.heappush(frontier, PriorityEntry(heuristic(tree.root), tree.root))

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
                    max_depth_reached=max(max_depth, child.depth),
                )
            heapq.heappush(frontier, PriorityEntry(heuristic(child), child))

    return SearchResult(
        goal_node=None,
        nodes_expanded=nodes_expanded,
        max_depth_reached=max_depth,
    )
