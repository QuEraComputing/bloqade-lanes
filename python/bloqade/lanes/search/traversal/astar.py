"""A* search traversal."""

from __future__ import annotations

__all__ = ["astar"]

import heapq

from bloqade.lanes.search.generators import MoveGenerator
from bloqade.lanes.search.traversal.goal import (
    CostFunction,
    GoalPredicate,
    HeuristicFunction,
    PriorityEntry,
    SearchResult,
)
from bloqade.lanes.search.tree import ConfigurationTree


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

    frontier: list[PriorityEntry] = []
    f_score = cost_fn(tree.root) + heuristic(tree.root)
    heapq.heappush(frontier, PriorityEntry(f_score, tree.root))

    nodes_expanded = 0
    max_depth = 0

    while frontier:
        if max_expansions is not None and nodes_expanded >= max_expansions:
            break

        entry = heapq.heappop(frontier)
        node = entry.node

        # A* guarantees optimality when the goal is popped from the
        # frontier (confirmed lowest f-score), not when first generated.
        if goal(node):
            return SearchResult(
                goal_node=node,
                nodes_expanded=nodes_expanded,
                max_depth_reached=max(max_depth, node.depth),
            )

        nodes_expanded += 1
        max_depth = max(max_depth, node.depth)

        for child in tree.expand_node(node, generator, strict=False):
            f_score = cost_fn(child) + heuristic(child)
            heapq.heappush(frontier, PriorityEntry(f_score, child))

    return SearchResult(
        goal_node=None,
        nodes_expanded=nodes_expanded,
        max_depth_reached=max_depth,
    )
