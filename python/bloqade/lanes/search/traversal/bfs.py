"""Breadth-first search traversal."""

from __future__ import annotations

__all__ = ["bfs"]

from collections import deque

from bloqade.lanes.search.configuration import ConfigurationNode
from bloqade.lanes.search.generators import MoveGenerator
from bloqade.lanes.search.traversal.goal import GoalPredicate, SearchResult
from bloqade.lanes.search.tree import ConfigurationTree


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
