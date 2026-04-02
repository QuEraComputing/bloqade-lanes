"""Breadth-first search traversal."""

from __future__ import annotations

__all__ = ["BFSTraversal", "bfs"]

from collections import deque

from bloqade.lanes.search.configuration import ConfigurationNode
from bloqade.lanes.search.generators import MoveGenerator
from bloqade.lanes.search.traversal.driver import run_frontier_search
from bloqade.lanes.search.traversal.goal import GoalPredicate, SearchResult
from bloqade.lanes.search.traversal.interface import TraversalStrategyABC
from bloqade.lanes.search.tree import ConfigurationTree


class BFSTraversal(TraversalStrategyABC):
    """Breadth-first traversal policy."""

    def search(
        self,
        *,
        tree: ConfigurationTree,
        generator: MoveGenerator,
        goal: GoalPredicate,
        max_expansions: int | None = None,
        max_depth: int | None = None,
    ) -> SearchResult:
        frontier: deque[ConfigurationNode] = deque([tree.root])
        return run_frontier_search(
            tree=tree,
            generator=generator,
            goal=goal,
            pop_next=lambda: frontier.popleft() if frontier else None,
            push_child=frontier.append,
            frontier_has_items=lambda: bool(frontier),
            max_expansions=max_expansions,
            max_depth=max_depth,
            goal_on_pop=False,
        )


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
    return BFSTraversal().search(
        tree=tree,
        generator=generator,
        goal=goal,
        max_expansions=max_expansions,
        max_depth=max_depth,
    )
