"""Greedy best-first search traversal."""

from __future__ import annotations

__all__ = ["GreedyBestFirstTraversal", "greedy_best_first"]

import heapq
from dataclasses import dataclass

from bloqade.lanes.search.generators import MoveGenerator
from bloqade.lanes.search.traversal.driver import run_frontier_search
from bloqade.lanes.search.traversal.goal import (
    GoalPredicate,
    HeuristicFunction,
    PriorityEntry,
    SearchResult,
)
from bloqade.lanes.search.traversal.interface import TraversalStrategyABC
from bloqade.lanes.search.tree import ConfigurationTree


@dataclass(frozen=True)
class GreedyBestFirstTraversal(TraversalStrategyABC):
    """Greedy best-first traversal policy."""

    heuristic: HeuristicFunction

    def search(
        self,
        *,
        tree: ConfigurationTree,
        generator: MoveGenerator,
        goal: GoalPredicate,
        max_expansions: int | None = None,
        max_depth: int | None = None,
    ) -> SearchResult:
        _ = max_depth
        frontier: list[PriorityEntry] = []
        heapq.heappush(frontier, PriorityEntry(self.heuristic(tree.root), tree.root))
        return run_frontier_search(
            tree=tree,
            generator=generator,
            goal=goal,
            pop_next=lambda: heapq.heappop(frontier).node if frontier else None,
            push_child=lambda child: heapq.heappush(
                frontier, PriorityEntry(self.heuristic(child), child)
            ),
            frontier_has_items=lambda: bool(frontier),
            max_expansions=max_expansions,
            max_depth=None,
            goal_on_pop=False,
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
    return GreedyBestFirstTraversal(heuristic=heuristic).search(
        tree=tree,
        generator=generator,
        goal=goal,
        max_expansions=max_expansions,
    )
