"""A* search traversal."""

from __future__ import annotations

__all__ = ["AStarTraversal", "astar"]

import heapq
from dataclasses import dataclass

from bloqade.lanes.search.generators import MoveGenerator
from bloqade.lanes.search.traversal.driver import run_frontier_search
from bloqade.lanes.search.traversal.goal import (
    CostFunction,
    GoalPredicate,
    HeuristicFunction,
    PriorityEntry,
    SearchResult,
)
from bloqade.lanes.search.traversal.interface import TraversalStrategyABC
from bloqade.lanes.search.tree import ConfigurationTree


@dataclass(frozen=True)
class AStarTraversal(TraversalStrategyABC):
    """A* traversal policy."""

    heuristic: HeuristicFunction
    cost: CostFunction | None = None

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
        cost_fn: CostFunction = (
            self.cost if self.cost is not None else lambda node: float(node.depth)
        )
        frontier: list[PriorityEntry] = []
        root_f = cost_fn(tree.root) + self.heuristic(tree.root)
        heapq.heappush(frontier, PriorityEntry(root_f, tree.root))
        return run_frontier_search(
            tree=tree,
            generator=generator,
            goal=goal,
            pop_next=lambda: heapq.heappop(frontier).node if frontier else None,
            push_child=lambda child: heapq.heappush(
                frontier, PriorityEntry(cost_fn(child) + self.heuristic(child), child)
            ),
            frontier_has_items=lambda: bool(frontier),
            max_expansions=max_expansions,
            max_depth=None,
            goal_on_pop=True,
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
    return AStarTraversal(heuristic=heuristic, cost=cost).search(
        tree=tree,
        generator=generator,
        goal=goal,
        max_expansions=max_expansions,
    )
