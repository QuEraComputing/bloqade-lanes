"""Traversal interfaces for configuration-tree search."""

from __future__ import annotations

import abc

from bloqade.lanes.search.generators import MoveGenerator
from bloqade.lanes.search.traversal.goal import GoalPredicate, SearchResult
from bloqade.lanes.search.tree import ConfigurationTree


class TraversalStrategyABC(abc.ABC):
    """Strategy interface for traversing a configuration tree.

    Traversal strategies choose which frontier node to examine next.
    Candidate move generation is delegated to the provided MoveGenerator.
    """

    @abc.abstractmethod
    def search(
        self,
        *,
        tree: ConfigurationTree,
        generator: MoveGenerator,
        goal: GoalPredicate,
        max_expansions: int | None = None,
        max_depth: int | None = None,
    ) -> SearchResult:
        """Execute search over the tree using the supplied generator."""
        ...
