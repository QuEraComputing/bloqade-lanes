"""Base protocol and types for move generators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, Protocol, runtime_checkable

from bloqade.lanes.layout import LaneAddress

if TYPE_CHECKING:
    from bloqade.lanes.search.configuration import ConfigurationNode
    from bloqade.lanes.search.tree import ConfigurationTree


class EntropyNode(Protocol):
    """Minimal node metadata required by HeuristicMoveGenerator."""

    entropy: int


@runtime_checkable
class MoveGenerator(Protocol):
    """Interface for generating candidate move sets from a configuration.

    Implementations yield candidate move sets. Validation and node
    creation are handled by ConfigurationTree.expand_node.
    """

    def generate(
        self,
        node: ConfigurationNode,
        tree: ConfigurationTree,
    ) -> Iterator[frozenset[LaneAddress]]:
        """Yield candidate move sets from the given configuration.

        Args:
            node: The configuration to generate moves from.
            tree: The configuration tree (provides arch_spec, path_finder).

        Yields:
            frozenset[LaneAddress] — each candidate parallel move set.
        """
        ...
