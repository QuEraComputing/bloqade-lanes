"""Shared traversal driver for frontier-based searches."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from bloqade.lanes.search.configuration import ConfigurationNode
from bloqade.lanes.search.generators import MoveGenerator
from bloqade.lanes.search.traversal.goal import GoalPredicate, SearchResult
from bloqade.lanes.search.tree import ConfigurationTree


def run_frontier_search(
    *,
    tree: ConfigurationTree,
    generator: MoveGenerator,
    goal: GoalPredicate,
    pop_next: Callable[[], ConfigurationNode | None],
    push_child: Callable[[ConfigurationNode], None],
    frontier_has_items: Callable[[], bool],
    max_expansions: int | None = None,
    max_depth: int | None = None,
    goal_on_pop: bool = False,
) -> SearchResult:
    """Execute a generic frontier-driven search loop.

    The traversal policy is captured by pop_next/push_child/frontier_has_items.
    """
    if goal(tree.root):
        return SearchResult(goal_node=tree.root, nodes_expanded=0, max_depth_reached=0)

    nodes_expanded = 0
    max_depth_reached = 0

    while frontier_has_items():
        if max_expansions is not None and nodes_expanded >= max_expansions:
            break

        node = pop_next()
        if node is None:
            break

        if goal_on_pop and goal(node):
            return SearchResult(
                goal_node=node,
                nodes_expanded=nodes_expanded,
                max_depth_reached=max(max_depth_reached, node.depth),
            )

        nodes_expanded += 1
        max_depth_reached = max(max_depth_reached, node.depth)

        if max_depth is not None and node.depth >= max_depth:
            continue

        children = _expand_children(tree=tree, node=node, generator=generator)
        for child in children:
            if goal(child):
                return SearchResult(
                    goal_node=child,
                    nodes_expanded=nodes_expanded,
                    max_depth_reached=max(max_depth_reached, child.depth),
                )
            push_child(child)

    return SearchResult(
        goal_node=None,
        nodes_expanded=nodes_expanded,
        max_depth_reached=max_depth_reached,
    )


def _expand_children(
    *,
    tree: ConfigurationTree,
    node: ConfigurationNode,
    generator: MoveGenerator,
) -> Iterable[ConfigurationNode]:
    return tree.expand_node(node, generator, strict=False)
