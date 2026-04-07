"""Tests for traversal interface and strategy implementations."""

import pytest

from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.layout import LocationAddress
from bloqade.lanes.search.generators import ExhaustiveMoveGenerator
from bloqade.lanes.search.traversal import (
    AStarTraversal,
    BFSTraversal,
    GreedyBestFirstTraversal,
    TraversalStrategyABC,
)
from bloqade.lanes.search.tree import ConfigurationTree


def _make_tree() -> ConfigurationTree:
    arch_spec = logical.get_arch_spec()
    placement = {
        0: LocationAddress(0, 0, 0),
        1: LocationAddress(0, 1, 0),
    }
    return ConfigurationTree.from_initial_placement(arch_spec, placement)


def test_traversal_interface_is_abstract():
    with pytest.raises(TypeError):
        TraversalStrategyABC()  # type: ignore[abstract]


def test_bfs_traversal_class_runs_search():
    tree = _make_tree()
    target = LocationAddress(0, 0, 1)
    gen = ExhaustiveMoveGenerator(max_x_capacity=1, max_y_capacity=1)
    traversal = BFSTraversal()

    result = traversal.search(
        tree=tree,
        generator=gen,
        goal=lambda node: node.configuration.get(0) == target,
        max_depth=3,
    )

    assert result.goal_node is not None
    assert result.goal_node.configuration[0] == target


def test_greedy_traversal_class_runs_search():
    tree = _make_tree()
    target = LocationAddress(0, 0, 1)
    gen = ExhaustiveMoveGenerator(max_x_capacity=1, max_y_capacity=1)
    traversal = GreedyBestFirstTraversal(
        heuristic=lambda node: 0.0 if node.configuration.get(0) == target else 1.0
    )

    result = traversal.search(
        tree=tree,
        generator=gen,
        goal=lambda node: node.configuration.get(0) == target,
    )

    assert result.goal_node is not None
    assert result.goal_node.configuration[0] == target


def test_astar_traversal_class_runs_search():
    tree = _make_tree()
    target = LocationAddress(0, 0, 1)
    gen = ExhaustiveMoveGenerator(max_x_capacity=1, max_y_capacity=1)
    traversal = AStarTraversal(
        heuristic=lambda node: 0.0 if node.configuration.get(0) == target else 1.0
    )

    result = traversal.search(
        tree=tree,
        generator=gen,
        goal=lambda node: node.configuration.get(0) == target,
    )

    assert result.goal_node is not None
    assert result.goal_node.configuration[0] == target
