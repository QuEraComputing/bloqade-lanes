"""Tests for goal predicates."""

from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.layout import LocationAddress
from bloqade.lanes.search.configuration import ConfigurationNode
from bloqade.lanes.search.traversal import (
    partial_placement_goal,
    placement_goal,
    zone_goal,
)


def _make_node(config: dict[int, LocationAddress]) -> ConfigurationNode:
    return ConfigurationNode(configuration=config)


def test_placement_goal_all_match():
    target = {0: LocationAddress(0, 0, 0), 1: LocationAddress(0, 1, 0)}
    goal = placement_goal(target)

    node = _make_node({0: LocationAddress(0, 0, 0), 1: LocationAddress(0, 1, 0)})
    assert goal(node) is True


def test_placement_goal_partial_match():
    target = {0: LocationAddress(0, 0, 0), 1: LocationAddress(0, 1, 0)}
    goal = placement_goal(target)

    node = _make_node({0: LocationAddress(0, 0, 0), 1: LocationAddress(0, 0, 1)})
    assert goal(node) is False


def test_placement_goal_ignores_extra_qubits():
    target = {0: LocationAddress(0, 0, 0)}
    goal = placement_goal(target)

    # Qubit 1 is extra — should be ignored
    node = _make_node({0: LocationAddress(0, 0, 0), 1: LocationAddress(0, 1, 0)})
    assert goal(node) is True


def test_partial_placement_goal_min_placed():
    target = {
        0: LocationAddress(0, 0, 0),
        1: LocationAddress(0, 2, 0),
        2: LocationAddress(0, 0, 1),
    }
    goal = partial_placement_goal(target, min_placed=2)

    # 2 out of 3 at target
    node = _make_node(
        {
            0: LocationAddress(0, 0, 0),
            1: LocationAddress(0, 2, 0),
            2: LocationAddress(0, 2, 1),  # not at target
        }
    )
    assert goal(node) is True


def test_partial_placement_goal_not_enough():
    target = {
        0: LocationAddress(0, 0, 0),
        1: LocationAddress(0, 2, 0),
        2: LocationAddress(0, 0, 1),
    }
    goal = partial_placement_goal(target, min_placed=3)

    node = _make_node(
        {
            0: LocationAddress(0, 0, 0),
            1: LocationAddress(0, 2, 0),
            2: LocationAddress(0, 2, 1),
        }
    )
    assert goal(node) is False


def test_partial_placement_goal_none_means_all():
    target = {0: LocationAddress(0, 0, 0)}
    goal = partial_placement_goal(target, min_placed=None)

    node_match = _make_node({0: LocationAddress(0, 0, 0)})
    node_miss = _make_node({0: LocationAddress(0, 0, 1)})
    assert goal(node_match) is True
    assert goal(node_miss) is False


def test_zone_goal_all_in_zone():
    arch_spec = logical.get_arch_spec()
    # Zone 0 contains even-col words [0, 2, 4, 6, 8] in the logical arch
    goal = zone_goal(0, arch_spec)

    node = _make_node({0: LocationAddress(0, 0, 0), 1: LocationAddress(0, 2, 0)})
    assert goal(node) is True


def test_zone_goal_some_outside():
    arch_spec = logical.get_arch_spec()
    goal = zone_goal(0, arch_spec)

    # Word 99 is not in any zone
    node = _make_node({0: LocationAddress(0, 0, 0), 1: LocationAddress(0, 99, 0)})
    assert goal(node) is False
