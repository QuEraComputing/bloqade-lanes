"""Tests for search strategies."""

from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.layout import LocationAddress
from bloqade.lanes.search.generators import ExhaustiveMoveGenerator
from bloqade.lanes.search.strategies import greedy_best_first
from bloqade.lanes.search.tree import ConfigurationTree


def _make_tree() -> ConfigurationTree:
    arch_spec = logical.get_arch_spec()
    placement = {
        0: LocationAddress(0, 0),
        1: LocationAddress(1, 0),
    }
    return ConfigurationTree.from_initial_placement(arch_spec, placement)


def test_greedy_root_is_goal():
    """If the root already satisfies the goal, return immediately."""
    tree = _make_tree()
    gen = ExhaustiveMoveGenerator(max_x_capacity=1, max_y_capacity=1)

    result = greedy_best_first(tree, gen, goal=lambda _: True, heuristic=lambda _: 0.0)

    assert result.goal_node is tree.root
    assert result.nodes_expanded == 0


def test_greedy_finds_goal_one_step():
    """Search should find a goal reachable in one move."""
    tree = _make_tree()
    gen = ExhaustiveMoveGenerator(max_x_capacity=1, max_y_capacity=1)

    # Goal: qubit 0 at (0, 5) — one site bus forward move from (0, 0)
    target = LocationAddress(0, 5)

    def goal(node):
        return node.configuration.get(0) == target

    # Heuristic: 0 if at target, 1 otherwise
    def heuristic(node):
        return 0.0 if node.configuration.get(0) == target else 1.0

    result = greedy_best_first(tree, gen, goal=goal, heuristic=heuristic)

    assert result.goal_node is not None
    assert result.goal_node.configuration[0] == target
    assert result.goal_node.depth == 1
    assert result.nodes_expanded >= 1


def test_greedy_max_expansions_limit():
    """Search should stop after max_expansions."""
    tree = _make_tree()
    gen = ExhaustiveMoveGenerator(max_x_capacity=1, max_y_capacity=1)

    # Unreachable goal
    result = greedy_best_first(
        tree,
        gen,
        goal=lambda _: False,
        heuristic=lambda _: 1.0,
        max_expansions=5,
    )

    assert result.goal_node is None
    assert result.nodes_expanded <= 5


def test_greedy_returns_search_stats():
    """SearchResult should contain valid statistics."""
    tree = _make_tree()
    gen = ExhaustiveMoveGenerator(max_x_capacity=1, max_y_capacity=1)

    result = greedy_best_first(
        tree,
        gen,
        goal=lambda _: False,
        heuristic=lambda _: 1.0,
        max_expansions=3,
    )

    assert result.nodes_expanded <= 3
    assert result.max_depth_reached >= 0


def test_greedy_move_program_extraction():
    """The goal node should produce a valid move program."""
    tree = _make_tree()
    gen = ExhaustiveMoveGenerator(max_x_capacity=1, max_y_capacity=1)

    target = LocationAddress(0, 5)

    result = greedy_best_first(
        tree,
        gen,
        goal=lambda node: node.configuration.get(0) == target,
        heuristic=lambda node: 0.0 if node.configuration.get(0) == target else 1.0,
    )

    assert result.goal_node is not None
    program = result.goal_node.to_move_program()
    assert len(program) == result.goal_node.depth
    # Each step should have at least one lane
    for step in program:
        assert len(step) >= 1
