"""Tests for tree traversal strategies."""

from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.layout import LocationAddress
from bloqade.lanes.search.generators import ExhaustiveMoveGenerator
from bloqade.lanes.search.traversal import astar, bfs, greedy_best_first
from bloqade.lanes.search.tree import ConfigurationTree


def _make_tree() -> ConfigurationTree:
    arch_spec = logical.get_arch_spec()
    placement = {
        0: LocationAddress(0, 0),
        1: LocationAddress(1, 0),
    }
    return ConfigurationTree.from_initial_placement(arch_spec, placement)


_TARGET = LocationAddress(0, 1)
_GEN = ExhaustiveMoveGenerator(max_x_capacity=1, max_y_capacity=1)


def _goal(node):
    return node.configuration.get(0) == _TARGET


def _heuristic(node):
    return 0.0 if node.configuration.get(0) == _TARGET else 1.0


# ── greedy_best_first ──


def test_greedy_root_is_goal():
    tree = _make_tree()
    result = greedy_best_first(tree, _GEN, goal=lambda _: True, heuristic=lambda _: 0.0)
    assert result.goal_node is tree.root
    assert result.nodes_expanded == 0


def test_greedy_finds_goal_one_step():
    tree = _make_tree()
    result = greedy_best_first(tree, _GEN, goal=_goal, heuristic=_heuristic)
    assert result.goal_node is not None
    assert result.goal_node.configuration[0] == _TARGET
    assert result.goal_node.depth == 1


def test_greedy_max_expansions_limit():
    tree = _make_tree()
    result = greedy_best_first(
        tree, _GEN, goal=lambda _: False, heuristic=lambda _: 1.0, max_expansions=5
    )
    assert result.goal_node is None
    assert result.nodes_expanded <= 5


def test_greedy_move_program_extraction():
    tree = _make_tree()
    result = greedy_best_first(tree, _GEN, goal=_goal, heuristic=_heuristic)
    assert result.goal_node is not None
    program = result.goal_node.to_move_program()
    assert len(program) == result.goal_node.depth
    for step in program:
        assert len(step) >= 1


# ── bfs ──


def test_bfs_root_is_goal():
    tree = _make_tree()
    result = bfs(tree, _GEN, goal=lambda _: True)
    assert result.goal_node is tree.root
    assert result.nodes_expanded == 0


def test_bfs_finds_goal_one_step():
    tree = _make_tree()
    result = bfs(tree, _GEN, goal=_goal)
    assert result.goal_node is not None
    assert result.goal_node.configuration[0] == _TARGET
    assert result.goal_node.depth == 1


def test_bfs_finds_shallowest():
    """BFS should find the shallowest goal first."""
    tree = _make_tree()
    result = bfs(tree, _GEN, goal=_goal)
    assert result.goal_node is not None
    # BFS guarantees depth 1 for a 1-step reachable goal
    assert result.goal_node.depth == 1


def test_bfs_max_expansions_limit():
    tree = _make_tree()
    result = bfs(tree, _GEN, goal=lambda _: False, max_expansions=5)
    assert result.goal_node is None
    assert result.nodes_expanded <= 5


def test_bfs_max_depth_limit():
    tree = _make_tree()
    # Goal is unreachable at depth 0 (root is not the goal)
    result = bfs(tree, _GEN, goal=_goal, max_depth=0)
    assert result.goal_node is None


def test_bfs_move_program():
    tree = _make_tree()
    result = bfs(tree, _GEN, goal=_goal)
    assert result.goal_node is not None
    program = result.goal_node.to_move_program()
    assert len(program) == result.goal_node.depth


# ── astar ──


def test_astar_root_is_goal():
    tree = _make_tree()
    result = astar(tree, _GEN, goal=lambda _: True, heuristic=lambda _: 0.0)
    assert result.goal_node is tree.root
    assert result.nodes_expanded == 0


def test_astar_finds_goal_one_step():
    tree = _make_tree()
    result = astar(tree, _GEN, goal=_goal, heuristic=_heuristic)
    assert result.goal_node is not None
    assert result.goal_node.configuration[0] == _TARGET
    assert result.goal_node.depth == 1


def test_astar_with_default_cost():
    """A* with default cost (depth) should find optimal solution."""
    tree = _make_tree()
    result = astar(tree, _GEN, goal=_goal, heuristic=_heuristic)
    assert result.goal_node is not None
    assert result.goal_node.depth == 1


def test_astar_with_custom_cost():
    """A* with custom cost function."""
    tree = _make_tree()
    # Cost = 2x depth (penalizes deeper solutions more)
    result = astar(
        tree,
        _GEN,
        goal=_goal,
        heuristic=_heuristic,
        cost=lambda node: float(node.depth) * 2.0,
    )
    assert result.goal_node is not None
    assert result.goal_node.depth == 1


def test_astar_max_expansions_limit():
    tree = _make_tree()
    result = astar(
        tree, _GEN, goal=lambda _: False, heuristic=lambda _: 1.0, max_expansions=5
    )
    assert result.goal_node is None
    assert result.nodes_expanded <= 5


def test_astar_move_program():
    tree = _make_tree()
    result = astar(tree, _GEN, goal=_goal, heuristic=_heuristic)
    assert result.goal_node is not None
    program = result.goal_node.to_move_program()
    assert len(program) == result.goal_node.depth
    for step in program:
        assert len(step) >= 1


def test_astar_multiple_goals_matches_bfs():
    """A* should find the same shallowest goal as BFS when multiple goals exist."""
    tree = _make_tree()

    def multi_goal(node):
        return node.depth in (1, 2)

    bfs_result = bfs(tree, _GEN, goal=multi_goal, max_depth=3)

    # Fresh tree for A* (transposition table is shared)
    tree2 = _make_tree()
    astar_result = astar(tree2, _GEN, goal=multi_goal, heuristic=lambda _: 0.0)

    assert bfs_result.goal_node is not None
    assert astar_result.goal_node is not None
    assert astar_result.goal_node.depth == bfs_result.goal_node.depth
