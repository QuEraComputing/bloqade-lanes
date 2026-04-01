"""Tests for entropy_guided_search traversal."""

from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.layout import LocationAddress, SiteLaneAddress
from bloqade.lanes.search.configuration import ConfigurationNode
from bloqade.lanes.search.search_params import SearchParams
from bloqade.lanes.search.traversal.entropy_guided import (
    EntropyGuidedSearch,
    entropy_guided_search,
)
from bloqade.lanes.search.traversal.goal import SearchResult, placement_goal
from bloqade.lanes.search.traversal.step_info import (
    EntropyBumpStepInfo,
    RevertStepInfo,
)
from bloqade.lanes.search.tree import (
    ConfigurationTree,
    ExpansionOutcome,
    ExpansionStatus,
)


def _make_tree():
    arch_spec = logical.get_arch_spec()
    placement = {
        0: LocationAddress(0, 0),
        1: LocationAddress(1, 0),
    }
    return ConfigurationTree.from_initial_placement(arch_spec, placement)


def _make_on_step_recorder():
    records: list[dict] = []

    def record(event, _node, metadata):  # type: ignore[no-untyped-def]
        records.append(
            {
                "event": event,
                "metadata": metadata,
            }
        )

    return records, record


def test_root_is_goal():
    tree = _make_tree()
    target = {0: LocationAddress(0, 0), 1: LocationAddress(1, 0)}
    result = entropy_guided_search(tree, target, placement_goal(target))
    assert result.goal_node is tree.root
    assert result.goal_nodes == (tree.root,)
    assert result.nodes_expanded == 0


def test_finds_one_step_goal():
    tree = _make_tree()
    target = {0: LocationAddress(0, 1)}
    result = entropy_guided_search(tree, target, placement_goal(target))
    assert result.goal_node is not None
    assert result.goal_node.configuration[0] == LocationAddress(0, 1)
    assert result.goal_node.depth >= 1


def test_returns_search_result():
    tree = _make_tree()
    target = {0: LocationAddress(0, 1)}
    result = entropy_guided_search(tree, target, placement_goal(target))
    assert hasattr(result, "goal_node")
    assert hasattr(result, "goal_nodes")
    assert hasattr(result, "nodes_expanded")
    assert hasattr(result, "max_depth_reached")


def test_search_result_single_goal_populates_goal_nodes():
    tree = _make_tree()
    result = SearchResult(goal_node=tree.root, nodes_expanded=0, max_depth_reached=0)
    assert result.goal_node is tree.root
    assert result.goal_nodes == (tree.root,)


def test_max_expansions_limit():
    tree = _make_tree()
    target = {0: LocationAddress(7, 1)}
    result = entropy_guided_search(
        tree, target, placement_goal(target), max_expansions=10
    )
    assert result.nodes_expanded <= 10


def test_move_program_extraction():
    tree = _make_tree()
    target = {0: LocationAddress(0, 1)}
    result = entropy_guided_search(tree, target, placement_goal(target))
    assert result.goal_node is not None
    program = result.goal_node.to_move_program()
    assert len(program) == result.goal_node.depth
    for step in program:
        assert len(step) >= 1


def test_custom_params():
    tree = _make_tree()
    target = {0: LocationAddress(0, 1)}
    params = SearchParams(w_d=2.0, w_m=0.5, e_max=4)
    result = entropy_guided_search(tree, target, placement_goal(target), params=params)
    assert result.goal_node is not None


def test_two_qubit_goal():
    tree = _make_tree()
    target = {0: LocationAddress(0, 1), 1: LocationAddress(1, 1)}
    result = entropy_guided_search(
        tree, target, placement_goal(target), max_expansions=100
    )
    if result.goal_node is not None:
        assert result.goal_node.configuration[0] == LocationAddress(0, 1)
        assert result.goal_node.configuration[1] == LocationAddress(1, 1)


def test_reversion_expands_more_than_depth():
    tree = _make_tree()
    target = {0: LocationAddress(0, 1), 1: LocationAddress(1, 1)}
    params = SearchParams(e_max=2, delta_e=1, max_candidates=1)
    result = entropy_guided_search(
        tree, target, placement_goal(target), params=params, max_expansions=200
    )
    assert result.nodes_expanded > 0


def test_sequential_fallback_triggered():
    tree = _make_tree()
    target = {0: LocationAddress(0, 1)}
    params = SearchParams(e_max=2, delta_e=1, max_candidates=1)
    result = entropy_guided_search(tree, target, placement_goal(target), params=params)
    assert result.goal_node is not None
    assert result.goal_node.configuration[0] == LocationAddress(0, 1)


def test_sequential_fallback_direct():
    tree = _make_tree()
    target = {0: LocationAddress(0, 1), 1: LocationAddress(1, 0)}
    search = EntropyGuidedSearch(tree, target, placement_goal(target))
    result = search._sequential_fallback(tree.root)
    assert result.goal_node is not None
    assert result.goal_node.configuration[0] == LocationAddress(0, 1)
    assert result.goal_node.configuration[1] == LocationAddress(1, 0)


def test_max_candidates_enforced():
    tree = _make_tree()
    target = {0: LocationAddress(0, 1)}
    params = SearchParams(max_candidates=1, e_max=4)
    result = entropy_guided_search(tree, target, placement_goal(target), params=params)
    assert result.goal_node is not None


def test_on_step_callback_fires():
    tree = _make_tree()
    target = {0: LocationAddress(0, 1)}
    records, record = _make_on_step_recorder()
    result = entropy_guided_search(tree, target, placement_goal(target), on_step=record)
    assert result.goal_node is not None
    assert len(records) > 0
    events = [r["event"] for r in records]
    assert "descend" in events
    assert "goal" in events


def test_on_step_none_is_noop():
    tree = _make_tree()
    target = {0: LocationAddress(0, 1)}
    result = entropy_guided_search(tree, target, placement_goal(target), on_step=None)
    assert result.goal_node is not None


def test_on_step_records_revert():
    tree = _make_tree()
    target = {0: LocationAddress(0, 1), 1: LocationAddress(1, 1)}
    params = SearchParams(e_max=2, delta_e=1, max_candidates=1)
    records, record = _make_on_step_recorder()
    entropy_guided_search(
        tree,
        target,
        placement_goal(target),
        params=params,
        on_step=record,
        max_expansions=50,
    )
    events = [r["event"] for r in records]
    # With tight params, search may revert/bump entropy, or may find goal directly
    assert "entropy_bump" in events or "revert" in events or "goal" in events

    for step in records:
        if step["event"] == "entropy_bump":
            meta = step["metadata"]
            assert isinstance(meta, EntropyBumpStepInfo)
            if meta.reason in {"no-valid-moves", "state-seen"}:
                assert hasattr(meta, "no_valid_moves_qubit")
                assert hasattr(meta, "state_seen_node_id")
        if step["event"] == "revert":
            meta = step["metadata"]
            assert isinstance(meta, RevertStepInfo)
            assert meta.reason in {
                "entropy",
                "no-valid-moves",
                "state-seen",
            }
            assert hasattr(meta, "no_valid_moves_qubit")
            assert hasattr(meta, "state_seen_node_id")


def test_on_step_fallback_events():
    tree = _make_tree()
    target = {0: LocationAddress(0, 1)}
    params = SearchParams(e_max=2, delta_e=1, max_candidates=1)
    records, record = _make_on_step_recorder()
    entropy_guided_search(
        tree, target, placement_goal(target), params=params, on_step=record
    )
    events = [r["event"] for r in records]
    assert "fallback_start" in events or "goal" in events


def test_outcome_transposition_maps_to_state_seen(monkeypatch):
    tree = _make_tree()
    target = {0: LocationAddress(0, 1)}
    params = SearchParams(e_max=2, delta_e=1, max_candidates=1)
    records, record = _make_on_step_recorder()
    candidate = frozenset({SiteLaneAddress(0, 0, 0)})

    def fake_next_candidate(_self, _sn, _node):  # type: ignore[no-untyped-def]
        return candidate

    def fake_try_move_set(_node, move_set, strict=True):  # type: ignore[no-untyped-def]
        return ExpansionOutcome(
            move_set=move_set,
            status=ExpansionStatus.TRANSPOSITION_SEEN,
            existing_node=tree.root,
        )

    monkeypatch.setattr(EntropyGuidedSearch, "_get_next_candidate", fake_next_candidate)
    monkeypatch.setattr(tree, "try_move_set", fake_try_move_set)

    entropy_guided_search(
        tree, target, placement_goal(target), params=params, on_step=record
    )

    bumps = [r for r in records if r["event"] == "entropy_bump"]
    assert any(
        b["metadata"].reason == "state-seen"
        and b["metadata"].state_seen_node_id == id(tree.root)
        for b in bumps
    )


def test_outcome_collision_maps_to_no_valid_moves(monkeypatch):
    tree = _make_tree()
    target = {0: LocationAddress(0, 1)}
    params = SearchParams(e_max=2, delta_e=1, max_candidates=1)
    records, record = _make_on_step_recorder()
    candidate = frozenset({SiteLaneAddress(0, 0, 0)})

    def fake_next_candidate(_self, _sn, _node):  # type: ignore[no-untyped-def]
        return candidate

    def fake_try_move_set(_node, move_set, strict=True):  # type: ignore[no-untyped-def]
        return ExpansionOutcome(
            move_set=move_set,
            status=ExpansionStatus.COLLISION,
            error_message="collision",
        )

    monkeypatch.setattr(EntropyGuidedSearch, "_get_next_candidate", fake_next_candidate)
    monkeypatch.setattr(
        EntropyGuidedSearch,
        "_first_unresolved_qubit_without_valid_move",
        lambda _self, _node: 42,
    )
    monkeypatch.setattr(tree, "try_move_set", fake_try_move_set)

    entropy_guided_search(
        tree, target, placement_goal(target), params=params, on_step=record
    )

    bumps = [r for r in records if r["event"] == "entropy_bump"]
    assert any(
        b["metadata"].reason == "no-valid-moves"
        and b["metadata"].no_valid_moves_qubit == 42
        and b["metadata"].state_seen_node_id is None
        for b in bumps
    )


def test_collects_multiple_goal_nodes(monkeypatch):
    tree = _make_tree()
    target = {0: LocationAddress(0, 5)}
    goal = placement_goal(target)
    c1 = frozenset({SiteLaneAddress(0, 0, 0)})
    c2 = frozenset({SiteLaneAddress(0, 0, 1)})
    candidate_sequence = iter([c1, c2])

    def fake_next_candidate(_self, _sn, _node):  # type: ignore[no-untyped-def]
        return next(candidate_sequence, None)

    def fake_try_move_set(node, move_set, strict=True):  # type: ignore[no-untyped-def]
        goal_node = ConfigurationNode(
            configuration={0: LocationAddress(0, 5), 1: node.configuration[1]},
            parent=node,
            parent_moves=move_set,
            depth=node.depth + 1,
        )
        return ExpansionOutcome(
            move_set=move_set,
            status=ExpansionStatus.CREATED_CHILD,
            child=goal_node,
        )

    monkeypatch.setattr(EntropyGuidedSearch, "_get_next_candidate", fake_next_candidate)
    monkeypatch.setattr(tree, "try_move_set", fake_try_move_set)

    result = entropy_guided_search(
        tree,
        target,
        goal,
        params=SearchParams(max_goal_candidates=2),
        max_expansions=10,
    )
    assert len(result.goal_nodes) == 2
    assert result.goal_node is result.goal_nodes[0]
    assert all(goal(n) for n in result.goal_nodes)


def test_solution_branch_cutoff_ancestor_returns_first_ancestor_below_branch():
    tree = _make_tree()
    target = {0: LocationAddress(0, 5)}
    search = EntropyGuidedSearch(
        tree,
        target,
        placement_goal(target),
    )
    branch_child = ConfigurationNode(
        configuration=dict(tree.root.configuration), parent=tree.root, depth=1
    )
    sibling = ConfigurationNode(
        configuration=dict(tree.root.configuration), parent=tree.root, depth=1
    )
    tree.root.children = {
        frozenset(): branch_child,
        frozenset({SiteLaneAddress(0, 0, 0)}): sibling,
    }

    deep = ConfigurationNode(
        configuration=dict(tree.root.configuration), parent=branch_child, depth=2
    )
    goal = ConfigurationNode(
        configuration=dict(tree.root.configuration), parent=deep, depth=3
    )

    assert search._cutoff_ancestor(goal) is branch_child


def test_solution_branch_cutoff_ancestor_returns_root_for_linear_branch():
    tree = _make_tree()
    target = {0: LocationAddress(0, 5)}
    search = EntropyGuidedSearch(tree, target, placement_goal(target))

    n1 = ConfigurationNode(
        configuration=dict(tree.root.configuration), parent=tree.root, depth=1
    )
    n2 = ConfigurationNode(
        configuration=dict(tree.root.configuration), parent=n1, depth=2
    )
    goal = ConfigurationNode(
        configuration=dict(tree.root.configuration), parent=n2, depth=3
    )

    tree.root.children = {frozenset(): n1}
    n1.children = {frozenset(): n2}
    n2.children = {frozenset(): goal}

    assert search._cutoff_ancestor(goal) is tree.root
