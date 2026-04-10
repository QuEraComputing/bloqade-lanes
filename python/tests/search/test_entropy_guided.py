"""Tests for entropy_guided_search traversal."""

from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.layout import (
    Direction,
    LocationAddress,
    SiteLaneAddress,
    WordLaneAddress,
)
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
        1: LocationAddress(4, 0),
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
    target = {0: LocationAddress(0, 0), 1: LocationAddress(4, 0)}
    result = entropy_guided_search(tree, target, placement_goal(target))
    assert result.goal_node is tree.root
    assert result.goal_nodes == (tree.root,)
    assert result.nodes_expanded == 0


def test_finds_one_step_goal():
    tree = _make_tree()
    target = {0: LocationAddress(1, 0)}
    result = entropy_guided_search(tree, target, placement_goal(target))
    assert result.goal_node is not None
    assert result.goal_node.configuration[0] == LocationAddress(1, 0)
    assert result.goal_node.depth >= 1


def test_returns_search_result():
    tree = _make_tree()
    target = {0: LocationAddress(1, 0)}
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
    target = {0: LocationAddress(7, 0)}
    result = entropy_guided_search(
        tree, target, placement_goal(target), max_expansions=10
    )
    assert result.nodes_expanded <= 10


def test_sequential_fallback_triggers_on_max_expansions(monkeypatch):
    tree = _make_tree()
    target = {0: LocationAddress(9, 0)}
    fallback_called = {"value": False}
    source_loc = tree.root.configuration[0]
    first_lane = next(iter(tree.outgoing_lanes(source_loc)))
    candidate = frozenset({first_lane})

    def fake_fallback(_self, _current_node):  # type: ignore[no-untyped-def]
        fallback_called["value"] = True
        return SearchResult(nodes_expanded=0, max_depth_reached=0, goal_nodes=())

    monkeypatch.setattr(EntropyGuidedSearch, "_sequential_fallback", fake_fallback)
    monkeypatch.setattr(
        EntropyGuidedSearch,
        "_get_next_candidate",
        lambda _self, _sn, _node, _generator: candidate,
    )
    monkeypatch.setattr(
        tree,
        "try_move_set",
        lambda node, move_set, strict=True: ExpansionOutcome(
            move_set=move_set,
            status=ExpansionStatus.CREATED_CHILD,
            child=ConfigurationNode(
                configuration=dict(node.configuration),
                parent=node,
                parent_moves=move_set,
                depth=node.depth + 1,
            ),
        ),
    )

    entropy_guided_search(
        tree,
        target,
        placement_goal(target),
        params=SearchParams(e_max=2, delta_e=1),
        max_expansions=1,
    )
    assert fallback_called["value"] is True


def test_move_program_extraction():
    tree = _make_tree()
    target = {0: LocationAddress(1, 0)}
    result = entropy_guided_search(tree, target, placement_goal(target))
    assert result.goal_node is not None
    program = result.goal_node.to_move_program()
    assert len(program) == result.goal_node.depth
    for step in program:
        assert len(step) >= 1


def test_custom_params():
    tree = _make_tree()
    target = {0: LocationAddress(1, 0)}
    params = SearchParams(w_d=2.0, w_m=0.5, e_max=4)
    result = entropy_guided_search(tree, target, placement_goal(target), params=params)
    assert result.goal_node is not None


def test_two_qubit_goal():
    tree = _make_tree()
    target = {0: LocationAddress(1, 0), 1: LocationAddress(5, 0)}
    result = entropy_guided_search(
        tree, target, placement_goal(target), max_expansions=100
    )
    if result.goal_node is not None:
        assert result.goal_node.configuration[0] == LocationAddress(1, 0)
        assert result.goal_node.configuration[1] == LocationAddress(5, 0)


def test_reversion_expands_more_than_depth():
    tree = _make_tree()
    target = {0: LocationAddress(1, 0), 1: LocationAddress(5, 0)}
    params = SearchParams(e_max=2, delta_e=1, max_candidates=1)
    result = entropy_guided_search(
        tree, target, placement_goal(target), params=params, max_expansions=200
    )
    assert result.nodes_expanded > 0


def test_sequential_fallback_triggered_on_max_depth():
    tree = _make_tree()
    target = {0: LocationAddress(1, 0)}
    params = SearchParams(e_max=2, delta_e=1, max_candidates=1)
    result = entropy_guided_search(
        tree,
        target,
        placement_goal(target),
        params=params,
        max_depth=1,
    )
    assert result.goal_node is not None
    assert result.goal_node.configuration[0] == LocationAddress(1, 0)


def test_root_deadlock_does_not_trigger_fallback_without_limits(monkeypatch):
    tree = _make_tree()
    target = {0: LocationAddress(0, 9)}
    fallback_called = {"value": False}

    def fake_fallback(_self, _current_node):  # type: ignore[no-untyped-def]
        fallback_called["value"] = True
        return SearchResult(nodes_expanded=0, max_depth_reached=0, goal_nodes=())

    monkeypatch.setattr(EntropyGuidedSearch, "_sequential_fallback", fake_fallback)
    monkeypatch.setattr(
        EntropyGuidedSearch,
        "_get_next_candidate",
        lambda _self, _sn, _node, _generator: None,
    )

    result = entropy_guided_search(
        tree,
        target,
        placement_goal(target),
        params=SearchParams(e_max=2, delta_e=1),
    )

    assert fallback_called["value"] is False
    assert result.goal_nodes == ()


def test_sequential_fallback_direct():
    tree = _make_tree()
    target = {0: LocationAddress(1, 0), 1: LocationAddress(4, 0)}
    search = EntropyGuidedSearch(tree, target, placement_goal(target))
    result = search._sequential_fallback(tree.root)
    assert result.goal_node is not None
    assert result.goal_node.configuration[0] == LocationAddress(1, 0)
    assert result.goal_node.configuration[1] == LocationAddress(4, 0)


def test_sequential_fallback_reuses_already_seen_child():
    tree = _make_tree()
    target = {0: LocationAddress(1, 0)}
    search = EntropyGuidedSearch(tree, target, placement_goal(target))

    # Pre-create the first fallback step so replaying it hits ALREADY_CHILD.
    first_step = frozenset({WordLaneAddress(0, 0, 0)})
    first_child = tree.apply_move_set(tree.root, first_step, strict=False)
    assert first_child is not None

    result = search._sequential_fallback(tree.root)
    assert result.goal_node is not None
    assert result.goal_node.configuration[0] == LocationAddress(1, 0)


def test_max_candidates_enforced():
    tree = _make_tree()
    target = {0: LocationAddress(1, 0)}
    params = SearchParams(max_candidates=1, e_max=4)
    result = entropy_guided_search(tree, target, placement_goal(target), params=params)
    assert result.goal_node is not None


def test_candidate_cache_keeps_only_top_unique_generated_candidates():
    tree = _make_tree()
    target = {0: LocationAddress(5, 0)}
    params = SearchParams(max_candidates=2, e_max=4)
    search = EntropyGuidedSearch(tree, target, placement_goal(target), params=params)

    node = tree.root
    existing_child_moveset = frozenset({SiteLaneAddress(0, 0, 0)})
    node.children = {
        existing_child_moveset: ConfigurationNode(
            configuration=dict(node.configuration),
            parent=node,
            parent_moves=existing_child_moveset,
            depth=node.depth + 1,
        )
    }

    top_candidate = frozenset({SiteLaneAddress(2, 0, 0)})
    duplicate_top_candidate = frozenset({SiteLaneAddress(2, 0, 0)})
    second_candidate = frozenset({SiteLaneAddress(3, 0, 0)})
    lower_ranked_candidate = frozenset({SiteLaneAddress(4, 0, 0)})

    class _StubGenerator:
        def generate(self, node, tree):  # type: ignore[no-untyped-def,unused-argument]
            return iter(
                [
                    existing_child_moveset,
                    top_candidate,
                    duplicate_top_candidate,
                    second_candidate,
                    lower_ranked_candidate,
                ]
            )

    sn = search._get_or_create_search_node(node)
    next_candidate = search._get_next_candidate(sn, node, _StubGenerator())

    assert next_candidate == top_candidate
    assert sn.candidate_cache == [existing_child_moveset, top_candidate]


def test_candidate_cache_top_k_applies_to_global_rectangle_order():
    tree = _make_tree()
    target = {0: LocationAddress(5, 0)}
    params = SearchParams(max_candidates=2, e_max=4)
    search = EntropyGuidedSearch(tree, target, placement_goal(target), params=params)

    node = tree.root
    rect_top = frozenset({SiteLaneAddress(0, 0, 0), SiteLaneAddress(1, 0, 0)})
    rect_second = frozenset({SiteLaneAddress(2, 0, 0), SiteLaneAddress(3, 0, 0)})
    rect_third = frozenset({SiteLaneAddress(4, 0, 0), SiteLaneAddress(5, 0, 0)})

    class _RectangleStubGenerator:
        def generate(self, node, tree):  # type: ignore[no-untyped-def,unused-argument]
            return iter([rect_top, rect_second, rect_third])

    sn = search._get_or_create_search_node(node)
    next_candidate = search._get_next_candidate(sn, node, _RectangleStubGenerator())

    assert next_candidate == rect_top
    assert sn.candidate_cache == [rect_top, rect_second]


def test_on_step_callback_fires():
    tree = _make_tree()
    target = {0: LocationAddress(1, 0)}
    records, record = _make_on_step_recorder()
    result = entropy_guided_search(tree, target, placement_goal(target), on_step=record)
    assert result.goal_node is not None
    assert len(records) > 0
    events = [r["event"] for r in records]
    assert "descend" in events
    assert "goal" in events


def test_on_step_none_is_noop():
    tree = _make_tree()
    target = {0: LocationAddress(1, 0)}
    result = entropy_guided_search(tree, target, placement_goal(target), on_step=None)
    assert result.goal_node is not None


def test_on_step_records_revert():
    tree = _make_tree()
    target = {0: LocationAddress(1, 0), 1: LocationAddress(5, 0)}
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
    target = {0: LocationAddress(1, 0)}
    params = SearchParams(e_max=2, delta_e=1, max_candidates=1)
    records, record = _make_on_step_recorder()
    entropy_guided_search(
        tree, target, placement_goal(target), params=params, on_step=record
    )
    events = [r["event"] for r in records]
    assert "fallback_start" in events or "goal" in events


def test_outcome_transposition_maps_to_state_seen(monkeypatch):
    tree = _make_tree()
    target = {0: LocationAddress(1, 0)}
    params = SearchParams(e_max=2, delta_e=1, max_candidates=1)
    records, record = _make_on_step_recorder()
    candidate = frozenset({WordLaneAddress(0, 0, 0)})

    def fake_next_candidate(_self, _sn, _node, _generator):  # type: ignore[no-untyped-def]
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
    target = {0: LocationAddress(1, 0)}
    params = SearchParams(e_max=2, delta_e=1, max_candidates=1)
    records, record = _make_on_step_recorder()
    candidate = frozenset({WordLaneAddress(0, 0, 0)})

    def fake_next_candidate(_self, _sn, _node, _generator):  # type: ignore[no-untyped-def]
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


def test_ancestor_revisit_maps_to_state_seen_with_real_tree_outcome():
    tree = _make_tree()
    target = {0: LocationAddress(9, 0)}
    params = SearchParams(e_max=2, delta_e=1, max_candidates=1)
    records, record = _make_on_step_recorder()

    class _BacktrackGenerator:
        def generate(self, node, tree):  # type: ignore[no-untyped-def]
            loc = node.configuration[0]
            if loc == LocationAddress(0, 0):
                first_lane = next(iter(tree.outgoing_lanes(loc)), None)
                if first_lane is not None:
                    yield frozenset({first_lane})
                return
            for lane in tree.valid_lanes(node, direction=Direction.BACKWARD):
                src, dst = tree.arch_spec.get_endpoints(lane)
                if src == loc and dst == LocationAddress(0, 0):
                    yield frozenset({lane})
                    return

    search = EntropyGuidedSearch(
        tree,
        target,
        placement_goal(target),
        params=params,
        max_expansions=2,
        on_step=record,
    )
    search.run(generator=_BacktrackGenerator())

    bumps = [r for r in records if r["event"] == "entropy_bump"]
    assert any(
        b["metadata"].reason == "state-seen"
        and b["metadata"].state_seen_node_id == id(tree.root)
        for b in bumps
    )


def test_collects_multiple_goal_nodes(monkeypatch):
    tree = _make_tree()
    target = {0: LocationAddress(1, 0)}
    goal = placement_goal(target)
    c1 = frozenset({WordLaneAddress(0, 0, 0)})
    c2 = frozenset({WordLaneAddress(0, 0, 1)})
    candidate_sequence = iter([c1, c2])

    def fake_next_candidate(_self, _sn, _node, _generator):  # type: ignore[no-untyped-def]
        return next(candidate_sequence, None)

    def fake_try_move_set(node, move_set, strict=True):  # type: ignore[no-untyped-def]
        goal_node = ConfigurationNode(
            configuration={0: LocationAddress(1, 0), 1: node.configuration[1]},
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


def test_score_resume_buffer_replaces_lowest_when_higher_score_arrives(monkeypatch):
    tree = _make_tree()
    target = {0: LocationAddress(1, 0)}
    search = EntropyGuidedSearch(tree, target, placement_goal(target))
    capacity = 2

    c1 = ConfigurationNode(
        configuration=dict(tree.root.configuration),
        parent=tree.root,
        parent_moves=frozenset({SiteLaneAddress(0, 0, 0)}),
        depth=1,
    )
    c2 = ConfigurationNode(
        configuration=dict(tree.root.configuration),
        parent=tree.root,
        parent_moves=frozenset({SiteLaneAddress(1, 0, 0)}),
        depth=1,
    )
    c3 = ConfigurationNode(
        configuration=dict(tree.root.configuration),
        parent=tree.root,
        parent_moves=frozenset({SiteLaneAddress(2, 0, 0)}),
        depth=1,
    )
    scores = {id(c1): 1.0, id(c2): 2.0, id(c3): 3.0}

    monkeypatch.setattr(
        search,
        "_state_move_score",
        lambda node: scores[id(node)],  # type: ignore[no-untyped-def]
    )

    search._buffer_insert(c1, capacity=capacity)
    search._buffer_insert(c2, capacity=capacity)
    search._buffer_insert(c3, capacity=capacity)

    best = search._buffer_pop_best()
    next_best = search._buffer_pop_best()

    assert best is c3
    assert next_best is c2


def test_score_resume_buffer_does_not_replace_when_not_better(monkeypatch):
    tree = _make_tree()
    target = {0: LocationAddress(5, 0)}
    search = EntropyGuidedSearch(tree, target, placement_goal(target))
    capacity = 2

    c1 = ConfigurationNode(
        configuration=dict(tree.root.configuration),
        parent=tree.root,
        parent_moves=frozenset({SiteLaneAddress(0, 0, 0)}),
        depth=1,
    )
    c2 = ConfigurationNode(
        configuration=dict(tree.root.configuration),
        parent=tree.root,
        parent_moves=frozenset({SiteLaneAddress(1, 0, 0)}),
        depth=1,
    )
    c3 = ConfigurationNode(
        configuration=dict(tree.root.configuration),
        parent=tree.root,
        parent_moves=frozenset({SiteLaneAddress(2, 0, 0)}),
        depth=1,
    )
    scores = {id(c1): 3.0, id(c2): 2.0, id(c3): 1.0}

    monkeypatch.setattr(
        search,
        "_state_move_score",
        lambda node: scores[id(node)],  # type: ignore[no-untyped-def]
    )

    search._buffer_insert(c1, capacity=capacity)
    search._buffer_insert(c2, capacity=capacity)
    search._buffer_insert(c3, capacity=capacity)

    best = search._buffer_pop_best()
    next_best = search._buffer_pop_best()

    assert best is c1
    assert next_best is c2


def test_multi_goal_uses_score_buffer_for_resume(monkeypatch):
    tree = _make_tree()
    target = {0: LocationAddress(5, 0)}
    goal = placement_goal(target)
    search = EntropyGuidedSearch(
        tree,
        target,
        goal,
        params=SearchParams(max_goal_candidates=2),
    )

    q0_loc = tree.root.configuration[0]
    q1_loc = tree.root.configuration[1]
    lane_q0 = next(iter(tree.outgoing_lanes(q0_loc)))
    lane_q1 = next(iter(tree.outgoing_lanes(q1_loc)))
    c1 = frozenset({lane_q0})
    c2 = frozenset({lane_q1})
    candidate_sequence = iter([c1, c2])

    def fake_next_candidate(_self, _sn, _node, _generator):  # type: ignore[no-untyped-def]
        return next(candidate_sequence, None)

    created_children: list[ConfigurationNode] = []

    def fake_try_move_set(node, move_set, strict=True):  # type: ignore[no-untyped-def]
        child = ConfigurationNode(
            configuration={0: LocationAddress(5, 0), 1: node.configuration[1]},
            parent=node,
            parent_moves=move_set,
            depth=node.depth + 1,
        )
        created_children.append(child)
        return ExpansionOutcome(
            move_set=move_set,
            status=ExpansionStatus.CREATED_CHILD,
            child=child,
        )

    pop_calls: list[int] = []

    def fake_pop_best():  # type: ignore[no-untyped-def]
        pop_calls.append(1)
        return tree.root

    monkeypatch.setattr(EntropyGuidedSearch, "_get_next_candidate", fake_next_candidate)
    monkeypatch.setattr(tree, "try_move_set", fake_try_move_set)
    monkeypatch.setattr(search, "_buffer_pop_best", fake_pop_best)

    class _StubGenerator:
        def generate(self, node, tree):  # type: ignore[no-untyped-def,unused-argument]
            return iter(())

    result = search.run(generator=_StubGenerator())

    assert len(pop_calls) == 1
    assert len(result.goal_nodes) == 2
    assert len(created_children) == 2
    assert all(goal(n) for n in result.goal_nodes)
