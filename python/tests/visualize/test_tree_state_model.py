from __future__ import annotations

import importlib
import sys
from pathlib import Path

from bloqade.lanes.layout import LocationAddress


def _load_debug_modules():
    repo_root = Path(__file__).resolve().parents[3]
    viz_dir = repo_root / "debug" / "viz"
    viz_dir_str = str(viz_dir)
    if viz_dir_str not in sys.path:
        sys.path.insert(0, viz_dir_str)
    trace_tree = importlib.import_module("trace_tree")
    tree_state_model = importlib.import_module("tree_state_model")
    return trace_tree, tree_state_model


def test_state_seen_goals_render_as_distinct_nodes() -> None:
    trace_tree, tree_state_model = _load_debug_modules()
    TreeTraceStep = trace_tree.TreeTraceStep
    TreeStateReducer = tree_state_model.TreeStateReducer

    root_cfg = {0: LocationAddress(0, 0, 0)}
    branch_a_cfg = {0: LocationAddress(0, 1, 0)}
    branch_b_cfg = {0: LocationAddress(0, 2, 0)}
    goal_cfg = {0: LocationAddress(0, 3, 0)}

    steps = [
        TreeTraceStep(
            step_index=0,
            event="descend",
            node_id=10,
            parent_node_id=0,
            depth=1,
            entropy=1,
            unresolved_count=1,
            moveset=None,
            candidate_movesets=tuple(),
            candidate_index=None,
            reason=None,
            state_seen_node_id=None,
            no_valid_moves_qubit=None,
            trigger_node_id=None,
            configuration=branch_a_cfg,
            parent_configuration=root_cfg,
            moveset_score=5.0,
        ),
        TreeTraceStep(
            step_index=1,
            event="goal",
            node_id=99,
            parent_node_id=50,
            depth=2,
            entropy=1,
            unresolved_count=0,
            moveset=None,
            candidate_movesets=tuple(),
            candidate_index=None,
            reason="state-seen-goal",
            state_seen_node_id=99,
            no_valid_moves_qubit=None,
            trigger_node_id=10,
            configuration=goal_cfg,
            parent_configuration=branch_a_cfg,
            moveset_score=None,
        ),
        TreeTraceStep(
            step_index=2,
            event="descend",
            node_id=11,
            parent_node_id=0,
            depth=1,
            entropy=1,
            unresolved_count=1,
            moveset=None,
            candidate_movesets=tuple(),
            candidate_index=None,
            reason=None,
            state_seen_node_id=None,
            no_valid_moves_qubit=None,
            trigger_node_id=None,
            configuration=branch_b_cfg,
            parent_configuration=root_cfg,
            moveset_score=4.0,
        ),
        TreeTraceStep(
            step_index=3,
            event="goal",
            node_id=99,
            parent_node_id=50,
            depth=1,
            entropy=1,
            unresolved_count=0,
            moveset=None,
            candidate_movesets=tuple(),
            candidate_index=None,
            reason="state-seen-goal",
            state_seen_node_id=99,
            no_valid_moves_qubit=None,
            trigger_node_id=11,
            configuration=goal_cfg,
            parent_configuration=branch_b_cfg,
            moveset_score=None,
        ),
    ]

    reducer = TreeStateReducer(steps, root_node_id=0, best_buffer_size=2)
    first_goal_frame = reducer.frame_at(2)
    second_goal_frame = reducer.frame_at(4)

    assert first_goal_frame.current_node_id < 0
    assert second_goal_frame.current_node_id < 0
    assert second_goal_frame.current_node_id != first_goal_frame.current_node_id

    first_goal_node = first_goal_frame.nodes[first_goal_frame.current_node_id]
    second_goal_node = second_goal_frame.nodes[second_goal_frame.current_node_id]
    assert first_goal_node.parent_id == 10
    assert second_goal_node.parent_id == 11
    assert first_goal_node.is_goal
    assert second_goal_node.is_goal

    synthetic_goals = [
        node
        for node in second_goal_frame.nodes.values()
        if node.node_id < 0 and node.is_goal
    ]
    assert len(synthetic_goals) == 2
    assert second_goal_frame.best_goal_depth == 1


def test_buffer_panel_uses_exact_traced_buffer_snapshot() -> None:
    trace_tree, tree_state_model = _load_debug_modules()
    TreeTraceStep = trace_tree.TreeTraceStep
    TreeStateReducer = tree_state_model.TreeStateReducer

    root_cfg = {0: LocationAddress(0, 0, 0)}
    cfg_a = {0: LocationAddress(0, 1, 0)}
    cfg_b = {0: LocationAddress(0, 2, 0)}

    steps = [
        TreeTraceStep(
            step_index=0,
            event="descend",
            node_id=10,
            parent_node_id=0,
            depth=1,
            entropy=1,
            unresolved_count=1,
            moveset=None,
            candidate_movesets=tuple(),
            candidate_index=None,
            reason=None,
            state_seen_node_id=None,
            no_valid_moves_qubit=None,
            trigger_node_id=None,
            configuration=cfg_a,
            parent_configuration=root_cfg,
            moveset_score=8.0,
            best_buffer_node_ids=(10,),
        ),
        TreeTraceStep(
            step_index=1,
            event="descend",
            node_id=11,
            parent_node_id=0,
            depth=1,
            entropy=1,
            unresolved_count=1,
            moveset=None,
            candidate_movesets=tuple(),
            candidate_index=None,
            reason=None,
            state_seen_node_id=None,
            no_valid_moves_qubit=None,
            trigger_node_id=None,
            configuration=cfg_b,
            parent_configuration=root_cfg,
            moveset_score=3.0,
            best_buffer_node_ids=(10, 11),
        ),
        TreeTraceStep(
            step_index=2,
            event="goal",
            node_id=10,
            parent_node_id=0,
            depth=1,
            entropy=1,
            unresolved_count=0,
            moveset=None,
            candidate_movesets=tuple(),
            candidate_index=None,
            reason=None,
            state_seen_node_id=None,
            no_valid_moves_qubit=None,
            trigger_node_id=None,
            configuration=cfg_a,
            parent_configuration=root_cfg,
            moveset_score=None,
            best_buffer_node_ids=(11,),
        ),
    ]

    reducer = TreeStateReducer(steps, root_node_id=0, best_buffer_size=2)
    frame = reducer.frame_at(3)
    assert frame.best_buffer_node_display_ids == (2, None)
