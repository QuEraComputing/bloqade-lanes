from __future__ import annotations

from bloqade.lanes.visualize.entropy_tree.state import TreeStateReducer
from bloqade.lanes.visualize.entropy_tree.tracer import TreeTraceStep


def _step(
    idx: int,
    event: str,
    node_id: int,
    parent: int | None,
    depth: int,
    entropy: int = 0,
) -> TreeTraceStep:
    return TreeTraceStep(
        step_index=idx,
        event=event,
        node_id=node_id,
        parent_node_id=parent,
        depth=depth,
        entropy=entropy,
        unresolved_count=0,
        moveset=None,
        candidate_movesets=(),
        candidate_index=None,
        reason=None,
        state_seen_node_id=None,
        no_valid_moves_qubit=None,
        trigger_node_id=None,
        configuration={},
        parent_configuration=None,
        moveset_score=None,
        best_buffer_node_ids=None,
    )


def test_empty_trace_has_single_root_frame():
    reducer = TreeStateReducer(steps=(), root_node_id=0, best_buffer_size=0)
    assert reducer.frame_count == 1
    frame = reducer.frame_at(0)
    assert frame.event == "initial"
    assert frame.current_node_id == 0


def test_descend_steps_grow_frame_count():
    steps = (
        _step(0, "descend", node_id=1, parent=0, depth=1),
        _step(1, "descend", node_id=2, parent=1, depth=2),
    )
    reducer = TreeStateReducer(steps=steps, root_node_id=0, best_buffer_size=0)
    # frame_count = len(actions) + 1; with 2 real descends and no synthetic seed
    # (candidate_movesets is empty), actions = steps so frame_count = 3.
    assert reducer.frame_count == 3
    first = reducer.frame_at(0)
    last = reducer.frame_at(reducer.frame_count - 1)
    assert first.event == "initial"
    assert last.event == "descend"
    assert last.current_node_id == 2


def test_frame_at_zero_returns_initial_event():
    steps = (_step(0, "descend", node_id=1, parent=0, depth=1),)
    reducer = TreeStateReducer(steps=steps, root_node_id=0, best_buffer_size=0)
    frame = reducer.frame_at(0)
    assert frame.event == "initial"
    assert frame.current_node_id == 0
