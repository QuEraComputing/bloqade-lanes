from __future__ import annotations

import json

from bloqade.lanes.visualize import rust_entropy_trace as module


def test_load_rust_entropy_trace_parses_payload() -> None:
    payload = json.dumps(
        {
            "version": 1,
            "root_node_id": 0,
            "best_buffer_size": 2,
            "steps": [
                {
                    "event": "descend",
                    "node_id": 1,
                    "parent_node_id": 0,
                    "depth": 1,
                    "entropy": 1,
                    "unresolved_count": 1,
                    "moveset": [[0, 0, 0, 0, 0, 0]],
                    "candidate_movesets": [[[0, 0, 0, 0, 0, 0]]],
                    "candidate_index": 0,
                    "reason": None,
                    "state_seen_node_id": None,
                    "no_valid_moves_qubit": None,
                    "trigger_node_id": None,
                    "configuration": [[0, 0, 0, 1]],
                    "parent_configuration": [[0, 0, 0, 0]],
                    "moveset_score": 12.5,
                    "best_buffer_node_ids": [1, 4],
                }
            ],
        }
    )

    trace = module.load_rust_entropy_trace(payload)
    assert trace.root_node_id == 0
    assert trace.best_buffer_size == 2
    assert len(trace.steps) == 1
    step = trace.steps[0]
    assert step.event == "descend"
    assert step.node_id == 1
    assert step.depth == 1
    assert step.candidate_index == 0
    assert step.moveset == [(0, 0, 0, 0, 0, 0)]
    assert step.best_buffer_node_ids == [1, 4]
