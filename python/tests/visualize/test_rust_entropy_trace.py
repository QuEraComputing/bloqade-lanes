from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_adapter():
    repo_root = Path(__file__).resolve().parents[3]
    adapter_path = (
        repo_root
        / "python"
        / "bloqade"
        / "lanes"
        / "visualize"
        / "rust_entropy_trace.py"
    )
    spec = importlib.util.spec_from_file_location("_rust_entropy_trace", adapter_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_rust_entropy_trace_parses_payload() -> None:
    module = _load_adapter()
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
