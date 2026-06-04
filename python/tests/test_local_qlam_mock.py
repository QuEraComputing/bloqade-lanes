import importlib.util
import sys
from pathlib import Path


def load_local_qlam_mock():
    path = Path(__file__).parents[2] / "demo" / "gemini_mvp" / "local_qlam_mock.py"
    spec = importlib.util.spec_from_file_location("local_qlam_mock", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def squin_gate(class_name: str) -> dict:
    return {
        "module_name": "bloqade.squin.gate.stmts",
        "class_name": class_name,
    }


def test_local_qlam_mock_infers_double_x_as_zeros():
    local_qlam_mock = load_local_qlam_mock()

    program_content = {
        "body": [
            squin_gate("X"),
            squin_gate("X"),
        ]
    }

    assert (
        local_qlam_mock.infer_mock_measurement_mode(program_content)
        == local_qlam_mock.MOCK_MODE_ZEROS
    )


def test_local_qlam_mock_returns_zero_bitstring_for_double_x_program():
    local_qlam_mock = load_local_qlam_mock()

    task = local_qlam_mock.MockTask(
        qpu_mode="logical",
        definition={
            "programs": [
                {
                    "content": {
                        "body": [
                            squin_gate("X"),
                            squin_gate("X"),
                        ]
                    }
                }
            ],
            "subtasks": [{"program_index": 0, "num_shots": 1}],
        },
        bitstring_width=5,
        execution_latency_seconds=0.0,
    )

    assert task._bitstring(subtask_index=0, shot_index=0) == [False] * 5
