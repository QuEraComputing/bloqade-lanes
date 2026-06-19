from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .tasks import DemoTask
from .types import TsimCircuit

_NONUNITARY_PREFIXES = (
    "M",
    "MX",
    "MY",
    "MR",
    "MRX",
    "MRY",
    "MPP",
    "DETECTOR",
    "OBSERVABLE_INCLUDE",
)


def _clear_task_tsim_artifacts(task: object) -> None:
    for attr in (
        "tsim_circuit",
        "noiseless_tsim_circuit",
        "measurement_sampler",
        "noiseless_measurement_sampler",
        "detector_sampler",
        "noiseless_detector_sampler",
        "detector_error_model",
    ):
        task.__dict__.pop(attr, None)


def _first_nonunitary_instruction_index(circuit: TsimCircuit) -> int:
    for idx in range(len(circuit)):
        if str(circuit[idx]).startswith(_NONUNITARY_PREFIXES):
            return idx
    return len(circuit)


def _task_impl(task: object) -> Any:
    return task.task if isinstance(task, DemoTask) else task


def _apply_special_tsim_circuit_strategy(
    task_map: Mapping[str, object],
) -> dict[str, object]:
    """Prepend the inverse compiled unitary prefix to each task's circuits."""

    transformed = dict(task_map)
    for wrapped_task in transformed.values():
        task = _task_impl(wrapped_task)
        compiled_prefix = task.tsim_circuit[
            : _first_nonunitary_instruction_index(task.tsim_circuit)
        ]
        inverse_prefix = compiled_prefix.without_noise().inverse()
        _clear_task_tsim_artifacts(task)
        task.__dict__["tsim_circuit"] = inverse_prefix + task.tsim_circuit
        task.__dict__["noiseless_tsim_circuit"] = (
            inverse_prefix + task.noiseless_tsim_circuit
        )
    return transformed


__all__ = ["_apply_special_tsim_circuit_strategy"]
