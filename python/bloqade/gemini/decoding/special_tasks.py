from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

import stim

from bloqade.gemini.device import GeminiLogicalSimulatorTask

_TaskT = TypeVar("_TaskT", bound=GeminiLogicalSimulatorTask[Any])

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
        "stim_circuit",
        "noiseless_stim_circuit",
    ):
        task.__dict__.pop(attr, None)


def _first_nonunitary_instruction_index(circuit: stim.Circuit) -> int:
    for idx in range(len(circuit)):
        if str(circuit[idx]).startswith(_NONUNITARY_PREFIXES):
            return idx
    return len(circuit)


def _apply_noiseless_unitary_prefix(
    circuit_map: Mapping[str, stim.Circuit],
) -> dict[str, stim.Circuit]:
    """Prepend the noiseless inverse unitary prefix to each Stim circuit."""
    transformed: dict[str, stim.Circuit] = {}
    for basis_label, circuit in circuit_map.items():
        compiled_prefix = circuit[: _first_nonunitary_instruction_index(circuit)]
        inverse_prefix = compiled_prefix.without_noise().inverse()
        transformed[basis_label] = inverse_prefix + circuit
    return transformed


# NOTE: this function is currently unused in the source code and can be deleted
def _apply_special_tsim_circuit_strategy(
    task_map: Mapping[str, _TaskT],
) -> dict[str, _TaskT]:
    """Prepend the inverse compiled unitary prefix to each task's circuits."""

    transformed = dict(task_map)
    special_circuits = _apply_noiseless_unitary_prefix(
        {basis: task.stim_circuit for basis, task in transformed.items()}
    )
    special_noiseless_circuits = _apply_noiseless_unitary_prefix(
        {basis: task.noiseless_stim_circuit for basis, task in transformed.items()}
    )
    for basis, task in transformed.items():
        _clear_task_tsim_artifacts(task)
        vars(task)["stim_circuit"] = special_circuits[basis]
        vars(task)["noiseless_stim_circuit"] = special_noiseless_circuits[basis]
    return transformed
