from __future__ import annotations

from collections.abc import Mapping

from .tasks import DemoTask
from .types import KirinKernel, MeasurementMap, TsimCircuit

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


def apply_special_tsim_circuit_strategy(
    task_map: Mapping[str, DemoTask],
) -> dict[str, DemoTask]:
    """Prepend the inverse compiled unitary prefix to each task's circuits."""

    transformed = dict(task_map)
    for demo_task in transformed.values():
        task = demo_task.task
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


def build_task_map(
    simulator,
    kernel_map: Mapping[str, KirinKernel],
    *,
    m2dets: MeasurementMap | None,
    m2obs: MeasurementMap | None,
    append_measurements: bool = True,
) -> dict[str, DemoTask]:
    """Build demo tasks for each basis kernel in a kernel map."""

    return {
        basis: DemoTask(
            task=simulator.task(
                kernel.similar(),
                m2dets if append_measurements else None,
                m2obs if append_measurements else None,
            )
        )
        for basis, kernel in kernel_map.items()
    }


__all__ = ["apply_special_tsim_circuit_strategy", "build_task_map"]
