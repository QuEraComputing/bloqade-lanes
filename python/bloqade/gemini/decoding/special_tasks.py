from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from .tasks import DemoTask
from .types import KirinKernel, MeasurementMap, TsimCircuit

if TYPE_CHECKING:
    from bloqade.gemini.device import GeminiLogicalSimulatorTask
    from bloqade.lanes import GeminiLogicalSimulator

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


def _set_task_override(
    task: GeminiLogicalSimulatorTask,
    attr: str,
    value: object,
) -> None:
    """Set or shadow a cached Gemini task attribute."""

    try:
        setattr(task, attr, value)
    except AttributeError:
        task.__dict__[attr] = value


def _clear_task_tsim_artifacts(task: GeminiLogicalSimulatorTask) -> None:
    """Clear cached tsim/sampler artifacts after overriding a circuit."""

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
    """Return the first instruction index that cannot be inverted as unitary."""

    for idx in range(len(circuit)):
        if str(circuit[idx]).startswith(_NONUNITARY_PREFIXES):
            return idx
    return len(circuit)


def _override_task_tsim_circuit(
    task: GeminiLogicalSimulatorTask,
    circuit: TsimCircuit,
    *,
    noiseless_circuit: TsimCircuit | None = None,
) -> None:
    """Replace a task's tsim circuits and clear dependent caches."""

    _clear_task_tsim_artifacts(task)
    _set_task_override(task, "tsim_circuit", circuit)
    if noiseless_circuit is not None:
        _set_task_override(task, "noiseless_tsim_circuit", noiseless_circuit)


def _build_compiled_unitary_prefix_circuit(
    task: GeminiLogicalSimulatorTask,
) -> TsimCircuit:
    """Return the initial unitary portion of a compiled task circuit."""

    compiled_circuit = task.tsim_circuit
    return compiled_circuit[: _first_nonunitary_instruction_index(compiled_circuit)]


def _prepend_inverse_tsim_circuit(
    task: GeminiLogicalSimulatorTask,
    circuit_to_invert: TsimCircuit,
) -> None:
    """Prepend the inverse of a unitary prefix circuit to a task."""

    inverse_prefix = circuit_to_invert.without_noise().inverse()
    _override_task_tsim_circuit(
        task,
        inverse_prefix + task.tsim_circuit,
        noiseless_circuit=inverse_prefix + task.noiseless_tsim_circuit,
    )


def _build_task(
    simulator: GeminiLogicalSimulator,
    kernel: KirinKernel,
    *,
    m2dets: MeasurementMap | None,
    m2obs: MeasurementMap | None,
    append_measurements: bool = True,
) -> DemoTask:
    """Build a demo task from a logical kernel and measurement maps."""

    task = simulator.task(
        kernel.similar(),
        m2dets if append_measurements else None,
        m2obs if append_measurements else None,
    )
    return DemoTask(task=task)


def apply_special_tsim_circuit_strategy(
    task_map: Mapping[str, DemoTask],
) -> dict[str, DemoTask]:
    """Prepend the inverse compiled unitary prefix to every task."""

    transformed = dict(task_map)
    for demo_task in transformed.values():
        _prepend_inverse_tsim_circuit(
            demo_task.task,
            _build_compiled_unitary_prefix_circuit(demo_task.task),
        )
    return transformed


def build_task_map(
    simulator: GeminiLogicalSimulator,
    kernel_map: Mapping[str, KirinKernel],
    *,
    m2dets: MeasurementMap | None,
    m2obs: MeasurementMap | None,
    append_measurements: bool = True,
) -> dict[str, DemoTask]:
    """Build demo tasks for each basis kernel in a kernel map."""

    return {
        basis: _build_task(
            simulator,
            kernel,
            m2dets=m2dets,
            m2obs=m2obs,
            append_measurements=append_measurements,
        )
        for basis, kernel in kernel_map.items()
    }
