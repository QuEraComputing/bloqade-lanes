from __future__ import annotations

from bloqade.lanes.steane_defaults import steane7_m2dets, steane7_m2obs

from .types import MeasurementMap


def build_measurement_maps(
    num_logical_qubits: int,
) -> tuple[MeasurementMap, MeasurementMap]:
    """Return Steane-7 measurement maps for a logical-qubit register.

    Args:
        num_logical_qubits: Number of logical Steane blocks in the task.

    Returns:
        A ``(m2dets, m2obs)`` pair suitable for Gemini logical task
        construction.
    """

    return steane7_m2dets(num_logical_qubits), steane7_m2obs(num_logical_qubits)
