from __future__ import annotations

from bloqade.lanes.steane_defaults import steane7_m2dets, steane7_m2obs

from .types import MeasurementMap


def build_measurement_maps(
    num_logical_qubits: int,
) -> tuple[MeasurementMap, MeasurementMap]:
    return steane7_m2dets(num_logical_qubits), steane7_m2obs(num_logical_qubits)
