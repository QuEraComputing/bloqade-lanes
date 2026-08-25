from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any, TypeVar

import numpy as np
from kirin import ir

from bloqade.lanes.analysis import atom
from bloqade.lanes.arch.gemini import physical
from bloqade.lanes.bytecode.encoding import ZoneAddress
from bloqade.lanes.transform import LogicalPipeline

RetType = TypeVar("RetType")


def _shot_identity(shot) -> tuple[str, int, int, int]:
    """Return the storage identity shared by a shot's frame records."""
    return (
        shot.task_id,
        shot.subtask_index,
        shot.subtask_shot_index,
        shot.shot_index,
    )


def _shots_for_subtask_frame(storage, shot_filter, subtask_index, frame_type):
    subtask_filter = replace(
        shot_filter,
        subtask_indices=(subtask_index,),
        frame_type=frame_type,
    )
    return sorted(
        storage.get_shots(shot_filter=subtask_filter),
        key=_shot_identity,
    )


def shot_results_for_subtasks(
    storage,
    shot_filter,
    subtasks: Sequence[dict],
    *,
    frame_type: str | None = None,
) -> list[np.ndarray]:
    """Return selected shot bitstrings grouped and sorted by subtask.

    ``StorageBackend.get_shots`` does not promise an ordering.  This helper
    preserves the caller-provided subtask order and sorts each subtask by its
    full storage identity. ``frame_type`` optionally overrides the frame type
    already present in ``shot_filter``.
    """
    results = []
    for subtask in subtasks:
        selected_frame = shot_filter.frame_type if frame_type is None else frame_type
        rows = _shots_for_subtask_frame(
            storage,
            shot_filter,
            subtask["subtask_index"],
            selected_frame,
        )
        results.append(np.asarray([row.bitstring for row in rows], dtype=bool))

    return results


def aligned_detected_and_sorted_shots_for_subtasks(
    storage,
    shot_filter,
    subtasks: Sequence[dict],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return DETECTED and SORTED frames aligned shot-for-shot per subtask.

    Each frame is ordered by the complete storage identity. A mismatch means
    that the two frames cannot safely be compared by array position.
    """
    detected_results = []
    sorted_results = []

    for subtask in subtasks:
        subtask_index = subtask["subtask_index"]
        detected_rows = _shots_for_subtask_frame(
            storage, shot_filter, subtask_index, "DETECTED"
        )
        sorted_rows = _shots_for_subtask_frame(
            storage, shot_filter, subtask_index, "SORTED"
        )

        detected_keys = tuple(map(_shot_identity, detected_rows))
        sorted_keys = tuple(map(_shot_identity, sorted_rows))
        if detected_keys != sorted_keys:
            detected_only = sorted(set(detected_keys) - set(sorted_keys))
            sorted_only = sorted(set(sorted_keys) - set(detected_keys))
            raise ValueError(
                "DETECTED and SORTED frames are not aligned for "
                f"subtask_index={subtask_index}: "
                f"missing SORTED={detected_only}, missing DETECTED={sorted_only}."
            )

        detected_results.append(
            np.asarray([row.bitstring for row in detected_rows], dtype=bool)
        )
        sorted_results.append(
            np.asarray([row.bitstring for row in sorted_rows], dtype=bool)
        )

    return detected_results, sorted_results


def get_slm_mapping_postprocessing(
    sim_kernel: ir.Method[..., RetType], *, invert_bits=False
) -> tuple[Callable[[np.ndarray], Any], atom.PostProcessing[RetType]]:
    """Create a logical_results-compatible postprocessor for Zone-0 shots."""

    arch_spec = physical.get_arch_spec()
    physical_move_kernel = LogicalPipeline(transversal_rewrite=True).emit(sim_kernel)
    zone0 = ZoneAddress(0)

    interpreter = atom.AtomInterpreter(
        physical_move_kernel.dialects,
        arch_spec=arch_spec,
    )
    frame, _ = interpreter.run(physical_move_kernel)

    # Each MeasureResult identifies both:
    # - its compact measurement-record column expected by post-processing
    # - the physical SLM location at which it was measured.
    records_by_id = {}

    for value in frame.entries.values():
        if not isinstance(value, atom.MeasureResult):
            continue

        previous = records_by_id.get(value.measurement_id)
        if previous is not None and (
            previous.location_address != value.location_address
            or previous.qubit_id != value.qubit_id
        ):
            raise ValueError(
                f"Measurement record {value.measurement_id} maps inconsistently: "
                f"{previous} versus {value}."
            )

        records_by_id[value.measurement_id] = value

    if not records_by_id:
        raise ValueError("No physical measurement records were found.")

    ids = sorted(records_by_id)
    if ids != list(range(len(ids))):
        raise ValueError(f"Unexpected measurement-record IDs: {ids}")

    records = [records_by_id[measurement_id] for measurement_id in ids]

    if not records:
        raise ValueError("No physical measurement records were found.")

    ids = [record.measurement_id for record in records]
    if ids != list(range(len(records))):
        raise ValueError(f"Unexpected measurement-record IDs: {ids}")

    # `get_zone_index()` is word-major; QLAM-style SLM frames are conventionally
    # arranged bottom-to-top in rows, left-to-right within each row. Translate
    # through physical coordinates instead of treating a Zone-0 index as an SLM
    # bitstring column.
    zone0_locations = list(arch_spec.yield_zone_locations(zone0))
    positions_by_location = {
        location: arch_spec.get_position(location) for location in zone0_locations
    }
    x_coordinates = sorted({position[0] for position in positions_by_location.values()})
    y_coordinates = sorted({position[1] for position in positions_by_location.values()})

    if len(x_coordinates) * len(y_coordinates) != len(zone0_locations):
        raise ValueError(
            "Zone 0 is not a rectangular SLM grid; cannot derive a row-major "
            "SLM bitstring mapping."
        )

    x_index = {coordinate: index for index, coordinate in enumerate(x_coordinates)}
    y_index = {coordinate: index for index, coordinate in enumerate(y_coordinates)}
    raw_slm_index_by_location = {
        location: y_index[y] * len(x_coordinates) + x_index[x]
        for location, (x, y) in positions_by_location.items()
    }

    mapping = np.asarray(
        [raw_slm_index_by_location[record.location_address] for record in records],
        dtype=int,
    )
    # print(f"row-major SLM mapping: {mapping}")
    expected_zone0_sites = len(zone0_locations)

    post_processing = interpreter.get_post_processing(physical_move_kernel)

    def postprocess(zone0_shots):
        zone0_shots = np.asarray(zone0_shots, dtype=bool)

        if zone0_shots.ndim != 2:
            raise ValueError(f"Expected a 2-D array, got {zone0_shots.shape}.")
        if zone0_shots.shape[1] != expected_zone0_sites:
            raise ValueError(
                f"Expected {expected_zone0_sites} Zone-0 columns, "
                f"got {zone0_shots.shape[1]}."
            )

        measurement_shots = zone0_shots[:, mapping]

        # Enable only if the stored QLAM bit convention is opposite to the
        # simulator/post-processing convention.
        if invert_bits:
            measurement_shots = ~measurement_shots

        return measurement_shots.tolist()

    return postprocess, post_processing
