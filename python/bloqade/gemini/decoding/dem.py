"""Detector error model conversion and projection helpers."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt
import stim
from beliefmatching import detector_error_model_to_check_matrices


class DetectorErrorModelTask(Protocol):
    """Protocol for objects exposing a Stim detector error model."""

    @property
    def detector_error_model(self) -> stim.DetectorErrorModel: ...


def make_layout_only_dem(
    num_detectors: int,
    num_observables: int,
) -> stim.DetectorErrorModel:
    """Create a minimal DEM carrying detector and observable dimensions."""

    terms: list[str] = []
    if num_detectors:
        terms.append(" ".join(f"D{i}" for i in range(int(num_detectors))))
    if num_observables:
        terms.append(" ".join(f"L{i}" for i in range(int(num_observables))))
    if not terms:
        raise ValueError("Need at least one detector or observable.")
    return stim.DetectorErrorModel("\n".join(f"error(0.5) {term}" for term in terms))


def matrix_to_dem(
    check_matrix: np.ndarray,
    observables_matrix: np.ndarray,
    priors: np.ndarray,
) -> stim.DetectorErrorModel:
    """Convert binary detector/observable matrices into a Stim DEM."""

    check = np.asarray(check_matrix, dtype=np.uint8)
    observables = np.asarray(observables_matrix, dtype=np.uint8)
    prior_arr = np.asarray(priors, dtype=np.float64)
    if check.ndim != 2 or observables.ndim != 2:
        raise ValueError("check_matrix and observables_matrix must be 2D.")
    if check.shape[1] != observables.shape[1] or check.shape[1] != len(prior_arr):
        raise ValueError("Matrices and priors must describe the same errors.")

    lines: list[str] = []
    for col, prior in enumerate(prior_arr):
        det_targets = [f"D{i}" for i in np.flatnonzero(check[:, col])]
        obs_targets = [f"L{i}" for i in np.flatnonzero(observables[:, col])]
        if not det_targets and not obs_targets:
            continue
        safe_prior = float(np.clip(prior, 1e-12, 1.0 - 1e-12))
        lines.append(f"error({safe_prior:.16g}) " + " ".join(det_targets + obs_targets))
    if not lines:
        raise ValueError("Matrix reduction produced an empty DEM.")
    return stim.DetectorErrorModel("\n".join(lines))


def detector_error_model_matrices(
    task_or_dem: DetectorErrorModelTask | stim.DetectorErrorModel,
) -> dict[str, npt.NDArray[np.float64] | npt.NDArray[np.int64]]:
    """Extract check matrices, observable matrices, and priors from a DEM."""

    dem = (
        task_or_dem
        if isinstance(task_or_dem, stim.DetectorErrorModel)
        else task_or_dem.detector_error_model
    )
    dem_matrix = detector_error_model_to_check_matrices(
        dem,
        allow_undecomposed_hyperedges=True,
    )
    return {
        "H": dem_matrix.check_matrix.toarray().astype(np.int64),
        "O": dem_matrix.observables_matrix.toarray().astype(np.int64),
        "priors": np.asarray(dem_matrix.priors, dtype=np.float64),
    }


_TargetKey = tuple[str, int]


def _compose_independent_flip_probabilities(p_old: float, p_new: float) -> float:
    return p_old * (1.0 - p_new) + p_new * (1.0 - p_old)


def _selected_index(sorted_indices: Sequence[int], value: int) -> int | None:
    index = bisect_left(sorted_indices, value)
    if index == len(sorted_indices) or sorted_indices[index] != value:
        return None
    return index


def _target_key(
    target: stim.DemTarget,
    *,
    detector_indices: Sequence[int],
    observable_indices: Sequence[int],
) -> _TargetKey | None:
    if target.is_relative_detector_id():
        index = _selected_index(detector_indices, target.val)
        if index is None:
            return None
        return ("D", index)
    if target.is_logical_observable_id():
        index = _selected_index(observable_indices, target.val)
        if index is None:
            return None
        return ("L", index)
    return None


def _dem_target(key: _TargetKey) -> stim.DemTarget:
    kind, index = key
    if kind == "D":
        return stim.target_relative_detector_id(index)
    return stim.target_logical_observable_id(index)


def sub_detector_error_model(
    dem: stim.DetectorErrorModel,
    detector_indices: Sequence[int],
    observable_indices: Sequence[int],
) -> stim.DetectorErrorModel:
    """Project a DEM onto selected detectors and logical observables.

    Duplicate projected error mechanisms are composed using XOR-flip
    probability semantics. This is less lossy than converting through a binary
    matrix and reconstructing a DEM.
    """

    sorted_detectors = sorted(int(index) for index in detector_indices)
    sorted_observables = sorted(int(index) for index in observable_indices)
    error_probabilities: dict[tuple[_TargetKey, ...], float] = {}

    for instruction in dem.flattened():
        if not isinstance(instruction, stim.DemInstruction):
            continue
        if instruction.type != "error":
            continue

        projected_keys: set[_TargetKey] = set()
        for target in cast(Sequence[stim.DemTarget], instruction.targets_copy()):
            if target.is_separator():
                continue
            key = _target_key(
                target,
                detector_indices=sorted_detectors,
                observable_indices=sorted_observables,
            )
            if key is None:
                continue
            if key in projected_keys:
                projected_keys.remove(key)
            else:
                projected_keys.add(key)

        if not projected_keys:
            continue

        projected_key = tuple(
            sorted(projected_keys, key=lambda key: (key[0] != "D", key[1]))
        )
        probability = float(cast(Sequence[float], instruction.args_copy())[0])
        if projected_key in error_probabilities:
            error_probabilities[projected_key] = (
                _compose_independent_flip_probabilities(
                    error_probabilities[projected_key],
                    probability,
                )
            )
        else:
            error_probabilities[projected_key] = probability

    projected_dem = stim.DetectorErrorModel()
    for projected_key, probability in error_probabilities.items():
        if probability == 0.0:
            continue
        projected_dem.append(
            "error",
            parens_arguments=[probability],
            targets=[_dem_target(key) for key in projected_key],
        )

    if sorted_detectors:
        projected_dem.append(
            "detector",
            parens_arguments=[],
            targets=[stim.target_relative_detector_id(len(sorted_detectors) - 1)],
        )
    if sorted_observables:
        projected_dem.append(
            "logical_observable",
            parens_arguments=[],
            targets=[stim.target_logical_observable_id(len(sorted_observables) - 1)],
        )

    return projected_dem


_make_layout_only_dem = make_layout_only_dem
_matrix_to_dem = matrix_to_dem
_compute_dem_data = detector_error_model_matrices


__all__ = [
    "DetectorErrorModelTask",
    "_compute_dem_data",
    "_make_layout_only_dem",
    "_matrix_to_dem",
    "detector_error_model_matrices",
    "detector_error_model_to_check_matrices",
    "make_layout_only_dem",
    "matrix_to_dem",
    "sub_detector_error_model",
]
