from __future__ import annotations

import numpy as np
import stim
from beliefmatching import detector_error_model_to_check_matrices

from ..domain.layout import DEFAULT_SYNDROME_LAYOUT, SyndromeLayout
from .types import DetectorErrorModelTask


def _make_layout_only_dem(
    num_detectors: int, num_observables: int
) -> stim.DetectorErrorModel:
    """Create a minimal DEM carrying only detector and observable dimensions."""

    terms = []
    if num_detectors:
        terms.append(" ".join(f"D{i}" for i in range(num_detectors)))
    if num_observables:
        terms.append(" ".join(f"L{i}" for i in range(num_observables)))
    if not terms:
        raise ValueError("Need at least one detector or observable.")
    # NOTE: this DEM only carries detector/observable layout metadata.
    return stim.DetectorErrorModel("\n".join(f"error(0.5) {term}" for term in terms))


def _matrix_to_dem(
    check_matrix: np.ndarray,
    observables_matrix: np.ndarray,
    priors: np.ndarray,
) -> stim.DetectorErrorModel:
    """Convert binary detector/observable matrices into a Stim DEM."""

    lines = []
    for col, prior in enumerate(np.asarray(priors, dtype=np.float64)):
        det_targets = [f"D{i}" for i in np.flatnonzero(check_matrix[:, col])]
        obs_targets = [f"L{i}" for i in np.flatnonzero(observables_matrix[:, col])]
        if not det_targets and not obs_targets:
            continue
        safe_prior = float(np.clip(prior, 1e-12, 1.0 - 1e-12))
        lines.append(f"error({safe_prior:.16g}) " + " ".join(det_targets + obs_targets))
    if not lines:
        raise ValueError("Matrix reduction produced an empty DEM.")
    return stim.DetectorErrorModel("\n".join(lines))


def _compute_dem_data(task: DetectorErrorModelTask) -> dict[str, np.ndarray]:
    """Extract check matrices and priors from a task detector error model."""

    dem_matrix = detector_error_model_to_check_matrices(
        task.detector_error_model,
        allow_undecomposed_hyperedges=True,
    )
    return {
        "H": dem_matrix.check_matrix.toarray().astype(np.int64),
        "O": dem_matrix.observables_matrix.toarray().astype(np.int64),
        "priors": np.asarray(dem_matrix.priors, dtype=np.float64),
    }


def _select_output_observables(
    observables: np.ndarray,
    *,
    layout: SyndromeLayout = DEFAULT_SYNDROME_LAYOUT,
) -> np.ndarray:
    """Select logical-output observable columns according to a syndrome layout."""

    output_obs = np.asarray(observables, dtype=np.uint8)[
        :, : layout.output_observable_count
    ]
    if output_obs.shape[1] != layout.output_observable_count:
        raise ValueError(
            "Observable array does not contain the requested number of output "
            "observables."
        )
    return output_obs
