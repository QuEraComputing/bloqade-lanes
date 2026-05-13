from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyndromeLayout:
    """Column layout describing output versus factory syndrome bits.

    Attributes:
        output_detector_count: Number of leading detector columns that belong
            to the output logical qubit.
        output_observable_count: Number of leading observable columns that
            belong to the output logical qubit.
    """

    output_detector_count: int = 3
    output_observable_count: int = 1


DEFAULT_SYNDROME_LAYOUT = SyndromeLayout()


def split_factory_bits(
    detectors: np.ndarray,
    observables: np.ndarray,
    *,
    layout: SyndromeLayout = DEFAULT_SYNDROME_LAYOUT,
) -> tuple[np.ndarray, np.ndarray]:
    """Return factory detector and observable bits after output columns.

    Args:
        detectors: Detector sample matrix.
        observables: Observable sample matrix.
        layout: Syndrome layout specifying how many leading columns are output
            bits.

    Returns:
        A pair ``(factory_detectors, factory_observables)``.
    """

    return (
        detectors[:, layout.output_detector_count :],
        observables[:, layout.output_observable_count :],
    )


# This is used for us to help us, via simulation, get the noiseless expected
# observable from the circuit.
def _normalize_valid_factory_targets(
    valid_factory_targets: np.ndarray | Sequence[Sequence[int]] | Sequence[int],
) -> np.ndarray:
    """Normalize one or more valid factory targets into a unique 2D array."""

    targets = np.asarray(valid_factory_targets, dtype=np.uint8)
    if targets.ndim == 1:
        targets = targets.reshape(1, -1)
    if targets.ndim != 2:
        raise ValueError(
            "valid_factory_targets must be a 1D factory syndrome or a 2D array "
            "of valid factory syndromes."
        )
    if targets.shape[0] == 0 or targets.shape[1] == 0:
        raise ValueError("Need at least one non-empty valid factory syndrome.")
    return np.unique(targets, axis=0)


def _ancilla_matches_valid_targets(
    ancilla_observables: np.ndarray,
    valid_factory_targets: np.ndarray | Sequence[Sequence[int]] | Sequence[int],
) -> np.ndarray | bool:
    """Test whether ancilla observable rows match any valid target pattern."""

    targets = _normalize_valid_factory_targets(valid_factory_targets)
    ancilla_observables = np.asarray(ancilla_observables, dtype=np.uint8)
    if ancilla_observables.ndim == 1:
        if ancilla_observables.shape[0] != targets.shape[1]:
            raise ValueError(
                "Ancilla syndrome length does not match valid factory target length."
            )
        return bool(np.any(np.all(targets == ancilla_observables, axis=1)))
    if ancilla_observables.ndim == 2:
        if ancilla_observables.shape[1] != targets.shape[1]:
            raise ValueError(
                "Ancilla syndrome width does not match valid factory target length."
            )
        return np.any(
            np.all(
                ancilla_observables[:, None, :] == targets[None, :, :],
                axis=2,
            ),
            axis=1,
        )
    raise ValueError("Ancilla observables must be a 1D shot or 2D batch array.")
