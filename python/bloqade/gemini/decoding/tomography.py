"""Minimal single-qubit tomography helpers for MSD postselection demos."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypedDict

import numpy as np


class SimpleFidelitySummary(TypedDict):
    """Target-specific point fidelity summary."""

    point: float


DEFAULT_TARGET_BLOCH = np.ones(3, dtype=np.float64) / np.sqrt(3.0)


def _density_matrix_from_bloch(bloch: np.ndarray) -> np.ndarray:
    bloch = np.asarray(bloch, dtype=np.float64)
    if bloch.shape != (3,):
        raise ValueError("bloch must be a length-3 vector.")
    x, y, z = bloch
    return 0.5 * np.array(
        [[1.0 + z, x - 1j * y], [x + 1j * y, 1.0 - z]],
        dtype=np.complex128,
    )


@dataclass(frozen=True, init=False)
class TomographyResult:
    """Point-estimate single-qubit tomography result."""

    density_matrix: np.ndarray

    def __init__(
        self,
        zero_counts: np.ndarray | Sequence[int],
        one_counts: np.ndarray | Sequence[int],
    ) -> None:
        zero_arr = np.asarray(zero_counts, dtype=np.int64).reshape(-1)
        one_arr = np.asarray(one_counts, dtype=np.int64).reshape(-1)
        if zero_arr.shape != (3,) or one_arr.shape != (3,):
            raise ValueError("zero_counts and one_counts must be length-3 arrays.")

        totals = np.maximum(zero_arr + one_arr, 1)
        bloch = ((zero_arr - one_arr) / totals).astype(np.float64)
        object.__setattr__(self, "density_matrix", _density_matrix_from_bloch(bloch))

    def fidelity_bloch(
        self,
        target_bloch: np.ndarray | Sequence[float],
    ) -> SimpleFidelitySummary:
        """Return the overlap with a pure target state from its Bloch vector."""

        target_density_matrix = _density_matrix_from_bloch(
            np.asarray(target_bloch, dtype=np.float64)
        )
        # NOTE: only works for pure states
        point = float(np.real(np.trace(self.density_matrix @ target_density_matrix)))
        return {"point": point}


__all__ = [
    "DEFAULT_TARGET_BLOCH",
    "SimpleFidelitySummary",
    "TomographyResult",
]
