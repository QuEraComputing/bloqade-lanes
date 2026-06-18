from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tasks import DemoTask


@dataclass(frozen=True)
class BasisDataset:
    """Detector and observable samples for one tomography basis."""

    detectors: np.ndarray
    observables: np.ndarray


def run_task(
    task: DemoTask,
    shots: int,
    *,
    with_noise: bool = True,
    chunk_size: int | None = None,
    sim_type: str = "tsim",
) -> BasisDataset:
    """Sample detector/observable arrays from a demo task."""

    # TODO: This is a wrapper to support both GeminiLogicalSimulatorTask and DemoTask.
    # Ideally, we can substitute this with something that better allows the user to
    # swap either using tsim or clifft as simulator backends.
    if shots < 0:
        raise ValueError("shots must be non-negative.")
    if chunk_size is None:
        chunk_size = shots
    if chunk_size <= 0 and shots > 0:
        raise ValueError("chunk_size must be positive when provided.")

    detector_chunks: list[np.ndarray] = []
    observable_chunks: list[np.ndarray] = []
    remaining = int(shots)
    while remaining > 0:
        batch = min(int(chunk_size), remaining)
        detectors, observables = task.sample_detector_observables(
            batch,
            with_noise=with_noise,
            sim_type=sim_type,
        )
        detector_chunks.append(detectors)
        observable_chunks.append(observables)
        remaining -= batch

    if not detector_chunks:
        return BasisDataset(
            detectors=np.zeros((0, 0), dtype=np.uint8),
            observables=np.zeros((0, 0), dtype=np.uint8),
        )
    return BasisDataset(
        detectors=np.concatenate(detector_chunks, axis=0),
        observables=np.concatenate(observable_chunks, axis=0),
    )


__all__ = ["BasisDataset", "run_task"]
