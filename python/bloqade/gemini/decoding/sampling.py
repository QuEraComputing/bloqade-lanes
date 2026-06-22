from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tasks import _CliffTSimulatorTask


@dataclass(frozen=True)
class _BasisDataset:
    """Detector and observable samples for one tomography basis."""

    detectors: np.ndarray
    observables: np.ndarray


def _run_task(
    task: object,
    shots: int,
    *,
    with_noise: bool = True,
) -> _BasisDataset:
    """Sample detector/observable arrays from a demo task."""

    if shots < 0:
        raise ValueError("shots must be non-negative.")
    if isinstance(task, _CliffTSimulatorTask):
        detectors, observables = task._sample_detector_observables(
            shots,
            with_noise=with_noise,
        )
        return _BasisDataset(
            detectors=detectors,
            observables=observables,
        )

    result = task.run(shots, with_noise=with_noise, run_detectors=True)  # type: ignore[attr-defined]
    return _BasisDataset(
        detectors=np.asarray(result.detectors, dtype=np.uint8),
        observables=np.asarray(result.observables, dtype=np.uint8),
    )
