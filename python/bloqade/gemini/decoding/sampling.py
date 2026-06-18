from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, Protocol, overload

import numpy as np

from .tasks import DemoTask


@dataclass(frozen=True)
class BasisDataset:
    """Detector and observable samples for one tomography basis.

    Attributes:
        detectors: Detector samples with shape ``(shots, num_detectors)``.
        observables: Observable samples with shape ``(shots, num_observables)``.
    """

    detectors: np.ndarray
    observables: np.ndarray


class DetectorObservableResult(Protocol):
    @property
    def detectors(self) -> object: ...

    @property
    def observables(self) -> object: ...


class SimulatorTask(Protocol):
    @overload
    def run(
        self,
        shots: int,
        with_noise: bool = True,
        *,
        run_detectors: Literal[False] = ...,
    ) -> object: ...

    @overload
    def run(
        self,
        shots: int,
        with_noise: bool = True,
        *,
        run_detectors: Literal[True],
    ) -> DetectorObservableResult: ...

    @overload
    def run(
        self,
        shots: int,
        with_noise: bool = True,
        *,
        run_detectors: bool,
    ) -> object | DetectorObservableResult: ...


# TODO: This is a wrapper to support both GeminiLogicalSimulatorTask and DemoTask.
# Ideally, we can substitute this with something that better allows the user to
# swap either using tsim or clifft as simulator backends.
def _run_simulator_task(
    task: SimulatorTask,
    shots: int,
    *,
    with_noise: bool,
    run_detectors: bool,
    sim_type: str,
) -> DetectorObservableResult:
    """Run a simulator task through the selected backend."""

    if not run_detectors:
        raise ValueError("_run_simulator_task is only used for detector sampling.")
    if isinstance(task, DemoTask):
        return task.run(
            shots,
            with_noise=with_noise,
            run_detectors=run_detectors,
            sim_type=sim_type,
        )
    if sim_type != "tsim":
        raise ValueError(
            f"sim_type is {sim_type}; currently, the only supported simulator "
            "backends are 'tsim' and 'clifft'"
        )
    return task.run(shots, with_noise=with_noise, run_detectors=run_detectors)


# TODO: has separate clifft explicit check to avoid converting from
# np.array -> list -> np.array. I think the problem is that DetectorResult
# contains lists, so we have to do conversions to numpy arrays. So that's why
# we are currently calling a clifft method that returns np.array's directly.
# To change this, we'd need to change the task.run() interface/the DetectorResult
# return type.
def _iter_task_datasets(
    task: SimulatorTask,
    shots: int,
    *,
    with_noise: bool = True,
    chunk_size: int | None = None,
    sim_type: str = "tsim",
) -> Iterator[BasisDataset]:
    """Yield sampled datasets in chunks, applying observable-frame normalization."""

    remaining = int(shots)
    if remaining < 0:
        raise ValueError("shots must be non-negative.")
    if remaining == 0:
        return

    if chunk_size is None:
        chunk_size = remaining
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive when provided.")

    while remaining > 0:
        batch = min(int(chunk_size), remaining)
        if isinstance(task, DemoTask):
            detectors, observables = task.sample_detector_observables(
                batch,
                with_noise=with_noise,
                sim_type=sim_type,
            )
            dataset = BasisDataset(detectors=detectors, observables=observables)
        else:
            result = _run_simulator_task(
                task,
                batch,
                with_noise=with_noise,
                run_detectors=True,
                sim_type=sim_type,
            )
            dataset = BasisDataset(
                detectors=np.asarray(result.detectors, dtype=np.uint8),
                observables=np.asarray(result.observables, dtype=np.uint8),
            )
        yield dataset
        remaining -= batch


def run_task(
    task: SimulatorTask,
    shots: int,
    *,
    with_noise: bool = True,
    chunk_size: int | None = None,
    sim_type: str = "tsim",
) -> BasisDataset:
    """Sample a simulator task and return detector/observable arrays.
    Note that the simulator backend is currently configured through the "sim_type" argument.

    Args:
        task: Simulator task or ``DemoTask`` to sample.
        shots: Number of shots to sample.
        with_noise: Whether to sample the noisy circuit path.
        chunk_size: Optional maximum shots per simulator call. ``None`` samples
            all shots in one call.
        sim_type: Simulator backend, currently ``"tsim"`` or ``"clifft"`` for
            ``DemoTask`` instances.

    Returns:
        A ``BasisDataset`` containing detector and observable arrays.
    """

    if isinstance(task, DemoTask) and (chunk_size is None or shots <= chunk_size):
        detectors, observables = task.sample_detector_observables(
            shots,
            with_noise=with_noise,
            sim_type=sim_type,
        )
        return BasisDataset(detectors=detectors, observables=observables)

    datasets = list(
        _iter_task_datasets(
            task,
            shots,
            with_noise=with_noise,
            chunk_size=chunk_size,
            sim_type=sim_type,
        )
    )
    if not datasets:
        return BasisDataset(
            detectors=np.zeros((0, 0), dtype=np.uint8),
            observables=np.zeros((0, 0), dtype=np.uint8),
        )
    return BasisDataset(
        detectors=np.concatenate([dataset.detectors for dataset in datasets], axis=0),
        observables=np.concatenate(
            [dataset.observables for dataset in datasets],
            axis=0,
        ),
    )
