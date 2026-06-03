from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, Protocol, overload

import numpy as np

from .tasks import DemoTask, ObservableFrame


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
def _sample_task_raw(
    task: SimulatorTask,
    shots: int,
    *,
    with_noise: bool = True,
    chunk_size: int | None = None,
    sim_type: str = "tsim",
) -> BasisDataset:
    """Sample detector and observable arrays without observable-frame rebasing."""

    if isinstance(task, DemoTask) and sim_type == "clifft":
        if chunk_size is None or shots <= chunk_size:
            detectors, observables = task.sample_clifft_det_obs(
                shots,
                with_noise=with_noise,
            )
            return BasisDataset(detectors=detectors, observables=observables)

        det_chunks = []
        obs_chunks = []
        remaining = shots
        while remaining > 0:
            batch = min(chunk_size, remaining)
            detectors, observables = task.sample_clifft_det_obs(
                batch,
                with_noise=with_noise,
            )
            det_chunks.append(detectors)
            obs_chunks.append(observables)
            remaining -= batch

        return BasisDataset(
            detectors=np.concatenate(det_chunks, axis=0),
            observables=np.concatenate(obs_chunks, axis=0),
        )

    if chunk_size is None or shots <= chunk_size:
        result = _run_simulator_task(
            task,
            shots,
            with_noise=with_noise,
            run_detectors=True,
            sim_type=sim_type,
        )
        return BasisDataset(
            detectors=np.asarray(result.detectors, dtype=np.uint8),
            observables=np.asarray(result.observables, dtype=np.uint8),
        )

    det_chunks = []
    obs_chunks = []
    remaining = shots
    while remaining > 0:
        batch = min(chunk_size, remaining)
        result = _run_simulator_task(
            task,
            batch,
            with_noise=with_noise,
            run_detectors=True,
            sim_type=sim_type,
        )
        det_chunks.append(np.asarray(result.detectors, dtype=np.uint8))
        obs_chunks.append(np.asarray(result.observables, dtype=np.uint8))
        remaining -= batch

    return BasisDataset(
        detectors=np.concatenate(det_chunks, axis=0),
        observables=np.concatenate(obs_chunks, axis=0),
    )


def _compute_observable_reference(
    task: DemoTask,
    *,
    shots: int = 64,
    sim_type: str = "tsim",
) -> np.ndarray:
    """Compute the deterministic noiseless observable reference for a task."""

    if task.observable_reference is not None:
        return np.asarray(task.observable_reference, dtype=np.uint8)

    reference_result = _run_simulator_task(
        task,
        shots,
        with_noise=False,
        run_detectors=True,
        sim_type=sim_type,
    )
    reference_obs = np.asarray(reference_result.observables, dtype=np.uint8)
    unique_rows = np.unique(reference_obs, axis=0)
    if len(unique_rows) != 1:
        raise RuntimeError(
            "Expected a deterministic noiseless observable reference row for this task."
        )
    task.observable_reference = unique_rows[0].copy()
    return np.asarray(task.observable_reference, dtype=np.uint8)


def _rebase_dataset_observables(
    dataset: BasisDataset,
    reference: np.ndarray,
) -> BasisDataset:
    """XOR dataset observables by a reference observable row."""

    return BasisDataset(
        detectors=dataset.detectors,
        observables=dataset.observables ^ reference.reshape(1, -1),
    )


def _normalize_observable_frame(
    task: SimulatorTask,
    dataset: BasisDataset,
    *,
    sim_type: str = "tsim",
) -> BasisDataset:
    """Apply task-specific observable-frame normalization to sampled data."""

    if not isinstance(task, DemoTask):
        return dataset
    if task.observable_frame != ObservableFrame.NOISELESS_REFERENCE_FLIPS:
        return dataset
    reference = _compute_observable_reference(task, sim_type=sim_type)
    return _rebase_dataset_observables(dataset, reference)


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

    while remaining > 0:
        batch = min(int(chunk_size), remaining)
        if isinstance(task, DemoTask) and sim_type == "clifft":
            detectors, observables = task.sample_clifft_det_obs(
                batch,
                with_noise=with_noise,
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
        yield _normalize_observable_frame(task, dataset, sim_type=sim_type)
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

    return _normalize_observable_frame(
        task,
        _sample_task_raw(
            task,
            shots,
            with_noise=with_noise,
            chunk_size=chunk_size,
            sim_type=sim_type,
        ),
        sim_type=sim_type,
    )
