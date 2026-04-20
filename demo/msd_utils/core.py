from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import binomtest

from .bayesian_tomography import posterior_fidelity_summary
from .common import (
    DEFAULT_SYNDROME_LAYOUT,
    DemoTask,
    ObservableFrame,
    SyndromeLayout,
)

DEFAULT_BASIS_LABELS = ("X", "Y", "Z")
DEFAULT_IDEAL_FACTORY_ACCEPTANCE = 1.0 / 6.0
DEFAULT_TARGET_BLOCH = np.ones(3, dtype=np.float64) / np.sqrt(3.0)


@dataclass(frozen=True)
class BasisDataset:
    detectors: np.ndarray
    observables: np.ndarray


def bits_to_key(bits: np.ndarray | Sequence[bool] | Sequence[int]) -> str:
    return "".join("1" if int(x) else "0" for x in bits)


def key_to_bits(key: str) -> np.ndarray:
    return np.fromiter((1 if c == "1" else 0 for c in key), dtype=np.uint8)


def pack_boolean_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.uint64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return np.sum(arr << np.arange(arr.shape[1], dtype=np.uint64), axis=1)


def packed_bits_to_int(bits: np.ndarray | Sequence[bool] | Sequence[int]) -> int:
    return int(pack_boolean_array(np.asarray(bits, dtype=np.uint8))[0])


def unpack_packed_bits(packed: int, length: int) -> np.ndarray:
    return ((int(packed) >> np.arange(length, dtype=np.uint64)) & 1).astype(np.uint8)


def logical_expectation(bits: np.ndarray) -> float:
    if len(bits) == 0:
        return float("nan")
    return float(np.mean(1.0 - 2.0 * np.asarray(bits, dtype=np.float64)))


def expectation_conf_interval(
    zero_count: int,
    one_count: int,
    *,
    confidence: float = 0.95,
    method: str = "wilson",
) -> np.ndarray:
    n = max(1, int(zero_count) + int(one_count))
    zero_interval = binomtest(int(zero_count), n).proportion_ci(
        confidence, method=method
    )
    return (
        2.0 * np.array([zero_interval.low, zero_interval.high], dtype=np.float64) - 1.0
    )


def expectation_with_error_bar(
    zero_count: int,
    one_count: int,
    *,
    confidence: float = 0.95,
    method: str = "wilson",
) -> tuple[float, float]:
    num_shots = max(int(zero_count) + int(one_count), 1)
    exp_val = float((int(zero_count) - int(one_count)) / num_shots)
    exp_interval = expectation_conf_interval(
        zero_count, one_count, confidence=confidence, method=method
    )
    exp_err = float((exp_interval[1] - exp_interval[0]) / 2.0)
    return exp_val, exp_err


def fidelity_from_counts(
    x_bits: np.ndarray,
    y_bits: np.ndarray,
    z_bits: np.ndarray,
    posterior_samples: int,
    *,
    sign_vector: Sequence[float] = (1.0, 1.0, 1.0),
    target_bloch: np.ndarray = DEFAULT_TARGET_BLOCH,
    uncertainty_backend: str = "wilson",
) -> dict[str, Any]:
    x_zero = int(np.sum(np.asarray(x_bits) == 0))
    x_one = int(np.sum(np.asarray(x_bits) != 0))
    y_zero = int(np.sum(np.asarray(y_bits) == 0))
    y_one = int(np.sum(np.asarray(y_bits) != 0))
    z_zero = int(np.sum(np.asarray(z_bits) == 0))
    z_one = int(np.sum(np.asarray(z_bits) != 0))

    return fidelity_from_zero_one_counts(
        x_zero,
        x_one,
        y_zero,
        y_one,
        z_zero,
        z_one,
        posterior_samples,
        sign_vector=sign_vector,
        target_bloch=target_bloch,
        uncertainty_backend=uncertainty_backend,
    )


def fidelity_from_zero_one_counts(
    x_zero: int,
    x_one: int,
    y_zero: int,
    y_one: int,
    z_zero: int,
    z_one: int,
    posterior_samples: int,
    *,
    sign_vector: Sequence[float] = (1.0, 1.0, 1.0),
    target_bloch: np.ndarray = DEFAULT_TARGET_BLOCH,
    uncertainty_backend: str = "wilson",
) -> dict[str, Any]:
    sign = np.asarray(sign_vector, dtype=np.float64)
    target = np.asarray(target_bloch, dtype=np.float64)

    ex, ex_err = expectation_with_error_bar(x_zero, x_one)
    ey, ey_err = expectation_with_error_bar(y_zero, y_one)
    ez, ez_err = expectation_with_error_bar(z_zero, z_one)

    bloch = np.array([ex, ey, ez], dtype=np.float64) * sign
    point = 0.5 + float(np.dot(bloch, target_bloch)) / 2.0
    if uncertainty_backend == "wilson":
        fidelity_err = 0.5 * float(
            np.sqrt(
                np.sum(
                    (target * np.array([ex_err, ey_err, ez_err], dtype=np.float64)) ** 2
                )
            )
        )
        low = point - fidelity_err
        high = point + fidelity_err
    elif uncertainty_backend == "bayesian_bloch_ball":
        n = np.array(
            [x_zero + x_one, y_zero + y_one, z_zero + z_one],
            dtype=np.int64,
        )
        k = np.array(
            [
                x_zero,
                y_zero,
                z_zero,
            ],
            dtype=np.int64,
        )
        posterior = posterior_fidelity_summary(
            n,
            k,
            sign=sign,
            target_bloch=target,
            posterior_samples=posterior_samples,
        )
        low = posterior["low"]
        high = posterior["high"]
        fidelity_err = posterior["error"]
        return {
            "point": float(point),
            "median": float(posterior["median"]),
            "low": float(low),
            "high": float(high),
            "error": float(fidelity_err),
            "bloch": tuple(float(x) for x in bloch),
        }
    else:
        raise ValueError(
            "uncertainty_backend must be 'wilson' or 'bayesian_bloch_ball'."
        )
    return {
        "point": float(point),
        "median": float(point),
        "low": float(low),
        "high": float(high),
        "error": float(fidelity_err),
        "bloch": tuple(float(x) for x in bloch),
    }


def magic_state_fidelity_point_from_counts(
    x_bits: np.ndarray,
    y_bits: np.ndarray,
    z_bits: np.ndarray,
    *,
    sign_vector: Sequence[float] = (1.0, -1.0, 1.0),
    target_bloch: np.ndarray = DEFAULT_TARGET_BLOCH,
) -> float:
    return float(
        fidelity_from_counts(
            x_bits,
            y_bits,
            z_bits,
            posterior_samples=4096,
            sign_vector=sign_vector,
            target_bloch=target_bloch,
            uncertainty_backend="wilson",
        )["point"]
    )


def sample_task_raw(
    task: Any,
    shots: int,
    *,
    with_noise: bool = True,
    chunk_size: int | None = 1_000_000,
) -> BasisDataset:
    if chunk_size is None or shots <= chunk_size:
        result = task.run(shots, with_noise=with_noise, run_detectors=True)
        return BasisDataset(
            detectors=np.asarray(result.detectors, dtype=np.uint8),
            observables=np.asarray(result.observables, dtype=np.uint8),
        )

    det_chunks = []
    obs_chunks = []
    remaining = shots
    while remaining > 0:
        batch = min(chunk_size, remaining)
        result = task.run(batch, with_noise=with_noise, run_detectors=True)
        det_chunks.append(np.asarray(result.detectors, dtype=np.uint8))
        obs_chunks.append(np.asarray(result.observables, dtype=np.uint8))
        remaining -= batch

    return BasisDataset(
        detectors=np.concatenate(det_chunks, axis=0),
        observables=np.concatenate(obs_chunks, axis=0),
    )


def compute_observable_reference(task: DemoTask, *, shots: int = 64) -> np.ndarray:
    if task.observable_reference is not None:
        return np.asarray(task.observable_reference, dtype=np.uint8)

    reference_result = task.run(shots, with_noise=False, run_detectors=True)
    reference_obs = np.asarray(reference_result.observables, dtype=np.uint8)
    unique_rows = np.unique(reference_obs, axis=0)
    if len(unique_rows) != 1:
        raise RuntimeError(
            "Expected a deterministic noiseless observable reference row for this task."
        )
    task.observable_reference = unique_rows[0].copy()
    return np.asarray(task.observable_reference, dtype=np.uint8)


def rebase_dataset_observables(
    dataset: BasisDataset,
    reference: np.ndarray,
) -> BasisDataset:
    return BasisDataset(
        detectors=dataset.detectors,
        observables=dataset.observables ^ reference.reshape(1, -1),
    )


def normalize_observable_frame(task: Any, dataset: BasisDataset) -> BasisDataset:
    if not isinstance(task, DemoTask):
        return dataset
    if task.observable_frame != ObservableFrame.NOISELESS_REFERENCE_FLIPS:
        return dataset
    reference = compute_observable_reference(task)
    return rebase_dataset_observables(dataset, reference)


def iter_task_datasets(
    task: Any,
    shots: int,
    *,
    with_noise: bool = True,
    chunk_size: int | None = 1_000_000,
):
    remaining = int(shots)
    if remaining < 0:
        raise ValueError("shots must be non-negative.")
    if remaining == 0:
        return

    if chunk_size is None:
        chunk_size = remaining

    while remaining > 0:
        batch = min(int(chunk_size), remaining)
        result = task.run(batch, with_noise=with_noise, run_detectors=True)
        dataset = BasisDataset(
            detectors=np.asarray(result.detectors, dtype=np.uint8),
            observables=np.asarray(result.observables, dtype=np.uint8),
        )
        yield normalize_observable_frame(task, dataset)
        remaining -= batch


def run_task(
    task: Any,
    shots: int,
    *,
    with_noise: bool = True,
    chunk_size: int | None = 1_000_000,
) -> BasisDataset:
    return normalize_observable_frame(
        task,
        sample_task_raw(
            task,
            shots,
            with_noise=with_noise,
            chunk_size=chunk_size,
        ),
    )


def split_factory_bits(
    detectors: np.ndarray,
    observables: np.ndarray,
    *,
    layout: SyndromeLayout = DEFAULT_SYNDROME_LAYOUT,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        detectors[:, layout.output_detector_count :],
        observables[:, layout.output_observable_count :],
    )


def normalize_valid_factory_targets(
    factory_targets: np.ndarray | Sequence[Sequence[int]] | Sequence[int],
) -> np.ndarray:
    targets = np.asarray(factory_targets, dtype=np.uint8)
    if targets.ndim == 1:
        targets = targets.reshape(1, -1)
    if targets.ndim != 2:
        raise ValueError(
            "Factory targets must be a 1D syndrome or a 2D array of valid syndromes."
        )
    if targets.shape[0] == 0 or targets.shape[1] == 0:
        raise ValueError("Need at least one non-empty valid factory syndrome.")
    return np.unique(targets, axis=0)


def resolve_valid_factory_targets(
    *,
    factory_target: np.ndarray | Sequence[int] | None = None,
    valid_factory_targets: np.ndarray | Sequence[Sequence[int]] | None = None,
) -> np.ndarray:
    if factory_target is not None and valid_factory_targets is not None:
        raise ValueError(
            "Pass either factory_target or valid_factory_targets, not both."
        )
    targets = (
        valid_factory_targets if valid_factory_targets is not None else factory_target
    )
    if targets is None:
        raise ValueError(
            "Need either factory_target or valid_factory_targets for postselection."
        )
    return normalize_valid_factory_targets(targets)


def ancilla_matches_valid_targets(
    ancilla_observables: np.ndarray,
    valid_factory_targets: np.ndarray | Sequence[Sequence[int]] | Sequence[int],
) -> np.ndarray | bool:
    targets = normalize_valid_factory_targets(valid_factory_targets)
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


def infer_factory_target(
    task_map: Mapping[str, Any],
    *,
    shots: int = 12_000,
    basis_labels: Sequence[str] = DEFAULT_BASIS_LABELS,
    ideal_factory_acceptance: float = DEFAULT_IDEAL_FACTORY_ACCEPTANCE,
) -> np.ndarray:
    counts: Counter[tuple[int, ...]] = Counter()
    for basis in basis_labels:
        data = run_task(task_map[basis], shots, with_noise=False)
        for row in data.observables[:, 1:]:
            counts[tuple(map(int, row))] += 1

    total = sum(counts.values())
    ranked = sorted(
        counts.items(),
        key=lambda item: (abs(item[1] / total - ideal_factory_acceptance), -item[1]),
    )
    print("Top noiseless ancilla branches:")
    for pattern, count in ranked[:8]:
        print(pattern, count / total)
    return np.asarray(ranked[0][0], dtype=np.uint8)


def infer_distilled_sign_vector(
    task_map: Mapping[str, Any],
    factory_target: np.ndarray | Sequence[int] | None = None,
    *,
    valid_factory_targets: np.ndarray | Sequence[Sequence[int]] | None = None,
    shots: int = 12_000,
    basis_labels: Sequence[str] = DEFAULT_BASIS_LABELS,
    target_bloch: np.ndarray = DEFAULT_TARGET_BLOCH,
) -> np.ndarray:
    targets = resolve_valid_factory_targets(
        factory_target=factory_target,
        valid_factory_targets=valid_factory_targets,
    )
    corrected: dict[str, np.ndarray] = {}
    for basis in basis_labels:
        data = run_task(task_map[basis], shots, with_noise=False)
        mask = ancilla_matches_valid_targets(data.observables[:, 1:], targets)
        corrected[basis] = data.observables[mask, 0].astype(np.uint8)

    raw_bloch = np.array(
        [
            logical_expectation(corrected["X"]),
            logical_expectation(corrected["Y"]),
            logical_expectation(corrected["Z"]),
        ]
    )

    sign_candidates = [
        np.array([sx, sy, sz], dtype=np.float64)
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    ]
    scored = sorted(
        (
            (float(np.dot(raw_bloch * sign, target_bloch)), sign)
            for sign in sign_candidates
        ),
        key=lambda item: item[0],
        reverse=True,
    )

    print("Noiseless accepted-branch Bloch:", raw_bloch)
    print("Chosen distilled sign vector:", scored[0][1], "score:", scored[0][0])
    return scored[0][1]


# NOTE: is NOT used in the decoders notebook, but is used in the reprod notebook (for naive postselection)
def naive_injected_summary(
    task_map: Mapping[str, Any],
    *,
    sign_vector: Sequence[float],
    posterior_samples: int,
    shots: int,
    require_zero_detectors: bool = False,
    min_accepted_per_basis: int = 50,
    basis_labels: Sequence[str] = DEFAULT_BASIS_LABELS,
    target_bloch: np.ndarray = DEFAULT_TARGET_BLOCH,
) -> dict[str, Any]:
    corrected: dict[str, np.ndarray] = {}
    accepted_fraction_by_basis: dict[str, float] = {}

    for basis in basis_labels:
        data = run_task(task_map[basis], shots, with_noise=True)
        mask = np.ones(len(data.observables), dtype=bool)
        if require_zero_detectors:
            mask &= np.all(data.detectors == 0, axis=1)

        corrected[basis] = data.observables[mask, 0].astype(np.uint8)
        accepted_fraction_by_basis[basis] = float(np.mean(mask))

    if min(len(corrected[basis]) for basis in basis_labels) < min_accepted_per_basis:
        raise RuntimeError("Too few accepted injected shots.")

    summary = fidelity_from_counts(
        corrected["X"],
        corrected["Y"],
        corrected["Z"],
        posterior_samples,
        sign_vector=sign_vector,
        target_bloch=target_bloch,
    )
    summary["accepted_fraction"] = float(
        np.mean(list(accepted_fraction_by_basis.values()))
    )
    summary["accepted_fraction_by_basis"] = accepted_fraction_by_basis
    return summary


# NOTE: is NOT used in the decoders notebook, but is used in the reprod notebook (for naive postselection)
def naive_distilled_summary(
    task_map: Mapping[str, Any],
    factory_target: np.ndarray | Sequence[int] | None = None,
    *,
    valid_factory_targets: np.ndarray | Sequence[Sequence[int]] | None = None,
    sign_vector: Sequence[float],
    posterior_samples: int,
    shots: int,
    require_zero_ancilla_detectors: bool = False,
    min_accepted_per_basis: int = 50,
    basis_labels: Sequence[str] = DEFAULT_BASIS_LABELS,
    target_bloch: np.ndarray = DEFAULT_TARGET_BLOCH,
) -> dict[str, Any]:
    targets = resolve_valid_factory_targets(
        factory_target=factory_target,
        valid_factory_targets=valid_factory_targets,
    )
    corrected: dict[str, np.ndarray] = {}
    accepted_fraction_by_basis: dict[str, float] = {}
    total_kept = 0
    total_shots = 0

    for basis in basis_labels:
        data = run_task(task_map[basis], shots, with_noise=True)
        anc_det, anc_obs = split_factory_bits(data.detectors, data.observables)
        mask = ancilla_matches_valid_targets(anc_obs, targets)
        if require_zero_ancilla_detectors:
            mask &= np.all(anc_det == 0, axis=1)

        corrected[basis] = data.observables[mask, 0].astype(np.uint8)
        accepted_fraction_by_basis[basis] = float(np.mean(mask))
        total_kept += int(np.sum(mask))
        total_shots += len(mask)

    if min(len(corrected[basis]) for basis in basis_labels) < min_accepted_per_basis:
        raise RuntimeError("Too few accepted distilled shots.")

    summary = fidelity_from_counts(
        corrected["X"],
        corrected["Y"],
        corrected["Z"],
        posterior_samples,
        sign_vector=sign_vector,
        target_bloch=target_bloch,
    )
    summary["accepted_fraction"] = total_kept / total_shots
    summary["accepted_fraction_by_basis"] = accepted_fraction_by_basis
    summary["valid_factory_targets"] = tuple(
        tuple(int(x) for x in row.tolist()) for row in targets
    )
    if len(targets) == 1:
        summary["factory_target"] = tuple(int(x) for x in targets[0].tolist())
    return summary
