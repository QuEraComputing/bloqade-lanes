from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.special import logsumexp

DEFAULT_BASIS_LABELS = ("X", "Y", "Z")
DEFAULT_IDEAL_FACTORY_ACCEPTANCE = 1.0 / 6.0
DEFAULT_TARGET_BLOCH = np.array([1.0, -1.0, 1.0], dtype=np.float64) / np.sqrt(3.0)


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


def logical_expectation(bits: np.ndarray) -> float:
    if len(bits) == 0:
        return float("nan")
    return float(np.mean(1.0 - 2.0 * np.asarray(bits, dtype=np.float64)))


def weighted_quantile(
    values: np.ndarray,
    quantiles: Sequence[float],
    weights: np.ndarray,
) -> np.ndarray:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    return np.interp(np.asarray(quantiles, dtype=np.float64), cdf, values)


def sample_bloch_ball(num_samples: int, seed: int = 1234) -> np.ndarray:
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(num_samples, 3))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    zero_mask = norms[:, 0] == 0.0
    while np.any(zero_mask):
        directions[zero_mask] = rng.normal(size=(int(np.sum(zero_mask)), 3))
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        zero_mask = norms[:, 0] == 0.0
    directions /= norms
    radii = rng.random(num_samples) ** (1.0 / 3.0)
    return directions * radii[:, None]


def fidelity_from_counts(
    x_bits: np.ndarray,
    y_bits: np.ndarray,
    z_bits: np.ndarray,
    posterior_samples: int,
    *,
    sign_vector: Sequence[float] = (1.0, 1.0, 1.0),
    target_bloch: np.ndarray = DEFAULT_TARGET_BLOCH,
) -> dict[str, Any]:
    ex = logical_expectation(x_bits)
    ey = logical_expectation(y_bits)
    ez = logical_expectation(z_bits)

    sign = np.asarray(sign_vector, dtype=np.float64)
    bloch = np.array([ex, ey, ez], dtype=np.float64) * sign
    point = 0.5 + float(np.dot(bloch, target_bloch)) / 2.0

    n = np.array([len(x_bits), len(y_bits), len(z_bits)], dtype=np.int64)
    k = np.array(
        [
            int(np.sum(np.asarray(x_bits) == 0)),
            int(np.sum(np.asarray(y_bits) == 0)),
            int(np.sum(np.asarray(z_bits) == 0)),
        ],
        dtype=np.int64,
    )

    points = sample_bloch_ball(posterior_samples)
    probs = np.clip((1.0 + points) / 2.0, 1e-12, 1.0 - 1e-12)
    log_weights = (k * np.log(probs) + (n - k) * np.log1p(-probs)).sum(axis=1)
    weights = np.exp(log_weights - logsumexp(log_weights))

    corrected_points = points * sign
    fidelities = 0.5 + (corrected_points @ target_bloch) / 2.0
    q16, q50, q84 = weighted_quantile(fidelities, [0.16, 0.5, 0.84], weights)
    return {
        "point": float(point),
        "median": float(q50),
        "low": float(q16),
        "high": float(q84),
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
        )["point"]
    )


def run_task(
    task: Any,
    shots: int,
    *,
    with_noise: bool = True,
    chunk_size: int | None = None,
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


def split_factory_bits(
    detectors: np.ndarray,
    observables: np.ndarray,
    *,
    detector_prefix: int = 3,
    observable_prefix: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    return detectors[:, detector_prefix:], observables[:, observable_prefix:]


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
    factory_target: np.ndarray,
    *,
    shots: int = 12_000,
    basis_labels: Sequence[str] = DEFAULT_BASIS_LABELS,
    target_bloch: np.ndarray = DEFAULT_TARGET_BLOCH,
) -> np.ndarray:
    corrected: dict[str, np.ndarray] = {}
    for basis in basis_labels:
        data = run_task(task_map[basis], shots, with_noise=False)
        mask = np.all(data.observables[:, 1:] == factory_target, axis=1)
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


def naive_distilled_summary(
    task_map: Mapping[str, Any],
    factory_target: np.ndarray,
    *,
    sign_vector: Sequence[float],
    posterior_samples: int,
    shots: int,
    require_zero_ancilla_detectors: bool = False,
    min_accepted_per_basis: int = 50,
    basis_labels: Sequence[str] = DEFAULT_BASIS_LABELS,
    target_bloch: np.ndarray = DEFAULT_TARGET_BLOCH,
) -> dict[str, Any]:
    corrected: dict[str, np.ndarray] = {}
    accepted_fraction_by_basis: dict[str, float] = {}
    total_kept = 0
    total_shots = 0

    for basis in basis_labels:
        data = run_task(task_map[basis], shots, with_noise=True)
        anc_det, anc_obs = split_factory_bits(data.detectors, data.observables)
        mask = np.all(anc_obs == factory_target, axis=1)
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
    summary["factory_target"] = tuple(int(x) for x in factory_target.tolist())
    return summary
