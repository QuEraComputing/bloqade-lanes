from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.special import logsumexp
from scipy.stats import binomtest

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


def _bures_measure(points: np.ndarray) -> np.ndarray:
    radii_sq = np.sum(points * points, axis=1)
    weights = np.zeros(len(points), dtype=np.float64)
    mask = radii_sq < 1.0
    weights[mask] = 1.0 / (np.pi**2 * np.sqrt(np.maximum(1.0 - radii_sq[mask], 1e-12)))
    return weights


def _grid_axis_points(posterior_samples: int) -> int:
    # Use a much finer 1D grid than the previous cube-root heuristic, closer in
    # spirit to the paper's binary-precision Bloch grid. We keep the total work
    # manageable later by adaptively cropping around the measured expectations.
    binary_precision = max(
        4,
        int(round(np.log2(float(max(posterior_samples, 1))) / 2.0)),
    )
    return min(513, 2**binary_precision + 1)


@lru_cache(maxsize=None)
def _grid_axis_values(axis_points: int) -> np.ndarray:
    # Use cell centers instead of endpoints so the Bures prior stays finite.
    edges = np.linspace(-1.0, 1.0, axis_points + 1, dtype=np.float64)
    return (edges[:-1] + edges[1:]) / 2.0


def _axis_window(
    values: np.ndarray, n_i: int, k_i: int, *, min_points: int = 33
) -> np.ndarray:
    if n_i <= 0 or len(values) <= min_points:
        return values

    p = float(np.clip(k_i / n_i, 1e-6, 1.0 - 1e-6))
    mean = 2.0 * p - 1.0
    sigma = 2.0 * np.sqrt(max(p * (1.0 - p), 1.0 / max(4 * n_i, 1)) / n_i)
    half_width = max(0.08, 8.0 * sigma)
    low = max(-1.0, mean - half_width)
    high = min(1.0, mean + half_width)

    mask = (values >= low) & (values <= high)
    if int(mask.sum()) >= min_points:
        return values[mask]

    center = int(np.argmin(np.abs(values - mean)))
    radius = min_points // 2
    start = max(0, center - radius)
    stop = min(len(values), center + radius + 1)
    if stop - start < min_points:
        if start == 0:
            stop = min(len(values), min_points)
        else:
            start = max(0, len(values) - min_points)
    return values[start:stop]


def _downsample_axis(values: np.ndarray, keep: int) -> np.ndarray:
    if len(values) <= keep:
        return values
    indices = np.linspace(0, len(values) - 1, num=keep, dtype=int)
    return values[np.unique(indices)]


def _adaptive_bloch_ball_grid(
    axis_points: int, n: np.ndarray, k: np.ndarray
) -> np.ndarray:
    axis_values = _grid_axis_values(axis_points)
    subsets = [
        _axis_window(axis_values, int(n_i), int(k_i))
        for n_i, k_i in zip(n, k, strict=True)
    ]

    max_grid_points = 1_500_000
    total_points = len(subsets[0]) * len(subsets[1]) * len(subsets[2])
    if total_points > max_grid_points:
        scale = (total_points / max_grid_points) ** (1.0 / 3.0)
        subsets = [
            _downsample_axis(values, max(33, int(round(len(values) / scale))))
            for values in subsets
        ]

    x, y, z = np.meshgrid(*subsets, indexing="ij")
    points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    points = points[np.sum(points * points, axis=1) <= 1.0]
    if len(points):
        return points

    # Rare edge case: with very few accepted shots, the empirical expectations
    # can sit near +/-1 on all three axes. A tight crop around that corner of
    # the cube can miss the physical Bloch ball entirely and spuriously drive
    # the posterior fallback to fidelity 0.5. When that happens, fall back to a
    # broad but still downsampled full-axis grid.
    broad_axis = _downsample_axis(axis_values, min(len(axis_values), 113))
    x, y, z = np.meshgrid(broad_axis, broad_axis, broad_axis, indexing="ij")
    broad_points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    return broad_points[np.sum(broad_points * broad_points, axis=1) <= 1.0]


def _histogram_quantiles(
    edges: np.ndarray,
    probabilities: np.ndarray,
    quantiles: Sequence[float],
) -> np.ndarray:
    cdf = np.cumsum(probabilities)
    values = []
    for quantile in quantiles:
        idx = int(np.searchsorted(cdf, quantile, side="left"))
        idx = min(max(idx, 0), len(probabilities) - 1)
        low = cdf[idx - 1] if idx > 0 else 0.0
        high = cdf[idx]
        frac = 0.0 if high <= low else (quantile - low) / (high - low)
        values.append(edges[idx] + (edges[idx + 1] - edges[idx]) * frac)
    return np.asarray(values, dtype=np.float64)


def _posterior_fidelity_quantiles(
    n: np.ndarray,
    k: np.ndarray,
    *,
    sign: np.ndarray,
    target_bloch: np.ndarray,
    posterior_samples: int,
) -> np.ndarray:
    axis_points = _grid_axis_points(posterior_samples)
    points = _adaptive_bloch_ball_grid(axis_points, n, k)

    probs = np.clip((1.0 + points) / 2.0, 1e-12, 1.0 - 1e-12)
    log_likelihood = (k * np.log(probs) + (n - k) * np.log1p(-probs)).sum(axis=1)

    prior = _bures_measure(points)
    log_prior = np.full(len(points), -np.inf, dtype=np.float64)
    positive_prior = prior > 0.0
    log_prior[positive_prior] = np.log(prior[positive_prior])

    log_weights = log_likelihood + log_prior
    finite = np.isfinite(log_weights)
    if not np.any(finite):
        point = 0.5
        return np.array([point, point, point], dtype=np.float64)

    weights = np.exp(log_weights[finite] - logsumexp(log_weights[finite]))
    corrected_points = points[finite] * sign
    fidelities = np.clip(
        0.5 + np.sum(corrected_points * target_bloch.reshape(1, 3), axis=1) / 2.0,
        0.0,
        1.0,
    )
    if weights.sum() <= 0.0:
        point = 0.5
        return np.array([point, point, point], dtype=np.float64)
    return weighted_quantile(fidelities, [0.16, 0.5, 0.84], weights)


def fidelity_from_counts(
    x_bits: np.ndarray,
    y_bits: np.ndarray,
    z_bits: np.ndarray,
    posterior_samples: int,
    *,
    sign_vector: Sequence[float] = (1.0, 1.0, 1.0),
    target_bloch: np.ndarray = DEFAULT_TARGET_BLOCH,
) -> dict[str, Any]:
    sign = np.asarray(sign_vector, dtype=np.float64)
    target = np.asarray(target_bloch, dtype=np.float64)

    x_zero = int(np.sum(np.asarray(x_bits) == 0))
    x_one = int(np.sum(np.asarray(x_bits) != 0))
    y_zero = int(np.sum(np.asarray(y_bits) == 0))
    y_one = int(np.sum(np.asarray(y_bits) != 0))
    z_zero = int(np.sum(np.asarray(z_bits) == 0))
    z_one = int(np.sum(np.asarray(z_bits) != 0))

    ex, ex_err = expectation_with_error_bar(x_zero, x_one)
    ey, ey_err = expectation_with_error_bar(y_zero, y_one)
    ez, ez_err = expectation_with_error_bar(z_zero, z_one)

    bloch = np.array([ex, ey, ez], dtype=np.float64) * sign
    point = 0.5 + float(np.dot(bloch, target_bloch)) / 2.0
    del posterior_samples

    # Propagate Wilson-style expectation errors into a fidelity interval,
    # matching the uncertainty model used in distillation_sim.
    fidelity_err = 0.5 * float(
        np.sqrt(
            np.sum((target * np.array([ex_err, ey_err, ez_err], dtype=np.float64)) ** 2)
        )
    )
    low = point - fidelity_err
    high = point + fidelity_err
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
        )["point"]
    )


def run_task(
    task: Any,
    shots: int,
    *,
    with_noise: bool = True,
    chunk_size: int | None = 1_000_000,
) -> BasisDataset:
    def _observable_reference() -> np.ndarray:
        cached = task.__dict__.get("_observable_reference")
        if cached is not None:
            return np.asarray(cached, dtype=np.uint8)

        reference_result = task.run(64, with_noise=False, run_detectors=True)
        reference_obs = np.asarray(reference_result.observables, dtype=np.uint8)
        unique_rows = np.unique(reference_obs, axis=0)
        if len(unique_rows) != 1:
            raise RuntimeError(
                "Expected a deterministic noiseless observable reference row for this task."
            )
        reference = unique_rows[0].copy()
        task.__dict__["_observable_reference"] = reference
        return reference

    def _maybe_rebase_observables(observables: np.ndarray) -> np.ndarray:
        if not task.__dict__.get("_rebase_observables_to_noiseless_reference", False):
            return observables
        reference = _observable_reference().reshape(1, -1)
        return observables ^ reference

    if chunk_size is None or shots <= chunk_size:
        result = task.run(shots, with_noise=with_noise, run_detectors=True)
        return BasisDataset(
            detectors=np.asarray(result.detectors, dtype=np.uint8),
            observables=_maybe_rebase_observables(
                np.asarray(result.observables, dtype=np.uint8)
            ),
        )

    det_chunks = []
    obs_chunks = []
    remaining = shots
    while remaining > 0:
        batch = min(chunk_size, remaining)
        result = task.run(batch, with_noise=with_noise, run_detectors=True)
        det_chunks.append(np.asarray(result.detectors, dtype=np.uint8))
        obs_chunks.append(
            _maybe_rebase_observables(np.asarray(result.observables, dtype=np.uint8))
        )
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
