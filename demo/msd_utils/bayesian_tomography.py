from __future__ import annotations

from functools import lru_cache
from typing import Sequence, TypedDict

import numpy as np
from scipy.special import logsumexp

# REFACTOR: this should be an internal constant for stdlibs
DEFAULT_SIGN = np.array((1.0, 1.0, 1.0), dtype=np.float64)


# NOTE: technically, this type won't be exposed to the user
class PosteriorFidelitySummary(TypedDict):
    point: float
    median: float
    low: float
    high: float
    error: float


# REFACTOR: should be an internal function for stdlib
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


# TODO: make the prior for fidelity computation configurable
# REFACTOR: should be an internal function for stdlib
def _bures_measure(points: np.ndarray) -> np.ndarray:
    radii_sq = np.sum(points * points, axis=1)
    weights = np.zeros(len(points), dtype=np.float64)
    mask = radii_sq < 1.0
    weights[mask] = 1.0 / (np.pi**2 * np.sqrt(np.maximum(1.0 - radii_sq[mask], 1e-12)))
    return weights


# TODO: get rid of the 9 binary precision cap, to allow for the user to get arbitrarily precise tomography estimates.
# REFACTOR: should be an internal function for stdlib
def _grid_axis_points(binary_precision: int) -> int:
    return 2 ** min(9, max(4, int(binary_precision))) + 1


# REFACTOR: should be an internal function for stdlib
@lru_cache(maxsize=None)
def _grid_axis_values(axis_points: int) -> np.ndarray:
    edges = np.linspace(-1.0, 1.0, axis_points + 1, dtype=np.float64)
    return (edges[:-1] + edges[1:]) / 2.0


# REFACTOR: should be an internal function for stdlib
def _axis_likelihood_window(
    values: np.ndarray,
    n_i: int,
    k_i: int,
    *,
    binary_precision: int,
    min_points: int = 33,
) -> np.ndarray:
    if n_i <= 0 or len(values) <= min_points:
        return values

    probs = np.clip((1.0 + values) / 2.0, 1e-12, 1.0 - 1e-12)
    log_likelihood = k_i * np.log(probs) + (n_i - k_i) * np.log1p(-probs)
    relative_log_likelihood = log_likelihood - np.max(log_likelihood)
    mask = relative_log_likelihood > -float(binary_precision) * np.log(2.0)

    if np.any(mask) and int(mask.sum()) >= min_points:
        return values[mask]

    center = int(np.argmax(log_likelihood))
    radius = min_points // 2
    start = max(0, center - radius)
    stop = min(len(values), center + radius + 1)
    if stop - start < min_points:
        if start == 0:
            stop = min(len(values), min_points)
        else:
            start = max(0, len(values) - min_points)
    return values[start:stop]


# REFACTOR: should be an internal function for stdlib
def _downsample_axis(values: np.ndarray, keep: int) -> np.ndarray:
    if len(values) <= keep:
        return values
    indices = np.linspace(0, len(values) - 1, num=keep, dtype=int)
    return values[np.unique(indices)]


# REFACTOR: should be an internal function for stdlib
# TODO: I think this "adaptive bloch ball thing" can give us maybe 'faster' estimates of the fidelity.
# If you really wanted to be more precise at the cost of more compute, then you could consider switching this implementation. -- a TODO would
# be to make this implementation more extensible/allow the user to choose the precision of their bloch ball estimates.
def _adaptive_bloch_ball_grid(
    axis_points: int,
    n: np.ndarray,
    k: np.ndarray,
    *,
    binary_precision: int,
    max_grid_points: int,
) -> np.ndarray:
    axis_values = _grid_axis_values(axis_points)
    subsets = [
        _axis_likelihood_window(
            axis_values,
            int(n_i),
            int(k_i),
            binary_precision=binary_precision,
        )
        for n_i, k_i in zip(n, k, strict=True)
    ]

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

    broad_axis = _downsample_axis(axis_values, min(len(axis_values), 113))
    x, y, z = np.meshgrid(broad_axis, broad_axis, broad_axis, indexing="ij")
    broad_points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    return broad_points[np.sum(broad_points * broad_points, axis=1) <= 1.0]


# REFACTOR: should be a standard library function
def posterior_fidelity_summary(
    n: np.ndarray,
    k: np.ndarray,
    target_bloch: np.ndarray,
    *,
    sign: np.ndarray = DEFAULT_SIGN,
    binary_precision: int | None = None,
    max_grid_points: int = 1_500_000,
) -> PosteriorFidelitySummary:
    if binary_precision is None:
        binary_precision = 9
    else:
        binary_precision = max(4, int(binary_precision))
    axis_points = _grid_axis_points(binary_precision)
    points = _adaptive_bloch_ball_grid(
        axis_points,
        n,
        k,
        binary_precision=binary_precision,
        max_grid_points=max_grid_points,
    )

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
        return {
            "point": float(point),
            "median": float(point),
            "low": float(point),
            "high": float(point),
            "error": 0.0,
        }

    weights = np.exp(log_weights[finite] - logsumexp(log_weights[finite]))
    corrected_points = points[finite] * sign
    fidelities = np.clip(
        0.5 + np.sum(corrected_points * target_bloch.reshape(1, 3), axis=1) / 2.0,
        0.0,
        1.0,
    )
    if weights.sum() <= 0.0:
        point = 0.5
        return {
            "point": float(point),
            "median": float(point),
            "low": float(point),
            "high": float(point),
            "error": 0.0,
        }

    low, median, high = weighted_quantile(fidelities, [0.16, 0.5, 0.84], weights)
    error = max(median - low, high - median)
    point = float(np.average(fidelities, weights=weights))
    return {
        "point": point,
        "median": float(median),
        "low": float(low),
        "high": float(high),
        "error": float(error),
    }
