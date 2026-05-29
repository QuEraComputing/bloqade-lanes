from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import numpy as np
from scipy.special import logsumexp
from scipy.stats import binomtest

from .types import FidelitySummary, PosteriorFidelitySummary

# TODO: should (1, 1, 1) be the default bloch vector in the standard library???
DEFAULT_TARGET_BLOCH = np.ones(3, dtype=np.float64) / np.sqrt(3.0)
_DEFAULT_SIGN = np.array((1.0, 1.0, 1.0), dtype=np.float64)


def _weighted_quantile(
    values: np.ndarray,
    quantiles: Sequence[float],
    weights: np.ndarray,
) -> np.ndarray:
    """Compute weighted quantiles from explicitly weighted samples."""

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    return np.interp(np.asarray(quantiles, dtype=np.float64), cdf, values)


# TODO: make the prior for fidelity computation configurable
def _bures_measure(points: np.ndarray) -> np.ndarray:
    """Evaluate the Bures prior density at Bloch-ball points."""

    radii_sq = np.sum(points * points, axis=1)
    weights = np.zeros(len(points), dtype=np.float64)
    mask = radii_sq < 1.0
    weights[mask] = 1.0 / (np.pi**2 * np.sqrt(np.maximum(1.0 - radii_sq[mask], 1e-12)))
    return weights


# TODO: get rid of the 9 binary precision cap, to allow for the user to get
# arbitrarily precise tomography estimates.
def _grid_axis_points(binary_precision: int) -> int:
    """Resolve binary precision into a capped number of grid cells per axis."""

    return 2 ** min(9, max(4, int(binary_precision))) + 1


@lru_cache(maxsize=None)
def _grid_axis_values(axis_points: int) -> np.ndarray:
    """Return centered grid values spanning the Bloch coordinate interval."""

    edges = np.linspace(-1.0, 1.0, axis_points + 1, dtype=np.float64)
    return (edges[:-1] + edges[1:]) / 2.0


def _axis_likelihood_window(
    values: np.ndarray,
    n_i: int,
    k_i: int,
    *,
    binary_precision: int,
    min_points: int = 33,
) -> np.ndarray:
    """Restrict one Bloch coordinate grid to values supported by the likelihood."""

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


def _downsample_axis(values: np.ndarray, keep: int) -> np.ndarray:
    """Downsample an axis grid while preserving endpoints approximately."""

    if len(values) <= keep:
        return values
    indices = np.linspace(0, len(values) - 1, num=keep, dtype=int)
    return values[np.unique(indices)]


# TODO: I think this "adaptive bloch ball thing" can give us maybe 'faster'
# estimates of the fidelity. If you really wanted to be more precise at the
# cost of more compute, then you could consider switching this implementation.
# A TODO would be to make this implementation more extensible/allow the user
# to choose the precision of their bloch ball estimates.
def _adaptive_bloch_ball_grid(
    axis_points: int,
    n: np.ndarray,
    k: np.ndarray,
    *,
    binary_precision: int,
    max_grid_points: int,
) -> np.ndarray:
    """Build a likelihood-adapted grid of valid Bloch-ball points."""

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


def posterior_fidelity_summary(
    n: np.ndarray,
    k: np.ndarray,
    target_bloch: np.ndarray,
    *,
    sign: np.ndarray = _DEFAULT_SIGN,
    binary_precision: int | None = None,
    max_grid_points: int = 1_500_000,
) -> PosteriorFidelitySummary:
    """Estimate fidelity and credible interval from tomography counts.
    Applies the Bayesian analysis on Bloch vectors described in the "Estimation
    of Confidence Intervals" section in the MSD paper: https://arxiv.org/pdf/2412.15165

    Args:
        n: Number of shots per tomography axis, ordered as X/Y/Z.
        k: Number of zero outcomes per tomography axis, ordered as X/Y/Z.
        target_bloch: Target Bloch vector used to convert sampled Bloch points
            into fidelities.
        sign: Sign convention applied to candidate Bloch vectors before
            computing fidelity.
        binary_precision: Controls the Bloch-ball grid resolution. ``None``
            uses the default precision.
        max_grid_points: Maximum number of adaptive grid points to evaluate.

    Returns:
        Posterior mean, median, credible interval, and error bar.
    """

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

    low, median, high = _weighted_quantile(fidelities, [0.16, 0.5, 0.84], weights)
    error = max(median - low, high - median)
    point = float(np.average(fidelities, weights=weights))
    return {
        "point": point,
        "median": float(median),
        "low": float(low),
        "high": float(high),
        "error": float(error),
    }


def logical_expectation(bits: np.ndarray) -> float:
    """Convert logical 0/1 samples into a Pauli expectation value.

    Args:
        bits: Logical observable bits, where 0 maps to ``+1`` and 1 maps to
            ``-1``.

    Returns:
        Mean Pauli expectation value, or ``nan`` for an empty input.
    """

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
    """Return a binomial confidence interval for a Pauli expectation.
    Used to compute the likelihood term in Bayes' rule.

    Args:
        zero_count: Number of logical-zero outcomes.
        one_count: Number of logical-one outcomes.
        confidence: Confidence level for the interval.
        method: Method passed to ``scipy.stats.binomtest(...).proportion_ci``.

    Returns:
        Two-element array containing the lower and upper expectation bounds.
    """

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
    """Return a Pauli expectation and symmetric error bar.
    Used in the "Wilson" uncertainty method for estimating uncertainty from tomography.

    Args:
        zero_count: Number of logical-zero outcomes.
        one_count: Number of logical-one outcomes.
        confidence: Confidence level for the interval used to derive the error.
        method: Method passed to ``expectation_conf_interval``.

    Returns:
        A pair ``(expectation, error_bar)``.
    """

    num_shots = max(int(zero_count) + int(one_count), 1)
    exp_val = float((int(zero_count) - int(one_count)) / num_shots)
    exp_interval = expectation_conf_interval(
        zero_count, one_count, confidence=confidence, method=method
    )
    exp_err = float((exp_interval[1] - exp_interval[0]) / 2.0)
    return exp_val, exp_err


# TODO: as a standard library, DEFAULT_TARGET_BLOCH should not be the default
# argument here.
def fidelity_from_counts(
    x_bits: np.ndarray,
    y_bits: np.ndarray,
    z_bits: np.ndarray,
    binary_precision: int | None = None,
    *,
    sign_vector: Sequence[float] = (1.0, 1.0, 1.0),
    target_bloch: np.ndarray = DEFAULT_TARGET_BLOCH,
    uncertainty_backend: str = "wilson",
    max_grid_points: int = 1_500_000,
) -> FidelitySummary:
    """Compute fidelity summary from X/Y/Z logical observable bit arrays.
    Switches between "bayesian_bloch_ball" for confidence interval estimation (used
    in the MSD paper) or "wilson".

    Args:
        x_bits: Logical observable bits from X-basis tomography.
        y_bits: Logical observable bits from Y-basis tomography.
        z_bits: Logical observable bits from Z-basis tomography.
        binary_precision: Precision used by the Bayesian Bloch-ball backend.
        sign_vector: Per-axis sign convention applied to reconstructed
            expectations.
        target_bloch: Target Bloch vector for fidelity calculation.
        uncertainty_backend: Either ``"wilson"`` or ``"bayesian_bloch_ball"``.
        max_grid_points: Maximum adaptive grid size for the Bayesian backend.

    Returns:
        Fidelity summary including point estimate, interval, error, and Bloch
        vector.
    """

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
        binary_precision,
        sign_vector=sign_vector,
        target_bloch=target_bloch,
        uncertainty_backend=uncertainty_backend,
        max_grid_points=max_grid_points,
    )


# TODO: as a standard library, DEFAULT_TARGET_BLOCH should not be the default
# argument here.
# TODO: make this function more general than just logical single qubit tomography?
# I think we had to explicitly pass in x_zero, x_one, ... because of some
# speed/runtime issues to not convert massive numpy arrays back and forth, but
# i'm not sure if that's really generic enough of a speed bottleneck to optimize
# for in general
def fidelity_from_zero_one_counts(
    x_zero: int,
    x_one: int,
    y_zero: int,
    y_one: int,
    z_zero: int,
    z_one: int,
    binary_precision: int | None = None,
    *,
    sign_vector: Sequence[float] = (1.0, 1.0, 1.0),
    target_bloch: np.ndarray = DEFAULT_TARGET_BLOCH,
    uncertainty_backend: str = "wilson",
    max_grid_points: int = 1_500_000,
) -> FidelitySummary:
    """Compute fidelity summary from X/Y/Z zero/one count pairs.
    Switches between "bayesian_bloch_ball" for confidence interval estimation (used
    in the MSD paper) or "wilson".

    Args:
        x_zero: Number of X-basis zero outcomes.
        x_one: Number of X-basis one outcomes.
        y_zero: Number of Y-basis zero outcomes.
        y_one: Number of Y-basis one outcomes.
        z_zero: Number of Z-basis zero outcomes.
        z_one: Number of Z-basis one outcomes.
        binary_precision: Precision used by the Bayesian Bloch-ball backend.
        sign_vector: Per-axis sign convention applied to reconstructed
            expectations.
        target_bloch: Target Bloch vector for fidelity calculation.
        uncertainty_backend: Either ``"wilson"`` or ``"bayesian_bloch_ball"``.
        max_grid_points: Maximum adaptive grid size for the Bayesian backend.

    Returns:
        Fidelity summary including point estimate, interval, error, and Bloch
        vector.
    """

    sign = np.asarray(sign_vector, dtype=np.float64)
    target = np.asarray(target_bloch, dtype=np.float64)

    ex, ex_err = expectation_with_error_bar(x_zero, x_one)
    ey, ey_err = expectation_with_error_bar(y_zero, y_one)
    ez, ez_err = expectation_with_error_bar(z_zero, z_one)

    print(f"fidelity_from_zero_one_counts, ex: {ex}, ey: {ey}, ez: {ez}")

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
            target,
            sign=sign,
            binary_precision=binary_precision,
            max_grid_points=max_grid_points,
        )
        low = posterior["low"]
        high = posterior["high"]
        fidelity_err = posterior["error"]
        return {
            "point": float(point),
            "median": posterior["median"],
            "low": float(low),
            "high": float(high),
            "error": float(fidelity_err),
            "bloch": (float(bloch[0]), float(bloch[1]), float(bloch[2])),
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
        "bloch": (float(bloch[0]), float(bloch[1]), float(bloch[2])),
    }
