from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Protocol, TypeVar

import numpy as np
from bloqade.decoders import BaseDecoder
from demo.msd_utils.domain.confidence import ConfidenceDecoder
from demo.msd_utils.standard.bit_packing import (
    pack_boolean_array,
    packed_bits_to_int,
    packed_pattern_targets,
    unpack_packed_bits,
)
from demo.msd_utils.standard.tomography import (
    DEFAULT_TARGET_BLOCH,
    fidelity_from_counts,
    fidelity_from_zero_one_counts,
)

from .constants import DEFAULT_BASIS_LABELS
from .layout import (
    DEFAULT_SYNDROME_LAYOUT,
    SyndromeLayout,
    _ancilla_matches_valid_targets,
    _normalize_valid_factory_targets,
    split_factory_bits,
)
from .sampling import BasisDataset

DecodeResult = TypeVar("DecodeResult")


class _ProgressBar(Protocol):
    """Small subset of the tqdm progress-bar API used here."""

    total: float | None

    def update(self, n: int = 1) -> object: ...

    def set_description_str(self, desc: str, refresh: bool = True) -> object: ...

    def refresh(self) -> object: ...

    def close(self) -> object: ...


@dataclass(frozen=True)
class _PackedThresholdDataset:
    """Packed detector/observable arrays used during threshold evaluation."""

    anc_det: np.ndarray
    anc_obs: np.ndarray
    packed_full_det: np.ndarray
    packed_anc_det: np.ndarray
    packed_anc_obs: np.ndarray
    output_bits: np.ndarray


@dataclass(frozen=True)
class DecoderAdapter:
    """Decoder pair plus cached full/factory decode callables.

    Attributes:
        full_decoder: Decoder used for full detector syndromes, if retained.
        factory_decoder: Confidence decoder used for factory syndromes, if
            retained.
        decode_factory: Callable from packed factory syndrome to
            ``(factory_correction, confidence_score)``.
        decode_full: Callable from packed full syndrome to output correction.
        factory_score_mode: Label for the factory confidence score.
    """

    # NOTE: full_decoder and factory_decoder fields might not be strictly needed; can maybe remove them (and core functionality is preserved.)
    full_decoder: BaseDecoder | None
    factory_decoder: ConfidenceDecoder | None
    decode_factory: Callable[[int], tuple[tuple[int, ...], float]]
    decode_full: Callable[[int], tuple[int, ...]]
    # Names the score returned by decode_factory so callers can label plots/logs.
    factory_score_mode: str
    # NOTE, mtg: add the name of the decoder?


def _make_decoder_adapter(
    *,
    full_decoder: BaseDecoder,
    factory_decoder: ConfidenceDecoder,
    full_syndrome_length: int,
    factory_syndrome_length: int,
    factory_decode_impl: Callable[[np.ndarray], tuple[np.ndarray, float]],
    factory_score_mode: str,
) -> DecoderAdapter:
    """Create a decoder adapter with integer-keyed cached decode functions."""

    # NOTE: the reason why we have to pack/unpack these bits is because lru_cache doesn't work for numpy arrays
    # but works for ints. So that's why the signature here for decode_factory takes in an int for the syndrome.
    @lru_cache(maxsize=None)
    def decode_factory(packed_syndrome: int):
        syndrome = unpack_packed_bits(
            int(packed_syndrome),
            factory_syndrome_length,
        ).astype(bool)
        correction, score = factory_decode_impl(syndrome)
        return tuple(
            int(x) for x in np.asarray(correction, dtype=np.uint8).tolist()
        ), float(score)

    @lru_cache(maxsize=None)
    def decode_full(packed_syndrome: int):
        syndrome = unpack_packed_bits(
            int(packed_syndrome),
            full_syndrome_length,
        ).astype(bool)
        correction = np.asarray(full_decoder.decode(syndrome), dtype=np.uint8)
        return tuple(int(x) for x in correction.tolist())

    return DecoderAdapter(
        full_decoder=full_decoder,
        factory_decoder=factory_decoder,
        decode_factory=decode_factory,
        decode_full=decode_full,
        factory_score_mode=factory_score_mode,
    )


def _call_decoder_fn(
    fn: Callable[[int], DecodeResult],
    bits: np.ndarray,
) -> DecodeResult:
    """Pack syndrome bits and call an integer-keyed decoder function."""

    return fn(packed_bits_to_int(bits))


# TODO: refactor this; probably reused logic from quantile logic in
# bayesian_tomography.py? Can continue here on 05/11
# not sure if this is really repeated logic..
def _weighted_quantiles_from_counts(
    values: np.ndarray,
    weights: np.ndarray,
    quantiles: np.ndarray,
) -> np.ndarray:
    """Compute quantiles from compressed value/count data."""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.int64).reshape(-1)
    quantiles = np.asarray(quantiles, dtype=np.float64).reshape(-1)
    if len(values) != len(weights):
        raise ValueError("values and weights must have the same length.")
    if len(values) == 0:
        return np.array([], dtype=np.float64)

    keep = weights > 0
    values = values[keep]
    weights = weights[keep]
    if len(values) == 0:
        return np.array([], dtype=np.float64)

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative_end = np.cumsum(weights) - 1
    total = int(np.sum(weights))
    if total <= 1:
        return np.full_like(quantiles, values[0], dtype=np.float64)

    def value_at_rank(rank: int) -> float:
        idx = int(np.searchsorted(cumulative_end, rank, side="left"))
        return float(values[idx])

    out = np.empty_like(quantiles, dtype=np.float64)
    for i, q in enumerate(quantiles):
        q = float(np.clip(q, 0.0, 1.0))
        rank = q * (total - 1)
        lo = int(np.floor(rank))
        hi = int(np.ceil(rank))
        frac = rank - lo
        vlo = value_at_rank(lo)
        vhi = value_at_rank(hi)
        out[i] = (1.0 - frac) * vlo + frac * vhi
    return out


def _pack_threshold_dataset(
    dataset: BasisDataset,
    *,
    layout: SyndromeLayout,
) -> _PackedThresholdDataset:
    """Pack repeated threshold-evaluation columns into integer arrays."""

    anc_det, anc_obs = split_factory_bits(
        dataset.detectors,
        dataset.observables,
        layout=layout,
    )
    return _PackedThresholdDataset(
        anc_det=anc_det,
        # NOTE: anc_obs is not used elsewhere in the code...
        anc_obs=anc_obs,
        packed_full_det=pack_boolean_array(dataset.detectors).astype(np.uint64),
        packed_anc_det=pack_boolean_array(anc_det).astype(np.uint64),
        packed_anc_obs=pack_boolean_array(anc_obs).astype(np.uint64),
        output_bits=np.asarray(dataset.observables[:, 0], dtype=np.uint8).astype(
            np.uint64
        ),
    )


# TODO: can modify the arguments here; don't need the whole DecoderAdapter; can
# just take in the decode_factory.
def _build_ancilla_decode_cache(
    decoder: DecoderAdapter,
    packed_anc_det: np.ndarray,
    *,
    syndrome_length: int,
    progress_update: Callable[[int], None] | None = None,
) -> dict[int, tuple[int, float]]:
    """Decode each unique ancilla detector pattern once."""

    unique_anc_det, shot_counts = np.unique(packed_anc_det, return_counts=True)
    ancilla_decode_cache: dict[int, tuple[int, float]] = {}
    for packed, shot_count in zip(unique_anc_det.tolist(), shot_counts, strict=True):
        anc_flip, score = _call_decoder_fn(
            decoder.decode_factory,
            unpack_packed_bits(int(packed), syndrome_length),
        )
        anc_flip_bits = np.asarray(anc_flip, dtype=np.uint8)
        ancilla_decode_cache[int(packed)] = (
            int(pack_boolean_array(anc_flip_bits)[0]) if len(anc_flip_bits) else 0,
            float(score),
        )
        if progress_update is not None:
            progress_update(int(shot_count))
    return ancilla_decode_cache


# TODO: can modify the arguments here; don't need the whole DecoderAdapter; can
# just take in the decode_full.
def _build_full_decode_cache(
    decoder: DecoderAdapter,
    packed_full_det: np.ndarray,
    *,
    syndrome_length: int,
    shot_counts: np.ndarray | None = None,
    progress_update: Callable[[int], None] | None = None,
) -> dict[int, int]:
    """Decode each unique full detector pattern once."""

    if shot_counts is None:
        unique_full_det, unique_shot_counts = np.unique(
            packed_full_det,
            return_counts=True,
        )
    else:
        unique_full_det = np.asarray(packed_full_det, dtype=np.uint64)
        unique_shot_counts = np.asarray(shot_counts, dtype=np.int64)
        if unique_full_det.shape != unique_shot_counts.shape:
            raise ValueError(
                "packed_full_det and shot_counts must have the same shape."
            )

    full_decode_cache: dict[int, int] = {}
    for packed, shot_count in zip(
        unique_full_det.tolist(),
        unique_shot_counts,
        strict=True,
    ):
        full_flip = np.asarray(
            _call_decoder_fn(
                decoder.decode_full,
                unpack_packed_bits(int(packed), syndrome_length),
            ),
            dtype=np.uint8,
        )
        full_decode_cache[int(packed)] = int(full_flip[0]) if len(full_flip) else 0
        if progress_update is not None:
            progress_update(int(shot_count))
    return full_decode_cache


def _score_count_dict_to_arrays(
    score_to_counts: Mapping[float, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert score-to-zero/one-count mappings into sorted arrays."""

    if not score_to_counts:
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
        )
    scores = np.array(sorted(score_to_counts), dtype=np.float64)
    zero_counts = np.array(
        [int(score_to_counts[score][0]) for score in scores],
        dtype=np.int64,
    )
    one_counts = np.array(
        [int(score_to_counts[score][1]) for score in scores],
        dtype=np.int64,
    )
    return scores, zero_counts, one_counts


def _build_generic_threshold_tables(
    actual_data: Mapping[str, BasisDataset],
    decoder_map: Mapping[str, DecoderAdapter],
    *,
    targets: np.ndarray,
    basis_labels: Sequence[str],
    layout: SyndromeLayout,
    progress_label: str | None = None,
) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    np.ndarray,
    np.ndarray,
    int,
]:
    """Build score/count tables shared by threshold-curve evaluators."""

    packed_targets = packed_pattern_targets(targets)
    per_basis_tables: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    global_score_weights: defaultdict[float, int] = defaultdict(int)
    total_shots = 0
    progress_bars: dict[str, _ProgressBar] = {}
    if progress_label is not None:
        from tqdm.auto import tqdm

        progress_bars = {
            basis: tqdm(
                total=len(actual_data[basis].observables),
                desc=f"{progress_label} {basis}: factory decode",
                unit="shots",
                position=index,
                leave=True,
            )
            for index, basis in enumerate(basis_labels)
        }

    try:
        for basis in basis_labels:
            dataset = actual_data[basis]
            total_shots += len(dataset.observables)
            packed_dataset = _pack_threshold_dataset(dataset, layout=layout)
            progress_bar = progress_bars.get(basis)

            def update_basis_progress(advance: int) -> None:
                if progress_bar is not None:
                    progress_bar.update(advance)

            ancilla_decode_cache = _build_ancilla_decode_cache(
                decoder_map[basis],
                packed_dataset.packed_anc_det,
                syndrome_length=packed_dataset.anc_det.shape[1],
                progress_update=update_basis_progress,
            )
            grouped = np.column_stack(
                [
                    packed_dataset.packed_anc_det,
                    packed_dataset.packed_anc_obs,
                    packed_dataset.packed_full_det,
                    packed_dataset.output_bits,
                ]
            )
            unique_groups, group_counts = np.unique(grouped, axis=0, return_counts=True)

            score_to_counts: defaultdict[float, np.ndarray] = defaultdict(
                lambda: np.zeros(2, dtype=np.int64)
            )
            accepted_groups: list[tuple[float, int, int, int]] = []
            full_decode_weights: defaultdict[int, int] = defaultdict(int)
            for row, count in zip(unique_groups, group_counts, strict=True):
                packed_a_det = int(row[0])
                packed_a_obs = int(row[1])
                packed_det = int(row[2])
                output_bit = int(row[3])

                anc_flip_packed, score = ancilla_decode_cache[packed_a_det]
                if not np.isfinite(score):
                    continue
                corrected_anc_packed = packed_a_obs ^ anc_flip_packed
                if corrected_anc_packed not in packed_targets:
                    continue

                accepted_groups.append(
                    (float(score), packed_det, output_bit, int(count))
                )
                full_decode_weights[packed_det] += int(count)

            if progress_bar is not None:
                accepted_shots = int(sum(full_decode_weights.values()))
                progress_bar.set_description_str(
                    f"{progress_label} {basis}: full decode"
                )
                progress_bar.total = len(dataset.observables) + accepted_shots
                progress_bar.refresh()

            accepted_full_dets = np.fromiter(
                full_decode_weights.keys(),
                dtype=np.uint64,
                count=len(full_decode_weights),
            )
            accepted_full_counts = np.fromiter(
                full_decode_weights.values(),
                dtype=np.int64,
                count=len(full_decode_weights),
            )
            full_decode_cache = _build_full_decode_cache(
                decoder_map[basis],
                accepted_full_dets,
                syndrome_length=dataset.detectors.shape[1],
                shot_counts=accepted_full_counts,
                progress_update=update_basis_progress,
            )

            if progress_bar is not None:
                progress_bar.set_description_str(f"{progress_label} {basis}: decoded")
                progress_bar.refresh()

            for score, packed_det, output_bit, count in accepted_groups:
                corrected_output_bit = output_bit ^ full_decode_cache[packed_det]
                score_to_counts[score][corrected_output_bit] += count
                global_score_weights[score] += count

            per_basis_tables[basis] = _score_count_dict_to_arrays(score_to_counts)
    finally:
        for progress_bar in progress_bars.values():
            progress_bar.close()

    global_scores = np.array(sorted(global_score_weights), dtype=np.float64)
    global_weights = np.array(
        [int(global_score_weights[score]) for score in global_scores], dtype=np.int64
    )
    return per_basis_tables, global_scores, global_weights, total_shots


def _counts_for_threshold(
    scores: np.ndarray,
    zero_counts: np.ndarray,
    one_counts: np.ndarray,
    threshold: float,
) -> tuple[int, int]:
    """Sum zero/one counts whose scores are at least the threshold."""

    if len(scores) == 0:
        return 0, 0
    idx = int(np.searchsorted(scores, threshold, side="left"))
    if idx >= len(scores):
        return 0, 0
    return int(np.sum(zero_counts[idx:])), int(np.sum(one_counts[idx:]))


# NOTE: in principle this is pretty generic (just depends on you having scores
# for your counts), but because the fn's it depend on are application-level, i
# think this should also be application-level.
def _evaluate_cached_threshold_curve(
    per_basis_tables: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    score_array: np.ndarray,
    *,
    score_weights: np.ndarray | None = None,
    binary_precision: int | None,
    threshold_points: int,
    sign_vector: Sequence[float],
    target_bloch: np.ndarray,
    basis_labels: Sequence[str],
    min_accepted_per_basis: int,
    threshold_policy: str,
    total_shots: int,
    uncertainty_backend: str,
    max_grid_points: int,
) -> dict[str, np.ndarray]:
    """Evaluate a fidelity curve from precomputed score/count tables."""

    if len(score_array) == 0:
        raise RuntimeError("No factory-accepted shots found for threshold sweep")

    if threshold_policy == "quantile":
        if score_weights is None:
            thresholds = np.unique(
                np.quantile(score_array, np.linspace(0.0, 1.0, threshold_points))
            )
        else:
            thresholds = np.unique(
                _weighted_quantiles_from_counts(
                    score_array,
                    score_weights,
                    np.linspace(0.0, 1.0, threshold_points),
                )
            )
    elif threshold_policy == "unique_values":
        thresholds = np.unique(score_array)
    elif threshold_policy == "linear_range":
        score_min = float(np.min(score_array))
        score_max = float(np.max(score_array))
        if np.isclose(score_min, score_max):
            thresholds = np.array([score_min], dtype=np.float64)
        else:
            thresholds = np.linspace(score_min, score_max, threshold_points)
    else:
        raise ValueError(
            "threshold_policy must be 'quantile', 'unique_values', or 'linear_range'."
        )

    accepted_fractions = []
    fidelities = []
    point_fidelities = []
    credibility = []

    for threshold in thresholds:
        counts_by_basis: dict[str, tuple[int, int]] = {}
        total_kept = 0
        for basis in basis_labels:
            scores, zero_counts, one_counts = per_basis_tables[basis]
            basis_zero, basis_one = _counts_for_threshold(
                scores,
                zero_counts,
                one_counts,
                float(threshold),
            )
            counts_by_basis[basis] = (basis_zero, basis_one)
            total_kept += basis_zero + basis_one

        if (
            min(sum(counts_by_basis[basis]) for basis in basis_labels)
            < min_accepted_per_basis
        ):
            continue

        # TODO: generalize beyond just single-qubit tomography?
        summary = fidelity_from_zero_one_counts(
            counts_by_basis["X"][0],
            counts_by_basis["X"][1],
            counts_by_basis["Y"][0],
            counts_by_basis["Y"][1],
            counts_by_basis["Z"][0],
            counts_by_basis["Z"][1],
            binary_precision,
            sign_vector=sign_vector,
            target_bloch=target_bloch,
            uncertainty_backend=uncertainty_backend,
            max_grid_points=max_grid_points,
        )
        accepted_fractions.append(total_kept / total_shots)
        point_fidelities.append(summary["point"])
        fidelities.append(summary["median"])
        credibility.append((summary["low"], summary["high"]))

    accepted_array = np.asarray(accepted_fractions, dtype=np.float64)
    fidelity_array = np.asarray(fidelities, dtype=np.float64)
    point_fidelity_array = np.asarray(point_fidelities, dtype=np.float64)
    credible_array = np.asarray(credibility, dtype=np.float64)

    if len(accepted_array) > 0:
        order = np.argsort(accepted_array)
        accepted_array = accepted_array[order]
        fidelity_array = fidelity_array[order]
        point_fidelity_array = point_fidelity_array[order]
        credible_array = credible_array[order]

    return {
        "accepted_fraction": accepted_array,
        "fidelity": fidelity_array,
        "point_fidelity": point_fidelity_array,
        "credible": credible_array,
    }


def evaluate_curve(
    actual_data: Mapping[str, BasisDataset],
    decoder_map: Mapping[str, DecoderAdapter],
    *,
    binary_precision: int | None = None,
    threshold_points: int,
    metric: str,
    valid_factory_targets: np.ndarray | Sequence[Sequence[int]] | Sequence[int],
    sign_vector: Sequence[float],
    target_bloch: np.ndarray = DEFAULT_TARGET_BLOCH,
    basis_labels: Sequence[str] = DEFAULT_BASIS_LABELS,
    min_accepted_per_basis: int = 50,
    threshold_policy: str = "quantile",
    selection_mode: str = "threshold",
    layout: SyndromeLayout = DEFAULT_SYNDROME_LAYOUT,
    uncertainty_backend: str = "wilson",
    max_grid_points: int = 1_500_000,
    progress_label: str | None = None,
) -> dict[str, np.ndarray]:
    """Evaluate a postselection curve for decoder confidence thresholds.

    Args:
        actual_data: Basis-labeled detector/observable samples to evaluate.
        decoder_map: Basis-labeled decoder adapters.
        binary_precision: Precision used by Bayesian tomography scoring.
        threshold_points: Number of thresholds to evaluate for continuous
            threshold policies.
        metric: Human-readable metric name used in error messages.
        valid_factory_targets: Valid corrected factory observable patterns.
        sign_vector: Per-axis sign convention for fidelity reconstruction.
        target_bloch: Target Bloch vector for fidelity calculation.
        basis_labels: Tomography basis labels to evaluate.
        min_accepted_per_basis: Minimum accepted samples required per basis.
        threshold_policy: Threshold selection policy.
        selection_mode: ``"threshold"`` or ``"pattern_rank"``.
        layout: Syndrome layout separating output and factory bits.
        uncertainty_backend: Fidelity uncertainty backend.
        max_grid_points: Maximum adaptive grid size for Bayesian tomography.
        progress_label: Optional Rich progress label shown while decoding
            actual-data syndromes.

    Returns:
        Dictionary containing accepted fractions, fidelities, and intervals.
    """

    if selection_mode == "pattern_rank":
        return evaluate_mld_curve(
            actual_data,
            decoder_map,
            binary_precision=binary_precision,
            valid_factory_targets=valid_factory_targets,
            sign_vector=sign_vector,
            target_bloch=target_bloch,
            basis_labels=basis_labels,
            min_accepted_per_basis=min_accepted_per_basis,
            layout=layout,
            uncertainty_backend=uncertainty_backend,
            max_grid_points=max_grid_points,
        )
    if selection_mode != "threshold":
        raise ValueError("selection_mode must be 'threshold' or 'pattern_rank'.")

    targets = _normalize_valid_factory_targets(valid_factory_targets)
    (
        per_basis_tables,
        score_array,
        score_weights,
        total_shots,
    ) = _build_generic_threshold_tables(
        actual_data,
        decoder_map,
        targets=targets,
        basis_labels=basis_labels,
        layout=layout,
        progress_label=progress_label,
    )
    try:
        return _evaluate_cached_threshold_curve(
            per_basis_tables,
            score_array,
            score_weights=score_weights,
            binary_precision=binary_precision,
            threshold_points=threshold_points,
            sign_vector=sign_vector,
            target_bloch=target_bloch,
            basis_labels=basis_labels,
            min_accepted_per_basis=min_accepted_per_basis,
            threshold_policy=threshold_policy,
            total_shots=total_shots,
            uncertainty_backend=uncertainty_backend,
            max_grid_points=max_grid_points,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"No factory-accepted shots found for {metric} threshold sweep"
        ) from exc


# This is SPECIFICALLY for the pattern rank case where we don't compute explicit
# thresholds, but rather literally have the points plotted on the curve based on
# MLD fidelity.
def evaluate_mld_curve(
    actual_data: Mapping[str, BasisDataset],
    decoder_map: Mapping[str, DecoderAdapter],
    *,
    binary_precision: int | None = None,
    valid_factory_targets: np.ndarray | Sequence[Sequence[int]] | Sequence[int],
    sign_vector: Sequence[float],
    target_bloch: np.ndarray = DEFAULT_TARGET_BLOCH,
    basis_labels: Sequence[str] = DEFAULT_BASIS_LABELS,
    min_accepted_per_basis: int = 50,
    layout: SyndromeLayout = DEFAULT_SYNDROME_LAYOUT,
    uncertainty_backend: str = "wilson",
    max_grid_points: int = 1_500_000,
) -> dict[str, np.ndarray]:
    """Evaluate the MLD pattern-rank postselection curve.

    Args:
        actual_data: Basis-labeled detector/observable samples to evaluate.
        decoder_map: Basis-labeled MLD decoder adapters.
        binary_precision: Precision used by Bayesian tomography scoring.
        valid_factory_targets: Valid corrected factory observable patterns.
        sign_vector: Per-axis sign convention for fidelity reconstruction.
        target_bloch: Target Bloch vector for fidelity calculation.
        basis_labels: Tomography basis labels to evaluate.
        min_accepted_per_basis: Minimum accepted samples required per basis.
        layout: Syndrome layout separating output and factory bits.
        uncertainty_backend: Fidelity uncertainty backend.
        max_grid_points: Maximum adaptive grid size for Bayesian tomography.

    Returns:
        Dictionary containing accepted fractions, fidelities, and intervals.
    """

    targets = _normalize_valid_factory_targets(valid_factory_targets)
    pattern_counts_by_basis: dict[str, dict[int, int]] = {}
    corrected_bits_by_basis: dict[str, dict[int, list[int]]] = {}
    pattern_scores: dict[int, float] = {}
    total_shots = 0

    for basis in basis_labels:
        dataset = actual_data[basis]
        total_shots += len(dataset.observables)
        anc_det, anc_obs = split_factory_bits(
            dataset.detectors,
            dataset.observables,
            layout=layout,
        )
        decode_factory = decoder_map[basis].decode_factory
        decode_full = decoder_map[basis].decode_full

        pattern_counts = defaultdict(int)
        corrected_bits = defaultdict(list)

        for det, obs, a_det, a_obs in zip(
            dataset.detectors,
            dataset.observables,
            anc_det,
            anc_obs,
            strict=True,
        ):
            packed = int(pack_boolean_array(a_det)[0])
            pattern_counts[packed] += 1

            anc_flip, score = _call_decoder_fn(decode_factory, a_det)
            anc_flip = np.asarray(anc_flip, dtype=np.uint8)
            if np.isfinite(score):
                existing = pattern_scores.get(packed)
                if existing is None:
                    pattern_scores[packed] = float(score)
                elif not np.isclose(existing, score, rtol=1e-9, atol=1e-12):
                    raise ValueError(
                        f"Inconsistent MLD score for ancilla detector pattern {packed}."
                    )
            if not _ancilla_matches_valid_targets(a_obs ^ anc_flip, targets):
                continue

            full_flip = np.asarray(_call_decoder_fn(decode_full, det), dtype=np.uint8)
            corrected_bits[packed].append(int(obs[0] ^ full_flip[0]))

        pattern_counts_by_basis[basis] = dict(pattern_counts)
        corrected_bits_by_basis[basis] = corrected_bits

    all_patterns = sorted(
        set().union(*(pattern_counts_by_basis[basis].keys() for basis in basis_labels))
    )
    ranked_patterns = []
    for pattern in all_patterns:
        score = pattern_scores.get(pattern)
        if score is None or not np.isfinite(score):
            continue
        total_count = sum(
            pattern_counts_by_basis[basis].get(pattern, 0) for basis in basis_labels
        )
        if total_count <= 0:
            continue
        ranked_patterns.append((pattern, float(score), total_count))

    ranked_patterns.sort(key=lambda row: (row[1], row[2]), reverse=True)

    cumulative_bits = {basis: [] for basis in basis_labels}
    accepted_fractions = []
    fidelities = []
    credibility = []

    for pattern, _score, _count in ranked_patterns:
        for basis in basis_labels:
            cumulative_bits[basis].extend(
                corrected_bits_by_basis[basis].get(pattern, ())
            )

        if (
            min(len(cumulative_bits[basis]) for basis in basis_labels)
            < min_accepted_per_basis
        ):
            continue

        summary = fidelity_from_counts(
            np.asarray(cumulative_bits["X"], dtype=np.uint8),
            np.asarray(cumulative_bits["Y"], dtype=np.uint8),
            np.asarray(cumulative_bits["Z"], dtype=np.uint8),
            binary_precision,
            sign_vector=sign_vector,
            target_bloch=target_bloch,
            uncertainty_backend=uncertainty_backend,
            max_grid_points=max_grid_points,
        )
        total_kept = sum(len(cumulative_bits[basis]) for basis in basis_labels)
        accepted_fractions.append(total_kept / total_shots)
        fidelities.append(summary["median"])
        credibility.append((summary["low"], summary["high"]))

    return {
        "accepted_fraction": np.asarray(accepted_fractions, dtype=np.float64),
        "fidelity": np.asarray(fidelities, dtype=np.float64),
        "credible": np.asarray(credibility, dtype=np.float64),
    }
