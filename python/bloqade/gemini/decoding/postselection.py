from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from demo.msd_utils.standard.bit_packing import (
    pack_boolean_array,
    packed_pattern_targets,
    unpack_packed_bits,
)
from demo.msd_utils.standard.tomography import DEFAULT_TARGET_BLOCH, TomographyResult

from .constants import DEFAULT_BASIS_LABELS
from .layout import DEFAULT_SYNDROME_LAYOUT, split_factory_bits
from .sampling import BasisDataset


@dataclass(frozen=True)
class DecoderAdapter:
    """Factory and full-syndrome decoder callables."""

    # NOTE: full_decoder and factory_decoder fields might not be strictly needed; can maybe remove them (and core functionality is preserved.)
    decode_factory: Callable[[np.ndarray], tuple[np.ndarray, float]]
    decode_full: Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class _PackedThresholdDataset:
    anc_det: np.ndarray
    packed_full_det: np.ndarray
    packed_anc_det: np.ndarray
    packed_anc_obs: np.ndarray
    output_bits: np.ndarray


def _weighted_quantiles_from_counts(
    values: np.ndarray,
    weights: np.ndarray,
    quantiles: np.ndarray,
) -> np.ndarray:
    # TODO: refactor this; probably reused logic from quantile logic in
    # bayesian_tomography.py? Can continue here on 05/11
    # not sure if this is really repeated logic..
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.int64).reshape(-1)
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

    out = np.empty_like(quantiles, dtype=np.float64)
    for i, q in enumerate(np.asarray(quantiles, dtype=np.float64).reshape(-1)):
        rank = float(np.clip(q, 0.0, 1.0)) * (total - 1)
        lo = int(np.floor(rank))
        hi = int(np.ceil(rank))
        frac = rank - lo
        lo_idx = int(np.searchsorted(cumulative_end, lo, side="left"))
        hi_idx = int(np.searchsorted(cumulative_end, hi, side="left"))
        out[i] = (1.0 - frac) * float(values[lo_idx]) + frac * float(values[hi_idx])
    return out


def _pack_threshold_dataset(dataset: BasisDataset) -> _PackedThresholdDataset:
    anc_det, anc_obs = split_factory_bits(
        dataset.detectors,
        dataset.observables,
        layout=DEFAULT_SYNDROME_LAYOUT,
    )
    return _PackedThresholdDataset(
        anc_det=anc_det,
        # NOTE: anc_obs is not used elsewhere in the code...
        packed_full_det=pack_boolean_array(dataset.detectors).astype(np.uint64),
        packed_anc_det=pack_boolean_array(anc_det).astype(np.uint64),
        packed_anc_obs=pack_boolean_array(anc_obs).astype(np.uint64),
        output_bits=np.asarray(dataset.observables[:, 0], dtype=np.uint64),
    )


def _decode_factory_patterns(
    decoder: DecoderAdapter,
    packed_anc_det: np.ndarray,
    *,
    syndrome_length: int,
    progress_update: Callable[[int], None] | None,
) -> dict[int, tuple[int, float]]:
    # TODO: can modify the arguments here; don't need the whole DecoderAdapter; can
    # just take in the decode_factory.
    unique_anc_det, shot_counts = np.unique(packed_anc_det, return_counts=True)
    decoded: dict[int, tuple[int, float]] = {}
    for packed, shot_count in zip(unique_anc_det.tolist(), shot_counts, strict=True):
        syndrome = unpack_packed_bits(int(packed), syndrome_length).astype(np.bool_)
        correction, score = decoder.decode_factory(syndrome)
        correction_bits = np.asarray(correction, dtype=np.uint8)
        decoded[int(packed)] = (
            int(pack_boolean_array(correction_bits)[0]) if len(correction_bits) else 0,
            float(score),
        )
        if progress_update is not None:
            progress_update(int(shot_count))
    return decoded


def _decode_full_patterns(
    decoder: DecoderAdapter,
    packed_full_det: np.ndarray,
    shot_counts: np.ndarray,
    *,
    syndrome_length: int,
    progress_update: Callable[[int], None] | None,
) -> dict[int, int]:
    # TODO: can modify the arguments here; don't need the whole DecoderAdapter; can
    # just take in the decode_full.
    decoded: dict[int, int] = {}
    for packed, shot_count in zip(
        np.asarray(packed_full_det, dtype=np.uint64).tolist(),
        np.asarray(shot_counts, dtype=np.int64),
        strict=True,
    ):
        syndrome = unpack_packed_bits(int(packed), syndrome_length).astype(np.bool_)
        correction = np.asarray(decoder.decode_full(syndrome), dtype=np.uint8)
        decoded[int(packed)] = int(correction[0]) if len(correction) else 0
        if progress_update is not None:
            progress_update(int(shot_count))
    return decoded


def _score_count_dict_to_arrays(
    score_to_counts: Mapping[float, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not score_to_counts:
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
        )
    scores = np.array(sorted(score_to_counts), dtype=np.float64)
    return (
        scores,
        np.array([int(score_to_counts[score][0]) for score in scores], dtype=np.int64),
        np.array([int(score_to_counts[score][1]) for score in scores], dtype=np.int64),
    )


def _build_generic_threshold_tables(
    actual_data: Mapping[str, BasisDataset],
    decoder_map: Mapping[str, DecoderAdapter],
    *,
    targets: np.ndarray,
    basis_labels: Sequence[str],
    progress_label: str | None = None,
) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    np.ndarray,
    np.ndarray,
    int,
]:
    """Decode, postselect, and compress shots into score-indexed counts."""

    # NOTE: in principle this is pretty generic (just depends on you having scores
    # for your counts), but because the fn's it depend on are application-level, i
    # think this should also be application-level.
    packed_targets = packed_pattern_targets(np.asarray(targets, dtype=np.uint8))
    per_basis_tables: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    global_score_weights: defaultdict[float, int] = defaultdict(int)
    total_shots = 0
    progress_bars = {}

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
            packed_dataset = _pack_threshold_dataset(dataset)
            progress_bar = progress_bars.get(basis)

            def update_progress(advance: int) -> None:
                if progress_bar is not None:
                    progress_bar.update(advance)

            factory_cache = _decode_factory_patterns(
                decoder_map[basis],
                packed_dataset.packed_anc_det,
                syndrome_length=packed_dataset.anc_det.shape[1],
                progress_update=update_progress,
            )
            unique_groups, group_counts = np.unique(
                np.column_stack(
                    [
                        packed_dataset.packed_anc_det,
                        packed_dataset.packed_anc_obs,
                        packed_dataset.packed_full_det,
                        packed_dataset.output_bits,
                    ]
                ),
                axis=0,
                return_counts=True,
            )

            accepted_groups: list[tuple[float, int, int, int]] = []
            full_decode_weights: defaultdict[int, int] = defaultdict(int)
            for row, count in zip(unique_groups, group_counts, strict=True):
                packed_a_det = int(row[0])
                packed_a_obs = int(row[1])
                packed_det = int(row[2])
                output_bit = int(row[3])

                anc_flip_packed, score = factory_cache[packed_a_det]
                if not np.isfinite(score):
                    continue
                if packed_a_obs ^ anc_flip_packed not in packed_targets:
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

            full_cache = _decode_full_patterns(
                decoder_map[basis],
                np.fromiter(full_decode_weights.keys(), dtype=np.uint64),
                np.fromiter(full_decode_weights.values(), dtype=np.int64),
                syndrome_length=dataset.detectors.shape[1],
                progress_update=update_progress,
            )

            score_to_counts: defaultdict[float, np.ndarray] = defaultdict(
                lambda: np.zeros(2, dtype=np.int64)
            )
            for score, packed_det, output_bit, count in accepted_groups:
                corrected_output_bit = output_bit ^ full_cache[packed_det]
                score_to_counts[score][corrected_output_bit] += count
                global_score_weights[score] += count

            if progress_bar is not None:
                progress_bar.set_description_str(f"{progress_label} {basis}: decoded")
                progress_bar.refresh()

            per_basis_tables[basis] = _score_count_dict_to_arrays(score_to_counts)
    finally:
        for progress_bar in progress_bars.values():
            progress_bar.close()

    global_scores = np.array(sorted(global_score_weights), dtype=np.float64)
    global_weights = np.array(
        [int(global_score_weights[score]) for score in global_scores],
        dtype=np.int64,
    )
    return per_basis_tables, global_scores, global_weights, total_shots


def _counts_for_threshold(
    scores: np.ndarray,
    zero_counts: np.ndarray,
    one_counts: np.ndarray,
    threshold: float,
) -> tuple[int, int]:
    idx = int(np.searchsorted(scores, threshold, side="left"))
    if idx >= len(scores):
        return 0, 0
    return int(np.sum(zero_counts[idx:])), int(np.sum(one_counts[idx:]))


def _evaluate_cached_threshold_curve(
    per_basis_tables: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    score_array: np.ndarray,
    *,
    score_weights: np.ndarray,
    threshold_points: int = 64,
    target_bloch: np.ndarray = DEFAULT_TARGET_BLOCH,
    basis_labels: Sequence[str] = DEFAULT_BASIS_LABELS,
    min_accepted_per_basis: int = 50,
    total_shots: int,
) -> dict[str, np.ndarray]:
    """Evaluate a point-fidelity curve from precomputed score/count tables."""

    if len(score_array) == 0:
        raise RuntimeError("No factory-accepted shots found for threshold sweep.")

    thresholds = np.unique(
        _weighted_quantiles_from_counts(
            score_array,
            score_weights,
            np.linspace(0.0, 1.0, threshold_points),
        )
    )

    accepted_fractions: list[float] = []
    fidelities: list[float] = []
    for threshold in thresholds:
        counts_by_basis = {
            basis: _counts_for_threshold(*per_basis_tables[basis], float(threshold))
            for basis in basis_labels
        }
        if (
            min(sum(counts_by_basis[basis]) for basis in basis_labels)
            < min_accepted_per_basis
        ):
            continue

        zero_counts = np.array(
            [counts_by_basis[basis][0] for basis in basis_labels],
            dtype=np.int64,
        )
        one_counts = np.array(
            [counts_by_basis[basis][1] for basis in basis_labels],
            dtype=np.int64,
        )
        point = TomographyResult(zero_counts, one_counts).fidelity_bloch(target_bloch)[
            "point"
        ]
        accepted_fractions.append(
            sum(sum(counts_by_basis[basis]) for basis in basis_labels) / total_shots
        )
        fidelities.append(point)

    order = np.argsort(accepted_fractions)
    accepted = np.asarray(accepted_fractions, dtype=np.float64)[order]
    fidelity = np.asarray(fidelities, dtype=np.float64)[order]
    return {
        "accepted_fraction": accepted,
        "fidelity": fidelity,
        "point_fidelity": fidelity.copy(),
    }


__all__ = [
    "DecoderAdapter",
    "_build_generic_threshold_tables",
    "_evaluate_cached_threshold_curve",
]
