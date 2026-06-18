from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Callable, Protocol, TypeVar

import numpy as np
from bloqade.decoders import BaseDecoder
from demo.msd_utils.domain.confidence import ConfidenceDecoder
from demo.msd_utils.standard.bit_packing import (
    pack_boolean_array,
    packed_pattern_targets,
    unpack_packed_bits,
)
from demo.msd_utils.standard.tomography import (
    DEFAULT_TARGET_BLOCH,
    TomographyResult,
)

from .constants import DEFAULT_BASIS_LABELS
from .layout import (
    DEFAULT_SYNDROME_LAYOUT,
    SyndromeLayout,
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
        decode_factory: Callable from factory syndrome bits to
            ``(factory_correction, confidence_score)``.
        decode_full: Callable from full syndrome bits to output correction.
    """

    # NOTE: full_decoder and factory_decoder fields might not be strictly needed; can maybe remove them (and core functionality is preserved.)
    full_decoder: BaseDecoder | None
    factory_decoder: ConfidenceDecoder | None
    decode_factory: Callable[[np.ndarray], tuple[np.ndarray, float]]
    decode_full: Callable[[np.ndarray], np.ndarray]


def _call_decoder_fn(
    fn: Callable[[np.ndarray], DecodeResult],
    bits: np.ndarray,
) -> DecodeResult:
    """Call a decoder function with boolean syndrome bits."""

    return fn(np.asarray(bits, dtype=np.bool_))


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
            packed_dataset = _pack_threshold_dataset(
                dataset,
                layout=DEFAULT_SYNDROME_LAYOUT,
            )
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
    threshold_points: int = 64,
    target_bloch: np.ndarray,
    basis_labels: Sequence[str],
    min_accepted_per_basis: int = 50,
    total_shots: int,
) -> dict[str, np.ndarray]:
    """Evaluate a fidelity curve from precomputed score/count tables."""

    if len(score_array) == 0:
        raise RuntimeError("No factory-accepted shots found for threshold sweep")

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

    accepted_fractions = []
    fidelities = []
    point_fidelities = []

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

        zero_counts = np.array(
            [counts_by_basis[basis][0] for basis in basis_labels],
            dtype=np.int64,
        )
        one_counts = np.array(
            [counts_by_basis[basis][1] for basis in basis_labels],
            dtype=np.int64,
        )
        summary = TomographyResult(
            zero_counts=zero_counts,
            one_counts=one_counts,
        ).fidelity_bloch(target_bloch)
        accepted_fractions.append(total_kept / total_shots)
        point_fidelities.append(summary["point"])
        fidelities.append(summary["point"])

    accepted_array = np.asarray(accepted_fractions, dtype=np.float64)
    fidelity_array = np.asarray(fidelities, dtype=np.float64)
    point_fidelity_array = np.asarray(point_fidelities, dtype=np.float64)

    if len(accepted_array) > 0:
        order = np.argsort(accepted_array)
        accepted_array = accepted_array[order]
        fidelity_array = fidelity_array[order]
        point_fidelity_array = point_fidelity_array[order]

    return {
        "accepted_fraction": accepted_array,
        "fidelity": fidelity_array,
        "point_fidelity": point_fidelity_array,
    }


def evaluate_curve(
    actual_data: Mapping[str, BasisDataset],
    decoder_map: Mapping[str, DecoderAdapter],
    *,
    threshold_points: int = 64,
    metric: str,
    valid_factory_targets: np.ndarray | Sequence[Sequence[int]] | Sequence[int],
    target_bloch: np.ndarray = DEFAULT_TARGET_BLOCH,
    basis_labels: Sequence[str] = DEFAULT_BASIS_LABELS,
    min_accepted_per_basis: int = 50,
    progress_label: str | None = None,
) -> dict[str, np.ndarray]:
    """Evaluate a quantile-threshold postselection curve."""

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
        progress_label=progress_label,
    )
    try:
        return _evaluate_cached_threshold_curve(
            per_basis_tables,
            score_array,
            score_weights=score_weights,
            threshold_points=threshold_points,
            target_bloch=target_bloch,
            basis_labels=basis_labels,
            min_accepted_per_basis=min_accepted_per_basis,
            total_shots=total_shots,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"No factory-accepted shots found for {metric} threshold sweep"
        ) from exc
