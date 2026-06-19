from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .bit_packing import pack_boolean_array, packed_pattern_targets
from .confidence import ConfidenceDecoder
from .constants import DEFAULT_BASIS_LABELS
from .layout import DEFAULT_SYNDROME_LAYOUT, split_factory_bits
from .sampling import BasisDataset
from .tomography import DEFAULT_TARGET_BLOCH, TomographyResult

DecoderPair = tuple[ConfidenceDecoder, Any]


@dataclass(frozen=True)
class DecodedPostselectionResult:
    """Decoded output observables and one factory confidence per accepted shot."""

    observables: np.ndarray
    confidence: np.ndarray

    def __post_init__(self) -> None:
        observables = np.asarray(self.observables, dtype=np.uint8)
        confidence = np.asarray(self.confidence, dtype=np.float64).reshape(-1)
        if observables.ndim != 2:
            raise ValueError("observables must have shape (shots, num_observables).")
        if confidence.shape != (observables.shape[0],):
            raise ValueError("confidence must have one entry per shot.")
        object.__setattr__(self, "observables", observables)
        object.__setattr__(self, "confidence", confidence)


def _pack_row(bits: np.ndarray) -> int:
    return int(pack_boolean_array(np.asarray(bits, dtype=np.uint8).reshape(1, -1))[0])


def _build_generic_threshold_tables(
    actual_data: Mapping[str, BasisDataset],
    decoder_map: Mapping[str, DecoderPair],
    *,
    targets: np.ndarray,
    basis_labels: Sequence[str],
    progress_label: str | None = None,
) -> dict[str, DecodedPostselectionResult]:
    """Decode, factory-postselect, and keep output bits with confidence.

    Each basis result stores decoded output observable shots with shape
    ``(shots, output_observables)`` and a separate confidence vector with shape
    ``(shots,)``.
    """

    packed_targets = packed_pattern_targets(np.asarray(targets, dtype=np.uint8))
    decoded_results: dict[str, DecodedPostselectionResult] = {}
    progress_bars = {}

    if progress_label is not None:
        from tqdm.auto import tqdm

        progress_bars = {
            basis: tqdm(
                total=len(actual_data[basis].observables),
                desc=f"{progress_label} {basis}: decoded",
                unit="shots",
                position=index,
                leave=True,
            )
            for index, basis in enumerate(basis_labels)
        }

    try:
        for basis in basis_labels:
            dataset = actual_data[basis]
            factory_decoder, full_decoder = decoder_map[basis]
            anc_det, anc_obs = split_factory_bits(
                dataset.detectors,
                dataset.observables,
                layout=DEFAULT_SYNDROME_LAYOUT,
            )
            factory_cache: dict[int, tuple[np.ndarray, float]] = {}
            full_cache: dict[int, np.ndarray] = {}
            observable_rows: list[np.ndarray] = []
            confidence_rows: list[float] = []
            progress_bar = progress_bars.get(basis)

            for shot_index in range(dataset.detectors.shape[0]):
                anc_detector_bits = anc_det[shot_index]
                packed_anc_det = _pack_row(anc_detector_bits)
                factory_result = factory_cache.get(packed_anc_det)
                if factory_result is None:
                    correction, confidence = factory_decoder.decode_with_confidence(
                        anc_detector_bits.astype(np.bool_)
                    )
                    factory_result = (
                        np.asarray(correction, dtype=np.uint8).reshape(-1),
                        float(np.float64(confidence)),
                    )
                    factory_cache[packed_anc_det] = factory_result

                anc_correction, confidence = factory_result
                if not np.isfinite(confidence):
                    if progress_bar is not None:
                        progress_bar.update(1)
                    continue

                corrected_factory = _pack_row(anc_obs[shot_index]) ^ _pack_row(
                    anc_correction
                )
                if corrected_factory not in packed_targets:
                    if progress_bar is not None:
                        progress_bar.update(1)
                    continue

                detector_bits = dataset.detectors[shot_index]
                packed_full_det = _pack_row(detector_bits)
                full_correction = full_cache.get(packed_full_det)
                if full_correction is None:
                    full_correction = np.asarray(
                        full_decoder.decode(detector_bits.astype(np.bool_)),
                        dtype=np.uint8,
                    ).reshape(-1)
                    full_cache[packed_full_det] = full_correction

                output_bits = np.asarray(
                    dataset.observables[shot_index, :1], dtype=np.uint8
                )
                correction_bits = np.asarray(
                    full_correction[: len(output_bits)], dtype=np.uint8
                )
                decoded_output = output_bits ^ correction_bits
                observable_rows.append(decoded_output.astype(np.uint8, copy=False))
                confidence_rows.append(confidence)
                if progress_bar is not None:
                    progress_bar.update(1)

            decoded_results[basis] = DecodedPostselectionResult(
                observables=(
                    np.stack(observable_rows, axis=0)
                    if observable_rows
                    else np.zeros((0, 1), dtype=np.uint8)
                ),
                confidence=np.asarray(confidence_rows, dtype=np.float64),
            )
    finally:
        for progress_bar in progress_bars.values():
            progress_bar.close()

    return decoded_results


def _shots_at_accepted_fraction(
    decoded_results: Mapping[str, DecodedPostselectionResult],
    accepted_fraction: float,
    *,
    basis_labels: Sequence[str] = DEFAULT_BASIS_LABELS,
) -> dict[str, np.ndarray]:
    """Return decoded observable shots after a global confidence top-k cut."""

    if not 0.0 <= accepted_fraction <= 1.0:
        raise ValueError("accepted_fraction must be between 0 and 1.")

    total_postselected = sum(
        int(decoded_results[basis].observables.shape[0]) for basis in basis_labels
    )
    empty = {
        basis: np.zeros(
            (0, decoded_results[basis].observables.shape[1]), dtype=np.uint8
        )
        for basis in basis_labels
    }
    if total_postselected == 0 or accepted_fraction == 0.0:
        return empty

    target_count = min(
        total_postselected,
        max(1, int(np.ceil(float(accepted_fraction) * total_postselected))),
    )
    confidences: list[np.ndarray] = []
    basis_ids: list[np.ndarray] = []
    shot_ids: list[np.ndarray] = []
    for basis_index, basis in enumerate(basis_labels):
        basis_results = decoded_results[basis]
        count = basis_results.observables.shape[0]
        confidences.append(basis_results.confidence)
        basis_ids.append(np.full(count, basis_index, dtype=np.int64))
        shot_ids.append(np.arange(count, dtype=np.int64))

    all_confidences = np.concatenate(confidences)
    all_basis_ids = np.concatenate(basis_ids)
    all_shot_ids = np.concatenate(shot_ids)
    order = np.argsort(-all_confidences, kind="stable")[:target_count]

    selected: dict[str, list[np.ndarray]] = {basis: [] for basis in basis_labels}
    for flat_index in order:
        basis = basis_labels[int(all_basis_ids[flat_index])]
        shot_index = int(all_shot_ids[flat_index])
        selected[basis].append(decoded_results[basis].observables[shot_index])

    return {
        basis: (
            np.asarray(selected[basis], dtype=np.uint8)
            if selected[basis]
            else np.zeros(
                (0, decoded_results[basis].observables.shape[1]), dtype=np.uint8
            )
        )
        for basis in basis_labels
    }


def _evaluate_cached_threshold_curve(
    decoded_results: Mapping[str, DecodedPostselectionResult],
    *,
    total_shots: int,
    threshold_points: int = 64,
    target_bloch: np.ndarray = DEFAULT_TARGET_BLOCH,
    basis_labels: Sequence[str] = DEFAULT_BASIS_LABELS,
    min_accepted_per_basis: int = 50,
) -> dict[str, np.ndarray]:
    """Evaluate a point-fidelity curve from decoded shots and confidences."""

    confidence_arrays = [
        decoded_results[basis].confidence
        for basis in basis_labels
        if decoded_results[basis].observables.shape[0] > 0
    ]
    if not confidence_arrays:
        raise RuntimeError("No factory-accepted shots found for threshold sweep.")
    confidences = np.sort(np.concatenate(confidence_arrays))
    threshold_indices = np.rint(
        np.linspace(0, len(confidences) - 1, threshold_points)
    ).astype(np.int64)
    thresholds = np.unique(confidences[threshold_indices])

    accepted_fractions: list[float] = []
    fidelities: list[float] = []
    for threshold in thresholds:
        shots_by_basis: dict[str, np.ndarray] = {}
        accepted_count = 0
        for basis in basis_labels:
            basis_results = decoded_results[basis]
            keep = basis_results.confidence >= float(threshold)
            shots = basis_results.observables[keep].astype(np.uint8, copy=False)
            shots_by_basis[basis] = shots
            accepted_count += int(shots.shape[0])

        if (
            min(shots_by_basis[basis].shape[0] for basis in basis_labels)
            < min_accepted_per_basis
        ):
            continue

        point = TomographyResult(shots_by_basis).fidelity_bloch(target_bloch)["point"]
        accepted_fractions.append(accepted_count / total_shots)
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
    "DecodedPostselectionResult",
    "_build_generic_threshold_tables",
    "_evaluate_cached_threshold_curve",
    "_shots_at_accepted_fraction",
]
