from __future__ import annotations

import inspect
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Mapping, Sequence, TypeAlias

import numpy as np
import stim
from beliefmatching import detector_error_model_to_check_matrices

from .common import DEFAULT_SYNDROME_LAYOUT, SyndromeLayout
from .core import (
    DEFAULT_BASIS_LABELS,
    DEFAULT_TARGET_BLOCH,
    BasisDataset,
    ancilla_matches_valid_targets,
    bits_to_key,
    fidelity_from_counts,
    magic_state_fidelity_point_from_counts,
    pack_boolean_array,
    packed_bits_to_int,
    resolve_valid_factory_targets,
    run_task,
    split_factory_bits,
    unpack_packed_bits,
)

SyndromeKey: TypeAlias = int | str


@dataclass(frozen=True)
class DecoderAdapter:
    full_decoder: Any
    factory_decoder: Any
    decode_factory: Callable[[Any], tuple[tuple[int, ...], float]]
    decode_full: Callable[[Any], tuple[int, ...]]
    factory_score_mode: str


@dataclass(frozen=True)
class TableDecoderWithConfidence:
    decoder: Any
    syndrome_confidence: np.ndarray
    confidence_score_mode: str = "mld_output_fidelity"

    def decode(self, detector_bits: np.ndarray) -> np.ndarray:
        return np.asarray(self.decoder.decode(detector_bits), dtype=np.bool_)

    def decode_with_confidence(
        self,
        detector_bits: np.ndarray,
    ) -> tuple[np.ndarray, np.float64]:
        if detector_bits.ndim != 1:
            raise ValueError(
                "decode_with_confidence expects a single detector shot (1D array)."
            )
        correction = self.decode(detector_bits)
        packed = int(pack_boolean_array(np.asarray(detector_bits, dtype=np.uint8))[0])
        score = (
            float(self.syndrome_confidence[packed])
            if packed < len(self.syndrome_confidence)
            else float("nan")
        )
        return correction, np.float64(score)


def make_layout_only_dem(
    num_detectors: int, num_observables: int
) -> stim.DetectorErrorModel:
    terms = []
    if num_detectors:
        terms.append(" ".join(f"D{i}" for i in range(num_detectors)))
    if num_observables:
        terms.append(" ".join(f"L{i}" for i in range(num_observables)))
    if not terms:
        raise ValueError("Need at least one detector or observable.")
    # NOTE: this DEM only carries detector/observable layout metadata.
    return stim.DetectorErrorModel("\n".join(f"error(0.5) {term}" for term in terms))


def make_shape_only_dem(
    num_detectors: int, num_observables: int
) -> stim.DetectorErrorModel:
    return make_layout_only_dem(num_detectors, num_observables)


def matrix_to_dem(
    check_matrix: np.ndarray,
    observables_matrix: np.ndarray,
    priors: np.ndarray,
) -> stim.DetectorErrorModel:
    lines = []
    for col, prior in enumerate(np.asarray(priors, dtype=np.float64)):
        det_targets = [f"D{i}" for i in np.flatnonzero(check_matrix[:, col])]
        obs_targets = [f"L{i}" for i in np.flatnonzero(observables_matrix[:, col])]
        if not det_targets and not obs_targets:
            continue
        safe_prior = float(np.clip(prior, 1e-12, 1.0 - 1e-12))
        lines.append(f"error({safe_prior:.16g}) " + " ".join(det_targets + obs_targets))
    if not lines:
        raise ValueError("Matrix reduction produced an empty DEM.")
    return stim.DetectorErrorModel("\n".join(lines))


def compute_dem_data(task: Any) -> dict[str, np.ndarray]:
    dem_matrix = detector_error_model_to_check_matrices(
        task.detector_error_model,
        allow_undecomposed_hyperedges=True,
    )
    return {
        "H": dem_matrix.check_matrix.toarray().astype(np.int64),
        "O": dem_matrix.observables_matrix.toarray().astype(np.int64),
        "priors": np.asarray(dem_matrix.priors, dtype=np.float64),
    }


def train_mld_decoder_pair(
    training_dataset: BasisDataset,
    *,
    table_decoder_cls: type,
    layout: SyndromeLayout = DEFAULT_SYNDROME_LAYOUT,
) -> tuple[Any, Any]:
    anc_det, anc_obs = split_factory_bits(
        training_dataset.detectors,
        training_dataset.observables,
        layout=layout,
    )

    full_decoder = table_decoder_cls.from_det_obs_shots(
        make_layout_only_dem(
            training_dataset.detectors.shape[1],
            training_dataset.observables.shape[1],
        ),
        np.concatenate(
            [training_dataset.detectors, training_dataset.observables], axis=1
        ).astype(bool),
    )
    factory_decoder = table_decoder_cls.from_det_obs_shots(
        make_layout_only_dem(anc_det.shape[1], anc_obs.shape[1]),
        np.concatenate([anc_det, anc_obs], axis=1).astype(bool),
    )
    return full_decoder, factory_decoder


def _make_decoder_adapter(
    *,
    full_decoder: Any,
    factory_decoder: Any,
    full_syndrome_length: int,
    factory_syndrome_length: int,
    factory_decode_impl: Callable[[np.ndarray], tuple[np.ndarray, float]],
    factory_score_mode: str,
) -> DecoderAdapter:
    def _normalize_syndrome_key(key: SyndromeKey, length: int) -> np.ndarray:
        if isinstance(key, str):
            bits = np.fromiter((1 if c == "1" else 0 for c in key), dtype=np.uint8)
            if len(bits) != length:
                raise ValueError(
                    f"Syndrome key has length {len(bits)} but expected {length} bits."
                )
            return bits.astype(bool)
        return unpack_packed_bits(int(key), length).astype(bool)

    @lru_cache(maxsize=None)
    def decode_factory(packed_syndrome: SyndromeKey):
        syndrome = _normalize_syndrome_key(packed_syndrome, factory_syndrome_length)
        correction, score = factory_decode_impl(syndrome)
        return tuple(
            int(x) for x in np.asarray(correction, dtype=np.uint8).tolist()
        ), float(score)

    @lru_cache(maxsize=None)
    def decode_full(packed_syndrome: SyndromeKey):
        syndrome = _normalize_syndrome_key(packed_syndrome, full_syndrome_length)
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
    fn: Callable[[SyndromeKey], Any],
    bits: np.ndarray,
) -> Any:
    packed = packed_bits_to_int(bits)
    try:
        return fn(packed)
    except TypeError:
        return fn(bits_to_key(bits))


def estimate_mld_ancilla_scores(
    decoder_by_basis: Mapping[str, tuple[Any, Any]],
    ranking_data_by_basis: Mapping[str, BasisDataset],
    *,
    factory_target: np.ndarray | Sequence[int] | None = None,
    valid_factory_targets: np.ndarray | Sequence[Sequence[int]] | None = None,
    basis_labels: Sequence[str] = DEFAULT_BASIS_LABELS,
    sign_vector: Sequence[float] = (1.0, -1.0, 1.0),
    target_bloch: np.ndarray = DEFAULT_TARGET_BLOCH,
    layout: SyndromeLayout = DEFAULT_SYNDROME_LAYOUT,
) -> np.ndarray:
    targets = resolve_valid_factory_targets(
        factory_target=factory_target,
        valid_factory_targets=valid_factory_targets,
    )
    if set(decoder_by_basis) != set(basis_labels):
        raise ValueError(
            "Need X/Y/Z decoder pairs to estimate shared MLD postselection scores."
        )
    if set(ranking_data_by_basis) != set(basis_labels):
        raise ValueError(
            "Need X/Y/Z ranking datasets to estimate shared MLD postselection scores."
        )

    corrected_by_pattern = {basis: defaultdict(list) for basis in basis_labels}
    ancilla_detectors: int | None = None

    for basis in basis_labels:
        full_decoder, factory_decoder = decoder_by_basis[basis]
        score_dataset = ranking_data_by_basis[basis]
        anc_det, anc_obs = split_factory_bits(
            score_dataset.detectors,
            score_dataset.observables,
            layout=layout,
        )
        if ancilla_detectors is None:
            ancilla_detectors = anc_det.shape[1]
        elif ancilla_detectors != anc_det.shape[1]:
            raise ValueError(
                "Inconsistent ancilla detector counts across MLD training datasets."
            )

        for det, obs, a_det, a_obs in zip(
            score_dataset.detectors,
            score_dataset.observables,
            anc_det,
            anc_obs,
            strict=True,
        ):
            anc_flip = np.asarray(
                factory_decoder.decode(a_det.astype(bool)), dtype=np.uint8
            )
            corrected_anc = a_obs ^ anc_flip
            if not ancilla_matches_valid_targets(corrected_anc, targets):
                continue
            full_flip = np.asarray(
                full_decoder.decode(det.astype(bool)), dtype=np.uint8
            )
            packed = int(pack_boolean_array(a_det)[0])
            corrected_by_pattern[basis][packed].append(int(obs[0] ^ full_flip[0]))

    assert ancilla_detectors is not None
    scores = np.full(1 << ancilla_detectors, np.nan, dtype=np.float64)
    all_patterns = set()
    for basis in basis_labels:
        all_patterns.update(corrected_by_pattern[basis].keys())

    for packed in all_patterns:
        basis_counts = {
            basis: np.asarray(
                corrected_by_pattern[basis].get(packed, ()), dtype=np.uint8
            )
            for basis in basis_labels
        }
        if min(len(basis_counts[basis]) for basis in basis_labels) == 0:
            continue
        scores[packed] = magic_state_fidelity_point_from_counts(
            basis_counts["X"],
            basis_counts["Y"],
            basis_counts["Z"],
            sign_vector=sign_vector,
            target_bloch=target_bloch,
        )
    return scores


def build_shared_mld_postselection_scores(
    training_data_by_basis: Mapping[str, BasisDataset],
    *,
    table_decoder_cls: type,
    factory_target: np.ndarray | Sequence[int] | None = None,
    valid_factory_targets: np.ndarray | Sequence[Sequence[int]] | None = None,
    ranking_data_by_basis: Mapping[str, BasisDataset] | None = None,
    basis_labels: Sequence[str] = DEFAULT_BASIS_LABELS,
    sign_vector: Sequence[float] = (1.0, -1.0, 1.0),
    target_bloch: np.ndarray = DEFAULT_TARGET_BLOCH,
    layout: SyndromeLayout = DEFAULT_SYNDROME_LAYOUT,
) -> np.ndarray:
    if set(training_data_by_basis) != set(basis_labels):
        raise ValueError(
            "Need X/Y/Z training datasets to build shared MLD postselection scores."
        )
    score_data_by_basis = (
        ranking_data_by_basis
        if ranking_data_by_basis is not None
        else training_data_by_basis
    )
    decoder_by_basis = {
        basis: train_mld_decoder_pair(
            training_data_by_basis[basis],
            table_decoder_cls=table_decoder_cls,
            layout=layout,
        )
        for basis in basis_labels
    }
    return estimate_mld_ancilla_scores(
        decoder_by_basis,
        score_data_by_basis,
        factory_target=factory_target,
        valid_factory_targets=valid_factory_targets,
        basis_labels=basis_labels,
        sign_vector=sign_vector,
        target_bloch=target_bloch,
        layout=layout,
    )


def build_mld_decoders(
    training_dataset: BasisDataset,
    ancilla_scores: np.ndarray,
    *,
    table_decoder_cls: type,
    layout: SyndromeLayout = DEFAULT_SYNDROME_LAYOUT,
) -> DecoderAdapter:
    anc_det, anc_obs = split_factory_bits(
        training_dataset.detectors,
        training_dataset.observables,
        layout=layout,
    )
    full_decoder, factory_decoder = train_mld_decoder_pair(
        training_dataset,
        table_decoder_cls=table_decoder_cls,
        layout=layout,
    )
    return build_mld_decoders_from_pair(
        full_decoder=full_decoder,
        factory_decoder=factory_decoder,
        full_syndrome_length=training_dataset.detectors.shape[1],
        factory_syndrome_length=anc_det.shape[1],
        ancilla_scores=ancilla_scores,
    )


def build_mld_decoders_from_pair(
    *,
    full_decoder: Any,
    factory_decoder: Any,
    full_syndrome_length: int,
    factory_syndrome_length: int,
    ancilla_scores: np.ndarray,
) -> DecoderAdapter:
    if len(ancilla_scores) != (1 << factory_syndrome_length):
        raise ValueError(
            "Ancilla score table has the wrong size for this decoder pair."
        )

    wrapped_factory_decoder = TableDecoderWithConfidence(
        decoder=factory_decoder,
        syndrome_confidence=np.asarray(ancilla_scores, dtype=np.float64),
    )

    def factory_decode_impl(syndrome: np.ndarray) -> tuple[np.ndarray, float]:
        correction, score = wrapped_factory_decoder.decode_with_confidence(
            syndrome.astype(bool)
        )
        return np.asarray(correction, dtype=np.uint8), float(score)

    return _make_decoder_adapter(
        full_decoder=full_decoder,
        factory_decoder=wrapped_factory_decoder,
        full_syndrome_length=full_syndrome_length,
        factory_syndrome_length=factory_syndrome_length,
        factory_decode_impl=factory_decode_impl,
        factory_score_mode=wrapped_factory_decoder.confidence_score_mode,
    )


class MLEFactoryScorer:
    def __init__(self, decoder: Any, *, score_mode: str):
        self.decoder = decoder
        self.score_mode = score_mode
        self.decode_signature = inspect.signature(decoder.decode)

    def decode(self, syndrome: np.ndarray) -> tuple[np.ndarray, float, str]:
        return _score_from_decoder(
            self.decoder,
            syndrome,
            score_mode=self.score_mode,
        )


def _score_from_decoder(
    decoder: Any,
    syndrome: np.ndarray,
    *,
    score_mode: str,
) -> tuple[np.ndarray, float, str]:
    if hasattr(decoder, "decode_with_confidence"):
        correction, confidence = decoder.decode_with_confidence(syndrome.astype(bool))
        score_kind = getattr(decoder, "confidence_score_mode", "confidence")
        return (
            np.asarray(correction, dtype=np.uint8),
            float(np.float64(confidence)),
            str(score_kind),
        )

    decode_signature = inspect.signature(decoder.decode)

    if "return_logical_gap" in decode_signature.parameters:
        result = decoder.decode(syndrome, return_logical_gap=True)
        correction, logical_gap = result[:2]
        logical_gap = np.asarray(logical_gap, dtype=np.float64)
        return (
            np.asarray(correction, dtype=np.uint8),
            float(logical_gap[0]),
            "logical_gap",
        )

    if hasattr(decoder, "decode_with_logical_gap"):
        correction, logical_gap = decoder.decode_with_logical_gap(syndrome)
        logical_gap = np.asarray(logical_gap, dtype=np.float64)
        return (
            np.asarray(correction, dtype=np.uint8),
            float(logical_gap[0]),
            "logical_gap",
        )

    if score_mode == "logical_gap":
        raise RuntimeError(
            "Requested logical-gap MLE scoring, but this GurobiDecoder does not expose logical-gap support."
        )

    if "return_weights" in decode_signature.parameters:
        correction, weights = decoder.decode(syndrome, return_weights=True)
        weights = np.asarray(weights, dtype=np.float64)
        return np.asarray(correction, dtype=np.uint8), float(weights[0]), "weight"

    correction = np.asarray(decoder.decode(syndrome), dtype=np.uint8)
    return correction, float("nan"), "none"


def build_mle_decoders(
    task: Any,
    *,
    gurobi_decoder_cls: type,
    score_mode: str = "best_available",
    layout: SyndromeLayout = DEFAULT_SYNDROME_LAYOUT,
) -> DecoderAdapter:
    if score_mode not in {"best_available", "logical_gap"}:
        raise ValueError("score_mode must be 'best_available' or 'logical_gap'.")

    dem_data = compute_dem_data(task)
    full_dem = matrix_to_dem(dem_data["H"], dem_data["O"], dem_data["priors"])
    factory_dem = matrix_to_dem(
        dem_data["H"][layout.output_detector_count :, :],
        dem_data["O"][layout.output_observable_count :, :],
        dem_data["priors"],
    )

    full_decoder = gurobi_decoder_cls(full_dem)
    factory_decoder = gurobi_decoder_cls(factory_dem)
    scorer = MLEFactoryScorer(factory_decoder, score_mode=score_mode)
    resolved_mode: str = score_mode

    def factory_decode_impl(syndrome: np.ndarray) -> tuple[np.ndarray, float]:
        nonlocal resolved_mode
        correction, score, used_mode = scorer.decode(syndrome)
        resolved_mode = used_mode
        return correction, score

    adapter = _make_decoder_adapter(
        full_decoder=full_decoder,
        factory_decoder=factory_decoder,
        full_syndrome_length=full_dem.num_detectors,
        factory_syndrome_length=factory_dem.num_detectors,
        factory_decode_impl=factory_decode_impl,
        factory_score_mode=resolved_mode,
    )
    sample_syndrome = np.zeros(factory_dem.num_detectors, dtype=np.uint8)
    adapter.decode_factory(int(pack_boolean_array(sample_syndrome)[0]))
    return DecoderAdapter(
        full_decoder=adapter.full_decoder,
        factory_decoder=adapter.factory_decoder,
        decode_factory=adapter.decode_factory,
        decode_full=adapter.decode_full,
        factory_score_mode=resolved_mode,
    )


def evaluate_curve(
    actual_data: Mapping[str, BasisDataset],
    decoder_map: Mapping[str, DecoderAdapter],
    *,
    posterior_samples: int,
    threshold_points: int,
    metric: str,
    factory_target: np.ndarray | Sequence[int] | None = None,
    valid_factory_targets: np.ndarray | Sequence[Sequence[int]] | None = None,
    sign_vector: Sequence[float],
    target_bloch: np.ndarray = DEFAULT_TARGET_BLOCH,
    basis_labels: Sequence[str] = DEFAULT_BASIS_LABELS,
    min_accepted_per_basis: int = 50,
    threshold_policy: str = "quantile",
    selection_mode: str = "threshold",
    layout: SyndromeLayout = DEFAULT_SYNDROME_LAYOUT,
    uncertainty_backend: str = "wilson",
) -> dict[str, np.ndarray]:
    if selection_mode == "pattern_rank":
        return evaluate_mld_curve(
            actual_data,
            decoder_map,
            posterior_samples=posterior_samples,
            factory_target=factory_target,
            valid_factory_targets=valid_factory_targets,
            sign_vector=sign_vector,
            target_bloch=target_bloch,
            basis_labels=basis_labels,
            min_accepted_per_basis=min_accepted_per_basis,
            layout=layout,
            uncertainty_backend=uncertainty_backend,
        )
    if selection_mode != "threshold":
        raise ValueError("selection_mode must be 'threshold' or 'pattern_rank'.")

    targets = resolve_valid_factory_targets(
        factory_target=factory_target,
        valid_factory_targets=valid_factory_targets,
    )
    all_scores = []
    for basis in basis_labels:
        dataset = actual_data[basis]
        anc_det, anc_obs = split_factory_bits(
            dataset.detectors,
            dataset.observables,
            layout=layout,
        )
        decode_factory = decoder_map[basis].decode_factory
        for a_det, a_obs in zip(anc_det, anc_obs, strict=True):
            anc_flip, score = _call_decoder_fn(decode_factory, a_det)
            anc_flip = np.asarray(anc_flip, dtype=np.uint8)
            if ancilla_matches_valid_targets(
                a_obs ^ anc_flip,
                targets,
            ) and np.isfinite(score):
                all_scores.append(score)
    if not all_scores:
        raise RuntimeError(
            f"No factory-accepted shots found for {metric} threshold sweep"
        )

    score_array = np.asarray(all_scores, dtype=np.float64)
    if threshold_policy == "quantile":
        thresholds = np.unique(
            np.quantile(score_array, np.linspace(0.0, 1.0, threshold_points))
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
    credibility = []

    for threshold in thresholds:
        corrected: dict[str, np.ndarray] = {}
        total_kept = 0
        total_shots = 0
        for basis in basis_labels:
            dataset = actual_data[basis]
            anc_det, anc_obs = split_factory_bits(
                dataset.detectors,
                dataset.observables,
                layout=layout,
            )
            decode_factory = decoder_map[basis].decode_factory
            decode_full = decoder_map[basis].decode_full
            corrected_bits = []
            for det, obs, a_det, a_obs in zip(
                dataset.detectors,
                dataset.observables,
                anc_det,
                anc_obs,
                strict=True,
            ):
                anc_flip, score = _call_decoder_fn(decode_factory, a_det)
                anc_flip = np.asarray(anc_flip, dtype=np.uint8)
                corrected_anc = a_obs ^ anc_flip
                if not ancilla_matches_valid_targets(corrected_anc, targets):
                    continue
                if score < threshold:
                    continue
                full_flip = np.asarray(
                    _call_decoder_fn(decode_full, det), dtype=np.uint8
                )
                corrected_bits.append(int(obs[0] ^ full_flip[0]))
            corrected[basis] = np.asarray(corrected_bits, dtype=np.uint8)
            total_kept += len(corrected[basis])
            total_shots += len(dataset.observables)

        if (
            min(len(corrected[basis]) for basis in basis_labels)
            < min_accepted_per_basis
        ):
            continue

        summary = fidelity_from_counts(
            corrected["X"],
            corrected["Y"],
            corrected["Z"],
            posterior_samples,
            sign_vector=sign_vector,
            target_bloch=target_bloch,
            uncertainty_backend=uncertainty_backend,
        )
        accepted_fractions.append(total_kept / total_shots)
        fidelities.append(summary["median"])
        credibility.append((summary["low"], summary["high"]))

    accepted_array = np.asarray(accepted_fractions, dtype=np.float64)
    fidelity_array = np.asarray(fidelities, dtype=np.float64)
    credible_array = np.asarray(credibility, dtype=np.float64)

    if len(accepted_array) > 0:
        order = np.argsort(accepted_array)
        accepted_array = accepted_array[order]
        fidelity_array = fidelity_array[order]
        credible_array = credible_array[order]

    return {
        "accepted_fraction": accepted_array,
        "fidelity": fidelity_array,
        "credible": credible_array,
    }


def evaluate_mld_curve(
    actual_data: Mapping[str, BasisDataset],
    decoder_map: Mapping[str, DecoderAdapter],
    *,
    posterior_samples: int,
    factory_target: np.ndarray | Sequence[int] | None = None,
    valid_factory_targets: np.ndarray | Sequence[Sequence[int]] | None = None,
    sign_vector: Sequence[float],
    target_bloch: np.ndarray = DEFAULT_TARGET_BLOCH,
    basis_labels: Sequence[str] = DEFAULT_BASIS_LABELS,
    min_accepted_per_basis: int = 50,
    layout: SyndromeLayout = DEFAULT_SYNDROME_LAYOUT,
    uncertainty_backend: str = "wilson",
) -> dict[str, np.ndarray]:
    targets = resolve_valid_factory_targets(
        factory_target=factory_target,
        valid_factory_targets=valid_factory_targets,
    )
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
            if not ancilla_matches_valid_targets(a_obs ^ anc_flip, targets):
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
            posterior_samples,
            sign_vector=sign_vector,
            target_bloch=target_bloch,
            uncertainty_backend=uncertainty_backend,
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


def injected_baseline(
    task_map: Mapping[str, Any],
    *,
    eval_shots: int,
    posterior_samples: int,
    table_decoder_cls: type,
    sign_vector: Sequence[float],
    target_bloch: np.ndarray = DEFAULT_TARGET_BLOCH,
    raw: bool = False,
    training_task_map: Mapping[str, Any] | None = None,
    basis_labels: Sequence[str] = DEFAULT_BASIS_LABELS,
    uncertainty_backend: str = "wilson",
) -> dict[str, Any]:
    corrected = {}
    for basis in basis_labels:
        evaluation_dataset = run_task(task_map[basis], eval_shots, with_noise=True)
        if raw:
            corrected[basis] = evaluation_dataset.observables[:, 0].astype(np.uint8)
            continue

        training_dataset = evaluation_dataset
        if training_task_map is not None:
            training_dataset = run_task(
                training_task_map[basis], eval_shots, with_noise=True
            )

        decoder = table_decoder_cls.from_det_obs_shots(
            make_shape_only_dem(
                training_dataset.detectors.shape[1],
                training_dataset.observables.shape[1],
            ),
            np.concatenate(
                [training_dataset.detectors, training_dataset.observables], axis=1
            ).astype(bool),
        )
        bits = []
        for det, obs in zip(
            evaluation_dataset.detectors,
            evaluation_dataset.observables,
            strict=True,
        ):
            flip = np.asarray(decoder.decode(det.astype(bool)), dtype=np.uint8)
            bits.append(int(obs[0] ^ flip[0]))
        corrected[basis] = np.asarray(bits, dtype=np.uint8)

    return fidelity_from_counts(
        corrected["X"],
        corrected["Y"],
        corrected["Z"],
        posterior_samples,
        sign_vector=sign_vector,
        target_bloch=target_bloch,
        uncertainty_backend=uncertainty_backend,
    )
