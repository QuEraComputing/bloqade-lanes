from __future__ import annotations

import inspect
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import stim
from beliefmatching import detector_error_model_to_check_matrices

from .core import (
    DEFAULT_BASIS_LABELS,
    DEFAULT_TARGET_BLOCH,
    BasisDataset,
    bits_to_key,
    fidelity_from_counts,
    key_to_bits,
    magic_state_fidelity_point_from_counts,
    pack_boolean_array,
    run_task,
    split_factory_bits,
)


@dataclass(frozen=True)
class DecoderAdapter:
    full_decoder: Any
    factory_decoder: Any
    decode_factory: Callable[[str], tuple[tuple[int, ...], float]]
    decode_full: Callable[[str], tuple[int, ...]]
    factory_score_mode: str


def _infer_checks_per_logical(num_detectors: int, num_observables: int) -> int:
    if num_observables <= 0 or num_detectors % num_observables != 0:
        raise ValueError(
            "Unable to infer checks_per_logical from detector/observable counts."
        )
    return num_detectors // num_observables


def _factory_detector_indices(
    num_detectors: int,
    num_observables: int,
    *,
    output_logical: int,
    checks_per_logical: int | None = None,
) -> np.ndarray:
    checks = checks_per_logical or _infer_checks_per_logical(
        num_detectors, num_observables
    )
    start = output_logical * checks
    stop = start + checks
    return np.concatenate([np.arange(start), np.arange(stop, num_detectors)])


def _factory_observable_indices(
    num_observables: int,
    *,
    output_logical: int,
) -> np.ndarray:
    return np.concatenate(
        [np.arange(output_logical), np.arange(output_logical + 1, num_observables)]
    )


def make_shape_only_dem(
    num_detectors: int, num_observables: int
) -> stim.DetectorErrorModel:
    terms = []
    if num_detectors:
        terms.append(" ".join(f"D{i}" for i in range(num_detectors)))
    if num_observables:
        terms.append(" ".join(f"L{i}" for i in range(num_observables)))
    if not terms:
        raise ValueError("Need at least one detector or observable.")
    return stim.DetectorErrorModel("\n".join(f"error(0.5) {term}" for term in terms))


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


def dem_target_map(
    target: stim.DemTarget,
    detector_indices: Sequence[int],
    observable_indices: Sequence[int],
) -> stim.DemTarget | None:
    if stim.DemTarget.is_relative_detector_id(target):
        det_idx = bisect_left(detector_indices, target.val)
        if det_idx == len(detector_indices) or detector_indices[det_idx] != target.val:
            return None
        return stim.target_relative_detector_id(det_idx)
    if stim.DemTarget.is_logical_observable_id(target):
        obs_idx = bisect_left(observable_indices, target.val)
        if (
            obs_idx == len(observable_indices)
            or observable_indices[obs_idx] != target.val
        ):
            return None
        return stim.target_logical_observable_id(obs_idx)
    return target


def sub_detector_error_model(
    dem: stim.DetectorErrorModel,
    detector_indices: Sequence[int],
    observable_indices: Sequence[int],
) -> stim.DetectorErrorModel:
    sorted_detectors = sorted(int(value) for value in detector_indices)
    sorted_observables = sorted(int(value) for value in observable_indices)
    reduced_dem = stim.DetectorErrorModel()
    error_probabilities: dict[tuple[stim.DemTarget, ...], float] = {}

    for instruction in dem:
        targets = instruction.targets_copy()
        mapped_targets = tuple(
            mapped
            for target in targets
            if (mapped := dem_target_map(target, sorted_detectors, sorted_observables))
            is not None
        )
        if not mapped_targets:
            continue
        probability = float(instruction.args_copy()[0])
        if mapped_targets in error_probabilities:
            previous = error_probabilities[mapped_targets]
            error_probabilities[mapped_targets] = (1.0 - previous) * probability + (
                previous * (1.0 - probability)
            )
        else:
            error_probabilities[mapped_targets] = probability

    for mapped_targets, probability in error_probabilities.items():
        reduced_dem.append(
            "error", parens_arguments=probability, targets=list(mapped_targets)
        )
    return reduced_dem


def partial_detector_observable_counts(
    det_obs_counts: np.ndarray,
    num_detectors: int,
    num_observables: int,
    detector_indices: Sequence[int],
    observable_indices: Sequence[int],
) -> np.ndarray:
    reshaped = det_obs_counts.reshape([2] * (num_detectors + num_observables))
    detector_indices = np.asarray(detector_indices, dtype=int)
    observable_indices = np.asarray(observable_indices, dtype=int)
    kept_indices = np.concatenate(
        [detector_indices, observable_indices + num_detectors]
    )
    reduced_indices = [
        index
        for index in range(num_detectors + num_observables)
        if index not in kept_indices
    ]
    reversed_reduced_indices = (num_detectors + num_observables - 1) - np.asarray(
        reduced_indices, dtype=int
    )[::-1]
    partial = np.sum(reshaped, axis=tuple(reversed_reduced_indices))
    return partial.reshape(-1)


def build_partial_table_decoder(
    full_decoder: Any,
    *,
    detector_indices: Sequence[int],
    observable_indices: Sequence[int],
    table_decoder_cls: type,
) -> Any:
    partial_counts = partial_detector_observable_counts(
        full_decoder._det_obs_counts,
        full_decoder.num_detectors,
        full_decoder.num_observables,
        detector_indices,
        observable_indices,
    )
    partial_dem = sub_detector_error_model(
        full_decoder._dem,
        detector_indices,
        observable_indices,
    )
    return table_decoder_cls(partial_dem, partial_counts)


def build_shared_mld_postselection_scores(
    training_data_by_basis: Mapping[str, BasisDataset],
    *,
    table_decoder_cls: type,
    factory_target: np.ndarray,
    basis_labels: Sequence[str] = DEFAULT_BASIS_LABELS,
    sign_vector: Sequence[float] = (1.0, -1.0, 1.0),
    target_bloch: np.ndarray = DEFAULT_TARGET_BLOCH,
) -> np.ndarray:
    if set(training_data_by_basis) != set(basis_labels):
        raise ValueError(
            "Need X/Y/Z training datasets to build shared MLD postselection scores."
        )

    corrected_by_pattern = {basis: defaultdict(list) for basis in basis_labels}
    ancilla_detectors: int | None = None

    for basis in basis_labels:
        training_dataset = training_data_by_basis[basis]
        anc_det, anc_obs = split_factory_bits(
            training_dataset.detectors, training_dataset.observables
        )
        if ancilla_detectors is None:
            ancilla_detectors = anc_det.shape[1]
        elif ancilla_detectors != anc_det.shape[1]:
            raise ValueError(
                "Inconsistent ancilla detector counts across MLD training datasets."
            )

        full_decoder = table_decoder_cls.from_det_obs_shots(
            make_shape_only_dem(
                training_dataset.detectors.shape[1],
                training_dataset.observables.shape[1],
            ),
            np.concatenate(
                [training_dataset.detectors, training_dataset.observables], axis=1
            ).astype(bool),
        )
        factory_decoder = table_decoder_cls.from_det_obs_shots(
            make_shape_only_dem(anc_det.shape[1], anc_obs.shape[1]),
            np.concatenate([anc_det, anc_obs], axis=1).astype(bool),
        )

        for det, obs, a_det, a_obs in zip(
            training_dataset.detectors,
            training_dataset.observables,
            anc_det,
            anc_obs,
            strict=True,
        ):
            anc_flip = np.asarray(
                factory_decoder.decode(a_det.astype(bool)), dtype=np.uint8
            )
            corrected_anc = a_obs ^ anc_flip
            if not np.array_equal(corrected_anc, factory_target):
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


def build_mld_decoders(
    training_dataset: BasisDataset,
    ancilla_scores: np.ndarray,
    *,
    table_decoder_cls: type,
    output_logical: int = 0,
    checks_per_logical: int | None = None,
) -> DecoderAdapter:
    full_decoder = table_decoder_cls.from_det_obs_shots(
        make_shape_only_dem(
            training_dataset.detectors.shape[1], training_dataset.observables.shape[1]
        ),
        np.concatenate(
            [training_dataset.detectors, training_dataset.observables], axis=1
        ).astype(bool),
    )
    factory_decoder = build_partial_table_decoder(
        full_decoder,
        detector_indices=_factory_detector_indices(
            training_dataset.detectors.shape[1],
            training_dataset.observables.shape[1],
            output_logical=output_logical,
            checks_per_logical=checks_per_logical,
        ),
        observable_indices=_factory_observable_indices(
            training_dataset.observables.shape[1],
            output_logical=output_logical,
        ),
        table_decoder_cls=table_decoder_cls,
    )
    anc_det, _anc_obs = split_factory_bits(
        training_dataset.detectors,
        training_dataset.observables,
        output_logical=output_logical,
        checks_per_logical=checks_per_logical,
    )
    if len(ancilla_scores) != (1 << anc_det.shape[1]):
        raise ValueError(
            "Ancilla score table has the wrong size for this training dataset."
        )

    @lru_cache(maxsize=None)
    def decode_factory(syndrome_key: str):
        syndrome = key_to_bits(syndrome_key).astype(bool)
        correction = np.asarray(factory_decoder.decode(syndrome), dtype=np.uint8)
        packed = int(pack_boolean_array(syndrome)[0])
        score = (
            float(ancilla_scores[packed])
            if packed < len(ancilla_scores)
            else float("nan")
        )
        return tuple(int(x) for x in correction.tolist()), score

    @lru_cache(maxsize=None)
    def decode_full(syndrome_key: str):
        syndrome = key_to_bits(syndrome_key).astype(bool)
        correction = np.asarray(full_decoder.decode(syndrome), dtype=np.uint8)
        return tuple(int(x) for x in correction.tolist())

    return DecoderAdapter(
        full_decoder=full_decoder,
        factory_decoder=factory_decoder,
        decode_factory=decode_factory,
        decode_full=decode_full,
        factory_score_mode="mld_output_fidelity",
    )


def _score_from_decoder(
    decoder: Any,
    syndrome: np.ndarray,
    *,
    score_mode: str,
) -> tuple[np.ndarray, float, str]:
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
    output_logical: int = 0,
    checks_per_logical: int | None = None,
) -> DecoderAdapter:
    if score_mode not in {"best_available", "logical_gap"}:
        raise ValueError("score_mode must be 'best_available' or 'logical_gap'.")

    dem_data = compute_dem_data(task)
    full_dem = matrix_to_dem(dem_data["H"], dem_data["O"], dem_data["priors"])
    factory_detector_indices = _factory_detector_indices(
        dem_data["H"].shape[0],
        dem_data["O"].shape[0],
        output_logical=output_logical,
        checks_per_logical=checks_per_logical,
    )
    factory_observable_indices = _factory_observable_indices(
        dem_data["O"].shape[0],
        output_logical=output_logical,
    )
    factory_dem = matrix_to_dem(
        dem_data["H"][factory_detector_indices, :],
        dem_data["O"][factory_observable_indices, :],
        dem_data["priors"],
    )

    full_decoder = gurobi_decoder_cls(full_dem)
    factory_decoder = gurobi_decoder_cls(factory_dem)
    resolved_mode: str | None = None

    @lru_cache(maxsize=None)
    def decode_factory(syndrome_key: str):
        nonlocal resolved_mode
        syndrome = key_to_bits(syndrome_key).astype(bool)
        correction, score, used_mode = _score_from_decoder(
            factory_decoder,
            syndrome,
            score_mode=score_mode,
        )
        resolved_mode = used_mode
        return tuple(int(x) for x in correction.tolist()), score

    @lru_cache(maxsize=None)
    def decode_full(syndrome_key: str):
        syndrome = key_to_bits(syndrome_key).astype(bool)
        correction = np.asarray(full_decoder.decode(syndrome), dtype=np.uint8)
        return tuple(int(x) for x in correction.tolist())

    # Prime the adapter so downstream code can inspect the effective score mode
    sample_syndrome = np.zeros(factory_dem.num_detectors, dtype=np.uint8)
    decode_factory(bits_to_key(sample_syndrome))

    return DecoderAdapter(
        full_decoder=full_decoder,
        factory_decoder=factory_decoder,
        decode_factory=decode_factory,
        decode_full=decode_full,
        factory_score_mode=resolved_mode or score_mode,
    )


def evaluate_curve(
    actual_data: Mapping[str, BasisDataset],
    decoder_map: Mapping[str, DecoderAdapter],
    *,
    posterior_samples: int,
    threshold_points: int,
    metric: str,
    factory_target: np.ndarray,
    sign_vector: Sequence[float],
    target_bloch: np.ndarray = DEFAULT_TARGET_BLOCH,
    basis_labels: Sequence[str] = DEFAULT_BASIS_LABELS,
    min_accepted_per_basis: int = 50,
    output_logical: int = 0,
    checks_per_logical: int | None = None,
) -> dict[str, np.ndarray]:
    all_scores = []
    for basis in basis_labels:
        dataset = actual_data[basis]
        anc_det, anc_obs = split_factory_bits(
            dataset.detectors,
            dataset.observables,
            output_logical=output_logical,
            checks_per_logical=checks_per_logical,
        )
        decode_factory = decoder_map[basis].decode_factory
        for a_det, a_obs in zip(anc_det, anc_obs, strict=True):
            anc_flip, score = decode_factory(bits_to_key(a_det))
            anc_flip = np.asarray(anc_flip, dtype=np.uint8)
            if np.array_equal(a_obs ^ anc_flip, factory_target) and np.isfinite(score):
                all_scores.append(score)
    if not all_scores:
        raise RuntimeError(
            f"No factory-accepted shots found for {metric} threshold sweep"
        )

    thresholds = np.unique(
        np.quantile(np.asarray(all_scores), np.linspace(0.0, 1.0, threshold_points))
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
                output_logical=output_logical,
                checks_per_logical=checks_per_logical,
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
                anc_flip, score = decode_factory(bits_to_key(a_det))
                anc_flip = np.asarray(anc_flip, dtype=np.uint8)
                corrected_anc = a_obs ^ anc_flip
                if not np.array_equal(corrected_anc, factory_target):
                    continue
                if score < threshold:
                    continue
                full_flip = np.asarray(decode_full(bits_to_key(det)), dtype=np.uint8)
                corrected_bits.append(
                    int(obs[output_logical] ^ full_flip[output_logical])
                )
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


def _collect_mld_pattern_statistics(
    data_by_basis: Mapping[str, BasisDataset],
    decoder_map: Mapping[str, DecoderAdapter],
    *,
    factory_target: np.ndarray,
    basis_labels: Sequence[str],
    output_logical: int,
    checks_per_logical: int | None,
) -> tuple[dict[str, dict[int, int]], dict[str, dict[int, list[int]]], int]:
    pattern_counts_by_basis: dict[str, dict[int, int]] = {}
    corrected_bits_by_basis: dict[str, dict[int, list[int]]] = {}
    total_shots = 0

    for basis in basis_labels:
        dataset = data_by_basis[basis]
        total_shots += len(dataset.observables)
        anc_det, anc_obs = split_factory_bits(
            dataset.detectors,
            dataset.observables,
            output_logical=output_logical,
            checks_per_logical=checks_per_logical,
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

            anc_flip, _score = decode_factory(bits_to_key(a_det))
            anc_flip = np.asarray(anc_flip, dtype=np.uint8)
            if not np.array_equal(a_obs ^ anc_flip, factory_target):
                continue

            full_flip = np.asarray(decode_full(bits_to_key(det)), dtype=np.uint8)
            corrected_bits[packed].append(
                int(obs[output_logical] ^ full_flip[output_logical])
            )

        pattern_counts_by_basis[basis] = dict(pattern_counts)
        corrected_bits_by_basis[basis] = corrected_bits

    return pattern_counts_by_basis, corrected_bits_by_basis, total_shots


def evaluate_mld_curve(
    actual_data: Mapping[str, BasisDataset],
    decoder_map: Mapping[str, DecoderAdapter],
    *,
    posterior_samples: int,
    factory_target: np.ndarray,
    sign_vector: Sequence[float],
    target_bloch: np.ndarray = DEFAULT_TARGET_BLOCH,
    basis_labels: Sequence[str] = DEFAULT_BASIS_LABELS,
    min_accepted_per_basis: int = 50,
    ranking_data: Mapping[str, BasisDataset] | None = None,
    output_logical: int = 0,
    checks_per_logical: int | None = None,
) -> dict[str, np.ndarray]:
    ranking_source = ranking_data or actual_data
    (
        ranking_pattern_counts,
        ranking_corrected_bits,
        _ranking_total_shots,
    ) = _collect_mld_pattern_statistics(
        ranking_source,
        decoder_map,
        factory_target=factory_target,
        basis_labels=basis_labels,
        output_logical=output_logical,
        checks_per_logical=checks_per_logical,
    )
    pattern_counts_by_basis, corrected_bits_by_basis, total_shots = (
        _collect_mld_pattern_statistics(
            actual_data,
            decoder_map,
            factory_target=factory_target,
            basis_labels=basis_labels,
            output_logical=output_logical,
            checks_per_logical=checks_per_logical,
        )
    )

    all_patterns = sorted(
        set().union(*(ranking_pattern_counts[basis].keys() for basis in basis_labels))
    )
    ranked_patterns = []
    for pattern in all_patterns:
        basis_counts = {
            basis: np.asarray(
                ranking_corrected_bits[basis].get(pattern, ()), dtype=np.uint8
            )
            for basis in basis_labels
        }
        if min(len(basis_counts[basis]) for basis in basis_labels) == 0:
            continue
        mean_score = magic_state_fidelity_point_from_counts(
            basis_counts["X"],
            basis_counts["Y"],
            basis_counts["Z"],
            sign_vector=sign_vector,
            target_bloch=target_bloch,
        )
        total_count = sum(
            ranking_pattern_counts[basis].get(pattern, 0) for basis in basis_labels
        )
        ranked_patterns.append((pattern, mean_score, total_count))

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
    basis_labels: Sequence[str] = DEFAULT_BASIS_LABELS,
    training_data_by_basis: Mapping[str, BasisDataset] | None = None,
    eval_data_by_basis: Mapping[str, BasisDataset] | None = None,
    run_detectors: bool = False,
) -> dict[str, Any]:
    corrected = {}
    for basis in basis_labels:
        dataset = (
            eval_data_by_basis[basis]
            if eval_data_by_basis is not None
            else run_task(
                task_map[basis],
                eval_shots,
                with_noise=True,
                run_detectors=run_detectors,
            )
        )
        if raw:
            corrected[basis] = dataset.observables[:, 0].astype(np.uint8)
            continue
        training_dataset = (
            training_data_by_basis.get(basis, dataset)
            if training_data_by_basis is not None
            else dataset
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
        for det, obs in zip(dataset.detectors, dataset.observables, strict=True):
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
    )
