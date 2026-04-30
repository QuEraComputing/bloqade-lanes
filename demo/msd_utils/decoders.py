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
    fidelity_from_zero_one_counts,
    iter_task_datasets,
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


def select_output_observables(
    observables: np.ndarray,
    *,
    layout: SyndromeLayout = DEFAULT_SYNDROME_LAYOUT,
) -> np.ndarray:
    output_obs = np.asarray(observables, dtype=np.uint8)[
        :, : layout.output_observable_count
    ]
    if output_obs.shape[1] != layout.output_observable_count:
        raise ValueError(
            "Observable array does not contain the requested number of output "
            "observables."
        )
    return output_obs


def train_mld_decoder_pair(
    training_dataset: BasisDataset,
    *,
    table_decoder_cls: type,
    layout: SyndromeLayout = DEFAULT_SYNDROME_LAYOUT,
) -> tuple[Any, Any]:
    output_obs = select_output_observables(
        training_dataset.observables,
        layout=layout,
    )
    anc_det, anc_obs = split_factory_bits(
        training_dataset.detectors,
        training_dataset.observables,
        layout=layout,
    )

    full_decoder = table_decoder_cls.from_det_obs_shots(
        make_layout_only_dem(
            training_dataset.detectors.shape[1],
            output_obs.shape[1],
        ),
        np.concatenate([training_dataset.detectors, output_obs], axis=1).astype(bool),
    )
    factory_decoder = table_decoder_cls.from_det_obs_shots(
        make_layout_only_dem(anc_det.shape[1], anc_obs.shape[1]),
        np.concatenate([anc_det, anc_obs], axis=1).astype(bool),
    )
    return full_decoder, factory_decoder


def _det_obs_arrays_for_chunk(
    dataset: BasisDataset,
    *,
    layout: SyndromeLayout,
) -> tuple[np.ndarray, np.ndarray]:
    output_obs = select_output_observables(dataset.observables, layout=layout)
    anc_det, anc_obs = split_factory_bits(
        dataset.detectors,
        dataset.observables,
        layout=layout,
    )
    full_det_obs = np.concatenate([dataset.detectors, output_obs], axis=1).astype(bool)
    factory_det_obs = np.concatenate([anc_det, anc_obs], axis=1).astype(bool)
    return full_det_obs, factory_det_obs


def train_mld_decoder_pair_from_task(
    task: Any,
    shots: int,
    *,
    table_decoder_cls: type,
    layout: SyndromeLayout = DEFAULT_SYNDROME_LAYOUT,
    chunk_size: int | None = 1_000_000,
    with_noise: bool = True,
) -> tuple[Any, Any]:
    chunk_iter = iter_task_datasets(
        task,
        shots,
        with_noise=with_noise,
        chunk_size=chunk_size,
    )

    try:
        first_chunk = next(chunk_iter)
    except StopIteration as exc:
        raise ValueError(
            "Need at least one shot to train an MLD decoder pair."
        ) from exc

    full_det_obs, factory_det_obs = _det_obs_arrays_for_chunk(
        first_chunk,
        layout=layout,
    )
    anc_det, anc_obs = split_factory_bits(
        first_chunk.detectors,
        first_chunk.observables,
        layout=layout,
    )

    full_decoder = table_decoder_cls.from_det_obs_shots(
        make_layout_only_dem(
            first_chunk.detectors.shape[1],
            select_output_observables(
                first_chunk.observables,
                layout=layout,
            ).shape[1],
        ),
        full_det_obs,
    )
    factory_decoder = table_decoder_cls.from_det_obs_shots(
        make_layout_only_dem(anc_det.shape[1], anc_obs.shape[1]),
        factory_det_obs,
    )

    for chunk in chunk_iter:
        full_chunk_det_obs, factory_chunk_det_obs = _det_obs_arrays_for_chunk(
            chunk,
            layout=layout,
        )
        full_decoder.update_det_obs_counts(full_chunk_det_obs)
        factory_decoder.update_det_obs_counts(factory_chunk_det_obs)

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
    # TODO: the types need to be more specific here
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

    packed_targets = _packed_pattern_targets(targets)
    corrected_by_pattern = {
        basis: defaultdict(lambda: np.zeros(2, dtype=np.int64))
        for basis in basis_labels
    }
    ancilla_detectors: int | None = None

    for basis in basis_labels:
        full_decoder, factory_decoder = decoder_by_basis[basis]
        score_dataset = ranking_data_by_basis[basis]
        anc_det, _ = split_factory_bits(
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
        _accumulate_mld_pattern_counts(
            corrected_by_pattern[basis],
            score_dataset,
            full_decoder=full_decoder,
            factory_decoder=factory_decoder,
            packed_targets=packed_targets,
            layout=layout,
        )

    return _mld_scores_from_pattern_counts(
        corrected_by_pattern,
        ancilla_detectors=ancilla_detectors,
        basis_labels=basis_labels,
        sign_vector=sign_vector,
        target_bloch=target_bloch,
    )


def _mld_scores_from_pattern_counts(
    corrected_by_pattern: Mapping[str, Mapping[int, np.ndarray]],
    *,
    ancilla_detectors: int | None,
    basis_labels: Sequence[str],
    sign_vector: Sequence[float],
    target_bloch: np.ndarray,
) -> np.ndarray:
    if ancilla_detectors is None:
        raise ValueError("Need at least one ancilla detector to score MLD patterns.")

    scores = np.full(1 << ancilla_detectors, np.nan, dtype=np.float64)
    all_patterns = set()
    for basis in basis_labels:
        all_patterns.update(corrected_by_pattern[basis].keys())

    for packed in all_patterns:
        counts_x = corrected_by_pattern["X"].get(packed)
        counts_y = corrected_by_pattern["Y"].get(packed)
        counts_z = corrected_by_pattern["Z"].get(packed)
        if counts_x is None or counts_y is None or counts_z is None:
            continue
        if (
            min(int(np.sum(counts_x)), int(np.sum(counts_y)), int(np.sum(counts_z)))
            == 0
        ):
            continue
        scores[packed] = fidelity_from_zero_one_counts(
            int(counts_x[0]),
            int(counts_x[1]),
            int(counts_y[0]),
            int(counts_y[1]),
            int(counts_z[0]),
            int(counts_z[1]),
            posterior_samples=1,
            sign_vector=sign_vector,
            target_bloch=target_bloch,
            uncertainty_backend="wilson",
        )["point"]
    return scores


# TODO: continue reading here, 4/21 11:56 AM
def _accumulate_mld_pattern_counts(
    pattern_counts: defaultdict[int, np.ndarray],
    dataset: BasisDataset,
    *,
    full_decoder: Any,
    factory_decoder: Any,
    packed_targets: set[int],
    layout: SyndromeLayout,
) -> int:
    anc_det, anc_obs = split_factory_bits(
        dataset.detectors,
        dataset.observables,
        layout=layout,
    )
    if anc_det.shape[1] == 0:
        return 0

    packed_full_det = pack_boolean_array(dataset.detectors).astype(np.uint64)
    packed_anc_det = pack_boolean_array(anc_det).astype(np.uint64)
    packed_anc_obs = pack_boolean_array(anc_obs).astype(np.uint64)
    output_bits = np.asarray(dataset.observables[:, 0], dtype=np.uint8).astype(
        np.uint64
    )

    grouped = np.column_stack(
        [packed_anc_det, packed_anc_obs, packed_full_det, output_bits]
    )
    unique_groups, group_counts = np.unique(grouped, axis=0, return_counts=True)

    unique_anc_det = np.unique(unique_groups[:, 0])
    ancilla_decode_cache: dict[int, int] = {}
    for packed in unique_anc_det.tolist():
        anc_flip = np.asarray(
            factory_decoder.decode(
                unpack_packed_bits(int(packed), anc_det.shape[1]).astype(bool)
            ),
            dtype=np.uint8,
        )
        ancilla_decode_cache[int(packed)] = (
            int(pack_boolean_array(anc_flip)[0]) if len(anc_flip) else 0
        )

    unique_full_det = np.unique(unique_groups[:, 2])
    full_decode_cache: dict[int, int] = {}
    for packed in unique_full_det.tolist():
        full_flip = np.asarray(
            full_decoder.decode(
                unpack_packed_bits(int(packed), dataset.detectors.shape[1]).astype(bool)
            ),
            dtype=np.uint8,
        )
        full_decode_cache[int(packed)] = int(full_flip[0]) if len(full_flip) else 0

    accepted = 0
    for row, count in zip(unique_groups, group_counts, strict=True):
        packed_a_det = int(row[0])
        packed_a_obs = int(row[1])
        packed_det = int(row[2])
        output_bit = int(row[3])

        corrected_anc_packed = packed_a_obs ^ ancilla_decode_cache[packed_a_det]
        if corrected_anc_packed not in packed_targets:
            continue

        corrected_output_bit = output_bit ^ full_decode_cache[packed_det]
        pattern_counts[packed_a_det][corrected_output_bit] += int(count)
        accepted += int(count)
    return accepted


def estimate_mld_ancilla_scores_from_tasks(
    decoder_by_basis: Mapping[str, tuple[Any, Any]],
    ranking_tasks_by_basis: Mapping[str, Any],
    shots: int,
    *,
    factory_target: np.ndarray | Sequence[int] | None = None,
    valid_factory_targets: np.ndarray | Sequence[Sequence[int]] | None = None,
    basis_labels: Sequence[str] = DEFAULT_BASIS_LABELS,
    sign_vector: Sequence[float] = (1.0, -1.0, 1.0),
    target_bloch: np.ndarray = DEFAULT_TARGET_BLOCH,
    layout: SyndromeLayout = DEFAULT_SYNDROME_LAYOUT,
    chunk_size: int | None = 1_000_000,
    with_noise: bool = True,
) -> np.ndarray:
    targets = resolve_valid_factory_targets(
        factory_target=factory_target,
        valid_factory_targets=valid_factory_targets,
    )
    if set(decoder_by_basis) != set(basis_labels):
        raise ValueError(
            "Need X/Y/Z decoder pairs to estimate shared MLD postselection scores."
        )
    if set(ranking_tasks_by_basis) != set(basis_labels):
        raise ValueError(
            "Need X/Y/Z ranking tasks to estimate shared MLD postselection scores."
        )

    packed_targets = _packed_pattern_targets(targets)
    corrected_by_pattern = {
        basis: defaultdict(lambda: np.zeros(2, dtype=np.int64))
        for basis in basis_labels
    }
    ancilla_detectors: int | None = None

    for basis in basis_labels:
        full_decoder, factory_decoder = decoder_by_basis[basis]
        for dataset in iter_task_datasets(
            ranking_tasks_by_basis[basis],
            shots,
            with_noise=with_noise,
            chunk_size=chunk_size,
        ):
            anc_det, _ = split_factory_bits(
                dataset.detectors,
                dataset.observables,
                layout=layout,
            )
            if ancilla_detectors is None:
                ancilla_detectors = anc_det.shape[1]
            elif ancilla_detectors != anc_det.shape[1]:
                raise ValueError(
                    "Inconsistent ancilla detector counts across MLD ranking tasks."
                )
            _accumulate_mld_pattern_counts(
                corrected_by_pattern[basis],
                dataset,
                full_decoder=full_decoder,
                factory_decoder=factory_decoder,
                packed_targets=packed_targets,
                layout=layout,
            )

    return _mld_scores_from_pattern_counts(
        corrected_by_pattern,
        ancilla_detectors=ancilla_detectors,
        basis_labels=basis_labels,
        sign_vector=sign_vector,
        target_bloch=target_bloch,
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


def _packed_pattern_targets(
    targets: np.ndarray,
) -> set[int]:
    return {
        int(x) for x in pack_boolean_array(np.asarray(targets, dtype=np.uint8)).tolist()
    }


def _weighted_quantiles_from_counts(
    values: np.ndarray,
    weights: np.ndarray,
    quantiles: np.ndarray,
) -> np.ndarray:
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    anc_det, anc_obs = split_factory_bits(
        dataset.detectors,
        dataset.observables,
        layout=layout,
    )
    packed_full_det = pack_boolean_array(dataset.detectors).astype(np.uint64)
    packed_anc_det = pack_boolean_array(anc_det).astype(np.uint64)
    packed_anc_obs = pack_boolean_array(anc_obs).astype(np.uint64)
    output_bits = np.asarray(dataset.observables[:, 0], dtype=np.uint8).astype(
        np.uint64
    )
    return (
        anc_det,
        anc_obs,
        packed_full_det,
        packed_anc_det,
        packed_anc_obs,
        output_bits,
    )


def _build_ancilla_decode_cache(
    decoder: DecoderAdapter,
    packed_anc_det: np.ndarray,
    *,
    syndrome_length: int,
) -> dict[int, tuple[int, float]]:
    unique_anc_det = np.unique(packed_anc_det)
    ancilla_decode_cache: dict[int, tuple[int, float]] = {}
    for packed in unique_anc_det.tolist():
        anc_flip, score = _call_decoder_fn(
            decoder.decode_factory,
            unpack_packed_bits(int(packed), syndrome_length),
        )
        anc_flip_bits = np.asarray(anc_flip, dtype=np.uint8)
        ancilla_decode_cache[int(packed)] = (
            int(pack_boolean_array(anc_flip_bits)[0]) if len(anc_flip_bits) else 0,
            float(score),
        )
    return ancilla_decode_cache


def _build_full_decode_cache(
    decoder: DecoderAdapter,
    packed_full_det: np.ndarray,
    *,
    syndrome_length: int,
) -> dict[int, int]:
    unique_full_det = np.unique(packed_full_det)
    full_decode_cache: dict[int, int] = {}
    for packed in unique_full_det.tolist():
        full_flip = np.asarray(
            _call_decoder_fn(
                decoder.decode_full,
                unpack_packed_bits(int(packed), syndrome_length),
            ),
            dtype=np.uint8,
        )
        full_decode_cache[int(packed)] = int(full_flip[0]) if len(full_flip) else 0
    return full_decode_cache


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
) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    np.ndarray,
    np.ndarray,
    int,
]:
    packed_targets = _packed_pattern_targets(targets)
    per_basis_tables: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    global_score_weights: defaultdict[float, int] = defaultdict(int)
    total_shots = 0

    for basis in basis_labels:
        dataset = actual_data[basis]
        total_shots += len(dataset.observables)
        (
            anc_det,
            anc_obs,
            packed_full_det,
            packed_anc_det,
            packed_anc_obs,
            output_bits,
        ) = _pack_threshold_dataset(
            dataset,
            layout=layout,
        )

        ancilla_decode_cache = _build_ancilla_decode_cache(
            decoder_map[basis],
            packed_anc_det,
            syndrome_length=anc_det.shape[1],
        )
        grouped = np.column_stack(
            [packed_anc_det, packed_anc_obs, packed_full_det, output_bits]
        )
        unique_groups, group_counts = np.unique(grouped, axis=0, return_counts=True)

        score_to_counts: defaultdict[float, np.ndarray] = defaultdict(
            lambda: np.zeros(2, dtype=np.int64)
        )
        accepted_groups: list[tuple[float, int, int, int]] = []
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

            accepted_groups.append((float(score), packed_det, output_bit, int(count)))

        accepted_full_dets = np.fromiter(
            (packed_det for _score, packed_det, _output_bit, _count in accepted_groups),
            dtype=np.uint64,
            count=len(accepted_groups),
        )
        full_decode_cache = _build_full_decode_cache(
            decoder_map[basis],
            accepted_full_dets,
            syndrome_length=dataset.detectors.shape[1],
        )

        for score, packed_det, output_bit, count in accepted_groups:
            corrected_output_bit = output_bit ^ full_decode_cache[packed_det]
            score_to_counts[score][corrected_output_bit] += count
            global_score_weights[score] += count

        per_basis_tables[basis] = _score_count_dict_to_arrays(score_to_counts)

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
    if len(scores) == 0:
        return 0, 0
    idx = int(np.searchsorted(scores, threshold, side="left"))
    if idx >= len(scores):
        return 0, 0
    return int(np.sum(zero_counts[idx:])), int(np.sum(one_counts[idx:]))


def _evaluate_cached_threshold_curve(
    per_basis_tables: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    score_array: np.ndarray,
    *,
    score_weights: np.ndarray | None = None,
    posterior_samples: int,
    threshold_points: int,
    sign_vector: Sequence[float],
    target_bloch: np.ndarray,
    basis_labels: Sequence[str],
    min_accepted_per_basis: int,
    threshold_policy: str,
    total_shots: int,
    uncertainty_backend: str,
) -> dict[str, np.ndarray]:
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

        summary = fidelity_from_zero_one_counts(
            counts_by_basis["X"][0],
            counts_by_basis["X"][1],
            counts_by_basis["Y"][0],
            counts_by_basis["Y"][1],
            counts_by_basis["Z"][0],
            counts_by_basis["Z"][1],
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
    )
    try:
        return _evaluate_cached_threshold_curve(
            per_basis_tables,
            score_array,
            score_weights=score_weights,
            posterior_samples=posterior_samples,
            threshold_points=threshold_points,
            sign_vector=sign_vector,
            target_bloch=target_bloch,
            basis_labels=basis_labels,
            min_accepted_per_basis=min_accepted_per_basis,
            threshold_policy=threshold_policy,
            total_shots=total_shots,
            uncertainty_backend=uncertainty_backend,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"No factory-accepted shots found for {metric} threshold sweep"
        ) from exc


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
