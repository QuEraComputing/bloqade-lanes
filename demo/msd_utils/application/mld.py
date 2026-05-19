from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from bloqade.decoders import BaseDecoder

from ..application.constants import DEFAULT_BASIS_LABELS
from ..application.table_decoders import TableDecoderClass
from ..application.thresholds import (
    DecoderAdapter,
    _make_decoder_adapter,
    _pack_threshold_dataset,
)
from ..domain.confidence import TableDecoderWithConfidence
from ..domain.layout import (
    DEFAULT_SYNDROME_LAYOUT,
    SyndromeLayout,
    _normalize_valid_factory_targets,
    split_factory_bits,
)
from ..standard.bit_packing import (
    _packed_pattern_targets,
    pack_boolean_array,
    unpack_packed_bits,
)
from ..standard.dem import _make_layout_only_dem, _select_output_observables
from ..standard.sampling import BasisDataset, _iter_task_datasets
from ..standard.tomography import DEFAULT_TARGET_BLOCH, fidelity_from_zero_one_counts
from ..standard.types import SimulatorTask


@dataclass(frozen=True)
class _MLDTrainingArrays:
    """Detector/observable arrays and dimensions used to train MLD decoders."""

    full_det_obs: np.ndarray
    factory_det_obs: np.ndarray
    full_detector_count: int
    full_observable_count: int
    factory_detector_count: int
    factory_observable_count: int


# TODO: Not sure if this helper function is the best abstraction that we want.
# Is the logic here (a) really reused across train_mld_decoder_pair and
# train_mld_decoder_pair_from_task, and (b) is there a cleaner way to write/do
# we need train_mld_decoder_pair_from_task? Maybe the task itself should handle
# the batching/streaming logic. -- see below comment about implementing like a
# DataLoader class.
def _mld_training_arrays(
    dataset: BasisDataset,
    *,
    layout: SyndromeLayout,
) -> _MLDTrainingArrays:
    """Build full and factory training arrays from one sampled dataset."""

    output_obs = _select_output_observables(
        dataset.observables,
        layout=layout,
    )
    anc_det, anc_obs = split_factory_bits(
        dataset.detectors,
        dataset.observables,
        layout=layout,
    )
    return _MLDTrainingArrays(
        full_det_obs=np.concatenate([dataset.detectors, output_obs], axis=1).astype(
            bool
        ),
        factory_det_obs=np.concatenate([anc_det, anc_obs], axis=1).astype(bool),
        full_detector_count=dataset.detectors.shape[1],
        full_observable_count=output_obs.shape[1],
        factory_detector_count=anc_det.shape[1],
        factory_observable_count=anc_obs.shape[1],
    )


def train_mld_decoder_pair(
    training_dataset: BasisDataset,
    *,
    table_decoder_cls: TableDecoderClass,
    layout: SyndromeLayout = DEFAULT_SYNDROME_LAYOUT,
) -> tuple[BaseDecoder, BaseDecoder]:
    """Train full and factory MLD table decoders from sampled data.

    Args:
        training_dataset: Detector/observable samples used for table training.
        table_decoder_cls: Table decoder class to train.
        layout: Syndrome layout separating output and factory syndrome bits.

    Returns:
        Pair ``(full_decoder, factory_decoder)``.
    """

    training_arrays = _mld_training_arrays(training_dataset, layout=layout)
    full_decoder = table_decoder_cls.from_det_obs_shots(
        _make_layout_only_dem(
            training_arrays.full_detector_count,
            training_arrays.full_observable_count,
        ),
        training_arrays.full_det_obs,
    )
    factory_decoder = table_decoder_cls.from_det_obs_shots(
        _make_layout_only_dem(
            training_arrays.factory_detector_count,
            training_arrays.factory_observable_count,
        ),
        training_arrays.factory_det_obs,
    )
    return full_decoder, factory_decoder


# TODO: a better fix would to make this independent of a "Task" object and to
# implement some kind of batched dataloader. however, that might be too
# complicated for the first iteration of stdlibs
def train_mld_decoder_pair_from_task(
    task: SimulatorTask,
    shots: int,
    *,
    table_decoder_cls: TableDecoderClass,
    layout: SyndromeLayout = DEFAULT_SYNDROME_LAYOUT,
    chunk_size: int | None = None,
    with_noise: bool = True,
    sim_type: str = "tsim",
) -> tuple[BaseDecoder, BaseDecoder]:
    """Train full and factory MLD table decoders by sampling a task in chunks.

    Args:
        task: Simulator task used to generate training data.
        shots: Number of shots to sample.
        table_decoder_cls: Table decoder class to train.
        layout: Syndrome layout separating output and factory syndrome bits.
        chunk_size: Optional maximum shots sampled per task call. ``None``
            samples all requested shots in one call.
        with_noise: Whether to sample the noisy circuit path.
        sim_type: Simulator backend for ``DemoTask`` instances.

    Returns:
        Pair ``(full_decoder, factory_decoder)``.

    Raises:
        ValueError: If ``shots`` yields no training data.
    """

    chunk_iter = _iter_task_datasets(
        task,
        shots,
        with_noise=with_noise,
        chunk_size=chunk_size,
        sim_type=sim_type,
    )

    try:
        first_chunk = next(chunk_iter)
    except StopIteration as exc:
        raise ValueError(
            "Need at least one shot to train an MLD decoder pair."
        ) from exc

    training_arrays = _mld_training_arrays(first_chunk, layout=layout)

    full_decoder = table_decoder_cls.from_det_obs_shots(
        _make_layout_only_dem(
            training_arrays.full_detector_count,
            training_arrays.full_observable_count,
        ),
        training_arrays.full_det_obs,
    )
    factory_decoder = table_decoder_cls.from_det_obs_shots(
        _make_layout_only_dem(
            training_arrays.factory_detector_count,
            training_arrays.factory_observable_count,
        ),
        training_arrays.factory_det_obs,
    )

    for chunk in chunk_iter:
        chunk_arrays = _mld_training_arrays(chunk, layout=layout)
        full_decoder.update_det_obs_counts(chunk_arrays.full_det_obs)
        factory_decoder.update_det_obs_counts(chunk_arrays.factory_det_obs)

    return full_decoder, factory_decoder


def estimate_mld_ancilla_scores(
    decoder_by_basis: Mapping[str, tuple[BaseDecoder, BaseDecoder]],
    ranking_data_by_basis: Mapping[str, BasisDataset],
    *,
    valid_factory_targets: np.ndarray | Sequence[Sequence[int]] | Sequence[int],
    basis_labels: Sequence[str] = DEFAULT_BASIS_LABELS,
    sign_vector: Sequence[float] = (1.0, -1.0, 1.0),
    target_bloch: np.ndarray = DEFAULT_TARGET_BLOCH,
    binary_precision: int | None = 4,
    uncertainty_backend: str = "wilson",
    layout: SyndromeLayout = DEFAULT_SYNDROME_LAYOUT,
) -> np.ndarray:
    """Estimate shared ancilla-pattern confidence scores for MLD decoders.

    Args:
        decoder_by_basis: Mapping from basis label to
            ``(full_decoder, factory_decoder)`` pairs.
        ranking_data_by_basis: Mapping from basis label to sampled ranking data.
        valid_factory_targets: Valid corrected factory observable patterns.
        basis_labels: Tomography basis labels to evaluate.
        sign_vector: Per-axis sign convention for fidelity reconstruction.
        target_bloch: Target Bloch vector for fidelity scoring.
        binary_precision: Precision used by Bayesian tomography scoring.
        uncertainty_backend: Fidelity uncertainty backend.
        layout: Syndrome layout separating output and factory bits.

    Returns:
        Array indexed by packed ancilla detector pattern containing estimated
        fidelity scores. Missing patterns are ``nan``.
    """

    targets = _normalize_valid_factory_targets(valid_factory_targets)
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
        binary_precision=binary_precision,
        uncertainty_backend=uncertainty_backend,
    )


def _mld_scores_from_pattern_counts(
    corrected_by_pattern: Mapping[str, Mapping[int, np.ndarray]],
    *,
    ancilla_detectors: int | None,
    basis_labels: Sequence[str],
    sign_vector: Sequence[float],
    target_bloch: np.ndarray,
    binary_precision: int | None = 4,
    uncertainty_backend: str = "wilson",
) -> np.ndarray:
    """Convert per-pattern tomography counts into fidelity scores."""

    if ancilla_detectors is None:
        raise ValueError("Need at least one ancilla detector to score MLD patterns.")

    scores = np.full(1 << ancilla_detectors, np.nan, dtype=np.float64)
    all_patterns = set()
    # TODO: ideally, we don't hardcode to just single qubit logical tomography; but this would require additional refactoring
    # Plus, I think our current fidelity path is coupled to single qubit (it's bloch-sphere based).. not sure how it would
    # work with multi-qubit logical tomography
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
            binary_precision=binary_precision,
            sign_vector=sign_vector,
            target_bloch=target_bloch,
            uncertainty_backend=uncertainty_backend,
        )["point"]
    return scores


# TODO: continue reading here, 4/21 11:56 AM
def _accumulate_mld_pattern_counts(
    pattern_counts: defaultdict[int, np.ndarray],
    dataset: BasisDataset,
    *,
    full_decoder: BaseDecoder,
    factory_decoder: BaseDecoder,
    packed_targets: set[int],
    layout: SyndromeLayout,
) -> int:
    """Accumulate corrected output counts keyed by ancilla detector pattern."""

    packed_dataset = _pack_threshold_dataset(dataset, layout=layout)
    if packed_dataset.anc_det.shape[1] == 0:
        return 0

    grouped = np.column_stack(
        [
            packed_dataset.packed_anc_det,
            packed_dataset.packed_anc_obs,
            packed_dataset.packed_full_det,
            packed_dataset.output_bits,
        ]
    )
    unique_groups, group_counts = np.unique(grouped, axis=0, return_counts=True)

    unique_anc_det = np.unique(unique_groups[:, 0])
    ancilla_decode_cache: dict[int, int] = {}
    for packed in unique_anc_det.tolist():
        anc_flip = np.asarray(
            factory_decoder.decode(
                unpack_packed_bits(int(packed), packed_dataset.anc_det.shape[1]).astype(
                    bool
                )
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


# TODO: ideally, estimating the scores from the tasks is really a batching
# problem, which should be solved at the dataloader level, NOT by creating an
# extra function.
def estimate_mld_ancilla_scores_from_tasks(
    decoder_by_basis: Mapping[str, tuple[BaseDecoder, BaseDecoder]],
    ranking_tasks_by_basis: Mapping[str, SimulatorTask],
    shots: int,
    *,
    valid_factory_targets: np.ndarray | Sequence[Sequence[int]] | Sequence[int],
    basis_labels: Sequence[str] = DEFAULT_BASIS_LABELS,
    sign_vector: Sequence[float] = (1.0, -1.0, 1.0),
    target_bloch: np.ndarray = DEFAULT_TARGET_BLOCH,
    binary_precision: int | None = 4,
    uncertainty_backend: str = "wilson",
    layout: SyndromeLayout = DEFAULT_SYNDROME_LAYOUT,
    chunk_size: int | None = None,
    with_noise: bool = True,
    sim_type: str = "tsim",
) -> np.ndarray:
    """Estimate MLD ancilla scores by sampling ranking tasks in chunks.

    Args:
        decoder_by_basis: Mapping from basis label to
            ``(full_decoder, factory_decoder)`` pairs.
        ranking_tasks_by_basis: Mapping from basis label to simulator task.
        shots: Number of ranking shots to sample per basis.
        valid_factory_targets: Valid corrected factory observable patterns.
        basis_labels: Tomography basis labels to evaluate.
        sign_vector: Per-axis sign convention for fidelity reconstruction.
        target_bloch: Target Bloch vector for fidelity scoring.
        binary_precision: Precision used by Bayesian tomography scoring.
        uncertainty_backend: Fidelity uncertainty backend.
        layout: Syndrome layout separating output and factory bits.
        chunk_size: Optional maximum shots sampled per task call. ``None``
            samples all requested shots in one call.
        with_noise: Whether to sample the noisy circuit path.
        sim_type: Simulator backend for ``DemoTask`` instances.

    Returns:
        Array indexed by packed ancilla detector pattern containing estimated
        fidelity scores. Missing patterns are ``nan``.
    """

    targets = _normalize_valid_factory_targets(valid_factory_targets)
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
        for dataset in _iter_task_datasets(
            ranking_tasks_by_basis[basis],
            shots,
            with_noise=with_noise,
            chunk_size=chunk_size,
            sim_type=sim_type,
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
        binary_precision=binary_precision,
        uncertainty_backend=uncertainty_backend,
    )


# NOTE: this function is really just putting different pieces/data together.
def build_mld_decoders_from_pair(
    *,
    full_decoder: BaseDecoder,
    factory_decoder: BaseDecoder,
    full_syndrome_length: int,
    factory_syndrome_length: int,
    ancilla_scores: np.ndarray,
) -> DecoderAdapter:
    """Build a decoder adapter from trained full/factory MLD decoders.

    Args:
        full_decoder: Decoder for full detector syndromes and output flips.
        factory_decoder: Decoder for factory detector syndromes and factory
            observable flips.
        full_syndrome_length: Number of detector bits consumed by
            ``full_decoder``.
        factory_syndrome_length: Number of detector bits consumed by
            ``factory_decoder``.
        ancilla_scores: Score table indexed by packed factory detector
            syndrome.

    Returns:
        Decoder adapter suitable for threshold-curve evaluation.
    """

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
