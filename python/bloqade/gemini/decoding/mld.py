from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import ContextManager, Protocol

import numpy as np
from bloqade.decoders import BaseDecoder
from demo.msd_utils.domain.confidence import TableDecoderWithConfidence
from demo.msd_utils.standard.bit_packing import (
    pack_boolean_array,
    packed_pattern_targets,
)
from demo.msd_utils.standard.dem import make_layout_only_dem
from demo.msd_utils.standard.tomography import (
    DEFAULT_TARGET_BLOCH,
    fidelity_from_zero_one_counts,
)

from .constants import DEFAULT_BASIS_LABELS
from .layout import (
    DEFAULT_SYNDROME_LAYOUT,
    SyndromeLayout,
    _normalize_valid_factory_targets,
    split_factory_bits,
)
from .postselection import (
    DecoderAdapter,
    _make_decoder_adapter,
    _pack_threshold_dataset,
)
from .sampling import BasisDataset, SimulatorTask, _iter_task_datasets
from .types import TableDecoderClass


class _ProgressBar(Protocol):
    """Minimal progress-bar protocol used by streamed MLD helpers."""

    def update(self, n: int | float = 1) -> object:
        """Advance the progress bar."""


def _select_output_observables(
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


def _tqdm_progress(
    *,
    label: str | None,
    total: int,
) -> ContextManager[_ProgressBar | None]:
    """Return a tqdm progress bar when requested and available."""

    if label is None:
        return nullcontext(None)

    try:
        from tqdm.auto import tqdm
    except ImportError:
        return nullcontext(None)

    return tqdm(
        total=int(total),
        desc=label,
        unit="shots",
        dynamic_ncols=True,
    )


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
    log: bool = True,
    label: str | None = None,
) -> tuple[BaseDecoder, BaseDecoder]:
    """Train full and factory MLD table decoders from sampled data.

    Args:
        training_dataset: Detector/observable samples used for table training.
        table_decoder_cls: Table decoder class to train.
        layout: Syndrome layout separating output and factory syndrome bits.
        log: If true, print table-decoder training progress.
        label: Optional basis/task label included in log messages.

    Returns:
        Pair ``(full_decoder, factory_decoder)``.
    """

    prefix = f" for {label}" if label is not None else ""
    if log:
        print(f"Preparing MLD table-training arrays{prefix}...", flush=True)
    training_arrays = _mld_training_arrays(training_dataset, layout=layout)
    if log:
        print(
            f"Training full MLD table decoder{prefix} "
            f"from {len(training_arrays.full_det_obs):,} shots...",
            flush=True,
        )
    full_decoder = table_decoder_cls.from_det_obs_shots(
        make_layout_only_dem(
            training_arrays.full_detector_count,
            training_arrays.full_observable_count,
        ),
        training_arrays.full_det_obs,
    )
    if log:
        print(
            f"Training factory MLD table decoder{prefix} "
            f"from {len(training_arrays.factory_det_obs):,} shots...",
            flush=True,
        )
    factory_decoder = table_decoder_cls.from_det_obs_shots(
        make_layout_only_dem(
            training_arrays.factory_detector_count,
            training_arrays.factory_observable_count,
        ),
        training_arrays.factory_det_obs,
    )
    if log:
        print(f"Finished MLD table decoders{prefix}.", flush=True)
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
    progress_label: str | None = None,
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
        progress_label: Optional tqdm label used to show streamed training
            progress.

    Returns:
        Pair ``(full_decoder, factory_decoder)``.

    Raises:
        ValueError: If ``shots`` yields no training data.
    """

    with _tqdm_progress(label=progress_label, total=shots) as progress:
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
            make_layout_only_dem(
                training_arrays.full_detector_count,
                training_arrays.full_observable_count,
            ),
            training_arrays.full_det_obs,
        )
        factory_decoder = table_decoder_cls.from_det_obs_shots(
            make_layout_only_dem(
                training_arrays.factory_detector_count,
                training_arrays.factory_observable_count,
            ),
            training_arrays.factory_det_obs,
        )
        if progress is not None:
            progress.update(len(first_chunk.detectors))

        for chunk in chunk_iter:
            chunk_arrays = _mld_training_arrays(chunk, layout=layout)
            full_decoder.update_det_obs_counts(chunk_arrays.full_det_obs)
            factory_decoder.update_det_obs_counts(chunk_arrays.factory_det_obs)
            if progress is not None:
                progress.update(len(chunk.detectors))

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

    packed_targets = packed_pattern_targets(targets)
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
        raise ValueError("Need ranking data to score MLD patterns.")

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


def _unpack_packed_array(packed: np.ndarray, length: int) -> np.ndarray:
    """Unpack little-endian packed integers into a 2D boolean bit matrix."""

    packed = np.asarray(packed, dtype=np.uint64).reshape(-1, 1)
    if length == 0:
        return np.zeros((packed.shape[0], 0), dtype=bool)
    shifts = np.arange(length, dtype=np.uint64).reshape(1, -1)
    return ((packed >> shifts) & 1).astype(bool)


def _batch_decode_packed_corrections(
    decoder: BaseDecoder,
    packed_syndromes: np.ndarray,
    *,
    syndrome_length: int,
) -> dict[int, int]:
    """Decode unique packed syndromes and return packed corrections by syndrome."""

    packed_syndromes = np.asarray(packed_syndromes, dtype=np.uint64).reshape(-1)
    if len(packed_syndromes) == 0:
        return {}

    cache_correction = getattr(decoder, "cache_correction", None)
    if callable(cache_correction):
        cache_correction()
    correction_table = getattr(decoder, "_maximum_likelihood_correction", None)
    if isinstance(correction_table, np.ndarray):
        packed_corrections = np.asarray(
            correction_table[packed_syndromes],
            dtype=np.uint64,
        )
    elif isinstance(correction_table, dict):
        packed_corrections = np.array(
            [int(correction_table.get(int(packed), 0)) for packed in packed_syndromes],
            dtype=np.uint64,
        )
    else:
        syndrome_bits = _unpack_packed_array(packed_syndromes, syndrome_length)
        corrections = np.asarray(decoder.decode(syndrome_bits), dtype=np.uint8)
        if corrections.ndim == 1:
            corrections = corrections.reshape(1, -1)

        if corrections.shape[1] == 0:
            packed_corrections = np.zeros(len(packed_syndromes), dtype=np.uint64)
        else:
            packed_corrections = pack_boolean_array(corrections).astype(np.uint64)

    return {
        int(syndrome): int(correction)
        for syndrome, correction in zip(
            packed_syndromes,
            packed_corrections,
            strict=True,
        )
    }


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
    ancilla_decode_cache = _batch_decode_packed_corrections(
        factory_decoder,
        unique_anc_det,
        syndrome_length=packed_dataset.anc_det.shape[1],
    )

    full_decode_cache = {
        packed: correction & 1
        for packed, correction in _batch_decode_packed_corrections(
            full_decoder,
            np.unique(unique_groups[:, 2]),
            syndrome_length=dataset.detectors.shape[1],
        ).items()
    }

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
    progress_label: str | None = None,
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
        progress_label: Optional tqdm label prefix used to show streamed
            ranking progress for each basis.

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

    packed_targets = packed_pattern_targets(targets)
    corrected_by_pattern = {
        basis: defaultdict(lambda: np.zeros(2, dtype=np.int64))
        for basis in basis_labels
    }
    ancilla_detectors: int | None = None

    for basis in basis_labels:
        full_decoder, factory_decoder = decoder_by_basis[basis]
        basis_progress_label = (
            f"{progress_label} {basis}" if progress_label is not None else None
        )
        with _tqdm_progress(label=basis_progress_label, total=shots) as progress:
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
                if progress is not None:
                    progress.update(len(dataset.detectors))

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
