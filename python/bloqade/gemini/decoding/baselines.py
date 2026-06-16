from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np
from demo.msd_utils.standard.dem import make_layout_only_dem
from demo.msd_utils.standard.tomography import (
    DEFAULT_TARGET_BLOCH,
    FidelitySummary,
    fidelity_from_counts,
    logical_expectation,
)

from .constants import (
    DEFAULT_BASIS_LABELS,
    DEFAULT_IDEAL_FACTORY_ACCEPTANCE,
)
from .layout import (
    _ancilla_matches_valid_targets,
    _normalize_valid_factory_targets,
    split_factory_bits,
)
from .sampling import SimulatorTask, run_task
from .types import TableDecoderClass


# TODO: technically maybe we don't need these cases where we
# infer the sign vector. Ideally, this should be fixed/taken care of by the user..
def infer_factory_target(
    task_map: Mapping[str, SimulatorTask],
    *,
    shots: int = 12_000,
    basis_labels: Sequence[str] = DEFAULT_BASIS_LABELS,
    ideal_factory_acceptance: float = DEFAULT_IDEAL_FACTORY_ACCEPTANCE,
) -> np.ndarray:
    """Infer the noiseless factory target branch from simulated task data.

    Args:
        task_map: Basis-labeled task map to sample without noise.
        shots: Number of noiseless shots to sample per basis.
        basis_labels: Basis labels to evaluate.
        ideal_factory_acceptance: Expected factory acceptance fraction used to
            choose among observed branches.

    Returns:
        Factory target observable pattern as a ``uint8`` array.
    """

    counts: Counter[tuple[int, ...]] = Counter()
    for basis in basis_labels:
        data = run_task(task_map[basis], shots, with_noise=False)
        for row in data.observables[:, 1:]:
            counts[tuple(map(int, row))] += 1

    total = sum(counts.values())
    ranked = sorted(
        counts.items(),
        key=lambda item: (abs(item[1] / total - ideal_factory_acceptance), -item[1]),
    )
    print("Top noiseless ancilla branches:")
    for pattern, count in ranked[:8]:
        print(pattern, count / total)
    return np.asarray(ranked[0][0], dtype=np.uint8)


def infer_distilled_sign_vector(
    task_map: Mapping[str, SimulatorTask],
    *,
    valid_factory_targets: np.ndarray | Sequence[Sequence[int]] | Sequence[int],
    shots: int = 12_000,
    basis_labels: Sequence[str] = DEFAULT_BASIS_LABELS,
    target_bloch: np.ndarray = DEFAULT_TARGET_BLOCH,
) -> np.ndarray:
    """Infer the sign convention aligning accepted outputs to a target state.

    Args:
        task_map: Basis-labeled task map to sample without noise.
        valid_factory_targets: Valid corrected factory observable patterns.
        shots: Number of noiseless shots to sample per basis.
        basis_labels: Basis labels to evaluate.
        target_bloch: Target Bloch vector used to choose the sign convention.

    Returns:
        Three-element sign vector for X/Y/Z tomography.
    """

    targets = _normalize_valid_factory_targets(valid_factory_targets)
    corrected: dict[str, np.ndarray] = {}
    for basis in basis_labels:
        data = run_task(task_map[basis], shots, with_noise=False)
        mask = _ancilla_matches_valid_targets(data.observables[:, 1:], targets)
        corrected[basis] = data.observables[mask, 0].astype(np.uint8)

    raw_bloch = np.array(
        [
            logical_expectation(corrected["X"]),
            logical_expectation(corrected["Y"]),
            logical_expectation(corrected["Z"]),
        ]
    )

    sign_candidates = [
        np.array([sx, sy, sz], dtype=np.float64)
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    ]
    scored = sorted(
        (
            (float(np.dot(raw_bloch * sign, target_bloch)), sign)
            for sign in sign_candidates
        ),
        key=lambda item: item[0],
        reverse=True,
    )

    print("Noiseless accepted-branch Bloch:", raw_bloch)
    print("Chosen distilled sign vector:", scored[0][1], "score:", scored[0][0])
    return scored[0][1]


# NOTE: is NOT used in the decoders notebook, but is used in the reprod notebook
# (for naive postselection) -- ideally, customize decoders path to take in a
# decoder that just postselects on 0
def naive_injected_summary(
    task_map: Mapping[str, SimulatorTask],
    *,
    sign_vector: Sequence[float],
    binary_precision: int | None = None,
    shots: int,
    require_zero_detectors: bool = False,
    min_accepted_per_basis: int = 50,
    basis_labels: Sequence[str] = DEFAULT_BASIS_LABELS,
    target_bloch: np.ndarray = DEFAULT_TARGET_BLOCH,
    max_grid_points: int = 1_500_000,
) -> dict[str, object]:
    """Summarize the injected baseline without a decoder.

    Args:
        task_map: Basis-labeled injected task map.
        sign_vector: Per-axis sign convention for fidelity reconstruction.
        binary_precision: Precision used by Bayesian tomography scoring.
        shots: Number of shots to sample per basis.
        require_zero_detectors: Whether to postselect on all detector bits zero.
        min_accepted_per_basis: Minimum accepted samples required per basis.
        basis_labels: Basis labels to evaluate.
        target_bloch: Target Bloch vector for fidelity calculation.
        max_grid_points: Maximum adaptive grid size for Bayesian tomography.

    Returns:
        Fidelity summary augmented with accepted-fraction metadata.
    """

    corrected: dict[str, np.ndarray] = {}
    accepted_fraction_by_basis: dict[str, float] = {}

    for basis in basis_labels:
        data = run_task(task_map[basis], shots, with_noise=True)
        mask = np.ones(len(data.observables), dtype=bool)
        if require_zero_detectors:
            mask &= np.all(data.detectors == 0, axis=1)

        corrected[basis] = data.observables[mask, 0].astype(np.uint8)
        accepted_fraction_by_basis[basis] = float(np.mean(mask))

    if min(len(corrected[basis]) for basis in basis_labels) < min_accepted_per_basis:
        raise RuntimeError("Too few accepted injected shots.")

    return {
        **fidelity_from_counts(
            corrected["X"],
            corrected["Y"],
            corrected["Z"],
            binary_precision,
            sign_vector=sign_vector,
            target_bloch=target_bloch,
            max_grid_points=max_grid_points,
        ),
        "accepted_fraction": float(np.mean(list(accepted_fraction_by_basis.values()))),
        "accepted_fraction_by_basis": accepted_fraction_by_basis,
    }


# NOTE: is NOT used in the decoders notebook, but is used in the reprod notebook
# (for naive postselection) -- ideally, customize decoders path to take in a
# decoder that just postselects on 0
def naive_distilled_summary(
    task_map: Mapping[str, SimulatorTask],
    *,
    valid_factory_targets: np.ndarray | Sequence[Sequence[int]] | Sequence[int],
    sign_vector: Sequence[float],
    binary_precision: int | None = None,
    shots: int,
    require_zero_ancilla_detectors: bool = False,
    min_accepted_per_basis: int = 50,
    basis_labels: Sequence[str] = DEFAULT_BASIS_LABELS,
    target_bloch: np.ndarray = DEFAULT_TARGET_BLOCH,
    max_grid_points: int = 1_500_000,
) -> dict[str, object]:
    """Summarize distilled-task performance with naive factory postselection.

    Args:
        task_map: Basis-labeled distilled task map.
        valid_factory_targets: Valid factory observable patterns.
        sign_vector: Per-axis sign convention for fidelity reconstruction.
        binary_precision: Precision used by Bayesian tomography scoring.
        shots: Number of shots to sample per basis.
        require_zero_ancilla_detectors: Whether to also require zero factory
            detector bits.
        min_accepted_per_basis: Minimum accepted samples required per basis.
        basis_labels: Basis labels to evaluate.
        target_bloch: Target Bloch vector for fidelity calculation.
        max_grid_points: Maximum adaptive grid size for Bayesian tomography.

    Returns:
        Fidelity summary augmented with accepted-fraction metadata and targets.
    """

    targets = _normalize_valid_factory_targets(valid_factory_targets)
    corrected: dict[str, np.ndarray] = {}
    accepted_fraction_by_basis: dict[str, float] = {}
    total_kept = 0
    total_shots = 0

    for basis in basis_labels:
        data = run_task(task_map[basis], shots, with_noise=True)
        anc_det, anc_obs = split_factory_bits(data.detectors, data.observables)
        mask = np.asarray(_ancilla_matches_valid_targets(anc_obs, targets), dtype=bool)
        if require_zero_ancilla_detectors:
            mask &= np.all(anc_det == 0, axis=1)

        corrected[basis] = data.observables[mask, 0].astype(np.uint8)
        accepted_fraction_by_basis[basis] = float(np.mean(mask))
        total_kept += int(np.sum(mask))
        total_shots += len(mask)

    if min(len(corrected[basis]) for basis in basis_labels) < min_accepted_per_basis:
        raise RuntimeError("Too few accepted distilled shots.")

    return {
        **fidelity_from_counts(
            corrected["X"],
            corrected["Y"],
            corrected["Z"],
            binary_precision,
            sign_vector=sign_vector,
            target_bloch=target_bloch,
            max_grid_points=max_grid_points,
        ),
        "accepted_fraction": total_kept / total_shots,
        "accepted_fraction_by_basis": accepted_fraction_by_basis,
        "valid_factory_targets": tuple(
            tuple(int(x) for x in row.tolist()) for row in targets
        ),
    }


def injected_baseline(
    task_map: Mapping[str, SimulatorTask],
    *,
    eval_shots: int,
    binary_precision: int | None = None,
    table_decoder_cls: TableDecoderClass,
    sign_vector: Sequence[float],
    target_bloch: np.ndarray = DEFAULT_TARGET_BLOCH,
    raw: bool = False,
    training_task_map: Mapping[str, SimulatorTask] | None = None,
    basis_labels: Sequence[str] = DEFAULT_BASIS_LABELS,
    uncertainty_backend: str = "wilson",
    sim_type: str = "tsim",
    max_grid_points: int = 1_500_000,
) -> FidelitySummary:
    """Estimate injected-state fidelity with an optional table-decoder correction.

    Args:
        task_map: Basis-labeled injected task map to evaluate.
        eval_shots: Number of shots to sample per basis.
        binary_precision: Precision used by Bayesian tomography scoring.
        table_decoder_cls: Table decoder class used when ``raw`` is false.
        sign_vector: Per-axis sign convention for fidelity reconstruction.
        target_bloch: Target Bloch vector for fidelity calculation.
        raw: If true, skip decoder training and use raw observable bits.
        training_task_map: Optional separate task map used for decoder training.
        basis_labels: Basis labels to evaluate.
        uncertainty_backend: Fidelity uncertainty backend.
        sim_type: Simulator backend for ``DemoTask`` instances.
        TODO: be more precise for `max_grid_points`
        max_grid_points: Maximum adaptive grid size for Bayesian tomography.

    Returns:
        Fidelity summary for the injected baseline.
    """

    corrected: dict[str, np.ndarray] = {}
    for basis in basis_labels:
        evaluation_dataset = run_task(
            task_map[basis],
            eval_shots,
            with_noise=True,
            sim_type=sim_type,
        )
        if raw:
            corrected[basis] = evaluation_dataset.observables[:, 0].astype(np.uint8)
            continue

        training_dataset = evaluation_dataset
        if training_task_map is not None:
            training_dataset = run_task(
                training_task_map[basis],
                eval_shots,
                with_noise=True,
                sim_type=sim_type,
            )

        decoder = table_decoder_cls.from_det_obs_shots(
            make_layout_only_dem(
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
        binary_precision,
        sign_vector=sign_vector,
        target_bloch=target_bloch,
        uncertainty_backend=uncertainty_backend,
        max_grid_points=max_grid_points,
    )
