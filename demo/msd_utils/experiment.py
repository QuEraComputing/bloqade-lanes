from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from bloqade.lanes import GeminiLogicalSimulator

from .circuits import (
    apply_special_tsim_circuit_strategy,
    build_decoder_kernel_bundle,
    build_injected_decoder_kernel_map,
    build_measurement_maps,
    build_task_map,
    make_noisy_steane7_initializer,
)
from .common import DEFAULT_SYNDROME_LAYOUT, DemoTask, SyndromeLayout
from .core import (
    DEFAULT_BASIS_LABELS,
    DEFAULT_TARGET_BLOCH,
    BasisDataset,
    run_task,
    split_factory_bits,
)
from .decoders import (
    DecoderAdapter,
    build_mld_decoders_from_pair,
    build_mle_decoders,
    estimate_mld_ancilla_scores,
    evaluate_curve,
    evaluate_mld_curve,
    injected_baseline,
    train_mld_decoder_pair,
)


@dataclass(frozen=True)
class ExperimentTaskMaps:
    simulator: GeminiLogicalSimulator
    noisy_initializer: Any
    actual_tasks: dict[str, DemoTask]
    special_tasks: dict[str, DemoTask]
    injected_tasks: dict[str, DemoTask]
    injected_decoder_tasks: dict[str, DemoTask]


@dataclass(frozen=True)
class ExperimentDiagnostics:
    compiled_task_keys: dict[str, tuple[str, ...]]
    injected_tomography_means: dict[str, float]
    special_noiseless_observables: dict[str, tuple[tuple[int, ...], ...]]
    injected_decoder_noiseless_observables: dict[str, tuple[tuple[int, ...], ...]]


@dataclass(frozen=True)
class MLDTrainingArtifacts:
    training_data_by_basis: dict[str, BasisDataset]
    ranking_data_by_basis: dict[str, BasisDataset]
    decoder_pairs_by_basis: dict[str, tuple[Any, Any]]
    ancilla_scores: np.ndarray
    decoder_map_by_basis: dict[str, DecoderAdapter]


@dataclass(frozen=True)
class DecoderSweepEvaluation:
    actual_data_by_basis: dict[str, BasisDataset]
    mle_decoders: dict[str, DecoderAdapter]
    mld_curve: dict[str, np.ndarray]
    mle_curve: dict[str, np.ndarray]
    injected_summary_corrected: dict[str, Any]
    injected_summary_raw: dict[str, Any]


@dataclass(frozen=True)
class MSDPrepConfig:
    theta_offset: float = 0.30
    phi_offset: float = 0.0
    lam_offset: float = 0.0
    output_qubit: int = 0
    ideal_theta_scale_pi: float = 0.3041
    ideal_phi_scale_pi: float = 0.25
    ideal_lam: float = 0.0


@dataclass(frozen=True)
class MSDRunConfig:
    mld_train_shots: int = 10_000_000
    eval_shots: int = 1_000_000
    posterior_samples: int = 200_000
    mle_threshold_points: int = 64
    mle_threshold_policy: str = "quantile"
    uncertainty_backend: str = "wilson"
    mld_rank_train_shots: int | None = None
    valid_factory_targets: tuple[tuple[int, ...], ...] = ((0, 0, 0, 0),)
    basis_labels: tuple[str, ...] = DEFAULT_BASIS_LABELS
    mld_sign_vector: tuple[float, ...] = (1.0, -1.0, 1.0)
    injected_raw_sign_vector: tuple[float, ...] = (1.0, -1.0, 1.0)
    injected_corrected_sign_vector: tuple[float, ...] = (1.0, -1.0, 1.0)
    mle_score_mode: str = "best_available"
    target_bloch: tuple[float, float, float] = tuple(
        float(x) for x in DEFAULT_TARGET_BLOCH
    )

    @classmethod
    def fast(cls) -> "MSDRunConfig":
        return cls(
            mld_train_shots=20_000,
            eval_shots=4_000,
            posterior_samples=20_000,
            mle_threshold_points=24,
        )

    @classmethod
    def paper(cls) -> "MSDRunConfig":
        return cls()


@dataclass(frozen=True)
class MSDPlotConfig:
    min_accepted_fraction: float = 0.02
    figsize: tuple[float, float] = (6.0, 4.0)
    title: str = (
        "Reproduction of Fig. 3(b) with bloqade-decoders on GeminiLogicalSimulator"
    )
    show_credible_bands: bool = True


@dataclass(frozen=True)
class ResolvedPrepParameters:
    ideal_theta: float
    ideal_phi: float
    ideal_lam: float
    theta: float
    phi: float
    lam: float
    prep_bloch: tuple[float, float, float]
    prep_fidelity: float


@dataclass(frozen=True)
class MSDFig3BExperimentResult:
    prep: ResolvedPrepParameters
    run_config: MSDRunConfig
    plot_config: MSDPlotConfig
    task_maps: ExperimentTaskMaps
    diagnostics: ExperimentDiagnostics
    mld_training: MLDTrainingArtifacts
    evaluation: DecoderSweepEvaluation


def _sorted_unique_observable_rows(
    dataset: BasisDataset,
) -> tuple[tuple[int, ...], ...]:
    unique_rows = np.unique(np.asarray(dataset.observables, dtype=np.uint8), axis=0)
    return tuple(tuple(int(x) for x in row.tolist()) for row in unique_rows)


def u3_prep_bloch(theta: float, phi: float) -> np.ndarray:
    return np.array(
        [
            math.sin(theta) * math.cos(phi),
            math.sin(theta) * math.sin(phi),
            math.cos(theta),
        ],
        dtype=np.float64,
    )


def resolve_msd_prep_parameters(
    prep_config: MSDPrepConfig,
    *,
    target_bloch: Sequence[float] = DEFAULT_TARGET_BLOCH,
) -> ResolvedPrepParameters:
    ideal_theta = prep_config.ideal_theta_scale_pi * math.pi
    ideal_phi = prep_config.ideal_phi_scale_pi * math.pi
    ideal_lam = prep_config.ideal_lam
    theta = ideal_theta + prep_config.theta_offset
    phi = ideal_phi + prep_config.phi_offset
    lam = ideal_lam + prep_config.lam_offset
    prep_bloch = u3_prep_bloch(theta, phi)
    prep_fidelity = (
        0.5
        + float(np.dot(prep_bloch, np.asarray(target_bloch, dtype=np.float64))) / 2.0
    )
    return ResolvedPrepParameters(
        ideal_theta=ideal_theta,
        ideal_phi=ideal_phi,
        ideal_lam=ideal_lam,
        theta=theta,
        phi=phi,
        lam=lam,
        prep_bloch=tuple(float(x) for x in prep_bloch.tolist()),
        prep_fidelity=float(prep_fidelity),
    )


def build_experiment_task_maps(
    *,
    actual_kernels: Mapping[str, Any],
    special_kernels: Mapping[str, Any],
    injected_kernels: Mapping[str, Any],
    injected_decoder_kernels: Mapping[str, Any],
    msd_measurement_maps: tuple[Any, Any],
    injected_measurement_maps: tuple[Any, Any],
    simulator: GeminiLogicalSimulator | None = None,
    append_measurements: bool = False,
    special_tsim_circuit_strategy: str | None = "prefix_prepare",
) -> ExperimentTaskMaps:
    sim = simulator if simulator is not None else GeminiLogicalSimulator()
    noisy_initializer = make_noisy_steane7_initializer(sim)
    actual_tasks = build_task_map(
        sim,
        actual_kernels,
        m2dets=msd_measurement_maps[0],
        m2obs=msd_measurement_maps[1],
        noisy_initializer=noisy_initializer,
        append_measurements=append_measurements,
    )
    special_tasks = apply_special_tsim_circuit_strategy(
        build_task_map(
            sim,
            special_kernels,
            m2dets=msd_measurement_maps[0],
            m2obs=msd_measurement_maps[1],
            noisy_initializer=noisy_initializer,
            append_measurements=append_measurements,
        ),
        special_tsim_circuit_strategy,
    )
    injected_tasks = build_task_map(
        sim,
        injected_kernels,
        m2dets=injected_measurement_maps[0],
        m2obs=injected_measurement_maps[1],
        noisy_initializer=noisy_initializer,
        append_measurements=append_measurements,
    )
    injected_decoder_tasks = build_task_map(
        sim,
        injected_decoder_kernels,
        m2dets=injected_measurement_maps[0],
        m2obs=injected_measurement_maps[1],
        noisy_initializer=noisy_initializer,
        append_measurements=append_measurements,
    )
    return ExperimentTaskMaps(
        simulator=sim,
        noisy_initializer=noisy_initializer,
        actual_tasks=actual_tasks,
        special_tasks=special_tasks,
        injected_tasks=injected_tasks,
        injected_decoder_tasks=injected_decoder_tasks,
    )


def collect_experiment_diagnostics(
    task_maps: ExperimentTaskMaps,
    *,
    injected_tomography_shots: int = 4_000,
    special_reference_shots: int = 64,
    injected_decoder_reference_shots: int = 64,
) -> ExperimentDiagnostics:
    compiled_task_keys = {
        "actual": tuple(task_maps.actual_tasks),
        "special": tuple(task_maps.special_tasks),
        "injected": tuple(task_maps.injected_tasks),
        "injected_decoder": tuple(task_maps.injected_decoder_tasks),
    }
    injected_tomography_means = {}
    for basis, task in task_maps.injected_tasks.items():
        data = run_task(task, injected_tomography_shots, with_noise=False)
        mean_value = float(
            np.mean(1.0 - 2.0 * np.asarray(data.observables, dtype=np.float64)[:, 0])
        )
        injected_tomography_means[basis] = mean_value
    special_noiseless_observables = {
        basis: _sorted_unique_observable_rows(
            run_task(task, special_reference_shots, with_noise=False)
        )
        for basis, task in task_maps.special_tasks.items()
    }
    injected_decoder_noiseless_observables = {
        basis: _sorted_unique_observable_rows(
            run_task(task, injected_decoder_reference_shots, with_noise=False)
        )
        for basis, task in task_maps.injected_decoder_tasks.items()
    }
    return ExperimentDiagnostics(
        compiled_task_keys=compiled_task_keys,
        injected_tomography_means=injected_tomography_means,
        special_noiseless_observables=special_noiseless_observables,
        injected_decoder_noiseless_observables=injected_decoder_noiseless_observables,
    )


def print_experiment_diagnostics(diagnostics: ExperimentDiagnostics) -> None:
    print("Compiled tasks:")
    print(" actual:", list(diagnostics.compiled_task_keys["actual"]))
    print(" special:", list(diagnostics.compiled_task_keys["special"]))
    print(" injected:", list(diagnostics.compiled_task_keys["injected"]))
    print(
        " injected decoder:", list(diagnostics.compiled_task_keys["injected_decoder"])
    )
    print("Injected tomography diagnostic (noiseless <labeled basis>):")
    for basis, mean_value in diagnostics.injected_tomography_means.items():
        print(f"  {basis}: {mean_value:.6f}")
    for basis, rows in diagnostics.special_noiseless_observables.items():
        print(f"{basis} special-state noiseless observables: {list(rows)}")


def train_mld_experiment(
    *,
    special_tasks: Mapping[str, DemoTask],
    actual_tasks: Mapping[str, DemoTask],
    train_shots: int,
    table_decoder_cls: type,
    valid_factory_targets: np.ndarray,
    sign_vector: Sequence[float],
    ranking_train_shots: int | None = None,
    target_bloch: np.ndarray = DEFAULT_TARGET_BLOCH,
    basis_labels: Sequence[str] = DEFAULT_BASIS_LABELS,
    layout: SyndromeLayout = DEFAULT_SYNDROME_LAYOUT,
    verbose: bool = True,
) -> MLDTrainingArtifacts:
    resolved_ranking_shots = (
        train_shots if ranking_train_shots is None else ranking_train_shots
    )
    training_data_by_basis: dict[str, BasisDataset] = {}
    for basis, task in special_tasks.items():
        if verbose:
            print(
                f"Sampling MLD table-training data for {basis} with {train_shots:,} shots..."
            )
        dataset = run_task(task, train_shots, with_noise=True)
        training_data_by_basis[basis] = dataset
        if verbose:
            print("cached MLD table-training data")

    ranking_data_by_basis: dict[str, BasisDataset] = {}
    for basis, task in actual_tasks.items():
        if verbose:
            print(
                f"Sampling MLD ranking data for {basis} with {resolved_ranking_shots:,} shots..."
            )
        dataset = run_task(task, resolved_ranking_shots, with_noise=True)
        ranking_data_by_basis[basis] = dataset
        if verbose:
            print("cached MLD ranking data")

    decoder_pairs_by_basis = {
        basis: train_mld_decoder_pair(
            dataset,
            table_decoder_cls=table_decoder_cls,
            layout=layout,
        )
        for basis, dataset in training_data_by_basis.items()
    }
    ancilla_scores = estimate_mld_ancilla_scores(
        decoder_pairs_by_basis,
        ranking_data_by_basis,
        valid_factory_targets=valid_factory_targets,
        basis_labels=basis_labels,
        sign_vector=sign_vector,
        target_bloch=target_bloch,
        layout=layout,
    )
    decoder_map_by_basis = {}
    for basis, dataset in training_data_by_basis.items():
        anc_det, _anc_obs = split_factory_bits(
            dataset.detectors,
            dataset.observables,
            layout=layout,
        )
        full_decoder, factory_decoder = decoder_pairs_by_basis[basis]
        decoder_map_by_basis[basis] = build_mld_decoders_from_pair(
            full_decoder=full_decoder,
            factory_decoder=factory_decoder,
            full_syndrome_length=dataset.detectors.shape[1],
            factory_syndrome_length=anc_det.shape[1],
            ancilla_scores=ancilla_scores,
        )
    if verbose:
        print("built MLD decoders with shared ancilla-pattern fidelity scores")
    return MLDTrainingArtifacts(
        training_data_by_basis=training_data_by_basis,
        ranking_data_by_basis=ranking_data_by_basis,
        decoder_pairs_by_basis=decoder_pairs_by_basis,
        ancilla_scores=ancilla_scores,
        decoder_map_by_basis=decoder_map_by_basis,
    )


def evaluate_decoder_experiment(
    *,
    actual_tasks: Mapping[str, DemoTask],
    special_tasks: Mapping[str, DemoTask],
    injected_tasks: Mapping[str, DemoTask],
    injected_decoder_tasks: Mapping[str, DemoTask],
    mld_decoder_map: Mapping[str, DecoderAdapter],
    eval_shots: int,
    posterior_samples: int,
    table_decoder_cls: type,
    gurobi_decoder_cls: type,
    valid_factory_targets: np.ndarray,
    sign_vector: Sequence[float],
    injected_corrected_sign_vector: Sequence[float],
    injected_raw_sign_vector: Sequence[float],
    mle_threshold_points: int,
    mle_score_mode: str = "best_available",
    target_bloch: np.ndarray = DEFAULT_TARGET_BLOCH,
    basis_labels: Sequence[str] = DEFAULT_BASIS_LABELS,
    mle_threshold_policy: str = "quantile",
    uncertainty_backend: str = "wilson",
    layout: SyndromeLayout = DEFAULT_SYNDROME_LAYOUT,
) -> DecoderSweepEvaluation:
    actual_data_by_basis = {
        basis: run_task(task, eval_shots, with_noise=True)
        for basis, task in actual_tasks.items()
    }
    mle_decoders = {
        basis: build_mle_decoders(
            task,
            gurobi_decoder_cls=gurobi_decoder_cls,
            score_mode=mle_score_mode,
            layout=layout,
        )
        for basis, task in special_tasks.items()
    }
    mld_curve = evaluate_mld_curve(
        actual_data_by_basis,
        mld_decoder_map,
        posterior_samples=posterior_samples,
        valid_factory_targets=valid_factory_targets,
        sign_vector=sign_vector,
        target_bloch=target_bloch,
        basis_labels=basis_labels,
        layout=layout,
        uncertainty_backend=uncertainty_backend,
    )
    mle_curve = evaluate_curve(
        actual_data_by_basis,
        mle_decoders,
        posterior_samples=posterior_samples,
        threshold_points=mle_threshold_points,
        metric="MLE logical gap",
        valid_factory_targets=valid_factory_targets,
        sign_vector=sign_vector,
        target_bloch=target_bloch,
        basis_labels=basis_labels,
        threshold_policy=mle_threshold_policy,
        layout=layout,
        uncertainty_backend=uncertainty_backend,
    )
    injected_summary_corrected = injected_baseline(
        injected_tasks,
        eval_shots=eval_shots,
        posterior_samples=posterior_samples,
        table_decoder_cls=table_decoder_cls,
        sign_vector=injected_corrected_sign_vector,
        target_bloch=target_bloch,
        training_task_map=injected_decoder_tasks,
        basis_labels=basis_labels,
        uncertainty_backend=uncertainty_backend,
    )
    injected_summary_raw = injected_baseline(
        injected_tasks,
        eval_shots=eval_shots,
        posterior_samples=posterior_samples,
        table_decoder_cls=table_decoder_cls,
        sign_vector=injected_raw_sign_vector,
        target_bloch=target_bloch,
        raw=True,
        basis_labels=basis_labels,
        uncertainty_backend=uncertainty_backend,
    )
    return DecoderSweepEvaluation(
        actual_data_by_basis=actual_data_by_basis,
        mle_decoders=mle_decoders,
        mld_curve=mld_curve,
        mle_curve=mle_curve,
        injected_summary_corrected=injected_summary_corrected,
        injected_summary_raw=injected_summary_raw,
    )


def run_msd_fig3b_experiment(
    *,
    prep_config: MSDPrepConfig,
    run_config: MSDRunConfig,
    table_decoder_cls: type,
    gurobi_decoder_cls: type,
    plot_config: MSDPlotConfig | None = None,
    simulator: GeminiLogicalSimulator | None = None,
    append_measurements: bool = False,
    verbose: bool = True,
) -> MSDFig3BExperimentResult:
    resolved_plot_config = plot_config if plot_config is not None else MSDPlotConfig()
    target_bloch = np.asarray(run_config.target_bloch, dtype=np.float64)
    prep = resolve_msd_prep_parameters(prep_config, target_bloch=target_bloch)
    kernel_bundle = build_decoder_kernel_bundle(
        prep.theta,
        prep.phi,
        prep.lam,
        output_qubit=prep_config.output_qubit,
    )
    task_maps = build_experiment_task_maps(
        actual_kernels=kernel_bundle.actual,
        special_kernels=kernel_bundle.special,
        injected_kernels=kernel_bundle.injected,
        injected_decoder_kernels=build_injected_decoder_kernel_map(),
        msd_measurement_maps=build_measurement_maps(5),
        injected_measurement_maps=build_measurement_maps(1),
        simulator=simulator,
        append_measurements=append_measurements,
    )
    diagnostics = collect_experiment_diagnostics(task_maps)
    if verbose:
        print_experiment_diagnostics(diagnostics)
    mld_training = train_mld_experiment(
        special_tasks=task_maps.special_tasks,
        actual_tasks=task_maps.actual_tasks,
        train_shots=run_config.mld_train_shots,
        ranking_train_shots=run_config.mld_rank_train_shots,
        table_decoder_cls=table_decoder_cls,
        valid_factory_targets=np.asarray(
            run_config.valid_factory_targets,
            dtype=np.uint8,
        ),
        sign_vector=run_config.mld_sign_vector,
        target_bloch=target_bloch,
        basis_labels=run_config.basis_labels,
        verbose=verbose,
    )
    evaluation = evaluate_decoder_experiment(
        actual_tasks=task_maps.actual_tasks,
        special_tasks=task_maps.special_tasks,
        injected_tasks=task_maps.injected_tasks,
        injected_decoder_tasks=task_maps.injected_decoder_tasks,
        mld_decoder_map=mld_training.decoder_map_by_basis,
        eval_shots=run_config.eval_shots,
        posterior_samples=run_config.posterior_samples,
        table_decoder_cls=table_decoder_cls,
        gurobi_decoder_cls=gurobi_decoder_cls,
        valid_factory_targets=np.asarray(
            run_config.valid_factory_targets,
            dtype=np.uint8,
        ),
        sign_vector=run_config.mld_sign_vector,
        injected_corrected_sign_vector=run_config.injected_corrected_sign_vector,
        injected_raw_sign_vector=run_config.injected_raw_sign_vector,
        mle_threshold_points=run_config.mle_threshold_points,
        mle_score_mode=run_config.mle_score_mode,
        target_bloch=target_bloch,
        basis_labels=run_config.basis_labels,
        mle_threshold_policy=run_config.mle_threshold_policy,
        uncertainty_backend=run_config.uncertainty_backend,
    )
    return MSDFig3BExperimentResult(
        prep=prep,
        run_config=run_config,
        plot_config=resolved_plot_config,
        task_maps=task_maps,
        diagnostics=diagnostics,
        mld_training=mld_training,
        evaluation=evaluation,
    )


def plot_msd_fig3b(
    result: MSDFig3BExperimentResult,
    *,
    plot_config: MSDPlotConfig | None = None,
    ax: Any | None = None,
) -> Any:
    import matplotlib.pyplot as plt

    resolved_plot_config = (
        plot_config if plot_config is not None else result.plot_config
    )
    created_figure = False
    if ax is None:
        _, ax = plt.subplots(figsize=resolved_plot_config.figsize)
        created_figure = True

    mle_curve = result.evaluation.mle_curve
    mld_curve = result.evaluation.mld_curve
    injected_summary = result.evaluation.injected_summary_corrected
    min_fraction = resolved_plot_config.min_accepted_fraction
    mle_plot_mask = mle_curve["accepted_fraction"] >= min_fraction
    mld_plot_mask = mld_curve["accepted_fraction"] >= min_fraction

    ax.plot(
        mle_curve["accepted_fraction"][mle_plot_mask],
        mle_curve["fidelity"][mle_plot_mask],
        marker="o",
        label="Distilled (MLE)",
    )
    if resolved_plot_config.show_credible_bands and len(mle_curve["credible"]) > 0:
        ax.fill_between(
            mle_curve["accepted_fraction"][mle_plot_mask],
            mle_curve["credible"][mle_plot_mask, 0],
            mle_curve["credible"][mle_plot_mask, 1],
            alpha=0.15,
        )

    ax.plot(
        mld_curve["accepted_fraction"][mld_plot_mask],
        mld_curve["fidelity"][mld_plot_mask],
        marker="o",
        label="Distilled (MLD)",
    )
    if resolved_plot_config.show_credible_bands and len(mld_curve["credible"]) > 0:
        ax.fill_between(
            mld_curve["accepted_fraction"][mld_plot_mask],
            mld_curve["credible"][mld_plot_mask, 0],
            mld_curve["credible"][mld_plot_mask, 1],
            alpha=0.15,
        )

    ax.axhline(
        injected_summary["median"],
        color="green",
        linewidth=2,
        label="Injected",
    )
    if resolved_plot_config.show_credible_bands:
        ax.axhspan(
            injected_summary["low"],
            injected_summary["high"],
            color="green",
            alpha=0.12,
        )

    ax.set_xlabel("Total accepted fraction")
    ax.set_ylabel("Magic state fidelity")
    ax.set_title(resolved_plot_config.title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    if created_figure:
        plt.show()
    return ax


def print_msd_fig3b_summary(result: MSDFig3BExperimentResult) -> None:
    print("Magic-state prep parameters:")
    print(
        "  ideal (theta, phi, lam) =",
        (result.prep.ideal_theta, result.prep.ideal_phi, result.prep.ideal_lam),
    )
    print(
        "  actual (theta, phi, lam) =",
        (result.prep.theta, result.prep.phi, result.prep.lam),
    )
    print("  prep Bloch vector        =", result.prep.prep_bloch)
    print("  prep fidelity vs target  =", result.prep.prep_fidelity)
    print(
        "Injected corrected baseline fidelity:",
        result.evaluation.injected_summary_corrected["point"],
    )
    print(
        "Injected raw baseline fidelity:",
        result.evaluation.injected_summary_raw["point"],
    )
    print(
        "MLE score mode:",
        next(iter(result.evaluation.mle_decoders.values())).factory_score_mode,
    )
    print(
        "MLE curve points:",
        len(result.evaluation.mle_curve["accepted_fraction"]),
    )
    print(
        "MLD curve points:",
        len(result.evaluation.mld_curve["accepted_fraction"]),
    )
