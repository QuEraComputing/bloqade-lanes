from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from bloqade.decoders import ConfidenceDecoder

from bloqade.lanes import GeminiLogicalSimulator

from ..application.baselines import injected_baseline
from ..application.constants import DEFAULT_BASIS_LABELS
from ..application.mld import (
    build_mld_decoders_from_pair,
    estimate_mld_ancilla_scores,
    train_mld_decoder_pair,
)
from ..application.mle import build_mle_decoders
from ..application.msd_kernels import (
    TomographyKernels,
    build_decoder_kernel_bundle,
    build_injected_kernel_bundle,
)
from ..application.table_decoders import TableDecoderClass
from ..application.thresholds import DecoderAdapter, evaluate_curve
from ..domain.kernels import DecoderPrimitiveSet
from ..domain.layout import (
    DEFAULT_SYNDROME_LAYOUT,
    SyndromeLayout,
    _normalize_valid_factory_targets,
    split_factory_bits,
)
from ..domain.special_tasks import (
    apply_special_tsim_circuit_strategy,
    build_task_map,
)
from ..domain.tasks import DemoTask
from ..standard.measurement_maps import build_measurement_maps
from ..standard.sampling import BasisDataset, run_task
from ..standard.types import FidelitySummary

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


@dataclass(frozen=True)
class MSDDecoderWorkflowConfig:
    """Configuration for the high-level MSD decoder workflow.

    Args:
        mld_train_shots: Number of shots sampled per basis to train MLD table
            decoders.
        eval_shots: Number of shots sampled per basis for decoder evaluation.
        target_bloch_vector: Target Bloch vector used for fidelity
            reconstruction.
        theta: U3 ``theta`` angle used by the injected baseline workflow.
        phi: U3 ``phi`` angle used by the injected baseline workflow.
        lam: U3 ``lambda`` angle used by the injected baseline workflow.
        decoder_primitive_set: State-injection and logical Squin kernels for
            the MSD workflow.
        valid_factory_targets: Accepted corrected ancilla/factory observable
            patterns. A 1D pattern is accepted and normalized to a single row.
        num_logical_qubits: Number of logical qubits in the MSD task.
        output_qubit: Logical qubit containing the output state.
        sim_type: Simulator backend used by ``DemoTask`` sampling.
        chunk_size: Maximum shots per simulator call.
        mld_rank_train_shots: Optional separate shot count for MLD ranking data.
            Defaults to ``mld_train_shots`` when omitted.
        binary_precision: Bayesian tomography grid precision.
        uncertainty_backend: Fidelity uncertainty backend.
        max_grid_points: Maximum adaptive grid size for Bayesian tomography.
        special_kernel_strategy: Special-task circuit prefix strategy.
        append_measurements: Whether to pass measurement maps into Gemini task
            construction. The MSD demos use ``False`` because the generated
            Gemini logical kernels already include the measurement/postprocessing
            structure expected by the workflow.
        basis_labels: Tomography basis labels.
        sign_vector: Per-axis sign convention for reconstructed fidelities.
        layout: Syndrome layout used to split output/factory bits.
        log: Whether high-level workflow helpers should print progress
            messages.
    """

    mld_train_shots: int
    eval_shots: int
    target_bloch_vector: np.ndarray | Sequence[float]
    theta: float
    phi: float
    lam: float
    decoder_primitive_set: DecoderPrimitiveSet
    valid_factory_targets: np.ndarray | Sequence[Sequence[int]] | Sequence[int]
    num_logical_qubits: int
    output_qubit: int = 0
    sim_type: str = "tsim"
    chunk_size: int | None = 1_000_000
    mld_rank_train_shots: int | None = None
    binary_precision: int | None = None
    uncertainty_backend: str = "wilson"
    max_grid_points: int = 1_500_000
    special_kernel_strategy: Literal["prefix_prepare", "compiled_inverse_prefix"] = (
        "compiled_inverse_prefix"
    )
    append_measurements: bool = False
    basis_labels: Sequence[str] = DEFAULT_BASIS_LABELS
    sign_vector: Sequence[float] = (1.0, -1.0, 1.0)
    layout: SyndromeLayout = DEFAULT_SYNDROME_LAYOUT
    log: bool = True

    def __post_init__(self) -> None:
        target = np.asarray(self.target_bloch_vector, dtype=np.float64)
        if target.shape != (3,):
            raise ValueError("target_bloch_vector must contain exactly three values.")
        sign = np.asarray(self.sign_vector, dtype=np.float64)
        if sign.shape != (3,):
            raise ValueError("sign_vector must contain exactly three values.")
        if self.mld_train_shots < 0 or self.eval_shots < 0:
            raise ValueError("mld_train_shots and eval_shots must be non-negative.")
        if self.mld_rank_train_shots is not None and self.mld_rank_train_shots < 0:
            raise ValueError("mld_rank_train_shots must be non-negative.")

        object.__setattr__(self, "target_bloch_vector", target)
        object.__setattr__(self, "sign_vector", tuple(float(x) for x in sign))
        object.__setattr__(
            self,
            "valid_factory_targets",
            _normalize_valid_factory_targets(self.valid_factory_targets),
        )

    @property
    def resolved_mld_rank_train_shots(self) -> int:
        """Return the ranking shot count used for MLD confidence scoring."""

        if self.mld_rank_train_shots is None:
            return self.mld_train_shots
        return self.mld_rank_train_shots


@dataclass(frozen=True)
class TomographyTasks:
    """Actual simulator tasks plus private decoder-reference tasks.

    Attributes:
        actual: Basis-labeled tasks for noisy/evaluation data.
        _special: Private basis-labeled tasks for decoder training or
            reference data.
    """

    actual: dict[str, DemoTask]
    _special: dict[str, DemoTask] = field(repr=False)


@dataclass(frozen=True)
class DecoderCurveOptions:
    """Options for one decoder threshold curve.

    Args:
        threshold_points: Number of threshold points to evaluate.
        threshold_policy: Threshold selection policy passed to
            ``evaluate_curve``.
        selection_mode: Curve selection mode, either ``"threshold"`` or
            ``"pattern_rank"``.
        min_accepted_per_basis: Minimum accepted samples per basis.
    """

    threshold_points: int = 64
    threshold_policy: str = "quantile"
    selection_mode: str = "threshold"
    min_accepted_per_basis: int = 50


def build_msd_tomography_kernels(
    config: MSDDecoderWorkflowConfig,
    *,
    log: bool | None = None,
) -> TomographyKernels:
    """Build the actual/special tomography kernels for the MSD workflow.

    Args:
        config: Workflow configuration containing the MSD primitive set and
            logical-qubit layout.
        log: If true, print a progress message. Defaults to ``config.log`` when
            omitted.

    Returns:
        Basis-labeled actual and special MSD tomography kernels.
    """

    log = config.log if log is None else log
    if log:
        print("Building MSD tomography kernels...")

    return build_decoder_kernel_bundle(
        config.decoder_primitive_set,
        num_logical_qubits=config.num_logical_qubits,
        output_qubit=config.output_qubit,
        special_kernel_strategy=config.special_kernel_strategy,
    )


def build_injected_tomography_kernels(
    config: MSDDecoderWorkflowConfig,
    *,
    log: bool | None = None,
) -> TomographyKernels:
    """Build the actual/special tomography kernels for the injected workflow.

    Args:
        config: Workflow configuration containing injected U3 angles.
        log: If true, print a progress message. Defaults to ``config.log`` when
            omitted.

    Returns:
        Basis-labeled injected-state evaluation kernels and ideal injected
        decoder-reference kernels.
    """

    log = config.log if log is None else log
    if log:
        print("Building injected tomography kernels...")

    return build_injected_kernel_bundle(
        config.theta,
        config.phi,
        config.lam,
        output_qubit=0,
    )


def build_msd_tomography_tasks(
    simulator: GeminiLogicalSimulator,
    config: MSDDecoderWorkflowConfig,
    tomography_kernels: TomographyKernels,
    *,
    log: bool | None = None,
) -> TomographyTasks:
    """Compile MSD actual and private reference kernels into simulator tasks.

    Args:
        simulator: Gemini logical simulator used for task construction.
        config: Workflow configuration.
        tomography_kernels: MSD tomography kernels returned by
            ``build_msd_tomography_kernels``.
        log: If true, print a progress message. Defaults to ``config.log`` when
            omitted.

    Returns:
        Actual tasks and private reference tasks with the configured
        special-task circuit strategy applied.
    """

    log = config.log if log is None else log
    if log:
        print("Building MSD simulator tasks...")

    m2dets, m2obs = build_measurement_maps(config.num_logical_qubits)
    actual = build_task_map(
        simulator,
        tomography_kernels.actual,
        m2dets=m2dets,
        m2obs=m2obs,
        append_measurements=config.append_measurements,
    )
    special = build_task_map(
        simulator,
        tomography_kernels._special,
        m2dets=m2dets,
        m2obs=m2obs,
        append_measurements=config.append_measurements,
    )
    special = apply_special_tsim_circuit_strategy(
        special,
        config.special_kernel_strategy,
        normalize_observable_reference=True,
    )
    return TomographyTasks(actual=actual, _special=special)


def build_injected_tomography_tasks(
    simulator: GeminiLogicalSimulator,
    config: MSDDecoderWorkflowConfig,
    tomography_kernels: TomographyKernels,
    *,
    log: bool | None = None,
) -> TomographyTasks:
    """Compile injected actual and private reference kernels into tasks.

    Args:
        simulator: Gemini logical simulator used for task construction.
        config: Workflow configuration. Only common sampling/task settings are
            used.
        tomography_kernels: Injected tomography kernels returned by
            ``build_injected_tomography_kernels``.
        log: If true, print a progress message. Defaults to ``config.log`` when
            omitted.

    Returns:
        Actual injected tasks and private ideal decoder-reference tasks.
    """

    log = config.log if log is None else log
    if log:
        print("Building injected simulator tasks...")

    m2dets, m2obs = build_measurement_maps(1)
    return TomographyTasks(
        actual=build_task_map(
            simulator,
            tomography_kernels.actual,
            m2dets=m2dets,
            m2obs=m2obs,
            append_measurements=config.append_measurements,
        ),
        _special=build_task_map(
            simulator,
            tomography_kernels._special,
            m2dets=m2dets,
            m2obs=m2obs,
            append_measurements=config.append_measurements,
        ),
    )


def train_mld_decoder_suite(
    msd_tomography_tasks: TomographyTasks,
    config: MSDDecoderWorkflowConfig,
    *,
    table_decoder_cls: TableDecoderClass,
    log: bool | None = None,
) -> dict[str, DecoderAdapter]:
    """Train and score MLD decoders for all tomography bases.

    Args:
        msd_tomography_tasks: MSD tomography tasks whose private reference
            tasks train the tables and whose actual tasks rank ancilla patterns.
        config: Workflow configuration.
        table_decoder_cls: Table decoder implementation to train.
        log: If true, print progress messages. Defaults to ``config.log`` when
            omitted.

    Returns:
        Basis-labeled decoder adapters ready for threshold-curve evaluation.
    """

    log = config.log if log is None else log

    training_data: dict[str, BasisDataset] = {}
    for basis in config.basis_labels:
        if log:
            print(
                f"Sampling MLD table-training data for {basis} "
                f"with {config.mld_train_shots:,} shots..."
            )
        training_data[basis] = run_task(
            msd_tomography_tasks._special[basis],
            config.mld_train_shots,
            with_noise=True,
            chunk_size=config.chunk_size,
            sim_type=config.sim_type,
        )

    ranking_data: dict[str, BasisDataset] = {}
    rank_shots = config.resolved_mld_rank_train_shots
    for basis in config.basis_labels:
        if log:
            print(
                f"Sampling MLD ranking data for {basis} "
                f"with {rank_shots:,} shots..."
            )
        ranking_data[basis] = run_task(
            msd_tomography_tasks.actual[basis],
            rank_shots,
            with_noise=True,
            chunk_size=config.chunk_size,
            sim_type=config.sim_type,
        )

    decoder_pairs = {
        basis: train_mld_decoder_pair(
            training_data[basis],
            table_decoder_cls=table_decoder_cls,
            layout=config.layout,
        )
        for basis in config.basis_labels
    }
    ancilla_scores = estimate_mld_ancilla_scores(
        decoder_pairs,
        ranking_data,
        valid_factory_targets=config.valid_factory_targets,
        basis_labels=config.basis_labels,
        sign_vector=config.sign_vector,
        target_bloch=np.asarray(config.target_bloch_vector, dtype=np.float64),
        binary_precision=config.binary_precision,
        uncertainty_backend=config.uncertainty_backend,
        layout=config.layout,
    )

    adapters: dict[str, DecoderAdapter] = {}
    for basis in config.basis_labels:
        dataset = training_data[basis]
        anc_det, _ = split_factory_bits(
            dataset.detectors,
            dataset.observables,
            layout=config.layout,
        )
        full_decoder, factory_decoder = decoder_pairs[basis]
        adapters[basis] = build_mld_decoders_from_pair(
            full_decoder=full_decoder,
            factory_decoder=factory_decoder,
            full_syndrome_length=dataset.detectors.shape[1],
            factory_syndrome_length=anc_det.shape[1],
            ancilla_scores=ancilla_scores,
        )

    if log:
        print("Built MLD decoders with shared ancilla-pattern fidelity scores.")
    return adapters


def build_mle_decoder_suite(
    msd_tomography_tasks: TomographyTasks,
    *,
    gurobi_decoder_cls: type[ConfidenceDecoder],
    log: bool = True,
) -> dict[str, DecoderAdapter]:
    """Build MLE decoders from all private MSD reference tasks.

    Args:
        msd_tomography_tasks: MSD tomography tasks whose private reference tasks expose
            deterministic DEMs.
        gurobi_decoder_cls: Confidence-capable Gurobi decoder implementation.
        log: If true, print progress messages.

    Returns:
        Basis-labeled MLE decoder adapters.
    """

    decoders: dict[str, DecoderAdapter] = {}
    for basis, task in msd_tomography_tasks._special.items():
        if log:
            print(f"Building MLE decoder for {basis}...")
        decoders[basis] = build_mle_decoders(
            task,
            gurobi_decoder_cls=gurobi_decoder_cls,
        )
    return decoders


def sample_actual_data(
    tomography_tasks: TomographyTasks,
    config: MSDDecoderWorkflowConfig,
    *,
    with_noise: bool = True,
    log: bool | None = None,
) -> dict[str, BasisDataset]:
    """Sample actual-task detector/observable data for each basis.

    Args:
        tomography_tasks: Basis-labeled tomography tasks.
        config: Workflow configuration.
        with_noise: Whether to sample the noisy circuit path.
        log: If true, print progress messages. Defaults to ``config.log`` when
            omitted.

    Returns:
        Basis-labeled detector/observable datasets.
    """

    log = config.log if log is None else log

    data: dict[str, BasisDataset] = {}
    for basis in config.basis_labels:
        if log:
            print(
                f"Sampling actual data for {basis} with {config.eval_shots:,} shots..."
            )
        data[basis] = run_task(
            tomography_tasks.actual[basis],
            config.eval_shots,
            with_noise=with_noise,
            chunk_size=config.chunk_size,
            sim_type=config.sim_type,
        )
    return data


def evaluate_decoder_curves(
    actual_data: Mapping[str, BasisDataset],
    decoder_suites: Mapping[str, Mapping[str, DecoderAdapter]],
    config: MSDDecoderWorkflowConfig,
    *,
    curve_options: (
        Mapping[str, DecoderCurveOptions] | DecoderCurveOptions | None
    ) = None,
    curve_option_overrides: Mapping[str, DecoderCurveOptions] | None = None,
    log: bool | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Evaluate postselection curves for one or more decoder suites.

    Args:
        actual_data: Basis-labeled datasets sampled from actual MSD tasks.
        decoder_suites: Mapping from decoder label to basis-labeled decoder
            adapters, for example ``{"MLD": mld_decoders, "MLE": mle_decoders}``.
        config: Workflow configuration.
        curve_options: Optional shared or per-decoder curve options. If omitted,
            default ``DecoderCurveOptions`` are used for every decoder suite. If
            a single ``DecoderCurveOptions`` is provided, it is shared by every
            decoder suite. If a mapping is provided, labels not present in the
            mapping use default options.
        curve_option_overrides: Optional per-decoder options that override the
            shared or default options from ``curve_options``.
        log: If true, print progress messages. Defaults to ``config.log`` when
            omitted.

    Returns:
        Mapping from decoder label to curve arrays.
    """

    log = config.log if log is None else log

    def options_for(label: str) -> DecoderCurveOptions:
        if isinstance(curve_options, DecoderCurveOptions):
            options = curve_options
        elif curve_options is None:
            options = DecoderCurveOptions()
        else:
            options = curve_options.get(label, DecoderCurveOptions())
        if curve_option_overrides is not None:
            options = curve_option_overrides.get(label, options)
        return options

    curves: dict[str, dict[str, np.ndarray]] = {}
    for label, decoder_map in decoder_suites.items():
        options = options_for(label)

        if log:
            print(f"Evaluating {label} curve...")
        progress_label = label if log and label.upper() == "MLE" else None
        curves[label] = evaluate_curve(
            actual_data,
            decoder_map,
            binary_precision=config.binary_precision,
            threshold_points=options.threshold_points,
            metric=label,
            valid_factory_targets=config.valid_factory_targets,
            sign_vector=config.sign_vector,
            target_bloch=np.asarray(config.target_bloch_vector, dtype=np.float64),
            basis_labels=config.basis_labels,
            min_accepted_per_basis=options.min_accepted_per_basis,
            threshold_policy=options.threshold_policy,
            selection_mode=options.selection_mode,
            layout=config.layout,
            uncertainty_backend=config.uncertainty_backend,
            max_grid_points=config.max_grid_points,
            progress_label=progress_label,
        )
    return curves


def evaluate_injected_baseline(
    injected_tomography_tasks: TomographyTasks,
    config: MSDDecoderWorkflowConfig,
    *,
    table_decoder_cls: TableDecoderClass,
    raw: bool = False,
    log: bool | None = None,
) -> FidelitySummary:
    """Evaluate the injected-state baseline fidelity.

    Args:
        injected_tomography_tasks: Injected tomography tasks.
        config: Workflow configuration.
        table_decoder_cls: Table decoder implementation used for corrected
            injected baselines.
        raw: If true, skip decoder correction.
        log: If true, print a progress message. Defaults to ``config.log`` when
            omitted.

    Returns:
        Fidelity summary for the injected baseline.
    """

    log = config.log if log is None else log

    if log:
        kind = "raw" if raw else "corrected"
        print(f"Evaluating {kind} injected baseline...")
    return injected_baseline(
        injected_tomography_tasks.actual,
        eval_shots=config.eval_shots,
        binary_precision=config.binary_precision,
        table_decoder_cls=table_decoder_cls,
        sign_vector=config.sign_vector,
        target_bloch=np.asarray(config.target_bloch_vector, dtype=np.float64),
        raw=raw,
        training_task_map=None if raw else injected_tomography_tasks._special,
        basis_labels=config.basis_labels,
        uncertainty_backend=config.uncertainty_backend,
        sim_type=config.sim_type,
        max_grid_points=config.max_grid_points,
    )


def plot_decoder_curves(
    curves: Mapping[str, Mapping[str, np.ndarray]],
    *,
    injected_summary: FidelitySummary | None = None,
    min_accepted_fraction: float = 0.04,
    ax: "Axes | None" = None,
    title: str | None = None,
    log: bool = True,
) -> tuple["Figure", "Axes"]:
    """Plot decoder threshold curves.

    Args:
        curves: Mapping from decoder label to curve arrays as returned by
            ``evaluate_decoder_curves``.
        injected_summary: Optional injected baseline summary to overlay.
        min_accepted_fraction: Left x-axis limit for accepted fraction.
        ax: Optional Matplotlib axes to draw into.
        title: Optional plot title.
        log: If true, print a progress message.

    Returns:
        Matplotlib ``(figure, axes)`` pair.
    """

    if log:
        print("Plotting decoder curves...")

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    fig = cast("Figure", ax.figure)

    for label, curve in curves.items():
        accepted = np.asarray(curve["accepted_fraction"], dtype=np.float64)
        fidelity = np.asarray(curve["fidelity"], dtype=np.float64)
        credible = np.asarray(curve["credible"], dtype=np.float64)
        if len(accepted) == 0:
            continue
        ax.plot(accepted, fidelity, marker="o", linewidth=1.5, label=label)
        if credible.shape == (len(accepted), 2):
            ax.fill_between(
                accepted,
                credible[:, 0],
                credible[:, 1],
                alpha=0.15,
            )

    if injected_summary is not None:
        median = float(injected_summary["median"])
        low = float(injected_summary["low"])
        high = float(injected_summary["high"])
        ax.axhline(
            median,
            linestyle="--",
            linewidth=1.2,
            color="black",
            label="Injected baseline",
        )
        ax.axhspan(low, high, color="black", alpha=0.08)

    ax.set_xscale("log")
    ax.set_xlim(left=min_accepted_fraction)
    ax.set_xlabel("Accepted fraction")
    ax.set_ylabel("Fidelity")
    if title is not None:
        ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    return fig, ax


__all__ = [
    "DecoderCurveOptions",
    "TomographyTasks",
    "MSDDecoderWorkflowConfig",
    "build_injected_tomography_kernels",
    "build_injected_tomography_tasks",
    "build_mle_decoder_suite",
    "build_msd_tomography_kernels",
    "build_msd_tomography_tasks",
    "evaluate_decoder_curves",
    "evaluate_injected_baseline",
    "plot_decoder_curves",
    "sample_actual_data",
    "train_mld_decoder_suite",
]
