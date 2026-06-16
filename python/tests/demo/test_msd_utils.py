from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import stim
from bloqade.decoders import ConfidenceDecoder, TableDecoder

from bloqade.lanes import GeminiLogicalSimulator

# pyright: reportAttributeAccessIssue=false


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from demo.msd_extras.qet import build_qet_kernel_maps, build_qet_primitives
from demo.msd_utils import (
    DEFAULT_TARGET_BLOCH,
    BasisDataset,
    DecoderAdapter,
    DecoderCurveOptions,
    DecoderPrimitiveSet,
    DemoTask,
    MSDDecoderWorkflowConfig,
    PostSelectionExperiment,
    SparseTableDecoder,
    SyndromeLayout,
    TableDecoderWithConfidence,
    TomographyResult,
    TomographyTasks,
    apply_special_tsim_circuit_strategy,
    build_decoder_kernel_bundle,
    build_injected_decoder_kernel_map,
    build_injected_kernel_bundle,
    build_injected_tomography_kernels,
    build_measurement_maps,
    build_mld_decoders_from_pair,
    build_mle_decoder_suite,
    build_mle_decoders,
    build_msd_primitives,
    build_msd_tomography_kernels,
    build_msd_tomography_tasks,
    build_task_map,
    estimate_mld_ancilla_scores,
    estimate_mld_ancilla_scores_from_tasks,
    evaluate_curve,
    evaluate_decoder_curves,
    evaluate_injected_baseline,
    evaluate_mld_curve,
    expectation_conf_interval,
    expectation_with_error_bar,
    fidelity_from_counts,
    fidelity_from_zero_one_counts,
    infer_distilled_sign_vector,
    infer_factory_target,
    injected_baseline,
    naive_distilled_summary,
    naive_injected_summary,
    pack_boolean_array,
    packed_bits_to_int,
    plot_decoder_curves,
    posterior_fidelity_summary,
    sample_actual_data,
    split_factory_bits,
    sub_detector_error_model,
    train_mld_decoder_pair,
    train_mld_decoder_pair_from_task,
    train_mld_decoder_suite,
    unpack_packed_bits,
)
from demo.msd_utils.domain.layout import _normalize_valid_factory_targets
from demo.msd_utils.domain.tasks import _ObservableFrame
from demo.msd_utils.standard.dem import make_layout_only_dem

import bloqade.gemini.decoding.special_tasks as circuits


def test_fidelity_from_counts_returns_ordered_interval():
    summary = fidelity_from_counts(
        np.array([0, 0, 1, 0], dtype=np.uint8),
        np.array([0, 1, 0, 0], dtype=np.uint8),
        np.array([0, 0, 0, 1], dtype=np.uint8),
        binary_precision=4,
    )
    assert set(summary) >= {"point", "median", "low", "high", "bloch"}
    assert summary["low"] <= summary["median"] <= summary["high"]
    assert len(summary["bloch"]) == 3


def test_fidelity_from_zero_one_counts_matches_array_counts():
    x_bits = np.array([0, 0, 1, 0], dtype=np.uint8)
    y_bits = np.array([1, 1, 0, 1], dtype=np.uint8)
    z_bits = np.array([0, 1, 1, 1], dtype=np.uint8)

    from_arrays = fidelity_from_counts(
        x_bits,
        y_bits,
        z_bits,
        sign_vector=(1.0, -1.0, 1.0),
        target_bloch=np.array([0.0, 1.0, 0.0], dtype=np.float64),
    )
    from_counts = fidelity_from_zero_one_counts(
        3,
        1,
        1,
        3,
        1,
        3,
        sign_vector=(1.0, -1.0, 1.0),
        target_bloch=np.array([0.0, 1.0, 0.0], dtype=np.float64),
    )

    assert from_counts["point"] == pytest.approx(from_arrays["point"])
    assert from_counts["bloch"] == pytest.approx(from_arrays["bloch"])


def test_expectation_helpers_return_ordered_interval_and_error_bar():
    interval = expectation_conf_interval(3, 1)
    expectation, error = expectation_with_error_bar(3, 1)

    assert interval.shape == (2,)
    assert interval[0] <= expectation <= interval[1]
    assert expectation == pytest.approx(0.5)
    assert error == pytest.approx((interval[1] - interval[0]) / 2.0)


def test_posterior_fidelity_summary_returns_ordered_interval():
    summary = posterior_fidelity_summary(
        np.array([8, 8, 8], dtype=np.int64),
        np.array([7, 6, 7], dtype=np.int64),
        DEFAULT_TARGET_BLOCH,
        binary_precision=4,
        max_grid_points=2_000,
    )

    assert np.isfinite(summary["point"])
    assert summary["low"] <= summary["median"] <= summary["high"]


def test_fidelity_from_counts_realistic_interval_is_finite_and_noncollapsed():
    x_bits = np.array([0] * 3060 + [1] * 940, dtype=np.uint8)
    y_bits = np.array([0] * 3040 + [1] * 960, dtype=np.uint8)
    z_bits = np.array([0] * 3090 + [1] * 910, dtype=np.uint8)

    summary = fidelity_from_counts(
        x_bits,
        y_bits,
        z_bits,
        binary_precision=7,
    )

    assert np.isfinite(summary["point"])
    assert np.isfinite(summary["median"])
    assert np.isfinite(summary["low"])
    assert np.isfinite(summary["high"])
    assert summary["high"] - summary["low"] > 1e-3


def test_split_factory_bits_and_pack_boolean_array():
    det = np.array([[1, 0, 1, 1, 0], [0, 1, 0, 0, 1]], dtype=np.uint8)
    obs = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8)
    anc_det, anc_obs = split_factory_bits(det, obs)
    assert anc_det.tolist() == [[1, 0], [0, 1]]
    assert anc_obs.tolist() == [[0, 1], [1, 0]]
    assert pack_boolean_array(anc_det).tolist() == [0b01, 0b10]


def test_bit_packing_round_trip_helpers():
    bits = np.array([1, 0, 1, 1], dtype=np.uint8)
    packed = packed_bits_to_int(bits)

    assert packed == 0b1101
    assert unpack_packed_bits(packed, len(bits)).tolist() == bits.tolist()


def test_normalize_valid_factory_targets_wraps_single_target():
    targets = _normalize_valid_factory_targets([0, 1, 0])
    assert targets.tolist() == [[0, 1, 0]]


def test_empty_valid_factory_target_is_accept_all_degenerate_pattern():
    targets = _normalize_valid_factory_targets(np.zeros((1, 0), dtype=np.uint8))

    assert targets.shape == (1, 0)
    assert targets.dtype == np.uint8


def test_mld_confidence_supports_empty_factory_stage():
    basis_labels = ("X", "Y", "Z")
    layout = SyndromeLayout(output_detector_count=1, output_observable_count=1)
    dataset = BasisDataset(
        detectors=np.array([[0], [1], [0], [1]], dtype=np.uint8),
        observables=np.array([[0], [0], [0], [0]], dtype=np.uint8),
    )
    full_decoder = TableDecoder.from_det_obs_shots(
        make_layout_only_dem(1, 1),
        np.concatenate([dataset.detectors, dataset.observables], axis=1).astype(bool),
    )
    factory_decoder = TableDecoder.from_det_obs_shots(
        stim.DetectorErrorModel(""),
        np.zeros((len(dataset.detectors), 0), dtype=bool),
    )
    decoder_pairs = {basis: (full_decoder, factory_decoder) for basis in basis_labels}

    scores = estimate_mld_ancilla_scores(
        decoder_pairs,
        {basis: dataset for basis in basis_labels},
        valid_factory_targets=np.zeros((1, 0), dtype=np.uint8),
        basis_labels=basis_labels,
        sign_vector=(1.0, 1.0, 1.0),
        target_bloch=np.array([1.0, 1.0, 1.0], dtype=np.float64) / np.sqrt(3.0),
        layout=layout,
    )
    assert scores.shape == (1,)
    assert np.isfinite(scores[0])

    adapter = build_mld_decoders_from_pair(
        full_decoder=full_decoder,
        factory_decoder=factory_decoder,
        full_syndrome_length=1,
        factory_syndrome_length=0,
        ancilla_scores=scores,
    )
    factory_flip, score = adapter.decode_factory(0)
    assert factory_flip == ()
    assert score == pytest.approx(scores[0])

    curves = evaluate_curve(
        {basis: dataset for basis in basis_labels},
        {basis: adapter for basis in basis_labels},
        binary_precision=4,
        threshold_points=2,
        metric="empty-factory",
        valid_factory_targets=np.zeros((1, 0), dtype=np.uint8),
        sign_vector=(1.0, 1.0, 1.0),
        target_bloch=np.array([1.0, 1.0, 1.0], dtype=np.float64) / np.sqrt(3.0),
        basis_labels=basis_labels,
        min_accepted_per_basis=1,
        layout=layout,
    )
    assert curves["accepted_fraction"].tolist() == [1.0]


def test_kernel_builders_return_expected_basis_maps():
    decoder = build_decoder_kernel_bundle(
        build_msd_primitives(0.1, 0.2, 0.3),
    )
    injected = build_injected_kernel_bundle(0.1, 0.2, 0.3)

    assert set(decoder.actual) == {"X", "Y", "Z"}
    assert set(decoder._special) == {"X", "Y", "Z"}
    assert not hasattr(decoder, "special")
    assert set(injected.actual) == {"X", "Y", "Z"}
    assert set(injected._special) == {"X", "Y", "Z"}
    assert not hasattr(injected, "special")


def test_build_injected_decoder_kernel_map_returns_basis_kernels():
    kernel_map = build_injected_decoder_kernel_map()

    assert set(kernel_map) == {"X", "Y", "Z"}
    assert all(hasattr(kernel, "code") for kernel in kernel_map.values())


def test_special_strategy_observable_frame_flag(monkeypatch: pytest.MonkeyPatch):
    def do_nothing(_: DemoTask[object]) -> None:
        return None

    monkeypatch.setattr(circuits, "_apply_prefix_prepare_to_task", do_nothing)

    normalized = DemoTask(task=object())  # type: ignore[arg-type]
    raw = DemoTask(task=object())  # type: ignore[arg-type]

    apply_special_tsim_circuit_strategy({"X": normalized}, "prefix_prepare")
    apply_special_tsim_circuit_strategy(
        {"X": raw},
        "prefix_prepare",
        normalize_observable_reference=False,
    )

    assert normalized.observable_frame is _ObservableFrame.NOISELESS_REFERENCE_FLIPS
    assert raw.observable_frame is _ObservableFrame.RAW


def test_prefix_prepare_uses_tsim_prefix_and_remains_deterministic():
    sim = GeminiLogicalSimulator()
    m2dets, m2obs = build_measurement_maps(5)
    decoder = build_decoder_kernel_bundle(
        build_msd_primitives(0.1, 0.2, 0.3),
        special_kernel_strategy="prefix_prepare",
    )

    special_tasks = build_task_map(
        sim,
        {"X": decoder._special["X"]},
        m2dets=m2dets,
        m2obs=m2obs,
        append_measurements=False,
    )
    demo_task = apply_special_tsim_circuit_strategy(
        special_tasks,
        "prefix_prepare",
    )["X"]

    assert "prepare_inverse" not in str(demo_task.task.physical_squin_kernel)

    result = demo_task.task.run(16, with_noise=False, run_detectors=True)
    assert len(np.unique(np.asarray(result.observables, dtype=np.uint8), axis=0)) == 1
    assert len(np.unique(np.asarray(result.detectors, dtype=np.uint8), axis=0)) == 1


def test_demo_task_clifft_backend_matches_result_shapes():
    pytest.importorskip("clifft")

    sim = GeminiLogicalSimulator()
    m2dets, m2obs = build_measurement_maps(5)
    decoder = build_decoder_kernel_bundle(
        build_msd_primitives(0.1, 0.2, 0.3),
        special_kernel_strategy="prefix_prepare",
    )
    special_tasks = build_task_map(
        sim,
        {"X": decoder._special["X"]},
        m2dets=m2dets,
        m2obs=m2obs,
        append_measurements=False,
    )
    demo_task = apply_special_tsim_circuit_strategy(
        special_tasks,
        "prefix_prepare",
    )["X"]

    detector_result = demo_task.run(
        4,
        with_noise=False,
        run_detectors=True,
        sim_type="clifft",
        seed=123,
    )
    assert np.asarray(detector_result.detectors, dtype=np.uint8).shape == (4, 15)
    assert np.asarray(detector_result.observables, dtype=np.uint8).shape == (4, 5)

    measurement_result = demo_task.run(
        4,
        with_noise=False,
        run_detectors=False,
        sim_type="clifft",
        seed=123,
    )
    assert np.asarray(measurement_result.measurements, dtype=np.uint8).shape == (4, 35)
    assert np.asarray(measurement_result.detectors, dtype=np.uint8).shape == (4, 15)
    assert np.asarray(measurement_result.observables, dtype=np.uint8).shape == (4, 5)


def test_decoder_kernel_bundle_accepts_decoder_primitive_set():
    primitive_set = build_qet_primitives(theta=0.1, phi0=0.2, phi1=0.3, phi2=0.4)
    decoder = build_decoder_kernel_bundle(
        primitive_set,
        num_logical_qubits=9,
        special_kernel_strategy="compiled_inverse_prefix",
    )

    assert set(decoder.actual) == {"X", "Y", "Z"}
    assert set(decoder._special) == {"X", "Y", "Z"}


def test_qet_primitives_integrate_with_decoder_kernel_bundle():
    primitive_set = build_qet_primitives(theta=0.1, phi0=0.2, phi1=0.3, phi2=0.4)
    assert isinstance(primitive_set, DecoderPrimitiveSet)

    actual, special = build_qet_kernel_maps(theta=0.1, phi0=0.2, phi1=0.3, phi2=0.4)
    assert set(actual) == {"X", "Y", "Z"}
    assert set(special) == {"X", "Y", "Z"}


class _FakeDense:
    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def toarray(self):
        return self._arr


class _FakeDemMatrix:
    def __init__(self, check_matrix, observables_matrix, priors):
        self.check_matrix = _FakeDense(check_matrix)
        self.observables_matrix = _FakeDense(observables_matrix)
        self.priors = priors


class _ConfidenceGurobi(ConfidenceDecoder):
    instances = []

    def __init__(self, dem):
        self.dem = dem
        self.instances.append(self)

    def _decode(self, detector_bits):
        return np.zeros(self.dem.num_observables, dtype=bool)

    def decode_with_confidence(self, detector_bits):
        return self._decode(detector_bits), np.float64(2.5)


class _FakeTask:
    detector_error_model = stim.DetectorErrorModel("""
        error(0.1) D0
        error(0.2) D0 L0
        error(0.3) D3 L1
        """)


class _ChunkResult:
    def __init__(self, detectors: np.ndarray, observables: np.ndarray):
        self.detectors = detectors.tolist()
        self.observables = observables.tolist()


class _StaticTask:
    def __init__(self, dataset: BasisDataset):
        self._dataset = dataset

    def run(self, shots: int, with_noise: bool = True, *, run_detectors: bool = False):
        assert run_detectors
        return _ChunkResult(
            self._dataset.detectors[:shots],
            self._dataset.observables[:shots],
        )


class _ChunkTask:
    def __init__(self, dataset: BasisDataset):
        self._dataset = dataset
        self._offset = 0

    def run(self, shots: int, with_noise: bool = True, *, run_detectors: bool = False):
        assert run_detectors
        start = self._offset
        stop = start + int(shots)
        self._offset = stop
        return _ChunkResult(
            self._dataset.detectors[start:stop],
            self._dataset.observables[start:stop],
        )


def _basis_task_map(
    bits_by_basis: dict[str, list[int]],
    *,
    factory_bits_by_basis: dict[str, list[int]] | None = None,
    detector_rows: np.ndarray | None = None,
) -> dict[str, _StaticTask]:
    tasks = {}
    for basis, bits in bits_by_basis.items():
        output = np.asarray(bits, dtype=np.uint8).reshape(-1, 1)
        if factory_bits_by_basis is None:
            observables = output
        else:
            factory = np.asarray(
                factory_bits_by_basis[basis],
                dtype=np.uint8,
            ).reshape(-1, 1)
            observables = np.concatenate([output, factory], axis=1)
        detectors = (
            np.zeros((len(output), 1), dtype=np.uint8)
            if detector_rows is None
            else detector_rows[: len(output)]
        )
        tasks[basis] = _StaticTask(BasisDataset(detectors, observables))
    return tasks


def _workflow_config(
    *,
    layout: SyndromeLayout = SyndromeLayout(
        output_detector_count=1,
        output_observable_count=1,
    ),
) -> MSDDecoderWorkflowConfig:
    return MSDDecoderWorkflowConfig(
        mld_train_shots=4,
        eval_shots=4,
        target_bloch_vector=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        theta=0.1,
        phi=0.2,
        lam=0.3,
        decoder_primitive_set=build_msd_primitives(0.1, 0.2, 0.3),
        valid_factory_targets=[0],
        num_logical_qubits=5,
        binary_precision=4,
        basis_labels=("X", "Y", "Z"),
        sign_vector=(1.0, 1.0, 1.0),
        layout=layout,
    )


def test_workflow_config_normalizes_inputs():
    config = _workflow_config()

    assert np.asarray(config.target_bloch_vector).tolist() == [1.0, 0.0, 0.0]
    assert np.asarray(config.valid_factory_targets).tolist() == [[0]]
    assert config.resolved_mld_rank_train_shots == config.mld_train_shots


def test_workflow_tomography_kernels_builders_return_basis_maps():
    config = _workflow_config()

    msd_tomography_kernels = build_msd_tomography_kernels(config)
    injected_tomography_kernels = build_injected_tomography_kernels(config)

    assert isinstance(msd_tomography_kernels.actual, dict)
    assert set(msd_tomography_kernels.actual) == {"X", "Y", "Z"}
    assert set(msd_tomography_kernels._special) == {"X", "Y", "Z"}
    assert set(injected_tomography_kernels.actual) == {"X", "Y", "Z"}
    assert set(injected_tomography_kernels._special) == {"X", "Y", "Z"}


def test_workflow_tomography_tasks_builder_compiles_msd_tomography_tasks():
    sim = GeminiLogicalSimulator()
    config = _workflow_config()
    tomography_kernels = build_msd_tomography_kernels(config)

    tomography_tasks = build_msd_tomography_tasks(sim, config, tomography_kernels)

    assert isinstance(tomography_tasks, TomographyTasks)
    assert set(tomography_tasks.actual) == {"X", "Y", "Z"}
    assert set(tomography_tasks._special) == {"X", "Y", "Z"}
    assert not hasattr(tomography_tasks, "special")


def test_workflow_sample_actual_data_uses_config_sampling():
    config = _workflow_config()
    task_map = _basis_task_map(
        {"X": [0, 0, 0, 0], "Y": [1, 1, 1, 1], "Z": [0, 0, 0, 0]},
    )
    tomography_tasks = TomographyTasks(
        actual=cast(dict[str, DemoTask[object]], task_map),
        _special=cast(dict[str, DemoTask[object]], task_map),
    )

    data = sample_actual_data(tomography_tasks, config)

    assert set(data) == {"X", "Y", "Z"}
    assert data["X"].observables.shape == (4, 1)


def test_workflow_mld_suite_trains_from_tomography_tasks():
    config = _workflow_config()
    dataset = BasisDataset(
        detectors=np.zeros((4, 2), dtype=np.uint8),
        observables=np.zeros((4, 2), dtype=np.uint8),
    )
    task_map = {basis: _StaticTask(dataset) for basis in ("X", "Y", "Z")}
    tomography_tasks = TomographyTasks(
        actual=cast(dict[str, DemoTask[object]], task_map),
        _special=cast(dict[str, DemoTask[object]], task_map),
    )

    decoders = train_mld_decoder_suite(
        tomography_tasks,
        config,
        table_decoder_cls=SparseTableDecoder,
    )

    assert set(decoders) == {"X", "Y", "Z"}
    assert all(
        decoder.factory_score_mode == "mld_output_fidelity"
        for decoder in decoders.values()
    )


def test_workflow_mle_suite_builds_per_basis_decoders(monkeypatch):
    monkeypatch.setattr(
        "demo.msd_utils.standard.dem.detector_error_model_to_check_matrices",
        lambda *args, **kwargs: _FakeDemMatrix(
            check_matrix=np.array([[1, 0], [0, 1], [1, 1], [0, 1]], dtype=int),
            observables_matrix=np.array([[1, 0], [0, 1]], dtype=int),
            priors=np.array([0.1, 0.2], dtype=float),
        ),
    )
    tomography_tasks = TomographyTasks(
        actual=cast(dict[str, DemoTask[object]], {"X": _FakeTask()}),
        _special=cast(dict[str, DemoTask[object]], {"X": _FakeTask()}),
    )

    decoders = build_mle_decoder_suite(
        tomography_tasks,
        gurobi_decoder_cls=_ConfidenceGurobi,
    )

    assert set(decoders) == {"X"}
    assert decoders["X"].factory_score_mode == "confidence"


def test_workflow_evaluate_decoder_curves_supports_multiple_suites():
    config = _workflow_config()
    dataset = BasisDataset(
        detectors=np.zeros((4, 2), dtype=np.uint8),
        observables=np.zeros((4, 2), dtype=np.uint8),
    )
    actual_data = {basis: dataset for basis in ("X", "Y", "Z")}

    def make_adapter(score: float) -> DecoderAdapter:
        def decode_factory(key: int):
            return (0,), score

        def decode_full(key: int):
            return (0,)

        return DecoderAdapter(
            full_decoder=None,
            factory_decoder=None,
            decode_factory=decode_factory,
            decode_full=decode_full,
            factory_score_mode="score",
        )

    suite = {basis: make_adapter(1.0) for basis in ("X", "Y", "Z")}

    curves = evaluate_decoder_curves(
        actual_data,
        {"MLD": suite, "MLE": suite},
        config,
        curve_options=DecoderCurveOptions(
            threshold_points=2,
            min_accepted_per_basis=1,
        ),
    )

    assert set(curves) == {"MLD", "MLE"}
    assert all("accepted_fraction" in curve for curve in curves.values())


def test_workflow_evaluate_decoder_curves_applies_shared_options_and_overrides(
    monkeypatch,
):
    import bloqade.gemini.decoding.workflow as workflows

    config = _workflow_config()
    calls: dict[str, tuple[int, int, str]] = {}

    def fake_evaluate_curve(*args, **kwargs):
        calls[str(kwargs["metric"])] = (
            int(kwargs["threshold_points"]),
            int(kwargs["min_accepted_per_basis"]),
            str(kwargs["selection_mode"]),
        )
        return {
            "accepted_fraction": np.array([], dtype=np.float64),
            "fidelity": np.array([], dtype=np.float64),
            "credible": np.empty((0, 2), dtype=np.float64),
        }

    monkeypatch.setattr(workflows, "evaluate_curve", fake_evaluate_curve)
    suite = cast(dict[str, DecoderAdapter], {})

    curves = workflows.evaluate_decoder_curves(
        {},
        {"MLD": suite, "MLE": suite},
        config,
        curve_options=DecoderCurveOptions(
            threshold_points=24,
            min_accepted_per_basis=2,
            selection_mode="threshold",
        ),
        curve_option_overrides={
            "MLE": DecoderCurveOptions(
                threshold_points=48,
                min_accepted_per_basis=3,
                selection_mode="pattern_rank",
            ),
        },
        log=False,
    )

    assert set(curves) == {"MLD", "MLE"}
    assert calls == {
        "MLD": (24, 2, "threshold"),
        "MLE": (48, 3, "pattern_rank"),
    }


def test_workflow_injected_baseline_and_plot_helpers():
    config = _workflow_config()
    task_map = _basis_task_map(
        {"X": [0, 0, 0, 0], "Y": [0, 0, 0, 0], "Z": [0, 0, 0, 0]},
    )
    tomography_tasks = TomographyTasks(
        actual=cast(dict[str, DemoTask[object]], task_map),
        _special=cast(dict[str, DemoTask[object]], task_map),
    )

    summary = evaluate_injected_baseline(
        tomography_tasks,
        config,
        table_decoder_cls=SparseTableDecoder,
        raw=True,
    )
    fig, ax = plot_decoder_curves(
        {
            "MLD": {
                "accepted_fraction": np.array([0.5, 1.0]),
                "fidelity": np.array([0.7, 0.8]),
                "credible": np.array([[0.6, 0.8], [0.7, 0.9]]),
            }
        },
        injected_summary=summary,
    )

    assert fig is ax.figure
    assert len(ax.lines) >= 2


def test_infer_factory_target_selects_branch_near_expected_acceptance():
    task_map = _basis_task_map(
        {
            "X": [0, 0, 0, 0],
            "Y": [0, 0, 0, 0],
            "Z": [0, 0, 0, 0],
        },
        factory_bits_by_basis={
            "X": [0, 0, 1, 1],
            "Y": [0, 0, 1, 1],
            "Z": [0, 0, 1, 1],
        },
    )

    target = infer_factory_target(
        task_map,
        shots=4,
        basis_labels=("X", "Y", "Z"),
        ideal_factory_acceptance=0.5,
    )

    assert target.tolist() == [0]


def test_infer_distilled_sign_vector_aligns_noiseless_bloch_vector():
    task_map = _basis_task_map(
        {
            "X": [0, 0, 0, 0],
            "Y": [1, 1, 1, 1],
            "Z": [0, 0, 0, 0],
        },
        factory_bits_by_basis={
            "X": [0, 0, 0, 0],
            "Y": [0, 0, 0, 0],
            "Z": [0, 0, 0, 0],
        },
    )

    sign = infer_distilled_sign_vector(
        task_map,
        valid_factory_targets=np.array([[0]], dtype=np.uint8),
        shots=4,
        basis_labels=("X", "Y", "Z"),
        target_bloch=DEFAULT_TARGET_BLOCH,
    )

    assert sign.tolist() == [1.0, -1.0, 1.0]


def test_naive_injected_summary_postselects_zero_detectors():
    detector_rows = np.array([[0], [1], [0], [1]], dtype=np.uint8)
    task_map = _basis_task_map(
        {"X": [0, 1, 0, 1], "Y": [0, 1, 0, 1], "Z": [0, 1, 0, 1]},
        detector_rows=detector_rows,
    )

    summary = naive_injected_summary(
        task_map,
        sign_vector=(1.0, 1.0, 1.0),
        shots=4,
        require_zero_detectors=True,
        basis_labels=("X", "Y", "Z"),
        target_bloch=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        min_accepted_per_basis=1,
    )

    assert summary["accepted_fraction"] == pytest.approx(0.5)
    assert summary["point"] == pytest.approx(1.0)


def test_naive_distilled_summary_postselects_factory_targets():
    task_map = _basis_task_map(
        {"X": [0, 1, 0, 1], "Y": [0, 1, 0, 1], "Z": [0, 1, 0, 1]},
        factory_bits_by_basis={
            "X": [0, 1, 0, 1],
            "Y": [0, 1, 0, 1],
            "Z": [0, 1, 0, 1],
        },
    )

    summary = naive_distilled_summary(
        task_map,
        valid_factory_targets=np.array([[0]], dtype=np.uint8),
        sign_vector=(1.0, 1.0, 1.0),
        shots=4,
        basis_labels=("X", "Y", "Z"),
        min_accepted_per_basis=1,
    )

    assert summary["accepted_fraction"] == pytest.approx(0.5)
    assert summary["valid_factory_targets"] == ((0,),)


def test_injected_baseline_raw_path_uses_observable_bits_directly():
    task_map = _basis_task_map(
        {"X": [0, 0, 0, 0], "Y": [1, 1, 1, 1], "Z": [0, 0, 0, 0]},
    )

    summary = injected_baseline(
        task_map,
        eval_shots=4,
        table_decoder_cls=SparseTableDecoder,
        sign_vector=(1.0, -1.0, 1.0),
        raw=True,
        basis_labels=("X", "Y", "Z"),
    )

    assert summary["bloch"] == pytest.approx((1.0, 1.0, 1.0))


def test_table_decoder_with_confidence_returns_syndrome_score():
    decoder = SparseTableDecoder.from_det_obs_shots(
        stim.DetectorErrorModel("error(0.5) D0 L0"),
        np.array([[0, 0], [1, 1], [1, 1]], dtype=np.uint8),
    )
    wrapped = TableDecoderWithConfidence(
        decoder=decoder,
        syndrome_confidence=np.array([0.25, 0.75], dtype=np.float64),
    )

    correction, score = wrapped.decode_with_confidence(
        np.array([1], dtype=np.bool_),
    )

    assert correction.tolist() == [True]
    assert score == pytest.approx(0.75)


def test_sub_detector_error_model_preserves_observable_distinctions():
    dem = stim.DetectorErrorModel("""
        error(0.1) D0
        error(0.2) D0 L0
        """)

    projected = sub_detector_error_model(dem, [0], [0])

    assert projected.num_detectors == 1
    assert projected.num_observables == 1
    assert "error(0.1) D0" in str(projected)
    assert "error(0.2) D0 L0" in str(projected)


def test_sub_detector_error_model_composes_duplicate_projected_errors():
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 D1 L0
        error(0.2) D0 D2 L0
        """)

    projected = sub_detector_error_model(dem, [0], [0])

    assert projected.num_errors == 1
    assert "error(0.26) D0 L0" in str(projected)


def test_build_mle_decoders_uses_confidence_decoder_api():
    _ConfidenceGurobi.instances = []

    adapter = build_mle_decoders(_FakeTask(), gurobi_decoder_cls=_ConfidenceGurobi)
    _, score = adapter.decode_factory(0)
    assert adapter.factory_score_mode == "confidence"
    assert score == pytest.approx(2.5)
    full_dem = _ConfidenceGurobi.instances[0].dem
    factory_dem = _ConfidenceGurobi.instances[1].dem
    assert "error(0.1) D0" in str(full_dem)
    assert "error(0.2) D0 L0" in str(full_dem)
    assert str(factory_dem).startswith("error(0.3) D0 L0")


def test_streaming_sparse_mld_decoder_pair_matches_batch():
    layout = SyndromeLayout(output_detector_count=1, output_observable_count=1)
    dataset = BasisDataset(
        detectors=np.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [1, 0, 0],
                [1, 1, 0],
                [1, 1, 1],
                [0, 1, 1],
            ],
            dtype=np.uint8,
        ),
        observables=np.array(
            [
                [0, 0],
                [1, 1],
                [0, 1],
                [1, 0],
                [1, 1],
                [0, 0],
            ],
            dtype=np.uint8,
        ),
    )

    batch_full, batch_factory = train_mld_decoder_pair(
        dataset,
        table_decoder_cls=SparseTableDecoder,
        layout=layout,
    )
    stream_full, stream_factory = train_mld_decoder_pair_from_task(
        _ChunkTask(dataset),
        len(dataset.detectors),
        table_decoder_cls=SparseTableDecoder,
        layout=layout,
        chunk_size=2,
    )

    test_full = np.array(
        [[0, 0, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1]],
        dtype=np.uint8,
    )
    test_factory = test_full[:, 1:]
    assert np.array_equal(
        batch_full.decode(test_full.astype(bool)),
        stream_full.decode(test_full.astype(bool)),
    )
    assert np.array_equal(
        batch_factory.decode(test_factory.astype(bool)),
        stream_factory.decode(test_factory.astype(bool)),
    )


def test_streaming_mld_ancilla_scores_match_batch():
    layout = SyndromeLayout(output_detector_count=1, output_observable_count=1)

    def make_dataset(seed: int) -> BasisDataset:
        rng = np.random.default_rng(seed)
        detectors = rng.integers(0, 2, size=(32, 3), dtype=np.uint8)
        observables = np.zeros((32, 2), dtype=np.uint8)
        observables[:, 0] = rng.integers(0, 2, size=32, dtype=np.uint8)
        observables[:, 1] = detectors[:, 2]
        return BasisDataset(detectors=detectors, observables=observables)

    ranking_data = {basis: make_dataset(i) for i, basis in enumerate("XYZ", start=1)}
    decoder_pairs = {
        basis: train_mld_decoder_pair(
            dataset,
            table_decoder_cls=SparseTableDecoder,
            layout=layout,
        )
        for basis, dataset in ranking_data.items()
    }

    batch_scores = estimate_mld_ancilla_scores(
        decoder_pairs,
        ranking_data,
        valid_factory_targets=np.array([[0]], dtype=np.uint8),
        basis_labels=("X", "Y", "Z"),
        sign_vector=(1.0, -1.0, 1.0),
        target_bloch=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        layout=layout,
    )
    streamed_scores = estimate_mld_ancilla_scores_from_tasks(
        decoder_pairs,
        {basis: _ChunkTask(dataset) for basis, dataset in ranking_data.items()},
        32,
        valid_factory_targets=np.array([[0]], dtype=np.uint8),
        basis_labels=("X", "Y", "Z"),
        sign_vector=(1.0, -1.0, 1.0),
        target_bloch=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        layout=layout,
        chunk_size=7,
    )

    assert np.allclose(batch_scores, streamed_scores, equal_nan=True)


def test_evaluate_curve_returns_monotone_acceptance():
    dataset = BasisDataset(
        detectors=np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 1, 1],
            ],
            dtype=np.uint8,
        ),
        observables=np.array(
            [
                [0, 0],
                [0, 0],
                [1, 0],
                [0, 0],
            ],
            dtype=np.uint8,
        ),
    )

    def make_adapter():
        def decode_factory(key: int):
            return (0,), 1.0 - 0.25 * (int(key) & 1)

        def decode_full(key: int):
            return (0, 0)

        return DecoderAdapter(
            full_decoder=None,
            factory_decoder=None,
            decode_factory=decode_factory,
            decode_full=decode_full,
            factory_score_mode="logical_gap",
        )

    curves = evaluate_curve(
        {"X": dataset, "Y": dataset, "Z": dataset},
        {"X": make_adapter(), "Y": make_adapter(), "Z": make_adapter()},
        binary_precision=4,
        threshold_points=3,
        metric="test",
        valid_factory_targets=np.array([[0]], dtype=np.uint8),
        sign_vector=(1.0, 1.0, 1.0),
        min_accepted_per_basis=1,
    )

    accepted = curves["accepted_fraction"]
    assert accepted.ndim == 1
    assert np.all(np.diff(accepted) >= -1e-12)


def test_evaluate_curve_cached_generic_threshold_matches_legacy_loop():
    layout = SyndromeLayout(output_detector_count=1, output_observable_count=1)
    dataset = BasisDataset(
        detectors=np.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
            ],
            dtype=np.uint8,
        ),
        observables=np.array(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [1, 1],
                [0, 0],
                [1, 1],
            ],
            dtype=np.uint8,
        ),
    )
    actual_data = {basis: dataset for basis in "XYZ"}

    def make_adapter(score_offset: float):
        def decode_factory(key):
            packed = int(key)
            return (packed & 1,), float(score_offset + packed)

        def decode_full(key):
            packed = int(key)
            return (packed & 1,)

        return DecoderAdapter(
            full_decoder=None,
            factory_decoder=None,
            decode_factory=decode_factory,
            decode_full=decode_full,
            factory_score_mode="logical_gap",
        )

    decoder_map = {
        "X": make_adapter(0.0),
        "Y": make_adapter(0.5),
        "Z": make_adapter(1.0),
    }

    curves = evaluate_curve(
        actual_data,
        decoder_map,
        binary_precision=4,
        threshold_points=5,
        metric="test",
        valid_factory_targets=np.array([[0]], dtype=np.uint8),
        sign_vector=(1.0, -1.0, 1.0),
        target_bloch=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        basis_labels=("X", "Y", "Z"),
        min_accepted_per_basis=1,
        threshold_policy="quantile",
        selection_mode="threshold",
        layout=layout,
        uncertainty_backend="wilson",
    )

    def legacy_curve():
        thresholds_source = []
        for basis in "XYZ":
            anc_det, anc_obs = split_factory_bits(
                actual_data[basis].detectors,
                actual_data[basis].observables,
                layout=layout,
            )
            for a_det, a_obs in zip(anc_det, anc_obs, strict=True):
                anc_flip, score = decoder_map[basis].decode_factory(
                    int(pack_boolean_array(a_det)[0])
                )
                anc_flip = np.asarray(anc_flip, dtype=np.uint8)
                if np.isfinite(score) and np.array_equal(
                    a_obs ^ anc_flip, np.array([0], dtype=np.uint8)
                ):
                    thresholds_source.append(score)

        thresholds = np.unique(
            np.quantile(
                np.asarray(thresholds_source, dtype=np.float64),
                np.linspace(0.0, 1.0, 5),
            )
        )
        accepted = []
        fidelity = []
        credible = []

        for threshold in thresholds:
            corrected = {}
            total_kept = 0
            total_shots = 0
            for basis in "XYZ":
                anc_det, anc_obs = split_factory_bits(
                    actual_data[basis].detectors,
                    actual_data[basis].observables,
                    layout=layout,
                )
                corrected_bits = []
                for det, obs, a_det, a_obs in zip(
                    actual_data[basis].detectors,
                    actual_data[basis].observables,
                    anc_det,
                    anc_obs,
                    strict=True,
                ):
                    anc_flip, score = decoder_map[basis].decode_factory(
                        int(pack_boolean_array(a_det)[0])
                    )
                    anc_flip = np.asarray(anc_flip, dtype=np.uint8)
                    if not np.array_equal(
                        a_obs ^ anc_flip, np.array([0], dtype=np.uint8)
                    ):
                        continue
                    if score < threshold:
                        continue
                    full_flip = np.asarray(
                        decoder_map[basis].decode_full(int(pack_boolean_array(det)[0])),
                        dtype=np.uint8,
                    )
                    corrected_bits.append(int(obs[0] ^ full_flip[0]))
                corrected[basis] = np.asarray(corrected_bits, dtype=np.uint8)
                total_kept += len(corrected[basis])
                total_shots += len(actual_data[basis].observables)

            if min(len(corrected[basis]) for basis in "XYZ") < 1:
                continue

            summary = fidelity_from_counts(
                corrected["X"],
                corrected["Y"],
                corrected["Z"],
                256,
                sign_vector=(1.0, -1.0, 1.0),
                target_bloch=np.array([0.0, 0.0, 1.0], dtype=np.float64),
                uncertainty_backend="wilson",
            )
            accepted.append(total_kept / total_shots)
            fidelity.append(summary["median"])
            credible.append((summary["low"], summary["high"]))

        accepted = np.asarray(accepted, dtype=np.float64)
        fidelity = np.asarray(fidelity, dtype=np.float64)
        credible = np.asarray(credible, dtype=np.float64)
        if len(accepted):
            order = np.argsort(accepted)
            accepted = accepted[order]
            fidelity = fidelity[order]
            credible = credible[order]
        return accepted, fidelity, credible

    legacy_accepted, legacy_fidelity, legacy_credible = legacy_curve()
    assert np.allclose(curves["accepted_fraction"], legacy_accepted)
    assert np.allclose(curves["fidelity"], legacy_fidelity)
    assert np.allclose(curves["credible"], legacy_credible)


def test_evaluate_mld_curve_uses_cumulative_pattern_ordering():
    dataset = BasisDataset(
        detectors=np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
            ],
            dtype=np.uint8,
        ),
        observables=np.array(
            [
                [0, 0],
                [0, 0],
                [1, 0],
                [1, 0],
            ],
            dtype=np.uint8,
        ),
    )

    def make_adapter():
        def decode_factory(key: int):
            # Deliberately misleading score: the lower-fidelity pattern gets the
            # higher decoder score. The legacy MLD evaluator ranks by this score
            # table, not by recomputing a separate fidelity-based ordering.
            return (0,), 0.5 if (int(key) & 1) == 0 else 1.0

        def decode_full(key: int):
            return (0, 0)

        return DecoderAdapter(
            full_decoder=None,
            factory_decoder=None,
            decode_factory=decode_factory,
            decode_full=decode_full,
            factory_score_mode="mld_output_fidelity",
        )

    curves = evaluate_mld_curve(
        {"X": dataset, "Y": dataset, "Z": dataset},
        {"X": make_adapter(), "Y": make_adapter(), "Z": make_adapter()},
        binary_precision=4,
        valid_factory_targets=np.array([[0]], dtype=np.uint8),
        sign_vector=(1.0, 1.0, 1.0),
        min_accepted_per_basis=1,
    )

    accepted = curves["accepted_fraction"]
    fidelity = curves["fidelity"]
    assert accepted.ndim == 1
    assert np.all(np.diff(accepted) >= -1e-12)
    assert accepted[0] == pytest.approx(0.5)
    assert accepted[-1] == pytest.approx(1.0)
    assert fidelity[0] <= fidelity[-1]


def test_evaluate_curve_pattern_rank_matches_legacy_mld_ordering():
    dataset = BasisDataset(
        detectors=np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
            ],
            dtype=np.uint8,
        ),
        observables=np.array(
            [
                [0, 0],
                [0, 0],
                [1, 0],
                [1, 0],
            ],
            dtype=np.uint8,
        ),
    )

    def make_adapter():
        def decode_factory(key: int):
            return (0,), 0.5 if (int(key) & 1) == 0 else 1.0

        def decode_full(key: int):
            return (0, 0)

        return DecoderAdapter(
            full_decoder=None,
            factory_decoder=None,
            decode_factory=decode_factory,
            decode_full=decode_full,
            factory_score_mode="mld_output_fidelity",
        )

    legacy_curves = evaluate_mld_curve(
        {"X": dataset, "Y": dataset, "Z": dataset},
        {"X": make_adapter(), "Y": make_adapter(), "Z": make_adapter()},
        binary_precision=4,
        valid_factory_targets=np.array([[0]], dtype=np.uint8),
        sign_vector=(1.0, 1.0, 1.0),
        min_accepted_per_basis=1,
    )

    curves = evaluate_curve(
        {"X": dataset, "Y": dataset, "Z": dataset},
        {"X": make_adapter(), "Y": make_adapter(), "Z": make_adapter()},
        binary_precision=4,
        threshold_points=4,
        metric="test",
        valid_factory_targets=np.array([[0]], dtype=np.uint8),
        sign_vector=(1.0, 1.0, 1.0),
        min_accepted_per_basis=1,
        selection_mode="pattern_rank",
    )

    accepted = curves["accepted_fraction"]
    # fidelity = curves["fidelity"]
    assert accepted.ndim == 1
    assert np.all(np.diff(accepted) >= -1e-12)
    assert accepted[0] == pytest.approx(0.5)
    assert accepted[-1] == pytest.approx(1.0)
    assert np.allclose(curves["accepted_fraction"], legacy_curves["accepted_fraction"])
    assert np.allclose(curves["fidelity"], legacy_curves["fidelity"])


def test_evaluate_curve_sparse_mld_threshold_matches_generic_threshold_path():
    dataset = BasisDataset(
        detectors=np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 1, 0],
            ],
            dtype=np.uint8,
        ),
        observables=np.array(
            [
                [0, 0],
                [1, 0],
                [0, 0],
                [1, 0],
                [0, 1],
                [1, 1],
            ],
            dtype=np.uint8,
        ),
    )

    def make_adapter(score_mode: str):
        def decode_factory(key: int):
            if (int(key) & 1) == 0:
                return (0,), 0.8
            return (0,), 0.4

        def decode_full(key: int):
            last = (int(key) >> 3) & 1
            return (last, 0)

        return DecoderAdapter(
            full_decoder=None,
            factory_decoder=None,
            decode_factory=decode_factory,
            decode_full=decode_full,
            factory_score_mode=score_mode,
        )

    sparse_curves = evaluate_curve(
        {"X": dataset, "Y": dataset, "Z": dataset},
        {
            "X": make_adapter("mld_output_fidelity"),
            "Y": make_adapter("mld_output_fidelity"),
            "Z": make_adapter("mld_output_fidelity"),
        },
        binary_precision=4,
        threshold_points=3,
        metric="test",
        valid_factory_targets=np.array([[0]], dtype=np.uint8),
        sign_vector=(1.0, 1.0, 1.0),
        min_accepted_per_basis=1,
        threshold_policy="quantile",
    )

    generic_curves = evaluate_curve(
        {"X": dataset, "Y": dataset, "Z": dataset},
        {
            "X": make_adapter("logical_gap"),
            "Y": make_adapter("logical_gap"),
            "Z": make_adapter("logical_gap"),
        },
        binary_precision=4,
        threshold_points=3,
        metric="test",
        valid_factory_targets=np.array([[0]], dtype=np.uint8),
        sign_vector=(1.0, 1.0, 1.0),
        min_accepted_per_basis=1,
        threshold_policy="quantile",
    )

    assert np.allclose(
        sparse_curves["accepted_fraction"], generic_curves["accepted_fraction"]
    )
    assert np.allclose(sparse_curves["fidelity"], generic_curves["fidelity"])
    assert np.allclose(sparse_curves["credible"], generic_curves["credible"])


def test_train_mld_decoder_pair_uses_only_output_observables_for_full_decoder():
    dataset = BasisDataset(
        detectors=np.zeros((8, 27), dtype=np.uint8),
        observables=np.zeros((8, 9), dtype=np.uint8),
    )

    full_decoder, factory_decoder = train_mld_decoder_pair(
        dataset,
        table_decoder_cls=SparseTableDecoder,
    )

    assert isinstance(full_decoder, SparseTableDecoder)
    assert isinstance(factory_decoder, SparseTableDecoder)

    assert full_decoder.num_detectors == 27
    assert full_decoder.num_observables == 1

    assert factory_decoder.num_detectors == 24
    assert factory_decoder.num_observables == 8


def test_postselection_experiment_tomography_result_uses_ranked_counts():
    per_basis_tables = {
        "X": (
            np.array([0.5, 0.9], dtype=np.float64),
            np.array([1, 5], dtype=np.int64),
            np.array([0, 0], dtype=np.int64),
        ),
        "Y": (
            np.array([0.4, 0.8], dtype=np.float64),
            np.array([2, 7], dtype=np.int64),
            np.array([0, 0], dtype=np.int64),
        ),
        "Z": (
            np.array([0.3, 0.7], dtype=np.float64),
            np.array([3, 11], dtype=np.int64),
            np.array([0, 0], dtype=np.int64),
        ),
    }
    score_weights = np.array([3, 2, 1, 11, 7, 5], dtype=np.int64)
    exp = object.__new__(PostSelectionExperiment)
    exp.postselection_exp_cache = SimpleNamespace(
        decoded_results=(
            per_basis_tables,
            np.array([0.3, 0.4, 0.5, 0.7, 0.8, 0.9], dtype=np.float64),
            score_weights,
            int(np.sum(score_weights)),
        )
    )

    result = exp.tomography_result(0.69, "wilson")

    assert isinstance(result, TomographyResult)
    assert np.array_equal(result.zero_counts, np.array([5, 7, 11]))
    assert np.array_equal(result.one_counts, np.array([0, 0, 0]))
    assert result.fidelity_bloch(np.array([1.0, 0.0, 0.0]))["point"] == 1.0


def test_notebooks_import_shared_msd_utils():
    for artifact in [
        Path("demo/msd_reprod_bloqade_decoders_workflow.py"),
        Path("demo/qet_reprod_bloqade_decoders_workflow.py"),
        Path("demo/msd_reprod_bloqade_decoders_workflow.ipynb"),
        Path("demo/qet_reprod_bloqade_decoders_workflow.ipynb"),
    ]:
        if artifact.suffix == ".ipynb":
            nb = json.loads(artifact.read_text())
            joined = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])
        else:
            joined = artifact.read_text()
        assert "demo.msd_utils" in joined
