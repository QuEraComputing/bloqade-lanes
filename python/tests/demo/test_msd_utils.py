from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from bloqade import squin
from bloqade.lanes import GeminiLogicalSimulator

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from demo.msd_utils.circuits import (
    DecoderPrimitiveSet,
    build_decoder_kernel_bundle,
    build_measurement_maps,
    build_naive_kernel_bundle,
    build_task,
    make_noisy_steane7_initializer,
)
from demo.msd_utils.common import SyndromeLayout
from demo.msd_utils.core import (
    BasisDataset,
    fidelity_from_counts,
    pack_boolean_array,
    split_factory_bits,
)
from demo.msd_utils.decoder_classes import SparseTableDecoder
from demo.msd_utils.decoders import (
    DecoderAdapter,
    build_mle_decoders,
    estimate_mld_ancilla_scores,
    estimate_mld_ancilla_scores_from_tasks,
    evaluate_curve,
    evaluate_mld_curve,
    train_mld_decoder_pair,
    train_mld_decoder_pair_from_task,
)
from demo.msd_utils.qet import build_qet_kernel_maps, build_qet_primitives


def test_fidelity_from_counts_returns_ordered_interval():
    summary = fidelity_from_counts(
        np.array([0, 0, 1, 0], dtype=np.uint8),
        np.array([0, 1, 0, 0], dtype=np.uint8),
        np.array([0, 0, 0, 1], dtype=np.uint8),
        posterior_samples=256,
    )
    assert set(summary) >= {"point", "median", "low", "high", "bloch"}
    assert summary["low"] <= summary["median"] <= summary["high"]
    assert len(summary["bloch"]) == 3


def test_fidelity_from_counts_realistic_interval_is_finite_and_noncollapsed():
    x_bits = np.array([0] * 3060 + [1] * 940, dtype=np.uint8)
    y_bits = np.array([0] * 3040 + [1] * 960, dtype=np.uint8)
    z_bits = np.array([0] * 3090 + [1] * 910, dtype=np.uint8)

    summary = fidelity_from_counts(
        x_bits,
        y_bits,
        z_bits,
        posterior_samples=20_000,
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


def test_kernel_builders_return_expected_basis_maps():
    naive = build_naive_kernel_bundle(0.1, 0.2, 0.3)
    decoder = build_decoder_kernel_bundle(0.1, 0.2, 0.3)
    assert set(naive.distilled) == {"X", "Y", "Z"}
    assert set(naive.injected) == {"X", "Y", "Z"}
    assert set(decoder.actual) == {"X", "Y", "Z"}
    assert set(decoder.special) == {"X", "Y", "Z"}
    assert set(decoder.injected) == {"X", "Y", "Z"}


def test_prefix_prepare_uses_tsim_prefix_and_remains_deterministic():
    sim = GeminiLogicalSimulator()
    noisy_initializer = make_noisy_steane7_initializer(sim)
    m2dets, m2obs = build_measurement_maps(5)
    decoder = build_decoder_kernel_bundle(
        0.1,
        0.2,
        0.3,
        special_kernel_strategy="prefix_prepare",
    )

    demo_task = build_task(
        sim,
        decoder.special["X"],
        m2dets=m2dets,
        m2obs=m2obs,
        noisy_initializer=noisy_initializer,
        append_measurements=False,
    )

    assert "prepare_inverse" not in str(demo_task.task.physical_squin_kernel)

    result = demo_task.task.run(16, with_noise=False, run_detectors=True)
    assert len(np.unique(np.asarray(result.observables, dtype=np.uint8), axis=0)) == 1
    assert len(np.unique(np.asarray(result.detectors, dtype=np.uint8), axis=0)) == 1


def test_demo_task_clifft_backend_matches_result_shapes():
    pytest.importorskip("clifft")

    sim = GeminiLogicalSimulator()
    noisy_initializer = make_noisy_steane7_initializer(sim)
    m2dets, m2obs = build_measurement_maps(5)
    decoder = build_decoder_kernel_bundle(
        0.1,
        0.2,
        0.3,
        special_kernel_strategy="prefix_prepare",
    )
    demo_task = build_task(
        sim,
        decoder.special["X"],
        m2dets=m2dets,
        m2obs=m2obs,
        noisy_initializer=noisy_initializer,
        append_measurements=False,
    )

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


def test_decoder_kernel_bundle_accepts_variadic_primitive_builder():
    captured: list[tuple[float, ...]] = []

    @squin.kernel
    def sentinel(reg):
        return

    def build_primitives(*args: float) -> DecoderPrimitiveSet:
        captured.append(args)
        return DecoderPrimitiveSet(
            state_injection_circuit=sentinel,
            logical_circuit=sentinel,
            logical_circuit_inverse=sentinel,
        )

    decoder = build_decoder_kernel_bundle(
        0.1,
        0.2,
        0.3,
        0.4,
        num_logical_qubits=9,
        build_primitives=build_primitives,
        injected_prep_args=None,
        special_kernel_strategy="compiled_inverse_prefix",
    )

    assert captured == [(0.1, 0.2, 0.3, 0.4)]
    assert set(decoder.actual) == {"X", "Y", "Z"}
    assert set(decoder.special) == {"X", "Y", "Z"}
    assert decoder.injected == {}


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


class _NewStyleGurobi:
    def __init__(self, dem):
        self.dem = dem

    def decode(
        self,
        detector_bits,
        verbose=False,
        return_weights=False,
        return_logical_gap=False,
    ):
        correction = np.zeros(self.dem.num_observables, dtype=bool)
        if return_logical_gap:
            return correction, np.array([2.5], dtype=float)
        if return_weights:
            return correction, np.array([1.0], dtype=float)
        return correction


class _OldStyleGurobi:
    def __init__(self, dem):
        self.dem = dem

    def decode(self, detector_bits, verbose=False, return_weights=False):
        correction = np.zeros(self.dem.num_observables, dtype=bool)
        if return_weights:
            return correction, np.array([0.75], dtype=float)
        return correction


class _FakeTask:
    detector_error_model = object()


class _ChunkResult:
    def __init__(self, detectors: np.ndarray, observables: np.ndarray):
        self.detectors = detectors.tolist()
        self.observables = observables.tolist()


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


def test_build_mle_decoders_supports_newer_and_older_decoder_apis(monkeypatch):
    monkeypatch.setattr(
        "demo.msd_utils.decoders.detector_error_model_to_check_matrices",
        lambda *args, **kwargs: _FakeDemMatrix(
            check_matrix=np.array([[1, 0], [0, 1], [1, 1], [0, 1]], dtype=int),
            observables_matrix=np.array([[1, 0], [0, 1]], dtype=int),
            priors=np.array([0.1, 0.2], dtype=float),
        ),
    )

    adapter_new = build_mle_decoders(_FakeTask(), gurobi_decoder_cls=_NewStyleGurobi)
    _, score_new = adapter_new.decode_factory("0")
    assert adapter_new.factory_score_mode == "logical_gap"
    assert score_new == pytest.approx(2.5)

    adapter_old = build_mle_decoders(_FakeTask(), gurobi_decoder_cls=_OldStyleGurobi)
    _, score_old = adapter_old.decode_factory("0")
    assert adapter_old.factory_score_mode == "weight"
    assert score_old == pytest.approx(0.75)


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
    assert np.array_equal(batch_full.decode(test_full), stream_full.decode(test_full))
    assert np.array_equal(
        batch_factory.decode(test_factory),
        stream_factory.decode(test_factory),
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
        def decode_factory(key: str):
            a_det = np.array([int(x) for x in key], dtype=np.uint8)
            return (0,), 1.0 - 0.25 * int(a_det[-1])

        def decode_full(key: str):
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
        posterior_samples=64,
        threshold_points=3,
        metric="test",
        factory_target=np.array([0], dtype=np.uint8),
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
        posterior_samples=256,
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
        def decode_factory(key: str):
            a_det = np.array([int(x) for x in key], dtype=np.uint8)
            # Deliberately misleading score: the lower-fidelity pattern gets the
            # higher decoder score. The legacy MLD evaluator ranks by this score
            # table, not by recomputing a separate fidelity-based ordering.
            return (0,), 0.5 if int(a_det[-1]) == 0 else 1.0

        def decode_full(key: str):
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
        posterior_samples=64,
        factory_target=np.array([0], dtype=np.uint8),
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
        def decode_factory(key: str):
            a_det = np.array([int(x) for x in key], dtype=np.uint8)
            return (0,), 0.5 if int(a_det[-1]) == 0 else 1.0

        def decode_full(key: str):
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
        posterior_samples=64,
        factory_target=np.array([0], dtype=np.uint8),
        sign_vector=(1.0, 1.0, 1.0),
        min_accepted_per_basis=1,
    )

    curves = evaluate_curve(
        {"X": dataset, "Y": dataset, "Z": dataset},
        {"X": make_adapter(), "Y": make_adapter(), "Z": make_adapter()},
        posterior_samples=64,
        threshold_points=4,
        metric="test",
        factory_target=np.array([0], dtype=np.uint8),
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
        def decode_factory(key: str | int):
            key_str = key if isinstance(key, str) else format(int(key), "b")
            last = int(key_str[-1]) if key_str else 0
            if last == 0:
                return (0,), 0.8
            return (0,), 0.4

        def decode_full(key: str | int):
            key_str = key if isinstance(key, str) else format(int(key), "b")
            last = int(key_str[-1]) if key_str else 0
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
        posterior_samples=64,
        threshold_points=3,
        metric="test",
        factory_target=np.array([0], dtype=np.uint8),
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
        posterior_samples=64,
        threshold_points=3,
        metric="test",
        factory_target=np.array([0], dtype=np.uint8),
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
    class _RecordingTableDecoder:
        @classmethod
        def from_det_obs_shots(cls, dem, det_obs_shots):
            return {
                "num_detectors": dem.num_detectors,
                "num_observables": dem.num_observables,
                "shape": tuple(det_obs_shots.shape),
            }

    dataset = BasisDataset(
        detectors=np.zeros((8, 27), dtype=np.uint8),
        observables=np.zeros((8, 9), dtype=np.uint8),
    )

    full_decoder, factory_decoder = train_mld_decoder_pair(
        dataset,
        table_decoder_cls=_RecordingTableDecoder,
    )

    assert full_decoder["num_detectors"] == 27
    assert full_decoder["num_observables"] == 1
    assert full_decoder["shape"] == (8, 28)

    assert factory_decoder["num_detectors"] == 24
    assert factory_decoder["num_observables"] == 8
    assert factory_decoder["shape"] == (8, 32)


def test_notebooks_import_shared_msd_utils():
    for notebook in [
        Path("demo/magic_state_distillation_reprod.ipynb"),
        Path("demo/msd_reprod_bloqade_decoders.ipynb"),
    ]:
        nb = json.loads(notebook.read_text())
        joined = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])
        assert "demo.msd_utils" in joined
