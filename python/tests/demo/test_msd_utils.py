from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from demo.msd_utils.circuits import (
    build_decoder_kernel_bundle,
    build_naive_kernel_bundle,
)
from demo.msd_utils.core import (
    BasisDataset,
    fidelity_from_counts,
    pack_boolean_array,
    split_factory_bits,
)
from demo.msd_utils.decoders import DecoderAdapter, build_mle_decoders, evaluate_curve


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


def test_notebooks_import_shared_msd_utils():
    for notebook in [
        Path("demo/magic_state_distillation_reprod.ipynb"),
        Path("demo/msd_reprod_bloqade_decoders.ipynb"),
    ]:
        nb = json.loads(notebook.read_text())
        joined = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])
        assert "demo.msd_utils" in joined
