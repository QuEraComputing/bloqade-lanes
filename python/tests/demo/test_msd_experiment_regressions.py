from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from demo.msd_utils.application.experiments import (  # noqa: E402
    PostSelectionExperiment,
    PostSelectionExperimentCache,
)
from demo.msd_utils.standard.tomography import TomographyResult  # noqa: E402

from bloqade.gemini.decoding.postselection import (  # noqa: E402
    DecoderAdapter,
    _build_generic_threshold_tables,
    _evaluate_cached_threshold_curve,
)
from bloqade.gemini.decoding.sampling import BasisDataset  # noqa: E402


def _dataset() -> BasisDataset:
    return BasisDataset(
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
                [1, 0],
                [0, 1],
                [1, 0],
            ],
            dtype=np.uint8,
        ),
    )


def _adapter() -> DecoderAdapter:
    def decode_factory(syndrome: np.ndarray) -> tuple[np.ndarray, float]:
        factory_bit = int(syndrome[0]) if len(syndrome) else 0
        return np.array([factory_bit], dtype=np.uint8), 0.9 if factory_bit == 0 else 0.1

    def decode_full(_syndrome: np.ndarray) -> np.ndarray:
        return np.array([0], dtype=np.uint8)

    return DecoderAdapter(
        decode_factory=decode_factory,
        decode_full=decode_full,
    )


def test_build_generic_threshold_tables_groups_counts_by_confidence_score():
    data = {basis: _dataset() for basis in ("X", "Y", "Z")}
    decoders = {basis: _adapter() for basis in ("X", "Y", "Z")}

    per_basis, scores, weights, total_shots = _build_generic_threshold_tables(
        data,
        decoders,
        targets=np.array([[0]], dtype=np.uint8),
        basis_labels=("X", "Y", "Z"),
    )

    assert total_shots == 12
    np.testing.assert_array_equal(scores, np.array([0.1, 0.9]))
    np.testing.assert_array_equal(weights, np.array([3, 6]))
    np.testing.assert_array_equal(per_basis["X"][0], np.array([0.1, 0.9]))
    np.testing.assert_array_equal(per_basis["X"][1], np.array([1, 1]))
    np.testing.assert_array_equal(per_basis["X"][2], np.array([0, 1]))


def test_evaluate_cached_threshold_curve_returns_point_estimate_only():
    per_basis = {
        basis: (
            np.array([0.1, 0.9], dtype=np.float64),
            np.array([1, 10], dtype=np.int64),
            np.array([0, 0], dtype=np.int64),
        )
        for basis in ("X", "Y", "Z")
    }

    curve = _evaluate_cached_threshold_curve(
        per_basis,
        np.array([0.1, 0.9], dtype=np.float64),
        score_weights=np.array([3, 30], dtype=np.int64),
        threshold_points=2,
        target_bloch=np.ones(3) / np.sqrt(3.0),
        basis_labels=("X", "Y", "Z"),
        min_accepted_per_basis=1,
        total_shots=300,
    )

    assert set(curve) == {"accepted_fraction", "fidelity", "point_fidelity"}
    assert "credible" not in curve
    assert np.all(curve["fidelity"] == curve["point_fidelity"])


def test_postselection_experiment_decode_and_tomography_result_with_cached_data():
    exp = object.__new__(PostSelectionExperiment)
    exp.postselection_condition = np.array([[0]], dtype=np.uint8)
    exp.postselection_exp_cache = PostSelectionExperimentCache()
    exp.postselection_exp_cache.raw_results = {
        basis: _dataset() for basis in ("X", "Y", "Z")
    }
    exp.postselection_exp_cache.decoders_with_confidence = {
        basis: _adapter() for basis in ("X", "Y", "Z")
    }

    decoded = exp.decode_and_postselect(decoder_name=None)
    assert decoded[3] == 12

    result = exp.tomography_result(accepted_fraction=0.5)
    assert isinstance(result, TomographyResult)
    assert result.density_matrix.shape == (2, 2)


def test_postselection_experiment_tomography_fraction_is_all_sampled_shots():
    exp = object.__new__(PostSelectionExperiment)
    exp.postselection_exp_cache = PostSelectionExperimentCache()
    exp.postselection_exp_cache.decoded_results = (
        {
            basis: (
                np.array([0.5], dtype=np.float64),
                np.array([10], dtype=np.int64),
                np.array([0], dtype=np.int64),
            )
            for basis in ("X", "Y", "Z")
        },
        np.array([0.5], dtype=np.float64),
        np.array([30], dtype=np.int64),
        300,
    )

    result = exp.tomography_result(accepted_fraction=0.1)

    assert result.fidelity_bloch(np.ones(3) / np.sqrt(3.0))["point"] == pytest.approx(
        (1.0 + np.sqrt(3.0)) / 2.0
    )
