from __future__ import annotations

import numpy as np

from bloqade.gemini.decoding.confidence import ConfidenceDecoder
from bloqade.gemini.decoding.experiments import (
    PostSelectionExperiment,
    _PostSelectionExperimentCache,
)
from bloqade.gemini.decoding.postselection import (
    DecodedPostselectionResult,
    _build_generic_threshold_tables,
    _evaluate_cached_threshold_curve,
    _shots_at_accepted_fraction,
)
from bloqade.gemini.decoding.sampling import BasisDataset
from bloqade.gemini.decoding.tomography import TomographyResult


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


class _FactoryDecoder(ConfidenceDecoder):
    def decode_with_confidence(
        self,
        detector_bits: np.ndarray,
    ) -> tuple[np.ndarray, np.float64]:
        factory_bit = int(detector_bits[0]) if len(detector_bits) else 0
        return (
            (
                np.array([factory_bit], dtype=np.bool_)
                if len(detector_bits)
                else np.zeros(0, dtype=np.bool_)
            ),
            np.float64(0.9 if factory_bit == 0 else 0.1),
        )


class _FullDecoder:
    def decode(self, _detector_bits: np.ndarray) -> np.ndarray:
        return np.array([0], dtype=np.bool_)


def _decoder_pair() -> tuple[ConfidenceDecoder, _FullDecoder]:
    return _FactoryDecoder(), _FullDecoder()


def test_build_generic_threshold_tables_returns_decoded_shots_with_confidence():
    data = {basis: _dataset() for basis in ("X", "Y", "Z")}
    decoders = {basis: _decoder_pair() for basis in ("X", "Y", "Z")}

    decoded = _build_generic_threshold_tables(
        data,
        decoders,
        targets=np.array([[0]], dtype=np.uint8),
        basis_labels=("X", "Y", "Z"),
    )

    assert decoded["X"].observables.shape == (3, 1)
    assert decoded["X"].confidence.shape == (3,)
    np.testing.assert_array_equal(decoded["X"].observables[:, 0], np.array([0, 1, 0]))
    np.testing.assert_allclose(decoded["X"].confidence, np.array([0.9, 0.9, 0.1]))


def test_build_generic_threshold_tables_supports_empty_factory_postselection():
    dataset = BasisDataset(
        detectors=np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
            ],
            dtype=np.uint8,
        ),
        observables=np.array([[0], [1], [0]], dtype=np.uint8),
    )

    data = {basis: dataset for basis in ("X", "Y", "Z")}
    decoders = {basis: _decoder_pair() for basis in ("X", "Y", "Z")}

    decoded = _build_generic_threshold_tables(
        data,
        decoders,
        targets=np.zeros((1, 0), dtype=np.uint8),
        basis_labels=("X", "Y", "Z"),
    )

    assert decoded["X"].observables.shape == (3, 1)
    np.testing.assert_array_equal(decoded["X"].observables[:, 0], np.array([0, 1, 0]))


def test_evaluate_cached_threshold_curve_returns_point_estimate_only():
    decoded = {
        basis: DecodedPostselectionResult(
            observables=np.array([[0], [0], [1]], dtype=np.uint8),
            confidence=np.array([0.9, 0.8, 0.1], dtype=np.float64),
        )
        for basis in ("X", "Y", "Z")
    }

    curve = _evaluate_cached_threshold_curve(
        decoded,
        threshold_points=3,
        target_bloch=np.ones(3) / np.sqrt(3.0),
        basis_labels=("X", "Y", "Z"),
        min_accepted_per_basis=1,
        total_shots=30,
    )

    assert set(curve) == {"accepted_fraction", "fidelity", "point_fidelity"}
    assert "credible" not in curve
    assert np.all(curve["fidelity"] == curve["point_fidelity"])


def test_postselection_experiment_decode_and_tomography_result_with_cached_data():
    exp = object.__new__(PostSelectionExperiment)
    exp.postselection_condition = np.array([[0]], dtype=np.uint8)
    exp._postselection_exp_cache = _PostSelectionExperimentCache()
    exp._postselection_exp_cache.raw_results = {
        basis: _dataset() for basis in ("X", "Y", "Z")
    }
    exp._postselection_exp_cache.decoders_with_confidence = {
        basis: _decoder_pair() for basis in ("X", "Y", "Z")
    }
    exp._postselection_exp_cache.decoded_results = None

    decoded = exp.decode_and_postselect(decoder_name=None)
    assert decoded["X"].observables.shape == (3, 1)

    result = exp.tomography_result(accepted_fraction=0.5)
    assert isinstance(result, TomographyResult)
    assert result.density_matrix.shape == (2, 2)


def test_shots_at_accepted_fraction_is_relative_to_postselected_shots():
    decoded = {
        basis: DecodedPostselectionResult(
            observables=np.array([[0], [1], [0]], dtype=np.uint8),
            confidence=np.array([0.9, 0.8, 0.1], dtype=np.float64),
        )
        for basis in ("X", "Y", "Z")
    }

    accepted = _shots_at_accepted_fraction(
        decoded,
        1 / 3,
        basis_labels=("X", "Y", "Z"),
    )

    assert sum(shots.shape[0] for shots in accepted.values()) == 3
