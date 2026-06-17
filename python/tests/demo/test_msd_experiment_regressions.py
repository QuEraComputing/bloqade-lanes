from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import stim
from bloqade.decoders import BaseDecoder

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from demo.msd_utils.application.experiments import (  # noqa: E402
    PostSelectionExperiment,
    PostSelectionExperimentCache,
    TomographyResult,
)
from demo.msd_utils.application.table_decoders import (  # noqa: E402
    SparseTableDecoder,
    TableDecoderWithSimplerConfidence,
)
from demo.msd_utils.domain.confidence import (  # noqa: E402
    ConfidenceDecoder,
    ConfidenceGurobiDecoder,
    TableDecoderWithConfidence,
)

from bloqade.gemini.decoding.postselection import DecoderAdapter  # noqa: E402
from bloqade.gemini.decoding.sampling import BasisDataset  # noqa: E402


class _FakeConfidenceDecoder(BaseDecoder, ConfidenceDecoder):
    confidence_score_mode = "fake-frequency"
    instances: list[_FakeConfidenceDecoder] = []

    def __init__(
        self,
        dem: stim.DetectorErrorModel,
        *,
        confidence: float = 0.5,
    ) -> None:
        super().__init__(dem)
        self.dem = dem
        self.confidence = confidence
        type(self).instances.append(self)

    def _decode(self, detector_bits: np.ndarray) -> np.ndarray:
        return np.zeros(self.dem.num_observables, dtype=np.bool_)

    def decode_with_confidence(
        self,
        detector_bits: np.ndarray,
    ) -> tuple[np.ndarray, np.float64]:
        return self._decode(detector_bits), np.float64(self.confidence)


def _fake_experiment_with_decoded_results(
    decoded_results: tuple[
        dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
        np.ndarray,
        np.ndarray,
        int,
    ],
) -> PostSelectionExperiment:
    exp = object.__new__(PostSelectionExperiment)
    exp.postselection_exp_cache = PostSelectionExperimentCache()
    exp.postselection_exp_cache.decoded_results = decoded_results
    return exp


def _fake_decoding_experiment() -> PostSelectionExperiment:
    exp = object.__new__(PostSelectionExperiment)
    exp.postselection_exp_cache = PostSelectionExperimentCache()
    exp.postselection_condition = np.array([[0]], dtype=np.uint8)

    detectors = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    observables = np.array(
        [
            [0, 0],
            [1, 1],
            [0, 1],
            [1, 0],
        ],
        dtype=np.uint8,
    )
    dataset = BasisDataset(detectors=detectors, observables=observables)
    exp.postselection_exp_cache.raw_results = {
        basis: dataset for basis in ("X", "Y", "Z")
    }

    def decode_factory(packed_syndrome: int) -> tuple[tuple[int], float]:
        factory_bit = int(packed_syndrome)
        return (factory_bit,), 0.75 if factory_bit else 0.25

    def decode_full(_packed_syndrome: int) -> tuple[int]:
        return (0,)

    adapter = DecoderAdapter(
        full_decoder=None,
        factory_decoder=None,
        decode_factory=decode_factory,
        decode_full=decode_full,
        factory_score_mode="test-score",
    )
    exp.postselection_exp_cache.decoders_with_confidence = {
        basis: adapter for basis in ("X", "Y", "Z")
    }
    return exp


def test_tomography_result_fidelity_accepts_vector_coordinates_and_angles():
    result = TomographyResult(
        basis_labels=("X", "Y", "Z"),
        zero_counts=np.array([3, 2, 1], dtype=np.int64),
        one_counts=np.array([1, 2, 3], dtype=np.int64),
        method="wilson",
        sign_vector=np.array([1.0, 1.0, 1.0]),
        binary_precision=4,
    )

    by_vector = result.fidelity_bloch(np.array([1.0, 0.0, 0.0]))
    by_coordinates = result.fidelity_bloch(1.0, 0.0, 0.0)
    by_angles = result.fidelity_bloch(np.pi / 2.0, 0.0)

    assert result.total_counts.tolist() == [4, 4, 4]
    assert result.bloch.tolist() == pytest.approx([0.5, 0.0, -0.5])
    assert by_vector["point"] == pytest.approx(0.75)
    assert by_coordinates["point"] == pytest.approx(by_vector["point"])
    assert by_angles["point"] == pytest.approx(by_vector["point"])
    assert by_vector["low"] <= by_vector["median"] <= by_vector["high"]


def test_tomography_result_accumulates_highest_confidence_groups_first():
    decoded_results = (
        {
            "X": (
                np.array([0.2, 0.9]),
                np.array([0, 3], dtype=np.int64),
                np.array([5, 0], dtype=np.int64),
            ),
            "Y": (
                np.array([0.1, 0.8]),
                np.array([0, 2], dtype=np.int64),
                np.array([4, 0], dtype=np.int64),
            ),
            "Z": (
                np.array([0.3, 0.7]),
                np.array([0, 1], dtype=np.int64),
                np.array([6, 0], dtype=np.int64),
            ),
        },
        np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9], dtype=np.float64),
        np.array([4, 5, 6, 1, 2, 3], dtype=np.int64),
        63,
    )
    exp = _fake_experiment_with_decoded_results(decoded_results)

    tiny_fraction = exp.tomography_result(
        1 / 63,
        "wilson",
        basis_labels=("X", "Y", "Z"),
    )
    high_groups = exp.tomography_result(
        6 / 63,
        "wilson",
        basis_labels=("X", "Y", "Z"),
    )

    assert tiny_fraction.zero_counts.tolist() == [3, 0, 0]
    assert tiny_fraction.one_counts.tolist() == [0, 0, 0]
    assert high_groups.zero_counts.tolist() == [3, 2, 1]
    assert high_groups.one_counts.tolist() == [0, 0, 0]


def test_postselection_experiment_decode_and_analysis_use_cached_tables():
    exp = _fake_decoding_experiment()

    per_basis_tables, scores, weights, total_shots = exp.decode_and_postselect(
        decoder_name=None,
    )
    curve = exp.analysis_f_vs_fraction(
        binary_precision=4,
        sign_vector=(1.0, 1.0, 1.0),
        target_bloch=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        basis_labels=("X", "Y", "Z"),
        threshold_points=2,
        min_accepted_per_basis=1,
    )

    assert set(per_basis_tables) == {"X", "Y", "Z"}
    assert scores.tolist() == [0.25, 0.75]
    assert weights.tolist() == [6, 6]
    assert total_shots == 12
    assert exp.postselection_exp_cache.decoded_results is not None
    assert exp.postselection_exp_cache.thresholded_data is curve
    assert curve["accepted_fraction"].shape == curve["fidelity"].shape


def test_initialize_decoders_constructs_confidence_adapters_from_decoder_class():
    _FakeConfidenceDecoder.instances.clear()
    exp = object.__new__(PostSelectionExperiment)
    exp.postselection_exp_cache = PostSelectionExperimentCache()
    exp.postselection_exp_cache.dems = {"X": stim.DetectorErrorModel("""
            error(0.1) D0 L0
            error(0.2) D3 L1
            """)}
    exp.decoder = _FakeConfidenceDecoder
    exp.decoder_init_args = {"confidence": 0.625}

    adapters = exp.initialize_decoders()
    factory_correction, score = adapters["X"].decode_factory(0)

    assert len(_FakeConfidenceDecoder.instances) == 2
    assert exp.postselection_exp_cache.decoders_with_confidence is adapters
    assert factory_correction == (0,)
    assert score == pytest.approx(0.625)
    assert adapters["X"].factory_score_mode == "fake-frequency"


def test_sparse_table_decoder_recomputes_correction_after_incremental_update():
    dem = stim.DetectorErrorModel("error(0.5) D0 L0\n")
    decoder = SparseTableDecoder.from_det_obs_shots(
        dem,
        np.array([[1, 0], [1, 0]], dtype=bool),
    )

    np.testing.assert_array_equal(
        decoder.decode(np.array([1], dtype=bool)),
        np.array([False]),
    )

    decoder.update_det_obs_counts(np.array([[1, 1], [1, 1], [1, 1]], dtype=bool))

    np.testing.assert_array_equal(
        decoder.decode(np.array([1], dtype=bool)),
        np.array([True]),
    )


def test_table_decoder_with_simpler_confidence_updates_uint32_counts():
    dem = stim.DetectorErrorModel("error(0.5) D0 L0\n")
    decoder = TableDecoderWithSimplerConfidence(dem, num_shots=0)

    decoder.update_det_obs_counts(
        np.array([[0, 0], [1, 1], [1, 1]], dtype=bool),
    )

    assert decoder._det_obs_counts.dtype == np.uint32
    assert decoder._det_obs_counts.tolist() == [1, 0, 0, 2]


def test_table_decoder_with_simpler_confidence_none_step_size_uses_one_batch(
    monkeypatch: pytest.MonkeyPatch,
):
    update_sizes: list[int] = []
    original_update = TableDecoderWithSimplerConfidence.update_det_obs_counts

    def track_update(
        self: TableDecoderWithSimplerConfidence,
        det_obs_shots: np.ndarray,
    ) -> None:
        update_sizes.append(det_obs_shots.shape[0])
        original_update(self, det_obs_shots)

    monkeypatch.setattr(
        TableDecoderWithSimplerConfidence,
        "update_det_obs_counts",
        track_update,
    )

    decoder = TableDecoderWithSimplerConfidence(
        stim.DetectorErrorModel(""),
        num_shots=5,
        step_size=None,
    )

    assert update_sizes == [5]
    assert decoder._det_obs_counts.tolist() == [5]


def test_table_decoder_with_simpler_confidence_decodes_with_frequency_score():
    dem = stim.DetectorErrorModel("error(0.5) D0 L0\n")
    decoder = TableDecoderWithSimplerConfidence(dem, num_shots=0)
    decoder.update_det_obs_counts(
        np.array(
            [
                [1, 0],
                [1, 1],
                [1, 1],
                [1, 1],
            ],
            dtype=bool,
        )
    )

    correction, confidence = decoder.decode_with_confidence(
        np.array([1], dtype=np.bool_),
    )

    np.testing.assert_array_equal(correction, np.array([True]))
    assert confidence == np.float64(0.75)
    assert decoder.confidence_score_mode == "correction_frequency"


def test_table_decoder_with_simpler_confidence_unseen_syndrome_has_nan_score():
    dem = stim.DetectorErrorModel("error(0.5) D0 L0\n")
    decoder = TableDecoderWithSimplerConfidence(dem, num_shots=0)
    decoder.update_det_obs_counts(np.array([[1, 1]], dtype=bool))

    correction, confidence = decoder.decode_with_confidence(
        np.array([0], dtype=np.bool_),
    )

    np.testing.assert_array_equal(correction, np.array([False]))
    assert np.isnan(confidence)


def test_table_decoder_with_simpler_confidence_update_rejects_uint32_overflow():
    dem = stim.DetectorErrorModel("error(0.5) D0 L0\n")
    counts = np.zeros(4, dtype=np.uint32)
    counts[3] = np.iinfo(np.uint32).max
    decoder = TableDecoderWithSimplerConfidence(dem, det_obs_counts=counts)

    with pytest.raises(OverflowError):
        decoder.update_det_obs_counts(np.array([[1, 1]], dtype=bool))


def test_confidence_decoder_interface_is_separate_from_base_decoder():
    assert not issubclass(ConfidenceDecoder, BaseDecoder)
    assert issubclass(ConfidenceGurobiDecoder, BaseDecoder)
    assert issubclass(ConfidenceGurobiDecoder, ConfidenceDecoder)
    assert issubclass(TableDecoderWithSimplerConfidence, BaseDecoder)
    assert issubclass(TableDecoderWithSimplerConfidence, ConfidenceDecoder)
    assert issubclass(TableDecoderWithConfidence, BaseDecoder)
    assert issubclass(TableDecoderWithConfidence, ConfidenceDecoder)


def test_confidence_gurobi_decoder_smoke_with_upstream_main_decoder():
    pytest.importorskip("gurobipy")

    dem = stim.DetectorErrorModel("""
        error(0.1) D0 L0
        error(0.2) D0
        """)
    decoder = ConfidenceGurobiDecoder(dem)

    correction, confidence = decoder.decode_with_confidence(
        np.array([1], dtype=np.bool_),
    )

    np.testing.assert_array_equal(correction, np.array([False]))
    assert float(confidence) == pytest.approx(0.8109302162163285)
