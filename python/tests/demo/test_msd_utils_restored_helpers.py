# ruff: noqa: D103

import sys
from pathlib import Path

import numpy as np
import pytest
import stim

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from demo.msd_utils.application.table_decoders import SparseTableDecoder
from demo.msd_utils.domain.confidence import TableDecoderWithConfidence
from demo.msd_utils.standard.bit_packing import (
    det_obs_shots_to_counts,
    pack_boolean_array,
    packed_bits_to_int,
    shots_to_counts,
    unpack_boolean_array,
    unpack_packed_bits,
)
from demo.msd_utils.standard.dem import make_layout_only_dem, matrix_to_dem
from demo.msd_utils.standard.tomography import (
    DEFAULT_TARGET_BLOCH,
    expectation_conf_interval,
    expectation_with_error_bar,
    fidelity_from_counts,
    fidelity_from_zero_one_counts,
    posterior_fidelity_summary,
)


def test_tomography_fidelity_from_counts_returns_ordered_interval():
    summary = fidelity_from_counts(
        np.array([0, 0, 1, 0], dtype=np.uint8),
        np.array([0, 1, 0, 0], dtype=np.uint8),
        np.array([0, 0, 0, 1], dtype=np.uint8),
        binary_precision=4,
    )

    assert set(summary) >= {"point", "median", "low", "high", "bloch"}
    assert summary["low"] <= summary["median"] <= summary["high"]
    assert len(summary["bloch"]) == 3


def test_tomography_fidelity_from_zero_one_counts_matches_array_counts():
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


def test_tomography_expectation_helpers_return_interval_and_error_bar():
    interval = expectation_conf_interval(3, 1)
    expectation, error = expectation_with_error_bar(3, 1)

    assert interval.shape == (2,)
    assert interval[0] <= expectation <= interval[1]
    assert expectation == pytest.approx(0.5)
    assert error == pytest.approx((interval[1] - interval[0]) / 2.0)


def test_tomography_posterior_fidelity_summary_returns_ordered_interval():
    summary = posterior_fidelity_summary(
        np.array([8, 8, 8], dtype=np.int64),
        np.array([7, 6, 7], dtype=np.int64),
        DEFAULT_TARGET_BLOCH,
        binary_precision=4,
        max_grid_points=2_000,
    )

    assert np.isfinite(summary["point"])
    assert summary["low"] <= summary["median"] <= summary["high"]


def test_bit_packing_helpers_round_trip_little_endian_bits():
    bits = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)

    packed = pack_boolean_array(bits)

    assert packed.tolist() == [0b101, 0b110]
    np.testing.assert_array_equal(unpack_boolean_array(packed, 3), bits.astype(bool))
    assert packed_bits_to_int([1, 0, 1, 1]) == 0b1101
    np.testing.assert_array_equal(
        unpack_packed_bits(0b1101, 4),
        np.array([1, 0, 1, 1], dtype=np.uint8),
    )


def test_count_helpers_pack_detector_observable_shots():
    det_shots = np.array([[0, 0], [1, 0]], dtype=bool)
    obs_shots = np.array([[0], [1]], dtype=bool)

    counts = det_obs_shots_to_counts(det_shots, obs_shots)

    expected = np.zeros(8, dtype=np.int64)
    expected[0] = 1
    expected[5] = 1
    np.testing.assert_array_equal(counts, expected)
    np.testing.assert_array_equal(shots_to_counts(det_shots), np.array([1, 1, 0, 0]))


def test_dem_helpers_create_layout_only_and_matrix_dems():
    layout_dem = make_layout_only_dem(num_detectors=2, num_observables=1)
    assert layout_dem.num_detectors == 2
    assert layout_dem.num_observables == 1

    dem = matrix_to_dem(
        check_matrix=np.array([[1, 0], [0, 1]], dtype=np.uint8),
        observables_matrix=np.array([[1, 0]], dtype=np.uint8),
        priors=np.array([0.25, 0.5], dtype=np.float64),
    )

    assert dem.num_detectors == 2
    assert dem.num_observables == 1
    assert "D0 L0" in str(dem)


def test_sparse_table_decoder_matches_dense_mld_argmax_semantics():
    dem = stim.DetectorErrorModel("error(0.5) D0 L0\nerror(0.5) D1 L0\n")
    shots = np.array(
        [
            [0, 0, 0],
            [0, 1, 1],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ],
        dtype=bool,
    )
    decoder = SparseTableDecoder.from_det_obs_shots(dem, shots)

    decoded = decoder.decode(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]))

    np.testing.assert_array_equal(decoded, np.array([[0], [1], [1], [0]], dtype=bool))


def test_table_decoder_with_confidence_uses_packed_syndrome_scores():
    dem = stim.DetectorErrorModel("error(0.5) D0 L0\n")
    decoder = SparseTableDecoder.from_det_obs_shots(
        dem,
        np.array([[0, 0], [1, 1], [1, 1]], dtype=bool),
    )
    wrapped = TableDecoderWithConfidence(
        decoder=decoder,
        syndrome_confidence=np.array([0.25, 0.75], dtype=np.float64),
    )

    correction, confidence = wrapped.decode_with_confidence(np.array([1], dtype=bool))

    np.testing.assert_array_equal(correction, np.array([True]))
    assert confidence == np.float64(0.75)
