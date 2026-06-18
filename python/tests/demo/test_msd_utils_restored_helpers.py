from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import stim

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from demo.msd_utils.application.table_decoders import (  # noqa: E402
    TableDecoderWithConfidence,
)
from demo.msd_utils.domain.confidence import ConfidenceDecoder  # noqa: E402
from demo.msd_utils.standard.bit_packing import (  # noqa: E402
    pack_boolean_array,
    shots_to_counts,
    unpack_packed_bits,
)
from demo.msd_utils.standard.tomography import TomographyResult  # noqa: E402


def test_tomography_result_builds_density_matrix_and_point_fidelity():
    result = TomographyResult(
        zero_counts=np.array([10, 5, 5], dtype=np.int64),
        one_counts=np.array([0, 5, 5], dtype=np.int64),
    )

    assert result.density_matrix.shape == (2, 2)
    assert result.fidelity_bloch(np.array([1.0, 0.0, 0.0]))["point"] == pytest.approx(
        1.0
    )


def test_bit_packing_helpers_round_trip_little_endian_bits():
    bits = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)

    packed = pack_boolean_array(bits)

    assert packed.tolist() == [0b101, 0b110]
    np.testing.assert_array_equal(
        unpack_packed_bits(0b1101, 4),
        np.array([1, 0, 1, 1], dtype=np.uint8),
    )


def test_table_decoder_with_confidence_decodes_and_scores_detector_patterns():
    dem = stim.DetectorErrorModel("error(0.1) D0 L0")
    shots = np.array(
        [
            [0, 0],
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 1],
            [1, 0],
        ],
        dtype=np.uint8,
    )
    decoder = TableDecoderWithConfidence(dem, det_obs_counts=shots_to_counts(shots))

    correction0, confidence0 = decoder.decode_with_confidence(
        np.array([0], dtype=np.uint8)
    )
    correction1, confidence1 = decoder.decode_with_confidence(
        np.array([1], dtype=np.uint8)
    )

    assert isinstance(decoder, ConfidenceDecoder)
    np.testing.assert_array_equal(correction0, np.array([False]))
    np.testing.assert_array_equal(correction1, np.array([True]))
    assert confidence0 == pytest.approx(2.0 / 3.0)
    assert confidence1 == pytest.approx(2.0 / 3.0)
    assert decoder._det_obs_counts.dtype == np.uint32


def test_table_decoder_with_confidence_allows_empty_dem_and_step_size_none():
    decoder = TableDecoderWithConfidence(
        stim.DetectorErrorModel(""),
        num_shots=0,
        step_size=None,
    )

    correction, confidence = decoder.decode_with_confidence(np.zeros(0, dtype=np.uint8))

    assert correction.shape == (0,)
    assert np.isnan(confidence)
