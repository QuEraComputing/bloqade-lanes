from __future__ import annotations

import numpy as np
import pytest
import stim

from bloqade.gemini.decoding.bit_packing import (
    pack_boolean_array,
    shots_to_counts,
    unpack_packed_bits,
)
from bloqade.gemini.decoding.confidence import (
    ConfidenceDecoder,
    GurobiDecoderWithConfidence,
)
from bloqade.gemini.decoding.dem import _sub_detector_error_model
from bloqade.gemini.decoding.table_decoders import TableDecoderWithConfidence
from bloqade.gemini.decoding.tomography import TomographyResult


def test_tomography_result_builds_density_matrix_and_point_fidelity():
    result = TomographyResult(
        {
            "X": np.array([[0], [0]], dtype=np.uint8),
            "Y": np.array([[0], [1]], dtype=np.uint8),
            "Z": np.array([[0], [1]], dtype=np.uint8),
        }
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


def test_bit_packing_helpers_support_zero_width_rows():
    bits = np.zeros((3, 0), dtype=np.uint8)

    packed = pack_boolean_array(bits)

    np.testing.assert_array_equal(packed, np.zeros(3, dtype=np.uint64))
    np.testing.assert_array_equal(unpack_packed_bits(0, 0), np.zeros(0, dtype=np.uint8))


def test_sub_detector_error_model_composes_duplicate_projected_errors():
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 L0
        error(0.2) D0 L0
        error(0.3) D1
        """)

    projected = _sub_detector_error_model(
        dem,
        detector_indices=[0],
        observable_indices=[0],
    )
    error_instructions = [
        instruction
        for instruction in projected.flattened()
        if isinstance(instruction, stim.DemInstruction) and instruction.type == "error"
    ]

    assert len(error_instructions) == 1
    assert error_instructions[0].args_copy()[0] == pytest.approx(0.26)
    assert str(projected).splitlines()[0] == "error(0.26) D0 L0"


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
    decoder = TableDecoderWithConfidence(dem, num_shots=0)
    decoder._update_det_obs_counts(shots)

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
    assert confidence == pytest.approx(1.0)


def test_gurobi_decoder_with_confidence_reports_gap_to_best_other_logical_solution():
    pytest.importorskip("gurobipy")
    dem = stim.DetectorErrorModel("""
        error(0.1) D0
        error(0.01) D0 L0
        """)
    decoder = GurobiDecoderWithConfidence(dem)

    correction, logical_gap = decoder.decode_with_confidence(
        np.array([1], dtype=np.bool_)
    )

    np.testing.assert_array_equal(correction, np.array([False]))
    expected_gap = np.log(0.1 / 0.9) - np.log(0.01 / 0.99)
    assert logical_gap == pytest.approx(expected_gap)


def test_shots_to_counts_uses_little_endian_packing():
    shots = np.array([[0, 0], [1, 0], [1, 0], [0, 1]], dtype=np.uint8)

    counts = shots_to_counts(shots)

    np.testing.assert_array_equal(counts, np.array([1, 2, 1, 0]))
