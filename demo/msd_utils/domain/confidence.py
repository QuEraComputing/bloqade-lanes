"""Confidence decoder wrappers used by MSD/QET table decoders."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from bloqade.decoders import BaseDecoder, ConfidenceDecoder
from demo.msd_utils.standard.bit_packing import pack_boolean_array


@dataclass(frozen=True)
class TableDecoderWithConfidence(ConfidenceDecoder):
    """Wrap a decoder with a syndrome-indexed confidence score."""

    decoder: BaseDecoder
    syndrome_confidence: np.ndarray
    confidence_score_mode: str = "mld_output_fidelity"

    def _decode(self, detector_bits: np.ndarray) -> np.ndarray:
        return np.asarray(self.decoder.decode(detector_bits), dtype=np.bool_)

    def decode_with_confidence(
        self,
        detector_bits: np.ndarray,
    ) -> tuple[np.ndarray, np.float64]:
        if detector_bits.ndim != 1:
            raise ValueError(
                "decode_with_confidence expects a single detector shot (1D array)."
            )
        correction = self.decode(detector_bits)
        packed = int(pack_boolean_array(np.asarray(detector_bits, dtype=np.uint8))[0])
        score = (
            float(self.syndrome_confidence[packed])
            if packed < len(self.syndrome_confidence)
            else float("nan")
        )
        return correction, np.float64(score)


__all__ = ["TableDecoderWithConfidence"]
