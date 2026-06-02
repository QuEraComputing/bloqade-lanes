from __future__ import annotations

import numpy as np
from bloqade.decoders import ConfidenceDecoder
from bloqade.decoders.bit_packing import pack_boolean_array
from bloqade.decoders.dem import detector_error_model_matrices, matrix_to_dem

from .layout import DEFAULT_SYNDROME_LAYOUT, SyndromeLayout
from .postselection import DecoderAdapter, _make_decoder_adapter
from .types import DetectorErrorModelTask


def build_mle_decoders(
    task: DetectorErrorModelTask,
    *,
    gurobi_decoder_cls: type[ConfidenceDecoder],
    layout: SyndromeLayout = DEFAULT_SYNDROME_LAYOUT,
) -> DecoderAdapter:
    """Build full and factory MLE decoder adapters from a task DEM.

    Args:
        task: Object exposing the full detector error model.
        gurobi_decoder_cls: Confidence-capable decoder class used for both the
            full and factory DEMs.
        layout: Syndrome layout separating output and factory syndrome bits.

    Returns:
        Decoder adapter with full-output decoding and factory confidence
        decoding.
    """

    dem_data = detector_error_model_matrices(task)
    full_dem = matrix_to_dem(dem_data["H"], dem_data["O"], dem_data["priors"])
    factory_dem = matrix_to_dem(
        dem_data["H"][layout.output_detector_count :, :],
        dem_data["O"][layout.output_observable_count :, :],
        dem_data["priors"],
    )

    full_decoder = gurobi_decoder_cls(full_dem)
    factory_decoder = gurobi_decoder_cls(factory_dem)
    # TODO: this attribute should be defined by the MLE decoder class
    score_mode = str(getattr(factory_decoder, "confidence_score_mode", "confidence"))

    def factory_decode_impl(syndrome: np.ndarray) -> tuple[np.ndarray, float]:
        correction, confidence = factory_decoder.decode_with_confidence(
            syndrome.astype(bool)
        )
        return np.asarray(correction, dtype=np.uint8), float(np.float64(confidence))

    adapter = _make_decoder_adapter(
        full_decoder=full_decoder,
        factory_decoder=factory_decoder,
        full_syndrome_length=full_dem.num_detectors,
        factory_syndrome_length=factory_dem.num_detectors,
        factory_decode_impl=factory_decode_impl,
        factory_score_mode=score_mode,
    )
    sample_syndrome = np.zeros(factory_dem.num_detectors, dtype=np.uint8)
    adapter.decode_factory(int(pack_boolean_array(sample_syndrome)[0]))
    return adapter
