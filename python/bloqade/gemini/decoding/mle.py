from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast

import numpy as np
from bloqade.decoders import BaseDecoder
from demo.msd_utils.domain.confidence import ConfidenceDecoder
from demo.msd_utils.standard.bit_packing import pack_boolean_array

from .dem import sub_detector_error_model
from .layout import DEFAULT_SYNDROME_LAYOUT, SyndromeLayout
from .postselection import DecoderAdapter, _make_decoder_adapter
from .types import DetectorErrorModelTask


def build_mle_decoders(
    task: DetectorErrorModelTask,
    *,
    # TODO: change this to type[ConfidenceDecoder] once pyright error on bloqade-decoders
    # for decode() signature from GurobiDecoder is fixed
    gurobi_decoder_cls: Callable[..., object],
    decoder_init_args: Mapping[str, Any] | None = None,
    layout: SyndromeLayout = DEFAULT_SYNDROME_LAYOUT,
) -> DecoderAdapter:
    """Build full and factory confidence-decoder adapters from a task DEM.

    Args:
        task: Object exposing the full detector error model.
        gurobi_decoder_cls: Confidence-capable decoder class used for both the
            full and factory DEMs. Despite the historical name, this can be any
            decoder constructor compatible with ``BaseDecoder`` and
            ``ConfidenceDecoder``.
        decoder_init_args: Optional keyword arguments forwarded to each decoder
            constructor.
        layout: Syndrome layout separating output and factory syndrome bits.

    Returns:
        Decoder adapter with full-output decoding and factory confidence
        decoding.
    """

    decoder_init_args = {} if decoder_init_args is None else dict(decoder_init_args)
    dem = task.detector_error_model
    full_dem = sub_detector_error_model(
        dem,
        detector_indices=range(dem.num_detectors),
        observable_indices=range(layout.output_observable_count),
    )
    factory_dem = sub_detector_error_model(
        dem,
        detector_indices=range(layout.output_detector_count, dem.num_detectors),
        observable_indices=range(layout.output_observable_count, dem.num_observables),
    )

    # TODO: get rid of these casts once pyright error on bloqade-decoders
    # for decode() signature from GurobiDecoder is fixed
    full_decoder = cast(BaseDecoder, gurobi_decoder_cls(full_dem, **decoder_init_args))
    factory_decoder = cast(
        ConfidenceDecoder,
        gurobi_decoder_cls(factory_dem, **decoder_init_args),
    )
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
