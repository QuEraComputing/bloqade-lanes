from __future__ import annotations

import matplotlib
import numpy as np

from bloqade.gemini.decoding import (
    ConfidenceDecoder,
    TableDecoderWithConfidence,
    TomographyResult,
    empty_logical_circuit,
    plot_decoder_curves,
    single_qubit_state_tomography,
)
from bloqade.gemini.decoding.kernels import _DecoderPrimitiveSet
from bloqade.gemini.decoding.msd import _build_decoder_kernel_bundle

matplotlib.use("Agg")


def test_public_facade_exports_simplified_decoder_and_tomography_types():
    assert issubclass(TableDecoderWithConfidence, ConfidenceDecoder)
    result = TomographyResult(
        {
            "X": np.array([[0], [0]], dtype=np.uint8),
            "Y": np.array([[0], [1]], dtype=np.uint8),
            "Z": np.array([[0], [1]], dtype=np.uint8),
        }
    )
    assert result.density_matrix.shape == (2, 2)


def test_build_decoder_kernel_bundle_contains_basis_tomography_kernels():
    tomography_kernels = single_qubit_state_tomography()
    kernels = _build_decoder_kernel_bundle(
        _DecoderPrimitiveSet(
            state_injection_circuit=empty_logical_circuit(),
            logical_circuit=empty_logical_circuit(),
        ),
        num_logical_qubits=1,
        tomography_kernels=tomography_kernels,
    )

    assert set(kernels) == {"X", "Y", "Z"}


def test_plot_decoder_curves_handles_curves_without_uncertainty_bands():
    fig, ax = plot_decoder_curves(
        {
            "decoder": {
                "accepted_fraction": np.array([0.05, 0.1]),
                "fidelity": np.array([0.99, 0.95]),
                "point_fidelity": np.array([0.99, 0.95]),
            }
        },
        injected_summary={"point": 0.96},
        log=False,
    )

    assert fig is ax.figure
    assert len(ax.lines) == 2
    assert len(ax.collections) == 0
