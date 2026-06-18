from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

matplotlib.use("Agg")

from demo.msd_utils import (  # noqa: E402
    DecoderAdapter,
    TableDecoderWithConfidence,
    TomographyResult,
    plot_decoder_curves,
)
from demo.msd_utils.application.experiments import (  # noqa: E402
    empty_logical_circuit,
    single_qubit_state_tomography,
)
from demo.msd_utils.domain.confidence import ConfidenceDecoder  # noqa: E402

from bloqade.gemini.decoding.kernels import DecoderPrimitiveSet  # noqa: E402
from bloqade.gemini.decoding.msd import build_decoder_kernel_bundle  # noqa: E402


def test_public_facade_exports_simplified_decoder_and_tomography_types():
    assert issubclass(TableDecoderWithConfidence, ConfidenceDecoder)
    assert TomographyResult(
        np.array([1, 1, 1]),
        np.array([1, 1, 1]),
    ).density_matrix.shape == (2, 2)


def test_decoder_adapter_accepts_array_callables_without_score_mode():
    def decode_factory(bits: np.ndarray) -> tuple[np.ndarray, float]:
        return bits.astype(np.uint8), 0.25

    def decode_full(bits: np.ndarray) -> np.ndarray:
        return np.array([int(np.any(bits))], dtype=np.uint8)

    adapter = DecoderAdapter(
        decode_factory=decode_factory,
        decode_full=decode_full,
    )

    assert not hasattr(adapter, "factory_score_mode")
    correction, score = adapter.decode_factory(np.array([1, 0], dtype=np.uint8))
    np.testing.assert_array_equal(correction, np.array([1, 0], dtype=np.uint8))
    assert score == pytest.approx(0.25)
    np.testing.assert_array_equal(
        adapter.decode_full(np.array([0, 1], dtype=np.uint8)),
        np.array([1], dtype=np.uint8),
    )


def test_build_decoder_kernel_bundle_contains_actual_tomography_kernels_only():
    tomography_kernels = single_qubit_state_tomography()
    bundle = build_decoder_kernel_bundle(
        DecoderPrimitiveSet(
            state_injection_circuit=empty_logical_circuit(),
            logical_circuit=empty_logical_circuit(),
        ),
        num_logical_qubits=1,
        tomography_kernels=tomography_kernels,
    )

    assert set(bundle.actual) == {"X", "Y", "Z"}
    assert not hasattr(bundle, "_special")


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
