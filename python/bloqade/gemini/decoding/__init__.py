# pyright: reportUnsupportedDunderAll=false

"""Notebook-focused Gemini decoding helpers."""

from bloqade.gemini.decoding.constants import (
    DEFAULT_BASIS_LABELS,
)
from bloqade.gemini.decoding.dem import sub_detector_error_model
from bloqade.gemini.decoding.kernels import DecoderPrimitiveSet
from bloqade.gemini.decoding.layout import (
    DEFAULT_SYNDROME_LAYOUT,
    SyndromeLayout,
    split_factory_bits,
)
from bloqade.gemini.decoding.measurement_maps import build_measurement_maps
from bloqade.gemini.decoding.msd import (
    TomographyKernels,
    build_decoder_kernel_bundle,
    build_msd_primitives,
)
from bloqade.gemini.decoding.postselection import DecoderAdapter
from bloqade.gemini.decoding.sampling import BasisDataset, run_task
from bloqade.gemini.decoding.special_tasks import (
    apply_special_tsim_circuit_strategy,
    build_task_map,
)
from bloqade.gemini.decoding.tasks import DemoTask, GeminiDecoderTask
from bloqade.gemini.decoding.types import (
    KirinKernel,
    MeasurementMap,
    SquinKernel,
    TsimCircuit,
)
from bloqade.gemini.decoding.workflow import plot_decoder_curves

__all__ = [
    "BasisDataset",
    "DEFAULT_BASIS_LABELS",
    "DEFAULT_SYNDROME_LAYOUT",
    "DecoderAdapter",
    "DecoderPrimitiveSet",
    "DemoTask",
    "GeminiDecoderTask",
    "KirinKernel",
    "MeasurementMap",
    "SquinKernel",
    "SyndromeLayout",
    "TomographyKernels",
    "TsimCircuit",
    "apply_special_tsim_circuit_strategy",
    "build_decoder_kernel_bundle",
    "build_measurement_maps",
    "build_msd_primitives",
    "build_task_map",
    "plot_decoder_curves",
    "run_task",
    "split_factory_bits",
    "sub_detector_error_model",
]
