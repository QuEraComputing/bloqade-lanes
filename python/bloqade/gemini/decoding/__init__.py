# pyright: reportUnsupportedDunderAll=false

"""Notebook-focused Gemini decoding helpers."""

from bloqade.gemini.decoding.confidence import (
    ConfidenceDecoder,
    ConfidenceGurobiDecoder,
)
from bloqade.gemini.decoding.experiments import (
    PostSelectionExperiment,
    PostSelectionExperimentCache,
    empty_logical_circuit,
    magic_state_dist_steane,
    single_qubit_state_tomography,
)
from bloqade.gemini.decoding.kernels import DecoderPrimitiveSet
from bloqade.gemini.decoding.layout import (
    DEFAULT_SYNDROME_LAYOUT,
    SyndromeLayout,
    split_factory_bits,
)
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
from bloqade.gemini.decoding.table_decoders import TableDecoderWithConfidence
from bloqade.gemini.decoding.tasks import DemoTask
from bloqade.gemini.decoding.tomography import DEFAULT_TARGET_BLOCH, TomographyResult
from bloqade.gemini.decoding.types import (
    KirinKernel,
    MeasurementMap,
    SquinKernel,
    TsimCircuit,
)
from bloqade.gemini.decoding.workflow import plot_decoder_curves

__all__ = [
    "BasisDataset",
    "ConfidenceDecoder",
    "ConfidenceGurobiDecoder",
    "DEFAULT_SYNDROME_LAYOUT",
    "DEFAULT_TARGET_BLOCH",
    "DecoderAdapter",
    "DecoderPrimitiveSet",
    "DemoTask",
    "KirinKernel",
    "MeasurementMap",
    "PostSelectionExperiment",
    "PostSelectionExperimentCache",
    "SquinKernel",
    "SyndromeLayout",
    "TableDecoderWithConfidence",
    "TomographyKernels",
    "TomographyResult",
    "TsimCircuit",
    "apply_special_tsim_circuit_strategy",
    "build_decoder_kernel_bundle",
    "build_msd_primitives",
    "build_task_map",
    "empty_logical_circuit",
    "magic_state_dist_steane",
    "plot_decoder_curves",
    "run_task",
    "single_qubit_state_tomography",
    "split_factory_bits",
]
