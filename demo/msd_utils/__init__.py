# pyright: reportUnsupportedDunderAll=false

"""Small facade for the MSD postselection notebook."""

from demo.msd_utils.application.experiments import PostSelectionExperiment
from demo.msd_utils.application.table_decoders import TableDecoderWithConfidence
from demo.msd_utils.standard.tomography import TomographyResult

from bloqade.gemini.decoding.kernels import DecoderPrimitiveSet
from bloqade.gemini.decoding.msd import (
    TomographyKernels,
    build_decoder_kernel_bundle,
    build_msd_primitives,
)
from bloqade.gemini.decoding.postselection import DecoderAdapter
from bloqade.gemini.decoding.sampling import BasisDataset, run_task
from bloqade.gemini.decoding.tasks import DemoTask
from bloqade.gemini.decoding.workflow import plot_decoder_curves

__all__ = [
    "BasisDataset",
    "DecoderAdapter",
    "DecoderPrimitiveSet",
    "DemoTask",
    "PostSelectionExperiment",
    "TableDecoderWithConfidence",
    "TomographyKernels",
    "TomographyResult",
    "build_decoder_kernel_bundle",
    "build_msd_primitives",
    "plot_decoder_curves",
    "run_task",
]
