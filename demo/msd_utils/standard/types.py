"""Compatibility type aliases for migrated MSD utilities."""

from demo.msd_utils.standard.tomography import (
    FidelitySummary,
    PosteriorFidelitySummary,
    SimpleFidelitySummary,
    TomographyResult,
)

from bloqade.gemini.decoding.sampling import DetectorObservableResult, SimulatorTask
from bloqade.gemini.decoding.types import (
    DetectorErrorModelTask,
    KirinKernel,
    MeasurementMap,
    SquinKernel,
    TableDecoderClass,
    TsimCircuit,
)

__all__ = [
    "DetectorErrorModelTask",
    "DetectorObservableResult",
    "FidelitySummary",
    "KirinKernel",
    "MeasurementMap",
    "PosteriorFidelitySummary",
    "SimpleFidelitySummary",
    "SimulatorTask",
    "SquinKernel",
    "TableDecoderClass",
    "TomographyResult",
    "TsimCircuit",
]
