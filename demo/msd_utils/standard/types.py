"""Compatibility type aliases for migrated MSD utilities."""

from bloqade.analysis.tomography import FidelitySummary, PosteriorFidelitySummary

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
    "SimulatorTask",
    "SquinKernel",
    "TableDecoderClass",
    "TsimCircuit",
]
