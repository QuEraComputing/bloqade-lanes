from __future__ import annotations

from typing import Any, Protocol, TypeAlias

import stim
import tsim as tsim_backend
from kirin import ir

from .table_decoders import TableDecoderWithConfidence

KirinKernel: TypeAlias = ir.Method[..., Any]
SquinKernel: TypeAlias = KirinKernel
TsimCircuit: TypeAlias = tsim_backend.Circuit
MeasurementMap: TypeAlias = list[list[int]]
TableDecoderClass: TypeAlias = type[TableDecoderWithConfidence]


class DetectorErrorModelTask(Protocol):
    @property
    def detector_error_model(self) -> stim.DetectorErrorModel: ...
