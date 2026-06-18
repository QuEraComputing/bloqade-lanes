from __future__ import annotations

from typing import Any, Protocol, TypeAlias

import stim
import tsim as tsim_backend
from demo.msd_utils.application.table_decoders import TableDecoderWithConfidence
from kirin import ir

KirinKernel: TypeAlias = ir.Method[..., Any]
SquinKernel: TypeAlias = KirinKernel
TsimCircuit: TypeAlias = tsim_backend.Circuit
MeasurementMap: TypeAlias = list[list[int]]
TableDecoderClass: TypeAlias = type[TableDecoderWithConfidence]


class DetectorErrorModelTask(Protocol):
    @property
    def detector_error_model(self) -> stim.DetectorErrorModel: ...
