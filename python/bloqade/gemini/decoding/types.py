from __future__ import annotations

from typing import Any, TypeAlias

import tsim as tsim_backend
from kirin import ir

KirinKernel: TypeAlias = ir.Method[..., Any]
SquinKernel: TypeAlias = KirinKernel
TsimCircuit: TypeAlias = tsim_backend.Circuit
MeasurementMap: TypeAlias = list[list[int]]
