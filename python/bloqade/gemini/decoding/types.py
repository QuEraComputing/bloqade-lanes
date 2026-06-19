from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

from kirin import ir

if TYPE_CHECKING:
    import tsim as tsim_backend

KirinKernel: TypeAlias = ir.Method[..., Any]
SquinKernel: TypeAlias = KirinKernel
TsimCircuit: TypeAlias = "tsim_backend.Circuit"
MeasurementMap: TypeAlias = list[list[int]]
