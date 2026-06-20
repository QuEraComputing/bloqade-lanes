from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    import tsim as tsim_backend

TsimCircuit: TypeAlias = "tsim_backend.Circuit"
MeasurementMap: TypeAlias = list[list[int]]
