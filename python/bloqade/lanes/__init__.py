from __future__ import annotations

from importlib import import_module
from typing import Any

from bloqade.gemini.device import (
    DetectorResult as DetectorResult,
    GeminiLogicalSimulator as GeminiLogicalSimulator,
    GeminiLogicalSimulatorTask as GeminiLogicalSimulatorTask,
    Result as Result,
)

# pyright: reportUnsupportedDunderAll=false


_EXPORTS = {
    "Metrics": ".metrics",
    "NoiseModelABC": ".rewrite.move2squin.noise",
    "generate_logical_noise_model": ".noise_model",
    "generate_simple_noise_model": ".noise_model",
    "steane7_m2dets": ".steane_defaults",
    "steane7_m2obs": ".steane_defaults",
}

__all__ = [
    "DetectorResult",
    "GeminiLogicalSimulator",
    "GeminiLogicalSimulatorTask",
    "Metrics",
    "NoiseModelABC",
    "Result",
    "generate_logical_noise_model",
    "generate_simple_noise_model",
    "steane7_m2dets",
    "steane7_m2obs",
]


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
