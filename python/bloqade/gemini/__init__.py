from __future__ import annotations

from importlib import import_module
from typing import Any

from .device import (
    DetectorResult as DetectorResult,
    GeminiLogicalSimulator as GeminiLogicalSimulator,
    GeminiLogicalSimulatorTask as GeminiLogicalSimulatorTask,
    Result as Result,
)

# pyright: reportUnsupportedDunderAll=false


_SUBMODULES = {"common", "decoding", "logical"}
_DEVICE_EXPORTS = {
    "GeminiLogicalDevice",
    "GeminiLogicalFuture",
    "GeminiLogicalResult",
}


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        return import_module(f"{__name__}.{name}")
    if name in _DEVICE_EXPORTS:
        device = import_module(f"{__name__}.device")
        return getattr(device, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DetectorResult",
    "GeminiLogicalDevice",
    "GeminiLogicalFuture",
    "GeminiLogicalResult",
    "GeminiLogicalSimulator",
    "GeminiLogicalSimulatorTask",
    "Result",
    "common",
    "decoding",
    "logical",
]
