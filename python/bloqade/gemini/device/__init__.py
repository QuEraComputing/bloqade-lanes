from __future__ import annotations

from importlib import import_module
from typing import Any

from .simulator import (
    DetectorResult as DetectorResult,
    GeminiLogicalSimulator as GeminiLogicalSimulator,
    GeminiLogicalSimulatorTask as GeminiLogicalSimulatorTask,
    Result as Result,
)

# pyright: reportUnsupportedDunderAll=false


_LOGICAL_EXPORTS = {
    "GeminiLogicalDevice": "device",
    "GeminiLogicalFuture": "future",
    "GeminiLogicalResult": "result",
}


def __getattr__(name: str) -> Any:
    module_name = _LOGICAL_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.logical.{module_name}")
    return getattr(module, name)


__all__ = [
    "DetectorResult",
    "GeminiLogicalDevice",
    "GeminiLogicalFuture",
    "GeminiLogicalResult",
    "GeminiLogicalSimulator",
    "GeminiLogicalSimulatorTask",
    "Result",
]
