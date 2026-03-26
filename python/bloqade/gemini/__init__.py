from importlib import import_module

from . import logical as logical

__all__ = [
    "DetectorResult",
    "GeminiLogicalSimulator",
    "GeminiLogicalSimulatorTask",
    "Result",
    "logical",
]

_DEVICE_EXPORTS = {
    "DetectorResult",
    "GeminiLogicalSimulator",
    "GeminiLogicalSimulatorTask",
    "Result",
}


def __getattr__(name: str):
    if name in _DEVICE_EXPORTS:
        return getattr(import_module("bloqade.gemini.device"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
